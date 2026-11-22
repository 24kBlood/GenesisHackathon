"""Replay and pair verification for the 20 open diagnostic profiles."""

import argparse
import asyncio
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from genesis.environment.auditor_env import AuditorEnv
from genesis.environment.episode import build_episode
from genesis.generation.company import SCHEMA_VERSION, Episode, _generation_config, generate_episode
from genesis.generation.datasets import GENERATOR_VERSION
from genesis.policies import make_policy

from .benchmark import BenchmarkEpisode, BenchmarkResult, LocalPolicyRunner, run_benchmark
from .judge import run_episode
from .oracle import OraclePolicy
from .permutation import TABLES, evidence_catalog, permute_episode, rewrite_ids, validate_links
from .public_proofs import verify_public_proofs


PUBLIC_POLICIES = ("random", "simple_reconciliation", "mark_all")
POLICY_SEED = 89131
PAIR_SEED = 74137
PROFILE_COUNTS = {"clean": 5, "interacting": 5, "standard": 10}


@dataclass(frozen=True)
class ReproducibilityReport:
    replay_passed: int
    pair_passed: int
    total: int


def _unique_fields(pairs: list[tuple[str, object]]) -> dict:
    fields: dict = {}
    for key, value in pairs:
        if key in fields:
            raise ValueError(f"duplicate diagnostic profile field: {key}")
        fields[key] = value
    return fields


def load_profiles(path: Path | str) -> tuple[BenchmarkEpisode, ...]:
    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_fields)
    required = {"profile_version", "generator_version", "schema_version", "policy_seed", "permutation_seed", "profiles"}
    if type(value) is not dict or set(value) != required:
        raise ValueError("invalid diagnostic profile index")
    if (value["profile_version"], value["generator_version"], value["schema_version"]) != (
        "1.0.0", GENERATOR_VERSION, SCHEMA_VERSION,
    ):
        raise ValueError("unsupported diagnostic profile version")
    if value["policy_seed"] != POLICY_SEED or value["permutation_seed"] != PAIR_SEED:
        raise ValueError("unexpected diagnostic policy or permutation seed")
    rows = value["profiles"]
    if type(rows) is not list or len(rows) != sum(PROFILE_COUNTS.values()):
        raise ValueError("expected 20 diagnostic profiles")
    profiles = []
    categories: Counter = Counter()
    for row in rows:
        if type(row) is not dict or set(row) != {"name", "seed", "category", "config"}:
            raise ValueError("invalid diagnostic profile")
        if type(row["name"]) is not str or not row["name"] or type(row["seed"]) is not int:
            raise ValueError("invalid diagnostic name or seed")
        config = _generation_config(row["config"])
        if config != row["config"]:
            raise ValueError("diagnostic configuration must be normalized")
        category = "interacting" if config.get("interaction_scenarios") else (
            "standard" if config["error_counts"] else "clean"
        )
        if row["category"] != category:
            raise ValueError("diagnostic category does not match configuration")
        categories[category] += 1
        profiles.append(BenchmarkEpisode(row["name"], row["seed"], config))
    if categories != PROFILE_COUNTS or len({row.name for row in profiles}) != len(profiles) or len({
        row.seed for row in profiles
    }) != len(profiles):
        raise ValueError("diagnostic names, seeds or category totals are invalid")
    return tuple(profiles)


class _CapturedEnv(AuditorEnv):
    def reset(self, seed, config):
        observation, info = super().reset(seed, config)
        self.initial = {"observation": observation, "info": info}
        return observation, info


def _env(make_episode=generate_episode, *, clock=lambda: 0.0):
    def factory(seed, config):
        episode = make_episode(seed, config)
        public = episode.public
        return build_episode(
            task="Проверьте открытые записи эпизода.", manifest=public.manifest,
            policy=public.policy, tables={name: getattr(public, name) for name in TABLES},
            truth=episode.evaluator.truth,
        )

    return _CapturedEnv(factory, mode="eval", clock=clock)


def _oracle(episode: Episode) -> OraclePolicy:
    return OraclePolicy(episode.evaluator.truth, evidence_catalog(episode.public))


def _reconciliation_runner() -> LocalPolicyRunner:
    return LocalPolicyRunner(lambda: make_policy("simple_reconciliation", seed=POLICY_SEED))


def _canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _first_difference(left, right, path="$") -> str:
    if type(left) is type(right) is dict:
        for key in sorted(left.keys() | right.keys()):
            if key not in left or key not in right:
                return f"{path}.{key}"
            if _canonical(left[key]) != _canonical(right[key]):
                return _first_difference(left[key], right[key], f"{path}.{key}")
    if type(left) is type(right) is list:
        for index, (a, b) in enumerate(zip(left, right)):
            if _canonical(a) != _canonical(b):
                return _first_difference(a, b, f"{path}[{index}]")
        if len(left) != len(right):
            return f"{path}[{min(len(left), len(right))}]"
    return path


def require_replay_equal(case: str, component: str, first, second) -> None:
    if sha256(_canonical(first)).digest() != sha256(_canonical(second)).digest():
        path = _first_difference(first, second)
        raise ValueError(f"{case}: {component} differs at {path}")


class _ReplayPolicy:
    def __init__(self, actions):
        self.pending = deque(actions)

    def act(self, observation, previous_action):
        if not self.pending:
            raise ValueError("replay requested an action beyond the captured sequence")
        return self.pending.popleft()


def _diagnostics(terminal) -> dict:
    match = terminal.matching
    assert match is not None
    return {
        "tp": match.tp, "fp": match.fp, "fn": match.fn, "score": asdict(terminal.score),
        "total_steps": terminal.total_steps, "terminated": terminal.terminated,
        "truncated": terminal.truncated, "reason": terminal.reason,
    }


async def _capture(profile, runner, make_episode=generate_episode):
    from genesis.trajectories import TrajectoryRecorder, validate_trajectory

    recorder = TrajectoryRecorder(mode="eval")
    env = _env(make_episode)
    outcome = await run_episode(
        env, seed=profile.seed, config=profile.config, runner=runner, on_step=recorder.record,
    )
    if outcome.transport_error is not None:
        raise ValueError(f"unexpected diagnostic transport outcome: {outcome.transport_error}")
    records = recorder.records
    validate_trajectory(records, mode="eval")
    return {"reset": env.initial, "steps": list(records)}, outcome.terminal


def _normalized_submission(submission: dict) -> dict:
    findings = [
        {**row, "affected_ids": sorted(row["affected_ids"]), "evidence_ids": sorted(row["evidence_ids"])}
        for row in submission["findings"]
    ]
    return {**submission, "findings": sorted(findings, key=_canonical)}


def _normalized_truth(truth: dict) -> dict:
    issues = []
    for row in truth["issues"]:
        proof_sets = sorted(sorted(items) for items in row["acceptable_evidence_sets"])
        issues.append({**row, "affected_ids": sorted(row["affected_ids"]), "acceptable_evidence_sets": proof_sets})
    return {**truth, "issues": sorted(issues, key=lambda item: item["issue_id"])}


def _normalized_public(public: dict) -> dict:
    return {field: sorted(data, key=_canonical) if field in TABLES else data for field, data in public.items()}


async def _verify(profiles) -> ReproducibilityReport:
    for profile in profiles:
        try:
            first = generate_episode(profile.seed, profile.config)
            second = generate_episode(profile.seed, profile.config)
            require_replay_equal(profile.name, "public records", asdict(first.public), asdict(second.public))
            require_replay_equal(profile.name, "trusted artifacts", asdict(first.evaluator), asdict(second.evaluator))
            first_runner = LocalPolicyRunner(lambda: make_policy("random", seed=POLICY_SEED))
            public_first, terminal_first = await _capture(profile, first_runner, lambda *_: first)
            replay = _ReplayPolicy(first_runner.actions)
            public_second, terminal_second = await _capture(profile, LocalPolicyRunner(lambda: replay), lambda *_: second)
            if replay.pending:
                raise ValueError("replay stopped before consuming all captured actions")
            require_replay_equal(profile.name, "public replay steps", public_first, public_second)
            require_replay_equal(
                profile.name, "terminal diagnostics", _diagnostics(terminal_first), _diagnostics(terminal_second),
            )
        except ValueError as error:
            raise ValueError(f"{profile.name}: replay: {error}") from None

        try:
            validate_links(first)
            moved, mapping = permute_episode(first, salt=PAIR_SEED + profile.seed)
            inverse = {value: key for key, value in mapping.items()}
            verify_public_proofs(first)
            verify_public_proofs(moved)
            require_replay_equal(
                profile.name, "mapped public records", _normalized_public(asdict(first.public)),
                _normalized_public(rewrite_ids(asdict(moved.public), inverse)),
            )
            require_replay_equal(
                profile.name, "mapped truth", _normalized_truth(first.evaluator.truth),
                _normalized_truth(rewrite_ids(moved.evaluator.truth, inverse)),
            )
            require_replay_equal(
                profile.name, "mapped provenance", first.evaluator.provenance,
                rewrite_ids(moved.evaluator.provenance, inverse),
            )
            _, original_score = await _capture(profile, LocalPolicyRunner(lambda: _oracle(first)), lambda *_: first)
            _, moved_score = await _capture(profile, LocalPolicyRunner(lambda: _oracle(moved)), lambda *_: moved)
            require_replay_equal(
                profile.name, "mapped Oracle answer", _normalized_submission(original_score.submission),
                _normalized_submission(rewrite_ids(moved_score.submission, inverse)),
            )
            if original_score.score.metrics.f1 != 1.0:
                raise ValueError("oracle did not reach full F1")
            require_replay_equal(profile.name, "oracle scores", _diagnostics(original_score), _diagnostics(moved_score))
            _, recon_original = await _capture(profile, _reconciliation_runner(), lambda *_: first)
            _, recon_moved = await _capture(profile, _reconciliation_runner(), lambda *_: moved)
            require_replay_equal(
                profile.name, "mapped public policy answer",
                _normalized_submission(recon_original.submission),
                _normalized_submission(rewrite_ids(recon_moved.submission, inverse)),
            )
            require_replay_equal(
                profile.name, "public policy scores",
                _diagnostics(recon_original), _diagnostics(recon_moved),
            )
        except ValueError as error:
            raise ValueError(f"{profile.name}: pair: {error}") from None
    return ReproducibilityReport(len(profiles), len(profiles), len(profiles))


def verify_reproducibility(profile_path: Path | str) -> ReproducibilityReport:
    return asyncio.run(_verify(load_profiles(profile_path)))


async def local_benchmark(profile_path: Path | str) -> BenchmarkResult:
    profiles = load_profiles(profile_path)
    episodes_by_key = {
        (profile.seed, _canonical(profile.config)): generate_episode(profile.seed, profile.config)
        for profile in profiles
    }

    def cached_episode(seed, config):
        return episodes_by_key[seed, _canonical(config)]

    def runner(name):
        return lambda _: LocalPolicyRunner(lambda: make_policy(name, seed=POLICY_SEED))

    def oracle_runner(profile):
        episode = episodes_by_key[profile.seed, _canonical(profile.config)]
        return LocalPolicyRunner(lambda: _oracle(episode))

    return await run_benchmark(
        lambda: _env(cached_episode, clock=perf_counter), profiles,
        {**{name: runner(name) for name in PUBLIC_POLICIES}, "oracle": oracle_runner},
    )


def benchmark_summary(result: BenchmarkResult) -> dict:
    groups = defaultdict(list)
    timing = defaultdict(list)
    for item, seconds in zip(result.diagnostics, result.wall_time_seconds, strict=True):
        groups[item.policy].append(item)
        timing[item.policy].append(seconds)
    return {
        "diagnostics": {
            name: {
                "episodes": len(group), "precision": mean(row.precision for row in group),
                "recall": mean(row.recall for row in group), "f1": mean(row.f1 for row in group),
                "return": mean(row.reward for row in group), "steps": mean(row.total_steps for row in group),
                "return_min": min(row.reward for row in group), "return_max": max(row.reward for row in group),
            }
            for name, group in groups.items()
        },
        "wall_time_seconds": {name: mean(values) for name, values in timing.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "benchmark"))
    parser.add_argument("--profiles", type=Path, default=Path("benchmarks/local_eval_profiles.json"))
    args = parser.parse_args()
    try:
        result = (
            asdict(verify_reproducibility(args.profiles)) if args.command == "verify"
            else benchmark_summary(asyncio.run(local_benchmark(args.profiles)))
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"local diagnostics: {error}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
