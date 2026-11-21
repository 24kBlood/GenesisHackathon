import asyncio
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from genesis.evaluation.benchmark import LocalPolicyRunner
from genesis.evaluation.judge import run_episode
from genesis.evaluation.permutation import TABLES, permute_episode, rewrite_ids, validate_links
from genesis.evaluation.public_proofs import verify_public_proofs
from genesis.evaluation.reproducibility import (
    PUBLIC_POLICIES, _diagnostics, _env, benchmark_summary, load_profiles,
    local_benchmark, require_replay_equal, verify_reproducibility,
)
from genesis.generation.company import generate_episode
from genesis.generation.datasets import load_dataset_entries
from genesis.policies import make_policy

import test_env_contract_and_isolation as contract


ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "benchmarks" / "local_eval_profiles.json"


class ReproducibilityTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profiles = load_profiles(PROFILES)

    def test_twenty_replays_and_pairs(self):
        self.assertEqual(asdict(verify_reproducibility(PROFILES)), {
            "replay_passed": 20, "pair_passed": 20, "total": 20,
        })

    def test_all_four_policies_receive_identical_open_case_set(self):
        result = asyncio.run(local_benchmark(PROFILES))
        policy_names = {*PUBLIC_POLICIES, "oracle"}
        self.assertEqual(len(result.diagnostics), 80)
        self.assertEqual(len(result.wall_time_seconds), 80)
        for policy in policy_names:
            actual = [item.case for item in result.diagnostics if item.policy == policy]
            self.assertEqual(actual, [profile.name for profile in self.profiles])
        self.assertFalse({profile.seed for profile in self.profiles} & {
            entry.seed for entry in load_dataset_entries(ROOT / "data" / "configs")
        })
        summary = benchmark_summary(result)
        diag = summary["diagnostics"]
        self.assertEqual(diag["oracle"]["f1"], 1.0)
        self.assertEqual(diag["oracle"]["precision"], 0.75)
        ordered = [diag[name]["return"] for name in (
            "oracle", "simple_reconciliation", "random", "mark_all",
        )]
        self.assertTrue(all(better > worse for better, worse in zip(ordered, ordered[1:])))
        self.assertEqual(diag["mark_all"]["return"], -1.0)
        self.assertGreater(diag["simple_reconciliation"]["f1"], 0.5)
        self.assertTrue(all(value >= 0 for value in summary["wall_time_seconds"].values()))

    def test_first_diff_and_changed_seed_and_posting(self):
        profile = self.profiles[0]
        a = generate_episode(profile.seed, profile.config)
        b = generate_episode(profile.seed + 1, profile.config)
        with self.assertRaisesRegex(ValueError, "changed-seed: records differs at"):
            require_replay_equal("changed-seed", "records", asdict(a.public), asdict(b.public))
        b = generate_episode(profile.seed, profile.config)
        b.public.ledger[0]["debit_minor"] += 1
        with self.assertRaisesRegex(ValueError, r"tampered: records differs at \$\.ledger\[0\]\.debit_minor"):
            require_replay_equal("tampered", "records", asdict(a.public), asdict(b.public))

    def test_bijection_changes_refunds_and_validates_all_shapes(self):
        contract.DataContractTests.setUpClass()
        validators = contract.DataContractTests.validators
        profile = next(
            case for case in self.profiles if case.config.get("include_distractors") and case.config.get("interaction_scenarios")
        )
        a = generate_episode(profile.seed, profile.config)
        b, mapping = permute_episode(a, salt=45)
        before = next(row for row in a.public.payments if row.get("payment_kind") == "refund")
        after = next(row for row in b.public.payments if row["payment_id"] == mapping[before["payment_id"]])
        self.assertEqual(after["refund_of_payment_id"], mapping[before["refund_of_payment_id"]])
        inverse = {value: key for key, value in mapping.items()}
        for name, schema in zip(TABLES, ("ledger-row", "invoice", "payment", "fx-row"), strict=True):
            rows = getattr(b.public, name)
            for row in rows:
                validators[schema].validate(row)
            self.assertCountEqual(
                [repr(sorted(row.items())) for row in getattr(a.public, name)],
                [repr(sorted(row.items())) for row in rewrite_ids(rows, inverse)],
            )
        validators["manifest"].validate(b.public.manifest)
        validators["policy"].validate(b.public.policy)
        validators["truth"].validate(b.evaluator.truth)
        validators["provenance"].validate(b.evaluator.provenance)
        keys = sorted(mapping)
        mapping[keys[0]] = mapping[keys[1]]
        with self.assertRaisesRegex(ValueError, "bijection"):
            permute_episode(a, salt=45, mapping=mapping)
        after["refund_of_payment_id"] = "missing-payment"
        with self.assertRaisesRegex(ValueError, "unresolved payment reference"):
            validate_links(b)

    def test_dropped_evidence_is_detected_by_public_reconciliation(self):
        profile = next(case for case in self.profiles if case.config["error_counts"] == {"fx_conversion": 1})
        episode = generate_episode(profile.seed, profile.config)
        proof = episode.evaluator.truth["issues"][0]["acceptable_evidence_sets"][0]
        proof.remove("policy")
        self.assertTrue(validate_links(episode))
        with self.assertRaisesRegex(ValueError, "conversion rule"):
            verify_public_proofs(episode)

    def test_injected_monotonic_timeout_and_wall_time_exclusion(self):
        profile = self.profiles[0]
        terminals = []
        for elapsed in (60.0, 71.0):
            now = [0.0]
            env = _env(clock=lambda: now[0])
            env.reset(profile.seed, profile.config)
            now[0] = elapsed
            observation, reward, terminated, truncated, info = env.step({"type": "calculator", "expression": "1"})
            self.assertEqual((observation, reward, terminated, truncated, info), (
                {"remaining_steps": 100, "status": "truncated"}, 0.0, False, True, {},
            ))
            terminal = env.terminal_result
            self.assertEqual(terminal.reason, "timeout")
            self.assertEqual(terminal.score.inspection_steps, 0)
            self.assertEqual(terminal.submission["findings"], [])
            terminals.append(terminal)
        self.assertNotEqual(terminals[0].wall_time_seconds, terminals[1].wall_time_seconds)
        require_replay_equal("timeout", "deterministic result", *map(_diagnostics, terminals))

    def test_load_profiles_rejects_duplicate_json_keys(self):
        raw = PROFILES.read_text(encoding="utf-8")
        duplicated = raw.replace('"policy_seed": 89131,', '"policy_seed": 1, "policy_seed": 89131,', 1)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "profiles.json"
            path.write_text(duplicated, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate diagnostic profile field"):
                load_profiles(path)

    def test_local_benchmark_reuses_generated_episodes_once_per_profile(self):
        with patch("genesis.evaluation.reproducibility.generate_episode", wraps=generate_episode) as mocked:
            result = asyncio.run(local_benchmark(PROFILES))
        self.assertEqual(len(result.diagnostics), 80)
        self.assertEqual(mocked.call_count, 20)

    def test_reconciliation_policy_is_invariant_under_permutation_and_id_renaming(self):
        profile = next(
            case for case in self.profiles
            if case.config.get("include_distractors") and case.config.get("interaction_scenarios")
        )
        original = generate_episode(profile.seed, profile.config)
        moved, mapping = permute_episode(original, salt=91234)
        inverse = {value: key for key, value in mapping.items()}

        async def run_both():
            first = await run_episode(
                _env(lambda *_: original), seed=profile.seed, config=profile.config,
                runner=LocalPolicyRunner(lambda: make_policy("simple_reconciliation")),
            )
            second = await run_episode(
                _env(lambda *_: moved), seed=profile.seed, config=profile.config,
                runner=LocalPolicyRunner(lambda: make_policy("simple_reconciliation")),
            )
            return first.terminal, second.terminal

        orig_term, moved_term = asyncio.run(run_both())
        self.assertEqual(_diagnostics(orig_term), _diagnostics(moved_term))
        self.assertEqual(orig_term.score.metrics.f1, 1.0)
        self.assertEqual(orig_term.total_steps, moved_term.total_steps)
        self.assertEqual(
            {tuple(sorted(item["affected_ids"])) for item in orig_term.submission["findings"]},
            {tuple(sorted(item["affected_ids"])) for item in rewrite_ids(moved_term.submission, inverse)["findings"]},
        )
