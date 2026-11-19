"""Command-line interface for the Audit RL environment.

Subcommands:
- generate-datasets: generate open train/dev datasets and trajectories/trajectories.jsonl
- run-baselines: evaluate all 4 baseline policies on the 20 held-out eval episodes (SubprocessAuditEnv, eval mode)
- verify: one-command full verification suite matching the 100-point rubric
- demo: readable step-by-step walkthrough of a single audit episode
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unittest
from pathlib import Path
from statistics import mean
from typing import Any

from audit_rl.env import AuditEnv, SubprocessAuditEnv
from audit_rl.generator import (
    generate_episode,
    permute_and_relabel_episode,
    verify_episode_integrity,
)
from audit_rl.policies import (
    FlagEverythingAttackPolicy,
    OraclePolicy,
    RandomExplorerPolicy,
    SimpleReconciliationPolicy,
    record_trajectories,
    run_policy_on_env,
    scan_trajectories_for_leaks,
)
from audit_rl.schemas import EpisodeConfig, save_episode


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_config_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def cmd_generate_datasets(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve() if args.root else _repo_root()
    configs_dir = root / "configs"
    data_dir = root / "data"
    traj_path = root / "trajectories" / "trajectories.jsonl"

    generated_summary: dict[str, Any] = {}
    traj_episodes: list[dict[str, Any]] = []

    for split in ("train", "dev"):
        cfg_file = configs_dir / f"{split}_configs.json"
        spec = _load_config_file(cfg_file)
        split_dir = data_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        index_entries: list[dict[str, Any]] = []

        for idx, ep_spec in enumerate(spec["episodes"], start=1):
            seed = int(ep_spec["seed"])
            ep_cfg = EpisodeConfig.from_dict(ep_spec["config"])
            episode = generate_episode(seed=seed, config=ep_cfg)
            ep_id = str(episode.manifest["episode_id"])
            ep_dir = split_dir / ep_id
            save_episode(episode, ep_dir, include_private=True)
            integrity = verify_episode_integrity(episode)
            if not integrity.get("ok", integrity.get("passed", False)):
                print(f"[ERROR] Integrity check failed for {split} seed={seed}: {integrity}", file=sys.stderr)
                return 1
            index_entries.append(
                {
                    "index": idx,
                    "episode_id": ep_id,
                    "seed": seed,
                    "description": ep_spec.get("description", ""),
                    "num_issues": len(episode.truth.get("issues", [])),
                    "path": str(ep_dir.relative_to(root)),
                }
            )
            if idx <= 5:
                traj_episodes.append(ep_spec)

        with (split_dir / "index.json").open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "config_version": spec.get("config_version", "1.0.0"),
                    "split": split,
                    "episodes": index_entries,
                },
                fh,
                indent=2,
            )
        generated_summary[split] = len(index_entries)

    policies = [
        SimpleReconciliationPolicy(),
        RandomExplorerPolicy(seed=0),
        FlagEverythingAttackPolicy(),
    ]
    traj_info = record_trajectories(traj_episodes, policies, traj_path)
    leak_check = scan_trajectories_for_leaks(traj_path)

    print(
        f"Generated datasets: train={generated_summary['train']} episodes, "
        f"dev={generated_summary['dev']} episodes"
    )
    print(
        f"Recorded trajectories: {traj_info['runs']} runs, {traj_info['steps']} steps -> {traj_path}"
    )
    print(
        f"Trajectory leak scan: {'PASS' if leak_check['passed'] else 'FAIL'} "
        f"({leak_check['lines']} lines)"
    )
    return 0 if leak_check["passed"] else 1


def evaluate_baselines(eval_config_path: Path) -> dict[str, Any]:
    """Run all 4 mandatory baseline policies across eval episodes using SubprocessAuditEnv(mode='eval')."""
    spec = _load_config_file(eval_config_path)
    episodes_spec = spec["episodes"]

    policy_names = [
        "random_explorer",
        "simple_reconciliation",
        "flag_everything_attack",
        "oracle",
    ]
    per_policy: dict[str, list[dict[str, Any]]] = {name: [] for name in policy_names}

    # Pre-generate episodes evaluator-side ONLY to obtain out-of-band truth for OraclePolicy.
    eval_truths: dict[int, dict[str, Any]] = {}
    for ep_spec in episodes_spec:
        seed = int(ep_spec["seed"])
        cfg = EpisodeConfig.from_dict(ep_spec["config"])
        ep = generate_episode(seed=seed, config=cfg)
        eval_truths[seed] = ep.truth

    with SubprocessAuditEnv(mode="eval") as env:
        for ep_spec in episodes_spec:
            seed = int(ep_spec["seed"])
            cfg_dict = ep_spec["config"]

            run_policies = [
                RandomExplorerPolicy(seed=0),
                SimpleReconciliationPolicy(),
                FlagEverythingAttackPolicy(),
                OraclePolicy(truth=eval_truths[seed]),
            ]
            for policy in run_policies:
                res = run_policy_on_env(
                    policy,
                    env,
                    seed=seed,
                    config=cfg_dict,
                    record_trajectory=False,
                )
                metrics = res.get("metrics", {})
                per_policy[policy.name].append(
                    {
                        "seed": seed,
                        "episode_id": res["episode_id"],
                        "precision": float(metrics.get("precision", 0.0)),
                        "recall": float(metrics.get("recall", 0.0)),
                        "f1": float(metrics.get("f1", 0.0)),
                        "return": float(res["reward"]),
                        "steps": int(res["steps"]),
                        "runtime_ms": float(res["runtime_ms"]),
                    }
                )

    summary: dict[str, dict[str, Any]] = {}
    for name in policy_names:
        rows = per_policy[name]
        summary[name] = {
            "episodes": len(rows),
            "precision": round(mean(r["precision"] for r in rows), 4),
            "recall": round(mean(r["recall"] for r in rows), 4),
            "f1": round(mean(r["f1"] for r in rows), 4),
            "return": round(mean(r["return"] for r in rows), 4),
            "steps": round(mean(r["steps"] for r in rows), 2),
            "runtime_ms": round(mean(r["runtime_ms"] for r in rows), 2),
            "total_runtime_ms": round(sum(r["runtime_ms"] for r in rows), 2),
        }

    return {
        "config_version": spec.get("config_version", "1.0.0"),
        "num_episodes": len(episodes_spec),
        "summary": summary,
        "details": per_policy,
    }


def _format_baseline_table(summary: dict[str, dict[str, Any]]) -> str:
    header = (
        f"{'Policy':<24} | {'P':>7} | {'R':>7} | {'F1':>7} | "
        f"{'Return':>8} | {'Steps':>6} | {'Avg ms':>8}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for name, stats in summary.items():
        lines.append(
            f"{name:<24} | {stats['precision']:>7.4f} | {stats['recall']:>7.4f} | "
            f"{stats['f1']:>7.4f} | {stats['return']:>8.4f} | "
            f"{stats['steps']:>6.2f} | {stats['runtime_ms']:>8.2f}"
        )
    return "\n".join(lines)


def cmd_run_baselines(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve() if args.root else _repo_root()
    eval_cfg = root / "configs" / "eval_configs.json"
    result = evaluate_baselines(eval_cfg)

    if args.json:
        print(json.dumps(result["summary"], indent=2))
    else:
        print(f"Baseline Evaluation ({result['num_episodes']} held-out eval episodes, SubprocessAuditEnv mode=eval)")
        print(_format_baseline_table(result["summary"]))
        print("\nJSON Summary:")
        print(json.dumps(result["summary"], indent=2))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve() if args.root else _repo_root()
    t0 = time.perf_counter()
    checks: list[tuple[str, bool, str]] = []

    # 1. Unit & integration tests
    tests_dir = root / "tests"
    if tests_dir.exists() and any(tests_dir.glob("test_*.py")):
        loader = unittest.defaultTestLoader
        suite = loader.discover(start_dir=str(tests_dir), top_level_dir=str(root))
        runner = unittest.TextTestRunner(verbosity=1 if args.verbose else 0)
        test_result = runner.run(suite)
        tests_ok = test_result.wasSuccessful() and test_result.testsRun > 0
        checks.append(
            (
                "Unit & Integration Tests",
                tests_ok,
                f"{test_result.testsRun} run, {len(test_result.failures)} failures, {len(test_result.errors)} errors",
            )
        )
    else:
        checks.append(("Unit & Integration Tests", False, "No tests found in tests/"))

    # 2. Independent integrity checks on 20 eval episodes
    eval_cfg_path = root / "configs" / "eval_configs.json"
    eval_spec = _load_config_file(eval_cfg_path)
    eval_episodes = []
    integrity_failures = 0
    for ep_spec in eval_spec["episodes"]:
        ep = generate_episode(seed=int(ep_spec["seed"]), config=EpisodeConfig.from_dict(ep_spec["config"]))
        eval_episodes.append((int(ep_spec["seed"]), ep))
        res = verify_episode_integrity(ep)
        if not res.get("ok", res.get("passed", False)):
            integrity_failures += 1
    checks.append(
        (
            "Eval Episode Integrity (20 episodes)",
            integrity_failures == 0 and len(eval_episodes) == 20,
            f"{len(eval_episodes) - integrity_failures}/{len(eval_episodes)} passed",
        )
    )

    # 3. 20 paired permutation & ID-remapping checks
    perm_passed = 0
    for idx, (seed, ep) in enumerate(eval_episodes):
        remapped, id_map = permute_and_relabel_episode(ep, perm_seed=seed + 50000 + idx)
        remapped_integrity = verify_episode_integrity(remapped)
        env_orig = AuditEnv(mode="train")
        orig_res = run_policy_on_env(OraclePolicy(ep.truth), env_orig, seed=seed, episode=ep, record_trajectory=False)
        env_perm = AuditEnv(mode="train")
        perm_res = run_policy_on_env(
            OraclePolicy(remapped.truth), env_perm, seed=seed, episode=remapped, record_trajectory=False
        )
        ok = (
            remapped_integrity.get("ok", remapped_integrity.get("passed", False))
            and bool(id_map)
            and remapped.manifest["episode_id"] != ep.manifest["episode_id"]
            and orig_res["metrics"].get("f1") == 1.0
            and perm_res["metrics"].get("f1") == 1.0
            and orig_res["reward"] == 1.0
            and perm_res["reward"] == 1.0
        )
        if ok:
            perm_passed += 1
    checks.append(
        (
            "Permutation & ID-Remapping Invariance (20 pairs)",
            perm_passed == len(eval_episodes) and len(eval_episodes) == 20,
            f"{perm_passed}/{len(eval_episodes)} passed",
        )
    )

    # 4. Trajectories leak scan (generate datasets if trajectories file missing)
    traj_path = root / "trajectories" / "trajectories.jsonl"
    if not traj_path.exists():
        gen_ns = argparse.Namespace(root=str(root))
        cmd_generate_datasets(gen_ns)
    leak_res = scan_trajectories_for_leaks(traj_path)
    checks.append(
        (
            "Trajectory Leak Scan",
            bool(leak_res["passed"]),
            f"{leak_res['lines']} lines scanned, {len(leak_res['errors'])} errors",
        )
    )

    # 5. All 4 baselines on 20 eval episodes via SubprocessAuditEnv (eval mode)
    baseline_res = evaluate_baselines(eval_cfg_path)
    summary = baseline_res["summary"]
    oracle_ok = summary["oracle"]["f1"] == 1.0 and summary["oracle"]["return"] == 1.0
    simple_beats_random = (
        summary["simple_reconciliation"]["f1"] > summary["random_explorer"]["f1"]
        and summary["simple_reconciliation"]["return"] > summary["random_explorer"]["return"]
    )
    simple_beats_attack = (
        summary["simple_reconciliation"]["f1"] > summary["flag_everything_attack"]["f1"]
        and summary["simple_reconciliation"]["return"] > summary["flag_everything_attack"]["return"]
    )
    attack_penalized = summary["flag_everything_attack"]["return"] < 0.0
    baselines_ok = oracle_ok and simple_beats_random and simple_beats_attack and attack_penalized
    checks.append(
        (
            "4 Baselines + Subprocess Isolation (20 eval episodes)",
            baselines_ok,
            (
                f"oracle_F1={summary['oracle']['f1']:.4f}, "
                f"simple_F1={summary['simple_reconciliation']['f1']:.4f}, "
                f"random_F1={summary['random_explorer']['f1']:.4f}, "
                f"attack_ret={summary['flag_everything_attack']['return']:.4f}"
            ),
        )
    )

    elapsed_s = time.perf_counter() - t0
    all_passed = all(ok for _, ok, _ in checks)

    print("\n=== Audit RL Verification Summary ===")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")
    print("\n=== Eval Baseline Table (SubprocessAuditEnv, mode=eval, 20 episodes) ===")
    print(_format_baseline_table(summary))
    print(f"\nOverall: {'PASS' if all_passed else 'FAIL'} ({elapsed_s:.2f}s)")
    return 0 if all_passed else 1


def cmd_demo(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve() if args.root else _repo_root()
    dev_cfg = _load_config_file(root / "configs" / "dev_configs.json")
    # Pick a multi-error episode (seed 2008, all 4 error types) for an illustrative walkthrough
    ep_spec = next((e for e in dev_cfg["episodes"] if e["seed"] == 2008), dev_cfg["episodes"][0])
    seed = int(ep_spec["seed"])
    cfg = ep_spec["config"]

    print(f"=== Audit RL Episode Walkthrough (seed={seed}: {ep_spec.get('description', '')}) ===\n")
    with SubprocessAuditEnv(mode="eval") as env:
        obs, info = env.reset(seed=seed, config=cfg)
        print("1. Reset Observation:")
        print(f"   episode_id       : {obs['manifest']['episode_id']}")
        print(f"   company          : {obs['manifest'].get('company_name', 'N/A')}")
        print(f"   base_currency    : {obs['manifest']['base_currency']}")
        print(f"   period           : {obs['manifest']['period_start']} -> {obs['manifest']['period_end']}")
        print(f"   fx_date_policy   : {obs['policy']['fx_date_policy']} ({obs['policy']['rounding_rule']})")
        print(f"   budget_remaining : {obs['budget_remaining']}")
        print(f"   available_actions: {obs.get('available_actions')}")

        policy = SimpleReconciliationPolicy(use_calculator=True)
        policy.reset(obs, info)

        step_num = 0
        shown_list = False
        shown_calc = False
        terminated = False
        truncated = False
        final_action: dict[str, Any] = {}

        while not (terminated or truncated):
            action = policy.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            step_num += 1
            act_type = action.get("action")
            if act_type == "list_records" and not shown_list:
                shown_list = True
                print(f"\n2. Sample Tool Call (step {step_num}): {json.dumps(action)}")
                recs = next_obs.get("records") or next_obs.get("items") or []
                print(f"   Returned {len(recs)} records, next_cursor={next_obs.get('next_cursor')}, budget={next_obs.get('budget_remaining')}")
            elif act_type == "calculator" and not shown_calc:
                shown_calc = True
                print(f"\n3. Safe Decimal Calculator Call (step {step_num}): {json.dumps(action)}")
                print(f"   Result: {next_obs.get('result', next_obs.get('value', next_obs))}")
            elif act_type == "submit":
                final_action = action
            obs = next_obs

        # Also demonstrate get_document on a fresh reset? No need, report submission & evaluation cleanly!
        submitted = final_action.get("findings", {}).get("findings", [])
        print(f"\n4. Submit Action (step {step_num}): {len(submitted)} findings submitted")
        for idx, f in enumerate(submitted, start=1):
            print(
                f"   [{idx}] {f['type']} | affected={f['affected_ids']} | "
                f"amount={f['claimed_amount_minor']} {f['currency']} | evidence={f['evidence_ids']}"
            )

        report = env.get_evaluator_report()
        metrics = report.get("metrics", {})
        print("\n5. Evaluator Report (out-of-process, mode=eval):")
        print(
            f"   TP={metrics.get('tp')} FP={metrics.get('fp')} FN={metrics.get('fn')} | "
            f"P={metrics.get('precision', 0.0):.4f} R={metrics.get('recall', 0.0):.4f} "
            f"F1={metrics.get('f1', 0.0):.4f}"
        )
        print(
            f"   reward={report.get('reward', 0.0):.4f} | "
            f"steps={report.get('steps', step_num)} (inspection_steps={report.get('inspection_steps', step_num - 1)})"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audit-rl",
        description="FinTech RL environment for financial audit model training.",
    )
    parser.add_argument("--root", default=None, help="Optional project root directory override.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("generate-datasets", help="Generate open train/dev datasets and trajectories.jsonl.")

    p_base = sub.add_parser("run-baselines", help="Run all 4 baseline policies on 20 eval episodes.")
    p_base.add_argument("--json", action="store_true", help="Print JSON summary only.")

    p_ver = sub.add_parser("verify", help="Run full one-command verification suite.")
    p_ver.add_argument("-v", "--verbose", action="store_true", help="Verbose unittest output.")

    sub.add_parser("demo", help="Run a readable walkthrough of one audit episode.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate-datasets":
        return cmd_generate_datasets(args)
    if args.command == "run-baselines":
        return cmd_run_baselines(args)
    if args.command == "verify":
        return cmd_verify(args)
    if args.command == "demo":
        return cmd_demo(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
