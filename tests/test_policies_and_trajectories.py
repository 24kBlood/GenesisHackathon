"""End-to-end tests for baseline policies, relative ranking, and trajectory leak scanning."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from audit_rl.env import AuditEnv, SubprocessAuditEnv
from audit_rl.generator import generate_episode
from audit_rl.policies import (
    FlagEverythingAttackPolicy,
    OraclePolicy,
    RandomExplorerPolicy,
    SimpleReconciliationPolicy,
    run_policy_on_env,
    scan_trajectories_for_leaks,
)
from audit_rl.schemas import EpisodeConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
TRAJECTORIES_PATH = REPO_ROOT / "trajectories" / "trajectories.jsonl"


def _load_specs(filename: str) -> list[dict[str, Any]]:
    with (CONFIGS_DIR / filename).open("r", encoding="utf-8") as fh:
        return list(json.load(fh)["episodes"])


class TestPoliciesAndTrajectories(unittest.TestCase):
    def test_all_four_policies_run_cleanly_on_train_dev_eval_samples(self) -> None:
        samples = [
            _load_specs("train_configs.json")[0],
            _load_specs("dev_configs.json")[0],
            _load_specs("eval_configs.json")[0],
            _load_specs("eval_configs.json")[5],
        ]
        for spec in samples:
            seed = int(spec["seed"])
            cfg = EpisodeConfig.from_dict(spec["config"])
            ep = generate_episode(seed=seed, config=cfg)

            policies = [
                RandomExplorerPolicy(seed=0),
                SimpleReconciliationPolicy(),
                FlagEverythingAttackPolicy(),
                OraclePolicy(ep.truth),
            ]
            for policy in policies:
                with self.subTest(seed=seed, policy=policy.name):
                    env = AuditEnv(mode="eval")
                    res = run_policy_on_env(policy, env, seed=seed, episode=ep, record_trajectory=True)
                    self.assertTrue(res["terminated"] or res["truncated"])
                    self.assertGreaterEqual(res["steps"], 1)
                    self.assertLessEqual(res["steps"], 100)
                    self.assertGreaterEqual(res["reward"], -1.0)
                    self.assertLessEqual(res["reward"], 1.0)

    def test_oracle_f1_one_on_clean_single_error_and_interacting_episodes(self) -> None:
        cases = [
            ("clean", EpisodeConfig(num_invoices=25, clean=True)),
            ("single_dup", EpisodeConfig(num_invoices=25, error_counts={"duplicate_invoice": 2})),
            ("single_fx", EpisodeConfig(num_invoices=25, error_counts={"fx_conversion_error": 2})),
            ("single_alloc", EpisodeConfig(num_invoices=25, error_counts={"payment_allocation_mismatch": 2})),
            ("single_cutoff", EpisodeConfig(num_invoices=25, error_counts={"period_cutoff_error": 2})),
            ("interacting_1", EpisodeConfig(num_invoices=30, interacting_scenarios=["fx_period_partial_payment"])),
            ("interacting_2", EpisodeConfig(num_invoices=30, interacting_scenarios=["duplicate_invoice_fx_mismatch"])),
        ]
        for label, cfg in cases:
            with self.subTest(case=label):
                ep = generate_episode(seed=555, config=cfg)
                env = AuditEnv(mode="eval")
                res = run_policy_on_env(
                    OraclePolicy(ep.truth), env, seed=555, episode=ep, record_trajectory=False
                )
                self.assertEqual(res["metrics"]["f1"], 1.0)
                self.assertEqual(res["reward"], 1.0)
                self.assertEqual(res["steps"], 1)
                self.assertEqual(res["inspection_steps"], 0)

    def test_baseline_ranking_on_eval_subset_via_subprocess_env(self) -> None:
        eval_specs = _load_specs("eval_configs.json")[:8]
        returns: dict[str, list[float]] = {
            "random_explorer": [],
            "simple_reconciliation": [],
            "flag_everything_attack": [],
            "oracle": [],
        }

        with SubprocessAuditEnv(mode="eval", max_steps=100) as env:
            for spec in eval_specs:
                seed = int(spec["seed"])
                cfg = EpisodeConfig.from_dict(spec["config"])
                ep = generate_episode(seed=seed, config=cfg)

                for policy in (
                    RandomExplorerPolicy(seed=0),
                    SimpleReconciliationPolicy(),
                    FlagEverythingAttackPolicy(),
                    OraclePolicy(ep.truth),
                ):
                    res = run_policy_on_env(policy, env, seed=seed, config=cfg, record_trajectory=False)
                    returns[policy.name].append(res["reward"])

        mean_random = sum(returns["random_explorer"]) / len(eval_specs)
        mean_simple = sum(returns["simple_reconciliation"]) / len(eval_specs)
        mean_flag = sum(returns["flag_everything_attack"]) / len(eval_specs)
        mean_oracle = sum(returns["oracle"]) / len(eval_specs)

        self.assertEqual(mean_oracle, 1.0)
        self.assertGreater(mean_simple, mean_random)
        self.assertGreater(mean_simple, mean_flag)
        self.assertLess(mean_flag, -0.5)

    def test_trajectories_jsonl_schema_and_zero_leak_scan(self) -> None:
        self.assertTrue(
            TRAJECTORIES_PATH.is_file(),
            f"Missing trajectories file at {TRAJECTORIES_PATH}",
        )
        scan = scan_trajectories_for_leaks(TRAJECTORIES_PATH)
        self.assertTrue(scan["passed"], msg=f"Leak scan failed: {scan['errors']}")
        self.assertGreater(scan["lines"], 0)

        required_fields = {
            "episode_id",
            "step",
            "action",
            "observation",
            "reward",
            "terminated",
            "truncated",
        }
        with TRAJECTORIES_PATH.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                raw = line.strip()
                if not raw:
                    continue
                rec = json.loads(raw)
                self.assertTrue(
                    required_fields.issubset(rec.keys()),
                    f"Line {line_no} missing fields: {required_fields - set(rec.keys())}",
                )
                self.assertIsInstance(rec["episode_id"], str)
                self.assertIsInstance(rec["step"], int)
                self.assertIsInstance(rec["action"], dict)
                self.assertIsInstance(rec["observation"], dict)
                self.assertIsInstance(rec["reward"], (int, float))
                self.assertIsInstance(rec["terminated"], bool)
                self.assertIsInstance(rec["truncated"], bool)


if __name__ == "__main__":
    unittest.main()
