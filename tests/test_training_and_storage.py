"""Tests for SQLite PlatformStore, val split integrity, RL training loop, and Evidence-First explanations."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from audit_rl.env import AuditEnv
from audit_rl.generator import generate_episode, verify_episode_integrity
from audit_rl.policies import (
    DEFAULT_UNTRAINED_PARAMS,
    LearnableAuditPolicy,
    SimpleReconciliationPolicy,
    run_policy_on_env,
)
from audit_rl.schemas import ERROR_TYPES, MAX_EXPLANATION_CHARS, EpisodeConfig
from audit_rl.storage import PlatformStore
from audit_rl.trainer import run_training_experiment

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"


class TestPlatformStore(unittest.TestCase):
    def test_record_list_and_reopen_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_platform.db"
            store = PlatformStore(db_path=db_path)

            exp1_id = store.record_experiment(
                kind="train",
                policy_name="learnable_audit_policy",
                summary={"eval_f1": 0.88, "train_reward": 0.70},
                details={"epochs": 2, "seed": 42},
                policy_version="v1",
                config_version="1.0.0",
                split="train",
            )
            exp2_id = store.record_experiment(
                kind="benchmark",
                policy_name="all_baselines",
                summary={"oracle_f1": 1.0},
                details={"episodes": 20},
                policy_version="1.0.0",
                config_version="1.0.0",
                split="eval",
            )
            self.assertGreater(exp2_id, exp1_id)

            ver_id = store.save_policy_version(
                policy_name="learnable_audit_policy",
                seed=42,
                params={"fx_tol_minor": 1, "check_dup": True},
                train_mean_reward=0.70,
                val_f1=0.94,
                eval_f1=0.88,
            )
            self.assertIsInstance(ver_id, str)
            self.assertTrue(len(ver_id) > 0)

            # Reopen fresh store instance against same SQLite file
            reopened = PlatformStore(db_path=db_path)
            all_exps = reopened.list_experiments(limit=10)
            self.assertEqual(len(all_exps), 2)
            # Newest first
            self.assertEqual(all_exps[0]["id"], exp2_id)
            self.assertEqual(all_exps[1]["id"], exp1_id)
            self.assertEqual(all_exps[1]["summary"]["eval_f1"], 0.88)

            train_only = reopened.list_experiments(limit=10, kind="train")
            self.assertEqual(len(train_only), 1)
            self.assertEqual(train_only[0]["kind"], "train")

            versions = reopened.list_policy_versions(limit=10)
            self.assertEqual(len(versions), 1)
            self.assertEqual(versions[0]["version_id"], ver_id)
            self.assertEqual(versions[0]["params"]["fx_tol_minor"], 1)
            self.assertAlmostEqual(versions[0]["eval_f1"], 0.88, places=4)

            overview = reopened.get_overview_stats()
            self.assertEqual(overview["total_experiments"], 2)
            self.assertEqual(overview["total_policy_versions"], 1)
            self.assertEqual(overview["counts_by_kind"]["train"], 1)
            self.assertEqual(overview["counts_by_kind"]["benchmark"], 1)
            self.assertIsNotNone(overview["best_policy_version"])
            self.assertEqual(overview["best_policy_version"]["version_id"], ver_id)


class TestValSplitIntegrity(unittest.TestCase):
    def test_val_configs_schema_distinct_seeds_and_integrity(self) -> None:
        val_path = CONFIGS_DIR / "val_configs.json"
        self.assertTrue(val_path.is_file(), "configs/val_configs.json must exist")

        with val_path.open("r", encoding="utf-8") as fh:
            val_doc = json.load(fh)

        self.assertIn("config_version", val_doc)
        val_episodes = val_doc["episodes"]
        self.assertEqual(len(val_episodes), 10)

        val_seeds = [int(e["seed"]) for e in val_episodes]
        self.assertEqual(len(val_seeds), len(set(val_seeds)), "Duplicate seeds in val_configs.json")

        # Distinct seeds from train, dev, eval
        other_seeds: set[int] = set()
        for split in ("train", "dev", "eval"):
            with (CONFIGS_DIR / f"{split}_configs.json").open("r", encoding="utf-8") as fh:
                doc = json.load(fh)
            for ep in doc["episodes"]:
                other_seeds.add(int(ep["seed"]))

        overlap = set(val_seeds) & other_seeds
        self.assertEqual(overlap, set(), f"val seeds overlap with train/dev/eval: {overlap}")

        # Verify every val episode generates cleanly and passes integrity checks
        for spec in val_episodes:
            cfg = EpisodeConfig.from_dict(spec["config"])
            ep = generate_episode(seed=int(spec["seed"]), config=cfg)
            report = verify_episode_integrity(ep)
            self.assertTrue(report["ok"], msg=f"Val seed {spec['seed']} failed: {report['errors']}")


class TestTrainingAndEvidenceFirst(unittest.TestCase):
    def test_learnable_policy_training_improves_on_unseen_splits_and_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db1 = Path(tmpdir) / "run1.db"
            db2 = Path(tmpdir) / "run2.db"

            res1 = run_training_experiment(
                seed=42,
                epochs=1,
                train_split="train",
                eval_split="eval",
                save_to_store=True,
                root=REPO_ROOT,
                db_path=db1,
            )
            res2 = run_training_experiment(
                seed=42,
                epochs=1,
                train_split="train",
                eval_split="eval",
                save_to_store=True,
                root=REPO_ROOT,
                db_path=db2,
            )

            # Deterministic reproducibility
            self.assertEqual(res1["params_after"], res2["params_after"])
            self.assertEqual(res1["trained_eval"]["f1"], res2["trained_eval"]["f1"])
            self.assertEqual(res1["trained_val"]["f1"], res2["trained_val"]["f1"])

            # Measurable improvement on unseen held-out val and eval splits
            self.assertGreater(
                res1["trained_eval"]["f1"],
                res1["untrained_eval"]["f1"],
                f"Expected trained eval F1 > untrained eval F1: {res1['trained_eval']} vs {res1['untrained_eval']}",
            )
            self.assertGreater(
                res1["trained_val"]["f1"],
                res1["untrained_val"]["f1"],
                f"Expected trained val F1 > untrained val F1: {res1['trained_val']} vs {res1['untrained_val']}",
            )
            self.assertGreater(res1["train_mean_reward_after"], res1["train_mean_reward_before"])
            self.assertGreater(res1["improvement_delta"]["eval_f1"], 0.0)

            # Persisted to SQLite store
            store = PlatformStore(db_path=db1)
            self.assertEqual(len(store.list_experiments(kind="train")), 1)
            self.assertEqual(len(store.list_policy_versions()), 1)

    def test_learnable_policy_runs_cleanly_with_default_untrained_params(self) -> None:
        cfg = EpisodeConfig(num_invoices=25, error_counts={"duplicate_invoice": 1})
        env = AuditEnv(mode="eval")
        pol = LearnableAuditPolicy(params=DEFAULT_UNTRAINED_PARAMS, version="untrained")
        res = run_policy_on_env(pol, env, seed=707, config=cfg, record_trajectory=False)
        self.assertTrue(res["terminated"] or res["truncated"])
        self.assertGreaterEqual(res["reward"], -1.0)
        self.assertLessEqual(res["reward"], 1.0)

    def test_evidence_first_explanations_in_simple_reconciliation_policy(self) -> None:
        cfg = EpisodeConfig(
            num_invoices=40,
            error_counts={et: 1 for et in ERROR_TYPES},
        )
        env = AuditEnv(mode="eval")
        policy = SimpleReconciliationPolicy()
        res = run_policy_on_env(policy, env, seed=8080, config=cfg, record_trajectory=True)

        self.assertEqual(res["metrics"]["f1"], 1.0)
        # Extract the submitted findings from the final trajectory step
        submit_steps = [
            step for step in res["trajectory"] if step["action"].get("action") == "submit"
        ]
        self.assertEqual(len(submit_steps), 1)
        findings = submit_steps[0]["action"]["findings"]["findings"]
        self.assertEqual(len(findings), len(ERROR_TYPES))

        for finding in findings:
            explanation = finding["explanation"]
            self.assertIsInstance(explanation, str)
            self.assertLessEqual(len(explanation), MAX_EXPLANATION_CHARS)
            self.assertIn("Observed:", explanation, f"Missing 'Observed:' in {explanation!r}")
            self.assertIn("Calculated:", explanation, f"Missing 'Calculated:' in {explanation!r}")
            self.assertIn("Rule:", explanation, f"Missing 'Rule:' in {explanation!r}")


if __name__ == "__main__":
    unittest.main()
