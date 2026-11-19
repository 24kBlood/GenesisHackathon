"""Exact reproducibility and 20 paired permutation & ID-remapping invariance tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from audit_rl.env import AuditEnv
from audit_rl.generator import (
    generate_episode,
    permute_and_relabel_episode,
    verify_episode_integrity,
)
from audit_rl.policies import OraclePolicy, SimpleReconciliationPolicy, run_policy_on_env
from audit_rl.schemas import EpisodeConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_CONFIGS_PATH = REPO_ROOT / "configs" / "eval_configs.json"


class TestReproducibilityAndPermutation(unittest.TestCase):
    def test_exact_reproducibility_same_seed_config_and_actions(self) -> None:
        cfg = EpisodeConfig(
            num_invoices=35,
            error_counts={
                "duplicate_invoice": 1,
                "payment_allocation_mismatch": 1,
                "fx_conversion_error": 1,
                "period_cutoff_error": 1,
            },
            interacting_scenarios=["fx_period_partial_payment"],
        )

        def _run_trace() -> list[dict[str, Any]]:
            env = AuditEnv(mode="train", max_steps=100)
            obs0, info0 = env.reset(seed=4242, config=cfg)
            trace: list[dict[str, Any]] = [{"reset_obs": obs0, "reset_info": info0}]

            p_inv, r1, t1, tr1, i1 = env.step(
                {"action": "list_records", "table": "invoices", "cursor": "0", "limit": 20}
            )
            trace.append({"obs": p_inv, "r": r1, "t": t1, "tr": tr1, "info": i1})

            inv0 = p_inv["records"][0]
            g_doc, r2, t2, tr2, i2 = env.step({"action": "get_document", "doc_id": inv0["doc_id"]})
            trace.append({"obs": g_doc, "r": r2, "t": t2, "tr": tr2, "info": i2})

            calc, r3, t3, tr3, i3 = env.step({"action": "calculator", "expression": "12345 * 1.085"})
            trace.append({"obs": calc, "r": r3, "t": t3, "tr": tr3, "info": i3})

            sub, r4, t4, tr4, i4 = env.step(
                {
                    "action": "submit",
                    "findings": {"episode_id": obs0["manifest"]["episode_id"], "findings": []},
                }
            )
            trace.append({"obs": sub, "r": r4, "t": t4, "tr": tr4, "info": i4})
            return trace

        trace_a = _run_trace()
        trace_b = _run_trace()
        self.assertEqual(trace_a, trace_b)

    def test_exact_reproducibility_with_simple_reconciliation_policy(self) -> None:
        cfg = EpisodeConfig(
            num_invoices=30,
            error_counts={"duplicate_invoice": 1, "fx_conversion_error": 1},
            interacting_scenarios=["duplicate_invoice_fx_mismatch"],
        )
        res1 = run_policy_on_env(
            SimpleReconciliationPolicy(), AuditEnv(mode="train"), seed=7777, config=cfg, record_trajectory=True
        )
        res2 = run_policy_on_env(
            SimpleReconciliationPolicy(), AuditEnv(mode="train"), seed=7777, config=cfg, record_trajectory=True
        )
        self.assertEqual(res1["trajectory"], res2["trajectory"])
        self.assertEqual(res1["reward"], res2["reward"])
        self.assertEqual(res1["metrics"], res2["metrics"])

    def test_20_paired_permutation_and_id_remapping_invariance(self) -> None:
        with EVAL_CONFIGS_PATH.open("r", encoding="utf-8") as fh:
            eval_doc = json.load(fh)
        episodes = eval_doc["episodes"]
        self.assertGreaterEqual(len(episodes), 20)

        for idx, spec in enumerate(episodes[:20]):
            seed = int(spec["seed"])
            cfg = EpisodeConfig.from_dict(spec["config"])
            with self.subTest(pair_index=idx, seed=seed):
                orig_ep = generate_episode(seed=seed, config=cfg)
                remapped_ep, id_map = permute_and_relabel_episode(orig_ep, perm_seed=seed * 31 + 17)

                # Episode ID and record IDs must change
                self.assertNotEqual(
                    remapped_ep.manifest["episode_id"], orig_ep.manifest["episode_id"]
                )
                orig_inv_ids = [i["invoice_id"] for i in orig_ep.invoices]
                remap_inv_ids = [i["invoice_id"] for i in remapped_ep.invoices]
                self.assertEqual(set(orig_inv_ids) & set(remap_inv_ids), set())
                self.assertEqual(len(orig_inv_ids), len(remap_inv_ids))
                self.assertGreater(len(id_map), len(orig_inv_ids))

                # Order should differ on tables (at least ledger order changes)
                orig_ledger_mapped = [id_map[r["row_id"]] for r in orig_ep.ledger]
                remap_ledger_ids = [r["row_id"] for r in remapped_ep.ledger]
                self.assertNotEqual(orig_ledger_mapped, remap_ledger_ids)

                # Both pass independent episode integrity checks
                orig_check = verify_episode_integrity(orig_ep)
                remap_check = verify_episode_integrity(remapped_ep)
                self.assertTrue(orig_check["ok"], msg=orig_check["errors"])
                self.assertTrue(remap_check["ok"], msg=remap_check["errors"])

                # Oracle produces identical F1=1.0 and reward=1.0 on both original and remapped
                env_orig = AuditEnv(mode="eval")
                res_orig = run_policy_on_env(
                    OraclePolicy(orig_ep.truth),
                    env_orig,
                    seed=seed,
                    episode=orig_ep,
                    record_trajectory=False,
                )
                env_remap = AuditEnv(mode="eval")
                res_remap = run_policy_on_env(
                    OraclePolicy(remapped_ep.truth),
                    env_remap,
                    seed=seed,
                    episode=remapped_ep,
                    record_trajectory=False,
                )

                self.assertEqual(res_orig["metrics"]["f1"], 1.0)
                self.assertEqual(res_remap["metrics"]["f1"], 1.0)
                self.assertEqual(res_orig["reward"], 1.0)
                self.assertEqual(res_remap["reward"], 1.0)


if __name__ == "__main__":
    unittest.main()
