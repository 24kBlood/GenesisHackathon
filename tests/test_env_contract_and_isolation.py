"""Contract, tool-interface, security, and out-of-process isolation tests for AuditEnv & SubprocessAuditEnv."""

from __future__ import annotations

import json
import unittest
from typing import Any

from audit_rl.calculator import CalculatorError, safe_calculate
from audit_rl.env import AuditEnv, SubprocessAuditEnv
from audit_rl.generator import generate_episode
from audit_rl.schemas import EpisodeConfig, MAX_PAGE_LIMIT

FORBIDDEN_KEYS = {
    "truth",
    "provenance",
    "seed",
    "acceptable_evidence_sets",
    "pre_injection_checks",
    "injected_events",
    "expected_amount_minor",
}


def _collect_all_keys(obj: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(str(k))
            keys.update(_collect_all_keys(v))
    elif isinstance(obj, list):
        for item in obj:
            keys.update(_collect_all_keys(item))
    return keys


def _contains_filesystem_path(obj: Any) -> bool:
    blob = json.dumps(obj)
    return "/Users/" in blob or "/home/" in blob or "truth.json" in blob or "provenance.json" in blob


class TestEnvContractAndIsolation(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = EpisodeConfig(
            num_invoices=25,
            error_counts={"duplicate_invoice": 1, "fx_conversion_error": 1},
        )

    def test_01_reset_returns_expected_observation_and_info_format(self) -> None:
        env = AuditEnv(mode="train", max_steps=100)
        obs, info = env.reset(seed=1001, config=self.cfg)
        for key in ("task", "manifest", "policy", "budget_remaining", "step", "available_actions"):
            self.assertIn(key, obs)
        self.assertEqual(obs["step"], 0)
        self.assertEqual(obs["budget_remaining"], 100)
        self.assertEqual(info["episode_id"], obs["manifest"]["episode_id"])
        self.assertEqual(info["mode"], "train")
        self.assertEqual(info["step"], 0)

    def test_02_step_returns_gymnasium_5_tuple(self) -> None:
        env = AuditEnv(mode="train")
        env.reset(seed=1001, config=self.cfg)
        out = env.step({"action": "list_records", "table": "invoices", "cursor": "0", "limit": 10})
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 5)
        obs, reward, terminated, truncated, info = out
        self.assertIsInstance(obs, dict)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_03_step_before_reset_raises_runtime_error(self) -> None:
        env = AuditEnv(mode="train")
        with self.assertRaises(RuntimeError):
            env.step({"action": "list_records", "table": "invoices"})

    def test_04_state_isolation_across_consecutive_resets(self) -> None:
        env = AuditEnv(mode="train")
        obs1, _ = env.reset(seed=11, config=self.cfg)
        ep1_id = obs1["manifest"]["episode_id"]
        first_page = env.step({"action": "list_records", "table": "invoices", "limit": 5})[0]
        inv1_id = first_page["records"][0]["invoice_id"]
        env.step({"action": "submit", "findings": {"episode_id": ep1_id, "findings": []}})

        obs2, info2 = env.reset(seed=22, config=EpisodeConfig(num_invoices=20, clean=True))
        ep2_id = obs2["manifest"]["episode_id"]
        self.assertNotEqual(ep1_id, ep2_id)
        self.assertEqual(obs2["step"], 0)
        self.assertEqual(obs2["budget_remaining"], 100)
        self.assertEqual(info2["step"], 0)
        # Old invoice ID from episode 1 must not exist in episode 2
        lookup_obs, _, _, _, _ = env.step({"action": "get_record", "table": "invoices", "id": inv1_id})
        if lookup_obs["status"] == "ok":
            # Only possible if same random ID happened to collide, which is astronomically unlikely;
            # verify record belongs to ep2's table
            self.assertEqual(lookup_obs["record"]["invoice_id"], inv1_id)
        else:
            self.assertEqual(lookup_obs["status"], "error")

    def test_05_pagination_limit_max_50_and_stable_cursors(self) -> None:
        env = AuditEnv(mode="train")
        env.reset(seed=1002, config=self.cfg)
        seen_ids: list[str] = []
        cursor: str | None = "0"
        pages = 0
        while cursor is not None:
            obs, reward, terminated, truncated, _ = env.step(
                {"action": "list_records", "table": "ledger", "cursor": cursor, "limit": MAX_PAGE_LIMIT}
            )
            self.assertEqual(obs["status"], "ok")
            self.assertEqual(reward, 0.0)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertLessEqual(len(obs["records"]), MAX_PAGE_LIMIT)
            seen_ids.extend(r["row_id"] for r in obs["records"])
            cursor = obs["next_cursor"]
            pages += 1
        self.assertGreater(pages, 1)
        self.assertEqual(len(seen_ids), len(set(seen_ids)))
        self.assertEqual(len(seen_ids), obs["total_count"])

        # Re-querying cursor "0" returns identical first page (stable deterministic cursor)
        obs_again = env.step({"action": "list_records", "table": "ledger", "cursor": "0", "limit": 10})[0]
        self.assertEqual([r["row_id"] for r in obs_again["records"]], seen_ids[:10])

    def test_06_pagination_limit_above_50_or_below_1_returns_bounded_error(self) -> None:
        env = AuditEnv(mode="train")
        env.reset(seed=1002, config=self.cfg)
        obs_over, r1, term1, trunc1, _ = env.step(
            {"action": "list_records", "table": "ledger", "cursor": "0", "limit": 51}
        )
        self.assertEqual(obs_over["status"], "error")
        self.assertEqual(obs_over["step"], 1)
        self.assertEqual(r1, 0.0)
        self.assertFalse(term1 or trunc1)

        obs_zero, _, _, _, _ = env.step(
            {"action": "list_records", "table": "ledger", "cursor": "0", "limit": 0}
        )
        self.assertEqual(obs_zero["status"], "error")
        self.assertEqual(obs_zero["step"], 2)

        obs_bool, _, _, _, _ = env.step(
            {"action": "list_records", "table": "ledger", "cursor": "0", "limit": True}
        )
        self.assertEqual(obs_bool["status"], "error")
        self.assertEqual(obs_bool["step"], 3)

    def test_07_get_record_across_all_tables(self) -> None:
        env = AuditEnv(mode="train")
        env.reset(seed=1003, config=self.cfg)
        inv = env.step({"action": "list_records", "table": "invoices", "limit": 1})[0]["records"][0]
        pay = env.step({"action": "list_records", "table": "payments", "limit": 1})[0]["records"][0]
        row = env.step({"action": "list_records", "table": "ledger", "limit": 1})[0]["records"][0]
        fx = env.step({"action": "list_records", "table": "fx", "limit": 1})[0]["records"][0]

        inv_obs = env.step({"action": "get_record", "table": "invoices", "id": inv["invoice_id"]})[0]
        self.assertEqual(inv_obs["status"], "ok")
        self.assertEqual(inv_obs["record"], inv)

        pay_obs = env.step({"action": "get_record", "table": "payments", "id": pay["payment_id"]})[0]
        self.assertEqual(pay_obs["status"], "ok")
        self.assertEqual(pay_obs["record"], pay)

        row_obs = env.step({"action": "get_record", "table": "ledger", "id": row["row_id"]})[0]
        self.assertEqual(row_obs["status"], "ok")
        self.assertEqual(row_obs["record"], row)

        fx_id = f"{fx['date']}:{fx['source_currency']}:{fx['target_currency']}"
        fx_obs = env.step({"action": "get_record", "table": "fx", "id": fx_id})[0]
        self.assertEqual(fx_obs["status"], "ok")
        self.assertEqual(fx_obs["record"], fx)

    def test_08_get_document_invoice_and_payment(self) -> None:
        env = AuditEnv(mode="train")
        env.reset(seed=1004, config=self.cfg)
        inv = env.step({"action": "list_records", "table": "invoices", "limit": 1})[0]["records"][0]
        pay = env.step({"action": "list_records", "table": "payments", "limit": 1})[0]["records"][0]

        doc_inv = env.step({"action": "get_document", "doc_id": inv["doc_id"]})[0]
        self.assertEqual(doc_inv["status"], "ok")
        self.assertEqual(doc_inv["document"]["invoice_id"], inv["invoice_id"])

        doc_pay = env.step({"action": "get_document", "doc_id": pay["doc_id"]})[0]
        self.assertEqual(doc_pay["status"], "ok")
        self.assertEqual(doc_pay["document"]["payment_id"], pay["payment_id"])

    def test_09_calculator_valid_decimal_arithmetic(self) -> None:
        self.assertEqual(safe_calculate("0.1 + 0.2"), "0.3")
        self.assertEqual(safe_calculate("round(2.5)"), "2")  # ROUND_HALF_EVEN
        self.assertEqual(safe_calculate("round(3.5)"), "4")  # ROUND_HALF_EVEN
        self.assertEqual(safe_calculate("abs(-123.45) + min(10, 5) + max(2, 8)"), "136.45")
        self.assertEqual(safe_calculate("100 // 3 + 100 % 3"), "34")

        env = AuditEnv(mode="train")
        env.reset(seed=1005, config=self.cfg)
        obs, reward, _, _, _ = env.step({"action": "calculator", "expression": "12500 * 1.085"})
        self.assertEqual(obs["status"], "ok")
        self.assertEqual(obs["result"], "13562.5")
        self.assertEqual(reward, 0.0)

    def test_10_calculator_rejects_unsafe_expressions_and_syntax_errors(self) -> None:
        unsafe_exprs = [
            "__import__('os').system('echo hi')",
            "eval('1+1')",
            "open('/etc/passwd')",
            "2 ** 1000",
            "(1).__class__.__mro__",
            "[x for x in range(10)]",
            "lambda x: x + 1",
            "1 / 0",
            "1 + * 2",
            "1 + " + "1+" * 150 + "1",  # > 256 chars
            "unknown_var + 1",
            "'hello'",
        ]
        for expr in unsafe_exprs:
            with self.subTest(expr=expr[:40]):
                with self.assertRaises(CalculatorError):
                    safe_calculate(expr)

        # Also verify via env.step returns bounded error and consumes 1 step
        env = AuditEnv(mode="train")
        env.reset(seed=1005, config=self.cfg)
        obs, reward, term, trunc, _ = env.step({"action": "calculator", "expression": "__import__('os')"})
        self.assertEqual(obs["status"], "error")
        self.assertEqual(obs["step"], 1)
        self.assertEqual(reward, 0.0)
        self.assertFalse(term or trunc)

    def test_11_unknown_action_table_id_consume_one_step_with_bounded_error(self) -> None:
        env = AuditEnv(mode="train")
        env.reset(seed=1006, config=self.cfg)

        o1, r1, t1, tr1, _ = env.step({"action": "hack_truth"})
        self.assertEqual(o1["status"], "error")
        self.assertEqual(o1["step"], 1)
        self.assertEqual(r1, 0.0)
        self.assertFalse(t1 or tr1)

        o2, _, _, _, _ = env.step({"action": "list_records", "table": "truth"})
        self.assertEqual(o2["status"], "error")
        self.assertEqual(o2["step"], 2)

        o3, _, _, _, _ = env.step({"action": "get_record", "table": "invoices", "id": "INV-DOES-NOT-EXIST"})
        self.assertEqual(o3["status"], "error")
        self.assertEqual(o3["step"], 3)

        o4, _, _, _, _ = env.step({"action": "get_document", "doc_id": "DOC-DOES-NOT-EXIST"})
        self.assertEqual(o4["status"], "error")
        self.assertEqual(o4["step"], 4)

    def test_12_submit_terminates_episode_cleanly(self) -> None:
        env = AuditEnv(mode="train")
        obs, _ = env.reset(seed=1007, config=EpisodeConfig(num_invoices=20, clean=True))
        ep_id = obs["manifest"]["episode_id"]
        sub_obs, reward, terminated, truncated, info = env.step(
            {"action": "submit", "findings": {"episode_id": ep_id, "findings": []}}
        )
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, 1.0)
        self.assertEqual(sub_obs["status"], "submitted")
        self.assertTrue(info["terminated"])

    def test_13_step_limit_100_truncates_with_empty_answer(self) -> None:
        env = AuditEnv(mode="train", max_steps=100)
        env.reset(seed=1008, config=EpisodeConfig(num_invoices=20, clean=True))
        terminated = False
        truncated = False
        reward = 0.0
        for i in range(100):
            _, reward, terminated, truncated, _ = env.step(
                {"action": "calculator", "expression": "1 + 1"}
            )
            if i < 99:
                self.assertFalse(terminated)
                self.assertFalse(truncated)
                self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        # Clean episode with empty answer and 100 inspection steps: 2*1.0 - 1.0 - 0.002*100 = 0.8
        self.assertAlmostEqual(reward, 0.8, places=6)
        report = env.get_evaluator_report()
        self.assertEqual(report["submitted_count"], 0)
        self.assertTrue(report["truncated"])
        self.assertFalse(report["terminated"])

    def test_14_stepping_after_termination_does_not_mutate_reward(self) -> None:
        env = AuditEnv(mode="train")
        obs, _ = env.reset(seed=1009, config=EpisodeConfig(num_invoices=20, clean=True))
        ep_id = obs["manifest"]["episode_id"]
        _, reward, terminated, truncated, _ = env.step(
            {"action": "submit", "findings": {"episode_id": ep_id, "findings": []}}
        )
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, 1.0)
        report_before = dict(env.get_evaluator_report())

        post_obs, post_reward, post_term, post_trunc, _ = env.step(
            {"action": "list_records", "table": "invoices", "limit": 5}
        )
        self.assertEqual(post_obs["status"], "error")
        self.assertEqual(post_reward, 0.0)
        self.assertTrue(post_term)
        self.assertFalse(post_trunc)
        self.assertEqual(env.get_evaluator_report(), report_before)

    def test_15_train_vs_eval_mode_reward_hiding(self) -> None:
        clean_cfg = EpisodeConfig(num_invoices=20, clean=True)

        train_env = AuditEnv(mode="train")
        t_obs, _ = train_env.reset(seed=1010, config=clean_cfg)
        _, t_reward, t_term, _, _ = train_env.step(
            {"action": "submit", "findings": {"episode_id": t_obs["manifest"]["episode_id"], "findings": []}}
        )
        self.assertTrue(t_term)
        self.assertEqual(t_reward, 1.0)

        eval_env = AuditEnv(mode="eval")
        e_obs, _ = eval_env.reset(seed=1010, config=clean_cfg)
        e_sub_obs, e_reward, e_term, _, e_info = eval_env.step(
            {"action": "submit", "findings": {"episode_id": e_obs["manifest"]["episode_id"], "findings": []}}
        )
        self.assertTrue(e_term)
        # Eval mode MUST hide reward (returns 0.0) in step()
        self.assertEqual(e_reward, 0.0)
        self.assertNotIn("reward", e_sub_obs)
        self.assertNotIn("metrics", e_sub_obs)
        self.assertNotIn("reward", e_info)
        self.assertNotIn("metrics", e_info)
        # Evaluator report still holds the true score
        report = eval_env.get_evaluator_report()
        self.assertEqual(report["reward"], 1.0)
        self.assertEqual(report["f1"], 1.0)

    def test_16_zero_leakage_of_truth_provenance_seed_or_paths(self) -> None:
        env = AuditEnv(mode="eval")
        obs0, info0 = env.reset(seed=1337, config=self.cfg)
        ep_id = obs0["manifest"]["episode_id"]

        payloads_to_check: list[Any] = [obs0, info0]
        for action in (
            {"action": "list_records", "table": "invoices", "limit": 5},
            {"action": "list_records", "table": "payments", "limit": 5},
            {"action": "list_records", "table": "ledger", "limit": 5},
            {"action": "list_records", "table": "fx", "limit": 5},
            {"action": "calculator", "expression": "100 + 200"},
            {"action": "submit", "findings": {"episode_id": ep_id, "findings": []}},
        ):
            o, _, _, _, inf = env.step(action)
            payloads_to_check.extend([o, inf])

        for item in payloads_to_check:
            keys = _collect_all_keys(item)
            leaked = keys & FORBIDDEN_KEYS
            self.assertEqual(leaked, set(), f"Leaked forbidden keys: {leaked}")
            self.assertFalse(_contains_filesystem_path(item), f"Leaked filesystem path in {item}")

    def test_17_subprocess_audit_env_out_of_process_isolation(self) -> None:
        with SubprocessAuditEnv(mode="eval", max_steps=100) as env:
            # Verify agent-side wrapper holds no truth/provenance/tables in its own __dict__
            self.assertNotIn("_truth", env.__dict__)
            self.assertNotIn("_provenance", env.__dict__)
            self.assertNotIn("_tables", env.__dict__)

            obs, info = env.reset(seed=2026, config=EpisodeConfig(num_invoices=20, clean=True))
            ep_id = obs["manifest"]["episode_id"]
            self.assertEqual(info["mode"], "eval")
            self.assertEqual(_collect_all_keys(obs) & FORBIDDEN_KEYS, set())

            list_obs, r1, term1, trunc1, _ = env.step(
                {"action": "list_records", "table": "invoices", "limit": 5}
            )
            self.assertEqual(list_obs["status"], "ok")
            self.assertEqual(r1, 0.0)
            self.assertFalse(term1 or trunc1)

            calc_obs, _, _, _, _ = env.step({"action": "calculator", "expression": "50 * 2"})
            self.assertEqual(calc_obs["result"], "100")

            sub_obs, r_sub, term_sub, trunc_sub, _ = env.step(
                {"action": "submit", "findings": {"episode_id": ep_id, "findings": []}}
            )
            self.assertTrue(term_sub)
            self.assertFalse(trunc_sub)
            self.assertEqual(r_sub, 0.0)  # eval mode hides reward
            self.assertEqual(sub_obs["status"], "submitted")

            report = env.get_evaluator_report()
            self.assertEqual(report["f1"], 1.0)
            # 2 inspection steps: 1.0 - 0.002 * 2 = 0.996
            self.assertAlmostEqual(report["reward"], 0.996, places=6)

    def test_18_subprocess_audit_env_train_mode_and_episode_override(self) -> None:
        ep = generate_episode(seed=3030, config=EpisodeConfig(num_invoices=20, clean=True))
        with SubprocessAuditEnv(mode="train", max_steps=50) as env:
            obs, _ = env.reset(seed=3030, episode=ep)
            self.assertEqual(obs["manifest"]["episode_id"], ep.manifest["episode_id"])
            _, reward, terminated, truncated, _ = env.step(
                {"action": "submit", "findings": {"episode_id": ep.manifest["episode_id"], "findings": []}}
            )
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(reward, 1.0)

    def test_19_invalid_cursor_and_non_dict_action_handled_gracefully(self) -> None:
        env = AuditEnv(mode="train")
        env.reset(seed=1011, config=self.cfg)
        o1, r1, t1, tr1, _ = env.step({"action": "list_records", "table": "invoices", "cursor": "-5", "limit": 10})
        self.assertEqual(o1["status"], "error")
        self.assertEqual(r1, 0.0)
        self.assertFalse(t1 or tr1)

        o2, r2, t2, tr2, _ = env.step("not-a-dict")  # type: ignore[arg-type]
        self.assertEqual(o2["status"], "error")
        self.assertEqual(o2["step"], 2)
        self.assertEqual(r2, 0.0)
        self.assertFalse(t2 or tr2)

    def test_20_invalid_init_args_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AuditEnv(mode="invalid")
        with self.assertRaises(ValueError):
            AuditEnv(max_steps=0)
        with self.assertRaises(ValueError):
            AuditEnv(timeout_seconds=-1.0)

    def test_21_observations_are_deep_copies_isolated_from_env_state(self) -> None:
        env = AuditEnv(mode="train")
        env.reset(seed=1012, config=self.cfg)
        o1 = env.step({"action": "list_records", "table": "invoices", "cursor": "0", "limit": 1})[0]
        orig_id = o1["records"][0]["invoice_id"]
        o1["records"][0]["invoice_id"] = "MUTATED-BY-AGENT"

        o2 = env.step({"action": "list_records", "table": "invoices", "cursor": "0", "limit": 1})[0]
        self.assertEqual(o2["records"][0]["invoice_id"], orig_id)


if __name__ == "__main__":
    unittest.main()
