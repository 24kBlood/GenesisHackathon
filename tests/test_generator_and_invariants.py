"""Unit and invariant tests for episode generation, schemas, and serialization."""

from __future__ import annotations

import re
import tempfile
import unittest
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from audit_rl.generator import (
    generate_episode,
    permute_and_relabel_episode,
    verify_episode_integrity,
)
from audit_rl.schemas import (
    ERROR_TYPES,
    INTERACTING_SCENARIOS,
    SCHEMA_VERSION,
    EpisodeConfig,
    load_episode,
    load_episode_open,
    load_episode_private,
    save_episode,
)

ISO_OFFSET_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2}|Z)$")


class TestGeneratorAndInvariants(unittest.TestCase):
    def test_clean_episode_zero_errors_and_balanced(self) -> None:
        cfg = EpisodeConfig(num_invoices=35, clean=True, include_distractors=True)
        ep = generate_episode(seed=101, config=cfg)

        self.assertEqual(ep.truth["issues"], [])
        self.assertEqual(ep.manifest["schema_version"], SCHEMA_VERSION)
        self.assertEqual(ep.truth["episode_id"], ep.manifest["episode_id"])

        # Pre-injection accounting balance checks
        pre = ep.provenance["pre_injection_checks"]
        self.assertTrue(pre["debits_equal_credits"])
        self.assertEqual(pre["total_debit_minor"], pre["total_credit_minor"])
        self.assertGreater(pre["total_debit_minor"], 0)

        # Post-injection ledger balance check
        total_debit = sum(r["debit_minor"] for r in ep.ledger)
        total_credit = sum(r["credit_minor"] for r in ep.ledger)
        self.assertEqual(total_debit, total_credit)

        report = verify_episode_integrity(ep)
        self.assertTrue(report["ok"], msg=f"Integrity errors: {report['errors']}")
        self.assertTrue(report["checks"]["pre_injection_balanced"])
        self.assertTrue(report["checks"]["post_injection_balanced"])

    def test_clean_flag_overrides_configured_errors(self) -> None:
        cfg = EpisodeConfig(
            num_invoices=30,
            error_counts={"duplicate_invoice": 2, "fx_conversion_error": 1},
            interacting_scenarios=["fx_period_partial_payment"],
            clean=True,
        )
        ep = generate_episode(seed=202, config=cfg)
        self.assertEqual(ep.truth["issues"], [])
        self.assertEqual(ep.provenance["injected_events"], [])
        self.assertEqual(ep.provenance["interacting_groups"], [])
        self.assertTrue(verify_episode_integrity(ep)["ok"])

    def test_distractors_exist_in_clean_episode_without_false_issues(self) -> None:
        cfg = EpisodeConfig(num_invoices=40, clean=True, include_distractors=True)
        ep = generate_episode(seed=303, config=cfg)

        self.assertEqual(ep.truth["issues"], [])
        distractors = ep.provenance["distractors"]
        self.assertIsInstance(distractors, dict)

        expected_keys = (
            "split_payments",
            "refunds",
            "recurring_invoices",
            "multicurrency_invoices",
            "boundary_timestamps",
        )
        for key in expected_keys:
            self.assertIn(key, distractors, f"Missing distractor category {key}")
            self.assertGreater(
                len(distractors[key]),
                0,
                f"Distractor category {key} should be non-empty in clean episode",
            )

        # Also check multi-currency invoices actually exist in open records
        foreign_invs = [i for i in ep.invoices if i["currency"] != ep.policy["base_currency"]]
        self.assertGreater(len(foreign_invs), 0)
        self.assertTrue(verify_episode_integrity(ep)["ok"])

    def test_each_error_type_independently_configurable(self) -> None:
        for idx, error_type in enumerate(ERROR_TYPES):
            with self.subTest(error_type=error_type):
                cfg = EpisodeConfig(
                    num_invoices=35,
                    error_counts={error_type: 2},
                    include_distractors=True,
                )
                ep = generate_episode(seed=400 + idx, config=cfg)
                issues = ep.truth["issues"]
                self.assertEqual(len(issues), 2)
                self.assertTrue(all(issue["type"] == error_type for issue in issues))
                report = verify_episode_integrity(ep)
                self.assertTrue(report["ok"], msg=f"{error_type} failed integrity: {report['errors']}")
                self.assertTrue(report["checks"]["all_evidence_ids_exist"])
                self.assertTrue(report["checks"]["all_issues_independently_verified"])

    def test_all_four_error_types_together(self) -> None:
        cfg = EpisodeConfig(
            num_invoices=50,
            error_counts={et: 1 for et in ERROR_TYPES},
            include_distractors=True,
        )
        ep = generate_episode(seed=505, config=cfg)
        issue_types = {issue["type"] for issue in ep.truth["issues"]}
        self.assertEqual(issue_types, set(ERROR_TYPES))
        self.assertEqual(len(ep.truth["issues"]), 4)
        report = verify_episode_integrity(ep)
        self.assertTrue(report["ok"], msg=report["errors"])

    def test_interacting_scenario_fx_period_partial_payment(self) -> None:
        cfg = EpisodeConfig(
            num_invoices=40,
            interacting_scenarios=["fx_period_partial_payment"],
            include_distractors=True,
        )
        ep = generate_episode(seed=601, config=cfg)
        issues = ep.truth["issues"]
        self.assertGreaterEqual(len(issues), 2)
        types = {i["type"] for i in issues}
        self.assertTrue(
            {"fx_conversion_error", "payment_allocation_mismatch"}.issubset(types),
            f"Unexpected issue types for fx_period_partial_payment: {types}",
        )
        groups = ep.provenance["interacting_groups"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["scenario"], "fx_period_partial_payment")
        self.assertGreaterEqual(len(groups[0]["issue_ids"]), 2)
        # Verify both issues are linked via the same shared invoice
        shared_inv = groups[0]["shared_invoice_id"]
        for iss in issues:
            self.assertIn(shared_inv, iss["affected_ids"])

        report = verify_episode_integrity(ep)
        self.assertTrue(report["ok"], msg=report["errors"])

    def test_interacting_scenario_duplicate_invoice_fx_mismatch(self) -> None:
        cfg = EpisodeConfig(
            num_invoices=40,
            interacting_scenarios=["duplicate_invoice_fx_mismatch"],
            include_distractors=True,
        )
        ep = generate_episode(seed=602, config=cfg)
        issues = ep.truth["issues"]
        self.assertGreaterEqual(len(issues), 2)
        types = {i["type"] for i in issues}
        self.assertTrue(
            {"duplicate_invoice", "fx_conversion_error"}.issubset(types),
            f"Unexpected issue types for duplicate_invoice_fx_mismatch: {types}",
        )
        groups = ep.provenance["interacting_groups"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["scenario"], "duplicate_invoice_fx_mismatch")
        self.assertGreaterEqual(len(groups[0]["issue_ids"]), 2)

        report = verify_episode_integrity(ep)
        self.assertTrue(report["ok"], msg=report["errors"])

    def test_both_interacting_scenarios_combined(self) -> None:
        cfg = EpisodeConfig(
            num_invoices=45,
            interacting_scenarios=list(INTERACTING_SCENARIOS),
        )
        ep = generate_episode(seed=603, config=cfg)
        self.assertGreaterEqual(len(ep.truth["issues"]), 4)
        self.assertEqual(len(ep.provenance["interacting_groups"]), 2)
        report = verify_episode_integrity(ep)
        self.assertTrue(report["ok"], msg=report["errors"])

    def test_row_and_doc_counts_and_value_formats(self) -> None:
        configs = [
            EpisodeConfig(num_invoices=20, timezone_offset="+00:00", period_start="2026-01-01T00:00:00+00:00", period_end="2026-03-31T23:59:59+00:00"),
            EpisodeConfig(num_invoices=60, base_currency="EUR", timezone_offset="-04:00", period_start="2026-04-01T00:00:00-04:00", period_end="2026-06-30T23:59:59-04:00", fx_date_policy="posting_date", rounding_rule="ROUND_HALF_UP"),
            EpisodeConfig(num_invoices=90, base_currency="KZT", timezone_offset="+05:00", fx_date_policy="payment_date"),
        ]
        for idx, cfg in enumerate(configs):
            with self.subTest(idx=idx, num_invoices=cfg.num_invoices):
                ep = generate_episode(seed=700 + idx, config=cfg)
                self.assertGreaterEqual(len(ep.ledger), 200)
                self.assertLessEqual(len(ep.ledger), 2000)
                doc_count = len(ep.invoices) + len(ep.payments)
                self.assertGreaterEqual(doc_count, 20)
                self.assertLessEqual(doc_count, 200)

                # ISO-8601 timestamps with explicit offset
                self.assertRegex(ep.manifest["period_start"], ISO_OFFSET_RE)
                self.assertRegex(ep.manifest["period_end"], ISO_OFFSET_RE)

                # Ledger value types and explicit nulls
                has_explicit_null_inv = False
                has_explicit_null_pay = False
                for row in ep.ledger:
                    self.assertRegex(row["timestamp"], ISO_OFFSET_RE)
                    datetime.fromisoformat(row["timestamp"])
                    self.assertIsInstance(row["debit_minor"], int)
                    self.assertNotIsInstance(row["debit_minor"], bool)
                    self.assertIsInstance(row["credit_minor"], int)
                    self.assertNotIsInstance(row["credit_minor"], bool)
                    self.assertIn("entity_id", row)
                    self.assertIn("invoice_id", row)
                    self.assertIn("payment_id", row)
                    if row["invoice_id"] is None:
                        has_explicit_null_inv = True
                    if row["payment_id"] is None:
                        has_explicit_null_pay = True
                self.assertTrue(has_explicit_null_inv)
                self.assertTrue(has_explicit_null_pay)

                # Invoices integer minor units & ISO timestamps
                for inv in ep.invoices:
                    self.assertRegex(inv["issued_at"], ISO_OFFSET_RE)
                    self.assertRegex(inv["due_at"], ISO_OFFSET_RE)
                    self.assertIsInstance(inv["total_minor"], int)
                    self.assertNotIsInstance(inv["total_minor"], bool)
                    for li in inv["line_items"]:
                        self.assertIsInstance(li["unit_price_minor"], int)
                        self.assertIsInstance(li["amount_minor"], int)

                # Payments integer minor units & ISO timestamps
                for pay in ep.payments:
                    self.assertRegex(pay["timestamp"], ISO_OFFSET_RE)
                    self.assertIsInstance(pay["amount_minor"], int)
                    self.assertNotIsInstance(pay["amount_minor"], bool)

                # FX rates are decimal strings
                for fx in ep.fx_rates:
                    self.assertIsInstance(fx["rate"], str)
                    val = Decimal(fx["rate"])
                    self.assertGreater(val, Decimal("0"))

                self.assertTrue(verify_episode_integrity(ep)["ok"])

    def test_save_load_roundtrip_preserves_exact_data(self) -> None:
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
        ep = generate_episode(seed=808, config=cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            full_dir = Path(tmpdir) / "full"
            save_episode(ep, full_dir, include_private=True)

            # All expected files exist
            for fname in (
                "manifest.json",
                "policy.json",
                "ledger.csv",
                "invoices.jsonl",
                "payments.jsonl",
                "fx.csv",
                "truth.json",
                "provenance.json",
            ):
                self.assertTrue((full_dir / fname).is_file(), f"Missing {fname}")

            loaded = load_episode(full_dir)
            self.assertEqual(loaded.manifest, ep.manifest)
            self.assertEqual(loaded.policy, ep.policy)
            self.assertEqual(loaded.ledger, ep.ledger)
            self.assertEqual(loaded.invoices, ep.invoices)
            self.assertEqual(loaded.payments, ep.payments)
            self.assertEqual(loaded.fx_rates, ep.fx_rates)
            self.assertEqual(loaded.truth, ep.truth)
            self.assertEqual(loaded.provenance, ep.provenance)

            # Open-only save omits private files
            open_dir = Path(tmpdir) / "open_only"
            save_episode(ep, open_dir, include_private=False)
            self.assertFalse((open_dir / "truth.json").exists())
            self.assertFalse((open_dir / "provenance.json").exists())
            open_loaded = load_episode_open(open_dir)
            self.assertEqual(open_loaded["ledger"], ep.ledger)
            self.assertEqual(open_loaded["invoices"], ep.invoices)
            self.assertEqual(open_loaded["payments"], ep.payments)
            self.assertEqual(open_loaded["fx_rates"], ep.fx_rates)

            # Private loader matches truth/provenance on full_dir
            priv_loaded = load_episode_private(full_dir)
            self.assertEqual(priv_loaded["truth"], ep.truth)
            self.assertEqual(priv_loaded["provenance"], ep.provenance)

    def test_episode_config_roundtrip(self) -> None:
        cfg = EpisodeConfig(
            num_invoices=42,
            base_currency="EUR",
            currencies=("EUR", "USD", "JPY"),
            fx_date_policy="posting_date",
            rounding_rule="ROUND_HALF_UP",
            error_counts={"duplicate_invoice": 1},
            interacting_scenarios=["duplicate_invoice_fx_mismatch"],
        )
        restored = EpisodeConfig.from_dict(cfg.to_dict())
        self.assertEqual(restored.to_dict(), cfg.to_dict())

    def test_permute_and_relabel_basic_integrity(self) -> None:
        cfg = EpisodeConfig(num_invoices=30, error_counts={"duplicate_invoice": 1, "fx_conversion_error": 1})
        ep = generate_episode(seed=909, config=cfg)
        remapped, id_map = permute_and_relabel_episode(ep, perm_seed=42)
        self.assertNotEqual(remapped.manifest["episode_id"], ep.manifest["episode_id"])
        self.assertIn(ep.manifest["episode_id"], id_map)
        report = verify_episode_integrity(remapped)
        self.assertTrue(report["ok"], msg=report["errors"])


if __name__ == "__main__":
    unittest.main()
