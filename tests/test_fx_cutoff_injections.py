from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from decimal import Decimal
from itertools import count
from unittest import TestCase

from genesis.domain.money import convert_minor
from genesis.evaluation.matcher import match
from genesis.evaluation.public_proofs import verify_public_proofs
from genesis.generation import generate_episode
from genesis.generation.fx_cutoff import apply_fx_cutoff_errors, in_period


class FxCutoffInjectionTests(TestCase):
    def setUp(self):
        clean = generate_episode(413, {"schema_version": "1.0.0", "error_counts": {}})
        self.public = clean.public

    def inject(self, counts, public=None):
        ids = count()
        return apply_fx_cutoff_errors(public or self.public, counts, lambda prefix: f"{prefix}-{next(ids)}")

    def test_fx_discrepancy_is_recomputed_using_public_quote_and_balanced_legs(self):
        ledger, [issue], [change] = self.inject({"fx_conversion": 1})
        self.assertEqual(issue["type"], "fx_conversion")
        debit, credit = sorted(
            (row for row in ledger if row["row_id"] in issue["affected_ids"]),
            key=lambda row: row["credit_minor"],
        )
        payment = next(doc for doc in self.public.payments if doc["payment_id"] == debit["payment_id"])
        day = payment["timestamp"][:10]
        rate = next(row["rate"] for row in self.public.fx if row["date"] == day)
        self.assertEqual(payment["currency"], "JPY")
        self.assertEqual(debit["currency"], "KZT")
        expected = (Decimal(payment["amount_minor"]) * Decimal(rate) * 100).quantize(Decimal(1))
        self.assertEqual(debit["debit_minor"], credit["credit_minor"])
        self.assertEqual(debit["credit_minor"], 0)
        self.assertEqual(credit["debit_minor"], 0)
        self.assertEqual(issue["expected_amount_minor"], abs(expected - debit["debit_minor"]))
        self.assertEqual(change["before"]["posted_minor"], expected)
        self.assertEqual(change["after"]["posted_minor"], debit["debit_minor"])
        self.assertEqual(issue["issue_id"], change["issue_id"])
        self.assertEqual(set(issue["acceptable_evidence_sets"][0]), {
            debit["row_id"], credit["row_id"], payment["doc_id"], f"fx:{day}:JPY:KZT", "policy",
        })

    def test_rounding_on_half_a_target_minor_unit(self):
        self.assertEqual(convert_minor(1, "JPY", "KZT", "1.005", {"JPY": 0, "KZT": 2}, "ROUND_HALF_EVEN"), 100)
        self.assertEqual(convert_minor(1, "JPY", "KZT", "1.005", {"JPY": 0, "KZT": 2}, "ROUND_HALF_UP"), 101)
        payment = next(doc for doc in self.public.payments if doc["currency"] == "JPY")
        invoice_id, = payment["invoice_refs"]
        public = replace(
            self.public,
            payments=[{**doc, "amount_minor": 1} if doc is payment else doc for doc in self.public.payments],
            invoices=[
                {**doc, "total_minor": 1, "line_items": [
                    {**doc["line_items"][0], "quantity": 1, "unit_price_minor": 1, "amount_minor": 1}
                ]} if doc["invoice_id"] == invoice_id else doc for doc in self.public.invoices
            ],
            fx=[{**row, "rate": "1.005"} for row in self.public.fx],
            ledger=[
                {**row, "debit_minor": 100 if row["debit_minor"] else 0,
                 "credit_minor": 100 if row["credit_minor"] else 0}
                if row["invoice_id"] == invoice_id else row for row in self.public.ledger
            ],
        )
        ledger, [issue], [change] = self.inject({"fx_conversion": 1}, public)
        self.assertEqual(change["before"]["posted_minor"], 100)
        self.assertEqual(change["after"]["posted_minor"], 102)
        self.assertEqual(issue["expected_amount_minor"], 2)
        self.assertEqual(
            {(row["debit_minor"], row["credit_minor"]) for row in ledger if row["row_id"] in issue["affected_ids"]},
            {(102, 0), (0, 102)},
        )

    def test_alternative_date_rules_use_public_dates_and_missing_quote_rejects(self):
        for rule, doc_date in (
            ("payment_date", "timestamp"), ("posting_date", "timestamp"), ("invoice_date", "issued_at"),
        ):
            with self.subTest(rule=rule):
                payment = next(doc for doc in self.public.payments if doc["currency"] == "JPY")
                invoice = next(doc for doc in self.public.invoices if doc["invoice_id"] in payment["invoice_refs"])
                day = (invoice if doc_date == "issued_at" else payment)[doc_date][:10]
                public = replace(self.public, policy={**self.public.policy, "fx_date_rule": rule})
                _, [issue], _ = self.inject({"fx_conversion": 1}, public)
                evidence = {
                    *issue["affected_ids"], payment["doc_id"], f"fx:{day}:JPY:KZT", "policy",
                }
                if rule == "invoice_date":
                    evidence.add(invoice["doc_id"])
                evidence_set, = issue["acceptable_evidence_sets"]
                self.assertCountEqual(evidence_set, evidence)
                if rule == "invoice_date":
                    finding = {
                        "type": issue["type"], "affected_ids": issue["affected_ids"],
                        "claimed_amount_minor": issue["expected_amount_minor"],
                        "currency": issue["currency"], "evidence_ids": sorted(evidence), "explanation": "",
                    }
                    self.assertEqual(match([finding], [issue], evidence).tp, 1)
                    finding["evidence_ids"].remove(invoice["doc_id"])
                    result = match([finding], [issue], evidence)
                    self.assertEqual((result.tp, result.fp, result.fn), (0, 1, 1))
                public.fx = [row for row in public.fx if row["date"] != day]
                with self.assertRaisesRegex(ValueError, "fx_conversion candidates"):
                    self.inject({"fx_conversion": 1}, public)

    def test_period_start_included_and_end_excluded_for_equivalent_offsets(self):
        policy = {"period_start": "2026-12-01T00:00:00+05:00", "period_end": "2027-01-01T00:00:00+05:00"}
        for timestamp in (policy["period_start"], "2026-11-30T19:00:00Z", "2026-12-31T18:59:59Z"):
            self.assertTrue(in_period(timestamp, policy), timestamp)
        for timestamp in (policy["period_end"], "2026-12-31T19:00:00Z", "2026-11-30T18:59:59Z"):
            self.assertFalse(in_period(timestamp, policy), timestamp)

    def test_moved_pair_is_balanced_and_primary_document_proves_cutoff(self):
        ledger, [issue], [change] = self.inject({"period_cutoff": 1})
        self.assertEqual(issue["type"], "period_cutoff")
        period_start = datetime.fromisoformat(self.public.policy["period_start"])
        period_end = datetime.fromisoformat(self.public.policy["period_end"])
        affected = [row for row in ledger if row["row_id"] in issue["affected_ids"]]
        invoice = next(doc for doc in self.public.invoices if doc["invoice_id"] == affected[0]["invoice_id"])
        self.assertLessEqual(period_start, datetime.fromisoformat(invoice["issued_at"]))
        self.assertLess(datetime.fromisoformat(invoice["issued_at"]), period_end)
        self.assertTrue(all(not in_period(row["timestamp"], self.public.policy) for row in affected))
        self.assertEqual(issue["expected_amount_minor"], invoice["total_minor"])
        self.assertEqual(issue["issue_id"], change["issue_id"])
        self.assertEqual(change["before"]["timestamp"], invoice["issued_at"])
        self.assertEqual({row["timestamp"] for row in affected}, {change["after"]["timestamp"]})
        self.assertEqual(set(issue["acceptable_evidence_sets"][0]), {
            *issue["affected_ids"], invoice["doc_id"], "policy",
        })
        balances = defaultdict(int)
        for row in ledger:
            balances[row["currency"], in_period(row["timestamp"], self.public.policy)] += (
                row["debit_minor"] - row["credit_minor"]
            )
        self.assertEqual(set(balances.values()), {0})

    def test_excess_counts_fail_clearly_and_mutations_do_not_overlap(self):
        for kind in ("fx_conversion", "period_cutoff"):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, f"{kind} candidates"):
                self.inject({kind: 2})
        counts = {"fx_conversion": 1, "period_cutoff": 1}
        actual = self.inject(counts)
        self.assertEqual(actual, self.inject(counts))
        ledger, [fx, period], _ = actual
        self.assertTrue(set(fx["affected_ids"]).isdisjoint(period["affected_ids"]))
        self.assertNotEqual(ledger, self.public.ledger)
        self.assertEqual(self.public.ledger, generate_episode(
            413, {"schema_version": "1.0.0", "error_counts": {}},
        ).public.ledger)

    def test_cutoff_covers_both_period_start_and_period_end_boundaries(self):
        sides = set()
        for seed in range(20):
            episode = generate_episode(seed, {"schema_version": "1.0.0", "error_counts": {"period_cutoff": 1}})
            policy = episode.public.policy
            ts = episode.evaluator.provenance["changes"][0]["after"]["timestamp"]
            self.assertFalse(in_period(ts, policy))
            if datetime.fromisoformat(ts) < datetime.fromisoformat(policy["period_start"]):
                sides.add("before_start")
            elif datetime.fromisoformat(ts) >= datetime.fromisoformat(policy["period_end"]):
                sides.add("after_end")
        self.assertEqual(sides, {"before_start", "after_end"})

        scaled = generate_episode(
            1001, {"schema_version": "1.0.0", "obligation_count": 20, "error_counts": {"period_cutoff": 2}},
        )
        policy = scaled.public.policy
        scaled_sides = {
            "before_start"
            if datetime.fromisoformat(change["after"]["timestamp"]) < datetime.fromisoformat(policy["period_start"])
            else "after_end"
            for change in scaled.evaluator.provenance["changes"]
        }
        self.assertEqual(scaled_sides, {"before_start", "after_end"})

    def test_invoice_date_rule_generates_reconcilable_fx_errors(self):
        episode = generate_episode(
            1004,
            {
                "schema_version": "1.0.0", "obligation_count": 20,
                "base_currency": "GBP", "foreign_currency": "USD",
                "timezone": "UTC", "fx_date_rule": "invoice_date",
                "error_counts": {"fx_conversion": 2},
            },
        )
        self.assertEqual(episode.public.policy["fx_date_rule"], "invoice_date")
        self.assertEqual(len(episode.evaluator.truth["issues"]), 2)
        invoice_docs = {doc["doc_id"] for doc in episode.public.invoices}
        for issue in episode.evaluator.truth["issues"]:
            self.assertEqual(issue["type"], "fx_conversion")
            self.assertTrue(set(issue["acceptable_evidence_sets"][0]) & invoice_docs)
        verify_public_proofs(episode)

        compact = generate_episode(
            413,
            {
                "schema_version": "1.0.0",
                "fx_date_rule": "invoice_date",
                "error_counts": {"fx_conversion": 1},
            },
        )
        verify_public_proofs(compact)
