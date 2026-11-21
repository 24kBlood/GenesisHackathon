from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, localcontext
from unittest import TestCase
from zoneinfo import ZoneInfo

from genesis.evaluation.matcher import match
from genesis.generation import generate_episode
from genesis.generation.interactions import DUPLICATE_WITH_FX, PARTIAL_FX_AT_CUTOFF


def _config(scenarios, counts=None):
    return {"schema_version": "1.0.0", "interaction_scenarios": scenarios, "error_counts": counts or {}}


def _finding(kind, amount, currency, affected, evidence):
    return {
        "type": kind, "claimed_amount_minor": amount, "currency": currency,
        "affected_ids": affected, "evidence_ids": evidence, "explanation": "Independent public reconciliation",
    }


class InteractionTests(TestCase):
    def _reconcile_public(self, public, scenario):
        fx = {f"fx:{row['date']}:{row['source_currency']}:{row['target_currency']}": row["rate"] for row in public.fx}
        zone = ZoneInfo(public.policy["timezone"])
        currency = public.manifest["base_currency"]
        exponents = public.manifest["currency_exponents"]

        def expected(amount, source, quote_key):
            with localcontext() as ctx:
                ctx.prec = 28
                return int((
                    Decimal(amount) / 10 ** exponents[source] * Decimal(fx[quote_key]) * 10 ** exponents[currency]
                ).quantize(Decimal(1), rounding=public.policy["rounding_mode"]))

        balances = defaultdict(int)
        for row in public.ledger:
            balances[row["currency"], row["timestamp"]] += row["debit_minor"] - row["credit_minor"]
        self.assertEqual(set(balances.values()), {0})
        self.assertEqual(public.policy["fx_date_rule"], "payment_date")

        invoices = {doc["invoice_id"]: doc for doc in public.invoices}
        mismatches = []
        for payment in public.payments:
            if payment["currency"] == currency:
                continue
            rows = [row for row in public.ledger if row["payment_id"] == payment["payment_id"]]
            bank, = [row for row in rows if row["account"] == "bank"]
            payment_day = datetime.fromisoformat(payment["timestamp"]).astimezone(zone).date()
            source = payment["currency"]
            key = f"fx:{payment_day}:{source}:{currency}"
            want = expected(payment["amount_minor"], source, key)
            if bank["credit_minor"] != want:
                mismatches.append((payment, bank, rows, payment_day, key, want))
        (payment, bank, rows, payment_day, key, want), = mismatches
        invoice_id, = payment["invoice_refs"]
        invoice = invoices[invoice_id]
        allocation, = [row for row in rows if row["account"] == "accounts_payable"]
        accrual = [row for row in public.ledger if row["invoice_id"] == invoice_id and row["payment_id"] is None]
        catalog = {"policy", *fx, *(row["row_id"] for row in public.ledger),
                   *(doc["doc_id"] for doc in (*public.invoices, *public.payments))}
        if scenario == PARTIAL_FX_AT_CUTOFF:
            paid_at = datetime.fromisoformat(payment["timestamp"])
            period_end = datetime.fromisoformat(public.policy["period_end"])
            self.assertEqual(period_end - paid_at, timedelta(minutes=15))
            self.assertEqual(paid_at.astimezone(timezone.utc).astimezone(zone).date(), payment_day)
            next_key = f"fx:{period_end.astimezone(zone).date()}:{payment['currency']}:{currency}"
            self.assertNotEqual(fx[key], fx[next_key])
            self.assertEqual(bank["credit_minor"], expected(payment["amount_minor"], payment["currency"], next_key))
            payable, = [row for row in accrual if row["account"] == "accounts_payable"]
            suspense, = [row for row in rows if row["account"] == "suspense"]
            amount = payable["credit_minor"] - allocation["debit_minor"]
            self.assertEqual(payable["credit_minor"], want)
            self.assertGreater(amount, 0)
            self.assertNotEqual(amount, abs(want - bank["credit_minor"]))
            evidence = [
                invoice["doc_id"], payment["doc_id"], "policy", key, payable["row_id"],
                bank["row_id"], allocation["row_id"], suspense["row_id"],
            ]
            affected = [invoice_id, payment["payment_id"], allocation["row_id"], suspense["row_id"]]
            causal = _finding(
                "payment_allocation", amount, currency, affected, evidence,
            )
            fx_ids = [bank["row_id"], suspense["row_id"]]
            further_evidence = [next_key]
        else:
            issued_day = datetime.fromisoformat(invoice["issued_at"]).astimezone(zone).date()
            accrual_key = f"fx:{issued_day}:{invoice['currency']}:{currency}"
            recognized = expected(invoice["total_minor"], invoice["currency"], accrual_key)
            expenses = [row["debit_minor"] for row in accrual if row["account"] == "expense"]
            self.assertCountEqual(expenses, [recognized, recognized])
            self.assertEqual(len(accrual), 4)
            row_ids = [row["row_id"] for row in accrual]
            evidence = [invoice["doc_id"], "policy", accrual_key, *row_ids]
            causal = _finding(
                "duplicate_invoice", sum(expenses) - recognized, currency, [invoice_id, *row_ids], evidence,
            )
            fx_ids = [bank["row_id"], allocation["row_id"]]
            further_evidence = []
        finding_fx = _finding(
            "fx_conversion", abs(want - bank["credit_minor"]), currency, fx_ids,
            [payment["doc_id"], "policy", key, *fx_ids, *further_evidence],
        )
        return [causal, finding_fx], catalog

    def test_both_scenarios_reconcile_publicly_and_match_independently(self):
        for scenario in (PARTIAL_FX_AT_CUTOFF, DUPLICATE_WITH_FX):
            with self.subTest(scenario=scenario):
                episode = generate_episode(413, _config([scenario]))
                findings, catalog = self._reconcile_public(episode.public, scenario)
                truth = episode.evaluator.truth["issues"]
                changes = episode.evaluator.provenance["changes"]
                self.assertEqual(len({entry["issue_id"] for entry in truth}), 2)
                self.assertEqual([entry["issue_id"] for entry in truth], [entry["issue_id"] for entry in changes])
                self.assertEqual([item["type"] for item in findings], [item["type"] for item in truth])
                self.assertEqual(match(findings, truth, catalog).tp, 2)
                for finding, expected in zip(findings, truth, strict=True):
                    accepted = expected["acceptable_evidence_sets"]
                    self.assertIn(set(finding["evidence_ids"]), [set(evidence) for evidence in accepted])
                    minimal = {**finding, "evidence_ids": min(accepted, key=len)}
                    self.assertEqual(match([minimal], [expected], catalog).tp, 1)
                    for removed in minimal["evidence_ids"]:
                        broken = {**minimal, "evidence_ids": [item for item in minimal["evidence_ids"] if item != removed]}
                        result = match([broken], [expected], catalog)
                        self.assertEqual((result.tp, result.fp, result.fn), (0, 1, 1))
                first, second = changes
                if scenario == PARTIAL_FX_AT_CUTOFF:
                    suspense = next(row for row in first["after"]["rows"] if row["account"] == "suspense")
                    self.assertIn(suspense, second["before"]["rows"])
                    self.assertNotIn(suspense, second["after"]["rows"])
                else:
                    self.assertEqual(len(first["before"]["rows"]), 2)
                    self.assertEqual(len(first["after"]["rows"]), 4)
                    self.assertTrue(all(row in episode.public.ledger for row in first["after"]["rows"]))

    def test_scenarios_are_additive_reproducible_and_keep_clean_control(self):
        scenarios = [PARTIAL_FX_AT_CUTOFF, DUPLICATE_WITH_FX]
        ordinary = {"duplicate_invoice": 1, "payment_allocation": 1, "fx_conversion": 1, "period_cutoff": 1}
        config = {**_config(scenarios, ordinary), "include_distractors": True}
        mixed = generate_episode(19, config)
        self.assertEqual(mixed, generate_episode(19, {**config, "interaction_scenarios": scenarios[::-1]}))
        self.assertEqual(Counter(entry["type"] for entry in mixed.evaluator.truth["issues"]), {
            "duplicate_invoice": 2, "payment_allocation": 2, "fx_conversion": 3, "period_cutoff": 1,
        })
        self.assertEqual(mixed.public.manifest["record_counts"]["ledger"], len(mixed.public.ledger))
        self.assertEqual(len({row["row_id"] for row in mixed.public.ledger}), len(mixed.public.ledger))
        self.assertTrue(any(doc.get("payment_kind") == "refund" for doc in mixed.public.payments))
        self.assertIn("distractor_operations", mixed.evaluator.provenance["initial_history"])
        clean = generate_episode(19, _config([]))
        self.assertEqual(clean.evaluator.truth["issues"], [])
        self.assertFalse(any(row["account"] == "suspense" for row in clean.public.ledger))
        self.assertEqual(len(generate_episode(19, _config([], ordinary)).evaluator.truth["issues"]), 4)
        for scenarios, error in (
            (["unknown"], "unknown interaction"), ([PARTIAL_FX_AT_CUTOFF] * 2, "duplicate interaction"),
        ):
            with self.assertRaisesRegex(ValueError, error):
                generate_episode(19, _config(scenarios))
        with self.assertRaisesRegex(ValueError, "fx_conversion candidates"):
            generate_episode(19, _config([PARTIAL_FX_AT_CUTOFF], {"fx_conversion": 2}))
