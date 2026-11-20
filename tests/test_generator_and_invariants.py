import json
import tempfile
import unittest
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal, localcontext
from pathlib import Path
from zoneinfo import ZoneInfo

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

import test_env_contract_and_isolation as contract
from genesis.generation import export_episode, generate_episode


CLEAN = {"schema_version": "1.0.0", "error_counts": {}}


class CleanGeneratorTests(unittest.TestCase):
    def test_export_is_reproducible_and_seed_changes_business_data(self):
        def files(directory):
            return {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*")
                    if path.is_file()}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, seed in (("first", 413), ("other", 29), ("replay", 413)):
                export_episode(generate_episode(seed, CLEAN), root / name)
            self.assertEqual(files(root / "first"), files(root / "replay"))
            self.assertNotEqual(
                sorted(row["total_minor"] for row in contract.load_lines(root / "first" / "public" / "invoices.jsonl")),
                sorted(row["total_minor"] for row in contract.load_lines(root / "other" / "public" / "invoices.jsonl")),
            )
            with self.assertRaises(FileExistsError):
                export_episode(generate_episode(0, CLEAN), root / "first")

    def test_exported_records_match_v1_schemas(self):
        schemas = {
            path.name.removesuffix(".schema.json"): contract.load(path)
            for path in (contract.ROOT / "schemas").glob("*.schema.json")
        }
        registry = Registry().with_resources(
            (schema["$id"], Resource.from_contents(schema)) for schema in schemas.values()
        )
        for include_distractors in (False, True):
            with self.subTest(include_distractors=include_distractors):
                self.assert_export_matches_schemas(schemas, registry, include_distractors)

    def assert_export_matches_schemas(self, schemas, registry, include_distractors):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "episode"
            export_episode(generate_episode(413, {**CLEAN, "include_distractors": include_distractors}), target)
            public, evaluator = target / "public", target / "evaluator-only"
            self.assertEqual({path.name for path in public.iterdir()}, {
                "manifest.json", "policy.json", "ledger.csv", "fx.csv", "invoices.jsonl", "payments.jsonl",
            })
            self.assertEqual({path.name for path in evaluator.iterdir()}, {"truth.json", "provenance.json"})
            artifacts = {
                "manifest": [contract.load(public / "manifest.json")],
                "policy": [contract.load(public / "policy.json")],
                "invoice": contract.load_lines(public / "invoices.jsonl"),
                "payment": contract.load_lines(public / "payments.jsonl"),
                "ledger-row": list(contract.load_csv(public / "ledger.csv", contract.LEDGER_COLUMNS)),
                "fx-row": list(contract.load_csv(public / "fx.csv", contract.FX_COLUMNS)),
                "truth": [contract.load(evaluator / "truth.json")],
                "provenance": [contract.load(evaluator / "provenance.json")],
            }
            for name, entries in artifacts.items():
                validator = Draft202012Validator(
                    schemas[name], registry=registry, format_checker=Draft202012Validator.FORMAT_CHECKER,
                )
                for entry in entries:
                    validator.validate(entry)
                    if name not in {"truth", "provenance"}:
                        self.assertTrue(set(contract.field_names(entry)).isdisjoint(contract.PRIVATE_KEYS))
            self.assertIn(",null,null", (public / "ledger.csv").read_text(encoding="utf-8"))
            self.assertEqual(artifacts["truth"][0]["issues"], [])
            self.assertEqual(artifacts["provenance"][0]["changes"], [])
            if include_distractors:
                refund, = [payment for payment in artifacts["payment"] if "payment_kind" in payment]
                payment_schema = Draft202012Validator(schemas["payment"], registry=registry)
                for invalid in (
                    {key: value for key, value in refund.items() if key != "refund_of_payment_id"},
                    {key: value for key, value in refund.items() if key != "payment_kind"},
                    {**refund, "amount_minor": -refund["amount_minor"]},
                    {**refund, "invoice_refs": None},
                ):
                    self.assertFalse(payment_schema.is_valid(invalid))

    def test_public_documents_reconcile_with_balanced_postings(self):
        public = generate_episode(413, CLEAN).public
        manifest = public.manifest
        invoices = {doc["invoice_id"]: doc for doc in public.invoices}
        payments = {doc["payment_id"]: doc for doc in public.payments}
        quotes = {(q["date"], q["source_currency"], q["target_currency"]): q["rate"] for q in public.fx}
        base = manifest["base_currency"]
        zone = ZoneInfo(manifest["timezone"])
        start = datetime.fromisoformat(manifest["period_start"])
        end = datetime.fromisoformat(manifest["period_end"])
        movements = defaultdict(int)
        self.assertEqual(manifest["record_counts"], {
            table: len(getattr(public, table)) for table in ("ledger", "invoices", "payments", "fx")
        })
        self.assertEqual(manifest["document_count"], len(invoices) + len(payments))
        self.assertEqual(len({row["row_id"] for row in public.ledger}), len(public.ledger))
        for invoice in invoices.values():
            self.assertEqual(invoice["total_minor"], sum(
                line["quantity"] * line["unit_price_minor"] for line in invoice["line_items"]
            ))
        for row in public.ledger:
            instant = datetime.fromisoformat(row["timestamp"])
            self.assertLessEqual(start, instant)
            self.assertLess(instant, end)
            self.assertEqual(instant.utcoffset(), instant.astimezone(zone).utcoffset())
            currency, invoice, payment = row["currency"], row["invoice_id"], row["payment_id"]
            self.assertEqual(currency, base)
            self.assertTrue(invoice is None or invoice in invoices)
            self.assertTrue(payment is None or payment in payments)
            movements[currency, invoice, payment] += row["debit_minor"] - row["credit_minor"]
        self.assertTrue(movements)
        self.assertEqual(set(movements.values()), {0})

        unclosed = set(invoices)
        foreign_count = 0
        for payment in payments.values():
            invoice_id, = payment["invoice_refs"]
            unclosed.remove(invoice_id)
            self.assertEqual(invoices[invoice_id]["total_minor"], payment["amount_minor"])
            source = payment["currency"]
            foreign_count += source != base
            rate = "1" if source == base else quotes[(payment["timestamp"][:10], source, base)]
            exponents = manifest["currency_exponents"]
            amount = Decimal(payment["amount_minor"]) / 10 ** exponents[source]
            expected = (amount * Decimal(rate) * 10 ** exponents[base]).quantize(
                Decimal(1), rounding=public.policy["rounding_mode"],
            )
            paid = [row for row in public.ledger if row["payment_id"] == payment["payment_id"]]
            self.assertEqual({row["invoice_id"] for row in paid}, {invoice_id})
            self.assertEqual(sum(row["debit_minor"] for row in paid if row["account"] == "accounts_payable"), expected)
            self.assertEqual(sum(row["credit_minor"] for row in paid if row["account"] == "bank"), expected)
        self.assertEqual(len(unclosed), 1)
        self.assertEqual(foreign_count, 1)

    def test_validation_and_decimal_context_isolation(self):
        for config in (
            {}, {"schema_version": "0.0.0", "error_counts": {}},
            {**CLEAN, "error_counts": {"made_up": 0}},
            {**CLEAN, "error_counts": {"fx_conversion": False}},
            {**CLEAN, "error_counts": {"fx_conversion": -1}},
            {**CLEAN, "error_counts": {"fx_conversion": 2}},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                generate_episode(0, config)
        with self.assertRaises(ValueError):
            generate_episode(True, CLEAN)
        expected = generate_episode(413, CLEAN)
        with localcontext() as context:
            context.prec = 3
            self.assertEqual(generate_episode(413, CLEAN), expected)
        self.assertEqual(generate_episode(413, json.loads(json.dumps(CLEAN))), expected)


class DistractorTests(unittest.TestCase):
    def setUp(self):
        self.config = {**CLEAN, "include_distractors": True}
        self.episode = generate_episode(413, self.config)
        self.public = self.episode.public

    def test_consecutive_installments_are_each_fully_allocated(self):
        grouped = defaultdict(list)
        for payment in self.public.payments:
            grouped[payment["invoice_refs"][0]].append(payment)
        # Identify the split only from public documents and ordinary payment roles.
        invoice_id, = [key for key, payments in grouped.items()
                       if len(payments) == 2 and all("payment_kind" not in p for p in payments)]
        invoice = next(invoice for invoice in self.public.invoices if invoice["invoice_id"] == invoice_id)
        payments = sorted(grouped[invoice_id], key=lambda p: p["timestamp"])
        balances = self.assert_installment_history(self.public, invoice, payments)
        self.assertEqual(balances, [invoice["total_minor"] - payments[0]["amount_minor"], 0])
        self.assertGreater(balances[0], 0)
        self.assertEqual(len({p["doc_id"] for p in payments}), 2)
        self.assertEqual(len({p["payment_id"] for p in payments}), 2)

        # An actually underallocated payment has the same public roles only
        # for part of its amount, so the independent reconciliation rejects it.
        anomalous = generate_episode(413, {**self.config, "error_counts": {"payment_allocation": 1}})
        issue, = anomalous.evaluator.truth["issues"]
        payment, = [p for p in anomalous.public.payments if p["payment_id"] in issue["affected_ids"]]
        invoice, = [i for i in anomalous.public.invoices if i["invoice_id"] in payment["invoice_refs"]]
        with self.assertRaisesRegex(AssertionError, "each payment"):
            self.assert_installment_history(anomalous.public, invoice, [payment])

    def assert_installment_history(self, public, invoice, payments):
        outstanding = invoice["total_minor"]
        balances = []
        for payment in payments:
            payment_id = payment["payment_id"]
            rows = [row for row in public.ledger if row["payment_id"] == payment_id]
            self.assertTrue(invoice["issued_at"] < payment["timestamp"] < invoice["due_at"])
            self.assertEqual(payment["currency"], invoice["currency"])
            allocated = sum(row["debit_minor"] for row in rows
                            if row["account"] == "accounts_payable"
                            and row["invoice_id"] == invoice["invoice_id"])
            self.assertEqual(allocated, payment["amount_minor"], "each payment must reconcile")
            self.assertEqual(sum(row["credit_minor"] for row in rows if row["account"] == "bank"),
                             payment["amount_minor"])
            self.assertEqual({row["invoice_id"] for row in rows}, {invoice["invoice_id"]})
            self.assertEqual({row["currency"] for row in rows}, {payment["currency"]})
            self.assertEqual({row["timestamp"] for row in rows}, {payment["timestamp"]})
            outstanding -= allocated
            self.assertGreaterEqual(outstanding, 0)
            balances.append(outstanding)
        return balances

    def assert_refund_reconciles(self, public, refund):
        original = next(
            (p for p in public.payments if p["payment_id"] == refund["refund_of_payment_id"]), None,
        )
        self.assertIsNotNone(original, "refund source must resolve")
        self.assertNotIn("payment_kind", original)
        self.assertEqual(original["currency"], refund["currency"])
        self.assertEqual(original["invoice_refs"], refund["invoice_refs"])
        invoice_id, = refund["invoice_refs"]
        invoice = next(invoice for invoice in public.invoices if invoice["invoice_id"] == invoice_id)
        self.assertEqual(invoice["currency"], refund["currency"])
        self.assertLess(invoice["issued_at"], original["timestamp"])
        self.assertLess(original["timestamp"], refund["timestamp"])
        net = original["amount_minor"] - refund["amount_minor"]
        self.assertEqual(net, invoice["total_minor"], "refund amount must reconcile")
        self.assertGreater(refund["amount_minor"], 0)

        def postings(payment):
            rows = [row for row in public.ledger if row["payment_id"] == payment["payment_id"]]
            self.assertEqual({row["currency"] for row in rows}, {payment["currency"]})
            self.assertEqual({row["invoice_id"] for row in rows}, {invoice_id})
            self.assertEqual({row["timestamp"] for row in rows}, {payment["timestamp"]})
            return Counter({account: sum(
                row["debit_minor"] - row["credit_minor"] for row in rows if row["account"] == account
            ) for account in {row["account"] for row in rows}})

        self.assertEqual(postings(original), {
            "accounts_payable": net, "vendor_receivable": refund["amount_minor"],
            "bank": -original["amount_minor"],
        })
        self.assertEqual(postings(refund), {
            "bank": refund["amount_minor"], "vendor_receivable": -refund["amount_minor"],
        })
        return invoice

    def test_refund_reverses_the_publicly_identifiable_overpayment(self):
        refund, = [p for p in self.public.payments if p.get("payment_kind") == "refund"]
        self.assert_refund_reconciles(self.public, refund)
        with self.assertRaisesRegex(AssertionError, "refund amount"):
            self.assert_refund_reconciles(self.public, {**refund, "amount_minor": refund["amount_minor"] + 1})
        with self.assertRaisesRegex(AssertionError, "refund source"):
            self.assert_refund_reconciles(self.public, {**refund, "refund_of_payment_id": "unknown"})

    def test_equal_amounts_are_independent_and_each_accrues_once(self):
        obligations = defaultdict(list)
        for invoice in self.public.invoices:
            signature = (invoice["entity_id"], invoice["currency"], invoice["total_minor"])
            obligations[signature].append(invoice)
        pair, = [invoices for invoices in obligations.values() if len(invoices) == 2]
        self.assertEqual(len({invoice["invoice_id"] for invoice in pair}), 2)
        self.assertEqual(len({invoice["doc_id"] for invoice in pair}), 2)
        self.assertEqual(len({invoice["issued_at"] for invoice in pair}), 2)
        self.assertEqual(len({invoice["line_items"][0]["description"] for invoice in pair}), 2)
        for invoice in pair:
            accrual = [r for r in self.public.ledger
                       if r["invoice_id"] == invoice["invoice_id"] and r["payment_id"] is None]
            amount = invoice["total_minor"]
            self.assertCountEqual(
                [(row["account"], row["debit_minor"], row["credit_minor"]) for row in accrual],
                [("expense", amount, 0), ("accounts_payable", 0, amount)],
            )
        self.assertEqual(self.episode.evaluator.truth["issues"], [])
        self.assertEqual(self.episode.evaluator.provenance["changes"], [])

    def test_seed_config_normalization_opening_funds_and_error_compatibility(self):
        all_counts = dict.fromkeys(("duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff"), 1)
        zero_counts = dict.fromkeys(all_counts, 0)
        for seed in (0, 413):
            for enabled in (False, True):
                with self.subTest(seed=seed, include_distractors=enabled):
                    current = generate_episode(seed, {**CLEAN, "include_distractors": enabled})
                    self.assertEqual(current, generate_episode(seed, {
                        **CLEAN, "include_distractors": enabled, "error_counts": zero_counts,
                    }))
                    if not enabled:
                        self.assertEqual(current, generate_episode(seed, CLEAN))
                    else:
                        self.assertIn("distractor_operations", current.evaluator.provenance["initial_history"])
                    public = current.public
                    self.assertEqual(public.manifest["document_count"], len(public.invoices) + len(public.payments))
                    self.assertEqual(len({row["row_id"] for row in public.ledger}), len(public.ledger))
                    self.assertEqual(sum(row["debit_minor"] - row["credit_minor"] for row in public.ledger), 0)
                    cash = 0
                    per_timestamp = defaultdict(int)
                    for row in public.ledger:
                        if row["account"] == "bank":
                            per_timestamp[row["timestamp"]] += row["debit_minor"] - row["credit_minor"]
                    for timestamp in sorted(per_timestamp):
                        cash += per_timestamp[timestamp]
                        self.assertGreaterEqual(cash, 0)
            combined = generate_episode(seed, {**self.config, "error_counts": all_counts})
            self.assertEqual(Counter(issue["type"] for issue in combined.evaluator.truth["issues"]), all_counts)
            refund, = [p for p in combined.public.payments if p.get("payment_kind") == "refund"]
            self.assert_refund_reconciles(combined.public, refund)


class ConfiguredGeneratorTests(unittest.TestCase):
    def test_all_four_counts_are_independent_and_reproducible(self):
        mixed = {
            "duplicate_invoice": 1, "payment_allocation": 1,
            "fx_conversion": 1, "period_cutoff": 1,
        }
        for seed in (0, 413):
            for counts in ({}, *({name: 1} for name in mixed), mixed):
                with self.subTest(seed=seed, counts=counts):
                    config = {**CLEAN, "error_counts": counts}
                    episode = generate_episode(seed, config)
                    with localcontext() as context:
                        context.prec = 3
                        self.assertEqual(episode, generate_episode(seed, config))
                    issues = episode.evaluator.truth["issues"]
                    changes = episode.evaluator.provenance["changes"]
                    self.assertEqual(Counter(issue["type"] for issue in issues), counts)
                    self.assertEqual({issue["issue_id"]: issue["affected_ids"] for issue in issues}, {
                        change["issue_id"]: change["affected_ids"] for change in changes
                    })
                    self.assertEqual(len(changes), len(issues))
                    if issues:
                        self.assertEqual(
                            len(set().union(*(set(issue["affected_ids"]) for issue in issues))),
                            sum(len(issue["affected_ids"]) for issue in issues),
                        )
