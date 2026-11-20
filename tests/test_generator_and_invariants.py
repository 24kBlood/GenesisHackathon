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
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "episode"
            export_episode(generate_episode(413, CLEAN), target)
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
