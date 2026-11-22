from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from decimal import Decimal
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

from genesis.domain.enums import FindingType
from genesis.evaluation.matcher import match
from genesis.evaluation.permutation import validate_links
from genesis.evaluation.public_proofs import verify_public_proofs
from genesis.generation.company import PublicEpisode, generate_episode
from genesis.generation.datasets import (
    _book_hash, check_public_dataset, export_public_dataset, load_dataset_entries,
)
from genesis.generation.interactions import DUPLICATE_WITH_FX, PARTIAL_FX_AT_CUTOFF

import test_env_contract_and_isolation as contract
import test_interaction_scenarios as interactions


DATA = Path(__file__).resolve().parents[1] / "data"
PUBLIC_NAMES = {"manifest.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv", "policy.json"}


def load_public(path: Path) -> PublicEpisode:
    return PublicEpisode(
        contract.load(path / "manifest.json"),
        contract.load(path / "policy.json"),
        list(contract.load_csv(path / "ledger.csv", contract.LEDGER_COLUMNS)),
        contract.load_lines(path / "invoices.jsonl"),
        contract.load_lines(path / "payments.jsonl"),
        list(contract.load_csv(path / "fx.csv", contract.FX_COLUMNS)),
    )


class PublicDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        contract.DataContractTests.setUpClass()
        cls.validators = contract.DataContractTests.validators
        cls.entries = load_dataset_entries(DATA / "configs")

    def test_twenty_public_books_follow_schemas_and_reconcile_references(self):
        self.assertEqual(check_public_dataset(DATA), 20)
        self.assertEqual(Counter(entry.split for entry in self.entries), {"train": 10, "dev": 10})
        self.assertEqual(len({entry.seed for entry in self.entries}), 20)
        observed_types = set()
        observed_scenarios = set()
        observed_subtypes = set()
        observed_cutoffs = set()
        observed_currencies = set()
        observed_timezones = set()
        scaled_episodes = 0
        non_exact_fx = 0
        for entry in self.entries:
            regenerated = generate_episode(entry.seed, entry.config)
            validate_links(regenerated)
            verify_public_proofs(regenerated)
            truth = regenerated.evaluator.truth
            episode_id = truth["episode_id"]
            with self.subTest(episode=episode_id):
                folder = DATA / entry.split / episode_id
                self.assertEqual({file.name for file in folder.iterdir()}, PUBLIC_NAMES)
                public = load_public(folder)
                self.assert_public_contract(public)
                self.validators["config"].validate(entry.config)
                observed_scenarios.update(entry.config.get("interaction_scenarios", []))
                observed_currencies.update(public.manifest["currency_exponents"].keys())
                observed_timezones.add(public.policy["timezone"])
                if len(public.ledger) >= 200 and public.manifest["document_count"] >= 20:
                    scaled_episodes += 1
                cash = 0
                per_ts = defaultdict(int)
                for row in public.ledger:
                    if row["account"] == "bank":
                        per_ts[row["timestamp"]] += row["debit_minor"] - row["credit_minor"]
                for ts in sorted(per_ts):
                    cash += per_ts[ts]
                    self.assertGreaterEqual(cash, 0)
                for change in regenerated.evaluator.provenance["changes"]:
                    after = change["after"] or {}
                    if "subtype" in after:
                        observed_subtypes.add(after["subtype"])
                    if "inside_period" in after:
                        observed_cutoffs.add(
                            "before_start"
                            if datetime.fromisoformat(after["timestamp"]) < datetime.fromisoformat(public.policy["period_start"])
                            else "after_end"
                        )
                quotes = {(q["date"], q["source_currency"], q["target_currency"]): q["rate"] for q in public.fx}
                zone = ZoneInfo(public.policy["timezone"])
                base = public.manifest["base_currency"]
                exps = public.manifest["currency_exponents"]
                for pay in public.payments:
                    if pay["currency"] != base:
                        day = datetime.fromisoformat(pay["timestamp"]).astimezone(zone).date().isoformat()
                        rate = quotes.get((day, pay["currency"], base))
                        if rate is not None:
                            raw = (Decimal(pay["amount_minor"]).scaleb(-exps[pay["currency"]]) * Decimal(rate)).scaleb(exps[base])
                            if raw != int(raw):
                                non_exact_fx += 1
                evidence = {
                    "policy", *(row["row_id"] for row in public.ledger),
                    *(row["doc_id"] for row in public.invoices + public.payments),
                    *(f"fx:{row['date']}:{row['source_currency']}:{row['target_currency']}" for row in public.fx),
                }
                for issue in truth["issues"]:
                    observed_types.add(issue["type"])
                    for accepted in issue["acceptable_evidence_sets"]:
                        self.assertLessEqual(set(accepted), evidence)
        self.assertEqual(observed_types, {kind.value for kind in FindingType})
        self.assertEqual(observed_scenarios, {PARTIAL_FX_AT_CUTOFF, DUPLICATE_WITH_FX})
        self.assertEqual(observed_subtypes, {"missing", "partial"})
        self.assertEqual(observed_cutoffs, {"before_start", "after_end"})
        self.assertGreaterEqual(len(observed_currencies), 4)
        self.assertGreaterEqual(len(observed_timezones), 3)
        self.assertGreaterEqual(scaled_episodes, 2)
        self.assertGreater(non_exact_fx, 0)

    def assert_public_contract(self, public):
        groups = {
            "ledger": ("ledger-row", public.ledger), "fx": ("fx-row", public.fx),
            "invoices": ("invoice", public.invoices), "payments": ("payment", public.payments),
        }
        declared = public.manifest["currency_exponents"]
        self.validators["manifest"].validate(public.manifest)
        self.validators["policy"].validate(public.policy)
        for key in ("period_start", "period_end", "timezone", "rounding_mode"):
            self.assertEqual(public.manifest[key], public.policy[key])
        self.assertIn(public.manifest["base_currency"], declared)
        self.assertEqual(public.manifest["record_counts"], {name: len(group[1]) for name, group in groups.items()})
        self.assertEqual(public.manifest["document_count"], len(public.invoices) + len(public.payments))
        for schema, rows in groups.values():
            for row in rows:
                self.validators[schema].validate(row)
                self.assertLessEqual(
                    {value for key, value in row.items() if key.endswith("currency")}, set(declared),
                )
        public_objects = [public.manifest, public.policy, *(row for _, rows in groups.values() for row in rows)]
        self.assertTrue(contract.PRIVATE_KEYS.isdisjoint(contract.field_names(public_objects)))
        invoices = {row["invoice_id"]: row for row in public.invoices}
        payments = {row["payment_id"]: row for row in public.payments}
        docs = {row["doc_id"] for row in public.invoices + public.payments}
        self.assertEqual(len(docs), public.manifest["document_count"])
        self.assertEqual((len(invoices), len(payments)), (len(public.invoices), len(public.payments)))
        self.assertEqual(len({row["row_id"] for row in public.ledger}), len(public.ledger))
        for row in public.payments:
            self.assertLessEqual(set(row["invoice_refs"] or []), invoices.keys())
            if "refund_of_payment_id" in row:
                self.assertIn(row["refund_of_payment_id"], payments)
        movements = defaultdict(int)
        cash_movements = defaultdict(int)
        for row in public.ledger:
            if row["invoice_id"] is not None:
                self.assertIn(row["invoice_id"], invoices)
                self.assertEqual(row["entity_id"], invoices[row["invoice_id"]]["entity_id"])
            if row["payment_id"] is not None:
                self.assertIn(row["payment_id"], payments)
            movements[row["currency"], row["timestamp"]] += row["debit_minor"] - row["credit_minor"]
            if row["account"] == "bank":
                cash_movements[row["timestamp"]] += row["debit_minor"] - row["credit_minor"]
        self.assertEqual(set(movements.values()), {0})
        balance = 0
        for timestamp in sorted(cash_movements):
            balance += cash_movements[timestamp]
            self.assertGreaterEqual(balance, 0)

    def test_clean_and_single_error_books_have_publicly_verifiable_distinctions(self):
        clean, duplicate = self.entries[:2]
        for entry in (clean, duplicate):
            regenerated = generate_episode(entry.seed, entry.config)
            public = load_public(DATA / entry.split / regenerated.public.manifest["episode_id"])
            postings = Counter(
                row["invoice_id"] for row in public.ledger if row["account"] == "expense"
            )
            self.assertEqual(set(postings), {row["invoice_id"] for row in public.invoices})
            if entry == clean:
                self.assertEqual(set(postings.values()), {1})
                self.assertTrue(any(row.get("payment_kind") == "refund" for row in public.payments))
                ordinary = Counter(
                    row["invoice_refs"][0] for row in public.payments
                    if "payment_kind" not in row and row["invoice_refs"]
                )
                self.assertTrue(any(count > 1 for count in ordinary.values()))
                self.assertEqual(regenerated.evaluator.truth["issues"], [])
                continue
            invoice_id, = [key for key, count in postings.items() if count == 2]
            duplicated_rows = [
                row for row in public.ledger
                if row["invoice_id"] == invoice_id and row["payment_id"] is None
            ]
            expense = [row["debit_minor"] for row in duplicated_rows if row["account"] == "expense"]
            payable = [row["credit_minor"] for row in duplicated_rows if row["account"] == "accounts_payable"]
            self.assertCountEqual(expense, payable)
            invoice = next(row for row in public.invoices if row["invoice_id"] == invoice_id)
            self.assertEqual(invoice["currency"], public.manifest["base_currency"])
            finding = interactions._finding(
                "duplicate_invoice", sum(expense) - invoice["total_minor"], invoice["currency"],
                [invoice_id, *(row["row_id"] for row in duplicated_rows)],
                [invoice["doc_id"], *(row["row_id"] for row in duplicated_rows)],
            )
            truth = regenerated.evaluator.truth["issues"]
            self.assertEqual(match([finding], truth, finding["evidence_ids"]).tp, 1)

    def test_allocation_matches_an_answer_derived_from_public_postings(self):
        entries = [entry for entry in self.entries if entry.config["error_counts"] == {"payment_allocation": 1}]
        self.assertTrue(entries)
        for entry in entries:
            episode = generate_episode(entry.seed, entry.config)
            public = load_public(DATA / entry.split / episode.public.manifest["episode_id"])
            suspense, = (row for row in public.ledger if row["account"] == "suspense")
            payment, = (row for row in public.payments if row["payment_id"] == suspense["payment_id"])
            invoice, = (row for row in public.invoices if row["invoice_id"] in payment["invoice_refs"])
            settlement = [row for row in public.ledger if row["payment_id"] == payment["payment_id"]]
            affected = [row["row_id"] for row in settlement if row["account"] in ("accounts_payable", "suspense")]
            evidence = [invoice["doc_id"], payment["doc_id"], *(row["row_id"] for row in settlement)]
            finding = interactions._finding(
                "payment_allocation", suspense["debit_minor"], payment["currency"],
                [invoice["invoice_id"], payment["payment_id"], *affected], evidence,
            )
            scores = match([finding], episode.evaluator.truth["issues"], evidence)
            self.assertEqual((scores.tp, scores.fp, scores.fn), (1, 0, 0))

    def test_two_committed_interactions_reconcile_using_public_files(self):
        for scenario in (PARTIAL_FX_AT_CUTOFF, DUPLICATE_WITH_FX):
            entry = next(entry for entry in self.entries if entry.config.get("interaction_scenarios") == [scenario])
            regenerated = generate_episode(entry.seed, entry.config)
            public = load_public(DATA / entry.split / regenerated.public.manifest["episode_id"])
            findings, evidence = interactions.InteractionTests()._reconcile_public(public, scenario)
            truth = regenerated.evaluator.truth["issues"]
            self.assertEqual(match(findings, truth, evidence).tp, 2)

    def test_independent_exports_match_and_no_existing_output_is_rewritten(self):
        with TemporaryDirectory() as temporary:
            first, second = (Path(temporary) / name for name in ("first", "second"))
            self.assertEqual(export_public_dataset(DATA / "configs", first), 20)
            self.assertEqual(export_public_dataset(DATA / "configs", second), 20)
            self.assertEqual(check_public_dataset(first), 20)
            self.assertEqual(check_public_dataset(second), 20)
            def payload(root):
                return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
            self.assertEqual(payload(first), payload(second))
            with self.assertRaises(FileExistsError):
                export_public_dataset(DATA / "configs", first)
            self.assertEqual(payload(first), payload(second))
            manifest = next(first.glob("train/*/manifest.json"))
            original = manifest.read_bytes()
            manifest.write_bytes(original.replace(b'"episode_id"', b'"seed":0,"episode_id"'))
            with self.assertRaisesRegex(ValueError, "content mismatch"):
                check_public_dataset(first)
            manifest.write_bytes(original)
            extra = manifest.with_name("truth.json")
            extra.write_bytes(b"{}\n")
            with self.assertRaisesRegex(ValueError, "unexpected dataset file"):
                check_public_dataset(first)
            extra.unlink()
            manifest.unlink()
            with self.assertRaisesRegex(ValueError, "missing dataset file"):
                check_public_dataset(first)

    def test_rejects_malformed_index_duplicate_key_and_duplicate_seed(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            train = base / "train_configs.json"
            dev = base / "dev_configs.json"
            train.write_bytes((DATA / "configs" / train.name).read_bytes())
            valid = contract.load(DATA / "configs" / dev.name)
            original = contract.load(train)["entries"][0]["seed"]
            for content, message in (
                ("not JSON", "dev_configs.json"),
                ('{"split":"train","split":"dev"}', "duplicate index field"),
                (json.dumps({**valid, "entries": []}), "expected 10 entries"),
                (json.dumps({**valid, "entries": [{**valid["entries"][0], "seed": original}, *valid["entries"][1:]]}),
                 "duplicate seed"),
                (json.dumps({**valid, "entries": [
                    {**valid["entries"][0], "config": {**valid["entries"][0]["config"], "interaction_scenarios": []}},
                    *valid["entries"][1:],
                ]}), "normalized configuration"),
            ):
                with self.subTest(message=message):
                    dev.write_text(content, encoding="utf-8")
                    target = base / "export"
                    with self.assertRaisesRegex(ValueError, message):
                        export_public_dataset(base, target)
                    self.assertFalse(target.exists())

    def test_duplicate_book_ignores_episode_id_row_order_and_opaque_ids(self):
        first = self.entries[0]
        episode = generate_episode(first.seed, first.config)
        equivalent = deepcopy(episode)
        equivalent.public.manifest["episode_id"] = "another-id"
        equivalent.public.ledger.reverse()
        for row in equivalent.public.ledger:
            row["row_id"] = f"changed-{row['row_id']}"
        self.assertEqual(_book_hash(equivalent.public), _book_hash(episode.public))
        with TemporaryDirectory() as temporary:
            target = Path(temporary) / "export"
            with patch("genesis.generation.datasets.generate_episode", side_effect=[episode, equivalent]):
                with self.assertRaisesRegex(ValueError, "duplicate episode or book"):
                    export_public_dataset(DATA / "configs", target)
            self.assertFalse(target.exists())
