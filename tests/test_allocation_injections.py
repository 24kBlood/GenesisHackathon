"""Integration tests for duplicate accrual and payment allocation errors."""

import tempfile
import unittest
from pathlib import Path
from random import Random

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

import test_env_contract_and_isolation as contract
from genesis.evaluation.matcher import match
from genesis.generation import export_episode, generate_episode
from genesis.generation.allocation import apply_allocation_errors


def config(**counts):
    return {"schema_version": "1.0.0", "error_counts": counts}


def evidence_catalog(public):
    return (
        {row["row_id"] for row in public.ledger}
        | {doc["doc_id"] for doc in public.invoices + public.payments}
        | {f"fx:{row['date']}:{row['source_currency']}:{row['target_currency']}" for row in public.fx}
        | {"policy"}
    )


def finding(issue, evidence):
    return {
        "type": issue["type"], "affected_ids": issue["affected_ids"],
        "claimed_amount_minor": issue["expected_amount_minor"], "currency": issue["currency"],
        "evidence_ids": evidence, "explanation": "Сверка публичных документов и проводок.",
    }


class AllocationTests(unittest.TestCase):
    def assert_verifiable(self, episode, issue):
        allowed, = issue["acceptable_evidence_sets"]
        catalog = evidence_catalog(episode.public)
        self.assertLessEqual(set(allowed), catalog)
        self.assertEqual(len(issue["affected_ids"]), len(set(issue["affected_ids"])))
        self.assertEqual(len(allowed), len(set(allowed)))
        result = match([finding(issue, allowed)], [issue], catalog)
        self.assertEqual((result.tp, result.fp, result.fn), (1, 0, 0))
        for evidence_id in allowed:
            with self.subTest(missing_evidence=evidence_id):
                incomplete = [ref for ref in allowed if ref != evidence_id]
                result = match([finding(issue, incomplete)], [issue], catalog)
                self.assertEqual((result.tp, result.fp, result.fn), (0, 1, 1))

    def test_duplicate_posts_two_full_accrual_pairs_for_one_document(self):
        clean = generate_episode(19, config())
        mutated = generate_episode(19, config(duplicate_invoice=1))
        issue, = mutated.evaluator.truth["issues"]
        change, = mutated.evaluator.provenance["changes"]
        invoice, = [doc for doc in mutated.public.invoices if doc["invoice_id"] in issue["affected_ids"]]

        def accruals(public):
            return [row for row in public.ledger
                    if row["payment_id"] is None and row["invoice_id"] == invoice["invoice_id"]]

        amount = invoice["total_minor"]
        self.assertEqual(issue["type"], "duplicate_invoice")
        self.assertEqual(issue["currency"], invoice["currency"])
        self.assertEqual(issue["expected_amount_minor"], amount)
        self.assertEqual(len(accruals(clean.public)), 2)
        self.assertCountEqual(
            [(row["account"], row["debit_minor"], row["credit_minor"])
             for row in accruals(mutated.public)],
            [("expense", amount, 0), ("accounts_payable", 0, amount)] * 2,
        )
        self.assertEqual(len(mutated.public.ledger), len(clean.public.ledger) + 2)
        self.assertEqual(change["issue_id"], issue["issue_id"])
        self.assertEqual(change["before"]["posted_minor"], amount)
        self.assertEqual(change["after"]["posted_minor"], 2 * amount)
        self.assertCountEqual(mutated.public.invoices, clean.public.invoices)
        self.assertCountEqual(mutated.public.payments, clean.public.payments)
        self.assert_verifiable(mutated, issue)

    def test_missing_and_partial_are_independently_provable_and_balanced(self):
        subtypes = set()
        for seed in range(12):
            with self.subTest(seed=seed):
                episode = generate_episode(seed, config(payment_allocation=1))
                issue, = episode.evaluator.truth["issues"]
                change, = episode.evaluator.provenance["changes"]
                payment, = [p for p in episode.public.payments if p["payment_id"] in issue["affected_ids"]]
                rows = [row for row in episode.public.ledger if row["payment_id"] == payment["payment_id"]]
                expected = payment["amount_minor"]
                posted = sum(row["debit_minor"] for row in rows if row["account"] == "accounts_payable")
                banks = [row for row in rows if row["account"] == "bank"]
                suspense, = [row for row in rows if row["account"] == "suspense"]
                subtype = change["after"]["subtype"]
                subtypes.add(subtype)

                self.assertEqual(issue["type"], "payment_allocation")
                self.assertEqual(issue["currency"], payment["currency"])
                self.assertEqual(issue["expected_amount_minor"], expected - posted)
                self.assertEqual(change["before"]["posted_minor"], expected)
                self.assertEqual(change["after"]["posted_minor"], posted)
                self.assertEqual(suspense["debit_minor"], expected - posted)
                self.assertIsNone(suspense["invoice_id"])
                self.assertEqual([row["credit_minor"] for row in banks], [expected])
                self.assertEqual(sum(row["debit_minor"] - row["credit_minor"] for row in rows), 0)
                if subtype == "partial":
                    self.assertGreater(posted, 0)
                else:
                    self.assertEqual(posted, 0)
                self.assert_verifiable(episode, issue)
        self.assertEqual(subtypes, {"missing", "partial"})

    def test_independent_counts_and_determinism(self):
        counts = config(duplicate_invoice=1, payment_allocation=1)
        episode = generate_episode(18, counts)
        self.assertEqual(episode, generate_episode(18, counts))
        self.assertEqual(episode.evaluator.provenance["config"], counts)
        duplicate, allocation = episode.evaluator.truth["issues"]
        self.assertFalse(set(duplicate["affected_ids"]) & set(allocation["affected_ids"]))
        self.assertEqual(len(generate_episode(18, config(duplicate_invoice=2)).evaluator.truth["issues"]), 2)
        self.assertEqual(len(generate_episode(18, config(payment_allocation=2)).evaluator.truth["issues"]), 2)
        self.assertEqual(generate_episode(18, config()).evaluator.truth["issues"], [])
        with self.assertRaisesRegex(ValueError, "disjoint fully paid domestic invoices"):
            generate_episode(18, config(duplicate_invoice=2, payment_allocation=2))

    def test_unbalanced_or_ambiguous_settlement_is_ineligible(self):
        for failure in ("unbalanced", "duplicate_bank", "wrong_amount"):
            with self.subTest(failure=failure):
                clean = generate_episode(4, config()).public
                payment = next(doc for doc in clean.payments if doc["currency"] == "KZT")
                rows = [row for row in clean.ledger if row["payment_id"] == payment["payment_id"]]
                bank, = [row for row in rows if row["account"] == "bank"]
                if failure in {"unbalanced", "wrong_amount"}:
                    bank["credit_minor"] += 1
                if failure == "wrong_amount":
                    allocation, = [row for row in rows if row["account"] == "accounts_payable"]
                    allocation["debit_minor"] += 1
                if failure == "duplicate_bank":
                    clean.ledger.append({**bank, "row_id": "test-extra-bank"})
                original_ids = [row["row_id"] for row in clean.ledger]

                with self.assertRaisesRegex(ValueError, "only 0 disjoint"):
                    apply_allocation_errors(
                        clean.ledger, clean.invoices, [payment],
                        base_currency="KZT", duplicate_count=0, allocation_count=1,
                        rng=Random(1), new_id=lambda prefix: f"test-{prefix}",
                    )
                self.assertEqual([row["row_id"] for row in clean.ledger], original_ids)

    def test_exported_error_artifacts_validate_and_counts_reconcile(self):
        schemas = {
            path.name.removesuffix(".schema.json"): contract.load(path)
            for path in (contract.ROOT / "schemas").glob("*.schema.json")
        }
        registry = Registry().with_resources(
            (schema["$id"], Resource.from_contents(schema)) for schema in schemas.values()
        )
        for seed in (0, 1):
            with self.subTest(seed=seed), tempfile.TemporaryDirectory() as temporary:
                episode = generate_episode(seed, config(duplicate_invoice=1, payment_allocation=1))
                target = Path(temporary) / "episode"
                export_episode(episode, target)
                public, evaluator = target / "public", target / "evaluator-only"
                ledger = list(contract.load_csv(public / "ledger.csv", contract.LEDGER_COLUMNS))
                manifest = contract.load(public / "manifest.json")
                self.assertEqual(manifest["record_counts"]["ledger"], len(ledger))
                self.assertEqual(sum(row["debit_minor"] - row["credit_minor"] for row in ledger), 0)
                for schema, artifacts in (
                    ("ledger-row", ledger), ("manifest", [manifest]),
                    ("truth", [contract.load(evaluator / "truth.json")]),
                    ("provenance", [contract.load(evaluator / "provenance.json")]),
                ):
                    validator = Draft202012Validator(
                        schemas[schema], registry=registry, format_checker=Draft202012Validator.FORMAT_CHECKER,
                    )
                    for artifact in artifacts:
                        validator.validate(artifact)
