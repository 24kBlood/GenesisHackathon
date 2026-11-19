"""Artifact contract tests. Runtime process isolation is tested separately."""

import csv
import json
import re
import unittest
from copy import deepcopy
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

from jsonschema import Draft202012Validator, ValidationError, validators
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from genesis.domain.enums import FindingType
from genesis.domain.models import EpisodeTruth, FindingsSubmission


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "contract"
PUBLIC = FIXTURE / "public"
EVALUATOR = FIXTURE / "evaluator-only"
LEDGER_COLUMNS = (
    "row_id", "timestamp", "account", "debit_minor", "credit_minor", "currency",
    "entity_id", "invoice_id", "payment_id",
)
FX_COLUMNS = ("date", "source_currency", "target_currency", "rate")
PRIVATE_KEYS = {
    "seed", "truth", "provenance", "initial_history", "changes", "issues",
    "expected_amount_minor", "acceptable_evidence_sets",
}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def load_csv(path, columns):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(columns):
            raise ValueError(f"Unexpected columns in {path.name}")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed row in {path.name}")
            for name in ("debit_minor", "credit_minor"):
                if name in row:
                    value = row[name]
                    if re.fullmatch(r"-?(0|[1-9][0-9]*)", value) is None:
                        raise ValueError(f"Noninteger {name} in {path.name}")
                    row[name] = int(value)
            for name in ("invoice_id", "payment_id"):
                if name in row and row[name] == "null":
                    row[name] = None
            yield row


def field_names(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from field_names(item)
    elif isinstance(value, list):
        for item in value:
            yield from field_names(item)


class DataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schemas = {
            path.name.removesuffix(".schema.json"): load(path)
            for path in (ROOT / "schemas").glob("*.schema.json")
        }
        # JSON Schema's mathematical integer type accepts 12.0. The wire
        # contract uses a narrower JSON representation for all integer fields.
        # Resources use the known dialect and inherit the strict validator at
        # cross-file refs, including an invoice nested inside an observation.
        registry = Registry().with_resources(
            (schema["$id"], Resource(
                {key: value for key, value in schema.items() if key != "$schema"},
                DRAFT202012,
            ))
            for schema in cls.schemas.values()
        )
        wire_validator = validators.extend(
            Draft202012Validator,
            type_checker=Draft202012Validator.TYPE_CHECKER.redefine(
                "integer", lambda _, value: type(value) is int,
            ),
        )
        cls.validators = {
            name: wire_validator(schema, registry=registry, format_checker=Draft202012Validator.FORMAT_CHECKER)
            for name, schema in cls.schemas.items()
        }
        cls.manifest = load(PUBLIC / "manifest.json")
        cls.policy = load(PUBLIC / "policy.json")
        cls.ledger = list(load_csv(PUBLIC / "ledger.csv", LEDGER_COLUMNS))
        cls.fx = list(load_csv(PUBLIC / "fx.csv", FX_COLUMNS))
        cls.invoices = load_lines(PUBLIC / "invoices.jsonl")
        cls.payments = load_lines(PUBLIC / "payments.jsonl")
        cls.findings = cast(FindingsSubmission, load(PUBLIC / "findings.json"))
        cls.trajectory = load_lines(PUBLIC / "trajectories.jsonl")
        cls.truth = cast(EpisodeTruth, load(EVALUATOR / "truth.json"))
        cls.provenance = load(EVALUATOR / "provenance.json")

    def expect_pass(self, schema, sample):
        self.validators[schema].validate(sample)

    def expect_failure(self, schema, sample):
        with self.assertRaises(ValidationError):
            self.expect_pass(schema, sample)

    def test_metaschema_and_finding_enum(self):
        self.assertGreater(len(self.schemas), 6)
        self.assertEqual(len({schema["$id"] for schema in self.schemas.values()}), len(self.schemas))
        for name, schema in self.schemas.items():
            with self.subTest(schema=name):
                Draft202012Validator.check_schema(schema)
        self.assertEqual(
            set(self.schemas["common"]["$defs"]["error_type"]["enum"]),
            {kind.value for kind in FindingType},
        )

    def test_fixture_validates_against_all_artifact_schemas(self):
        groups = (
            ("manifest", [self.manifest]),
            ("policy", [self.policy]),
            ("ledger-row", self.ledger),
            ("fx-row", self.fx),
            ("invoice", self.invoices),
            ("payment", self.payments),
            ("findings", [self.findings]),
            ("trajectories", self.trajectory),
            ("action", [step["action"] for step in self.trajectory]),
            ("observation", [step["observation"] for step in self.trajectory]),
            ("truth", [self.truth]),
            ("config", [self.provenance["config"]]),
            ("provenance", [self.provenance]),
        )
        for schema, samples in groups:
            for sample in samples:
                with self.subTest(schema=schema):
                    self.expect_pass(schema, sample)

    def test_time_requires_real_rfc3339_offset(self):
        invoice = self.invoices[0]
        for timestamp in (
            "2026-11-05T10:00:00", "2026-11-05T10:00:00+0500",
            "2026-11-05T10:00:00+25:00", "2026-02-30T10:00:00+05:00",
            "2026-11-05", "2026-11-05T10:00:00+05:00\n",
        ):
            with self.subTest(timestamp=timestamp):
                self.expect_failure("invoice", {**invoice, "issued_at": timestamp})
        self.expect_pass("invoice", {**invoice, "issued_at": "2026-11-05T05:00:00Z"})
        self.expect_failure("fx-row", {**self.fx[0], "date": "2026-02-30"})

    def test_amount_and_rate_wire_types(self):
        for amount in ("10000", 10000.0, 1.25, True, None):
            with self.subTest(amount=amount):
                self.expect_failure("payment", {**self.payments[0], "amount_minor": amount})
        self.expect_failure("observation", {
            "remaining_steps": 99,
            "document": {**self.payments[0], "amount_minor": 10000.0},
        })
        for rate in (1.5, "0", "00.00", "-5", "NaN", "Infinity", "2e3", "1,2", "2\n"):
            with self.subTest(rate=rate):
                self.expect_failure("fx-row", {**self.fx[0], "rate": rate})
        self.expect_pass("fx-row", {**self.fx[0], "rate": "0.0000001"})

    def test_absent_links_remain_explicit_null(self):
        ledger = {**self.ledger[0], "invoice_id": None, "payment_id": None}
        self.expect_pass("ledger-row", ledger)
        self.expect_failure("ledger-row", {k: v for k, v in ledger.items() if k != "payment_id"})
        self.expect_failure("ledger-row", {**ledger, "invoice_id": ""})
        payment = {**self.payments[0], "invoice_refs": None}
        self.expect_pass("payment", payment)
        self.expect_failure("payment", {k: v for k, v in payment.items() if k != "invoice_refs"})
        self.expect_failure("payment", {**payment, "invoice_refs": [""]})

    def test_findings_limits_and_valid_duplicates(self):
        verdict = deepcopy(self.findings["findings"][0])
        verdict["explanation"] = "Я" * 1000
        verdict["evidence_ids"] = [f"id-{index}" for index in range(10)]
        payload = {"episode_id": self.manifest["episode_id"], "findings": [verdict] * 30}
        # Repeated or unsupported findings are evaluated as FP, not rejected as
        # malformed submission. The schema checks shape and budgets.
        self.expect_pass("findings", payload)
        self.expect_pass("findings", {**payload, "findings": []})
        self.expect_failure("findings", {**payload, "findings": [verdict] * 31})
        self.expect_failure("findings", {**payload, "findings": [{**verdict, "explanation": "Я" * 1001}]})
        self.expect_failure("findings", {
            **payload, "findings": [{**verdict, "evidence_ids": [*verdict["evidence_ids"], "extra"]}],
        })
        self.expect_failure("findings", {**payload, "findings": [{**verdict, "type": "hidden-fifth-type"}]})

    def test_public_action_and_trajectory_shapes(self):
        self.expect_pass("action", {"type": "list_records", "table": "ledger", "cursor": None, "limit": 50})
        self.expect_failure("action", {"type": "list_records", "table": "ledger", "limit": 51})
        self.expect_failure("action", {"type": "submit", "findings": []})
        self.expect_failure("action", {"type": "get_record", "id": "row"})
        step = self.trajectory[0]
        self.expect_pass("trajectories", {
            **step, "action": {"type": "invalid"},
            "observation": {"error": {"code": "invalid_action"}, "remaining_steps": 99},
        })
        self.expect_failure("trajectories", {**step, "terminated": True, "truncated": True})
        self.expect_failure("trajectories", {**step, "step": 101})
        self.expect_failure("trajectories", {**step, "reward": 1.01})
        self.expect_failure("trajectories", {**step, "reward": 0.01})
        self.expect_pass("trajectories", {**self.trajectory[-1], "reward": 0.98})
        self.expect_failure("observation", {"remaining_steps": 99, "records": [{}] * 51})

    def test_manifest_counts_currency_and_timezones(self):
        self.assertEqual(self.manifest["record_counts"], {
            "ledger": len(self.ledger), "invoices": len(self.invoices),
            "payments": len(self.payments), "fx": len(self.fx),
        })
        self.assertEqual(self.manifest["document_count"], len(self.invoices) + len(self.payments))
        for key in ("period_start", "period_end", "timezone", "rounding_mode"):
            self.assertEqual(self.manifest[key], self.policy[key])
        zone = ZoneInfo(self.manifest["timezone"])
        start, end = (datetime.fromisoformat(self.manifest[field]) for field in ("period_start", "period_end"))
        self.assertLess(start, end)
        timestamps = [
            self.manifest["period_start"], self.manifest["period_end"],
            *(row["timestamp"] for row in [*self.ledger, *self.payments]),
            *(invoice[field] for invoice in self.invoices for field in ("issued_at", "due_at")),
        ]
        for timestamp in timestamps:
            parsed = datetime.fromisoformat(timestamp)
            self.assertEqual(parsed.utcoffset(), parsed.astimezone(zone).utcoffset())
        for row in [*self.ledger, *self.payments]:
            self.assertLessEqual(start, datetime.fromisoformat(row["timestamp"]))
            self.assertLess(datetime.fromisoformat(row["timestamp"]), end)
        currencies = {
            self.manifest["base_currency"],
            *(row["currency"] for row in [*self.ledger, *self.invoices, *self.payments]),
            *(row[field] for row in self.fx for field in ("source_currency", "target_currency")),
            *(finding["currency"] for finding in self.findings["findings"]),
            *(issue["currency"] for issue in self.truth["issues"]),
        }
        self.assertLessEqual(currencies, set(self.manifest["currency_exponents"]))

    def test_fixture_cross_file_id_and_evidence_references(self):
        episode_id = self.manifest["episode_id"]
        self.assertTrue(all(
            sample["episode_id"] == episode_id
            for sample in [self.findings, self.truth, self.provenance, *self.trajectory]
        ))
        invoices = {doc["invoice_id"]: doc for doc in self.invoices}
        payments = {doc["payment_id"]: doc for doc in self.payments}
        ledger_ids = {row["row_id"] for row in self.ledger}
        self.assertEqual(len(ledger_ids), len(self.ledger))
        self.assertEqual(len(invoices), len(self.invoices))
        self.assertEqual(len(payments), len(self.payments))
        doc_ids = {doc["doc_id"] for doc in [*self.invoices, *self.payments]}
        fx_ids = {
            f"fx:{row['date']}:{row['source_currency']}:{row['target_currency']}" for row in self.fx
        }
        self.assertEqual(len(doc_ids), self.manifest["document_count"])
        self.assertEqual(len(fx_ids), len(self.fx))
        self.assertTrue(ledger_ids.isdisjoint(doc_ids | fx_ids | {"policy"}))
        evidence_catalog = ledger_ids | doc_ids | fx_ids | {"policy"}
        affected_catalog = evidence_catalog | set(invoices) | set(payments)

        for payment in payments.values():
            self.assertLessEqual(set(payment["invoice_refs"] or []), set(invoices))
        for row in self.ledger:
            if row["invoice_id"] is not None:
                self.assertIn(row["invoice_id"], invoices)
                self.assertEqual(row["entity_id"], invoices[row["invoice_id"]]["entity_id"])
            if row["payment_id"] is not None:
                self.assertIn(row["payment_id"], payments)
        for issue in self.truth["issues"]:
            self.assertLessEqual(set(issue["affected_ids"]), affected_catalog)
            for evidence in issue["acceptable_evidence_sets"]:
                self.assertLessEqual(set(evidence), evidence_catalog)
        for finding in self.findings["findings"]:
            self.assertLessEqual(set(finding["affected_ids"]), affected_catalog)
            self.assertLessEqual(set(finding["evidence_ids"]), evidence_catalog)
        truth_ids = {issue["issue_id"] for issue in self.truth["issues"]}
        self.assertEqual(truth_ids, {change["issue_id"] for change in self.provenance["changes"]})

    def test_fixture_fx_discrepancy_recomputes_from_public_records(self):
        payment = self.payments[0]
        rate = self.fx[0]
        exponents = self.manifest["currency_exponents"]
        self.assertEqual(self.policy["fx_date_rule"], "payment_date")
        self.assertEqual(datetime.fromisoformat(payment["timestamp"]).date().isoformat(), rate["date"])
        self.assertEqual(payment["currency"], rate["source_currency"])
        self.assertEqual(rate["target_currency"], self.manifest["base_currency"])
        major = Decimal(payment["amount_minor"]).scaleb(-exponents[payment["currency"]])
        converted = (major * Decimal(rate["rate"])).scaleb(exponents[rate["target_currency"]])
        expected = int(converted.quantize(Decimal(1), rounding=self.policy["rounding_mode"]))
        recorded = sum(
            row["credit_minor"] for row in self.ledger
            if row["account"] == "bank" and row["payment_id"] == payment["payment_id"]
        )
        self.assertNotEqual(expected, recorded)
        self.assertEqual(abs(expected - recorded), self.truth["issues"][0]["expected_amount_minor"])

    def test_public_fixture_projections_have_no_private_keys(self):
        for sample in [
            self.manifest, self.policy, self.findings, *self.invoices, *self.payments,
            *self.ledger, *self.fx, *self.trajectory,
        ]:
            self.assertTrue(PRIVATE_KEYS.isdisjoint(field_names(sample)))
        for field in PRIVATE_KEYS:
            with self.subTest(field=field):
                self.expect_failure("observation", {**self.trajectory[0]["observation"], field: None})
        invoice = deepcopy(self.invoices[0])
        invoice["line_items"][0]["truth"] = {}
        self.expect_failure("observation", {"remaining_steps": 99, "document": invoice})
        for step_number, step in enumerate(self.trajectory, 1):
            self.assertEqual(step["step"], step_number)
            self.assertEqual(step["observation"]["remaining_steps"], 100 - step_number)
            self.assertEqual(step["reward"], 0)
        self.assertEqual(self.trajectory[-1]["action"]["findings"], self.findings)
        self.assertTrue(self.trajectory[-1]["terminated"])
        self.assertFalse(self.trajectory[-1]["truncated"])


if __name__ == "__main__":
    unittest.main()
