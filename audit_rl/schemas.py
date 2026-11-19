"""Shared schemas, constants, validation, and serialization for audit_rl."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_EVEN, ROUND_HALF_UP
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0.0"

ERROR_TYPES: tuple[str, ...] = (
    "duplicate_invoice",
    "payment_allocation_mismatch",
    "fx_conversion_error",
    "period_cutoff_error",
)

INTERACTING_SCENARIOS: tuple[str, ...] = (
    "fx_period_partial_payment",
    "duplicate_invoice_fx_mismatch",
)

CURRENCY_EXPONENTS: dict[str, int] = {
    "USD": 2,
    "EUR": 2,
    "GBP": 2,
    "KZT": 2,
    "CHF": 2,
    "JPY": 0,
}

DEFAULT_MAX_STEPS = 100
MAX_PAGE_LIMIT = 50
MAX_FINDINGS = 30
MAX_EVIDENCE_PER_FINDING = 10
MAX_EXPLANATION_CHARS = 1000

VALID_FX_DATE_POLICIES: tuple[str, ...] = (
    "invoice_issue_date",
    "payment_date",
    "posting_date",
)

VALID_ROUNDING_RULES: tuple[str, ...] = (
    "ROUND_HALF_EVEN",
    "ROUND_HALF_UP",
)

LEDGER_CSV_COLUMNS: tuple[str, ...] = (
    "row_id",
    "timestamp",
    "account",
    "debit_minor",
    "credit_minor",
    "currency",
    "entity_id",
    "invoice_id",
    "payment_id",
)

FX_CSV_COLUMNS: tuple[str, ...] = (
    "date",
    "source_currency",
    "target_currency",
    "rate",
)

_ROUNDING_MAP = {
    "ROUND_HALF_EVEN": ROUND_HALF_EVEN,
    "ROUND_HALF_UP": ROUND_HALF_UP,
}


def fx_record_key(date: str, source_currency: str, target_currency: str) -> str:
    """Canonical FX record key matching get_record('fx', id)."""
    return f"{date}:{source_currency}:{target_currency}"


def convert_minor(
    amount_minor: int,
    source_currency: str,
    target_currency: str,
    rate_str: str,
    rounding_rule: str = "ROUND_HALF_EVEN",
) -> int:
    """Convert integer minor units from source_currency to target_currency using Decimal."""
    if rounding_rule not in _ROUNDING_MAP:
        raise ValueError(f"Unsupported rounding_rule: {rounding_rule}")
    if source_currency == target_currency:
        return int(amount_minor)
    src_exp = CURRENCY_EXPONENTS.get(source_currency, 2)
    dst_exp = CURRENCY_EXPONENTS.get(target_currency, 2)
    rate = Decimal(str(rate_str))
    scale_shift = Decimal(10) ** (dst_exp - src_exp)
    converted = Decimal(int(amount_minor)) * rate * scale_shift
    return int(converted.quantize(Decimal("1"), rounding=_ROUNDING_MAP[rounding_rule]))


@dataclass
class EpisodeConfig:
    """Configuration for generating a synthetic financial audit episode."""

    num_invoices: int = 35
    base_currency: str = "USD"
    currencies: tuple[str, ...] = ("USD", "EUR", "GBP", "KZT", "JPY")
    timezone_offset: str = "+05:00"
    period_start: str = "2026-01-01T00:00:00+05:00"
    period_end: str = "2026-03-31T23:59:59+05:00"
    fx_date_policy: str = "invoice_issue_date"
    rounding_rule: str = "ROUND_HALF_EVEN"
    error_counts: dict[str, int] = field(
        default_factory=lambda: {error_type: 0 for error_type in ERROR_TYPES}
    )
    interacting_scenarios: list[str] = field(default_factory=list)
    include_distractors: bool = True
    clean: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.num_invoices, bool) or not isinstance(self.num_invoices, int):
            raise ValueError("num_invoices must be an int")
        if self.num_invoices < 10 or self.num_invoices > 150:
            raise ValueError("num_invoices must be between 10 and 150")

        self.base_currency = str(self.base_currency).upper().strip()
        if self.base_currency not in CURRENCY_EXPONENTS:
            raise ValueError(f"Unsupported base_currency: {self.base_currency}")

        norm_currencies: list[str] = []
        for cur in self.currencies:
            c = str(cur).upper().strip()
            if c not in CURRENCY_EXPONENTS:
                raise ValueError(f"Unsupported currency: {c}")
            if c not in norm_currencies:
                norm_currencies.append(c)
        if self.base_currency not in norm_currencies:
            norm_currencies.insert(0, self.base_currency)
        self.currencies = tuple(norm_currencies)

        if self.fx_date_policy not in VALID_FX_DATE_POLICIES:
            raise ValueError(f"Invalid fx_date_policy: {self.fx_date_policy}")
        if self.rounding_rule not in VALID_ROUNDING_RULES:
            raise ValueError(f"Invalid rounding_rule: {self.rounding_rule}")

        normalized_counts = {error_type: 0 for error_type in ERROR_TYPES}
        if self.error_counts:
            for key, val in self.error_counts.items():
                if key not in ERROR_TYPES:
                    raise ValueError(f"Unknown error type in error_counts: {key}")
                if isinstance(val, bool) or not isinstance(val, int) or val < 0:
                    raise ValueError(f"error_counts[{key}] must be a non-negative int")
                normalized_counts[key] = val

        scenarios = list(self.interacting_scenarios) if self.interacting_scenarios else []
        for scenario in scenarios:
            if scenario not in INTERACTING_SCENARIOS:
                raise ValueError(f"Unknown interacting scenario: {scenario}")

        if self.clean:
            normalized_counts = {error_type: 0 for error_type in ERROR_TYPES}
            scenarios = []

        self.error_counts = normalized_counts
        self.interacting_scenarios = scenarios

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_invoices": self.num_invoices,
            "base_currency": self.base_currency,
            "currencies": list(self.currencies),
            "timezone_offset": self.timezone_offset,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "fx_date_policy": self.fx_date_policy,
            "rounding_rule": self.rounding_rule,
            "error_counts": dict(self.error_counts),
            "interacting_scenarios": list(self.interacting_scenarios),
            "include_distractors": bool(self.include_distractors),
            "clean": bool(self.clean),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | EpisodeConfig | None) -> EpisodeConfig:
        if data is None:
            return cls()
        if isinstance(data, cls):
            return cls(**data.to_dict())
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict or EpisodeConfig, got {type(data).__name__}")
        kwargs: dict[str, Any] = {}
        valid_fields = {
            "num_invoices",
            "base_currency",
            "currencies",
            "timezone_offset",
            "period_start",
            "period_end",
            "fx_date_policy",
            "rounding_rule",
            "error_counts",
            "interacting_scenarios",
            "include_distractors",
            "clean",
        }
        for key, value in data.items():
            if key in valid_fields:
                if key == "currencies" and value is not None:
                    kwargs[key] = tuple(value)
                elif key == "error_counts" and value is not None:
                    kwargs[key] = dict(value)
                elif key == "interacting_scenarios" and value is not None:
                    kwargs[key] = list(value)
                else:
                    kwargs[key] = value
        return cls(**kwargs)


@dataclass
class GeneratedEpisode:
    """In-memory representation of a generated audit episode (open + private parts)."""

    manifest: dict[str, Any]
    policy: dict[str, Any]
    ledger: list[dict[str, Any]]
    invoices: list[dict[str, Any]]
    payments: list[dict[str, Any]]
    fx_rates: list[dict[str, Any]]
    truth: dict[str, Any]
    provenance: dict[str, Any]


def validate_findings_payload(payload: Any, expected_episode_id: str) -> list[str]:
    """Validate a submitted findings payload against the strict submission schema.

    Returns a list of human-readable error strings; empty list means valid.
    """
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["findings payload must be a JSON object"]

    episode_id = payload.get("episode_id")
    if not isinstance(episode_id, str) or episode_id != expected_episode_id:
        errors.append(
            f"episode_id mismatch: expected {expected_episode_id!r}, got {episode_id!r}"
        )

    findings = payload.get("findings")
    if not isinstance(findings, list):
        errors.append("findings must be a list")
        return errors

    if len(findings) > MAX_FINDINGS:
        errors.append(
            f"too many findings: {len(findings)} exceeds maximum {MAX_FINDINGS}"
        )

    for idx, finding in enumerate(findings):
        prefix = f"findings[{idx}]"
        if not isinstance(finding, dict):
            errors.append(f"{prefix} must be an object")
            continue

        ftype = finding.get("type")
        if not isinstance(ftype, str) or ftype.strip().lower() not in ERROR_TYPES:
            errors.append(f"{prefix}.type must be one of {ERROR_TYPES}, got {ftype!r}")

        affected_ids = finding.get("affected_ids")
        if (
            not isinstance(affected_ids, list)
            or len(affected_ids) == 0
            or not all(isinstance(x, str) and x.strip() != "" for x in affected_ids)
        ):
            errors.append(f"{prefix}.affected_ids must be a non-empty list of strings")

        claimed = finding.get("claimed_amount_minor")
        if isinstance(claimed, bool) or not isinstance(claimed, int):
            errors.append(
                f"{prefix}.claimed_amount_minor must be an int (not bool), got {type(claimed).__name__}"
            )

        currency = finding.get("currency")
        if not isinstance(currency, str) or currency.strip() == "":
            errors.append(f"{prefix}.currency must be a non-empty string")

        evidence_ids = finding.get("evidence_ids")
        if (
            not isinstance(evidence_ids, list)
            or len(evidence_ids) > MAX_EVIDENCE_PER_FINDING
            or not all(isinstance(x, str) and x.strip() != "" for x in evidence_ids)
        ):
            errors.append(
                f"{prefix}.evidence_ids must be a list of at most "
                f"{MAX_EVIDENCE_PER_FINDING} non-empty strings"
            )

        explanation = finding.get("explanation")
        if not isinstance(explanation, str) or len(explanation) > MAX_EXPLANATION_CHARS:
            errors.append(
                f"{prefix}.explanation must be a string of at most "
                f"{MAX_EXPLANATION_CHARS} characters"
            )

    return errors


def _write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False))
            fh.write("\n")


def save_episode(
    episode: GeneratedEpisode,
    output_dir: str | Path,
    include_private: bool = True,
) -> Path:
    """Save an episode's open (and optionally private) files to output_dir."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    _write_json(out_path / "manifest.json", episode.manifest)
    _write_json(out_path / "policy.json", episode.policy)

    with (out_path / "ledger.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(LEDGER_CSV_COLUMNS))
        writer.writeheader()
        for row in episode.ledger:
            writer.writerow(
                {
                    "row_id": row["row_id"],
                    "timestamp": row["timestamp"],
                    "account": row["account"],
                    "debit_minor": int(row["debit_minor"]),
                    "credit_minor": int(row["credit_minor"]),
                    "currency": row["currency"],
                    "entity_id": "" if row.get("entity_id") is None else row["entity_id"],
                    "invoice_id": "" if row.get("invoice_id") is None else row["invoice_id"],
                    "payment_id": "" if row.get("payment_id") is None else row["payment_id"],
                }
            )

    _write_jsonl(out_path / "invoices.jsonl", episode.invoices)
    _write_jsonl(out_path / "payments.jsonl", episode.payments)

    with (out_path / "fx.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FX_CSV_COLUMNS))
        writer.writeheader()
        for row in episode.fx_rates:
            writer.writerow(
                {
                    "date": row["date"],
                    "source_currency": row["source_currency"],
                    "target_currency": row["target_currency"],
                    "rate": str(row["rate"]),
                }
            )

    if include_private:
        _write_json(out_path / "truth.json", episode.truth)
        _write_json(out_path / "provenance.json", episode.provenance)

    return out_path


def load_episode_open(episode_dir: str | Path) -> dict[str, Any]:
    """Load only the open (agent-visible) files of a saved episode."""
    ep_path = Path(episode_dir)
    with (ep_path / "manifest.json").open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    with (ep_path / "policy.json").open("r", encoding="utf-8") as fh:
        policy = json.load(fh)

    ledger: list[dict[str, Any]] = []
    with (ep_path / "ledger.csv").open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ledger.append(
                {
                    "row_id": row["row_id"],
                    "timestamp": row["timestamp"],
                    "account": row["account"],
                    "debit_minor": int(row["debit_minor"]),
                    "credit_minor": int(row["credit_minor"]),
                    "currency": row["currency"],
                    "entity_id": row["entity_id"] if row["entity_id"] != "" else None,
                    "invoice_id": row["invoice_id"] if row["invoice_id"] != "" else None,
                    "payment_id": row["payment_id"] if row["payment_id"] != "" else None,
                }
            )

    invoices: list[dict[str, Any]] = []
    with (ep_path / "invoices.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                invoices.append(json.loads(line))

    payments: list[dict[str, Any]] = []
    with (ep_path / "payments.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                payments.append(json.loads(line))

    fx_rates: list[dict[str, Any]] = []
    with (ep_path / "fx.csv").open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fx_rates.append(
                {
                    "date": row["date"],
                    "source_currency": row["source_currency"],
                    "target_currency": row["target_currency"],
                    "rate": str(row["rate"]),
                }
            )

    return {
        "manifest": manifest,
        "policy": policy,
        "ledger": ledger,
        "invoices": invoices,
        "payments": payments,
        "fx_rates": fx_rates,
    }


def load_episode_private(episode_dir: str | Path) -> dict[str, Any]:
    """Load only the private (evaluator-only) files of a saved episode."""
    ep_path = Path(episode_dir)
    with (ep_path / "truth.json").open("r", encoding="utf-8") as fh:
        truth = json.load(fh)
    with (ep_path / "provenance.json").open("r", encoding="utf-8") as fh:
        provenance = json.load(fh)
    return {
        "truth": truth,
        "provenance": provenance,
    }


def load_episode(episode_dir: str | Path) -> GeneratedEpisode:
    """Load a complete GeneratedEpisode (open + private files) from disk."""
    open_data = load_episode_open(episode_dir)
    private_data = load_episode_private(episode_dir)
    return GeneratedEpisode(
        manifest=open_data["manifest"],
        policy=open_data["policy"],
        ledger=open_data["ledger"],
        invoices=open_data["invoices"],
        payments=open_data["payments"],
        fx_rates=open_data["fx_rates"],
        truth=private_data["truth"],
        provenance=private_data["provenance"],
    )
