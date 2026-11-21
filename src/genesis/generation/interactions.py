"""Causal mutations whose findings share an obligation or a settlement."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from random import Random
from typing import TYPE_CHECKING, Callable
from zoneinfo import ZoneInfo

from genesis.domain.enums import FindingType
from genesis.domain.models import TruthIssue
from genesis.domain.money import convert_minor

if TYPE_CHECKING:
    from .company import Invoice, LedgerRow, Payment, PublicEpisode


PARTIAL_FX_AT_CUTOFF = "partial_fx_payment_at_cutoff"
DUPLICATE_WITH_FX = "duplicate_invoice_with_fx"
_SCENARIOS = (PARTIAL_FX_AT_CUTOFF, DUPLICATE_WITH_FX)


@dataclass(frozen=True)
class InteractionTerms:
    issued_at: str
    paid_at: str
    amount_minor: int
    correct_rate: str
    wrong_rate: str
    alternative_quote_date: str | None = None


def normalize_scenarios(requested: object) -> list[str]:
    if not isinstance(requested, list):
        raise ValueError("interaction_scenarios must be an array")
    if any(type(name) is not str or name not in _SCENARIOS for name in requested):
        raise ValueError("unknown interaction scenario")
    if len(requested) != len(set(requested)):
        raise ValueError("duplicate interaction scenario")
    return [name for name in _SCENARIOS if name in requested]


def build_terms(scenarios: list[str], end: datetime, rng: Random) -> dict[str, InteractionTerms]:
    result = {}
    for name in scenarios:
        if name == PARTIAL_FX_AT_CUTOFF:
            issued = end - timedelta(days=1, hours=14)
            paid = end - timedelta(minutes=15)
            rate, wrong_rate, wrong_date = "3.40", "3.60", end.date().isoformat()
        else:
            issued = end - timedelta(days=6, hours=14)
            paid = end - timedelta(days=3, hours=12)
            rate, wrong_rate, wrong_date = "3.20", "3.25", None
        result[name] = InteractionTerms(
            issued.isoformat(), paid.isoformat(), rng.randint(400, 1400), rate, wrong_rate, wrong_date,
        )
    return result


def apply_interaction(
    public: PublicEpisode, name: str, invoice: Invoice, payment: Payment,
    terms: InteractionTerms, new_id: Callable[[str], str],
) -> tuple[list[TruthIssue], list[dict]]:
    """Apply two ordered mutations to a projected, clean scenario obligation."""
    ledger = public.ledger
    invoice_id, payment_id = invoice["invoice_id"], payment["payment_id"]
    (expense,) = (row for row in ledger if row["invoice_id"] == invoice_id and row["account"] == "expense")
    (accrual,) = (
        row for row in ledger
        if row["invoice_id"] == invoice_id and row["account"] == "accounts_payable" and row["payment_id"] is None
    )
    (allocation,) = (
        row for row in ledger if row["payment_id"] == payment_id and row["account"] == "accounts_payable"
    )
    (bank,) = (row for row in ledger if row["payment_id"] == payment_id and row["account"] == "bank")
    currency = bank["currency"]
    clean_posted = bank["credit_minor"]
    zone = ZoneInfo(public.policy["timezone"])
    payment_day = datetime.fromisoformat(payment["timestamp"]).astimezone(zone).date().isoformat()
    payment_key = f"fx:{payment_day}:{payment['currency']}:{currency}"
    issues: list[TruthIssue] = []
    changes: list[dict] = []

    def record(
        kind: FindingType, amount: int, affected: list[str], evidence: list[str],
        before: dict, after: dict, *, additional_evidence: list[str] | None = None,
    ) -> None:
        issue_id = new_id("issue")
        acceptable_evidence = [evidence]
        if additional_evidence is not None:
            acceptable_evidence.append(additional_evidence)
        issues.append({
            "issue_id": issue_id, "type": kind.value, "expected_amount_minor": amount,
            "currency": currency, "affected_ids": affected, "acceptable_evidence_sets": acceptable_evidence,
        })
        changes.append({
            "change_id": new_id("change"), "issue_id": issue_id, "affected_ids": affected.copy(),
            "before": before, "after": after,
        })

    if name == PARTIAL_FX_AT_CUTOFF:
        shortfall = clean_posted // 4
        allocated = {**allocation, "debit_minor": clean_posted - shortfall}
        suspense = {
            **allocation, "row_id": new_id("row"), "account": "suspense",
            "invoice_id": None, "debit_minor": shortfall,
        }
        ledger[ledger.index(allocation)] = allocated
        ledger.append(suspense)
        evidence = [
            invoice["doc_id"], payment["doc_id"], "policy", payment_key,
            bank["row_id"], allocated["row_id"], suspense["row_id"],
        ]
        affected = [invoice_id, payment_id, allocated["row_id"], suspense["row_id"]]
        record(
            FindingType.PAYMENT_ALLOCATION, shortfall, affected, evidence,
            {"rows": [allocation], "allocated_minor": clean_posted},
            {"rows": [allocated, suspense], "allocated_minor": allocated["debit_minor"]},
            additional_evidence=[*evidence, accrual["row_id"]],
        )
        fx_debit = suspense
    else:
        copied = [{**row, "row_id": new_id("row")} for row in (expense, accrual)]
        ledger.extend(copied)
        issuance_day = datetime.fromisoformat(invoice["issued_at"]).astimezone(zone).date().isoformat()
        accrual_key = f"fx:{issuance_day}:{invoice['currency']}:{currency}"
        original_rows = [expense, accrual]
        all_rows = [*original_rows, *copied]
        row_ids = [row["row_id"] for row in all_rows]
        evidence = [invoice["doc_id"], "policy", accrual_key, *row_ids]
        record(
            FindingType.DUPLICATE_INVOICE, expense["debit_minor"], [invoice_id, *row_ids], evidence,
            {"rows": original_rows, "recognized_minor": expense["debit_minor"]},
            {"rows": all_rows, "recognized_minor": 2 * expense["debit_minor"]},
        )
        fx_debit = allocation

    wrong_date = terms.alternative_quote_date
    wrong_rate = terms.wrong_rate
    if wrong_date is not None:
        (wrong_rate,) = (
            quote["rate"] for quote in public.fx
            if quote["date"] == wrong_date
            and quote["source_currency"] == payment["currency"]
            and quote["target_currency"] == currency
        )
    posted = convert_minor(
        payment["amount_minor"], payment["currency"], currency, wrong_rate,
        public.manifest["currency_exponents"], public.policy["rounding_mode"],
    )
    delta = posted - clean_posted
    amended_bank = {**bank, "credit_minor": posted}
    amended_debit = {**fx_debit, "debit_minor": fx_debit["debit_minor"] + delta}
    ledger[ledger.index(bank)] = amended_bank
    ledger[ledger.index(fx_debit)] = amended_debit
    affected = [bank["row_id"], fx_debit["row_id"]]
    evidence = [payment["doc_id"], "policy", payment_key, *affected]
    after = {"rows": [amended_bank, amended_debit], "posted_minor": posted, "applied_rate": wrong_rate}
    alternative = None
    if wrong_date is not None:
        alternative = [*evidence, f"fx:{wrong_date}:{payment['currency']}:{currency}"]
        after["quote_date"] = wrong_date
    record(
        FindingType.FX_CONVERSION, abs(delta), affected, evidence,
        {"rows": [bank, fx_debit], "posted_minor": clean_posted,
         "applied_rate": terms.correct_rate, "quote_date": payment_day},
        after, additional_evidence=alternative,
    )
    return issues, changes
