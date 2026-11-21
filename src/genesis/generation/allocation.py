"""Publish duplicate accruals and missing or partial payment allocations."""

from collections import defaultdict
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING, Any

from genesis.domain.enums import FindingType
from genesis.domain.models import TruthIssue

if TYPE_CHECKING:
    from .company import Invoice, LedgerRow, Payment


@dataclass(frozen=True)
class _Candidate:
    invoice: "Invoice"
    payment: "Payment"
    expense: "LedgerRow"
    payable: "LedgerRow"
    allocation: "LedgerRow"
    bank: "LedgerRow"


def _posting_pair(
    rows: list["LedgerRow"], debit: str, credit: str, amount: int,
) -> tuple["LedgerRow", "LedgerRow"] | None:
    if len(rows) != 2 or rows[0]["row_id"] == rows[1]["row_id"]:
        return None
    debit_row, credit_row = sorted(rows, key=lambda row: row["credit_minor"])
    if (
        (debit_row["account"], debit_row["debit_minor"], debit_row["credit_minor"])
        == (debit, amount, 0)
        and (credit_row["account"], credit_row["debit_minor"], credit_row["credit_minor"])
        == (credit, 0, amount)
    ):
        return debit_row, credit_row
    return None


def _candidates(
    ledger: list["LedgerRow"], invoices: list["Invoice"], payments: list["Payment"],
    base_currency: str,
) -> list[_Candidate]:
    accruals: dict[str, list[LedgerRow]] = defaultdict(list)
    settlements: dict[str, list[LedgerRow]] = defaultdict(list)
    for row in ledger:
        payment_id, invoice_id = row["payment_id"], row["invoice_id"]
        if payment_id is not None:
            settlements[payment_id].append(row)
        elif invoice_id is not None:
            accruals[invoice_id].append(row)

    invoices_by_id = {invoice["invoice_id"]: invoice for invoice in invoices}
    by_invoice: dict[str, list[_Candidate]] = defaultdict(list)
    for payment in payments:
        amount = payment["amount_minor"]
        if (
            payment["currency"] != base_currency
            or len(payment["invoice_refs"]) != 1
            or amount <= 0
        ):
            continue
        invoice_id, = payment["invoice_refs"]
        invoice = invoices_by_id.get(invoice_id)
        if (
            invoice is None or invoice["currency"] != base_currency
            or invoice["total_minor"] != amount
        ):
            continue
        accrual = _posting_pair(accruals[invoice_id], "expense", "accounts_payable", amount)
        settlement = _posting_pair(
            settlements[payment["payment_id"]], "accounts_payable", "bank", amount,
        )
        if accrual is None or settlement is None:
            continue
        if any(
            row["currency"] != base_currency or row["invoice_id"] != invoice_id
            or row["entity_id"] != invoice["entity_id"] or row["timestamp"] != timestamp
            for pair, timestamp in ((accrual, invoice["issued_at"]), (settlement, payment["timestamp"]))
            for row in pair
        ):
            continue
        by_invoice[invoice_id].append(_Candidate(invoice, payment, *accrual, *settlement))
    return [entries[0] for _, entries in sorted(by_invoice.items()) if len(entries) == 1]


def apply_allocation_errors(
    ledger: list["LedgerRow"], invoices: list["Invoice"], payments: list["Payment"],
    *, base_currency: str, duplicate_count: int, allocation_count: int,
    rng: Random, new_id: Callable[[str], str],
) -> tuple[list[TruthIssue], list[dict[str, Any]]]:
    """Inject GEN-04 errors on disjoint, fully paid domestic invoices.

    The caller owns the new episode and supplies validated counts. Ledger rows
    are changed only after the independent candidate counts can be satisfied.
    """
    issues: list[TruthIssue] = []
    changes: list[dict[str, Any]] = []
    if not (duplicate_count or allocation_count):
        return issues, changes

    candidates = _candidates(ledger, invoices, payments, base_currency)
    if duplicate_count + allocation_count > len(candidates):
        raise ValueError(
            f"Requested {duplicate_count} duplicate and {allocation_count} allocation errors; "
            f"only {len(candidates)} disjoint fully paid domestic invoices are available"
        )
    chosen = rng.sample(candidates, duplicate_count + allocation_count)
    duplicate_candidates = chosen[:duplicate_count]
    allocation_candidates = chosen[duplicate_count:]
    replacements: dict[str, LedgerRow] = {}
    additions: list[LedgerRow] = []

    def record(
        issue_type: FindingType, amount: int, affected: list[str], evidence: list[str],
        before: dict[str, Any], after: dict[str, Any],
    ) -> None:
        issue_id = new_id("issue")
        affected_ids = list(dict.fromkeys(affected))
        issues.append({
            "issue_id": issue_id,
            "type": str(issue_type),
            "affected_ids": affected_ids,
            "expected_amount_minor": amount,
            "currency": base_currency,
            "acceptable_evidence_sets": [list(dict.fromkeys(evidence))],
        })
        changes.append({
            "change_id": new_id("change"),
            "issue_id": issue_id,
            "affected_ids": affected_ids.copy(),
            "before": deepcopy(before),
            "after": deepcopy(after),
        })

    for candidate in duplicate_candidates:
        original = [candidate.expense, candidate.payable]
        copied = [{**row, "row_id": new_id("row")} for row in original]
        additions.extend(copied)
        evidence = [candidate.invoice["doc_id"], *(row["row_id"] for row in original + copied)]
        amount = candidate.invoice["total_minor"]
        record(
            FindingType.DUPLICATE_INVOICE, amount, [candidate.invoice["invoice_id"], *evidence],
            evidence,
            {"rows": original, "posted_minor": amount},
            {"rows": original + copied, "posted_minor": 2 * amount},
        )

    for candidate in allocation_candidates:
        row = candidate.allocation
        amount = candidate.payment["amount_minor"]
        partial = amount > 1 and rng.choice((False, True))
        shortfall = max(1, amount // 4) if partial else amount
        suspense_id = new_id("row") if partial else row["row_id"]
        suspense: LedgerRow = {
            **row, "row_id": suspense_id, "account": "suspense", "invoice_id": None,
            "debit_minor": shortfall,
        }
        if partial:
            allocated: LedgerRow = {**row, "debit_minor": amount - shortfall}
            replacements[row["row_id"]] = allocated
            additions.append(suspense)
            after_rows = [allocated, suspense]
        else:
            replacements[row["row_id"]] = suspense
            after_rows = [suspense]
        evidence = [
            candidate.invoice["doc_id"], candidate.payment["doc_id"],
            candidate.bank["row_id"], row["row_id"], suspense_id,
        ]
        record(
            FindingType.PAYMENT_ALLOCATION, shortfall,
            [candidate.invoice["invoice_id"], candidate.payment["payment_id"], *evidence],
            evidence,
            {"rows": [row], "posted_minor": amount},
            {"rows": after_rows, "posted_minor": amount - shortfall,
             "subtype": "partial" if partial else "missing"},
        )

    ledger[:] = [replacements.get(row["row_id"], row) for row in ledger] + additions
    return issues, changes
