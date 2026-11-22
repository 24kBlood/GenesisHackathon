"""Deterministic injections into independently reconcilable postings."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Context, Decimal, localcontext
from typing import TYPE_CHECKING, Any, Callable, Collection, Iterator
from zoneinfo import ZoneInfo

from genesis.domain.enums import FindingType
from genesis.domain.models import TruthIssue
from genesis.domain.money import convert_minor

if TYPE_CHECKING:
    from .company import Invoice, LedgerRow, Payment, PublicEpisode


@dataclass(frozen=True)
class _Mutation:
    rows: tuple[LedgerRow, LedgerRow]
    documents: tuple[str, ...]
    amount: int
    before: dict[str, Any]
    after: dict[str, Any]


def in_period(timestamp: str, policy: dict[str, str]) -> bool:
    """Compare aware instants against the half-open reporting interval."""
    instant = datetime.fromisoformat(timestamp).astimezone(timezone.utc)
    start = datetime.fromisoformat(policy["period_start"]).astimezone(timezone.utc)
    end = datetime.fromisoformat(policy["period_end"]).astimezone(timezone.utc)
    return start <= instant < end


def _paired_legs(rows: list[LedgerRow]) -> tuple[LedgerRow, LedgerRow] | None:
    if len(rows) != 2:
        return None
    debit, credit = sorted(rows, key=lambda row: row["credit_minor"])
    if (
        debit["debit_minor"] <= 0 or debit["credit_minor"] != 0
        or credit["debit_minor"] != 0 or credit["credit_minor"] != debit["debit_minor"]
        or debit["currency"] != credit["currency"]
        or debit["timestamp"] != credit["timestamp"]
    ):
        return None
    return debit, credit


def _quote_date(
    policy: dict[str, str], payment: Payment, invoice: Invoice | None, posting: LedgerRow,
) -> str | None:
    rule = policy["fx_date_rule"]
    if rule == "invoice_date":
        if invoice is None:
            return None
        timestamp = invoice["issued_at"]
    elif rule == "posting_date":
        timestamp = posting["timestamp"]
    elif rule == "payment_date":
        timestamp = payment["timestamp"]
    else:
        return None
    return datetime.fromisoformat(timestamp).astimezone(ZoneInfo(policy["timezone"])).date().isoformat()


def _fx_mutations(
    public: PublicEpisode, count: int, reserved_payment_ids: Collection[str],
    by_payment: dict[str, list[LedgerRow]],
) -> Iterator[_Mutation]:
    invoices = {invoice["invoice_id"]: invoice for invoice in public.invoices}
    quotes: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for row in public.fx:
        quotes[row["date"], row["source_currency"], row["target_currency"]].append(row["rate"])
    exponents = public.manifest["currency_exponents"]
    rounding = public.policy["rounding_mode"]
    for payment in sorted(public.payments, key=lambda doc: doc["payment_id"]):
        source = payment["currency"]
        if source == public.manifest["base_currency"] or payment["payment_id"] in reserved_payment_ids:
            continue
        legs = _paired_legs(by_payment.get(payment["payment_id"], []))
        if legs is None:
            continue
        debit, credit = legs
        target = debit["currency"]
        refs = payment["invoice_refs"] or []
        invoice = invoices.get(refs[0]) if len(refs) == 1 else None
        day = _quote_date(public.policy, payment, invoice, debit)
        if day is None or source == target:
            continue
        rates = quotes.get((day, source, target), [])
        if len(rates) != 1:
            continue
        rate = rates[0]
        source_amount = payment["amount_minor"]
        correct = convert_minor(source_amount, source, target, rate, exponents, rounding)
        if correct != debit["debit_minor"]:
            continue
        with localcontext(Context(prec=64)):
            wrong_rate = format(Decimal(rate) * Decimal("1.01"), "f")
        posted = convert_minor(source_amount, source, target, wrong_rate, exponents, rounding)
        if posted == correct:
            # Allow room for rounding when a one-percent change disappears in the ledger.
            with localcontext(Context(prec=64)):
                target_amount = Decimal(correct + 2).scaleb(-exponents[target])
                source_amount_major = Decimal(source_amount).scaleb(-exponents[source])
                wrong_rate = format(target_amount / source_amount_major, "f")
            posted = convert_minor(source_amount, source, target, wrong_rate, exponents, rounding)
        documents = (payment["doc_id"], f"fx:{day}:{source}:{target}", "policy")
        if public.policy["fx_date_rule"] == "invoice_date" and invoice is not None:
            documents += (invoice["doc_id"],)
        yield _Mutation(
            ({**debit, "debit_minor": posted}, {**credit, "credit_minor": posted}),
            documents,
            abs(correct - posted),
            {"posted_minor": correct, "applied_rate": rate},
            {"posted_minor": posted, "applied_rate": wrong_rate},
        )
        count -= 1
        if not count:
            return
    raise ValueError("not enough independently reconcilable fx_conversion candidates")


def _cutoff_mutations(
    public: PublicEpisode, count: int, accruals_by_invoice: dict[str, list[LedgerRow]],
) -> Iterator[_Mutation]:
    policy = public.policy
    if policy.get("recognition_date_rule") != "document_date":
        raise ValueError("period_cutoff requires policy recognition_date_rule=document_date")
    zone = ZoneInfo(policy["timezone"])
    start = datetime.fromisoformat(policy["period_start"]).astimezone(timezone.utc)
    end = datetime.fromisoformat(policy["period_end"]).astimezone(timezone.utc)
    paid_invoices = {ref for payment in public.payments for ref in (payment["invoice_refs"] or [])}
    near_start: list[_Mutation] = []
    near_end: list[_Mutation] = []
    for invoice in sorted(public.invoices, key=lambda doc: doc["invoice_id"]):
        invoice_id = invoice["invoice_id"]
        if invoice_id in paid_invoices or invoice["currency"] != public.manifest["base_currency"]:
            continue
        recognition = datetime.fromisoformat(invoice["issued_at"]).astimezone(timezone.utc)
        if not in_period(invoice["issued_at"], policy) or min(recognition - start, end - recognition) > timedelta(days=1):
            continue
        legs = _paired_legs(accruals_by_invoice.get(invoice_id, []))
        if legs is None:
            continue
        debit, credit = legs
        if debit["timestamp"] != invoice["issued_at"] or debit["debit_minor"] != invoice["total_minor"]:
            continue
        is_start = recognition - start < end - recognition
        outside = start - timedelta(minutes=1) if is_start else end
        timestamp = outside.astimezone(zone).isoformat()
        mutation = _Mutation(
            ({**debit, "timestamp": timestamp}, {**credit, "timestamp": timestamp}),
            (invoice["doc_id"], "policy"),
            debit["debit_minor"],
            {"timestamp": debit["timestamp"], "inside_period": True},
            {"timestamp": timestamp, "inside_period": False},
        )
        (near_start if is_start else near_end).append(mutation)
    interleaved: list[_Mutation] = []
    for index in range(max(len(near_start), len(near_end))):
        if index < len(near_start):
            interleaved.append(near_start[index])
        if index < len(near_end):
            interleaved.append(near_end[index])
    if len(interleaved) < count:
        raise ValueError("not enough independently reconcilable period_cutoff candidates")
    yield from interleaved[:count]


def apply_fx_cutoff_errors(
    public: PublicEpisode, counts: dict[str, int], new_id: Callable[[str], str],
    *, reserved_payment_ids: Collection[str] = (),
) -> tuple[list[LedgerRow], list[TruthIssue], list[dict[str, Any]]]:
    """Return the revised ledger and its private truth and provenance entries."""
    by_payment: dict[str, list[LedgerRow]] = defaultdict(list)
    accruals_by_invoice: dict[str, list[LedgerRow]] = defaultdict(list)
    for row in public.ledger:
        payment_id = row["payment_id"]
        if payment_id is not None:
            by_payment[payment_id].append(row)
        elif row["invoice_id"] is not None:
            accruals_by_invoice[row["invoice_id"]].append(row)

    replacements: dict[str, LedgerRow] = {}
    issues: list[TruthIssue] = []
    changes: list[dict[str, Any]] = []
    for name, mutations in (
        (FindingType.FX_CONVERSION, _fx_mutations(
            public, counts.get(FindingType.FX_CONVERSION, 0), reserved_payment_ids, by_payment,
        )),
        (FindingType.PERIOD_CUTOFF, _cutoff_mutations(
            public, counts.get(FindingType.PERIOD_CUTOFF, 0), accruals_by_invoice,
        )),
    ):
        if not counts.get(name, 0):
            continue
        for mutation in mutations:
            affected = [row["row_id"] for row in mutation.rows]
            issue_id = new_id("issue")
            issues.append({
                "issue_id": issue_id,
                "type": name.value,
                "affected_ids": affected,
                "expected_amount_minor": mutation.amount,
                "currency": mutation.rows[0]["currency"],
                "acceptable_evidence_sets": [[*mutation.documents, *affected]],
            })
            changes.append({
                "change_id": new_id("change"),
                "issue_id": issue_id,
                "affected_ids": affected,
                "before": mutation.before,
                "after": mutation.after,
            })
            replacements.update((row["row_id"], row) for row in mutation.rows)
    return [replacements.get(row["row_id"], row) for row in public.ledger], issues, changes
