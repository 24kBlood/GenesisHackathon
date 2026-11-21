"""Reproducible, correct transactions that resemble common audit findings."""

from collections.abc import Callable
from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .company import Invoice, Payment


class _Post(Protocol):
    def __call__(
        self, timestamp: str, entity_id: str, amount: int, debit: str, credit: str, *,
        invoice_id: str, payment_id: str | None = None,
    ) -> None: ...


@dataclass(frozen=True)
class DistractorPlan:
    amount: int
    first_installment: int
    deposit: int

    @property
    def opening_cash(self) -> int:
        return 2 * self.amount + self.deposit


def plan_clean_distractors(rng: Random) -> DistractorPlan:
    amount = rng.randint(9_000, 40_000)
    return DistractorPlan(amount, amount // rng.randint(2, 4), rng.randint(500, 5_000))


def add_clean_distractors(
    invoices: list["Invoice"], payments: list["Payment"], *, base_currency: str,
    plan: DistractorPlan, new_id: Callable[[str], str], date_at: Callable[[int], str],
    post: _Post,
) -> dict[str, Any]:
    """Publish two distinguishable obligations and their legitimate cash events."""
    vendor = new_id("entity")

    def issue_invoice(day: int, description: str) -> str:
        invoice_id = new_id("invoice")
        issued_at = date_at(day)
        invoices.append({
            "doc_id": new_id("doc"), "invoice_id": invoice_id, "entity_id": vendor,
            "issued_at": issued_at, "due_at": date_at(22), "currency": base_currency,
            "total_minor": plan.amount,
            "line_items": [{
                "description": description, "quantity": 1, "unit_price_minor": plan.amount,
                "amount_minor": plan.amount,
            }],
        })
        post(issued_at, vendor, plan.amount, "expense", "accounts_payable", invoice_id=invoice_id)
        return invoice_id

    def payment(invoice_id: str, day: int, amount: int) -> "Payment":
        return {
            "doc_id": new_id("doc"), "payment_id": new_id("payment"), "timestamp": date_at(day),
            "currency": base_currency, "amount_minor": amount, "invoice_refs": [invoice_id],
        }

    split_invoice = issue_invoice(10, "Archive inventory")
    split_ids = []
    for day, amount in ((12, plan.first_installment), (17, plan.amount - plan.first_installment)):
        installment = payment(split_invoice, day, amount)
        payments.append(installment)
        split_ids.append(installment["payment_id"])
        post(
            installment["timestamp"], vendor, amount, "accounts_payable", "bank",
            invoice_id=split_invoice, payment_id=installment["payment_id"],
        )

    refund_invoice = issue_invoice(11, "Courier service")
    original = payment(refund_invoice, 14, plan.amount + plan.deposit)
    payments.append(original)
    for amount, debit in ((plan.amount, "accounts_payable"), (plan.deposit, "vendor_receivable")):
        post(
            original["timestamp"], vendor, amount, debit, "bank",
            invoice_id=refund_invoice, payment_id=original["payment_id"],
        )
    refund = payment(refund_invoice, 19, plan.deposit)
    refund["payment_kind"] = "refund"
    refund["refund_of_payment_id"] = original["payment_id"]
    payments.append(refund)
    post(
        refund["timestamp"], vendor, plan.deposit, "bank", "vendor_receivable",
        invoice_id=refund_invoice, payment_id=refund["payment_id"],
    )

    return {
        "regular_split": {"invoice_id": split_invoice, "payment_ids": split_ids},
        "overpayment_refund": {
            "invoice_id": refund_invoice, "original_payment_id": original["payment_id"],
            "refund_payment_id": refund["payment_id"], "deposit_minor": plan.deposit,
        },
        "separate_obligations": {
            "entity_id": vendor, "invoice_ids": [split_invoice, refund_invoice],
            "obligation_minor": plan.amount,
        },
    }
