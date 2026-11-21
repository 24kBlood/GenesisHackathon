"""Independent arithmetic checks on the records cited by synthetic truth."""

from collections import defaultdict
from datetime import datetime
from decimal import Context, Decimal, localcontext
from zoneinfo import ZoneInfo

from genesis.generation.company import Episode


def _one(rows, predicate):
    records = [record for record in rows if predicate(record)]
    if len(records) != 1:
        raise ValueError("a required public record is missing or ambiguous")
    return records[0]


def _converted(amount, currency, target, when, quotes, public):
    if currency == target:
        return amount
    zone = ZoneInfo(public.policy["timezone"])
    day = datetime.fromisoformat(when).astimezone(zone).date().isoformat()
    quote = _one(
        quotes, lambda row: (row["date"], row["source_currency"], row["target_currency"]) == (day, currency, target),
    )
    with localcontext(Context(prec=64)):
        exponent = public.manifest["currency_exponents"]
        result = Decimal(amount).scaleb(-exponent[currency]) * Decimal(quote["rate"])
        return int(result.scaleb(exponent[target]).quantize(Decimal(1), rounding=public.policy["rounding_mode"]))


def _proof(kind, target, rows, invoices, payments, quotes, has_policy, public):
    if target != public.manifest["base_currency"] or any(row["currency"] != target for row in rows):
        raise ValueError("posting currency does not match the claimed currency")
    if kind == "duplicate_invoice":
        invoice = _one(invoices, lambda _: True)
        accrued = [row for row in rows if row["invoice_id"] == invoice["invoice_id"] and row["payment_id"] is None]
        if invoice["currency"] != target and not has_policy:
            raise ValueError("foreign duplicate requires the public policy")
        expected = _converted(invoice["total_minor"], invoice["currency"], target, invoice["issued_at"], quotes, public)
        expenses = [row["debit_minor"] for row in accrued if row["account"] == "expense"]
        payables = [row["credit_minor"] for row in accrued if row["account"] == "accounts_payable"]
        if len(expenses) < 2 or sorted(expenses) != sorted(payables) or set(expenses) != {expected}:
            raise ValueError("duplicate accrual is not established")
        return sum(expenses) - expected

    if kind == "period_cutoff":
        invoice = _one(invoices, lambda _: True)
        expense = _one(rows, lambda row: row["invoice_id"] == invoice["invoice_id"] and row["account"] == "expense")
        payable = _one(rows, lambda row: row["invoice_id"] == invoice["invoice_id"] and row["account"] == "accounts_payable")
        start, end, issued, posted = map(datetime.fromisoformat, (
            public.policy["period_start"], public.policy["period_end"], invoice["issued_at"], expense["timestamp"],
        ))
        if (
            not has_policy or public.policy.get("recognition_date_rule") != "document_date"
            or not start <= issued < end or start <= posted < end
            or payable["timestamp"] != expense["timestamp"] or payable["credit_minor"] != expense["debit_minor"]
        ):
            raise ValueError("cutoff discrepancy is not established")
        return expense["debit_minor"]

    payment = _one(payments, lambda _: True)
    payment_rows = [row for row in rows if row["payment_id"] == payment["payment_id"]]
    bank = _one(payment_rows, lambda row: row["account"] == "bank")
    if kind == "fx_conversion":
        if not has_policy or public.policy["fx_date_rule"] != "payment_date" or len(payment_rows) < 2:
            raise ValueError("conversion rule or paired posting is absent")
        expected = _converted(payment["amount_minor"], payment["currency"], target, payment["timestamp"], quotes, public)
        return abs(expected - bank["credit_minor"])
    if kind != "payment_allocation":
        raise ValueError("unsupported proof type")
    invoice = _one(invoices, lambda row: row["invoice_id"] in (payment["invoice_refs"] or []))
    _one(payment_rows, lambda row: row["account"] == "suspense")
    if payment["currency"] != target and not has_policy:
        raise ValueError("foreign allocation requires the public policy")
    expected = _converted(payment["amount_minor"], payment["currency"], target, payment["timestamp"], quotes, public)
    allocated = sum(
        row["debit_minor"] for row in payment_rows
        if row["account"] == "accounts_payable" and row["invoice_id"] == invoice["invoice_id"]
    )
    return expected - allocated


def verify_public_proofs(episode: Episode) -> None:
    """Require each accepted evidence set to establish its stated discrepancy."""
    public = episode.public
    by_id = {row["row_id"]: row for row in public.ledger}
    invoices_by_id = {doc["doc_id"]: doc for doc in public.invoices}
    payments_by_id = {doc["doc_id"]: doc for doc in public.payments}
    quotes_by_id = {
        f"fx:{row['date']}:{row['source_currency']}:{row['target_currency']}": row for row in public.fx
    }
    balances = defaultdict(int)
    for row in public.ledger:
        balances[row["currency"], row["timestamp"]] += row["debit_minor"] - row["credit_minor"]
    if any(balances.values()):
        raise ValueError("public ledger is unbalanced by currency and instant")

    for index, issue in enumerate(episode.evaluator.truth["issues"], 1):
        kind, target = issue["type"], issue["currency"]
        for evidence in issue["acceptable_evidence_sets"]:
            def cited(index):
                return [index[key] for key in evidence if key in index]
            try:
                amount = _proof(
                    kind, target, cited(by_id), cited(invoices_by_id), cited(payments_by_id),
                    cited(quotes_by_id), "policy" in evidence, public,
                )
            except ValueError as error:
                raise ValueError(f"issue {index} ({kind}): {error}") from None
            if amount <= 0 or amount != issue["expected_amount_minor"]:
                raise ValueError(f"issue {index} ({kind}): evidence does not establish the expected amount")
