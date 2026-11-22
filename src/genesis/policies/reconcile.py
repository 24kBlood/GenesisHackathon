"""A public, conservative rule-based reconciliation across all v1 finding types."""

from collections import defaultdict
from datetime import datetime, timezone
from decimal import DecimalException
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from genesis.domain.enums import FindingType
from genesis.domain.models import Finding
from genesis.domain.money import convert_minor

from .protocol import Action, Observation, submit


_TABLES = ("payments", "invoices", "fx", "ledger")
MAX_FINDINGS = 30
MAX_EVIDENCE_IDS = 10


def _parse_aware(timestamp: object) -> datetime | None:
    if not isinstance(timestamp, str):
        return None
    try:
        instant = datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    return None if instant.tzinfo is None else instant


def _fx_convert(
    amount: int,
    source: str,
    target: str,
    timestamp: object,
    zone: ZoneInfo,
    quotes: dict[tuple[str, str, str], list[str]],
    exponents: dict[str, int],
    rounding: str,
) -> tuple[int, str] | None:
    if source == target:
        return amount, ""
    instant = _parse_aware(timestamp)
    if instant is None:
        return None
    day = instant.astimezone(zone).date().isoformat()
    rates = quotes.get((day, source, target), [])
    if len(rates) != 1 or source not in exponents or target not in exponents:
        return None
    try:
        converted = convert_minor(amount, source, target, rates[0], exponents, rounding)
    except (DecimalException, KeyError, TypeError, ValueError, OverflowError):
        return None
    return converted, f"fx:{day}:{source}:{target}"


class FxReconciliationPolicy:
    """Inspect full public pages and reconcile duplicate, cutoff, allocation, and FX errors."""

    def __init__(self):
        self._manifest: dict = {}
        self._policy: dict = {}
        self._tables: dict[str, list[dict]] = {table: [] for table in _TABLES}
        self._table_index = 0
        self._cursor: str | None = None

    def act(self, observation: Observation, previous_action: Action | None) -> Action:
        if previous_action is None:
            self._manifest = observation["manifest"]
            self._policy = observation["policy"]
        elif "records" not in observation:
            return submit(self._manifest["episode_id"])
        else:
            self._tables[previous_action["table"]].extend(observation["records"])
            self._cursor = observation.get("cursor")
            if self._cursor is None:
                self._table_index += 1
        if self._table_index == len(_TABLES):
            return submit(self._manifest["episode_id"], self._findings())
        if observation["remaining_steps"] <= 1:
            return submit(self._manifest["episode_id"])
        return {
            "type": "list_records", "table": _TABLES[self._table_index],
            "cursor": self._cursor, "limit": 50,
        }

    def _findings(self) -> list[Finding]:
        base = self._manifest["base_currency"]
        exponents = self._manifest["currency_exponents"]
        rounding = self._policy["rounding_mode"]
        try:
            zone = ZoneInfo(self._policy["timezone"])
        except (KeyError, TypeError, ValueError, ZoneInfoNotFoundError):
            return []

        invoices = {row["invoice_id"]: row for row in self._tables["invoices"]}
        quotes: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        for row in self._tables["fx"]:
            quotes[row["date"], row["source_currency"], row["target_currency"]].append(row["rate"])

        accruals: dict[str, list[dict]] = defaultdict(list)
        settlements: dict[str, list[dict]] = defaultdict(list)
        for row in self._tables["ledger"]:
            payment_id = row["payment_id"]
            if payment_id is not None:
                settlements[payment_id].append(row)
            elif row["invoice_id"] is not None:
                accruals[row["invoice_id"]].append(row)

        findings: list[Finding] = []
        self._reconcile_invoices(findings, invoices, accruals, quotes, exponents, base, rounding, zone)
        self._reconcile_payments(findings, invoices, settlements, quotes, exponents, base, rounding, zone)
        return findings[:MAX_FINDINGS]

    def _reconcile_invoices(
        self,
        findings: list[Finding],
        invoices: dict[str, dict],
        accruals: dict[str, list[dict]],
        quotes: dict[tuple[str, str, str], list[str]],
        exponents: dict[str, int],
        base: str,
        rounding: str,
        zone: ZoneInfo,
    ) -> None:
        cutoff_enabled = self._policy.get("recognition_date_rule") == "document_date"
        period_start = _parse_aware(self._policy.get("period_start")) if cutoff_enabled else None
        period_end = _parse_aware(self._policy.get("period_end")) if cutoff_enabled else None
        start_utc = period_start.astimezone(timezone.utc) if period_start is not None else None
        end_utc = period_end.astimezone(timezone.utc) if period_end is not None else None

        for invoice_id in sorted(invoices):
            if len(findings) >= MAX_FINDINGS:
                return
            invoice = invoices[invoice_id]
            rows = accruals.get(invoice_id, [])
            if not rows or any(
                row["currency"] != base or row["entity_id"] != invoice["entity_id"]
                for row in rows
            ):
                continue

            expenses = [
                row for row in rows
                if row["account"] == "expense" and row["debit_minor"] > 0 and row["credit_minor"] == 0
            ]
            payables = [
                row for row in rows
                if row["account"] == "accounts_payable" and row["credit_minor"] > 0 and row["debit_minor"] == 0
            ]
            if len(expenses) + len(payables) != len(rows) or len(expenses) != len(payables):
                continue

            # 1) Duplicate invoice accrual check.
            if len(expenses) >= 2:
                converted = _fx_convert(
                    invoice["total_minor"], invoice["currency"], base, invoice["issued_at"],
                    zone, quotes, exponents, rounding,
                )
                if converted is not None:
                    expected, quote_key = converted
                    expense_amounts = sorted(row["debit_minor"] for row in expenses)
                    payable_amounts = sorted(row["credit_minor"] for row in payables)
                    if expense_amounts == payable_amounts and set(expense_amounts) == {expected}:
                        excess = sum(expense_amounts) - expected
                        row_ids = sorted(row["row_id"] for row in rows)
                        evidence = [invoice["doc_id"], *(["policy", quote_key] if quote_key else []), *row_ids]
                        if excess > 0 and len(evidence) <= MAX_EVIDENCE_IDS:
                            findings.append({
                                "type": FindingType.DUPLICATE_INVOICE.value,
                                "affected_ids": [invoice_id, *row_ids],
                                "claimed_amount_minor": excess,
                                "currency": base,
                                "evidence_ids": evidence,
                                "explanation": "Invoice obligation was accrued more than once in the ledger.",
                            })
                            if len(findings) >= MAX_FINDINGS:
                                return

            # 2) Period cutoff recognition check.
            if len(expenses) == 1 and start_utc is not None and end_utc is not None:
                expense, payable = expenses[0], payables[0]
                issued = _parse_aware(invoice["issued_at"])
                posted = _parse_aware(expense["timestamp"])
                if (
                    issued is not None and posted is not None
                    and expense["timestamp"] == payable["timestamp"]
                    and expense["debit_minor"] == payable["credit_minor"]
                ):
                    issued_utc = issued.astimezone(timezone.utc)
                    posted_utc = posted.astimezone(timezone.utc)
                    if start_utc <= issued_utc < end_utc and not (start_utc <= posted_utc < end_utc):
                        affected = sorted([expense["row_id"], payable["row_id"]])
                        findings.append({
                            "type": FindingType.PERIOD_CUTOFF.value,
                            "affected_ids": affected,
                            "claimed_amount_minor": expense["debit_minor"],
                            "currency": base,
                            "evidence_ids": [invoice["doc_id"], "policy", *affected],
                            "explanation": "In-period invoice accrual was posted outside the reporting period.",
                        })

    def _reconcile_payments(
        self,
        findings: list[Finding],
        invoices: dict[str, dict],
        settlements: dict[str, list[dict]],
        quotes: dict[tuple[str, str, str], list[str]],
        exponents: dict[str, int],
        base: str,
        rounding: str,
        zone: ZoneInfo,
    ) -> None:
        rule = self._policy.get("fx_date_rule")
        for payment in sorted(self._tables["payments"], key=lambda row: row["payment_id"]):
            if len(findings) >= MAX_FINDINGS:
                return
            refs = payment["invoice_refs"] or []
            paid = payment["amount_minor"]
            if payment.get("payment_kind") == "refund" or len(refs) != 1 or paid <= 0:
                continue
            invoice = invoices.get(refs[0])
            if invoice is None:
                continue
            rows = settlements.get(payment["payment_id"], [])
            if not rows or any(
                row["currency"] != base or row["entity_id"] != invoice["entity_id"]
                for row in rows
            ):
                continue
            if len({row["timestamp"] for row in rows}) != 1:
                continue

            bank_rows = [
                row for row in rows
                if row["account"] == "bank" and row["credit_minor"] > 0 and row["debit_minor"] == 0
            ]
            if len(bank_rows) != 1:
                continue
            bank = bank_rows[0]
            ap_rows = [
                row for row in rows
                if row["account"] == "accounts_payable"
                and row["invoice_id"] == invoice["invoice_id"]
                and row["debit_minor"] > 0 and row["credit_minor"] == 0
            ]
            suspense_rows = [
                row for row in rows
                if row["account"] == "suspense"
                and row["invoice_id"] is None
                and row["debit_minor"] > 0 and row["credit_minor"] == 0
            ]
            if sum(row["debit_minor"] for row in rows) != bank["credit_minor"]:
                continue

            # 3) Missing or partial payment allocation check (unallocated debit routed to suspense).
            if (
                len(suspense_rows) == 1
                and len(ap_rows) in (0, 1)
                and len(ap_rows) + len(suspense_rows) + 1 == len(rows)
            ):
                converted = _fx_convert(
                    paid, payment["currency"], base, payment["timestamp"],
                    zone, quotes, exponents, rounding,
                )
                if converted is not None:
                    expected, quote_key = converted
                    allocated = sum(row["debit_minor"] for row in ap_rows)
                    shortfall = expected - allocated
                    after_ids = sorted(row["row_id"] for row in (*ap_rows, *suspense_rows))
                    evidence = [
                        invoice["doc_id"], payment["doc_id"],
                        *(["policy", quote_key] if quote_key else []),
                        bank["row_id"], *after_ids,
                    ]
                    if shortfall > 0 and len(evidence) <= MAX_EVIDENCE_IDS:
                        findings.append({
                            "type": FindingType.PAYMENT_ALLOCATION.value,
                            "affected_ids": [invoice["invoice_id"], payment["payment_id"], *after_ids],
                            "claimed_amount_minor": shortfall,
                            "currency": base,
                            "evidence_ids": evidence,
                            "explanation": "Payment amount was only partially or not allocated to accounts payable.",
                        })
                        if len(findings) >= MAX_FINDINGS:
                            return

            # 4) Foreign-currency settlement conversion check (2-row clean or 3-row partial-allocation settlement).
            source = payment["currency"]
            if source == base or invoice["currency"] != source or invoice["total_minor"] != paid:
                continue
            fx_debit: dict | None = None
            if len(rows) == 2 and len(ap_rows) == 1 and ap_rows[0]["debit_minor"] == bank["credit_minor"]:
                fx_debit = ap_rows[0]
            elif (
                len(rows) == 3 and len(ap_rows) == 1 and len(suspense_rows) == 1
                and ap_rows[0]["debit_minor"] + suspense_rows[0]["debit_minor"] == bank["credit_minor"]
            ):
                fx_debit = suspense_rows[0]
            if fx_debit is None:
                continue

            timestamp = {
                "payment_date": payment["timestamp"],
                "invoice_date": invoice["issued_at"],
                "posting_date": bank["timestamp"],
            }.get(rule)
            if timestamp is None:
                continue
            converted = _fx_convert(paid, source, base, timestamp, zone, quotes, exponents, rounding)
            if converted is None:
                continue
            expected, quote_key = converted
            if expected == bank["credit_minor"]:
                continue
            affected = sorted([fx_debit["row_id"], bank["row_id"]])
            evidence = [payment["doc_id"], quote_key, "policy", *affected]
            if rule == "invoice_date":
                evidence.append(invoice["doc_id"])
            findings.append({
                "type": FindingType.FX_CONVERSION.value,
                "affected_ids": affected,
                "claimed_amount_minor": abs(expected - bank["credit_minor"]),
                "currency": base,
                "evidence_ids": evidence,
                "explanation": "Public quote differs from the recorded settlement.",
            })
