"""A public, conservative reconciliation of simple foreign-currency settlements."""

from collections import defaultdict
from datetime import datetime
from decimal import Context, Decimal, DecimalException, localcontext
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from genesis.domain.enums import FindingType
from genesis.domain.models import Finding

from .protocol import Action, Observation, submit


_TABLES = ("payments", "invoices", "fx", "ledger")


class FxReconciliationPolicy:
    """Inspect full public pages and abstain if the inputs cannot be reconciled."""

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
        invoices = {row["invoice_id"]: row for row in self._tables["invoices"]}
        quotes = {
            (row["date"], row["source_currency"], row["target_currency"]): row["rate"]
            for row in self._tables["fx"]
        }
        settlements = defaultdict(list)
        for row in self._tables["ledger"]:
            settlements[row["payment_id"]].append(row)
        try:
            zone = ZoneInfo(self._policy["timezone"])
        except (ValueError, ZoneInfoNotFoundError):
            return []
        findings: list[Finding] = []
        for payment in sorted(self._tables["payments"], key=lambda row: row["payment_id"]):
            source = payment["currency"]
            refs = payment["invoice_refs"] or []
            if source == base or payment.get("payment_kind") == "refund" or len(refs) != 1:
                continue
            invoice = invoices.get(refs[0])
            paid = payment["amount_minor"]
            if (
                invoice is None or invoice["currency"] != source
                or invoice["total_minor"] != paid or paid <= 0
            ):
                continue
            legs = _settlement_pair(settlements[payment["payment_id"]], invoice, base)
            if legs is None:
                continue
            debit, credit = legs
            rule = self._policy["fx_date_rule"]
            timestamp = {
                "payment_date": payment["timestamp"],
                "invoice_date": invoice["issued_at"],
                "posting_date": debit["timestamp"],
            }.get(rule)
            if timestamp is None:
                continue
            try:
                instant = datetime.fromisoformat(timestamp)
            except ValueError:
                continue
            if instant.tzinfo is None:
                continue
            day = instant.astimezone(zone).date().isoformat()
            rate = quotes.get((day, source, base))
            if rate is None or source not in exponents or base not in exponents:
                continue
            expected = _convert(paid, rate, exponents[source], exponents[base], self._policy["rounding_mode"])
            if expected is None or expected == debit["debit_minor"]:
                continue
            affected = [debit["row_id"], credit["row_id"]]
            evidence = [payment["doc_id"], f"fx:{day}:{source}:{base}", "policy", *affected]
            if rule == "invoice_date":
                evidence.append(invoice["doc_id"])
            findings.append({
                "type": FindingType.FX_CONVERSION.value, "affected_ids": affected,
                "claimed_amount_minor": abs(expected - debit["debit_minor"]), "currency": base,
                "evidence_ids": evidence, "explanation": "Public quote differs from the recorded settlement.",
            })
            if len(findings) == 30:
                break
        return findings


def _settlement_pair(rows: list[dict], invoice: dict, base: str) -> tuple[dict, dict] | None:
    if len(rows) != 2 or any(
        row["currency"] != base or row["invoice_id"] != invoice["invoice_id"]
        or row["entity_id"] != invoice["entity_id"]
        for row in rows
    ):
        return None
    debit, credit = sorted(rows, key=lambda row: row["credit_minor"])
    if (
        debit["account"] != "accounts_payable" or credit["account"] != "bank"
        or debit["debit_minor"] <= 0 or debit["debit_minor"] != credit["credit_minor"]
        or debit["credit_minor"] != 0 or credit["debit_minor"] != 0
        or debit["timestamp"] != credit["timestamp"]
    ):
        return None
    return debit, credit


def _convert(amount: int, rate: str, source_dp: int, target_dp: int, rounding: str) -> int | None:
    try:
        precision = max(28, len(str(amount)) + len(rate) + source_dp + target_dp)
        with localcontext(Context(prec=precision)):
            target = (Decimal(amount) * Decimal(rate)).scaleb(target_dp - source_dp)
            return int(target.quantize(Decimal(1), rounding=rounding))
    except (DecimalException, TypeError, ValueError, OverflowError):
        return None
