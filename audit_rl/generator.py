"""Causal financial audit episode generator, integrity verifier, and permutation tool."""

from __future__ import annotations

import copy
import random
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_EVEN
from typing import Any

from audit_rl.schemas import (
    CURRENCY_EXPONENTS,
    ERROR_TYPES,
    EpisodeConfig,
    GeneratedEpisode,
    INTERACTING_SCENARIOS,
    SCHEMA_VERSION,
    convert_minor,
    fx_record_key,
)

# Approximate reference exchange rates to USD (1 unit of currency = X USD)
_USD_BASE_RATES: dict[str, Decimal] = {
    "USD": Decimal("1.000000"),
    "EUR": Decimal("1.085000"),
    "GBP": Decimal("1.268000"),
    "CHF": Decimal("1.132000"),
    "KZT": Decimal("0.002080"),
    "JPY": Decimal("0.006550"),
}

_COMPANY_PREFIXES = (
    "Alatau",
    "Caspian",
    "Steppe",
    "Nordic",
    "Meridian",
    "Helios",
    "Vertex",
    "Cobalt",
    "Aurora",
    "Atlas",
    "Kestrel",
    "Boreal",
)

_COMPANY_SUFFIXES = (
    "Logistics LLP",
    "Cloud Systems Inc.",
    "Trading Group",
    "Industrial Solutions",
    "Payments Corp.",
    "Supply Chain Ltd.",
    "Analytics Holdings",
    "Energy Services",
)

_COUNTERPARTY_NAMES = (
    "Saryarka Telecom",
    "Eurasia Freight",
    "Balkhash Materials",
    "Danube Components",
    "Thames Advisory",
    "Kyoto Precision",
    "Zurich Instrumentation",
    "Vanguard Hosting",
    "Solstice Packaging",
    "Redwood Software",
    "Pinnacle Facilities",
    "Harbor Fulfillment",
    "Orion Security",
    "Summit Equipment",
    "Delta Data Services",
    "Cedar Consulting",
    "Ironclad Maintenance",
    "Pacific Microelectronics",
    "Alpine Logistics",
    "Crescent Office Spaces",
)

_LINE_ITEM_DESCRIPTIONS = (
    "Enterprise Platform License",
    "Managed Infrastructure Support",
    "Cross-Border Freight Handling",
    "Hardware Maintenance Retainer",
    "API Transaction Processing",
    "Security Compliance Audit",
    "Data Warehouse Storage",
    "Regional Distribution Fee",
    "Technical Implementation Hours",
    "Dedicated Compute Cluster",
)

_RECURRING_DESCRIPTIONS = (
    "Monthly Office Lease",
    "Monthly Cloud SaaS Subscription",
    "Monthly Managed Security Retainer",
)

_OPERATING_EXPENSES = (
    ("Payroll Expense", "Cash"),
    ("Bank Fee Expense", "Cash"),
    ("Depreciation Expense", "Accumulated Depreciation"),
    ("Utilities Expense", "Cash"),
    ("Software Expense", "Cash"),
    ("Insurance Expense", "Prepaid Insurance"),
)


def _parse_tz_offset(offset_str: str) -> timezone:
    s = offset_str.strip()
    if s in ("Z", "z"):
        return timezone.utc
    sign = 1 if s[0] == "+" else -1
    hh, mm = s[1:].split(":")
    return timezone(timedelta(hours=sign * int(hh), minutes=sign * int(mm)))


def _format_ts(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _random_ts_on_date(
    rng: random.Random,
    day: date,
    tz: timezone,
    min_hour: int = 8,
    max_hour: int = 18,
) -> datetime:
    hour = rng.randint(min_hour, max_hour)
    minute = rng.randint(0, 59)
    second = rng.randint(0, 59)
    return datetime(day.year, day.month, day.day, hour, minute, second, tzinfo=tz)


def _random_amount_minor(rng: random.Random, currency: str, low_usd: int = 250, high_usd: int = 8500) -> int:
    usd_major = rng.randint(low_usd, high_usd)
    rate_to_usd = _USD_BASE_RATES[currency]
    cur_major = Decimal(usd_major) / rate_to_usd
    exp = CURRENCY_EXPONENTS[currency]
    minor = int((cur_major * (Decimal(10) ** exp)).quantize(Decimal("1"), rounding=ROUND_HALF_EVEN))
    # Round to clean-ish integer while retaining minor-unit variation
    step = 100 if exp == 0 else 5
    return max(step * 10, (minor // step) * step + rng.randint(0, step - 1))


def _split_amount_across_lines(
    rng: random.Random,
    total_minor: int,
    num_lines: int,
    service_date_str: str,
) -> list[dict[str, Any]]:
    """Build line_items whose quantity * unit_price_minor == amount_minor and sum == total_minor."""
    lines: list[dict[str, Any]] = []
    remaining = total_minor
    chosen_descs = rng.sample(_LINE_ITEM_DESCRIPTIONS, k=min(num_lines, len(_LINE_ITEM_DESCRIPTIONS)))
    for i in range(num_lines):
        desc = chosen_descs[i % len(chosen_descs)]
        if i == num_lines - 1:
            qty = 1
            unit_price = remaining
            amt = remaining
        else:
            qty = rng.randint(1, 5)
            max_share = max(qty, remaining // (num_lines - i))
            min_share = max(qty, remaining // ((num_lines - i) * 3))
            if min_share > max_share:
                min_share = max_share
            unit_price = max(1, rng.randint(min_share, max_share) // qty)
            amt = qty * unit_price
            if amt >= remaining:
                qty = 1
                unit_price = max(1, remaining // 2)
                amt = unit_price
            remaining -= amt
        lines.append(
            {
                "description": desc,
                "quantity": qty,
                "unit_price_minor": unit_price,
                "amount_minor": amt,
                "service_date": service_date_str,
            }
        )
    return lines


def _allocate_base_across_lines(
    line_items: list[dict[str, Any]],
    total_base_minor: int,
) -> list[int]:
    """Split total_base_minor proportionally across line items so sum == total_base_minor."""
    total_orig = sum(int(li["amount_minor"]) for li in line_items)
    if total_orig <= 0 or len(line_items) == 1:
        return [total_base_minor]
    shares: list[int] = []
    allocated = 0
    for idx, li in enumerate(line_items):
        if idx == len(line_items) - 1:
            shares.append(total_base_minor - allocated)
        else:
            part = int(
                (Decimal(total_base_minor) * Decimal(int(li["amount_minor"])) / Decimal(total_orig)).quantize(
                    Decimal("1"), rounding=ROUND_HALF_EVEN
                )
            )
            shares.append(part)
            allocated += part
    return shares


def _generate_fx_rates(
    rng: random.Random,
    start_date: date,
    end_date: date,
    currencies: tuple[str, ...],
    base_currency: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Generate daily FX rates from each foreign currency to base_currency."""
    fx_rows: list[dict[str, Any]] = []
    fx_lookup: dict[str, str] = {}

    foreign_currencies = [c for c in currencies if c != base_currency]
    base_usd = _USD_BASE_RATES[base_currency]

    current_rates: dict[str, Decimal] = {}
    for cur in foreign_currencies:
        initial = (_USD_BASE_RATES[cur] / base_usd) * Decimal(str(round(rng.uniform(0.96, 1.04), 6)))
        current_rates[cur] = initial.quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN)

    num_days = (end_date - start_date).days + 1
    for offset in range(num_days):
        d = start_date + timedelta(days=offset)
        d_str = d.isoformat()
        for cur in foreign_currencies:
            if offset > 0:
                drift = Decimal(str(round(rng.uniform(-0.006, 0.006), 6)))
                new_rate = current_rates[cur] * (Decimal("1") + drift)
                new_rate = max(Decimal("0.000100"), new_rate)
                current_rates[cur] = new_rate.quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN)
            rate_str = f"{current_rates[cur]:.6f}"
            fx_rows.append(
                {
                    "date": d_str,
                    "source_currency": cur,
                    "target_currency": base_currency,
                    "rate": rate_str,
                }
            )
            fx_lookup[fx_record_key(d_str, cur, base_currency)] = rate_str

    return fx_rows, fx_lookup


def _resolve_invoice_fx_date(
    inv: dict[str, Any],
    ar_row: dict[str, Any] | None,
    linked_payments: list[dict[str, Any]],
    fx_date_policy: str,
) -> str:
    if fx_date_policy == "posting_date" and ar_row is not None:
        return str(ar_row["timestamp"])[:10]
    if fx_date_policy == "payment_date" and linked_payments:
        earliest = min(str(p["timestamp"])[:10] for p in linked_payments)
        return earliest
    return str(inv["issued_at"])[:10]


class _IdGenerator:
    """Deterministic unique ID generator per episode."""

    def __init__(self, rng: random.Random) -> None:
        self._entity_seq = rng.randint(100, 399)
        self._inv_seq = rng.randint(1000, 3999)
        self._pay_seq = rng.randint(5000, 7999)
        self._doc_seq = rng.randint(10000, 29999)
        self._row_seq = rng.randint(40000, 59999)
        self._issue_seq = 0

    def next_entity_id(self) -> str:
        self._entity_seq += 1
        return f"ENT-{self._entity_seq:04d}"

    def next_invoice_id(self) -> str:
        self._inv_seq += 1
        return f"INV-{self._inv_seq:05d}"

    def next_payment_id(self) -> str:
        self._pay_seq += 1
        return f"PAY-{self._pay_seq:05d}"

    def next_doc_id(self) -> str:
        self._doc_seq += 1
        return f"DOC-{self._doc_seq:06d}"

    def next_row_id(self) -> str:
        self._row_seq += 1
        return f"ROW-{self._row_seq:06d}"

    def next_issue_id(self) -> str:
        self._issue_seq += 1
        return f"ISS-{self._issue_seq:03d}"


def generate_episode(
    seed: int,
    config: EpisodeConfig | dict[str, Any] | None = None,
) -> GeneratedEpisode:
    """Generate a deterministic, causally structured financial audit episode."""
    cfg = EpisodeConfig.from_dict(config)
    rng = random.Random(seed)
    ids = _IdGenerator(rng)

    episode_id = f"ep-{rng.getrandbits(48):012x}"
    period_start_dt = datetime.fromisoformat(cfg.period_start)
    period_end_dt = datetime.fromisoformat(cfg.period_end)
    tz = period_start_dt.tzinfo or _parse_tz_offset(cfg.timezone_offset)
    if period_start_dt.tzinfo is None:
        period_start_dt = period_start_dt.replace(tzinfo=tz)
    if period_end_dt.tzinfo is None:
        period_end_dt = period_end_dt.replace(tzinfo=tz)

    period_start_date = period_start_dt.date()
    period_end_date = period_end_dt.date()
    total_period_days = max(10, (period_end_date - period_start_date).days)

    # Ensure at least one foreign currency is available if FX errors/scenarios are requested
    currencies = list(cfg.currencies)
    needs_fx = (
        cfg.error_counts.get("fx_conversion_error", 0) > 0
        or "fx_period_partial_payment" in cfg.interacting_scenarios
        or "duplicate_invoice_fx_mismatch" in cfg.interacting_scenarios
    )
    foreign_currencies = [c for c in currencies if c != cfg.base_currency]
    if needs_fx and not foreign_currencies:
        fallback = "EUR" if cfg.base_currency != "EUR" else "GBP"
        currencies.append(fallback)
        foreign_currencies.append(fallback)

    fx_start = period_start_date - timedelta(days=15)
    fx_end = period_end_date + timedelta(days=15)
    fx_rates, fx_lookup = _generate_fx_rates(
        rng, fx_start, fx_end, tuple(currencies), cfg.base_currency
    )

    company_name = f"{rng.choice(_COMPANY_PREFIXES)} {rng.choice(_COMPANY_SUFFIXES)}"
    num_entities = rng.randint(12, 18)
    entity_names = rng.sample(_COUNTERPARTY_NAMES, k=num_entities)
    entities = [{"entity_id": ids.next_entity_id(), "name": name} for name in entity_names]
    entity_ids = [e["entity_id"] for e in entities]

    ledger: list[dict[str, Any]] = []
    invoices: list[dict[str, Any]] = []
    payments: list[dict[str, Any]] = []
    # Track invoice ledger rows for fast reference
    invoice_ledger_rows: dict[str, list[dict[str, Any]]] = {}
    invoice_payments: dict[str, list[dict[str, Any]]] = {}
    seen_inv_signatures: set[tuple[str, str, int, str]] = set()
    distractor_records: dict[str, list[str]] = {
        "split_payments": [],
        "refunds": [],
        "recurring_invoices": [],
        "multicurrency_invoices": [],
        "boundary_timestamps": [],
    }

    def _get_rate(fx_date_str: str, cur: str) -> str:
        if cur == cfg.base_currency:
            return "1.000000"
        key = fx_record_key(fx_date_str, cur, cfg.base_currency)
        if key not in fx_lookup:
            # Clamp to nearest available date in fx table
            if fx_date_str < fx_start.isoformat():
                key = fx_record_key(fx_start.isoformat(), cur, cfg.base_currency)
            else:
                key = fx_record_key(fx_end.isoformat(), cur, cfg.base_currency)
        return fx_lookup[key]

    def _add_ledger_row(
        ts_str: str,
        account: str,
        debit_minor: int,
        credit_minor: int,
        entity_id: str | None = None,
        invoice_id: str | None = None,
        payment_id: str | None = None,
    ) -> dict[str, Any]:
        row = {
            "row_id": ids.next_row_id(),
            "timestamp": ts_str,
            "account": account,
            "debit_minor": int(debit_minor),
            "credit_minor": int(credit_minor),
            "currency": cfg.base_currency,
            "entity_id": entity_id,
            "invoice_id": invoice_id,
            "payment_id": payment_id,
        }
        ledger.append(row)
        if invoice_id is not None and payment_id is None:
            invoice_ledger_rows.setdefault(invoice_id, []).append(row)
        return row

    def _create_invoice_and_book(
        entity_id: str,
        currency: str,
        total_minor: int,
        service_day: date,
        issued_dt: datetime,
        due_dt: datetime,
        line_items: list[dict[str, Any]] | None = None,
        posting_dt: datetime | None = None,
        fx_rate_override: str | None = None,
        fx_date_override: str | None = None,
        allow_duplicate_sig: bool = False,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        s_str = service_day.isoformat()
        if not allow_duplicate_sig:
            while (entity_id, s_str, total_minor, currency) in seen_inv_signatures:
                total_minor += 1
            seen_inv_signatures.add((entity_id, s_str, total_minor, currency))

        if line_items is None:
            num_lines = rng.randint(2, 3)
            line_items = _split_amount_across_lines(rng, total_minor, num_lines, s_str)
        else:
            line_items = copy.deepcopy(line_items)
            total_minor = sum(int(li["amount_minor"]) for li in line_items)

        inv_id = ids.next_invoice_id()
        doc_id = ids.next_doc_id()
        inv = {
            "doc_id": doc_id,
            "invoice_id": inv_id,
            "entity_id": entity_id,
            "issued_at": _format_ts(issued_dt),
            "due_at": _format_ts(due_dt),
            "currency": currency,
            "total_minor": int(total_minor),
            "line_items": line_items,
        }
        invoices.append(inv)
        invoice_payments.setdefault(inv_id, [])

        post_dt = posting_dt if posting_dt is not None else issued_dt
        post_ts = _format_ts(post_dt)
        if fx_date_override is not None:
            fx_date_str = fx_date_override
        elif cfg.fx_date_policy == "posting_date":
            fx_date_str = post_ts[:10]
        else:
            fx_date_str = inv["issued_at"][:10]

        rate_str = fx_rate_override if fx_rate_override is not None else _get_rate(fx_date_str, currency)
        base_total = convert_minor(total_minor, currency, cfg.base_currency, rate_str, cfg.rounding_rule)
        line_bases = _allocate_base_across_lines(line_items, base_total)

        rows: list[dict[str, Any]] = []
        ar_row = _add_ledger_row(
            ts_str=post_ts,
            account="Accounts Receivable",
            debit_minor=base_total,
            credit_minor=0,
            entity_id=entity_id,
            invoice_id=inv_id,
            payment_id=None,
        )
        rows.append(ar_row)
        for idx, line_base in enumerate(line_bases):
            rev_account = "Service Revenue" if idx % 2 == 0 else "Product Revenue"
            rev_row = _add_ledger_row(
                ts_str=post_ts,
                account=rev_account,
                debit_minor=0,
                credit_minor=line_base,
                entity_id=entity_id,
                invoice_id=inv_id,
                payment_id=None,
            )
            rows.append(rev_row)
        return inv, rows

    def _create_payment_and_book(
        entity_id: str,
        currency: str,
        amount_minor: int,
        pay_dt: datetime,
        invoice_refs: list[dict[str, Any]] | None,
        is_refund: bool = False,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        pay_id = ids.next_payment_id()
        doc_id = ids.next_doc_id()
        pay_ts = _format_ts(pay_dt)
        pay = {
            "doc_id": doc_id,
            "payment_id": pay_id,
            "timestamp": pay_ts,
            "currency": currency,
            "amount_minor": int(amount_minor),
            "invoice_refs": copy.deepcopy(invoice_refs) if invoice_refs is not None else None,
        }
        payments.append(pay)

        rate_str = _get_rate(pay_ts[:10], currency)
        abs_base = convert_minor(abs(amount_minor), currency, cfg.base_currency, rate_str, cfg.rounding_rule)

        rows: list[dict[str, Any]] = []
        if is_refund:
            primary_inv = invoice_refs[0]["invoice_id"] if invoice_refs else None
            rows.append(
                _add_ledger_row(
                    ts_str=pay_ts,
                    account="Sales Returns & Allowances",
                    debit_minor=abs_base,
                    credit_minor=0,
                    entity_id=entity_id,
                    invoice_id=primary_inv,
                    payment_id=pay_id,
                )
            )
            rows.append(
                _add_ledger_row(
                    ts_str=pay_ts,
                    account="Cash",
                    debit_minor=0,
                    credit_minor=abs_base,
                    entity_id=entity_id,
                    invoice_id=primary_inv,
                    payment_id=pay_id,
                )
            )
            if invoice_refs:
                for ref in invoice_refs:
                    invoice_payments.setdefault(ref["invoice_id"], []).append(pay)
            return pay, rows

        if invoice_refs is None:
            rows.append(
                _add_ledger_row(
                    ts_str=pay_ts,
                    account="Cash",
                    debit_minor=abs_base,
                    credit_minor=0,
                    entity_id=entity_id,
                    invoice_id=None,
                    payment_id=pay_id,
                )
            )
            rows.append(
                _add_ledger_row(
                    ts_str=pay_ts,
                    account="Unapplied Cash",
                    debit_minor=0,
                    credit_minor=abs_base,
                    entity_id=entity_id,
                    invoice_id=None,
                    payment_id=pay_id,
                )
            )
            return pay, rows

        primary_inv = invoice_refs[0]["invoice_id"] if invoice_refs else None
        rows.append(
            _add_ledger_row(
                ts_str=pay_ts,
                account="Cash",
                debit_minor=abs_base,
                credit_minor=0,
                entity_id=entity_id,
                invoice_id=primary_inv,
                payment_id=pay_id,
            )
        )
        allocated_base_sum = 0
        allocated_orig_sum = sum(int(r["allocated_minor"]) for r in invoice_refs)
        for idx, ref in enumerate(invoice_refs):
            ref_inv_id = ref["invoice_id"]
            invoice_payments.setdefault(ref_inv_id, []).append(pay)
            if allocated_orig_sum == amount_minor and idx == len(invoice_refs) - 1:
                ref_base = abs_base - allocated_base_sum
            else:
                ref_base = convert_minor(
                    abs(int(ref["allocated_minor"])),
                    currency,
                    cfg.base_currency,
                    rate_str,
                    cfg.rounding_rule,
                )
                ref_base = min(ref_base, abs_base - allocated_base_sum)
            allocated_base_sum += ref_base
            rows.append(
                _add_ledger_row(
                    ts_str=pay_ts,
                    account="Accounts Receivable",
                    debit_minor=0,
                    credit_minor=ref_base,
                    entity_id=entity_id,
                    invoice_id=ref_inv_id,
                    payment_id=pay_id,
                )
            )

        remainder_base = abs_base - allocated_base_sum
        if remainder_base > 0:
            rows.append(
                _add_ledger_row(
                    ts_str=pay_ts,
                    account="Unapplied Cash",
                    debit_minor=0,
                    credit_minor=remainder_base,
                    entity_id=entity_id,
                    invoice_id=None,
                    payment_id=pay_id,
                )
            )
        return pay, rows

    # =========================================================================
    # STEP 1: Build clean hidden company history + realistic valid distractors
    # =========================================================================
    distractor_invoice_ids: set[str] = set()
    clean_candidates: list[dict[str, Any]] = []

    # 1a. Distractor: Legitimate recurring identical amounts (3 distinct months/dates)
    if cfg.include_distractors and cfg.num_invoices >= 15:
        rec_entity = entity_ids[0]
        rec_desc = rng.choice(_RECURRING_DESCRIPTIONS)
        rec_amt = _random_amount_minor(rng, cfg.base_currency, 1200, 3500)
        step_days = max(25, total_period_days // 3)
        for m_idx in range(3):
            day_offset = min(total_period_days - 3, 4 + m_idx * step_days)
            s_day = period_start_date + timedelta(days=day_offset)
            iss_dt = _random_ts_on_date(rng, s_day, tz)
            due_dt = iss_dt + timedelta(days=15)
            li = [
                {
                    "description": rec_desc,
                    "quantity": 1,
                    "unit_price_minor": rec_amt,
                    "amount_minor": rec_amt,
                    "service_date": s_day.isoformat(),
                }
            ]
            inv, _ = _create_invoice_and_book(
                entity_id=rec_entity,
                currency=cfg.base_currency,
                total_minor=rec_amt,
                service_day=s_day,
                issued_dt=iss_dt,
                due_dt=due_dt,
                line_items=li,
            )
            distractor_invoice_ids.add(inv["invoice_id"])
            distractor_records["recurring_invoices"].append(inv["invoice_id"])
            pay_day = min(period_end_date, s_day + timedelta(days=rng.randint(2, 6)))
            _create_payment_and_book(
                entity_id=rec_entity,
                currency=cfg.base_currency,
                amount_minor=rec_amt,
                pay_dt=_random_ts_on_date(rng, pay_day, tz),
                invoice_refs=[{"invoice_id": inv["invoice_id"], "allocated_minor": rec_amt}],
            )

    # 1b. Distractor: Split payment (1 invoice paid by 3 payments summing exactly to total_minor)
    if cfg.include_distractors and cfg.num_invoices >= 15:
        sp_entity = entity_ids[1]
        sp_day = period_start_date + timedelta(days=min(12, total_period_days // 4))
        sp_amt = _random_amount_minor(rng, cfg.base_currency, 2400, 6000)
        sp_iss = _random_ts_on_date(rng, sp_day, tz)
        sp_inv, _ = _create_invoice_and_book(
            entity_id=sp_entity,
            currency=cfg.base_currency,
            total_minor=sp_amt,
            service_day=sp_day,
            issued_dt=sp_iss,
            due_dt=sp_iss + timedelta(days=30),
        )
        distractor_invoice_ids.add(sp_inv["invoice_id"])
        p1 = sp_amt // 3
        p2 = sp_amt // 3
        p3 = sp_amt - p1 - p2
        for idx, part in enumerate((p1, p2, p3)):
            p_day = min(period_end_date, sp_day + timedelta(days=3 + idx * 5))
            pay, _ = _create_payment_and_book(
                entity_id=sp_entity,
                currency=cfg.base_currency,
                amount_minor=part,
                pay_dt=_random_ts_on_date(rng, p_day, tz),
                invoice_refs=[{"invoice_id": sp_inv["invoice_id"], "allocated_minor": part}],
            )
            distractor_records["split_payments"].append(pay["payment_id"])

    # 1c. Distractor: Refund / credit note (properly reversed in ledger & payments)
    if cfg.include_distractors and cfg.num_invoices >= 15:
        rf_entity = entity_ids[2]
        rf_day = period_start_date + timedelta(days=min(18, total_period_days // 3))
        rf_amt = _random_amount_minor(rng, cfg.base_currency, 1000, 4000)
        rf_iss = _random_ts_on_date(rng, rf_day, tz)
        rf_inv, _ = _create_invoice_and_book(
            entity_id=rf_entity,
            currency=cfg.base_currency,
            total_minor=rf_amt,
            service_day=rf_day,
            issued_dt=rf_iss,
            due_dt=rf_iss + timedelta(days=15),
        )
        distractor_invoice_ids.add(rf_inv["invoice_id"])
        pay_day = min(period_end_date, rf_day + timedelta(days=3))
        _create_payment_and_book(
            entity_id=rf_entity,
            currency=cfg.base_currency,
            amount_minor=rf_amt,
            pay_dt=_random_ts_on_date(rng, pay_day, tz),
            invoice_refs=[{"invoice_id": rf_inv["invoice_id"], "allocated_minor": rf_amt}],
        )
        refund_amt = rf_amt // 2
        refund_day = min(period_end_date, pay_day + timedelta(days=4))
        rf_pay, _ = _create_payment_and_book(
            entity_id=rf_entity,
            currency=cfg.base_currency,
            amount_minor=-refund_amt,
            pay_dt=_random_ts_on_date(rng, refund_day, tz),
            invoice_refs=[{"invoice_id": rf_inv["invoice_id"], "allocated_minor": -refund_amt}],
            is_refund=True,
        )
        distractor_records["refunds"].append(rf_pay["payment_id"])

    # 1d. Distractor: Valid boundary timestamp (23:30 local time on last day of period)
    if cfg.include_distractors and cfg.num_invoices >= 15:
        bt_entity = entity_ids[3]
        bt_amt = _random_amount_minor(rng, cfg.base_currency, 900, 3000)
        bt_iss = datetime(
            period_end_date.year,
            period_end_date.month,
            period_end_date.day,
            23,
            30,
            0,
            tzinfo=tz,
        )
        bt_post = datetime(
            period_end_date.year,
            period_end_date.month,
            period_end_date.day,
            23,
            35,
            0,
            tzinfo=tz,
        )
        bt_inv, _ = _create_invoice_and_book(
            entity_id=bt_entity,
            currency=cfg.base_currency,
            total_minor=bt_amt,
            service_day=period_end_date,
            issued_dt=bt_iss,
            due_dt=bt_iss + timedelta(days=30),
            posting_dt=bt_post,
        )
        distractor_invoice_ids.add(bt_inv["invoice_id"])
        distractor_records["boundary_timestamps"].append(bt_inv["invoice_id"])

    # 1e. Distractor: Clean multi-currency invoices converted at exact policy FX rate
    if cfg.include_distractors and foreign_currencies and cfg.num_invoices >= 15:
        for mc_idx in range(min(3, len(foreign_currencies))):
            mc_cur = foreign_currencies[mc_idx % len(foreign_currencies)]
            mc_entity = entity_ids[(4 + mc_idx) % len(entity_ids)]
            mc_day = period_start_date + timedelta(days=rng.randint(5, max(6, total_period_days - 8)))
            mc_amt = _random_amount_minor(rng, mc_cur, 800, 4500)
            mc_iss = _random_ts_on_date(rng, mc_day, tz)
            mc_inv, _ = _create_invoice_and_book(
                entity_id=mc_entity,
                currency=mc_cur,
                total_minor=mc_amt,
                service_day=mc_day,
                issued_dt=mc_iss,
                due_dt=mc_iss + timedelta(days=30),
            )
            distractor_invoice_ids.add(mc_inv["invoice_id"])
            distractor_records["multicurrency_invoices"].append(mc_inv["invoice_id"])
            if cfg.fx_date_policy != "payment_date":
                mc_pay_day = min(period_end_date, mc_day + timedelta(days=rng.randint(2, 7)))
                _create_payment_and_book(
                    entity_id=mc_entity,
                    currency=mc_cur,
                    amount_minor=mc_amt,
                    pay_dt=_random_ts_on_date(rng, mc_pay_day, tz),
                    invoice_refs=[{"invoice_id": mc_inv["invoice_id"], "allocated_minor": mc_amt}],
                )

    # 1f. Remaining regular clean invoices up to cfg.num_invoices
    remaining_invoices = max(0, cfg.num_invoices - len(invoices))
    # Cap total docs before error injection so we stay well below 200 even after errors
    expected_extra_docs = (
        cfg.error_counts.get("duplicate_invoice", 0)
        + cfg.error_counts.get("payment_allocation_mismatch", 0)
        + cfg.error_counts.get("period_cutoff_error", 0)
        + 3 * len(cfg.interacting_scenarios)
        + 4
    )
    max_clean_docs = max(cfg.num_invoices, 200 - expected_extra_docs)
    for idx in range(remaining_invoices):
        ent_id = rng.choice(entity_ids)
        # Ensure good mix of base and foreign currencies
        if foreign_currencies and (idx % 3 == 0):
            cur = foreign_currencies[idx % len(foreign_currencies)]
        else:
            cur = cfg.base_currency
        day_offset = rng.randint(2, max(3, total_period_days - 4))
        s_day = period_start_date + timedelta(days=day_offset)
        iss_dt = _random_ts_on_date(rng, s_day, tz)
        due_dt = iss_dt + timedelta(days=rng.choice((15, 30, 45)))
        amt = _random_amount_minor(rng, cur, 300, 7500)

        # If fx_date_policy == "payment_date" and foreign currency, we keep invoice unpaid
        # or pay on same date so invoice booking FX date matches payment date cleanly.
        inv, _ = _create_invoice_and_book(
            entity_id=ent_id,
            currency=cur,
            total_minor=amt,
            service_day=s_day,
            issued_dt=iss_dt,
            due_dt=due_dt,
        )
        clean_candidates.append(inv)

        projected_docs = len(invoices) + len(payments) + (remaining_invoices - idx - 1)
        if projected_docs < max_clean_docs and rng.random() < 0.72:
            if cfg.fx_date_policy == "payment_date" and cur != cfg.base_currency:
                p_day = s_day
            else:
                p_day = min(period_end_date, s_day + timedelta(days=rng.randint(1, 10)))
            _create_payment_and_book(
                entity_id=ent_id,
                currency=cur,
                amount_minor=amt,
                pay_dt=_random_ts_on_date(rng, p_day, tz),
                invoice_refs=[{"invoice_id": inv["invoice_id"], "allocated_minor": amt}],
            )

    # 1g. Balanced general operating ledger entries (ensure 200 <= ledger_rows <= 2000)
    min_target_rows = 240
    extra_pairs = max(25, (min_target_rows - len(ledger)) // 2 + 10)
    # Keep well under 2000 even for large num_invoices
    max_pairs = max(0, (1850 - len(ledger)) // 2)
    num_op_pairs = min(extra_pairs, max_pairs)
    for _ in range(num_op_pairs):
        exp_acct, credit_acct = rng.choice(_OPERATING_EXPENSES)
        op_day = period_start_date + timedelta(days=rng.randint(0, total_period_days))
        op_ts = _format_ts(_random_ts_on_date(rng, op_day, tz))
        op_amt = _random_amount_minor(rng, cfg.base_currency, 100, 2500)
        _add_ledger_row(op_ts, exp_acct, op_amt, 0, None, None, None)
        _add_ledger_row(op_ts, credit_acct, 0, op_amt, None, None, None)

    # =========================================================================
    # Pre-injection accounting identity checks
    # =========================================================================
    pre_total_debit = sum(int(r["debit_minor"]) for r in ledger)
    pre_total_credit = sum(int(r["credit_minor"]) for r in ledger)
    pre_payments_reconcile = all(
        p["invoice_refs"] is not None
        and sum(int(ref["allocated_minor"]) for ref in p["invoice_refs"]) == int(p["amount_minor"])
        for p in payments
    )
    pre_invoices_in_period = all(
        period_start_date.isoformat() <= str(li["service_date"]) <= period_end_date.isoformat()
        for inv in invoices
        for li in inv["line_items"]
    )
    pre_injection_checks = {
        "debits_equal_credits": pre_total_debit == pre_total_credit,
        "total_debit_minor": pre_total_debit,
        "total_credit_minor": pre_total_credit,
        "all_clean_payments_reconcile": pre_payments_reconcile,
        "all_clean_invoices_in_period": pre_invoices_in_period,
        "clean_invoice_count": len(invoices),
        "clean_payment_count": len(payments),
        "clean_ledger_row_count": len(ledger),
    }

    # =========================================================================
    # STEP 2: Causal error injection
    # =========================================================================
    truth_issues: list[dict[str, Any]] = []
    injected_events: list[dict[str, Any]] = []
    used_invoice_ids: set[str] = set(distractor_invoice_ids)

    def _pick_unused_invoice(foreign_only: bool = False, base_only: bool = False) -> dict[str, Any] | None:
        pool = [
            inv
            for inv in clean_candidates
            if inv["invoice_id"] not in used_invoice_ids
            and (not foreign_only or inv["currency"] != cfg.base_currency)
            and (not base_only or inv["currency"] == cfg.base_currency)
        ]
        if not pool:
            return None
        chosen = rng.choice(pool)
        used_invoice_ids.add(chosen["invoice_id"])
        return chosen

    def _create_fresh_foreign_invoice(
        service_day: date | None = None,
        issued_dt: datetime | None = None,
    ) -> dict[str, Any]:
        cur = rng.choice(foreign_currencies)
        ent_id = rng.choice(entity_ids)
        if service_day is None:
            service_day = period_start_date + timedelta(days=rng.randint(5, max(6, total_period_days - 5)))
        if issued_dt is None:
            issued_dt = _random_ts_on_date(rng, service_day, tz)
        amt = _random_amount_minor(rng, cur, 1200, 6500)
        inv, _ = _create_invoice_and_book(
            entity_id=ent_id,
            currency=cur,
            total_minor=amt,
            service_day=service_day,
            issued_dt=issued_dt,
            due_dt=issued_dt + timedelta(days=30),
        )
        used_invoice_ids.add(inv["invoice_id"])
        return inv

    def _corrupt_invoice_fx(inv: dict[str, Any]) -> dict[str, Any]:
        """Rewrite an invoice's ledger AR/Revenue rows using an intentionally wrong FX rate."""
        inv_rows = invoice_ledger_rows[inv["invoice_id"]]
        ar_row = next(r for r in inv_rows if r["account"] == "Accounts Receivable" and r["debit_minor"] > 0)
        rev_rows = [r for r in inv_rows if r["credit_minor"] > 0]

        linked_pays = invoice_payments.get(inv["invoice_id"], [])
        fx_date_str = _resolve_invoice_fx_date(inv, ar_row, linked_pays, cfg.fx_date_policy)
        expected_rate_str = _get_rate(fx_date_str, inv["currency"])
        expected_base = convert_minor(
            inv["total_minor"], inv["currency"], cfg.base_currency, expected_rate_str, cfg.rounding_rule
        )

        # Shift rate by +/- 8% to 14% so discrepancy is unambiguously > tolerance_minor=1
        direction = rng.choice((-1, 1))
        shift_factor = Decimal("1") + Decimal(direction) * Decimal(str(round(rng.uniform(0.08, 0.14), 4)))
        wrong_rate = (Decimal(expected_rate_str) * shift_factor).quantize(
            Decimal("0.000001"), rounding=ROUND_HALF_EVEN
        )
        wrong_rate_str = f"{wrong_rate:.6f}"
        posted_base = convert_minor(
            inv["total_minor"], inv["currency"], cfg.base_currency, wrong_rate_str, cfg.rounding_rule
        )
        if abs(posted_base - expected_base) < 50:
            posted_base = expected_base + direction * max(100, int(expected_base * 0.10))

        # Keep invoice journal entry balanced (Debit AR == Sum Credit Revenue)
        ar_row["debit_minor"] = posted_base
        new_line_bases = _allocate_base_across_lines(inv["line_items"], posted_base)
        for rev_row, new_val in zip(rev_rows, new_line_bases):
            rev_row["credit_minor"] = new_val
        if len(rev_rows) != len(new_line_bases) and rev_rows:
            rev_rows[0]["credit_minor"] = posted_base
            for r in rev_rows[1:]:
                r["credit_minor"] = 0

        diff_minor = abs(posted_base - expected_base)
        fx_key = fx_record_key(fx_date_str, inv["currency"], cfg.base_currency)
        issue = {
            "issue_id": ids.next_issue_id(),
            "type": "fx_conversion_error",
            "affected_ids": [inv["invoice_id"]],
            "expected_amount_minor": diff_minor,
            "currency": cfg.base_currency,
            "tolerance_minor": 1,
            "acceptable_evidence_sets": [
                [inv["doc_id"], ar_row["row_id"], fx_key],
                [inv["invoice_id"], ar_row["row_id"], fx_key],
            ],
            "description": (
                f"Invoice {inv['invoice_id']} ({inv['total_minor']} {inv['currency']}) posted as "
                f"{posted_base} {cfg.base_currency} instead of expected {expected_base} {cfg.base_currency} "
                f"using FX {fx_key} ({expected_rate_str})."
            ),
        }
        truth_issues.append(issue)
        injected_events.append(
            {
                "kind": "fx_conversion_error",
                "issue_id": issue["issue_id"],
                "invoice_id": inv["invoice_id"],
                "expected_rate": expected_rate_str,
                "wrong_rate": wrong_rate_str,
                "expected_base_minor": expected_base,
                "posted_base_minor": posted_base,
            }
        )
        return issue

    # 2a. Standalone duplicate_invoice errors
    for _ in range(cfg.error_counts.get("duplicate_invoice", 0)):
        orig = _pick_unused_invoice(base_only=True) or _pick_unused_invoice()
        if orig is None:
            s_day = period_start_date + timedelta(days=rng.randint(5, max(6, total_period_days - 5)))
            iss_dt = _random_ts_on_date(rng, s_day, tz)
            orig, _ = _create_invoice_and_book(
                entity_id=rng.choice(entity_ids),
                currency=cfg.base_currency,
                total_minor=_random_amount_minor(rng, cfg.base_currency, 1000, 5000),
                service_day=s_day,
                issued_dt=iss_dt,
                due_dt=iss_dt + timedelta(days=30),
            )
            used_invoice_ids.add(orig["invoice_id"])

        orig_iss = datetime.fromisoformat(orig["issued_at"])
        shift_days = rng.randint(0, 2)
        dup_iss = min(period_end_dt - timedelta(hours=1), orig_iss + timedelta(days=shift_days, hours=rng.randint(1, 5)))
        dup_due = dup_iss + timedelta(days=30)
        s_day = date.fromisoformat(orig["line_items"][0]["service_date"])

        dup_inv, dup_rows = _create_invoice_and_book(
            entity_id=orig["entity_id"],
            currency=orig["currency"],
            total_minor=orig["total_minor"],
            service_day=s_day,
            issued_dt=dup_iss,
            due_dt=dup_due,
            line_items=orig["line_items"],
            allow_duplicate_sig=True,
        )
        used_invoice_ids.add(dup_inv["invoice_id"])
        orig_ar = next(
            r for r in invoice_ledger_rows[orig["invoice_id"]]
            if r["account"] == "Accounts Receivable" and r["debit_minor"] > 0
        )
        dup_ar = next(r for r in dup_rows if r["account"] == "Accounts Receivable" and r["debit_minor"] > 0)

        issue = {
            "issue_id": ids.next_issue_id(),
            "type": "duplicate_invoice",
            "affected_ids": sorted([orig["invoice_id"], dup_inv["invoice_id"]]),
            "expected_amount_minor": int(dup_inv["total_minor"]),
            "currency": dup_inv["currency"],
            "tolerance_minor": 0,
            "acceptable_evidence_sets": [
                [orig["doc_id"], dup_inv["doc_id"], dup_ar["row_id"]],
                [orig["doc_id"], dup_inv["doc_id"], orig_ar["row_id"]],
                [orig["invoice_id"], dup_inv["invoice_id"], dup_ar["row_id"]],
                [orig["invoice_id"], dup_inv["invoice_id"], orig_ar["row_id"]],
            ],
            "description": (
                f"Duplicate invoice {dup_inv['invoice_id']} repeats {orig['invoice_id']} for entity "
                f"{orig['entity_id']} on service_date {s_day.isoformat()} ({dup_inv['total_minor']} {dup_inv['currency']})."
            ),
        }
        truth_issues.append(issue)
        injected_events.append(
            {
                "kind": "duplicate_invoice",
                "issue_id": issue["issue_id"],
                "orig_invoice_id": orig["invoice_id"],
                "dup_invoice_id": dup_inv["invoice_id"],
            }
        )

    # 2b. Standalone payment_allocation_mismatch errors
    for idx in range(cfg.error_counts.get("payment_allocation_mismatch", 0)):
        if idx % 2 == 0:
            # Variant A: Completely unallocated payment (invoice_refs is None)
            ent_id = rng.choice(entity_ids)
            pay_day = period_start_date + timedelta(days=rng.randint(3, max(4, total_period_days - 3)))
            pay_amt = _random_amount_minor(rng, cfg.base_currency, 600, 4000)
            pay, pay_rows = _create_payment_and_book(
                entity_id=ent_id,
                currency=cfg.base_currency,
                amount_minor=pay_amt,
                pay_dt=_random_ts_on_date(rng, pay_day, tz),
                invoice_refs=None,
            )
            cash_row = next(r for r in pay_rows if r["account"] == "Cash")
            unapp_row = next(r for r in pay_rows if r["account"] == "Unapplied Cash")
            issue = {
                "issue_id": ids.next_issue_id(),
                "type": "payment_allocation_mismatch",
                "affected_ids": [pay["payment_id"]],
                "expected_amount_minor": int(pay_amt),
                "currency": pay["currency"],
                "tolerance_minor": 0,
                "acceptable_evidence_sets": [
                    [pay["doc_id"], cash_row["row_id"]],
                    [pay["doc_id"], unapp_row["row_id"]],
                    [pay["payment_id"], cash_row["row_id"]],
                ],
                "description": (
                    f"Payment {pay['payment_id']} ({pay_amt} {pay['currency']}) has null invoice_refs "
                    f"and remains unallocated."
                ),
            }
            truth_issues.append(issue)
            injected_events.append(
                {
                    "kind": "payment_allocation_mismatch",
                    "variant": "unallocated",
                    "issue_id": issue["issue_id"],
                    "payment_id": pay["payment_id"],
                }
            )
        else:
            # Variant B: Partial/mismatched allocation against an invoice
            inv = _pick_unused_invoice()
            if inv is None:
                s_day = period_start_date + timedelta(days=rng.randint(5, max(6, total_period_days - 5)))
                iss_dt = _random_ts_on_date(rng, s_day, tz)
                inv, _ = _create_invoice_and_book(
                    entity_id=rng.choice(entity_ids),
                    currency=cfg.base_currency,
                    total_minor=_random_amount_minor(rng, cfg.base_currency, 1000, 5000),
                    service_day=s_day,
                    issued_dt=iss_dt,
                    due_dt=iss_dt + timedelta(days=30),
                )
                used_invoice_ids.add(inv["invoice_id"])
            inv_day = date.fromisoformat(inv["issued_at"][:10])
            pay_day = min(period_end_date, inv_day + timedelta(days=rng.randint(1, 7)))
            pay_amt = int(inv["total_minor"])
            alloc_amt = max(1, int(pay_amt * rng.uniform(0.55, 0.80)))
            if alloc_amt >= pay_amt:
                alloc_amt = pay_amt - 10
            diff_amt = pay_amt - alloc_amt
            pay, pay_rows = _create_payment_and_book(
                entity_id=inv["entity_id"],
                currency=inv["currency"],
                amount_minor=pay_amt,
                pay_dt=_random_ts_on_date(rng, pay_day, tz),
                invoice_refs=[{"invoice_id": inv["invoice_id"], "allocated_minor": alloc_amt}],
            )
            cash_row = next(r for r in pay_rows if r["account"] == "Cash")
            ar_credit_row = next(r for r in pay_rows if r["account"] == "Accounts Receivable")
            issue = {
                "issue_id": ids.next_issue_id(),
                "type": "payment_allocation_mismatch",
                "affected_ids": sorted([pay["payment_id"], inv["invoice_id"]]),
                "expected_amount_minor": int(diff_amt),
                "currency": pay["currency"],
                "tolerance_minor": 0,
                "acceptable_evidence_sets": [
                    [pay["doc_id"], inv["doc_id"], cash_row["row_id"]],
                    [pay["doc_id"], inv["doc_id"], ar_credit_row["row_id"]],
                    [pay["payment_id"], inv["invoice_id"], cash_row["row_id"]],
                ],
                "description": (
                    f"Payment {pay['payment_id']} ({pay_amt} {pay['currency']}) allocates only "
                    f"{alloc_amt} to invoice {inv['invoice_id']}, leaving {diff_amt} unallocated."
                ),
            }
            truth_issues.append(issue)
            injected_events.append(
                {
                    "kind": "payment_allocation_mismatch",
                    "variant": "partial_allocation",
                    "issue_id": issue["issue_id"],
                    "payment_id": pay["payment_id"],
                    "invoice_id": inv["invoice_id"],
                }
            )

    # 2c. Standalone fx_conversion_error errors
    for _ in range(cfg.error_counts.get("fx_conversion_error", 0)):
        inv = _pick_unused_invoice(foreign_only=True)
        if inv is None:
            inv = _create_fresh_foreign_invoice()
        _corrupt_invoice_fx(inv)

    # 2d. Standalone period_cutoff_error errors
    for idx in range(cfg.error_counts.get("period_cutoff_error", 0)):
        ent_id = rng.choice(entity_ids)
        amt = _random_amount_minor(rng, cfg.base_currency, 1000, 5500)
        if idx % 2 == 0:
            # Service rendered in NEXT period, but booked in CURRENT period
            out_service_day = period_end_date + timedelta(days=rng.randint(2, 6))
            iss_dt = _random_ts_on_date(rng, period_end_date, tz, min_hour=14, max_hour=21)
        else:
            # Service rendered in PRIOR period, but booked in CURRENT period
            out_service_day = period_start_date - timedelta(days=rng.randint(2, 6))
            iss_dt = _random_ts_on_date(rng, period_start_date, tz, min_hour=9, max_hour=16)

        inv, inv_rows = _create_invoice_and_book(
            entity_id=ent_id,
            currency=cfg.base_currency,
            total_minor=amt,
            service_day=out_service_day,
            issued_dt=iss_dt,
            due_dt=iss_dt + timedelta(days=30),
            posting_dt=iss_dt,
        )
        used_invoice_ids.add(inv["invoice_id"])
        ar_row = next(r for r in inv_rows if r["account"] == "Accounts Receivable")
        rev_row = next(r for r in inv_rows if r["credit_minor"] > 0)
        issue = {
            "issue_id": ids.next_issue_id(),
            "type": "period_cutoff_error",
            "affected_ids": [inv["invoice_id"]],
            "expected_amount_minor": int(inv["total_minor"]),
            "currency": inv["currency"],
            "tolerance_minor": 0,
            "acceptable_evidence_sets": [
                [inv["doc_id"], ar_row["row_id"]],
                [inv["doc_id"], rev_row["row_id"]],
                [inv["invoice_id"], ar_row["row_id"]],
            ],
            "description": (
                f"Invoice {inv['invoice_id']} has service_date {out_service_day.isoformat()} outside "
                f"period [{period_start_date.isoformat()}, {period_end_date.isoformat()}] but was "
                f"posted in ledger at {ar_row['timestamp']}."
            ),
        }
        truth_issues.append(issue)
        injected_events.append(
            {
                "kind": "period_cutoff_error",
                "issue_id": issue["issue_id"],
                "invoice_id": inv["invoice_id"],
                "service_date": out_service_day.isoformat(),
                "posted_at": ar_row["timestamp"],
            }
        )

    # 2e. Interacting scenarios
    interacting_groups: list[dict[str, Any]] = []
    for scenario in cfg.interacting_scenarios:
        if scenario == "fx_period_partial_payment":
            # Foreign-currency invoice near period boundary with wrong FX rate AND
            # a partial/mismatched payment crossing the period boundary.
            near_end_day = period_end_date - timedelta(days=1)
            iss_dt = _random_ts_on_date(rng, near_end_day, tz, min_hour=15, max_hour=20)
            inv = _create_fresh_foreign_invoice(service_day=near_end_day, issued_dt=iss_dt)

            # Create the partial/mismatched payment crossing the boundary first or after
            pay_dt = datetime(
                (period_end_date + timedelta(days=1)).year,
                (period_end_date + timedelta(days=1)).month,
                (period_end_date + timedelta(days=1)).day,
                10,
                15,
                0,
                tzinfo=tz,
            )
            pay_amt = int(inv["total_minor"])
            alloc_amt = max(1, int(pay_amt * rng.uniform(0.55, 0.75)))
            diff_amt = pay_amt - alloc_amt
            pay, pay_rows = _create_payment_and_book(
                entity_id=inv["entity_id"],
                currency=inv["currency"],
                amount_minor=pay_amt,
                pay_dt=pay_dt,
                invoice_refs=[{"invoice_id": inv["invoice_id"], "allocated_minor": alloc_amt}],
            )
            cash_row = next(r for r in pay_rows if r["account"] == "Cash")
            ar_credit_row = next(r for r in pay_rows if r["account"] == "Accounts Receivable")

            # Corrupt the invoice's FX conversion (now that linked payment exists for payment_date policy too)
            fx_issue = _corrupt_invoice_fx(inv)

            pay_issue = {
                "issue_id": ids.next_issue_id(),
                "type": "payment_allocation_mismatch",
                "affected_ids": sorted([pay["payment_id"], inv["invoice_id"]]),
                "expected_amount_minor": int(diff_amt),
                "currency": pay["currency"],
                "tolerance_minor": 0,
                "acceptable_evidence_sets": [
                    [pay["doc_id"], inv["doc_id"], cash_row["row_id"]],
                    [pay["doc_id"], inv["doc_id"], ar_credit_row["row_id"]],
                    [pay["payment_id"], inv["invoice_id"], cash_row["row_id"]],
                ],
                "description": (
                    f"Cross-boundary payment {pay['payment_id']} ({pay_amt} {pay['currency']}) "
                    f"allocates only {alloc_amt} to foreign-currency invoice {inv['invoice_id']} "
                    f"(unallocated diff {diff_amt})."
                ),
            }
            truth_issues.append(pay_issue)
            interacting_groups.append(
                {
                    "scenario": scenario,
                    "shared_invoice_id": inv["invoice_id"],
                    "payment_id": pay["payment_id"],
                    "issue_ids": [fx_issue["issue_id"], pay_issue["issue_id"]],
                }
            )

        elif scenario == "duplicate_invoice_fx_mismatch":
            # Duplicate foreign-currency invoice where the duplicate is posted with a WRONG FX rate,
            # so naive base-currency ledger dedup misses the duplicate!
            orig = _pick_unused_invoice(foreign_only=True)
            if orig is None:
                orig = _create_fresh_foreign_invoice()
            orig_iss = datetime.fromisoformat(orig["issued_at"])
            dup_iss = min(period_end_dt - timedelta(hours=1), orig_iss + timedelta(days=1, hours=2))
            s_day = date.fromisoformat(orig["line_items"][0]["service_date"])

            dup_inv, dup_rows = _create_invoice_and_book(
                entity_id=orig["entity_id"],
                currency=orig["currency"],
                total_minor=orig["total_minor"],
                service_day=s_day,
                issued_dt=dup_iss,
                due_dt=dup_iss + timedelta(days=30),
                line_items=orig["line_items"],
                allow_duplicate_sig=True,
            )
            used_invoice_ids.add(dup_inv["invoice_id"])
            orig_ar = next(
                r for r in invoice_ledger_rows[orig["invoice_id"]]
                if r["account"] == "Accounts Receivable" and r["debit_minor"] > 0
            )
            dup_ar = next(r for r in dup_rows if r["account"] == "Accounts Receivable" and r["debit_minor"] > 0)

            dup_issue = {
                "issue_id": ids.next_issue_id(),
                "type": "duplicate_invoice",
                "affected_ids": sorted([orig["invoice_id"], dup_inv["invoice_id"]]),
                "expected_amount_minor": int(dup_inv["total_minor"]),
                "currency": dup_inv["currency"],
                "tolerance_minor": 0,
                "acceptable_evidence_sets": [
                    [orig["doc_id"], dup_inv["doc_id"], dup_ar["row_id"]],
                    [orig["doc_id"], dup_inv["doc_id"], orig_ar["row_id"]],
                    [orig["invoice_id"], dup_inv["invoice_id"], dup_ar["row_id"]],
                ],
                "description": (
                    f"Duplicate foreign-currency invoice {dup_inv['invoice_id']} repeats {orig['invoice_id']} "
                    f"for entity {orig['entity_id']} on service_date {s_day.isoformat()} "
                    f"({dup_inv['total_minor']} {dup_inv['currency']})."
                ),
            }
            truth_issues.append(dup_issue)
            fx_issue = _corrupt_invoice_fx(dup_inv)
            interacting_groups.append(
                {
                    "scenario": scenario,
                    "orig_invoice_id": orig["invoice_id"],
                    "dup_invoice_id": dup_inv["invoice_id"],
                    "issue_ids": [dup_issue["issue_id"], fx_issue["issue_id"]],
                }
            )

    # Sort open tables deterministically by timestamp / date / id
    ledger.sort(key=lambda r: (r["timestamp"], r["row_id"]))
    invoices.sort(key=lambda r: (r["issued_at"], r["invoice_id"]))
    payments.sort(key=lambda r: (r["timestamp"], r["payment_id"]))
    fx_rates.sort(key=lambda r: (r["date"], r["source_currency"], r["target_currency"]))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "company_name": company_name,
        "base_currency": cfg.base_currency,
        "currencies": list(currencies),
        "timezone_offset": cfg.timezone_offset,
        "period_start": _format_ts(period_start_dt),
        "period_end": _format_ts(period_end_dt),
        "record_counts": {
            "ledger": len(ledger),
            "invoices": len(invoices),
            "payments": len(payments),
            "fx_rates": len(fx_rates),
        },
    }

    policy = {
        "schema_version": SCHEMA_VERSION,
        "base_currency": cfg.base_currency,
        "fx_date_policy": cfg.fx_date_policy,
        "rounding_rule": cfg.rounding_rule,
        "cutoff_basis": "service_date",
        "currency_exponents": {c: CURRENCY_EXPONENTS[c] for c in currencies},
        "error_types": list(ERROR_TYPES),
    }

    truth = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "issues": truth_issues,
    }

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "seed": seed,
        "config": cfg.to_dict(),
        "pre_injection_checks": pre_injection_checks,
        "distractors": distractor_records,
        "injected_events": injected_events,
        "interacting_groups": interacting_groups,
    }

    return GeneratedEpisode(
        manifest=manifest,
        policy=policy,
        ledger=ledger,
        invoices=invoices,
        payments=payments,
        fx_rates=fx_rates,
        truth=truth,
        provenance=provenance,
    )


def verify_episode_integrity(episode: GeneratedEpisode) -> dict[str, Any]:
    """Independently verify accounting invariants, bounds, evidence IDs, and every truth issue."""
    errors: list[str] = []
    manifest = episode.manifest
    policy = episode.policy
    ledger = episode.ledger
    invoices = episode.invoices
    payments = episode.payments
    fx_rates = episode.fx_rates
    truth = episode.truth
    provenance = episode.provenance

    # 1. Pre-injection balance check
    pre = provenance.get("pre_injection_checks", {})
    pre_balanced = bool(pre.get("debits_equal_credits")) and (
        pre.get("total_debit_minor") == pre.get("total_credit_minor")
    )
    if not pre_balanced:
        errors.append("pre-injection accounting balance check failed")

    # 2. Post-injection ledger balance check
    post_debit = sum(int(r["debit_minor"]) for r in ledger)
    post_credit = sum(int(r["credit_minor"]) for r in ledger)
    post_balanced = post_debit == post_credit
    if not post_balanced:
        errors.append(f"post-injection ledger unbalanced: debit={post_debit}, credit={post_credit}")

    # 3. Row and document count bounds
    row_count_ok = 200 <= len(ledger) <= 2000
    doc_count = len(invoices) + len(payments)
    doc_count_ok = 20 <= doc_count <= 200
    if not row_count_ok:
        errors.append(f"ledger row count {len(ledger)} out of bounds [200, 2000]")
    if not doc_count_ok:
        errors.append(f"document count {doc_count} out of bounds [20, 200]")

    # Build open-record indices
    inv_by_id = {inv["invoice_id"]: inv for inv in invoices}
    pay_by_id = {pay["payment_id"]: pay for pay in payments}
    row_by_id = {row["row_id"]: row for row in ledger}
    fx_by_key = {
        fx_record_key(r["date"], r["source_currency"], r["target_currency"]): r["rate"]
        for r in fx_rates
    }
    valid_open_ids: set[str] = (
        set(inv_by_id.keys())
        | set(pay_by_id.keys())
        | set(row_by_id.keys())
        | {inv["doc_id"] for inv in invoices}
        | {pay["doc_id"] for pay in payments}
        | set(fx_by_key.keys())
    )

    ar_row_by_inv: dict[str, dict[str, Any]] = {}
    for row in ledger:
        if (
            row.get("invoice_id")
            and row.get("payment_id") is None
            and row.get("account") == "Accounts Receivable"
            and int(row.get("debit_minor", 0)) > 0
        ):
            ar_row_by_inv[row["invoice_id"]] = row

    pays_by_inv: dict[str, list[dict[str, Any]]] = {}
    for pay in payments:
        if pay.get("invoice_refs"):
            for ref in pay["invoice_refs"]:
                pays_by_inv.setdefault(ref["invoice_id"], []).append(pay)

    # 4. Check all truth issues: evidence IDs exist + independent recalculation from open records
    all_evidence_ok = True
    all_issues_verified = True
    base_currency = policy["base_currency"]
    fx_date_policy = policy["fx_date_policy"]
    rounding_rule = policy["rounding_rule"]
    period_start_date = manifest["period_start"][:10]
    period_end_date = manifest["period_end"][:10]
    period_start_dt = datetime.fromisoformat(manifest["period_start"])
    period_end_dt = datetime.fromisoformat(manifest["period_end"])

    issues = truth.get("issues", [])
    for issue in issues:
        iid = issue.get("issue_id", "?")
        itype = issue.get("type")
        ev_sets = issue.get("acceptable_evidence_sets", [])
        if not ev_sets:
            all_evidence_ok = False
            errors.append(f"{iid}: empty acceptable_evidence_sets")
        for ev_set in ev_sets:
            for ev_id in ev_set:
                if ev_id not in valid_open_ids:
                    all_evidence_ok = False
                    errors.append(f"{iid}: evidence_id {ev_id!r} not found in open records")

        # Independent recalculation by error type
        if itype == "duplicate_invoice":
            affected = issue.get("affected_ids", [])
            if len(affected) != 2 or any(a not in inv_by_id for a in affected):
                all_issues_verified = False
                errors.append(f"{iid}: duplicate_invoice affected_ids invalid: {affected}")
            else:
                i1, i2 = inv_by_id[affected[0]], inv_by_id[affected[1]]
                same_core = (
                    i1["entity_id"] == i2["entity_id"]
                    and i1["currency"] == i2["currency"]
                    and int(i1["total_minor"]) == int(i2["total_minor"])
                    and i1["line_items"][0]["service_date"] == i2["line_items"][0]["service_date"]
                )
                if not same_core or int(issue["expected_amount_minor"]) != int(i1["total_minor"]):
                    all_issues_verified = False
                    errors.append(f"{iid}: duplicate_invoice independent verification failed")

        elif itype == "payment_allocation_mismatch":
            affected = issue.get("affected_ids", [])
            pay_ids = [a for a in affected if a in pay_by_id]
            if len(pay_ids) != 1:
                all_issues_verified = False
                errors.append(f"{iid}: payment_allocation_mismatch missing payment_id in {affected}")
            else:
                pay = pay_by_id[pay_ids[0]]
                refs = pay.get("invoice_refs")
                if refs is None:
                    recalc_diff = abs(int(pay["amount_minor"]))
                else:
                    recalc_diff = abs(
                        int(pay["amount_minor"]) - sum(int(r["allocated_minor"]) for r in refs)
                    )
                if recalc_diff <= 0 or recalc_diff != int(issue["expected_amount_minor"]):
                    all_issues_verified = False
                    errors.append(
                        f"{iid}: payment_allocation_mismatch recalc={recalc_diff} != "
                        f"expected={issue['expected_amount_minor']}"
                    )

        elif itype == "fx_conversion_error":
            affected = issue.get("affected_ids", [])
            if len(affected) != 1 or affected[0] not in inv_by_id:
                all_issues_verified = False
                errors.append(f"{iid}: fx_conversion_error invalid affected_ids {affected}")
            else:
                inv = inv_by_id[affected[0]]
                ar_row = ar_row_by_inv.get(inv["invoice_id"])
                if ar_row is None or inv["currency"] == base_currency:
                    all_issues_verified = False
                    errors.append(f"{iid}: fx_conversion_error missing AR row or base currency")
                else:
                    fx_date = _resolve_invoice_fx_date(
                        inv, ar_row, pays_by_inv.get(inv["invoice_id"], []), fx_date_policy
                    )
                    fk = fx_record_key(fx_date, inv["currency"], base_currency)
                    rate_str = fx_by_key.get(fk)
                    if rate_str is None:
                        all_issues_verified = False
                        errors.append(f"{iid}: missing FX rate {fk}")
                    else:
                        exp_base = convert_minor(
                            int(inv["total_minor"]),
                            inv["currency"],
                            base_currency,
                            rate_str,
                            rounding_rule,
                        )
                        recalc_diff = abs(int(ar_row["debit_minor"]) - exp_base)
                        tol = int(issue.get("tolerance_minor", 1))
                        if recalc_diff <= tol or abs(recalc_diff - int(issue["expected_amount_minor"])) > tol:
                            all_issues_verified = False
                            errors.append(
                                f"{iid}: fx_conversion_error recalc={recalc_diff} vs "
                                f"expected={issue['expected_amount_minor']}"
                            )

        elif itype == "period_cutoff_error":
            affected = issue.get("affected_ids", [])
            if len(affected) != 1 or affected[0] not in inv_by_id:
                all_issues_verified = False
                errors.append(f"{iid}: period_cutoff_error invalid affected_ids {affected}")
            else:
                inv = inv_by_id[affected[0]]
                ar_row = ar_row_by_inv.get(inv["invoice_id"])
                s_date = str(inv["line_items"][0]["service_date"])
                outside_period = s_date < period_start_date or s_date > period_end_date
                posted_in_period = (
                    ar_row is not None
                    and period_start_dt <= datetime.fromisoformat(ar_row["timestamp"]) <= period_end_dt
                )
                if (
                    not outside_period
                    or not posted_in_period
                    or int(issue["expected_amount_minor"]) != int(inv["total_minor"])
                ):
                    all_issues_verified = False
                    errors.append(f"{iid}: period_cutoff_error independent verification failed")
        else:
            all_issues_verified = False
            errors.append(f"{iid}: unknown issue type {itype!r}")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "checks": {
            "pre_injection_balanced": pre_balanced,
            "post_injection_balanced": post_balanced,
            "row_count_in_bounds": row_count_ok,
            "doc_count_in_bounds": doc_count_ok,
            "all_evidence_ids_exist": all_evidence_ok,
            "all_issues_independently_verified": all_issues_verified,
        },
        "stats": {
            "ledger_rows": len(ledger),
            "documents": doc_count,
            "invoices": len(invoices),
            "payments": len(payments),
            "fx_rates": len(fx_rates),
            "num_issues": len(issues),
        },
    }


def _remap_recursive(obj: Any, id_map: dict[str, str]) -> Any:
    if isinstance(obj, str):
        return id_map.get(obj, obj)
    if isinstance(obj, list):
        return [_remap_recursive(x, id_map) for x in obj]
    if isinstance(obj, dict):
        return {id_map.get(k, k) if isinstance(k, str) else k: _remap_recursive(v, id_map) for k, v in obj.items()}
    return obj


def permute_and_relabel_episode(
    episode: GeneratedEpisode,
    perm_seed: int,
) -> tuple[GeneratedEpisode, dict[str, str]]:
    """Deterministically shuffle row/document order and remap all IDs consistently."""
    rng = random.Random(perm_seed)
    id_map: dict[str, str] = {}

    old_ep_id = episode.manifest["episode_id"]
    new_ep_id = f"ep-{rng.getrandbits(48):012x}"
    while new_ep_id == old_ep_id:
        new_ep_id = f"ep-{rng.getrandbits(48):012x}"
    id_map[old_ep_id] = new_ep_id

    # Collect all unique IDs in deterministic sorted order before remapping
    entity_ids = sorted(
        {
            x
            for x in (
                [inv["entity_id"] for inv in episode.invoices]
                + [r["entity_id"] for r in episode.ledger if r.get("entity_id") is not None]
            )
            if x is not None
        }
    )
    invoice_ids = sorted({inv["invoice_id"] for inv in episode.invoices})
    payment_ids = sorted({pay["payment_id"] for pay in episode.payments})
    doc_ids = sorted(
        {inv["doc_id"] for inv in episode.invoices} | {pay["doc_id"] for pay in episode.payments}
    )
    row_ids = sorted({r["row_id"] for r in episode.ledger})

    def _assign_new_ids(old_ids: list[str], prefix: str, width: int) -> None:
        used_new: set[str] = set(old_ids)
        base_num = rng.randint(10 ** (width - 1), 5 * 10 ** (width - 1))
        shuffled_offsets = list(range(len(old_ids) * 3 + 17))
        rng.shuffle(shuffled_offsets)
        cursor = 0
        for old_id in old_ids:
            while True:
                cand = f"{prefix}-{base_num + shuffled_offsets[cursor]:0{width}d}"
                cursor += 1
                if cand not in used_new:
                    used_new.add(cand)
                    id_map[old_id] = cand
                    break

    _assign_new_ids(entity_ids, "ENT", 4)
    _assign_new_ids(invoice_ids, "INV", 5)
    _assign_new_ids(payment_ids, "PAY", 5)
    _assign_new_ids(doc_ids, "DOC", 6)
    _assign_new_ids(row_ids, "ROW", 6)

    new_manifest = copy.deepcopy(episode.manifest)
    new_manifest["episode_id"] = id_map[old_ep_id]

    new_policy = copy.deepcopy(episode.policy)

    new_ledger: list[dict[str, Any]] = []
    for row in episode.ledger:
        nr = copy.deepcopy(row)
        nr["row_id"] = id_map[nr["row_id"]]
        if nr.get("entity_id") is not None:
            nr["entity_id"] = id_map.get(nr["entity_id"], nr["entity_id"])
        if nr.get("invoice_id") is not None:
            nr["invoice_id"] = id_map.get(nr["invoice_id"], nr["invoice_id"])
        if nr.get("payment_id") is not None:
            nr["payment_id"] = id_map.get(nr["payment_id"], nr["payment_id"])
        new_ledger.append(nr)

    new_invoices: list[dict[str, Any]] = []
    for inv in episode.invoices:
        ni = copy.deepcopy(inv)
        ni["doc_id"] = id_map[ni["doc_id"]]
        ni["invoice_id"] = id_map[ni["invoice_id"]]
        ni["entity_id"] = id_map[ni["entity_id"]]
        new_invoices.append(ni)

    new_payments: list[dict[str, Any]] = []
    for pay in episode.payments:
        np_ = copy.deepcopy(pay)
        np_["doc_id"] = id_map[np_["doc_id"]]
        np_["payment_id"] = id_map[np_["payment_id"]]
        if np_.get("invoice_refs") is not None:
            for ref in np_["invoice_refs"]:
                ref["invoice_id"] = id_map.get(ref["invoice_id"], ref["invoice_id"])
        new_payments.append(np_)

    new_fx_rates = copy.deepcopy(episode.fx_rates)

    # Shuffle open tables deterministically
    rng.shuffle(new_ledger)
    rng.shuffle(new_invoices)
    rng.shuffle(new_payments)
    rng.shuffle(new_fx_rates)

    new_truth = copy.deepcopy(episode.truth)
    new_truth["episode_id"] = id_map[old_ep_id]
    for issue in new_truth.get("issues", []):
        issue["affected_ids"] = sorted(id_map.get(x, x) for x in issue["affected_ids"])
        issue["acceptable_evidence_sets"] = [
            [id_map.get(ev, ev) for ev in ev_set]
            for ev_set in issue["acceptable_evidence_sets"]
        ]

    new_provenance = _remap_recursive(copy.deepcopy(episode.provenance), id_map)

    new_episode = GeneratedEpisode(
        manifest=new_manifest,
        policy=new_policy,
        ledger=new_ledger,
        invoices=new_invoices,
        payments=new_payments,
        fx_rates=new_fx_rates,
        truth=new_truth,
        provenance=new_provenance,
    )
    return new_episode, id_map
