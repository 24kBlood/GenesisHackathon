"""Clean accounts payable history and its public and evaluator projections."""

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from functools import partial
from hashlib import sha256
from random import Random
import re
from secrets import randbits, token_hex
from typing import Any, Literal, Mapping, NotRequired, TypedDict
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from genesis.domain.enums import FindingType
from genesis.domain.models import EpisodeTruth
from genesis.domain.money import convert_minor

from .allocation import apply_allocation_errors
from .distractors import add_clean_distractors, plan_clean_distractors
from .fx_cutoff import apply_fx_cutoff_errors
from .interactions import apply_interaction, build_terms, normalize_scenarios


class LedgerRow(TypedDict):
    row_id: str
    timestamp: str
    account: str
    debit_minor: int
    credit_minor: int
    currency: str
    entity_id: str
    invoice_id: str | None
    payment_id: str | None


class Invoice(TypedDict):
    doc_id: str
    invoice_id: str
    entity_id: str
    issued_at: str
    due_at: str
    currency: str
    total_minor: int
    line_items: list[dict[str, Any]]


class Payment(TypedDict):
    doc_id: str
    payment_id: str
    timestamp: str
    currency: str
    amount_minor: int
    invoice_refs: list[str] | None
    payment_kind: NotRequired[Literal["refund"]]
    refund_of_payment_id: NotRequired[str]


class FxQuote(TypedDict):
    date: str
    source_currency: str
    target_currency: str
    rate: str


@dataclass
class PublicEpisode:
    manifest: dict[str, Any]
    policy: dict[str, str]
    ledger: list[LedgerRow]
    invoices: list[Invoice]
    payments: list[Payment]
    fx: list[FxQuote]


@dataclass
class EvaluatorEpisode:
    truth: EpisodeTruth
    provenance: dict[str, Any]


@dataclass
class Episode:
    public: PublicEpisode
    evaluator: EvaluatorEpisode


@dataclass(frozen=True)
class Obligation:
    currency: str
    quantity: int
    unit_price_minor: int
    issued_at: str
    due_at: str
    paid_at: str | None
    conversion_rate: str
    description: str = "Synthetic service"

    @property
    def total_minor(self) -> int:
        return self.quantity * self.unit_price_minor


SCHEMA_VERSION = "1.0.0"
BASE_CURRENCY = "KZT"
FOREIGN_CURRENCY = "JPY"
DEFAULT_TIMEZONE = "Asia/Almaty"
DEFAULT_FX_DATE_RULE = "payment_date"
DEFAULT_OBLIGATION_COUNT = 4
MIN_OBLIGATION_COUNT = 4
MAX_OBLIGATION_COUNT = 500
EXPONENTS = {"KZT": 2, "JPY": 0}
CURRENCY_EXPONENTS: dict[str, int] = {"KZT": 2, "JPY": 0, "USD": 2, "EUR": 2, "GBP": 2}
_USD_VALUE: dict[str, Decimal] = {
    "USD": Decimal("1"),
    "EUR": Decimal("1.08"),
    "GBP": Decimal("1.27"),
    "KZT": Decimal("0.00202"),
    "JPY": Decimal("0.00667"),
}
_FX_DATE_RULES = frozenset({"payment_date", "invoice_date"})
_OPENING_CASH_RESERVE = 200_000
_SERVICE_DESCRIPTIONS = (
    "Cloud compute hosting",
    "Legal advisory retainer",
    "Freight & customs brokerage",
    "Data center colocation",
    "Enterprise software license",
    "Equipment maintenance",
    "External audit support",
    "Telecom bandwidth lease",
    "Office facilities lease",
    "Payroll processing service",
    "Security monitoring retainer",
    "Warehouse handling fee",
)
FINDING_TYPES = frozenset(FindingType)


def _generation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping) or config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"configuration requires schema_version {SCHEMA_VERSION}")
    unexpected = config.keys() - {
        "schema_version", "error_counts", "interaction_scenarios", "include_distractors",
        "obligation_count", "base_currency", "foreign_currency", "timezone", "fx_date_rule", "episode_id",
    }
    if unexpected:
        raise ValueError("unknown generation configuration fields")
    counts = config.get("error_counts")
    if not isinstance(counts, Mapping):
        raise ValueError("error_counts must be a mapping")
    normalized = {}
    for name, count in counts.items():
        if name not in FINDING_TYPES:
            raise ValueError(f"unknown error type: {name!r}")
        if type(count) is not int or count < 0:
            raise ValueError("error counts must be non-negative integers")
        if count:
            normalized[str(name)] = count
    interactions = normalize_scenarios(config.get("interaction_scenarios", []))
    distractors = config.get("include_distractors", False)
    if type(distractors) is not bool:
        raise ValueError("include_distractors must be a boolean")
    obligation_count = config.get("obligation_count", DEFAULT_OBLIGATION_COUNT)
    if type(obligation_count) is not int or not MIN_OBLIGATION_COUNT <= obligation_count <= MAX_OBLIGATION_COUNT:
        raise ValueError(
            f"obligation_count must be an integer between {MIN_OBLIGATION_COUNT} and {MAX_OBLIGATION_COUNT}"
        )
    base_currency = config.get("base_currency", BASE_CURRENCY)
    if type(base_currency) is not str or base_currency not in CURRENCY_EXPONENTS:
        raise ValueError(f"unsupported base_currency: {base_currency!r}")
    foreign_currency = config.get("foreign_currency", FOREIGN_CURRENCY)
    if type(foreign_currency) is not str or foreign_currency not in CURRENCY_EXPONENTS:
        raise ValueError(f"unsupported foreign_currency: {foreign_currency!r}")
    if base_currency == foreign_currency:
        raise ValueError("base_currency and foreign_currency must differ")
    tz_name = config.get("timezone", DEFAULT_TIMEZONE)
    if type(tz_name) is not str or not tz_name:
        raise ValueError("timezone must be a non-empty IANA timezone name")
    try:
        ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError, KeyError):
        raise ValueError(f"unknown timezone: {tz_name!r}") from None
    fx_date_rule = config.get("fx_date_rule", DEFAULT_FX_DATE_RULE)
    if type(fx_date_rule) is not str or fx_date_rule not in _FX_DATE_RULES:
        raise ValueError(f"unsupported fx_date_rule: {fx_date_rule!r}")
    if interactions and fx_date_rule != "payment_date":
        raise ValueError("interaction_scenarios require fx_date_rule='payment_date'")
    result: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "error_counts": normalized}
    if "episode_id" in config:
        episode_id = config["episode_id"]
        if type(episode_id) is not str or re.fullmatch(r"episode-(?:[0-9a-f]{24}|[0-9a-f]{32})", episode_id) is None:
            raise ValueError("episode_id must have an episode- prefix and 24 or 32 lowercase hex digits")
        result["episode_id"] = episode_id
    if distractors:
        result["include_distractors"] = True
    if interactions:
        result["interaction_scenarios"] = interactions
    if obligation_count != DEFAULT_OBLIGATION_COUNT:
        result["obligation_count"] = obligation_count
    if base_currency != BASE_CURRENCY:
        result["base_currency"] = base_currency
    if foreign_currency != FOREIGN_CURRENCY:
        result["foreign_currency"] = foreign_currency
    if tz_name != DEFAULT_TIMEZONE:
        result["timezone"] = tz_name
    if fx_date_rule != DEFAULT_FX_DATE_RULE:
        result["fx_date_rule"] = fx_date_rule
    return result


def _rng(seed: int, stream: str) -> Random:
    material = f"genesis:{SCHEMA_VERSION}:{seed}:{stream}".encode("utf-8")
    return Random(int.from_bytes(sha256(material).digest(), "big"))


def _timestamp(day: date, tz: ZoneInfo, hour: int = 10) -> str:
    return datetime(day.year, day.month, day.day, hour, tzinfo=tz).isoformat()


def _format_rate(value: Decimal, places: int = 4) -> str:
    quantum = Decimal(1).scaleb(-places)
    rounded = value.quantize(quantum, rounding=ROUND_HALF_EVEN)
    if rounded <= 0:
        rounded = quantum
    return format(rounded, "f")


def _sample_rate(rng: Random, source: str, target: str, *, four_decimals: bool) -> str:
    if not four_decimals and (source, target) == ("JPY", "KZT"):
        major, hundredths = divmod(rng.randint(300, 380), 100)
        return f"{major}.{hundredths:02d}"
    base_ratio = _USD_VALUE[source] / _USD_VALUE[target]
    factor = Decimal(1) + Decimal(rng.randint(-800, 800)) / Decimal(10_000)
    return _format_rate(base_ratio * factor, places=4)


def _interaction_rates(source: str, target: str) -> tuple[tuple[str, str], tuple[str, str]]:
    if (source, target) == ("JPY", "KZT"):
        return (("3.40", "3.60"), ("3.20", "3.25"))
    base_ratio = _USD_VALUE[source] / _USD_VALUE[target]

    def rate_pair(correct_factor: str, wrong_factor: str) -> tuple[str, str]:
        correct = base_ratio * Decimal(correct_factor)
        wrong = base_ratio * Decimal(wrong_factor)
        pair = _format_rate(correct), _format_rate(wrong)
        if pair[0] == pair[1]:
            return _format_rate(correct, places=8), _format_rate(wrong, places=8)
        return pair

    return (
        rate_pair("1.02", "1.08"),
        rate_pair("0.98", "1.005"),
    )


def _history(
    seed: int,
    *,
    obligation_count: int,
    base_currency: str,
    foreign_currency: str,
    tz: ZoneInfo,
) -> tuple[date, date, list[Obligation]]:
    rng = _rng(seed, "history")
    month = rng.randint(1, 12)
    start = date(2026, month, 1)
    end = date(2026 + (month == 12), month % 12 + 1, 1)
    if obligation_count == DEFAULT_OBLIGATION_COUNT:
        foreign_rate = _sample_rate(
            rng, foreign_currency, base_currency,
            four_decimals=(foreign_currency, base_currency) != ("JPY", "KZT"),
        )
        unpaid_near_end = rng.choice((False, True))
        unpaid_issued = _timestamp(end - timedelta(days=1), tz, hour=16) if unpaid_near_end else _timestamp(start, tz)
        unpaid_due = _timestamp(end + timedelta(days=14), tz) if unpaid_near_end else _timestamp(start + timedelta(days=21), tz)
        obligations = [
            Obligation(
                currency=foreign_currency,
                quantity=rng.randint(1, 6),
                unit_price_minor=rng.randint(1, 50) * 100,
                issued_at=_timestamp(start + timedelta(days=3), tz),
                due_at=_timestamp(start + timedelta(days=21), tz),
                paid_at=_timestamp(start + timedelta(days=7), tz),
                conversion_rate=foreign_rate,
            ),
            Obligation(
                currency=base_currency,
                quantity=rng.randint(1, 6),
                unit_price_minor=rng.randint(1, 50) * 100,
                issued_at=_timestamp(start + timedelta(days=2), tz),
                due_at=_timestamp(start + timedelta(days=21), tz),
                paid_at=_timestamp(start + timedelta(days=5), tz),
                conversion_rate="1",
            ),
            Obligation(
                currency=base_currency,
                quantity=rng.randint(1, 6),
                unit_price_minor=rng.randint(1, 50) * 100,
                issued_at=_timestamp(start + timedelta(days=4), tz),
                due_at=_timestamp(start + timedelta(days=21), tz),
                paid_at=_timestamp(start + timedelta(days=9), tz),
                conversion_rate="1",
            ),
            Obligation(
                currency=base_currency,
                quantity=rng.randint(1, 6),
                unit_price_minor=rng.randint(1, 50) * 100,
                issued_at=unpaid_issued,
                due_at=unpaid_due,
                paid_at=None,
                conversion_rate="1",
            ),
        ]
        return start, end, obligations

    foreign_pool = [
        foreign_currency,
        *[code for code in ("USD", "EUR", "GBP", "JPY", "KZT") if code not in (base_currency, foreign_currency)][:2],
    ]
    rates = {code: _sample_rate(rng, code, base_currency, four_decimals=True) for code in foreign_pool}
    # Ensure at least one foreign currency rate produces fractional minor units (exercising ROUND_HALF_EVEN).
    primary = foreign_pool[0]
    if Decimal(rates[primary]).scaleb(CURRENCY_EXPONENTS[base_currency] - CURRENCY_EXPONENTS[primary]) % 1 == 0:
        rates[primary] = _format_rate(Decimal(rates[primary]) + Decimal("0.0035"), places=4)

    n_unpaid = max(2, obligation_count // 4)
    n_foreign = max(2, (obligation_count * 3) // 10)
    n_domestic_paid = obligation_count - n_unpaid - n_foreign
    obligations: list[Obligation] = []

    for index in range(n_foreign):
        currency = foreign_pool[index % len(foreign_pool)]
        issue_day = rng.randint(1, 13)
        payment_day = issue_day + rng.randint(2, 5)
        unit_price = rng.randint(137, 9873)
        if index == 0 and unit_price % 2 == 0:
            unit_price += 1
        quantity = 1 if index == 0 else rng.randint(1, 8)
        obligations.append(Obligation(
            currency=currency,
            quantity=quantity,
            unit_price_minor=unit_price,
            issued_at=_timestamp(start + timedelta(days=issue_day), tz, hour=rng.randint(9, 16)),
            due_at=_timestamp(start + timedelta(days=21), tz),
            paid_at=_timestamp(start + timedelta(days=payment_day), tz, hour=rng.randint(9, 16)),
            conversion_rate=rates[currency],
            description=_SERVICE_DESCRIPTIONS[index % len(_SERVICE_DESCRIPTIONS)],
        ))

    for index in range(n_domestic_paid):
        issue_day = rng.randint(1, 14)
        payment_day = issue_day + rng.randint(2, 5)
        obligations.append(Obligation(
            currency=base_currency,
            quantity=rng.randint(1, 10),
            unit_price_minor=rng.randint(250, 9500),
            issued_at=_timestamp(start + timedelta(days=issue_day), tz, hour=rng.randint(9, 16)),
            due_at=_timestamp(start + timedelta(days=21), tz),
            paid_at=_timestamp(start + timedelta(days=payment_day), tz, hour=rng.randint(9, 16)),
            conversion_rate="1",
            description=_SERVICE_DESCRIPTIONS[(index + n_foreign) % len(_SERVICE_DESCRIPTIONS)],
        ))

    for index in range(n_unpaid):
        near_end = index % 2 == 1
        issued = _timestamp(end - timedelta(days=1), tz, hour=16) if near_end else _timestamp(start, tz, hour=10)
        due = _timestamp(end + timedelta(days=14), tz) if near_end else _timestamp(start + timedelta(days=21), tz)
        obligations.append(Obligation(
            currency=base_currency,
            quantity=rng.randint(1, 8),
            unit_price_minor=rng.randint(300, 9000),
            issued_at=issued,
            due_at=due,
            paid_at=None,
            conversion_rate="1",
            description=_SERVICE_DESCRIPTIONS[(index + n_foreign + n_domestic_paid) % len(_SERVICE_DESCRIPTIONS)],
        ))

    return start, end, obligations


def _base_amount(
    amount: int, source_currency: str, rate: str,
    base_currency: str = BASE_CURRENCY, exponents: dict[str, int] = CURRENCY_EXPONENTS,
) -> int:
    return convert_minor(amount, source_currency, base_currency, rate, exponents, ROUND_HALF_EVEN)


class _Ids:
    def __init__(self, seed: int):
        self.public_random = _rng(seed, "identifiers")
        self.private_random = _rng(seed, "private-identifiers")
        self.used: set[str] = set()

    def new(self, namespace: str) -> str:
        stream = self.private_random if namespace in ("issue", "change") else self.public_random
        while True:
            value = f"{namespace}-{stream.getrandbits(96):024x}"
            if value not in self.used:
                self.used.add(value)
                return value


def _post(
    ledger: list[LedgerRow], ids: _Ids, timestamp: str, entity_id: str, amount: int,
    debit: str, credit: str, *, currency: str = BASE_CURRENCY,
    invoice_id: str | None = None, payment_id: str | None = None,
) -> None:
    for account, debit_minor, credit_minor in ((debit, amount, 0), (credit, 0, amount)):
        ledger.append({
            "row_id": ids.new("row"), "timestamp": timestamp, "account": account,
            "debit_minor": debit_minor, "credit_minor": credit_minor,
            "currency": currency, "entity_id": entity_id,
            "invoice_id": invoice_id, "payment_id": payment_id,
        })


def new_closed_case(config: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    """Create independent identifiers for a new case in the trusted coordinator."""
    prepared = _generation_config(config)
    prepared["episode_id"] = "episode-" + token_hex(16)
    return (1 << 128) | randbits(128), prepared


def generate_closed_episode(seed: int, config: Mapping[str, Any]) -> Episode:
    """Generate or replay a case provisioned by new_closed_case."""
    if type(seed) is not int or seed < (1 << 128):
        raise ValueError("closed generation requires a private random seed from new_closed_case")
    normalized = _generation_config(config)
    if re.fullmatch(r"episode-[0-9a-f]{32}", normalized.get("episode_id", "")) is None:
        raise ValueError("closed generation requires an independent 128-bit episode_id")
    return generate_episode(seed, normalized)


def generate_episode(seed: int, config: Mapping[str, Any]) -> Episode:
    """Generate one episode from a v1 configuration and a supplied seed.

    In the clean history, a bill credits accounts payable and debits expense
    in base_currency. A full payment debits that payable and credits bank in
    base_currency, while the payment document retains the source currency.
    Foreign accrual and payment use equal spot quotes, so the clean history
    books no FX gain or loss. Opening bank equity has no references and carries
    a positive working-capital buffer so cash stays non-negative after FX errors.
    Legacy public examples may omit episode_id. Use generate_closed_episode for hidden cases.
    """
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    normalized_config = _generation_config(config)
    obligation_count = normalized_config.get("obligation_count", DEFAULT_OBLIGATION_COUNT)
    base_currency = normalized_config.get("base_currency", BASE_CURRENCY)
    foreign_currency = normalized_config.get("foreign_currency", FOREIGN_CURRENCY)
    tz_name = normalized_config.get("timezone", DEFAULT_TIMEZONE)
    fx_date_rule = normalized_config.get("fx_date_rule", DEFAULT_FX_DATE_RULE)
    tz = ZoneInfo(tz_name)

    with localcontext(Context(prec=64, rounding=ROUND_HALF_EVEN)):
        start, end, obligations = _history(
            seed,
            obligation_count=obligation_count,
            base_currency=base_currency,
            foreign_currency=foreign_currency,
            tz=tz,
        )
        pair_rates = _interaction_rates(foreign_currency, base_currency)
    distractors = (
        plan_clean_distractors(_rng(seed, "clean-distractors"))
        if normalized_config.get("include_distractors") else None
    )
    terms = build_terms(
        normalized_config.get("interaction_scenarios", []),
        datetime.fromisoformat(_timestamp(end, tz, hour=0)),
        _rng(seed, "interaction-terms"),
        source_currency=foreign_currency,
        target_currency=base_currency,
        exponents=CURRENCY_EXPONENTS,
        pair_rates=pair_rates,
    )
    scenario_obligations = [
        Obligation(foreign_currency, 1, item.amount_minor, item.issued_at, item.paid_at, item.paid_at, item.correct_rate)
        for item in terms.values()
    ]
    history = [*obligations, *scenario_obligations]
    used_currencies = {base_currency, foreign_currency, *(item.currency for item in history)}
    exponents = {code: CURRENCY_EXPONENTS[code] for code in sorted(used_currencies)}
    obligation_total = sum(_base_amount(
        item.total_minor, item.currency, item.conversion_rate, base_currency, exponents,
    ) for item in history)
    opening_balance = obligation_total + max(_OPENING_CASH_RESERVE, obligation_total // 2)
    if distractors is not None:
        opening_balance += distractors.opening_cash
    ids = _Ids(seed)
    legacy_episode_id = ids.new("episode")
    episode_id = normalized_config.get("episode_id", legacy_episode_id)
    ledger: list[LedgerRow] = []
    invoices: list[Invoice] = []
    payments: list[Payment] = []
    fx_by_key: dict[tuple[str, str, str], FxQuote] = {}
    scenario_docs: dict[str, tuple[Invoice, Payment]] = {}

    def register_fx(day_str: str, source: str, target: str, rate: str) -> None:
        key = (day_str, source, target)
        existing = fx_by_key.get(key)
        if existing is not None:
            if existing["rate"] != rate:
                raise ValueError(f"conflicting FX quote for {key}")
            return
        fx_by_key[key] = {
            "date": day_str, "source_currency": source,
            "target_currency": target, "rate": rate,
        }

    _post(ledger, ids, _timestamp(start, tz), ids.new("entity"), opening_balance, "bank", "equity", currency=base_currency)
    names = [None] * len(obligations) + list(terms)
    for name, item in zip(names, history, strict=True):
        invoice_id = ids.new("invoice")
        entity_id = ids.new("entity")
        invoices.append({
            "doc_id": ids.new("doc"), "invoice_id": invoice_id, "entity_id": entity_id,
            "issued_at": item.issued_at, "due_at": item.due_at, "currency": item.currency,
            "total_minor": item.total_minor,
            "line_items": [{
                "description": item.description, "quantity": item.quantity,
                "unit_price_minor": item.unit_price_minor, "amount_minor": item.total_minor,
            }],
        })
        amount = _base_amount(item.total_minor, item.currency, item.conversion_rate, base_currency, exponents)
        _post(
            ledger, ids, item.issued_at, entity_id, amount, "expense", "accounts_payable",
            currency=base_currency, invoice_id=invoice_id,
        )
        if item.paid_at is not None:
            payment_id = ids.new("payment")
            payments.append({
                "doc_id": ids.new("doc"), "payment_id": payment_id, "timestamp": item.paid_at,
                "currency": item.currency, "amount_minor": item.total_minor,
                "invoice_refs": [invoice_id],
            })
            if name is not None:
                scenario_docs[name] = (invoices[-1], payments[-1])
            _post(
                ledger, ids, item.paid_at, entity_id, amount, "accounts_payable", "bank",
                currency=base_currency, invoice_id=invoice_id, payment_id=payment_id,
            )
        if item.currency != base_currency:
            for timestamp in (item.issued_at, item.paid_at):
                if timestamp is not None:
                    quote_day = datetime.fromisoformat(timestamp).astimezone(tz).date().isoformat()
                    register_fx(quote_day, item.currency, base_currency, item.conversion_rate)
        if name is not None and (quote_date := terms[name].alternative_quote_date):
            register_fx(quote_date, item.currency, base_currency, terms[name].wrong_rate)

    fx = list(fx_by_key.values())
    initial_history = {
        "opening_bank_minor": opening_balance,
        "obligations": [asdict(item) for item in history],
    }
    if terms:
        initial_history["market_quotes"] = [quote.copy() for quote in fx]
    if distractors is not None:
        initial_history["distractor_operations"] = add_clean_distractors(
            invoices, payments,
            base_currency=base_currency,
            plan=distractors,
            new_id=ids.new,
            date_at=lambda offset: _timestamp(start + timedelta(days=offset), tz),
            post=partial(_post, ledger, ids, currency=base_currency),
        )

    counts = normalized_config["error_counts"]
    issues, changes = apply_allocation_errors(
        ledger, invoices, payments,
        base_currency=base_currency,
        duplicate_count=counts.get(FindingType.DUPLICATE_INVOICE, 0),
        allocation_count=counts.get(FindingType.PAYMENT_ALLOCATION, 0),
        rng=_rng(seed, "allocation-errors"),
        new_id=ids.new,
    )

    policy = {
        "fx_date_rule": fx_date_rule, "timezone": tz_name,
        "period_start": _timestamp(start, tz, hour=0),
        "period_end": _timestamp(end, tz, hour=0),
        "rounding_mode": ROUND_HALF_EVEN,
        "recognition_date_rule": "document_date",
    }
    manifest = {
        "schema_version": SCHEMA_VERSION, "episode_id": episode_id,
        "base_currency": base_currency, "currency_exponents": exponents.copy(),
        "record_counts": {
            "ledger": len(ledger), "invoices": len(invoices),
            "payments": len(payments), "fx": len(fx),
        },
        "document_count": len(invoices) + len(payments),
        **{key: policy[key] for key in ("timezone", "period_start", "period_end", "rounding_mode")},
    }
    public = PublicEpisode(manifest, policy, ledger, invoices, payments, fx)
    if counts.get(FindingType.FX_CONVERSION) or counts.get(FindingType.PERIOD_CUTOFF):
        public.ledger, fx_cutoff_issues, fx_cutoff_changes = apply_fx_cutoff_errors(
            public, counts, ids.new,
            reserved_payment_ids={payment["payment_id"] for _, payment in scenario_docs.values()},
        )
        issues.extend(fx_cutoff_issues)
        changes.extend(fx_cutoff_changes)
    for name, (invoice, payment) in scenario_docs.items():
        scenario_issues, scenario_changes = apply_interaction(
            public, name, invoice, payment, terms[name], ids.new,
        )
        issues.extend(scenario_issues)
        changes.extend(scenario_changes)
    manifest["record_counts"]["ledger"] = len(public.ledger)
    order = _rng(seed, "publication-order")
    for rows in (public.ledger, invoices, payments, fx):
        order.shuffle(rows)
    return Episode(
        public=public,
        evaluator=EvaluatorEpisode(
            truth={"episode_id": episode_id, "issues": issues},
            provenance={
                "episode_id": episode_id, "seed": seed, "config": normalized_config,
                "initial_history": initial_history,
                "changes": changes,
            },
        ),
    )
