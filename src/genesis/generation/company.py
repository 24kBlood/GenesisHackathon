"""Clean accounts payable history and its public and evaluator projections."""

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_EVEN
from functools import partial
from hashlib import sha256
from random import Random
from typing import Any, Literal, Mapping, NotRequired, TypedDict

from genesis.domain.enums import FindingType
from genesis.domain.models import EpisodeTruth
from genesis.domain.money import convert_minor

from .allocation import apply_allocation_errors
from .distractors import add_clean_distractors, plan_clean_distractors
from .fx_cutoff import apply_fx_cutoff_errors


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
    invoice_refs: list[str]
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

    @property
    def total_minor(self) -> int:
        return self.quantity * self.unit_price_minor


SCHEMA_VERSION = "1.0.0"
BASE_CURRENCY = "KZT"
EXPONENTS = {"KZT": 2, "JPY": 0}
LOCAL_OFFSET = timezone(timedelta(hours=5))
FINDING_TYPES = frozenset(FindingType)


def _generation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping) or config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"configuration requires schema_version {SCHEMA_VERSION}")
    unexpected = config.keys() - {
        "schema_version", "error_counts", "interaction_scenarios", "include_distractors",
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
    interactions = config.get("interaction_scenarios", [])
    distractors = config.get("include_distractors", False)
    if not isinstance(interactions, list) or type(distractors) is not bool:
        raise ValueError("invalid interaction_scenarios or include_distractors")
    if interactions:
        raise ValueError("interactions are not supported")
    result = {"schema_version": SCHEMA_VERSION, "error_counts": normalized}
    if distractors:
        result["include_distractors"] = True
    if "interaction_scenarios" in config:
        result["interaction_scenarios"] = []
    return result


def _rng(seed: int, stream: str) -> Random:
    material = f"genesis:{SCHEMA_VERSION}:{seed}:{stream}".encode("utf-8")
    return Random(int.from_bytes(sha256(material).digest(), "big"))


def _timestamp(day: date, hour: int = 10) -> str:
    return datetime(day.year, day.month, day.day, hour, tzinfo=LOCAL_OFFSET).isoformat()


def _history(seed: int) -> tuple[date, date, list[Obligation]]:
    rng = _rng(seed, "history")
    month = rng.randint(1, 12)
    start = date(2026, month, 1)
    end = date(2026 + (month == 12), month % 12 + 1, 1)
    major, hundredths = divmod(rng.randint(300, 380), 100)
    jpy_rate = f"{major}.{hundredths:02d}"
    obligations = [
        Obligation(
            currency=currency,
            quantity=rng.randint(1, 6),
            unit_price_minor=rng.randint(1, 50) * 100,
            issued_at=_timestamp(start + timedelta(days=issue_day)),
            due_at=_timestamp(start + timedelta(days=21)),
            paid_at=None if payment_day is None else _timestamp(start + timedelta(days=payment_day)),
            conversion_rate=jpy_rate if currency == "JPY" else "1",
        )
        for currency, issue_day, payment_day in (
            ("JPY", 3, 7), ("KZT", 2, 5), ("KZT", 4, 9), ("KZT", 0, None),
        )
    ]
    return start, end, obligations


def _base_amount(amount: int, source_currency: str, rate: str) -> int:
    return convert_minor(amount, source_currency, BASE_CURRENCY, rate, EXPONENTS, ROUND_HALF_EVEN)


class _Ids:
    def __init__(self, seed: int):
        self.random = _rng(seed, "identifiers")
        self.used: set[str] = set()

    def new(self, namespace: str) -> str:
        while True:
            value = f"{namespace}-{self.random.getrandbits(96):024x}"
            if value not in self.used:
                self.used.add(value)
                return value


def _post(
    ledger: list[LedgerRow], ids: _Ids, timestamp: str, entity_id: str, amount: int,
    debit: str, credit: str, *, invoice_id: str | None = None,
    payment_id: str | None = None,
) -> None:
    for account, debit_minor, credit_minor in ((debit, amount, 0), (credit, 0, amount)):
        ledger.append({
            "row_id": ids.new("row"), "timestamp": timestamp, "account": account,
            "debit_minor": debit_minor, "credit_minor": credit_minor,
            "currency": BASE_CURRENCY, "entity_id": entity_id,
            "invoice_id": invoice_id, "payment_id": payment_id,
        })


def generate_episode(seed: int, config: Mapping[str, Any]) -> Episode:
    """Generate one episode from a v1 configuration and a supplied seed.

    In the clean history, a bill credits accounts payable and debits expense
    in KZT. A full payment debits that payable and credits bank in KZT, while
    the payment document retains the source currency. Foreign accrual and
    payment use equal spot quotes, so the clean history books no FX gain or
    loss. One invoice is left open. Opening bank equity has no references.
    """
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    normalized_config = _generation_config(config)
    start, end, obligations = _history(seed)
    distractors = (
        plan_clean_distractors(_rng(seed, "clean-distractors"))
        if normalized_config.get("include_distractors") else None
    )
    opening_balance = sum(_base_amount(
        item.total_minor, item.currency, item.conversion_rate,
    ) for item in obligations)
    if distractors is not None:
        opening_balance += distractors.opening_cash
    ids = _Ids(seed)
    episode_id = ids.new("episode")
    ledger: list[LedgerRow] = []
    invoices: list[Invoice] = []
    payments: list[Payment] = []
    fx: list[FxQuote] = []
    _post(ledger, ids, _timestamp(start), ids.new("entity"), opening_balance, "bank", "equity")
    for item in obligations:
        invoice_id = ids.new("invoice")
        entity_id = ids.new("entity")
        invoices.append({
            "doc_id": ids.new("doc"), "invoice_id": invoice_id, "entity_id": entity_id,
            "issued_at": item.issued_at, "due_at": item.due_at, "currency": item.currency,
            "total_minor": item.total_minor,
            "line_items": [{
                "description": "Synthetic service", "quantity": item.quantity,
                "unit_price_minor": item.unit_price_minor, "amount_minor": item.total_minor,
            }],
        })
        amount = _base_amount(item.total_minor, item.currency, item.conversion_rate)
        _post(
            ledger, ids, item.issued_at, entity_id, amount, "expense", "accounts_payable",
            invoice_id=invoice_id,
        )
        if item.paid_at is not None:
            payment_id = ids.new("payment")
            payments.append({
                "doc_id": ids.new("doc"), "payment_id": payment_id, "timestamp": item.paid_at,
                "currency": item.currency, "amount_minor": item.total_minor,
                "invoice_refs": [invoice_id],
            })
            _post(
                ledger, ids, item.paid_at, entity_id, amount, "accounts_payable", "bank",
                invoice_id=invoice_id, payment_id=payment_id,
            )
        if item.currency != BASE_CURRENCY:
            for timestamp in (item.issued_at, item.paid_at):
                if timestamp is not None:
                    fx.append({
                        "date": timestamp[:10], "source_currency": item.currency,
                        "target_currency": BASE_CURRENCY, "rate": item.conversion_rate,
                    })

    initial_history = {
        "opening_bank_minor": opening_balance,
        "obligations": [asdict(item) for item in obligations],
    }
    if distractors is not None:
        initial_history["distractor_operations"] = add_clean_distractors(
            invoices, payments,
            base_currency=BASE_CURRENCY,
            plan=distractors,
            new_id=ids.new,
            date_at=lambda offset: _timestamp(start + timedelta(days=offset)),
            post=partial(_post, ledger, ids),
        )

    counts = normalized_config["error_counts"]
    issues, changes = apply_allocation_errors(
        ledger, invoices, payments,
        base_currency=BASE_CURRENCY,
        duplicate_count=counts.get(FindingType.DUPLICATE_INVOICE, 0),
        allocation_count=counts.get(FindingType.PAYMENT_ALLOCATION, 0),
        rng=_rng(seed, "allocation-errors"),
        new_id=ids.new,
    )

    order = _rng(seed, "publication-order")
    for rows in (ledger, invoices, payments, fx):
        order.shuffle(rows)
    policy = {
        "fx_date_rule": "payment_date", "timezone": "Asia/Almaty",
        "period_start": _timestamp(start, hour=0),
        "period_end": _timestamp(end, hour=0),
        "rounding_mode": ROUND_HALF_EVEN,
        "recognition_date_rule": "document_date",
    }
    manifest = {
        "schema_version": SCHEMA_VERSION, "episode_id": episode_id,
        "base_currency": BASE_CURRENCY, "currency_exponents": EXPONENTS.copy(),
        "record_counts": {
            "ledger": len(ledger), "invoices": len(invoices),
            "payments": len(payments), "fx": len(fx),
        },
        "document_count": len(invoices) + len(payments),
        **{key: policy[key] for key in ("timezone", "period_start", "period_end", "rounding_mode")},
    }
    public = PublicEpisode(manifest, policy, ledger, invoices, payments, fx)
    if counts.get(FindingType.FX_CONVERSION) or counts.get(FindingType.PERIOD_CUTOFF):
        public.ledger, fx_cutoff_issues, fx_cutoff_changes = apply_fx_cutoff_errors(public, counts, ids.new)
        issues.extend(fx_cutoff_issues)
        changes.extend(fx_cutoff_changes)
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
