"""JSON-facing lifecycle. Only the trusted coordinator owns this instance."""

import json
import math
import re
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from time import monotonic
from typing import Literal, Protocol, cast

from genesis.domain.enums import FindingType
from genesis.domain.models import FindingsSubmission
from genesis.evaluation.matcher import MatchResult
from genesis.evaluation.reward import TerminalScore
from genesis.evaluation.terminal import TerminalEvaluation


MAX_STEPS = 100
MAX_ID_LENGTH = 128
MAX_AFFECTED_IDS = 64
MAX_EVIDENCE_IDS = 10
MAX_FINDINGS = 30
MAX_EXPLANATION_LENGTH = 1000
MAX_CLAIMED_AMOUNT_MINOR = 10**18
_TABLES = ("ledger", "invoices", "payments", "fx")
_FINDING_TYPES = {kind.value for kind in FindingType}
_FINDING_KEYS = {"type", "affected_ids", "claimed_amount_minor", "currency", "evidence_ids", "explanation"}
_FX_DATE_RULES = frozenset({"payment_date", "invoice_date", "posting_date"})
_ROUNDING_MODES = frozenset({
    "ROUND_CEILING", "ROUND_DOWN", "ROUND_FLOOR", "ROUND_HALF_DOWN",
    "ROUND_HALF_EVEN", "ROUND_HALF_UP", "ROUND_UP", "ROUND_05UP",
})
_CURRENCY = re.compile(r"[A-Z]{3}")
_TOOLS = {"list_records", "get_record", "get_document", "calculator"}
_MALFORMED_ACTION_CODES = frozenset({"invalid_action", "invalid_argument"})
_RESULT_KEYS = {
    "list_records": "records", "get_record": "record",
    "get_document": "document", "calculator": "result",
}
_PERIOD_FIELDS = ("timezone", "period_start", "period_end", "rounding_mode")
_MANIFEST_FIELDS = (
    "schema_version", "episode_id", "base_currency", *_PERIOD_FIELDS, "document_count",
)
_RECORD_FIELDS = {
    "row_id": (
        "row_id", "timestamp", "account", "debit_minor", "credit_minor", "currency",
        "entity_id", "invoice_id", "payment_id",
    ),
    "invoice_id": ("doc_id", "invoice_id", "entity_id", "issued_at", "due_at", "currency", "total_minor"),
    "payment_id": ("doc_id", "payment_id", "timestamp", "currency", "amount_minor", "invoice_refs"),
    "date": ("date", "source_currency", "target_currency", "rate"),
    "fx_date_rule": ("fx_date_rule", *_PERIOD_FIELDS),
}
_INTEGER_FIELDS = {
    "document_count", "debit_minor", "credit_minor", "total_minor", "amount_minor",
    "unit_price_minor", *_TABLES,
}
_NON_NEGATIVE_INTEGER_FIELDS = {"document_count", *_TABLES}
_ID_FIELDS = {"episode_id", "entity_id", "row_id", "doc_id", "invoice_id", "payment_id"}

Observation = dict[str, object]
PublicAction = dict[str, object]
Handler = Callable[[PublicAction], Mapping[str, object]]
StepResult = tuple[Observation, float, bool, bool, dict[str, object]]
TruncationReason = Literal["step_limit", "timeout", "transport"]
TerminalReason = Literal["submitted", "invalid_submission", "step_limit", "timeout", "transport"]


class ToolCode(StrEnum):
    NOT_FOUND = "not_found"
    INVALID_CURSOR = "invalid_cursor"
    INVALID_ARGUMENT = "invalid_argument"
    INVALID_EXPRESSION = "invalid_expression"
    DIVISION_BY_ZERO = "division_by_zero"
    CALCULATION_LIMIT = "calculation_limit"


class ToolError(Exception):
    """A handler can report a specific public error via a fixed code."""

    def __init__(self, code: ToolCode):
        self.code = code
        super().__init__(code.value)


class TerminalEvaluator(Protocol):
    def __call__(
        self, submission: FindingsSubmission, inspection_steps: int, *, invalid_submission: bool,
    ) -> TerminalEvaluation | TerminalScore: ...


@dataclass(frozen=True)
class Episode:
    """Trusted factory output. The handler and its cursor catalog belong to this episode."""

    task: str
    manifest: Mapping[str, object]
    policy: Mapping[str, object]
    handle_action: Handler
    evaluate: TerminalEvaluator


EpisodeFactory = Callable[[int, Mapping[str, object]], Episode]


@dataclass(frozen=True)
class TerminalResult:
    """Trusted report. The policy transport must never serialize this object."""

    submission: FindingsSubmission
    score: TerminalScore
    invalid_submission: bool
    terminated: bool
    truncated: bool
    matching: MatchResult | None
    total_steps: int
    wall_time_seconds: float
    reason: TerminalReason
    validation_error: str | None = None


@dataclass(frozen=True)
class _Completion:
    submission: FindingsSubmission
    invalid_submission: bool
    terminated: bool
    reason: TerminalReason
    wall_time_seconds: float
    validation_error: str | None = None

    @property
    def status(self) -> str:
        return "submitted" if self.terminated else "truncated"


def _valid_text(value: object) -> bool:
    if type(value) is not str:
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _is_id(value: object) -> bool:
    return _valid_text(value) and 1 <= len(value) <= MAX_ID_LENGTH


def _ids(
    value: object, *, nonempty: bool = False, maximum: int | None = None, unique: bool = False,
) -> bool:
    return (
        type(value) is list
        and (not nonempty or bool(value))
        and (maximum is None or len(value) <= maximum)
        and all(_is_id(item) for item in value)
        and (not unique or len(set(value)) == len(value))
    )


def _validate_finding(value: object, index: int) -> str | None:
    if type(value) is not dict or value.keys() != _FINDING_KEYS:
        return f"finding[{index}]: must be an object with required finding fields"
    if not isinstance(value["type"], str) or value["type"] not in _FINDING_TYPES:
        return f"finding[{index}]: invalid finding type"
    if not _ids(value["affected_ids"], nonempty=True, maximum=MAX_AFFECTED_IDS, unique=True):
        return f"finding[{index}]: affected_ids must be 1..{MAX_AFFECTED_IDS} unique valid IDs"
    amount = value["claimed_amount_minor"]
    if type(amount) is not int or not 0 <= amount <= MAX_CLAIMED_AMOUNT_MINOR:
        return f"finding[{index}]: claimed_amount_minor must be an integer in 0..{MAX_CLAIMED_AMOUNT_MINOR}"
    currency = value["currency"]
    if not _valid_text(currency) or _CURRENCY.fullmatch(currency) is None:
        return f"finding[{index}]: invalid currency code"
    if not _ids(value["evidence_ids"], maximum=MAX_EVIDENCE_IDS):
        return f"finding[{index}]: evidence_ids must be at most {MAX_EVIDENCE_IDS} valid IDs"
    explanation = value["explanation"]
    if not _valid_text(explanation) or len(explanation) > MAX_EXPLANATION_LENGTH:
        return f"finding[{index}]: explanation must be valid UTF-8 text of at most {MAX_EXPLANATION_LENGTH} chars"
    return None


def _validate_submission(action: dict, episode_id: str) -> tuple[FindingsSubmission | None, str | None]:
    if action.keys() != {"type", "findings"}:
        return None, "submit action must contain only type and findings"
    payload = action.get("findings")
    if type(payload) is not dict or payload.keys() != {"episode_id", "findings"}:
        return None, "findings payload must be an object with episode_id and findings"
    if not _valid_text(payload["episode_id"]) or payload["episode_id"] != episode_id:
        return None, "submission episode_id does not match active episode"
    findings = payload["findings"]
    if type(findings) is not list or len(findings) > MAX_FINDINGS:
        return None, f"findings must be a list of at most {MAX_FINDINGS} items"
    for index, finding in enumerate(findings):
        error = _validate_finding(finding, index)
        if error is not None:
            return None, error
    return cast(FindingsSubmission, deepcopy(payload)), None


def _validated_submission(action: dict, episode_id: str) -> FindingsSubmission | None:
    payload, _ = _validate_submission(action, episode_id)
    return payload


def _valid_tool_action(action: dict, kind: str) -> bool:
    keys = action.keys() - {"type"}
    if kind == "calculator":
        return keys == {"expression"} and _valid_text(action["expression"])
    if kind == "get_document":
        return keys == {"doc_id"} and _is_id(action["doc_id"])
    if action.get("table") not in _TABLES:
        return False
    if kind == "get_record":
        return keys == {"table", "id"} and _is_id(action["id"])
    return (
        keys <= {"table", "cursor", "limit"}
        and (action.get("cursor") is None or _is_id(action["cursor"]))
        and ("limit" not in action or type(action["limit"]) is int and 1 <= action["limit"] <= 50)
    )


def _field(key: str, value: object) -> object:
    if key in _NON_NEGATIVE_INTEGER_FIELDS:
        valid = type(value) is int and value >= 0
    elif key in _INTEGER_FIELDS:
        valid = type(value) is int
    elif key == "quantity":
        valid = type(value) in (int, float)
    elif key == "invoice_refs":
        valid = value is None or _ids(value)
    elif key in ("invoice_id", "payment_id") and value is None:
        valid = True  # Nullable ledger links.
    elif key in _ID_FIELDS:
        valid = _is_id(value)
    else:
        valid = _valid_text(value)
    if not valid:
        raise ValueError("invalid public field")
    return value


def _select(value: Mapping, fields: tuple[str, ...]) -> dict:
    return {key: _field(key, value[key]) for key in fields}


def _project_policy(value: Mapping) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("invalid public policy")
    policy = _select(value, ("fx_date_rule", *_PERIOD_FIELDS))
    if policy["fx_date_rule"] not in _FX_DATE_RULES:
        raise ValueError("invalid fx_date_rule")
    if policy["rounding_mode"] not in _ROUNDING_MODES:
        raise ValueError("invalid rounding_mode")
    if not policy["timezone"] or not policy["period_start"] or not policy["period_end"]:
        raise ValueError("invalid policy period or timezone")
    if "recognition_date_rule" in value:
        if value["recognition_date_rule"] != "document_date":
            raise ValueError("invalid recognition date rule")
        policy["recognition_date_rule"] = "document_date"
    return policy


def _project_record(value: Mapping, *, document: bool = False) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("invalid public record kind")
    # Ledger rows also have invoice/payment links, so test row_id first.
    discriminator = next((key for key in _RECORD_FIELDS if key in value), None)
    if discriminator is None:
        raise ValueError("invalid public record kind")
    if discriminator == "fx_date_rule" and document:
        return _project_policy(value)
    if discriminator == "fx_date_rule" or (document and discriminator in ("row_id", "date")):
        raise ValueError("invalid public record kind")
    record = _select(value, _RECORD_FIELDS[discriminator])
    if discriminator in ("invoice_id", "payment_id") and not _is_id(record[discriminator]):
        raise ValueError("invalid document identifier")
    if discriminator == "invoice_id":
        fields = ("description", "quantity", "unit_price_minor", "amount_minor")
        if type(value["line_items"]) is not list:
            raise ValueError("invalid line items")
        record["line_items"] = [
            _select(item, tuple(key for key in fields if key in item)) for item in value["line_items"]
        ]
    elif discriminator == "payment_id" and ("payment_kind" in value or "refund_of_payment_id" in value):
        if (
            value.get("payment_kind") != "refund" or not _is_id(value.get("refund_of_payment_id"))
            or record["amount_minor"] <= 0 or not record["invoice_refs"]
        ):
            raise ValueError("invalid refund")
        record.update(payment_kind="refund", refund_of_payment_id=value["refund_of_payment_id"])
    return record


def _copy_public(value: dict) -> dict:
    # The trusted adapters provide schema-typed fields; enforce JSON-compatible
    # values and detach nested containers before returning to the transport.
    return json.loads(json.dumps(value, allow_nan=False))


class AuditorEnv:
    """One session. Call reset/step serially from the trusted coordinator."""

    def __init__(
        self, episode_factory: EpisodeFactory, *, mode: Literal["train", "eval"] = "train",
        clock: Callable[[], float] = monotonic, timeout_seconds: float = 60,
    ):
        if mode not in ("train", "eval"):
            raise ValueError("invalid environment mode")
        if (
            type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive and finite")
        self._factory = episode_factory
        self._mode = mode
        self._clock = clock
        self._timeout = timeout_seconds
        self._episode: Episode | None = None
        self._completion: _Completion | None = None
        self._terminal_result: TerminalResult | None = None

    def reset(self, seed: int, config: Mapping[str, object]) -> tuple[Observation, dict[str, object]]:
        episode = self._factory(seed, deepcopy(dict(config)))
        manifest = _select(episode.manifest, _MANIFEST_FIELDS)
        if _CURRENCY.fullmatch(manifest["base_currency"]) is None:
            raise ValueError("invalid base_currency")
        if manifest["rounding_mode"] not in _ROUNDING_MODES:
            raise ValueError("invalid manifest rounding_mode")
        if not manifest["timezone"] or not manifest["period_start"] or not manifest["period_end"]:
            raise ValueError("invalid manifest period or timezone")
        raw_counts = episode.manifest["record_counts"]
        if not isinstance(raw_counts, Mapping):
            raise ValueError("invalid record_counts")
        manifest["record_counts"] = _select(raw_counts, _TABLES)
        raw_exponents = episode.manifest["currency_exponents"]
        if not isinstance(raw_exponents, Mapping) or not raw_exponents:
            raise ValueError("invalid currency_exponents")
        exponents: dict[str, int] = {}
        for key, value in raw_exponents.items():
            if not _valid_text(key) or _CURRENCY.fullmatch(key) is None or type(value) is not int or value < 0:
                raise ValueError("invalid currency_exponents")
            exponents[key] = value
        manifest["currency_exponents"] = exponents
        observation = _copy_public({
            "task": _field("task", episode.task), "manifest": manifest,
            "policy": _project_policy(episode.policy),
            "remaining_steps": MAX_STEPS,
        })
        started_at = self._clock()
        self._episode = episode
        self._episode_id = manifest["episode_id"]
        self._remaining = MAX_STEPS
        self._inspection_steps = 0
        self._cursors: set[tuple[str, str]] = set()
        self._started_at = started_at
        self._deadline = started_at + self._timeout
        self._completion = None
        self._terminal_result = None
        self._last_action_failed = False
        return observation, {}

    @property
    def terminal_result(self) -> TerminalResult | None:
        """Read once in the trusted coordinator after handing off the eval acknowledgement."""
        completion = self._completion
        if completion is None:
            return None
        if self._terminal_result is None:
            assert self._episode is not None
            evaluation = self._episode.evaluate(
                deepcopy(completion.submission), self._inspection_steps,
                invalid_submission=completion.invalid_submission,
            )
            if isinstance(evaluation, TerminalEvaluation):
                matching, score = evaluation.matching, evaluation.score
            else:
                matching, score = None, evaluation
            self._terminal_result = TerminalResult(
                submission=completion.submission,
                score=score,
                invalid_submission=completion.invalid_submission,
                terminated=completion.terminated,
                truncated=not completion.terminated,
                matching=matching,
                total_steps=MAX_STEPS - self._remaining,
                wall_time_seconds=completion.wall_time_seconds,
                reason=completion.reason,
                validation_error=completion.validation_error,
            )
        return deepcopy(self._terminal_result)

    @property
    def seconds_remaining(self) -> float:
        """Return the trusted coordinator's time budget after reset."""
        if self._episode is None:
            raise RuntimeError("reset is required before reading the deadline")
        return max(0, self._deadline - self._clock())

    def truncate(self, *, reason: TruncationReason = "transport") -> StepResult:
        """Close the episode after interruption of the external policy transport."""
        if self._episode is None:
            raise RuntimeError("reset is required before truncation")
        if self._completion is not None:
            return self._terminal_reply(0.0)
        return self._finish(None, terminated=False, reason=reason)

    @property
    def last_action_failed(self) -> bool:
        """Trusted recorder metadata, including errors masked by terminal status."""
        return self._last_action_failed

    def step(self, action: object) -> StepResult:
        if self._episode is None:
            raise RuntimeError("reset is required before step")
        if self._completion is not None:
            # A late action receives flags and status only. Reward was emitted once.
            return self._terminal_reply(0.0)
        if self._clock() >= self._deadline:
            self._last_action_failed = False
            return self._finish(None, terminated=False, reason="timeout")

        self._remaining -= 1
        kind = action.get("type") if type(action) is dict else None
        if type(kind) is str and kind == "submit":
            submission, validation_error = _validate_submission(action, self._episode_id)
            self._last_action_failed = submission is None
            return self._finish(submission, terminated=True, validation_error=validation_error)

        self._inspection_steps += 1
        observation = self._research(action, kind)
        error = observation.get("error")
        self._last_action_failed = (
            isinstance(error, Mapping) and error.get("code") in _MALFORMED_ACTION_CODES
        )
        if self._remaining == 0:
            return self._finish(None, terminated=False, reason="step_limit")
        return {"remaining_steps": self._remaining, **observation}, 0.0, False, False, {}

    def _research(self, action: object, kind: object) -> dict:
        if type(kind) is not str or kind not in _TOOLS:
            return {"error": {"code": "invalid_action"}}
        if not _valid_tool_action(action, kind):
            return {"error": {"code": ToolCode.INVALID_ARGUMENT.value}}
        if kind == "list_records" and action.get("cursor") is not None:
            if (action["table"], action["cursor"]) not in self._cursors:
                return {"error": {"code": ToolCode.INVALID_CURSOR.value}}

        assert self._episode is not None
        try:
            source = self._episode.handle_action(dict(action))
            key = _RESULT_KEYS[kind]
            if kind == "list_records":
                rows = source[key]
                cursor = source.get("cursor")
                if type(rows) is not list or len(rows) > action.get("limit", 50):
                    raise ValueError("invalid page")
                if cursor is not None and not _is_id(cursor):
                    raise ValueError("invalid cursor")
                result = _copy_public({"records": [_project_record(row) for row in rows], "cursor": cursor})
                if cursor is not None:
                    self._cursors.add((action["table"], cursor))
                return result
            value = source[key]
            if kind == "calculator":
                if type(value) is not str:
                    raise ValueError("invalid calculator result")
                return {"result": value}
            return _copy_public({key: _project_record(value, document=kind == "get_document")})
        except ToolError as error:
            code = error.code.value if isinstance(error.code, ToolCode) else "tool_error"
        except Exception:
            # A user-observable response must never include the adapter exception.
            code = "tool_error"
        return {"error": {"code": code}}

    def _finish(
        self, submission: FindingsSubmission | None, *, terminated: bool,
        reason: TruncationReason | None = None,
        validation_error: str | None = None,
    ) -> StepResult:
        empty: FindingsSubmission = {"episode_id": self._episode_id, "findings": []}
        completion_reason: TerminalReason
        if terminated:
            completion_reason = "submitted" if submission is not None else "invalid_submission"
        else:
            completion_reason = reason or "transport"
        self._completion = _Completion(
            submission=submission if submission is not None else empty,
            invalid_submission=terminated and submission is None,
            terminated=terminated,
            reason=completion_reason,
            wall_time_seconds=max(0.0, self._clock() - self._started_at),
            validation_error=validation_error if (terminated and submission is None) else None,
        )
        if self._mode == "train":
            result = self.terminal_result
            assert result is not None
            return self._terminal_reply(result.score.reward)
        return self._terminal_reply(0.0)

    def _terminal_reply(self, reward: float) -> StepResult:
        completion = self._completion
        assert completion is not None
        return (
            {"remaining_steps": self._remaining, "status": completion.status},
            reward, completion.terminated, not completion.terminated, {},
        )
