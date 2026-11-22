"""Allowlisted v1 trajectory recording and sequence checks."""

import json
import re
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from typing import Literal

from genesis.environment.auditor_env import (
    MAX_STEPS,
    StepResult,
    ToolCode,
    _RESULT_KEYS,
    _project_record,
    _valid_tool_action,
    _validated_submission,
)
from genesis.environment.calculator import MAX_RESULT_LENGTH
from genesis.environment.jsonl import FRAME_BYTES


_TOOLS = {"list_records", "get_record", "get_document", "calculator"}
_PAYLOADS = ("records", "record", "document", "result", "error", "status")
_ERROR_CODES = {"invalid_action", "tool_error", *(value.value for value in ToolCode)}
_TOOL_ERROR_CODES = {
    "list_records": frozenset({ToolCode.INVALID_CURSOR.value, "tool_error"}),
    "get_record": frozenset({ToolCode.NOT_FOUND.value, "tool_error"}),
    "get_document": frozenset({ToolCode.NOT_FOUND.value, "tool_error"}),
    "calculator": frozenset({
        ToolCode.INVALID_EXPRESSION.value,
        ToolCode.DIVISION_BY_ZERO.value,
        ToolCode.CALCULATION_LIMIT.value,
        "tool_error",
    }),
}
_PRIVATE_FIELDS = {
    "seed", "truth", "provenance", "initial_history", "changes", "issues",
    "expected_amount_minor", "acceptable_evidence_sets", "info", "stderr", "traceback",
    "wall_time_seconds", "transport_error",
}
_FIELDS = {"episode_id", "step", "action", "observation", "reward", "terminated", "truncated"}
_DECIMAL_RESULT = re.compile(r"-?[0-9]+(?:\.[0-9]+)?")
_INVALID = {"type": "invalid"}

Mode = Literal["train", "eval"]
TrajectoryStep = dict[str, object]


def _mode(mode: str) -> None:
    if mode not in ("train", "eval"):
        raise ValueError("invalid trajectory mode")


def _project_observation(source: Mapping[str, object]) -> dict[str, object]:
    values = [name for name in _PAYLOADS if name in source]
    if len(values) != 1:
        raise ValueError("trajectory requires a step observation")
    name, = values
    value = source[name]
    observation = {"remaining_steps": source["remaining_steps"]}
    if name == "records":
        if type(value) is not list or len(value) > 50:
            raise ValueError("invalid public page")
        cursor = source.get("cursor")
        if cursor is not None and (type(cursor) is not str or not cursor):
            raise ValueError("invalid public cursor")
        observation.update(records=[_project_record(record) for record in value], cursor=cursor)
    elif name in ("record", "document"):
        observation[name] = _project_record(value, document=name == "document")
    elif name == "result":
        if (
            type(value) is not str
            or len(value) > MAX_RESULT_LENGTH
            or _DECIMAL_RESULT.fullmatch(value) is None
        ):
            raise ValueError("invalid public calculator result")
        observation[name] = value
    elif name == "error":
        code = value.get("code") if isinstance(value, Mapping) else None
        if type(code) is not str or code not in _ERROR_CODES:
            raise ValueError("unknown public error code")
        observation[name] = {"code": code}
    else:
        if value not in ("submitted", "truncated"):
            raise ValueError("unknown public status")
        observation[name] = value
    return observation


def _project_action(
    action: object, episode_id: str, observation: Mapping[str, object], *, mode: Mode = "eval",
) -> dict:
    if type(action) is not dict:
        return _INVALID.copy()
    error = observation.get("error")
    error_code = error.get("code") if isinstance(error, Mapping) else None
    kind = action.get("type")
    if kind == "submit":
        if error_code is None:
            payload = _validated_submission(action, episode_id)
            if payload is not None:
                if mode == "eval":
                    # Closed eval exports blank free-text explanations; open train mode preserves them.
                    for finding in payload["findings"]:
                        finding["explanation"] = ""
                return {"type": "submit", "findings": payload}
    elif type(kind) is str and kind in _TOOLS and _valid_tool_action(action, kind):
        if error_code is None or error_code in _TOOL_ERROR_CODES[kind]:
            return deepcopy(action)
    return _INVALID.copy()


def _reject_private(value: object) -> None:
    if isinstance(value, Mapping):
        if any(type(key) is not str or key.casefold() in _PRIVATE_FIELDS for key in value):
            raise ValueError("private or non-string field in trajectory")
        for item in value.values():
            _reject_private(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_private(item)


def _encode(row: Mapping[str, object]) -> bytes:
    data = (json.dumps(
        row, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"),
    ) + "\n").encode("utf-8")
    if len(data) > 2 * FRAME_BYTES:
        raise ValueError("trajectory row exceeds frame budget")
    return data


def validate_trajectory(
    records: Iterable[Mapping[str, object]], *, mode: Mode = "eval",
    validate_shape: Callable[[Mapping[str, object]], None] | None = None,
) -> None:
    """Check public shape, order and terminal semantics. Accept partial prefixes.

    With `validate_shape`, each row is also checked by the caller's local v1
    schema registry. A timeout before accepting another action adds no row.
    """
    _mode(mode)
    seen: set[str] = set()
    episode_id = None
    step = 0
    finished = False
    for row in records:
        if not isinstance(row, Mapping):
            raise ValueError("invalid trajectory row")
        _reject_private(row)
        if row.keys() != _FIELDS:
            raise ValueError("invalid trajectory fields")
        row_episode = row["episode_id"]
        if type(row_episode) is not str or not row_episode:
            raise ValueError("invalid trajectory episode ID")
        if row_episode != episode_id:
            if row_episode in seen:
                raise ValueError("trajectory episode is not contiguous")
            seen.add(row_episode)
            episode_id, step, finished = row_episode, 0, False
        step += 1
        if finished or step > MAX_STEPS or type(row["step"]) is not int or row["step"] != step:
            raise ValueError("invalid trajectory step sequence")
        observation = row["observation"]
        if (
            not isinstance(observation, Mapping)
            or type(observation.get("remaining_steps")) is not int
            or observation["remaining_steps"] != MAX_STEPS - step
            or observation != _project_observation(observation)
        ):
            raise ValueError("invalid trajectory observation")
        action = row["action"]
        if action != _project_action(action, episode_id, observation, mode=mode):
            raise ValueError("invalid trajectory action")
        terminated, truncated, reward = row["terminated"], row["truncated"], row["reward"]
        if (
            type(terminated) is not bool or type(truncated) is not bool or terminated and truncated
            or type(reward) not in (int, float) or not -1 <= reward <= 1
        ):
            raise ValueError("invalid trajectory terminal fields")
        finished = terminated or truncated
        if (mode == "eval" or not finished) and reward != 0:
            raise ValueError("trajectory reward is private or non-terminal")
        expected_status = "submitted" if terminated else "truncated" if truncated else None
        if observation.get("status") != expected_status or (step == MAX_STEPS and not finished):
            raise ValueError("trajectory status disagrees with terminal flags")
        if truncated and step != MAX_STEPS:
            raise ValueError("only an accepted budget-ending action records truncation")
        kind = action["type"]
        if (kind == "submit" and not terminated) or (terminated and kind not in ("submit", "invalid")):
            raise ValueError("trajectory submission disagrees with terminal flags")
        if not finished:
            if kind == "invalid":
                if "error" not in observation:
                    raise ValueError("trajectory action disagrees with the observation")
            else:
                error = observation.get("error")
                error_code = error.get("code") if isinstance(error, Mapping) else None
                if _RESULT_KEYS[kind] not in observation and error_code not in _TOOL_ERROR_CODES[kind]:
                    raise ValueError("trajectory action disagrees with the observation")
        _encode(row)
        if validate_shape is not None:
            validate_shape(row)


class TrajectoryRecorder:
    """An observer attached to one run of an ordinary policy on public data."""

    def __init__(self, *, mode: Mode = "eval"):
        _mode(mode)
        self._mode = mode
        self._records: list[TrajectoryStep] = []

    @property
    def records(self) -> tuple[TrajectoryStep, ...]:
        return tuple(deepcopy(self._records))

    def record(self, episode_id: str, step: int, action: object, result: StepResult) -> None:
        if step != len(self._records) + 1 or not 1 <= step <= MAX_STEPS:
            raise ValueError("invalid recording step")
        if self._records and self._records[0]["episode_id"] != episode_id:
            raise ValueError("one recorder belongs to one episode")
        observation, reward, terminated, truncated, _ = result
        public = _project_observation(observation)
        self._records.append({
            "episode_id": episode_id, "step": step,
            "action": _project_action(action, episode_id, public, mode=self._mode),
            "observation": public, "reward": reward if self._mode == "train" else 0.0,
            "terminated": terminated, "truncated": truncated,
        })

    def to_jsonl(self) -> bytes:
        validate_trajectory(self._records, mode=self._mode)
        return b"".join(map(_encode, self._records))
