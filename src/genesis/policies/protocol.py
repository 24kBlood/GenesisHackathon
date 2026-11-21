"""The public interface shared by local and JSONL policy drivers."""

from collections.abc import Mapping, Sequence
from typing import Protocol

from genesis.domain.models import Finding


Action = dict[str, object]
Observation = Mapping[str, object]


class Policy(Protocol):
    """Fresh instance for one episode. No trusted inputs are supplied."""

    def act(self, observation: Observation, previous_action: Action | None) -> Action: ...


def submit(episode_id: str, findings: Sequence[Finding] = ()) -> Action:
    return {"type": "submit", "findings": {"episode_id": episode_id, "findings": list(findings)}}


class PolicySession:
    """Dispatch the two public messages to one policy without evaluator data."""

    def __init__(self, policy: Policy):
        self._policy = policy
        self._previous: Action | None = None
        self._started = False
        self._finished = False

    def respond(self, message: Mapping[str, object]) -> Action | None:
        kind = message.get("type")
        observation = message.get("observation")
        if self._finished or not isinstance(observation, Mapping):
            raise ValueError("invalid policy message")
        if kind == "reset" and not self._started:
            self._started = True
        elif kind == "step" and self._started:
            if message.get("terminated") is True or message.get("truncated") is True:
                self._finished = True
                return None
        else:
            raise ValueError("unexpected policy message")
        action = self._policy.act(observation, self._previous)
        self._previous = action
        return action
