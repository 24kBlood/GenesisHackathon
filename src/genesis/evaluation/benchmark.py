"""Trusted diagnostic runner for consistent episode and policy comparisons."""

from collections.abc import Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass
from time import monotonic

from genesis.environment.auditor_env import AuditorEnv
from genesis.environment.jsonl import TransportCode, TransportError
from genesis.policies.protocol import Action, Policy, PolicySession

from .judge import PolicyRunner, run_episode


class _LocalChannel:
    def __init__(self, policy: Policy, actions: list[Action]):
        self._session = PolicySession(policy)
        self._actions = actions
        self._next: Action | None = None

    async def send(self, message: dict[str, object], deadline: float) -> None:
        if monotonic() >= deadline:
            raise TransportError(TransportCode.TIMEOUT)
        self._next = self._session.respond(deepcopy(message))

    async def receive(self, deadline: float) -> Action:
        if monotonic() >= deadline:
            raise TransportError(TransportCode.TIMEOUT)
        if self._next is None:
            raise TransportError(TransportCode.EOF)
        action = deepcopy(self._next)
        self._actions.append(deepcopy(action))
        self._next = None
        return action


class LocalPolicyRunner:
    """In-process driver strictly for trusted built-ins and test fixtures."""

    def __init__(self, make_policy: Callable[[], Policy]):
        self._make_policy = make_policy
        self._actions: list[Action] = []

    @property
    def actions(self) -> tuple[Action, ...]:
        return tuple(deepcopy(self._actions))

    @asynccontextmanager
    async def open(self, deadline: float):
        self._actions.clear()
        yield _LocalChannel(self._make_policy(), self._actions)


@dataclass(frozen=True)
class BenchmarkEpisode:
    name: str
    seed: int
    config: Mapping[str, object]


@dataclass(frozen=True)
class PolicyDiagnostic:
    case: str
    episode_id: str
    policy: str
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    reward: float
    total_steps: int
    inspection_steps: int
    reason: str
    transport_error: TransportCode | None


@dataclass(frozen=True)
class BenchmarkResult:
    diagnostics: tuple[PolicyDiagnostic, ...]
    wall_time_seconds: tuple[float, ...]


async def run_benchmark(
    make_env: Callable[[], AuditorEnv],
    episodes: Sequence[BenchmarkEpisode],
    runners: Mapping[str, Callable[[BenchmarkEpisode], PolicyRunner]],
) -> BenchmarkResult:
    """Run episode-major and policy-insertion order. Keep timing out of diagnostics."""
    diagnostics: list[PolicyDiagnostic] = []
    wall_time_seconds: list[float] = []
    for episode in episodes:
        for name, make_runner in runners.items():
            result = await run_episode(
                make_env(), seed=episode.seed, config=episode.config, runner=make_runner(episode),
            )
            terminal = result.terminal
            matching = terminal.matching
            if matching is None:
                raise ValueError("benchmark requires the trusted matching adapter")
            metrics = terminal.score.metrics
            diagnostics.append(PolicyDiagnostic(
                case=episode.name, episode_id=terminal.submission["episode_id"], policy=name,
                tp=matching.tp, fp=matching.fp, fn=matching.fn, precision=metrics.precision,
                recall=metrics.recall, f1=metrics.f1, reward=terminal.score.reward,
                total_steps=terminal.total_steps, inspection_steps=terminal.score.inspection_steps,
                reason=terminal.reason, transport_error=result.transport_error,
            ))
            wall_time_seconds.append(terminal.wall_time_seconds)
    return BenchmarkResult(tuple(diagnostics), tuple(wall_time_seconds))
