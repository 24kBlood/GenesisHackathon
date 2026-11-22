"""Run a policy through the public protocol and keep the private result."""

from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from time import monotonic
from typing import Protocol

from genesis.environment.auditor_env import AuditorEnv, StepResult, TerminalResult
from genesis.environment.episode import build_episode
from genesis.environment.jsonl import TransportCode, TransportError
from genesis.generation import generate_closed_episode

from .admission import require_admissible_episode


class PolicyChannel(Protocol):
    async def send(self, message: dict[str, object], deadline: float) -> None: ...

    async def receive(self, deadline: float) -> object: ...


class PolicyRunner(Protocol):
    # The runner is a trusted transport factory, distinct from worker code.
    def open(self, deadline: float) -> AbstractAsyncContextManager[PolicyChannel]: ...


@dataclass(frozen=True)
class JudgeResult:
    terminal: TerminalResult
    transport_error: TransportCode | None


StepObserver = Callable[[str, int, object, StepResult], None]


def closed_environment(*, clock: Callable[[], float] = monotonic) -> AuditorEnv:
    """Build an eval environment for cases provisioned by the trusted coordinator."""
    def factory(seed, config):
        episode = generate_closed_episode(seed, config)
        require_admissible_episode(episode)
        public = episode.public
        return build_episode(
            task="Проверьте открытые записи эпизода.",
            manifest=public.manifest,
            policy=public.policy,
            tables={name: getattr(public, name) for name in ("ledger", "invoices", "payments", "fx")},
            truth=episode.evaluator.truth,
        )

    return AuditorEnv(factory, mode="eval", clock=clock)


async def run_episode(
    env: AuditorEnv, *, seed: int, config: Mapping[str, object], runner: PolicyRunner,
    on_step: StepObserver | None = None,
) -> JudgeResult:
    """Execute one episode. Score closed submissions only after policy teardown."""
    observation, _ = env.reset(seed, config)
    episode_id = observation["manifest"]["episode_id"]
    remaining = observation["remaining_steps"]
    accepted = 0
    deadline = monotonic() + env.seconds_remaining
    fault = None
    try:
        async with runner.open(deadline) as channel:
            env.begin_policy_interaction()
            deadline = monotonic() + env.seconds_remaining
            await channel.send({"type": "reset", "observation": observation, "info": {}}, deadline)
            while True:
                action = await channel.receive(deadline)
                transition = env.step(action)
                observation, reward, terminated, truncated, _ = transition
                consumed = observation["remaining_steps"] < remaining
                if consumed:
                    accepted += 1
                    remaining = observation["remaining_steps"]
                try:
                    await channel.send({
                        "type": "step", "observation": observation, "reward": reward,
                        "terminated": terminated, "truncated": truncated, "info": {},
                    }, deadline)
                except TransportError as error:
                    # The worker may exit immediately after submitting or reaching the step limit.
                    if not (terminated or truncated) or error.code not in (TransportCode.EOF, TransportCode.TIMEOUT):
                        raise
                finally:
                    if consumed and on_step is not None:
                        recorded_action = {"type": "invalid"} if env.last_action_failed else action
                        on_step(episode_id, accepted, recorded_action, transition)
                if terminated or truncated:
                    break
    except TransportError as error:
        # Infrastructure setup never reaches the policy. It is a coordinator
        # failure, not a graded episode outcome, regardless of its transport code.
        if error.phase == "setup" or error.code is TransportCode.PROCESS_ERROR:
            raise
        fault = error.code
        env.truncate(reason="timeout" if fault is TransportCode.TIMEOUT else "transport")
    terminal = env.terminal_result
    assert terminal is not None
    return JudgeResult(terminal, fault)
