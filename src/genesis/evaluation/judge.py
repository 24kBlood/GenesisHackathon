"""Run a policy through the public protocol and keep the private result."""

from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from time import monotonic
from typing import Protocol

from genesis.environment.auditor_env import AuditorEnv, TerminalResult
from genesis.environment.jsonl import TransportCode, TransportError


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


async def run_episode(
    env: AuditorEnv, *, seed: int, config: Mapping[str, object], runner: PolicyRunner,
) -> JudgeResult:
    """Execute one episode. Score closed submissions only after policy teardown."""
    observation, _ = env.reset(seed, config)
    deadline = monotonic() + env.seconds_remaining
    fault = None
    try:
        async with runner.open(deadline) as channel:
            await channel.send({"type": "reset", "observation": observation, "info": {}}, deadline)
            while True:
                action = await channel.receive(deadline)
                observation, reward, terminated, truncated, _ = env.step(action)
                await channel.send({
                    "type": "step", "observation": observation, "reward": reward,
                    "terminated": terminated, "truncated": truncated, "info": {},
                }, deadline)
                if terminated or truncated:
                    break
    except TransportError as error:
        fault = error.code
        env.truncate(reason="timeout" if fault is TransportCode.TIMEOUT else "transport")
    terminal = env.terminal_result
    assert terminal is not None
    return JudgeResult(terminal, fault)
