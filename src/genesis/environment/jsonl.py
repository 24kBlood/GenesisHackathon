"""Bounded UTF-8 JSONL on a policy process's standard streams."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Awaitable
from contextlib import suppress
from enum import StrEnum
from time import monotonic
from typing import TypeVar


FRAME_BYTES = 512 * 1024
STDERR_BYTES = 64 * 1024
STDERR_LINE_BYTES = 8 * 1024
_T = TypeVar("_T")


class TransportCode(StrEnum):
    TIMEOUT = "timeout"
    EOF = "eof"
    FRAME_LIMIT = "frame_limit"
    BAD_FRAME = "bad_frame"
    STDERR_LIMIT = "stderr_limit"
    PROCESS_ERROR = "process_error"


class TransportError(Exception):
    def __init__(self, code: TransportCode):
        self.code = code
        super().__init__(code.value)


async def before_deadline(operation: Awaitable[_T], deadline: float) -> _T:
    try:
        return await asyncio.wait_for(operation, timeout=max(0, deadline - monotonic()))
    except TimeoutError:
        raise TransportError(TransportCode.TIMEOUT) from None


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate JSON object member")
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError("non-finite JSON number")


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite JSON number")
    return result


class JsonlProcess:
    """Read/write one policy subprocess and drain diagnostics independently."""

    def __init__(self, process: asyncio.subprocess.Process):
        self._process = process
        assert process.stdin is not None and process.stdout is not None and process.stderr is not None
        self._stdin, self._stdout, self._stderr = process.stdin, process.stdout, process.stderr
        self._diagnostics = asyncio.create_task(self._drain_stderr())

    async def send(self, message: dict[str, object], deadline: float) -> None:
        frame = (json.dumps(message, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
        if len(frame) > FRAME_BYTES:
            raise TransportError(TransportCode.FRAME_LIMIT)
        if monotonic() >= deadline:
            raise TransportError(TransportCode.TIMEOUT)
        try:
            self._stdin.write(frame)
            await self._race_diagnostics(self._stdin.drain(), deadline)
        except OSError:
            raise TransportError(TransportCode.EOF) from None

    async def receive(self, deadline: float) -> object:
        try:
            frame = await self._race_diagnostics(self._stdout.readuntil(b"\n"), deadline)
        except asyncio.LimitOverrunError:
            raise TransportError(TransportCode.FRAME_LIMIT) from None
        except asyncio.IncompleteReadError as error:
            code = TransportCode.BAD_FRAME if error.partial else TransportCode.EOF
            raise TransportError(code) from None
        except OSError:
            raise TransportError(TransportCode.EOF) from None
        if len(frame) > FRAME_BYTES:
            raise TransportError(TransportCode.FRAME_LIMIT)
        try:
            return json.loads(
                frame.decode("utf-8"), object_pairs_hook=_object,
                parse_constant=_invalid_constant, parse_float=_finite_float,
            )
        except (ValueError, RecursionError):
            raise TransportError(TransportCode.BAD_FRAME) from None

    async def _race_diagnostics(self, operation: Awaitable[_T], deadline: float) -> _T:
        task = asyncio.ensure_future(operation)
        try:
            await asyncio.wait(
                (task, self._diagnostics), timeout=max(0, deadline - monotonic()),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if self._diagnostics.done():
                self._diagnostics.result()
            return await before_deadline(task, deadline) if not task.done() else task.result()
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            elif not task.cancelled():
                task.exception()

    async def _drain_stderr(self) -> None:
        total = line_size = 0
        while chunk := await self._stderr.read(4096):
            total += len(chunk)
            parts = chunk.split(b"\n")
            if (
                total > STDERR_BYTES or line_size + len(parts[0]) > STDERR_LINE_BYTES
                or any(len(part) > STDERR_LINE_BYTES for part in parts[1:])
            ):
                raise TransportError(TransportCode.STDERR_LIMIT)
            line_size = line_size + len(parts[0]) if len(parts) == 1 else len(parts[-1])

    async def close(self) -> None:
        """Signal EOF, allow a bounded exit, then reap the policy process."""
        self._stdin.close()
        await self._discard_and_reap()

    async def _discard_and_reap(self) -> None:
        async def discard(stream: asyncio.StreamReader) -> None:
            while await stream.read(4096):
                pass

        output = asyncio.create_task(discard(self._stdout))
        try:
            async with asyncio.timeout(0.25):
                await self._process.wait()
        except TimeoutError:
            with suppress(ProcessLookupError):
                self._process.kill()
        finally:
            if not self._diagnostics.done():
                self._diagnostics.cancel()
            await asyncio.gather(self._diagnostics, return_exceptions=True)
            error_output = asyncio.create_task(discard(self._stderr))
            with suppress(TimeoutError):
                await asyncio.wait_for(self._process.wait(), timeout=2)
            for task in (output, error_output):
                task.cancel()
            await asyncio.gather(output, error_output, return_exceptions=True)
