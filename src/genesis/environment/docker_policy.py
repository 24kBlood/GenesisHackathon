"""Production runner for a trusted, locally staged policy image."""

import asyncio
import os
import re
import shutil
from contextlib import asynccontextmanager, suppress
from time import monotonic
from uuid import uuid4

from .jsonl import FRAME_BYTES, JsonlProcess, TransportCode, TransportError, before_deadline


_IMAGE_REFERENCE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._:/@-]{0,255}")
_IMAGE_ID = re.compile(rb"sha256:[a-f0-9]{64}")
_DAEMON_RESPONSE_BYTES = 512
_CLEANUP_SECONDS = 5


class PolicyCleanupError(RuntimeError):
    """Container teardown could not be confirmed. Closed scoring must stop."""


def _host_environment() -> dict[str, str]:
    return {key: os.environ[key] for key in ("PATH", "SYSTEMROOT", "WINDIR") if key in os.environ}


def _create_arguments(name: str, image_id: str) -> tuple[str, ...]:
    return (
        "create", "--name", name, "--label", "genesis.policy=true", "--pull", "never",
        "--interactive", "--network", "none", "--ipc", "none", "--read-only",
        "--user", "65532:65532", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--pids-limit", "32", "--memory", "512m", "--memory-swap", "512m",
        "--cpus", "1", "--log-driver", "none", image_id,
    )


class DockerPolicy:
    """Own the lifetime of one container and expose only its attached streams.

    Choose a local Linux image prepared by the maintainer. The runner accepts
    the image reference only, with no policy-provided host paths or flags.
    """

    def __init__(self, image: str, *, executable: str = "docker"):
        if not _IMAGE_REFERENCE.fullmatch(image):
            raise ValueError("invalid local policy image reference")
        self._image = image
        self._executable = executable

    async def _command(self, *args: str, attached: bool = False):
        try:
            executable = shutil.which(self._executable)
            if executable is None:
                raise FileNotFoundError
            return await asyncio.create_subprocess_exec(
                executable, *args, env=_host_environment(), close_fds=True,
                stdin=asyncio.subprocess.PIPE if attached else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE if attached else asyncio.subprocess.DEVNULL,
                limit=FRAME_BYTES if attached else _DAEMON_RESPONSE_BYTES,
            )
        except OSError:
            raise TransportError(TransportCode.PROCESS_ERROR) from None

    async def _management(self, deadline: float, *args: str) -> bytes:
        process = await before_deadline(self._command(*args), deadline)
        assert process.stdout is not None
        try:
            try:
                await before_deadline(process.stdout.readexactly(_DAEMON_RESPONSE_BYTES), deadline)
            except asyncio.IncompleteReadError as error:
                result = error.partial.strip()
            else:
                raise TransportError(TransportCode.PROCESS_ERROR)
            if await before_deadline(process.wait(), deadline) != 0:
                raise TransportError(TransportCode.PROCESS_ERROR)
            return result
        finally:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.kill()
                with suppress(TimeoutError):
                    await asyncio.wait_for(process.wait(), timeout=1)

    async def _local_image_id(self, deadline: float) -> str:
        result = await self._management(
            deadline, "image", "inspect", "--format", "{{.Id}} {{json .Config.Volumes}}", self._image,
        )
        image_id, _, volumes = result.partition(b" ")
        if _IMAGE_ID.fullmatch(image_id) is None or volumes not in (b"null", b"{}"):
            raise TransportError(TransportCode.PROCESS_ERROR)
        return image_id.decode("ascii")

    async def _container_absent(self, deadline: float, name: str) -> bool:
        try:
            result = await self._management(
                deadline, "ps", "--all", "--filter", f"name=^/{name}$", "--format", "{{.Names}}",
            )
        except TransportError:
            return False
        return not result

    @asynccontextmanager
    async def open(self, deadline: float):
        image_id = await self._local_image_id(deadline)
        name = "genesis-policy-" + uuid4().hex
        created = False
        try:
            await self._management(deadline, *_create_arguments(name, image_id))
            created = True
            process = await before_deadline(
                self._command("start", "--attach", "--interactive", name, attached=True), deadline,
            )
            channel = JsonlProcess(process)
            try:
                yield channel
            finally:
                await channel.close()
        finally:
            cleanup_deadline = monotonic() + _CLEANUP_SECONDS
            try:
                await self._management(cleanup_deadline, "rm", "--force", "--volumes", name)
            except TransportError:
                if created or not await self._container_absent(cleanup_deadline, name):
                    raise PolicyCleanupError("policy container removal could not be confirmed") from None
