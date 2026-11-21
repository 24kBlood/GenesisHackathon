"""Production runner for a trusted, locally staged policy image."""

import asyncio
import json
import os
import re
import shutil
from contextlib import asynccontextmanager, suppress
from time import monotonic
from uuid import uuid4

from .jsonl import FRAME_BYTES, JsonlProcess, TransportCode, TransportError, before_deadline


_IMAGE_REFERENCE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._:/@-]{0,255}")
_IMAGE_ID = re.compile(r"sha256:[a-f0-9]{64}")
_DAEMON_RESPONSE_BYTES = 512
_IMAGE_FIELDS = '[{"Id":{{json .Id}},"Config":{"Volumes":{{json (index .Config "Volumes")}}}}]'
_CLEANUP_SECONDS = 5


class PolicyCleanupError(RuntimeError):
    """Container teardown could not be confirmed. Closed scoring must stop."""


def _host_environment() -> dict[str, str]:
    return {key: os.environ[key] for key in ("PATH", "SYSTEMROOT", "WINDIR") if key in os.environ}


def _create_arguments(
    name: str, image_id: str, *, cpus: int = 1, memory_mib: int = 512, tmpfs_mib: int = 0,
) -> tuple[str, ...]:
    memory = f"{memory_mib}m"
    tmpfs = ("--tmpfs", f"/tmp:rw,nosuid,nodev,noexec,size={tmpfs_mib}m,mode=1777") if tmpfs_mib else ()
    return (
        "create", "--name", name, "--label", "genesis.policy=true", "--pull", "never",
        "--interactive", "--network", "none", "--ipc", "none", "--read-only",
        "--user", "65532:65532", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--pids-limit", "32", "--memory", memory, "--memory-swap", memory,
        "--cpus", str(cpus), "--log-driver", "none", "--no-healthcheck", *tmpfs, image_id,
    )


class DockerPolicy:
    """Own the lifetime of one container and expose only its attached streams.

    Choose a local Linux image prepared by the maintainer. The runner accepts
    a local image reference and coordinator-owned limits.
    """

    def __init__(
        self, image: str, *, executable: str = "docker", cpus: int = 1,
        memory_mib: int = 512, tmpfs_mib: int = 0,
    ):
        if not _IMAGE_REFERENCE.fullmatch(image):
            raise ValueError("invalid local policy image reference")
        self._image = image
        self._executable = executable
        if (
            type(cpus) is not int or not 1 <= cpus <= 2
            or type(memory_mib) is not int or not 64 <= memory_mib <= 4096
            or type(tmpfs_mib) is not int or not 0 <= tmpfs_mib <= min(64, memory_mib)
        ):
            raise ValueError("invalid policy resource limits")
        self._limits = {"cpus": cpus, "memory_mib": memory_mib, "tmpfs_mib": tmpfs_mib}

    async def _command(self, *args: str, attached: bool = False, output_limit: int = _DAEMON_RESPONSE_BYTES):
        try:
            executable = shutil.which(self._executable)
            if executable is None:
                raise FileNotFoundError
            return await asyncio.create_subprocess_exec(
                executable, *args, env=_host_environment(), close_fds=True,
                stdin=asyncio.subprocess.PIPE if attached else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE if attached else asyncio.subprocess.DEVNULL,
                limit=FRAME_BYTES if attached else output_limit,
            )
        except OSError:
            raise TransportError(TransportCode.PROCESS_ERROR) from None

    async def _management(
        self, deadline: float, *args: str, output_limit: int = _DAEMON_RESPONSE_BYTES,
    ) -> bytes:
        process = await before_deadline(self._command(*args, output_limit=output_limit), deadline)
        assert process.stdout is not None
        try:
            try:
                await before_deadline(process.stdout.readexactly(output_limit), deadline)
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
        result = await self._management(deadline, "image", "inspect", "--format", _IMAGE_FIELDS, self._image)
        try:
            images = json.loads(result)
        except (ValueError, RecursionError):
            raise TransportError(TransportCode.PROCESS_ERROR) from None
        if type(images) is not list or len(images) != 1 or type(images[0]) is not dict:
            raise TransportError(TransportCode.PROCESS_ERROR)
        config = images[0].get("Config")
        image_id = images[0].get("Id")
        if (
            type(config) is not dict or config.get("Volumes") not in (None, {})
            or type(image_id) is not str or _IMAGE_ID.fullmatch(image_id) is None
        ):
            raise TransportError(TransportCode.PROCESS_ERROR)
        return image_id

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
            await self._management(deadline, *_create_arguments(name, image_id, **self._limits))
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
                if created or not await self._container_absent(monotonic() + _CLEANUP_SECONDS, name):
                    raise PolicyCleanupError("policy container removal could not be confirmed") from None
