"""Production runner for a trusted, locally staged policy image."""

import asyncio
import json
import math
import os
import re
import shutil
from contextlib import asynccontextmanager, suppress
from time import monotonic
from uuid import uuid4

from .jsonl import FRAME_BYTES, JsonlProcess, TransportCode, TransportError, before_deadline


_IMAGE_REFERENCE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._:/@-]{0,255}")
_IMAGE_ID = re.compile(r"sha256:[a-f0-9]{64}")
_MEMORY_LIMIT = re.compile(r"([1-9][0-9]{0,8})([bkmgBKMG]?)")
_CPU_LIMIT = re.compile(r"(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)")
_MEMORY_SCALE = {"": (1, 1024 * 1024), "b": (1, 1024 * 1024), "k": (1, 1024), "m": (1, 1), "g": (1024, 1)}
_MAX_CPUS = 2
_MAX_MEMORY_MIB = 4096
_HOST_ENV_KEYS = ("PATH", "SYSTEMROOT", "WINDIR", "DOCKER_HOST", "XDG_RUNTIME_DIR", "DOCKER_CONTEXT")
_IMAGE_INSPECT_FORMAT = '{"Id":{{json .Id}},"Config":{{json .Config}}}'
_IMAGE_FIELDS = _IMAGE_INSPECT_FORMAT
_DAEMON_RESPONSE_BYTES = 512
_IMAGE_INSPECT_BYTES = 64 * 1024
_CLEANUP_SECONDS = 5


class PolicyCleanupError(RuntimeError):
    """Container teardown could not be confirmed. Closed scoring must stop."""


def _host_environment() -> dict[str, str]:
    return {key: os.environ[key] for key in _HOST_ENV_KEYS if key in os.environ}


def _validated_cpus(cpus: str | int | float) -> str:
    if type(cpus) is int:
        if cpus <= 0 or cpus > _MAX_CPUS:
            raise ValueError(f"cpus must be between 1 and {_MAX_CPUS}")
        return str(cpus)
    if type(cpus) is float:
        if not math.isfinite(cpus) or cpus <= 0 or cpus > _MAX_CPUS:
            raise ValueError(f"cpus must be positive, finite and at most {_MAX_CPUS}")
        return f"{cpus:g}"
    if (
        type(cpus) is str and _CPU_LIMIT.fullmatch(cpus) is not None
        and 0 < float(cpus) <= _MAX_CPUS
    ):
        return cpus
    raise ValueError("invalid cpus limit")


def _validated_memory(value: str) -> str:
    match = _MEMORY_LIMIT.fullmatch(value) if type(value) is str else None
    if match is None:
        raise ValueError("invalid memory limit")
    amount, unit = int(match.group(1)), match.group(2).lower()
    numerator, denominator = _MEMORY_SCALE[unit]
    mib = (amount * numerator) // denominator
    if mib > _MAX_MEMORY_MIB:
        raise ValueError(f"memory must be at most {_MAX_MEMORY_MIB} MiB")
    return value


def _create_arguments(
    name: str,
    image_id: str,
    *,
    cpus: int | float | str = 1,
    memory_mib: int = 512,
    tmpfs_mib: int = 0,
    memory: str | None = None,
    pids_limit: int = 32,
    tmpfs_tmp_mb: int | None = None,
) -> tuple[str, ...]:
    resolved_memory = memory if memory is not None else f"{memory_mib}m"
    resolved_tmpfs = tmpfs_tmp_mb if tmpfs_tmp_mb is not None else (tmpfs_mib or None)
    args: list[str] = [
        "create", "--name", name, "--label", "genesis.policy=true", "--pull", "never",
        "--interactive", "--network", "none", "--ipc", "none", "--read-only",
        "--user", "65532:65532", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--no-healthcheck", "--ulimit", "nofile=256:256", "--cgroupns", "private",
        "--pids-limit", str(pids_limit), "--memory", resolved_memory, "--memory-swap", resolved_memory,
        "--cpus", str(cpus), "--log-driver", "none",
    ]
    if resolved_tmpfs is not None:
        args.extend(["--tmpfs", f"/tmp:rw,nosuid,nodev,noexec,size={resolved_tmpfs}m,mode=1777"])
    args.append(image_id)
    return tuple(args)


class DockerPolicy:
    """Own the lifetime of one container and expose only its attached streams.

    Choose a local Linux image prepared by the maintainer. The runner accepts
    a local image reference and coordinator-owned resource limits, with no
    policy-provided host paths or arbitrary flags.
    """

    def __init__(
        self,
        image: str,
        *,
        executable: str = "docker",
        cpus: int | float | str = 1,
        memory_mib: int = 512,
        tmpfs_mib: int = 0,
        memory: str | None = None,
        pids_limit: int = 32,
        tmpfs_tmp_mb: int | None = None,
    ):
        if type(image) is not str or not _IMAGE_REFERENCE.fullmatch(image):
            raise ValueError("invalid local policy image reference")
        normalized_cpus = _validated_cpus(cpus)
        if memory is not None:
            resolved_memory = _validated_memory(memory)
        else:
            if type(memory_mib) is not int or not 64 <= memory_mib <= 4096:
                raise ValueError("invalid memory_mib limit")
            resolved_memory = f"{memory_mib}m"
        if type(pids_limit) is not int or not 1 <= pids_limit <= 4096:
            raise ValueError("pids_limit must be an integer between 1 and 4096")
        if tmpfs_tmp_mb is not None:
            if type(tmpfs_tmp_mb) is not int or not 1 <= tmpfs_tmp_mb <= 1024:
                raise ValueError("tmpfs_tmp_mb must be None or an integer between 1 and 1024")
            resolved_tmpfs: int | None = tmpfs_tmp_mb
        else:
            max_tmpfs = min(64, memory_mib) if memory is None else 1024
            if type(tmpfs_mib) is not int or not 0 <= tmpfs_mib <= max_tmpfs:
                raise ValueError("invalid tmpfs_mib limit")
            resolved_tmpfs = tmpfs_mib or None
        self._image = image
        self._executable = executable
        self._memory = resolved_memory
        self._cpus = normalized_cpus
        self._pids_limit = pids_limit
        self._tmpfs_tmp_mb = resolved_tmpfs

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
        result = await self._management(
            deadline, "image", "inspect", "--format", _IMAGE_INSPECT_FORMAT, self._image,
            output_limit=_IMAGE_INSPECT_BYTES,
        )
        try:
            parsed = json.loads(result)
        except (ValueError, RecursionError):
            raise TransportError(TransportCode.PROCESS_ERROR) from None
        if type(parsed) is list:
            if len(parsed) != 1 or type(parsed[0]) is not dict:
                raise TransportError(TransportCode.PROCESS_ERROR)
            image = parsed[0]
        elif type(parsed) is dict:
            image = parsed
        else:
            raise TransportError(TransportCode.PROCESS_ERROR)
        config = image.get("Config")
        image_id = image.get("Id")
        if type(config) is not dict or type(image_id) is not str or _IMAGE_ID.fullmatch(image_id) is None:
            raise TransportError(TransportCode.PROCESS_ERROR)
        healthcheck = config.get("Healthcheck")
        healthcheck_disabled = (
            healthcheck in (None, {})
            or (type(healthcheck) is dict and healthcheck.get("Test") == ["NONE"])
        )
        if config.get("Volumes") not in (None, {}) or not healthcheck_disabled:
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
        try:
            image_id = await self._local_image_id(deadline)
        except TransportError as error:
            raise TransportError(error.code, phase="setup") from error
        name = "genesis-policy-" + uuid4().hex
        created = False
        try:
            try:
                await self._management(
                    deadline,
                    *_create_arguments(
                        name, image_id, memory=self._memory, cpus=self._cpus,
                        pids_limit=self._pids_limit, tmpfs_tmp_mb=self._tmpfs_tmp_mb,
                    ),
                )
                created = True
                process = await before_deadline(
                    self._command("start", "--attach", "--interactive", name, attached=True), deadline,
                )
            except TransportError as error:
                raise TransportError(error.code, phase="setup") from error
            channel = JsonlProcess(process)
            try:
                yield channel
            finally:
                await channel.close()
        finally:
            try:
                await self._management(monotonic() + _CLEANUP_SECONDS, "rm", "--force", "--volumes", name)
            except TransportError:
                if created or not await self._container_absent(monotonic() + _CLEANUP_SECONDS, name):
                    raise PolicyCleanupError("policy container removal could not be confirmed") from None
