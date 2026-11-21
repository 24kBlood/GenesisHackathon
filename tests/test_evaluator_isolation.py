import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from genesis.environment.auditor_env import AuditorEnv, Episode, ToolCode, ToolError
from genesis.environment.docker_policy import (
    DockerPolicy,
    PolicyCleanupError,
    _HOST_ENV_KEYS,
    _IMAGE_INSPECT_FORMAT,
    _create_arguments,
    _host_environment,
)
from genesis.environment.jsonl import (
    FRAME_BYTES,
    STDERR_BYTES,
    STDERR_LINE_BYTES,
    JsonlProcess,
    TransportCode,
    TransportError,
)
from genesis.evaluation.judge import run_episode
from genesis.evaluation.matcher import match
from genesis.evaluation.reward import score_terminal

import test_env_contract_and_isolation as contract


_WAITING_WORKER = "import sys, time; sys.stdin.readline(); time.sleep(20)"
_SCRIPTED_WORKER = """
import sys
from pathlib import Path

print("synthetic-worker-diagnostic", file=sys.stderr)
trace = [sys.stdin.buffer.readline()]
for frame in Path(sys.argv[2]).read_bytes().splitlines(keepends=True):
    sys.stdout.buffer.write(frame)
    sys.stdout.buffer.flush()
    trace.append(sys.stdin.buffer.readline())
Path(sys.argv[1]).write_bytes(b''.join(trace))
"""


class LocalPolicy:
    """Benign test subprocess. The production runner uses DockerPolicy."""

    def __init__(self, source, *arguments):
        self.source = source
        self.arguments = arguments
        self.process = None

    @asynccontextmanager
    async def open(self, deadline):
        self.process = await asyncio.create_subprocess_exec(
            sys.executable, "-I", "-u", "-c", self.source, *self.arguments,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env=_host_environment(), limit=FRAME_BYTES,
        )
        channel = JsonlProcess(self.process)
        try:
            yield channel
        finally:
            await channel.close()


class EvaluatorIsolationTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = contract.DataContractTests
        cls.fixture.setUpClass()

    def make_env(self, runner, *, mode="eval", timeout=10):
        scored = []

        def factory(seed, config):
            manifest = {**self.fixture.manifest, "seed": seed}
            policy = {**self.fixture.policy, "truth": config}
            evidence = self.fixture.findings["findings"][0]["evidence_ids"]

            def handle(action):
                if action["type"] == "get_document" and action["doc_id"] == "policy":
                    return {"document": policy, "provenance": config}
                raise ToolError(ToolCode.NOT_FOUND)

            def evaluate(submission, inspection_steps, *, invalid_submission):
                if mode == "eval":
                    self.assertIsNotNone(runner.process.returncode, "worker must exit before closed scoring")
                scored.append(submission)
                true_issues = [] if config["clean"] else self.fixture.truth["issues"]
                matched = match(submission["findings"], true_issues, evidence)
                return score_terminal(
                    matched.tp, matched.fp, matched.fn, inspection_steps, invalid_submission=invalid_submission,
                )

            return Episode("Synthetic evidence check", manifest, policy, handle, evaluate)

        return AuditorEnv(factory, mode=mode, timeout_seconds=timeout), scored

    async def replay(self, actions, *, clean=False, mode="eval"):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "public.jsonl"
            script = Path(directory) / "actions.jsonl"
            script.write_bytes("".join(json.dumps(action, ensure_ascii=False) + "\n" for action in actions).encode("utf-8"))
            runner = LocalPolicy(_SCRIPTED_WORKER, str(trace), str(script))
            env, scored = self.make_env(runner, mode=mode)
            report = await run_episode(env, seed=986, config={"clean": clean}, runner=runner)
            self.assertNotEqual(runner.process.pid, os.getpid())
            self.assertEqual(runner.process.returncode, 0)
            self.assertEqual(len(scored), 1)
            self.assertIsNone(report.transport_error)
            return report, trace.read_bytes()

    async def test_C19_identical_closed_wire_bytes_different_private_scores(self):
        actions = [
            {"type": "get_document", "doc_id": "policy"},
            {"type": "get_document", "doc_id": "missing-document"},
            {"type": "reset", "seed": 0},
            {"type": "submit", "findings": self.fixture.findings},
        ]
        first, public = await self.replay(actions)
        second, other_public = await self.replay(actions, clean=True)
        self.assertEqual(public, other_public)
        self.assertNotIn(b"synthetic-worker-diagnostic", public)
        frames = [json.loads(line) for line in public.splitlines()]
        self.assertEqual(len(frames), 5)
        self.assertEqual(frames[2]["observation"]["error"], {"code": "not_found"})
        self.assertEqual(frames[3]["observation"]["error"], {"code": "invalid_action"})
        self.assertEqual(frames[-1], {
            "type": "step", "observation": {"remaining_steps": 96, "status": "submitted"},
            "reward": 0.0, "terminated": True, "truncated": False, "info": {},
        })
        self.assertTrue(contract.PRIVATE_KEYS.isdisjoint(contract.field_names(frames)))
        self.assertEqual(first.terminal.score.reward, 0.994)
        self.assertEqual(second.terminal.score.reward, -1)

    async def test_open_receives_only_terminal_reward_and_public_ack(self):
        report, public = await self.replay([{"type": "submit", "findings": self.fixture.findings}], mode="train")
        reset, step = (json.loads(line) for line in public.splitlines())
        self.assertEqual((reset["type"], step["reward"], step["info"]), ("reset", 1.0, {}))
        self.assertTrue(contract.PRIVATE_KEYS.isdisjoint(contract.field_names(step)))
        self.assertEqual(report.terminal.score.reward, 1.0)

    async def test_valid_json_invalid_action_spends_step_then_invalid_submit_terminates(self):
        report, public = await self.replay([[], {"type": "submit", "findings": []}])
        _, invalid, terminal = (json.loads(line) for line in public.splitlines())
        self.assertEqual(invalid["observation"], {"remaining_steps": 99, "error": {"code": "invalid_action"}})
        self.assertEqual(terminal["observation"], {"remaining_steps": 98, "status": "submitted"})
        self.assertEqual(terminal["reward"], 0)
        self.assertTrue(report.terminal.invalid_submission)
        self.assertEqual(report.terminal.score.reward, -1)

    async def test_C17_transport_accepts_valid_frame_and_bounds_failures(self):
        await self._check_contract_sized_unicode_submission()
        for source, want in (
            (_WAITING_WORKER, TransportCode.TIMEOUT),
            ("import sys; sys.stdin.readline()", TransportCode.EOF),
            ("import sys; sys.stdin.readline(); print('{bad-json}')", TransportCode.BAD_FRAME),
            ("import sys; sys.stdin.readline(); print('1e999')", TransportCode.BAD_FRAME),
            ("import sys; sys.stdin.readline(); print(' ', end='')", TransportCode.BAD_FRAME),
            (f"import sys; sys.stdin.readline(); print('a' * {FRAME_BYTES})", TransportCode.FRAME_LIMIT),
            (f"import sys; sys.stdin.readline(); print('diagnostic' * {STDERR_LINE_BYTES}, file=sys.stderr)",
             TransportCode.STDERR_LIMIT),
        ):
            with self.subTest(want=want):
                runner = LocalPolicy(source)
                env, scored = self.make_env(runner, timeout=0.8 if want == TransportCode.TIMEOUT else 10)
                report = await run_episode(env, seed=986, config={"clean": True}, runner=runner)
                self.assertEqual(report.transport_error, want)
                self.assertTrue(report.terminal.truncated)
                self.assertEqual(report.terminal.reason, "timeout" if want is TransportCode.TIMEOUT else "transport")
                self.assertEqual(report.terminal.submission["findings"], [])
                self.assertEqual(report.terminal.score.inspection_steps, 0)
                self.assertEqual(report.terminal.score.reward, 1)
                self.assertEqual(len(scored), 1)

    async def _check_contract_sized_unicode_submission(self):
        findings = {
            **self.fixture.findings,
            "findings": [{**self.fixture.findings["findings"][0], "explanation": "🙂" * 1000}] * 30,
        }
        report, _ = await self.replay([{"type": "submit", "findings": findings}])
        self.assertFalse(report.terminal.invalid_submission)
        self.assertEqual(len(report.terminal.submission["findings"]), 30)

    async def test_cleanup_failure_aborts_closed_scoring_after_neutral_ack(self):
        runner = DockerPolicy("synthetic:local")
        env, scored = self.make_env(SimpleNamespace(process=SimpleNamespace(returncode=None)))
        channel = SimpleNamespace(
            send=AsyncMock(),
            receive=AsyncMock(return_value={"type": "submit", "findings": self.fixture.findings}),
            close=AsyncMock(),
        )
        commands = AsyncMock(side_effect=[
            json.dumps([{"Id": "sha256:" + "0" * 64, "Config": {}}]).encode("ascii"),
            b"created",
            TransportError(TransportCode.PROCESS_ERROR),
        ])
        with (
            patch.object(runner, "_management", commands),
            patch.object(runner, "_command", AsyncMock(return_value=object())),
            patch("genesis.environment.docker_policy.JsonlProcess", return_value=channel),
            self.assertRaises(PolicyCleanupError),
        ):
            await run_episode(env, seed=986, config={"clean": False}, runner=runner)
        self.assertEqual(channel.send.await_args.args[0]["reward"], 0)
        self.assertTrue(channel.send.await_args.args[0]["terminated"])
        self.assertEqual(commands.await_args.args[1:4], ("rm", "--force", "--volumes"))
        self.assertEqual(scored, [])

    async def test_missing_image_is_normal_truncation_without_worker(self):
        runner = DockerPolicy("synthetic:missing")
        env, scored = self.make_env(SimpleNamespace(process=SimpleNamespace(returncode=0)))
        command = AsyncMock(side_effect=TransportError(TransportCode.PROCESS_ERROR))
        with patch.object(runner, "_management", command):
            report = await run_episode(env, seed=986, config={"clean": True}, runner=runner)
        self.assertEqual(command.await_count, 1)
        self.assertEqual(report.transport_error, TransportCode.PROCESS_ERROR)
        self.assertEqual(report.terminal.submission["findings"], [])
        self.assertTrue(report.terminal.truncated)
        self.assertEqual(len(scored), 1)

    async def test_jsonl_guards_reject_duplicate_keys_nonfinite_utf8_overrun_stderr_and_send_errors(self):
        cases = (
            ("duplicate_keys", 'import sys; sys.stdin.readline(); print(\'{"a":1,"a":2}\')', TransportCode.BAD_FRAME),
            ("nan", "import sys; sys.stdin.readline(); print('{\"x\":NaN}')", TransportCode.BAD_FRAME),
            ("infinity", "import sys; sys.stdin.readline(); print('{\"x\":Infinity}')", TransportCode.BAD_FRAME),
            ("neg_infinity", "import sys; sys.stdin.readline(); print('{\"x\":-Infinity}')", TransportCode.BAD_FRAME),
            ("invalid_utf8", r"import sys; sys.stdin.readline(); sys.stdout.buffer.write(b'\xff\xfe\n'); sys.stdout.buffer.flush()", TransportCode.BAD_FRAME),
            ("limit_overrun_no_nl", f"import sys; sys.stdin.readline(); sys.stdout.buffer.write(b'a' * ({FRAME_BYTES} + 10)); sys.stdout.buffer.flush()", TransportCode.FRAME_LIMIT),
            (
                "cumulative_stderr",
                f"import sys; sys.stdin.readline(); [sys.stderr.write('x' * 4000 + '\\n') for _ in range({STDERR_BYTES // 4000 + 2})]; sys.stderr.flush(); sys.stdin.readline()",
                TransportCode.STDERR_LIMIT,
            ),
            (
                "lone_utf16_surrogate",
                r'import json, sys; m = json.loads(sys.stdin.readline()); ep = m["observation"]["manifest"]["episode_id"]; '
                r'sys.stdout.write("{\"type\":\"submit\",\"findings\":{\"episode_id\":\"" + ep + "\",\"findings\":[{\"type\":\"fx_conversion\",\"affected_ids\":[\"\\ud800\"],\"claimed_amount_minor\":1,\"currency\":\"KZT\",\"evidence_ids\":[\"policy\"],\"explanation\":\"x\"}]}}\n"); sys.stdout.flush()',
                TransportCode.BAD_FRAME,
            ),
        )
        for label, source, want in cases:
            with self.subTest(case=label):
                runner = LocalPolicy(source)
                env, scored = self.make_env(runner, timeout=10)
                report = await run_episode(env, seed=986, config={"clean": False}, runner=runner)
                self.assertEqual(report.transport_error, want)
                self.assertTrue(report.terminal.truncated)
                self.assertEqual(report.terminal.reason, "transport")
                self.assertEqual(report.terminal.score.reward, -1.0)
                self.assertEqual(len(scored), 1)

        # Verify JsonlProcess.send() error paths directly (FRAME_LIMIT, TIMEOUT, EOF).
        runner = LocalPolicy("import sys; sys.stdin.readline()")
        async with runner.open(monotonic() + 10) as channel:
            with self.assertRaises(TransportError) as frame_err:
                await channel.send({"blob": "x" * FRAME_BYTES}, monotonic() + 10)
            self.assertEqual(frame_err.exception.code, TransportCode.FRAME_LIMIT)
            with self.assertRaises(TransportError) as timeout_err:
                await channel.send({"ok": True}, monotonic() - 1)
            self.assertEqual(timeout_err.exception.code, TransportCode.TIMEOUT)

        closing_runner = LocalPolicy("import os, sys; os.close(0)")
        async with closing_runner.open(monotonic() + 10) as channel:
            await closing_runner.process.wait()
            with self.assertRaises(TransportError) as eof_err:
                await channel.send({"blob": "x" * 1024}, monotonic() + 10)
            self.assertEqual(eof_err.exception.code, TransportCode.EOF)

    async def test_post_submit_eof_and_timeout_ack_race_preserve_valid_submission(self):
        # Closing stdin before writing submit deterministically triggers EPIPE on terminal ack send.
        source = (
            "import json, os, sys; "
            "m = json.loads(sys.stdin.readline()); "
            "ep = m['observation']['manifest']['episode_id']; "
            "os.close(0); "
            "print(json.dumps({'type': 'submit', 'findings': {'episode_id': ep, 'findings': []}}), flush=True)"
        )
        runner = LocalPolicy(source)
        env, scored = self.make_env(runner, mode="eval", timeout=10)
        report = await run_episode(env, seed=986, config={"clean": True}, runner=runner)
        self.assertIsNone(report.transport_error)
        self.assertTrue(report.terminal.terminated)
        self.assertFalse(report.terminal.truncated)
        self.assertEqual(report.terminal.reason, "submitted")
        self.assertEqual(report.terminal.score.reward, 1.0)
        self.assertEqual(len(scored), 1)

        # Timeout on terminal ack send after submit is accepted is also non-fatal.
        class TimeoutAckRunner:
            def __init__(self, findings):
                self._findings = findings
                self._sends = 0
                self.process = SimpleNamespace(returncode=0)

            @asynccontextmanager
            async def open(self, deadline):
                yield self

            async def send(self, message, deadline):
                self._sends += 1
                if self._sends == 2:
                    raise TransportError(TransportCode.TIMEOUT)

            async def receive(self, deadline):
                return {"type": "submit", "findings": self._findings}

        ack_runner = TimeoutAckRunner(self.fixture.findings)
        env2, _ = self.make_env(ack_runner, mode="eval", timeout=10)
        report2 = await run_episode(env2, seed=986, config={"clean": False}, runner=ack_runner)
        self.assertIsNone(report2.transport_error)
        self.assertTrue(report2.terminal.terminated)
        self.assertEqual(report2.terminal.reason, "submitted")
        self.assertEqual(report2.terminal.score.reward, 1.0)


class DockerConfigurationTests(unittest.TestCase):
    def test_fixed_switches_and_host_env_allowlist(self):
        flags = _create_arguments("genesis-test", "sha256:" + "0" * 64)
        for flag, value in (
            ("--network", "none"), ("--ipc", "none"), ("--user", "65532:65532"),
            ("--cap-drop", "ALL"), ("--security-opt", "no-new-privileges"), ("--pull", "never"),
            ("--ulimit", "nofile=256:256"), ("--cgroupns", "private"),
            ("--memory", "512m"), ("--memory-swap", "512m"), ("--cpus", "1"), ("--pids-limit", "32"),
        ):
            self.assertEqual(flags[flags.index(flag) + 1], value)
        self.assertIn("--read-only", flags)
        self.assertIn("--no-healthcheck", flags)
        self.assertNotIn("--tmpfs", flags)
        self.assertTrue(set(flags).isdisjoint({
            "--env", "--env-file", "--mount", "--volume", "--privileged", "--device", "--pid", "--entrypoint",
        }))
        with patch.dict(
            os.environ,
            {"PATH": "/usr/bin", "DOCKER_HOST": "unix:///run/user/1000/docker.sock",
             "XDG_RUNTIME_DIR": "/run/user/1000", "DOCKER_CONTEXT": "desktop-linux",
             "GH_TOKEN": "secret", "PYTHONPATH": "/secret"},
            clear=True,
        ):
            host_env = _host_environment()
        self.assertLessEqual(set(host_env), set(_HOST_ENV_KEYS))
        self.assertEqual(host_env["DOCKER_HOST"], "unix:///run/user/1000/docker.sock")
        self.assertEqual(host_env["XDG_RUNTIME_DIR"], "/run/user/1000")
        self.assertEqual(host_env["DOCKER_CONTEXT"], "desktop-linux")
        self.assertNotIn("GH_TOKEN", host_env)
        self.assertNotIn("PYTHONPATH", host_env)

        custom = _create_arguments(
            "genesis-custom", "sha256:" + "0" * 64,
            memory="2g", cpus="2", pids_limit=128, tmpfs_tmp_mb=64,
        )
        self.assertEqual(custom[custom.index("--memory") + 1], "2g")
        self.assertEqual(custom[custom.index("--memory-swap") + 1], "2g")
        self.assertEqual(custom[custom.index("--cpus") + 1], "2")
        self.assertEqual(custom[custom.index("--pids-limit") + 1], "128")
        self.assertEqual(custom[custom.index("--tmpfs") + 1], "/tmp:rw,nosuid,nodev,noexec,size=64m,mode=1777")

        for bad_kwargs in (
            {"image": "--privileged"},
            {"image": "synthetic:local", "memory": "--privileged"},
            {"image": "synthetic:local", "cpus": True},
            {"image": "synthetic:local", "cpus": 0},
            {"image": "synthetic:local", "pids_limit": 0},
            {"image": "synthetic:local", "tmpfs_tmp_mb": 0},
        ):
            with self.subTest(bad_kwargs=bad_kwargs), self.assertRaises(ValueError):
                DockerPolicy(**bad_kwargs)


class DockerImageTests(unittest.IsolatedAsyncioTestCase):
    async def test_accepts_absent_null_or_empty_volumes_in_local_image_config(self):
        runner = DockerPolicy("synthetic:local")
        image_id = "sha256:" + "a" * 64
        deadline = monotonic() + 10
        for payload in (
            {"Id": image_id, "Config": {"Volumes": None, "Healthcheck": None}},
            {"Id": image_id, "Config": {"Volumes": {}, "Healthcheck": {"Test": ["NONE"]}}},
            [{"Id": image_id, "Config": {"Entrypoint": ["python"]}}],
            [{"Id": image_id, "Config": {"Volumes": None}}],
            [{"Id": image_id, "Config": {"Volumes": {}}}],
        ):
            with self.subTest(payload=payload):
                inspect = AsyncMock(return_value=json.dumps(payload).encode("ascii"))
                with patch.object(runner, "_management", inspect):
                    self.assertEqual(await runner._local_image_id(deadline), image_id)
                inspect.assert_awaited_once_with(
                    deadline, "image", "inspect", "--format", _IMAGE_INSPECT_FORMAT,
                    "synthetic:local", output_limit=65536,
                )

    async def test_rejects_declared_volumes_healthcheck_or_unverifiable_image_config(self):
        runner = DockerPolicy("synthetic:local")
        image_id = "sha256:" + "a" * 64
        for image in (
            {"Id": image_id, "Config": {"Volumes": {"/data": {}}}},
            {"Id": image_id, "Config": {"Healthcheck": {"Test": ["CMD-SHELL", "exit 0"]}}},
            {"Id": image_id, "Config": None},
            {"Id": "not-local-id", "Config": {}},
        ):
            with self.subTest(image=image):
                inspect = AsyncMock(return_value=json.dumps(image).encode("ascii"))
                with patch.object(runner, "_management", inspect), self.assertRaises(TransportError) as failure:
                    await runner._local_image_id(monotonic() + 10)
                self.assertEqual(failure.exception.code, TransportCode.PROCESS_ERROR)

    async def test_failed_create_with_slow_rm_does_not_falsely_raise_cleanup_error(self):
        runner = DockerPolicy("synthetic:local")
        image_id = "sha256:" + "a" * 64
        call_deadlines = []

        async def fake_management(deadline, *args, **kwargs):
            call_deadlines.append((args[0], deadline))
            if args[:2] == ("image", "inspect"):
                return json.dumps({"Id": image_id, "Config": {"Volumes": None, "Healthcheck": None}}).encode("ascii")
            if args[0] == "create":
                raise TransportError(TransportCode.PROCESS_ERROR)
            if args[0] == "rm":
                raise TransportError(TransportCode.TIMEOUT)
            if args[0] == "ps":
                return b""
            raise AssertionError(f"unexpected management call: {args}")

        with patch.object(runner, "_management", side_effect=fake_management):
            with self.assertRaises(TransportError) as failure:
                async with runner.open(monotonic() + 10):
                    pass
        self.assertEqual(failure.exception.code, TransportCode.PROCESS_ERROR)
        self.assertEqual([cmd for cmd, _ in call_deadlines], ["image", "create", "rm", "ps"])
        # _container_absent must receive its own fresh deadline (>= rm deadline).
        self.assertGreaterEqual(call_deadlines[3][1], call_deadlines[2][1])


@unittest.skipUnless(os.environ.get("GENESIS_POLICY_SMOKE_IMAGE"), "requires the local Docker smoke image")
class DockerPolicyTests(unittest.IsolatedAsyncioTestCase):
    def container_ids(self):
        return subprocess.check_output([
            "docker", "ps", "--all", "--filter", "label=genesis.policy=true", "--format", "{{.ID}}",
        ], text=True, timeout=5).splitlines()

    async def test_C18_real_container_configuration_and_lifecycle(self):
        runner = DockerPolicy(os.environ["GENESIS_POLICY_SMOKE_IMAGE"])
        deadline = monotonic() + 30
        async with runner.open(deadline) as channel:
            await channel.send({"type": "reset", "observation": {"manifest": {"episode_id": "synthetic"}}}, deadline)
            self.assertEqual(await channel.receive(deadline), {
                "type": "submit", "findings": {"episode_id": "synthetic", "findings": []},
            })
            container_id, = self.container_ids()
            details, = json.loads(subprocess.check_output(["docker", "inspect", container_id], timeout=5))
            host = details["HostConfig"]
            self.assertTrue(details["State"]["Running"])
            self.assertEqual(details["Config"]["User"], "65532:65532")
            self.assertEqual(host["NetworkMode"], "none")
            self.assertEqual(host["IpcMode"], "none")
            self.assertTrue(host["ReadonlyRootfs"])
            self.assertFalse(host["Privileged"])
            self.assertEqual(host["CapDrop"], ["ALL"])
            self.assertTrue(any(item.startswith("no-new-privileges") for item in host["SecurityOpt"]))
            self.assertEqual(host["PidsLimit"], 32)
            self.assertEqual(host["Memory"], 512 * 1024 ** 2)
            self.assertEqual(host["NanoCpus"], 10 ** 9)
            self.assertFalse(host["Binds"])
            self.assertFalse(details["Mounts"])
            self.assertTrue({"GH_TOKEN", "DOCKER_HOST", "PYTHONPATH"}.isdisjoint(
                item.split("=", 1)[0] for item in details["Config"]["Env"]
            ))
            await channel.send({"type": "step", "reward": 0.0, "terminated": True}, deadline)
        self.assertNotIn(container_id, self.container_ids())

    async def test_real_container_is_removed_after_read_deadline(self):
        runner = DockerPolicy(os.environ["GENESIS_POLICY_SMOKE_IMAGE"])
        deadline = monotonic() + 30
        with self.assertRaises(TransportError) as failure:
            async with runner.open(deadline) as channel:
                await channel.send({"type": "reset", "observation": {"manifest": {"episode_id": "timeout"}}}, deadline)
                await channel.receive(deadline)
                container_id, = self.container_ids()
                await channel.receive(monotonic() + 0.1)
        self.assertEqual(failure.exception.code, TransportCode.TIMEOUT)
        self.assertNotIn(container_id, self.container_ids())
