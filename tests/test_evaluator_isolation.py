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
from genesis.environment.docker_policy import DockerPolicy, PolicyCleanupError, _create_arguments, _host_environment
from genesis.environment.jsonl import FRAME_BYTES, STDERR_LINE_BYTES, JsonlProcess, TransportCode, TransportError
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

    async def test_identical_closed_wire_bytes_different_private_scores(self):
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

    async def test_transport_failures_are_bounded_empty_truncations(self):
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

    async def test_transport_accepts_contract_sized_unicode_submission(self):
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
            ("sha256:" + "0" * 64 + " null").encode("ascii"),
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


class DockerConfigurationTests(unittest.TestCase):
    def test_fixed_switches_and_host_env_allowlist(self):
        flags = _create_arguments("genesis-test", "sha256:" + "0" * 64)
        for flag, value in (
            ("--network", "none"), ("--ipc", "none"), ("--user", "65532:65532"),
            ("--cap-drop", "ALL"), ("--security-opt", "no-new-privileges"), ("--pull", "never"),
            ("--memory", "512m"), ("--pids-limit", "32"),
        ):
            self.assertEqual(flags[flags.index(flag) + 1], value)
        self.assertIn("--read-only", flags)
        self.assertTrue(set(flags).isdisjoint({
            "--env", "--env-file", "--mount", "--volume", "--privileged", "--device", "--pid", "--entrypoint",
        }))
        self.assertLessEqual(set(_host_environment()), {"PATH", "SYSTEMROOT", "WINDIR"})
        with self.assertRaises(ValueError):
            DockerPolicy("--privileged")


@unittest.skipUnless(os.environ.get("GENESIS_POLICY_SMOKE_IMAGE"), "requires the local Docker smoke image")
class DockerPolicyTests(unittest.IsolatedAsyncioTestCase):
    def container_ids(self):
        return subprocess.check_output([
            "docker", "ps", "--all", "--filter", "label=genesis.policy=true", "--format", "{{.ID}}",
        ], text=True, timeout=5).splitlines()

    async def test_real_container_configuration_and_lifecycle(self):
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
