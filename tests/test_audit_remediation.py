"""Regressions for the verified audit findings fixed on the remediation branch."""

from contextlib import asynccontextmanager, contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
import errno
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, patch

from genesis.cli import __main__ as cli_main
from genesis.cli.run_policy import _publish_trace, _resolve_profile, run_policy
from genesis.environment.docker_policy import DockerPolicy
from genesis.environment.jsonl import TransportCode, TransportError
from genesis.evaluation.admission import require_admissible_episode
from genesis.evaluation.judge import closed_environment, run_episode
from genesis.generation.company import generate_closed_episode, generate_episode, new_closed_case
from genesis.generation.datasets import load_dataset_entries


class _StubEnv:
    def __init__(self):
        self.observation = {
            "manifest": {"episode_id": "episode-" + "0" * 24}, "remaining_steps": 100,
        }
        self.seconds_remaining = 10
        self.truncated = False

    def reset(self, seed, config):
        return deepcopy(self.observation), {}

    def begin_policy_interaction(self):
        self.seconds_remaining = 10

    def truncate(self, *, reason):
        self.truncated = True

    @property
    def terminal_result(self):
        return object()


class _NoopSend:
    async def __call__(self, *args, **kwargs):
        return None


@contextmanager
def _temporary_trace():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        yield root, root / "trace.jsonl"


class AdmissionTests(TestCase):
    def test_over_budget_closed_case_is_rejected_before_a_policy_starts(self):
        config = {
            "schema_version": "1.0.0", "error_counts": {"duplicate_invoice": 31},
            "obligation_count": 100,
        }
        seed, closed = new_closed_case(config)
        with self.assertRaises(ValueError):
            closed_environment().reset(seed, closed)

    def test_admissible_closed_case_is_accepted(self):
        seed, closed = new_closed_case({"schema_version": "1.0.0", "error_counts": {"fx_conversion": 1}})
        observation, _ = closed_environment().reset(seed, closed)
        self.assertEqual(observation["manifest"]["episode_id"], closed["episode_id"])

    def test_corrupt_truth_amount_is_rejected_by_admission(self):
        seed, closed = new_closed_case({"schema_version": "1.0.0", "error_counts": {"fx_conversion": 1}})
        episode = generate_closed_episode(seed, closed)
        episode.evaluator.truth["issues"][0]["expected_amount_minor"] += 7
        with self.assertRaises(ValueError):
            require_admissible_episode(episode)


class SetupFailureTests(IsolatedAsyncioTestCase):
    async def test_attach_exit_is_attributed_using_docker_start_state(self):
        image = json.dumps({"Id": "sha256:" + "0" * 64, "Config": {}}).encode("ascii")
        for started_at, started in ((b"0001-01-01T00:00:00Z", False), (b"2026-11-22T10:00:00Z", True)):
            with self.subTest(started=started):
                runner = DockerPolicy("synthetic:local")
                process = SimpleNamespace(returncode=125)
                channel = SimpleNamespace(
                    send=AsyncMock(), receive=AsyncMock(side_effect=TransportError(TransportCode.EOF)),
                    close=AsyncMock(),
                )
                env = _StubEnv()
                management = AsyncMock(side_effect=[image, b"created", started_at, b"removed"])
                with (
                    patch.object(runner, "_management", management),
                    patch.object(runner, "_command", AsyncMock(return_value=process)),
                    patch("genesis.environment.docker_policy.JsonlProcess", return_value=channel),
                ):
                    if started:
                        result = await run_episode(env, seed=1, config={}, runner=runner)
                        self.assertEqual(result.transport_error, TransportCode.EOF)
                        self.assertTrue(env.truncated)
                    else:
                        with self.assertRaises(TransportError) as failure:
                            await run_episode(env, seed=1, config={}, runner=runner)
                        self.assertEqual(failure.exception.phase, "setup")
                        self.assertFalse(env.truncated)
                        channel.send.assert_not_awaited()
                channel.close.assert_awaited_once()
                self.assertEqual(management.await_args.args[1:4], ("rm", "--force", "--volumes"))

    async def test_setup_timeout_is_rethrown_without_scoring(self):
        @asynccontextmanager
        async def setup_runner(deadline):
            raise TransportError(TransportCode.TIMEOUT, phase="setup")
            yield

        env = _StubEnv()
        with self.assertRaises(TransportError) as failure:
            await run_episode(env, seed=1, config={}, runner=SimpleNamespace(open=setup_runner))
        self.assertEqual(failure.exception.phase, "setup")
        self.assertFalse(env.truncated)

    async def test_interaction_timeout_is_still_scored(self):
        async def receive(deadline):
            raise TransportError(TransportCode.TIMEOUT)

        channel = SimpleNamespace(send=_NoopSend(), receive=receive)

        @asynccontextmanager
        async def interaction_runner(deadline):
            yield channel

        env = _StubEnv()
        result = await run_episode(env, seed=1, config={}, runner=SimpleNamespace(open=interaction_runner))
        self.assertEqual(result.transport_error, TransportCode.TIMEOUT)
        self.assertTrue(env.truncated)


class FxAmountTests(TestCase):
    def test_zero_rounded_posting_does_not_exhaust_fx_candidates(self):
        config = {
            "schema_version": "1.0.0", "error_counts": {"fx_conversion": 1},
            "base_currency": "JPY", "foreign_currency": "KZT",
        }
        episode = generate_episode(50, config)
        self.assertEqual(len(episode.evaluator.truth["issues"]), 1)
        self.assertTrue(all(
            row["debit_minor"] > 0 or row["credit_minor"] > 0 for row in episode.public.ledger
        ))
        require_admissible_episode(episode)


class ResourceCeilingTests(TestCase):
    def test_documented_ceilings_apply_to_every_input_format(self):
        for bad in (
            {"cpus": 3}, {"cpus": 2.5}, {"cpus": "3"},
            {"memory": "5g"}, {"memory": "4097m"}, {"memory": "4194305k"},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                DockerPolicy("synthetic:local", **bad)
        runner = DockerPolicy("synthetic:local", cpus="2", memory="2g")
        self.assertEqual((runner._cpus, runner._memory), ("2", "2g"))
        self.assertEqual(DockerPolicy("synthetic:local", memory="4194304k")._memory, "4194304k")


class CliRemediationTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = cli_main._REPOSITORY
        checksums = json.loads((cls.root / "data" / "checksums.json").read_text(encoding="utf-8"))
        cls.episode_id = next(row["episode_id"] for row in checksums["episodes"] if row["split"] == "train")

    def test_explicit_episode_id_does_not_regenerate_the_corpus(self):
        with patch("genesis.cli.run_policy.generate_episode", side_effect=AssertionError("regenerated")):
            profile = _resolve_profile(self.root, self.episode_id)
        expected_seed = next(
            entry.seed for entry in load_dataset_entries(self.root / "data" / "configs")
            if entry.config.get("episode_id") == self.episode_id
        )
        self.assertEqual((profile.name, profile.seed), (self.episode_id, expected_seed))

    def test_trace_publication_is_atomic_and_no_clobber(self):
        with _temporary_trace() as (directory, destination):
            _publish_trace(destination, b'{"ok":true}\n')
            self.assertEqual(destination.read_bytes(), b'{"ok":true}\n')
            self.assertEqual(list(directory.iterdir()), [destination])
            with self.assertRaises(FileExistsError):
                _publish_trace(destination, b"replacement\n")
            self.assertEqual(destination.read_bytes(), b'{"ok":true}\n')
            self.assertEqual(list(directory.iterdir()), [destination])

    def test_trace_failure_does_not_hide_a_completed_policy_result(self):
        with _temporary_trace() as (_, destination):
            destination.write_bytes(b"original\n")
            args = SimpleNamespace(
                profile="standard-03", image=None, policy="simple_reconciliation",
                cpus=1, memory_mib=512, tmpfs_mib=0, trace=destination,
            )
            out = StringIO()
            errors = StringIO()
            with redirect_stdout(out), redirect_stderr(errors):
                code = run_policy(self.root, args)
            result = json.loads(out.getvalue())
            self.assertEqual(code, 0)
            self.assertTrue(result["terminated"])
            self.assertIn("failed", result["trace"])
            self.assertIn("trace publication failed", errors.getvalue())
            self.assertEqual(destination.read_bytes(), b"original\n")

    def test_trace_publication_reports_unsupported_link_without_overwriting(self):
        with _temporary_trace() as (directory, destination):
            destination.write_bytes(b"original\n")
            with (
                patch("genesis.cli.run_policy.os.link", side_effect=OSError(errno.ENOTSUP, "no hard links")),
                self.assertRaises(OSError) as failure,
            ):
                _publish_trace(destination, b"replacement\n")
            self.assertEqual(failure.exception.errno, errno.ENOTSUP)
            self.assertEqual(destination.read_bytes(), b"original\n")
            self.assertEqual(list(directory.iterdir()), [destination])
