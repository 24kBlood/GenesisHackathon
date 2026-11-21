"""Integration tests and regeneration of the recorded public FX sample."""

import asyncio
import hashlib
import json
import sys
import unittest
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path

import test_env_contract_and_isolation as contract
from genesis.environment.auditor_env import AuditorEnv
from genesis.environment.episode import build_episode
from genesis.evaluation.benchmark import LocalPolicyRunner
from genesis.evaluation.judge import run_episode
from genesis.policies import make_policy
from genesis.trajectories import TrajectoryRecorder, validate_trajectory


_ARTIFACT = contract.ROOT / "trajectories" / "trajectories.jsonl"
_SEED = 413


def _env(fixture, *, mode, clean=False, clock=None):
    truth = fixture.truth
    if clean:
        # Keep every public record identical and vary only evaluator truth.
        truth = {"episode_id": truth["episode_id"], "issues": []}
    options = {} if clock is None else {"clock": clock}
    return AuditorEnv(
        lambda _seed, _config: build_episode(
            task="Audit the public synthetic FX episode",
            manifest=fixture.manifest,
            policy=fixture.policy,
            tables={name: getattr(fixture, name) for name in ("ledger", "invoices", "payments", "fx")},
            truth=truth,
        ),
        mode=mode, **options,
    )


async def _record_fx(fixture, mode="train", clean=False):
    recorder = TrajectoryRecorder(mode=mode)
    runner = LocalPolicyRunner(lambda: make_policy("simple_reconciliation"))
    result = await run_episode(
        _env(fixture, mode=mode, clean=clean), seed=_SEED, config={}, runner=runner,
        on_step=recorder.record,
    )
    validate_trajectory(
        recorder.records, mode=mode, validate_shape=fixture.validators["trajectories"].validate,
    )
    return recorder, result


class _ScriptedPolicy:
    def __init__(self, actions):
        self._actions = iter(actions)

    def act(self, observation, previous_action):
        return next(self._actions)


class TrajectoryTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = contract.DataContractTests
        cls.fixture.setUpClass()

    async def test_open_recorded_fx_artifact_matches_real_run_and_replays(self):
        recorder, result = await _record_fx(self.fixture)
        expected = recorder.to_jsonl()
        self.assertEqual(_ARTIFACT.read_bytes(), expected)
        self.assertEqual(len(recorder.records), 5)
        self.assertEqual([row["step"] for row in recorder.records], [1, 2, 3, 4, 5])
        self.assertTrue(recorder.records[-1]["terminated"])
        self.assertFalse(recorder.records[-1]["truncated"])
        self.assertEqual(recorder.records[-1]["reward"], 0.992)
        self.assertEqual(result.terminal.score.reward, 0.992)
        self.assertTrue(all(row["reward"] == 0 for row in recorder.records[:-1]))
        self.assertIn(b'"status":"submitted"', expected)
        self.assertNotIn(b"\r", expected)
        self.assertTrue(expected.endswith(b"\n"))

        replay = TrajectoryRecorder(mode="train")
        actions = [json.loads(line)["action"] for line in expected.splitlines()]
        replay_result = await run_episode(
            _env(self.fixture, mode="train"), seed=_SEED, config={},
            runner=LocalPolicyRunner(lambda: _ScriptedPolicy(actions)), on_step=replay.record,
        )
        self.assertEqual(replay.to_jsonl(), expected)
        self.assertEqual(replay_result.terminal.matching, result.terminal.matching)
        snapshots = replay.records
        snapshots[0]["observation"].clear()
        self.assertEqual(replay.to_jsonl(), expected)

    async def test_closed_trajectories_do_not_depend_on_truth(self):
        first, actual = await _record_fx(self.fixture, mode="eval")
        second, other = await _record_fx(self.fixture, mode="eval", clean=True)
        self.assertEqual(first.to_jsonl(), second.to_jsonl())
        self.assertEqual(actual.terminal.score.reward, 0.992)
        self.assertEqual(other.terminal.score.reward, -1)
        self.assertTrue(all(row["reward"] == 0 for row in first.records))

    async def test_invalid_research_and_invalid_submit_are_sanitized(self):
        actions = [
            {"type": "report", "seed": "Private value"},
            {"type": "get_document", "doc_id": "Missing private path"},
            {"type": "submit", "findings": {"truth": "Private document"}},
        ]
        recorder = TrajectoryRecorder(mode="train")
        result = await run_episode(
            _env(self.fixture, mode="train"), seed=_SEED, config={},
            runner=LocalPolicyRunner(lambda: _ScriptedPolicy(actions)), on_step=recorder.record,
        )
        rows = recorder.records
        self.assertEqual([row["action"] for row in rows], [{"type": "invalid"}] * 3)
        self.assertEqual(
            [row["observation"].get("error") for row in rows],
            [{"code": "invalid_action"}, {"code": "not_found"}, None],
        )
        self.assertEqual([row["terminated"] for row in rows], [False, False, True])
        self.assertEqual(rows[-1]["observation"]["status"], "submitted")
        self.assertEqual(rows[-1]["reward"], -1)
        self.assertTrue(result.terminal.invalid_submission)
        self.assertNotIn(b"Private", recorder.to_jsonl())
        validate_trajectory(rows, mode="train", validate_shape=self.fixture.validators["trajectories"].validate)

    async def test_budget_end_records_only_the_hundredth_accepted_action(self):
        recorder = TrajectoryRecorder()
        actions = [{"type": "calculator", "expression": "1+2"}] * 99 + [
            {"type": "calculator", "expression": "private string"},
        ]
        result = await run_episode(
            _env(self.fixture, mode="eval"), seed=_SEED, config={},
            runner=LocalPolicyRunner(lambda: _ScriptedPolicy(actions)), on_step=recorder.record,
        )
        final = recorder.records[-1]
        self.assertEqual(len(recorder.records), 100)
        self.assertEqual(final["step"], 100)
        self.assertEqual(final["observation"], {"remaining_steps": 0, "status": "truncated"})
        self.assertTrue(final["truncated"])
        self.assertFalse(final["terminated"])
        self.assertEqual(final["action"], {"type": "invalid"})
        self.assertEqual(result.terminal.reason, "step_limit")
        self.assertNotIn(b"private string", recorder.to_jsonl())

    async def test_timeout_does_not_invent_an_action(self):
        for accepted in (0, 1):
            now = 0.0
            observed = 0

            class ClockedChannel:
                async def send(self, message, deadline):
                    pass

                async def receive(self, deadline):
                    nonlocal now, observed
                    if observed == accepted:
                        now = 60.0
                    observed += 1
                    return {"type": "calculator", "expression": "1+2"}

                @asynccontextmanager
                async def open(self, deadline):
                    yield self

            with self.subTest(accepted=accepted):
                recorder = TrajectoryRecorder()
                result = await run_episode(
                    _env(self.fixture, mode="eval", clock=lambda: now), seed=_SEED, config={},
                    runner=ClockedChannel(), on_step=recorder.record,
                )
                self.assertEqual(len(recorder.records), accepted)
                self.assertEqual(result.terminal.total_steps, accepted)
                self.assertEqual(result.terminal.reason, "timeout")
                self.assertFalse(any(row["truncated"] for row in recorder.records))
                self.assertEqual(len(recorder.to_jsonl().splitlines()), accepted)

    def test_projection_omits_nested_private_fields_and_info(self):
        episode_id = self.fixture.manifest["episode_id"]
        invoice = deepcopy(self.fixture.invoices[0])
        invoice["truth"] = "hidden"
        invoice["line_items"][0]["seed"] = "hidden"
        recorder = TrajectoryRecorder()
        recorder.record(
            episode_id, 1, {"type": "get_document", "doc_id": invoice["doc_id"]},
            ({"remaining_steps": 99, "document": invoice, "provenance": "hidden"},
             0, False, False, {"detail": "hidden"}),
        )
        data = recorder.to_jsonl()
        self.assertNotIn(b"hidden", data)
        validate_trajectory(recorder.records, validate_shape=self.fixture.validators["trajectories"].validate)
        bad = recorder.records
        bad[0]["observation"]["document"]["line_items"][0]["truth"] = "hidden"
        with self.assertRaisesRegex(ValueError, "private"):
            validate_trajectory(bad)

    async def test_validator_rejects_nonzero_eval_and_impossible_sequences(self):
        recorder, _ = await _record_fx(self.fixture, mode="eval")
        for index, fields, message in (
            (0, {"reward": 1}, "reward"),
            (4, {"reward": 0.992}, "reward"),
            (1, {"step": 3}, "sequence"),
            (4, {"truncated": True}, "terminal fields"),
        ):
            with self.subTest(fields=fields):
                damaged = recorder.records
                damaged[index].update(fields)
                with self.assertRaisesRegex(ValueError, message):
                    validate_trajectory(damaged)
        with self.assertRaisesRegex(ValueError, "sequence"):
            validate_trajectory((*recorder.records, recorder.records[-1]))


if __name__ == "__main__":
    if sys.argv[1:] == ["--export"]:
        fixture = contract.DataContractTests
        fixture.setUpClass()
        recorder, result = asyncio.run(_record_fx(fixture))
        data = recorder.to_jsonl()
        _ARTIFACT.parent.mkdir(exist_ok=True)
        _ARTIFACT.write_bytes(data)
        print(f"1 episode, {len(recorder.records)} steps, return {result.terminal.score.reward}")
        print(f"SHA-256 {hashlib.sha256(data).hexdigest()}")
    else:
        unittest.main()
