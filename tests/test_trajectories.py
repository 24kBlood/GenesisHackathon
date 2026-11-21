"""Integration tests and regeneration of the recorded public train trajectory."""

import asyncio
import hashlib
import json
import sys
import unittest
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path

import test_env_contract_and_isolation as contract
from genesis.domain.enums import FindingType
from genesis.environment.auditor_env import AuditorEnv
from genesis.environment.episode import build_episode
from genesis.evaluation.benchmark import LocalPolicyRunner
from genesis.evaluation.judge import run_episode
from genesis.evaluation.permutation import TABLES
from genesis.generation.company import generate_episode
from genesis.generation.datasets import load_dataset_entries
from genesis.policies import make_policy
from genesis.trajectories import TrajectoryRecorder, validate_trajectory


_ARTIFACT = contract.ROOT / "trajectories" / "trajectories.jsonl"
_TRAIN_CONFIGS = contract.ROOT / "data" / "configs"
_TRAIN_SEED = 20260110
_SEED = 413


def _train_entry():
    entries = load_dataset_entries(_TRAIN_CONFIGS)
    return next(entry for entry in entries if entry.split == "train" and entry.seed == _TRAIN_SEED)


def _train_env(entry, *, mode="train", clean=False, clock=None):
    episode = generate_episode(entry.seed, entry.config)
    public = episode.public
    truth = episode.evaluator.truth
    if clean:
        truth = {"episode_id": truth["episode_id"], "issues": []}
    options = {} if clock is None else {"clock": clock}
    return AuditorEnv(
        lambda _seed, _config: build_episode(
            task="Audit the public financial records and submit evidenced findings.",
            manifest=public.manifest,
            policy=public.policy,
            tables={name: getattr(public, name) for name in TABLES},
            truth=truth,
        ),
        mode=mode, **options,
    )


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
            tables={name: getattr(fixture, name) for name in TABLES},
            truth=truth,
        ),
        mode=mode, **options,
    )


async def _record_train(fixture, *, mode="train", clean=False):
    entry = _train_entry()
    recorder = TrajectoryRecorder(mode=mode)
    runner = LocalPolicyRunner(lambda: make_policy("simple_reconciliation"))
    result = await run_episode(
        _train_env(entry, mode=mode, clean=clean),
        seed=entry.seed, config=entry.config, runner=runner, on_step=recorder.record,
    )
    validate_trajectory(
        recorder.records, mode=mode, validate_shape=fixture.validators["trajectories"].validate,
    )
    return recorder, result, entry


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

    async def test_open_recorded_train_artifact_matches_real_run_and_replays(self):
        recorder, result, entry = await _record_train(self.fixture, mode="train")
        expected = recorder.to_jsonl()
        self.assertEqual(_ARTIFACT.read_bytes(), expected)
        rows = recorder.records
        self.assertEqual(len(rows), 11)
        self.assertEqual([row["step"] for row in rows], list(range(1, len(rows) + 1)))
        self.assertTrue(any(row["observation"].get("cursor") is not None for row in rows[:-1]))
        self.assertTrue(rows[-1]["terminated"])
        self.assertFalse(rows[-1]["truncated"])
        self.assertEqual(rows[-1]["reward"], 0.98)
        self.assertEqual(result.terminal.score.reward, 0.98)
        self.assertEqual((result.terminal.matching.tp, result.terminal.matching.fp, result.terminal.matching.fn), (8, 0, 0))
        self.assertTrue(all(row["reward"] == 0 for row in rows[:-1]))
        submitted_findings = rows[-1]["action"]["findings"]["findings"]
        self.assertEqual({item["type"] for item in submitted_findings}, {kind.value for kind in FindingType})
        self.assertTrue(all(item["explanation"] for item in submitted_findings))
        self.assertIn(b'"status":"submitted"', expected)
        self.assertNotIn(b"\r", expected)
        self.assertTrue(expected.endswith(b"\n"))
        self.assertTrue(contract.PRIVATE_KEYS.isdisjoint(contract.field_names(list(rows))))

        replay = TrajectoryRecorder(mode="train")
        actions = [json.loads(line)["action"] for line in expected.splitlines()]
        replay_result = await run_episode(
            _train_env(entry, mode="train"), seed=entry.seed, config=entry.config,
            runner=LocalPolicyRunner(lambda: _ScriptedPolicy(actions)), on_step=replay.record,
        )
        self.assertEqual(replay.to_jsonl(), expected)
        self.assertEqual(replay_result.terminal.matching, result.terminal.matching)
        snapshots = replay.records
        snapshots[0]["observation"].clear()
        self.assertEqual(replay.to_jsonl(), expected)

    async def test_closed_trajectories_do_not_depend_on_truth(self):
        first, actual, _ = await _record_train(self.fixture, mode="eval")
        second, other, _ = await _record_train(self.fixture, mode="eval", clean=True)
        self.assertEqual(first.to_jsonl(), second.to_jsonl())
        self.assertEqual(actual.terminal.score.reward, 0.98)
        self.assertEqual(other.terminal.score.reward, -1)
        self.assertTrue(all(row["reward"] == 0 for row in first.records))

    async def test_eval_mode_blanks_explanation_while_train_mode_preserves_it(self):
        train_rec, _ = await _record_fx(self.fixture, mode="train")
        eval_rec, _ = await _record_fx(self.fixture, mode="eval")
        train_findings = train_rec.records[-1]["action"]["findings"]["findings"]
        eval_findings = eval_rec.records[-1]["action"]["findings"]["findings"]
        self.assertTrue(all(item["explanation"] for item in train_findings))
        self.assertTrue(all(item["explanation"] == "" for item in eval_findings))
        # Eval validator rejects non-empty explanations.
        damaged = eval_rec.records
        damaged[-1]["action"]["findings"]["findings"][0]["explanation"] = "leaked text"
        with self.assertRaisesRegex(ValueError, "action"):
            validate_trajectory(damaged, mode="eval")

    def test_controlled_tool_errors_preserve_valid_tool_actions(self):
        episode_id = self.fixture.manifest["episode_id"]
        recorder = TrajectoryRecorder(mode="train")
        valid_error_steps = [
            (
                {"type": "get_document", "doc_id": "missing-doc"},
                ({"remaining_steps": 99, "error": {"code": "not_found"}}, 0.0, False, False, {}),
            ),
            (
                {"type": "calculator", "expression": "1 / 0"},
                ({"remaining_steps": 98, "error": {"code": "division_by_zero"}}, 0.0, False, False, {}),
            ),
            (
                {"type": "list_records", "table": "ledger", "cursor": "stale-cursor"},
                ({"remaining_steps": 97, "error": {"code": "invalid_cursor"}}, 0.0, False, False, {}),
            ),
            (
                {"type": "unknown_tool", "secret": "drop-me"},
                ({"remaining_steps": 96, "error": {"code": "invalid_action"}}, 0.0, False, False, {}),
            ),
            (
                {"type": "list_records", "table": "ledger", "limit": 999},
                ({"remaining_steps": 95, "error": {"code": "invalid_argument"}}, 0.0, False, False, {}),
            ),
        ]
        for step, (action, result) in enumerate(valid_error_steps, 1):
            recorder.record(episode_id, step, action, result)
        rows = recorder.records
        self.assertEqual(rows[0]["action"], {"type": "get_document", "doc_id": "missing-doc"})
        self.assertEqual(rows[1]["action"], {"type": "calculator", "expression": "1 / 0"})
        self.assertEqual(rows[2]["action"], {"type": "list_records", "table": "ledger", "cursor": "stale-cursor"})
        self.assertEqual(rows[3]["action"], {"type": "invalid"})
        self.assertEqual(rows[4]["action"], {"type": "invalid"})
        validate_trajectory(rows, mode="train", validate_shape=self.fixture.validators["trajectories"].validate)
        self.assertNotIn(b"drop-me", recorder.to_jsonl())

        # Incompatible tool + error code combination is rejected by validate_trajectory.
        mismatched = deepcopy(rows[:1])
        mismatched[0]["observation"]["error"]["code"] = "division_by_zero"
        with self.assertRaisesRegex(ValueError, "action"):
            validate_trajectory(mismatched, mode="train")

    async def test_run_episode_preserves_controlled_tool_errors_and_sanitizes_malformed_actions(self):
        actions = [
            {"type": "get_document", "doc_id": "nonexistent-doc"},
            {"type": "calculator", "expression": "1 / 0"},
            {"type": "list_records", "table": "ledger", "cursor": "stale-cursor"},
            {"type": "unknown", "secret": "drop-me"},
            {"type": "list_records", "table": "ledger", "limit": 999},
            {"type": "submit", "findings": {"episode_id": self.fixture.manifest["episode_id"], "findings": []}},
        ]
        recorder = TrajectoryRecorder(mode="train")
        result = await run_episode(
            _env(self.fixture, mode="train"), seed=_SEED, config={},
            runner=LocalPolicyRunner(lambda: _ScriptedPolicy(actions)), on_step=recorder.record,
        )
        rows = recorder.records
        self.assertEqual(rows[0]["action"], {"type": "get_document", "doc_id": "nonexistent-doc"})
        self.assertEqual(rows[0]["observation"]["error"], {"code": "not_found"})
        self.assertEqual(rows[1]["action"], {"type": "calculator", "expression": "1 / 0"})
        self.assertEqual(rows[1]["observation"]["error"], {"code": "division_by_zero"})
        self.assertEqual(rows[2]["action"], {"type": "list_records", "table": "ledger", "cursor": "stale-cursor"})
        self.assertEqual(rows[2]["observation"]["error"], {"code": "invalid_cursor"})
        self.assertEqual(rows[3]["action"], {"type": "invalid"})
        self.assertEqual(rows[3]["observation"]["error"], {"code": "invalid_action"})
        self.assertEqual(rows[4]["action"], {"type": "invalid"})
        self.assertEqual(rows[4]["observation"]["error"], {"code": "invalid_argument"})
        self.assertEqual(rows[5]["action"]["type"], "submit")
        self.assertFalse(result.terminal.invalid_submission)
        self.assertNotIn(b"drop-me", recorder.to_jsonl())
        validate_trajectory(rows, mode="train", validate_shape=self.fixture.validators["trajectories"].validate)

    async def test_invalid_research_and_invalid_submit_are_sanitized(self):
        actions = [
            {"type": "report", "seed": "Private value"},
            {"type": "get_document", "doc_id": ["Private path"]},
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
            [{"code": "invalid_action"}, {"code": "invalid_argument"}, None],
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
            {"type": "calculator", "expression": ["private string"]},
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

    async def test_validator_rejects_non_mapping_rows_and_multi_episode_violations(self):
        for bad_row in (None, [], 123, "not-a-dict"):
            with self.subTest(bad_row=bad_row):
                with self.assertRaisesRegex(ValueError, "invalid trajectory row"):
                    validate_trajectory([bad_row])

        # Contiguous multi-episode trajectories pass; non-contiguous episode IDs fail.
        first_rec, _, _ = await _record_train(self.fixture, mode="eval")
        second_rec, _ = await _record_fx(self.fixture, mode="eval")
        combined = (*first_rec.records, *second_rec.records)
        validate_trajectory(combined, mode="eval", validate_shape=self.fixture.validators["trajectories"].validate)

        interleaved = (first_rec.records[0], second_rec.records[0], first_rec.records[1])
        with self.assertRaisesRegex(ValueError, "contiguous"):
            validate_trajectory(interleaved, mode="eval")

        # Early truncation before step 100 is rejected.
        early_trunc = deepcopy(second_rec.records[:1])
        early_trunc[0].update({
            "observation": {"remaining_steps": 99, "status": "truncated"},
            "truncated": True,
        })
        with self.assertRaisesRegex(ValueError, "budget-ending"):
            validate_trajectory(early_trunc, mode="eval")


if __name__ == "__main__":
    if sys.argv[1:] == ["--export"]:
        fixture = contract.DataContractTests
        fixture.setUpClass()
        recorder, result, entry = asyncio.run(_record_train(fixture, mode="train"))
        data = recorder.to_jsonl()
        _ARTIFACT.parent.mkdir(exist_ok=True)
        _ARTIFACT.write_bytes(data)
        print(
            f"1 episode (train seed={entry.seed}, id={recorder.records[0]['episode_id']}), "
            f"{len(recorder.records)} steps, return {result.terminal.score.reward}"
        )
        print(f"SHA-256 {hashlib.sha256(data).hexdigest()}")
    else:
        unittest.main()
