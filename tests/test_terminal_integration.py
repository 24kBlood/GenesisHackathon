"""Run trusted terminal scoring through the same environment used by policies."""

import json
import unittest
from copy import deepcopy
from unittest.mock import patch

import test_env_contract_and_isolation as contract
from genesis.environment.auditor_env import AuditorEnv
from genesis.environment.episode import build_episode
from genesis.evaluation.matcher import match


class TerminalIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = contract.DataContractTests
        cls.fixture.setUpClass()

    def setUp(self):
        self.now = 100.0
        self.episode_id = self.fixture.manifest["episode_id"]

    def factory(self, seed, config):
        truth = self.fixture.truth if not config.get("clean") else {
            "episode_id": self.episode_id, "issues": [],
        }
        return build_episode(
            task="Find the discrepancy",
            manifest=self.fixture.manifest,
            policy=self.fixture.policy,
            tables={name: getattr(self.fixture, name) for name in ("ledger", "invoices", "payments", "fx")},
            truth=truth,
            tolerances_minor=config.get("tolerances_minor"),
        )

    def make_env(self, *, mode="train", **config):
        env = AuditorEnv(self.factory, mode=mode, clock=lambda: self.now)
        env.reset(17, config)
        return env

    def submission(self, *, findings=None, episode_id=None):
        if findings is None:
            findings = self.fixture.findings["findings"]
        return {"type": "submit", "findings": {
            "episode_id": self.episode_id if episode_id is None else episode_id,
            "findings": deepcopy(findings),
        }}

    def assert_score(self, result, counts, metrics, reward):
        self.assertEqual((result.matching.tp, result.matching.fp, result.matching.fn), counts)
        actual = result.score.metrics
        self.assertEqual((actual.precision, actual.recall, actual.f1), metrics)
        self.assertAlmostEqual(result.score.reward, reward)

    def test_research_and_submit_real_fx_fixture(self):
        env = self.make_env()
        for step in self.fixture.trajectory[:-1]:
            with self.subTest(step=step["step"]):
                observation, reward, terminated, truncated, info = env.step(step["action"])
                self.assertEqual(observation, step["observation"])
                self.assertEqual((reward, terminated, truncated, info), (0.0, False, False, {}))
        self.now = 103.5
        self.assertEqual(
            env.step(self.submission()),
            ({"remaining_steps": 95, "status": "submitted"}, 0.992, True, False, {}),
        )
        result = env.terminal_result
        self.assert_score(result, (1, 0, 0), (1, 1, 1), 0.992)
        self.assertEqual(result.score.inspection_steps, 4)
        self.assertEqual(result.total_steps, 5)
        self.assertEqual(result.wall_time_seconds, 3.5)
        self.assertEqual(result.reason, "submitted")

    def test_duplicate_and_published_tolerance(self):
        findings = self.fixture.findings["findings"]
        env = self.make_env()
        env.step(self.submission(findings=findings * 2))
        self.assert_score(env.terminal_result, (1, 1, 0), (0.5, 1, 2 / 3), 1 / 3)
        finding = findings[0]
        plus_one = [{**finding, "claimed_amount_minor": finding["claimed_amount_minor"] + 1}]
        env = self.make_env()
        env.step(self.submission(findings=plus_one))
        self.assert_score(env.terminal_result, (0, 1, 1), (0, 0, 0), -1)
        env = self.make_env(tolerances_minor={finding["type"]: 1})
        env.step(self.submission(findings=plus_one))
        self.assert_score(env.terminal_result, (1, 0, 0), (1, 1, 1), 1)

    def test_empty_false_and_invalid_findings_on_clean_and_anomalous_episodes(self):
        valid_findings = self.fixture.findings["findings"]
        for clean, findings, expected_counts, metrics, reward in (
            (True, [], (0, 0, 0), (0, 0, 1), 1),
            (True, valid_findings, (0, 1, 0), (0, 0, 0), -1),
            (False, [], (0, 0, 1), (0, 0, 0), -1),
        ):
            with self.subTest(clean=clean, counts=expected_counts):
                env = self.make_env(clean=clean)
                self.assertEqual(env.step(self.submission(findings=findings))[1:4], (reward, True, False))
                self.assert_score(env.terminal_result, expected_counts, metrics, reward)
        for action in (
            self.submission(episode_id="other-episode"),
            self.submission(findings=valid_findings * 31),
            {"type": "submit", "findings": None},
        ):
            with self.subTest(action=action):
                env = self.make_env(clean=True)
                self.assertEqual(env.step(action)[1:4], (-1, True, False))
                result = env.terminal_result
                self.assertEqual(result.reason, "invalid_submission")
                self.assertEqual(result.submission, {"episode_id": self.episode_id, "findings": []})
                self.assertEqual(result.total_steps, 1)
                self.assertEqual(result.score.inspection_steps, 0)
                self.assert_score(result, (0, 0, 0), (0, 0, 0), -1)

    def test_step_limit_and_final_step_submit(self):
        for final, expected_reason, expected_steps, reward in (
            ({"type": "calculator", "expression": "1"}, "step_limit", 100, 0.8),
            (self.submission(findings=[]), "submitted", 99, 0.802),
        ):
            with self.subTest(reason=expected_reason):
                env = self.make_env(clean=True)
                for _ in range(99):
                    env.step({"type": "calculator", "expression": "1"})
                transition = env.step(final)
                self.assertEqual(transition[1], reward)
                self.assertEqual(transition[2], expected_reason == "submitted")
                self.assertEqual(transition[3], expected_reason == "step_limit")
                result = env.terminal_result
                self.assertEqual(result.reason, expected_reason)
                self.assertEqual(result.total_steps, 100)
                self.assertEqual(result.score.inspection_steps, expected_steps)
                self.assertEqual(result.submission["findings"], [])
                self.assertEqual(env.step(self.submission())[1], 0.0)
                self.assertEqual(env.terminal_result, result)

    def test_timeout_before_first_action_does_not_charge_steps(self):
        env = self.make_env()
        self.now += 60.0
        self.assertEqual(env.step(self.submission())[1:4], (-1, False, True))
        result = env.terminal_result
        self.assertEqual(result.total_steps, 0)
        self.assertEqual(result.score.inspection_steps, 0)
        self.assertEqual(result.wall_time_seconds, 60.0)
        self.assertEqual(result.reason, "timeout")
        self.assertEqual(result.submission["findings"], [])
        self.assert_score(result, (0, 0, 1), (0, 0, 0), -1)

    def test_eval_acknowledges_before_match_and_keeps_diagnostics_private(self):
        acknowledgements = []
        rewards = []
        for action in (
            self.submission(),
            self.submission(findings=[]),
            {"type": "submit", "findings": None},
        ):
            env = self.make_env(mode="eval")
            with patch("genesis.evaluation.terminal.match", wraps=match) as matching:
                self.now += 1.5
                acknowledgement = env.step(action)
                acknowledgements.append(json.dumps(acknowledgement, sort_keys=True))
                matching.assert_not_called()
                self.now += 20.0
                result = env.terminal_result
                matching.assert_called_once()
                self.assertEqual(result.wall_time_seconds, 1.5)
                rewards.append(result.score.reward)
                self.assertEqual(env.terminal_result, result)
                matching.assert_called_once()
            self.assertEqual(acknowledgement, ({"remaining_steps": 99, "status": "submitted"}, 0.0, True, False, {}))
        self.assertEqual(len(set(acknowledgements)), 1)
        self.assertEqual(rewards, [1, -1, -1])

    def test_factory_rejects_truth_from_another_episode(self):
        with self.assertRaisesRegex(ValueError, "same episode"):
            build_episode(
                task="Check the episode", manifest=self.fixture.manifest,
                policy=self.fixture.policy, tables={},
                truth={"episode_id": "other", "issues": []},
            )
