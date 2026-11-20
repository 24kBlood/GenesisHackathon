import json
import unittest
from copy import deepcopy
from dataclasses import replace

from genesis.environment.auditor_env import AuditorEnv, Episode, ToolCode, ToolError
from genesis.evaluation.matcher import match
from genesis.evaluation.reward import score_terminal

import test_env_contract_and_isolation as contract


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


class LifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = contract.DataContractTests
        cls.fixture.setUpClass()

    def setUp(self):
        self.clock = Clock()
        self.handled = []
        self.scored = []
        self.episode_id = self.fixture.manifest["episode_id"]

    def factory(self, seed, config):
        episode_id = config.get("episode_id", self.episode_id)
        manifest = {**self.fixture.manifest, "episode_id": episode_id, "seed": seed}
        policy = {**self.fixture.policy, "provenance": {"seed": seed}}
        docs = {document["doc_id"]: document for document in self.fixture.invoices + self.fixture.payments}
        docs["policy"] = policy
        evidence = self.fixture.findings["findings"][0]["evidence_ids"]

        def handle(action):
            self.handled.append(action)
            if action["type"] == "get_document":
                if action["doc_id"] not in docs:
                    raise ToolError(ToolCode.NOT_FOUND)
                return {"document": docs[action["doc_id"]], "truth": self.fixture.truth}
            if action["type"] == "list_records":
                cursor = f"{episode_id}-page-two" if action.get("cursor") is None else None
                return {"records": self.fixture.ledger[:1], "cursor": cursor}
            return {"result": "1"}

        def evaluate(submission, inspection_steps, *, invalid_submission):
            self.scored.append((submission, inspection_steps, invalid_submission))
            truth = self.fixture.truth["issues"] if not config.get("clean") else []
            outcome = match(submission["findings"], truth, evidence)
            return score_terminal(
                outcome.tp, outcome.fp, outcome.fn, inspection_steps,
                invalid_submission=invalid_submission,
            )

        return Episode("Найдите расхождение", manifest, policy, handle, evaluate)

    def make_env(self, *, mode="train", config=None, factory=None):
        env = AuditorEnv(factory or self.factory, mode=mode, clock=self.clock)
        initial, info = env.reset(121, config or {})
        self.fixture.validators["observation"].validate(initial)
        self.assertEqual(initial["remaining_steps"], 100)
        self.assertEqual(info, {})
        self.assertTrue(contract.PRIVATE_KEYS.isdisjoint(contract.field_names(initial)))
        return env

    def step(self, env, action):
        value = env.step(action)
        self.assertEqual(json.loads(json.dumps(value, allow_nan=False)), list(value))
        self.fixture.validators["observation"].validate(value[0])
        self.assertTrue(contract.PRIVATE_KEYS.isdisjoint(contract.field_names(value)))
        self.assertEqual(value[4], {})
        return value[:4]

    def submit(self, payload=None):
        return {"type": "submit", "findings": deepcopy(self.fixture.findings if payload is None else payload)}

    def test_train_replay_uses_shared_matcher_and_score(self):
        env = self.make_env()
        actions = [step["action"] for step in self.fixture.trajectory]
        first = [self.step(env, action) for action in actions]
        self.assertEqual(first[-1][1:], (0.992, True, False))
        self.assertEqual(env.terminal_result.score.metrics.f1, 1)
        self.assertEqual(env.terminal_result.score.inspection_steps, 4)
        env.reset(121, {})
        self.assertIsNone(env.terminal_result)
        self.assertEqual(first, [self.step(env, action) for action in actions])

    def test_final_step_accepts_submit(self):
        env = self.make_env()
        for _ in range(99):
            observation, reward, terminated, truncated, _ = env.step({"type": "calculator", "expression": "1"})
            self.assertEqual((reward, terminated, truncated), (0, False, False))
        self.assertEqual(observation["remaining_steps"], 1)
        final = self.step(env, self.submit())
        self.assertEqual(final, ({"remaining_steps": 0, "status": "submitted"}, 0.802, True, False))
        self.assertEqual(env.terminal_result.score.inspection_steps, 99)

    def test_budget_exhaustion_scores_empty_and_ignores_late_submit(self):
        env = self.make_env(config={"clean": True})
        for _ in range(99):
            env.step({"type": "unknown"})
        self.assertEqual(
            self.step(env, {"type": "unknown"}),
            ({"remaining_steps": 0, "status": "truncated"}, 0.8, False, True),
        )
        self.assertEqual(env.terminal_result.submission["findings"], [])
        self.assertEqual(env.terminal_result.score.inspection_steps, 100)
        self.assertEqual(self.step(env, self.submit())[1:], (0, False, True))
        self.assertEqual(len(self.scored), 1)

    def test_exact_deadline_precedes_action_and_does_not_debit(self):
        env = self.make_env()
        self.clock.now = 60
        self.assertEqual(
            self.step(env, self.submit()),
            ({"remaining_steps": 100, "status": "truncated"}, -1, False, True),
        )
        self.assertEqual(self.handled, [])
        self.assertEqual(env.terminal_result.score.inspection_steps, 0)
        self.assertFalse(env.terminal_result.invalid_submission)

    def test_invalid_submit_still_terminates_once(self):
        bad_findings = [
            {},
            {**self.fixture.findings, "episode_id": "other"},
            {**self.fixture.findings, "findings": self.fixture.findings["findings"] * 31},
        ]
        baseline = self.fixture.findings["findings"][0]
        for changes in (
            {"type": "unknown"}, {"affected_ids": []}, {"claimed_amount_minor": True},
            {"claimed_amount_minor": 10000.0}, {"currency": "usd"},
            {"evidence_ids": ["id"] * 11}, {"explanation": "я" * 1001},
            {"extra": "private"},
        ):
            bad_findings.append({**self.fixture.findings, "findings": [{**baseline, **changes}]})
        env = self.make_env(config={"clean": True})
        for payload in bad_findings:
            with self.subTest(payload=payload):
                env.reset(121, {"clean": True})
                self.assertEqual(self.step(env, self.submit(payload))[1:], (-1, True, False))
                self.assertTrue(env.terminal_result.invalid_submission)
                self.assertEqual(env.terminal_result.submission["findings"], [])
                self.assertEqual(self.step(env, self.submit())[1:], (0, True, False))
        self.assertEqual(len(self.scored), len(bad_findings))

    def test_eval_acknowledgement_precedes_scoring_and_keeps_private_result(self):
        env = self.make_env(mode="eval")
        acknowledgement = self.step(env, self.submit())
        self.assertEqual(acknowledgement, ({"remaining_steps": 99, "status": "submitted"}, 0, True, False))
        self.assertEqual(self.scored, [])
        self.assertEqual(env.terminal_result.score.reward, 1)
        self.assertEqual(len(self.scored), 1)
        self.assertEqual(self.step(env, self.submit()), acknowledgement)
        self.assertEqual(env.terminal_result.score.reward, 1)
        self.assertEqual(len(self.scored), 1)
        env.reset(121, {"clean": True})
        self.assertEqual(self.step(env, self.submit()), acknowledgement)
        self.assertEqual(env.terminal_result.score.reward, -1)
        env.reset(121, {})
        self.assertEqual(self.step(env, {"type": "submit"}), acknowledgement)
        self.assertTrue(env.terminal_result.invalid_submission)

    def test_error_codes_and_cursor_reset(self):
        env = self.make_env()
        page = {"type": "list_records", "table": "ledger", "limit": 1}
        first, *_ = self.step(env, page)
        cursor = first["cursor"]
        self.assertIn("records", self.step(env, {**page, "cursor": cursor})[0])
        env.reset(222, {"episode_id": "ep-other"})
        for action, code, budget in (
            ({**page, "cursor": cursor}, "invalid_cursor", 99),
            ({"type": ["secret"]}, "invalid_action", 98),
            ({"type": "list_records", "table": "ledger", "limit": True}, "invalid_argument", 97),
            ({"type": "get_document", "doc_id": "private"}, "not_found", 96),
        ):
            self.assertEqual(self.step(env, action), ({"remaining_steps": budget, "error": {"code": code}}, 0, False, False))
        self.step(env, self.submit())
        self.assertEqual(env.terminal_result.score.inspection_steps, 4)

    def test_public_projection_is_detached_and_handler_failures_are_bounded(self):
        bad_document = {**self.fixture.invoices[0], "doc_id": {"seed": 42}}

        def broken(_):
            raise ValueError("private stack diagnostic")

        for handler in (broken, lambda _: {"document": bad_document}):
            def factory(seed, config):
                return replace(self.factory(seed, config), handle_action=handler)

            env = self.make_env(factory=factory)
            self.assertEqual(
                self.step(env, {"type": "get_document", "doc_id": "known"})[0],
                {"remaining_steps": 99, "error": {"code": "tool_error"}},
            )
        env = self.make_env()
        action = {"type": "get_document", "doc_id": self.fixture.invoices[0]["doc_id"]}
        observation, *_ = self.step(env, action)
        observation["document"]["line_items"].clear()
        self.assertTrue(self.step(env, action)[0]["document"]["line_items"])
        env.step(self.submit())
        env.terminal_result.submission["findings"].clear()
        self.assertTrue(env.terminal_result.submission["findings"])

    def test_projection_preserves_schema_fields_and_excludes_private_metadata(self):
        policy = {**self.fixture.policy, "recognition_date_rule": "document_date", "seed": 42}
        refund = {
            **self.fixture.payments[0], "payment_kind": "refund", "refund_of_payment_id": "pay-initial",
            "provenance": {"seed": 42},
        }

        def factory(seed, config):
            return replace(
                self.factory(seed, config), policy=policy,
                handle_action=lambda action: {"document": policy if action["doc_id"] == "policy" else refund},
            )

        env = AuditorEnv(factory, clock=self.clock)
        initial, _ = env.reset(121, {})
        self.fixture.validators["observation"].validate(initial)
        rule, *_ = self.step(env, {"type": "get_document", "doc_id": "policy"})
        document, *_ = self.step(env, {"type": "get_document", "doc_id": "refund"})
        self.assertEqual(initial["policy"], {**self.fixture.policy, "recognition_date_rule": "document_date"})
        self.assertEqual(rule["document"], initial["policy"])
        self.assertEqual(document["document"], {
            **self.fixture.payments[0], "payment_kind": "refund", "refund_of_payment_id": "pay-initial",
        })
        self.assertTrue(contract.PRIVATE_KEYS.isdisjoint(contract.field_names([initial, rule, document])))
