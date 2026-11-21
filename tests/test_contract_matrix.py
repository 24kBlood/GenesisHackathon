"""Fixed contract and isolation checks C01–C20 (see QA_CHECKLIST.md)."""

import json
from copy import deepcopy
from unittest import TestCase

from genesis.environment.auditor_env import AuditorEnv
from genesis.environment.episode import build_episode

import test_env_contract_and_isolation as contract


class FixtureCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = contract.DataContractTests
        cls.fixture.setUpClass()

    def setUp(self):
        self.now = 100.0
        self.tables = {name: deepcopy(getattr(self.fixture, name)) for name in ("ledger", "fx", "invoices", "payments")}
        self.manifest = deepcopy(self.fixture.manifest)
        self.policy = deepcopy(self.fixture.policy)
        self.episode_id = self.manifest["episode_id"]
        self.env = AuditorEnv(self._episode, clock=lambda: self.now)

    def _episode(self, seed, config):
        episode_id = config.get("episode_id", self.episode_id)
        return build_episode(
            task="Review the synthetic books",
            manifest={
                **self.manifest, "episode_id": episode_id,
                "record_counts": {name: len(rows) for name, rows in self.tables.items()},
            },
            policy=self.policy,
            tables=self.tables,
            truth={**self.fixture.truth, "episode_id": episode_id,
                   "issues": [] if config.get("clean") else self.fixture.truth["issues"]},
        )

    def reset(self, **config):
        return self.env.reset(41, config)

    def submit(self, findings=None):
        return self.env.step({
            "type": "submit", "findings": {"episode_id": self.episode_id, "findings": findings or []},
        })

    def steps(self, n):
        for _ in range(n):
            self.env.step({"type": "calculator", "expression": "1+2"})


class ContractMatrix(FixtureCase):
    def test_C01_reset_shape(self):
        result = self.reset()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        observation, info = result
        self.fixture.validators["observation"].validate(observation)
        self.assertEqual((observation["remaining_steps"], info), (100, {}))

    def test_C02_step_shape(self):
        self.reset()
        result = self.env.step({"type": "calculator", "expression": "1+2"})
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 5)
        observation, reward, terminated, truncated, info = result
        self.fixture.validators["observation"].validate(observation)
        self.assertIs(type(terminated), bool)
        self.assertIs(type(truncated), bool)
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), list(result))
        self.assertEqual((reward, terminated, truncated, info), (0, False, False, {}))

    def test_C03_reset_clears_budget_terminal_and_cursors(self):
        self.reset()
        first = self.env.step({"type": "list_records", "table": "ledger", "limit": 1})[0]
        self.submit()
        self.assertIsNotNone(self.env.terminal_result)
        initial, _ = self.reset(episode_id="next-episode")
        self.assertIsNone(self.env.terminal_result)
        self.assertEqual(initial["remaining_steps"], 100)
        stale = self.env.step({"type": "list_records", "table": "ledger", "cursor": first["cursor"]})[0]
        self.assertEqual(stale, {"remaining_steps": 99, "error": {"code": "invalid_cursor"}})

    def test_C04_same_public_inputs_replay(self):
        def replay():
            initial = self.reset()
            steps = [self.env.step(row["action"]) for row in self.fixture.trajectory]
            return json.dumps([initial, steps], sort_keys=True, allow_nan=False)

        self.assertEqual(replay(), replay())

    def test_C05_maximum_page_size(self):
        self.tables["ledger"] = [{**self.tables["ledger"][0], "row_id": f"row-{i}"} for i in range(51)]
        self.reset()
        page = self.env.step({"type": "list_records", "table": "ledger", "limit": 50})[0]
        self.assertEqual(len(page["records"]), 50)
        self.assertIsNotNone(page["cursor"])

    def test_C08_table_and_limit_validation(self):
        self.reset()
        for step, arguments in enumerate(({"table": "unknown"}, {"table": "ledger", "limit": 51}), 1):
            action = {"type": "list_records", **arguments}
            self.assertFalse(self.fixture.validators["action"].is_valid(action))
            self.assertEqual(self.env.step(action)[0], {
                "remaining_steps": 100 - step, "error": {"code": "invalid_argument"},
            })

    def test_C10_only_public_documents_are_addressable(self):
        self.reset()
        for doc_id, expected in (
            ("policy", self.policy),
            (self.tables["invoices"][0]["doc_id"], self.tables["invoices"][0]),
            ("unpublished-document", None),
        ):
            observation = self.env.step({"type": "get_document", "doc_id": doc_id})[0]
            if expected is None:
                self.assertEqual(observation["error"], {"code": "not_found"})
            else:
                self.assertEqual(observation["document"], expected)

    def test_C11_calculator_public_decimal_and_grammar(self):
        self.reset()
        for expression in ("0.1 + .2", "(.3 * 4) / 4"):
            self.assertEqual(self.env.step({"type": "calculator", "expression": expression})[0]["result"], "0.3")
        for expression in ("pi", "2,3"):
            self.assertEqual(self.env.step({"type": "calculator", "expression": expression})[0]["error"],
                             {"code": "invalid_expression"})

    def test_C12_unrecognized_action_and_wrong_types_are_bounded(self):
        self.reset()
        for index, (action, code) in enumerate((
            ({"type": "unknown"}, "invalid_action"),
            ({"type": []}, "invalid_action"),
            ({"type": "get_document", "doc_id": ["policy"]}, "invalid_argument"),
        ), 1):
            self.assertEqual(self.env.step(action)[0], {"remaining_steps": 100 - index, "error": {"code": code}})

    def test_C16_late_requests_preserve_the_completed_episode(self):
        self.reset()
        final = self.submit(self.fixture.findings["findings"])
        before = self.env.terminal_result
        for action in ({"type": "get_document", "doc_id": "policy"}, {"type": "submit"}):
            observation, reward, terminated, truncated, info = self.env.step(action)
            self.assertEqual((observation, reward, terminated, truncated, info), (final[0], 0, True, False, {}))
            self.assertEqual(self.env.terminal_result, before)
        self.assertEqual(before.total_steps, 1)

    def test_C20_private_synthetic_metadata_stays_out_of_public_projections(self):
        marker = "/test-only/evaluator/hidden-fixture"
        self.manifest.update(seed=41, internal_path=marker)
        self.policy.update(truth={"path": marker}, debug=marker)
        self.tables["invoices"][0]["provenance"] = {"path": marker}
        initial, info = self.reset()
        document = self.env.step({"type": "get_document", "doc_id": self.tables["invoices"][0]["doc_id"]})
        policy = self.env.step({"type": "get_document", "doc_id": "policy"})
        public = [initial, info, document, policy]
        self.assertNotIn(marker, json.dumps(public))
        self.assertTrue(contract.PRIVATE_KEYS.isdisjoint(contract.field_names(public)))
