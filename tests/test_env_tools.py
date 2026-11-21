import unittest
from copy import deepcopy

from genesis.environment.auditor_env import AuditorEnv, Episode, ToolCode, ToolError
from genesis.environment.calculator import calculate
from genesis.environment.tools import PublicTools
from genesis.evaluation.matcher import match
from genesis.evaluation.reward import score_terminal

import test_env_contract_and_isolation as contract


class PublicToolsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = contract.DataContractTests
        cls.fixture.setUpClass()

    def setUp(self):
        self.tables = {
            "ledger": deepcopy(self.fixture.ledger),
            "invoices": deepcopy(self.fixture.invoices),
            "payments": deepcopy(self.fixture.payments),
            "fx": deepcopy(self.fixture.fx),
        }
        self.policy = deepcopy(self.fixture.policy)
        self.episode_id = self.fixture.manifest["episode_id"]

    def make_tools(self, episode_id=None):
        return PublicTools(episode_id=episode_id or self.episode_id, policy=self.policy, tables=self.tables)

    def make_env(self, *, mode="train"):
        def factory(_seed, config):
            episode_id = config.get("episode_id", self.episode_id)
            toolset = self.make_tools(episode_id)

            def score(submission, inspection_steps, *, invalid_submission):
                matches = match(submission["findings"], self.fixture.truth["issues"], toolset.public_evidence_ids)
                return score_terminal(
                    matches.tp, matches.fp, matches.fn, inspection_steps,
                    invalid_submission=invalid_submission,
                )

            manifest = {
                **self.fixture.manifest, "episode_id": episode_id,
                "record_counts": {name: len(rows) for name, rows in self.tables.items()},
            }
            return Episode("Сверьте записи", manifest, self.policy, toolset, score)

        env = AuditorEnv(factory, mode=mode, clock=lambda: 0)
        env.reset(0, {})
        return env

    def observation(self, env, action):
        observation, reward, terminated, truncated, info = env.step(action)
        self.fixture.validators["observation"].validate(observation)
        self.assertEqual((reward, terminated, truncated, info), (0, False, False, {}))
        return observation

    def test_C06_pages_cover_rows_in_stable_order(self):
        self.tables["ledger"] = [{**self.tables["ledger"][0], "row_id": f"row-{i}"} for i in range(123)]
        env = self.make_env()
        action = {"type": "list_records", "table": "ledger"}
        first = self.observation(env, action)
        repeated = self.observation(env, {**action, "cursor": None, "limit": 50})
        self.assertEqual((first["records"], first["cursor"]), (repeated["records"], repeated["cursor"]))
        second = self.observation(env, {**action, "cursor": first["cursor"]})
        third = self.observation(env, {**action, "cursor": second["cursor"]})
        self.assertEqual([len(page["records"]) for page in (first, second, third)], [50, 50, 23])
        self.assertIsNone(third["cursor"])
        combined = [row for page in (first, second, third) for row in page["records"]]
        self.assertEqual(combined, self.tables["ledger"])
        self.assertEqual(len(self.observation(env, {**action, "limit": 1})["records"]), 1)

    def test_C07_empty_table_invalid_limits_and_nontransferable_cursors(self):
        self.tables["fx"] = []
        env = self.make_env()
        action = {"type": "list_records", "table": "ledger", "limit": 1}
        first = self.observation(env, action)
        cursor = first["cursor"]
        empty = self.observation(env, {**action, "table": "fx"})
        self.assertEqual((empty["records"], empty["cursor"]), ([], None))
        rejects = [
            *[({**action, "limit": limit}, "invalid_argument") for limit in (0, 51, True, "1")],
            ({**action, "table": "other"}, "invalid_argument"),
            ({**action, "table": "payments", "cursor": cursor}, "invalid_cursor"),
            ({**action, "cursor": "unknown"}, "invalid_cursor"),
        ]
        for step, (bad_action, code) in enumerate(rejects, start=3):
            self.assertEqual(self.observation(env, bad_action), {
                "remaining_steps": 100 - step, "error": {"code": code},
            })
        env.reset(0, {"episode_id": "other"})
        self.assertNotEqual(self.observation(env, action)["cursor"], cursor)
        self.assertEqual(self.observation(env, {**action, "cursor": cursor})["error"]["code"], "invalid_cursor")

    def test_C09_index_covers_each_table_docs_and_evidence_catalog(self):
        env = self.make_env()
        keys = {
            "ledger": self.tables["ledger"][0]["row_id"],
            "invoices": self.tables["invoices"][0]["invoice_id"],
            "payments": self.tables["payments"][0]["payment_id"],
            "fx": "fx:2026-11-11:USD:KZT",
        }
        for table, row_id in keys.items():
            action = {"type": "get_record", "table": table, "id": row_id}
            found = self.observation(env, action)
            self.assertEqual(found["record"], self.tables[table][0])
            self.assertEqual(self.observation(env, {**action, "id": "missing"}), {
                "remaining_steps": found["remaining_steps"] - 1, "error": {"code": "not_found"},
            })
        for document in (*self.tables["invoices"], *self.tables["payments"]):
            self.assertEqual(
                self.observation(env, {"type": "get_document", "doc_id": document["doc_id"]})["document"],
                document,
            )
        self.assertEqual(self.observation(env, {"type": "get_document", "doc_id": "policy"})["document"], self.policy)
        self.assertEqual(self.observation(env, {"type": "get_document", "doc_id": "truth.json"})["error"]["code"], "not_found")
        evidence = self.make_tools().public_evidence_ids
        self.assertIn(keys["fx"], evidence)
        self.assertIn("policy", evidence)
        self.assertIn(keys["ledger"], evidence)
        self.assertNotIn(keys["payments"], evidence)
        self.assertIn(self.tables["payments"][0]["doc_id"], evidence)

    def test_table_snapshot_is_detached_and_cursor_is_scoped_to_it(self):
        tools = self.make_tools()
        action = {"type": "list_records", "table": "ledger", "limit": 1}
        response = tools(action)
        row = self.tables["ledger"][0]
        self.assertEqual(response["records"][0], row)
        response["records"][0]["debit_minor"] = 0
        row["debit_minor"] += 1
        self.assertNotEqual(tools(action)["records"][0]["debit_minor"], 0)
        changed = self.make_tools()
        self.assertNotEqual(changed(action)["cursor"], response["cursor"])
        with self.assertRaises(ToolError) as error:
            changed({**action, "cursor": response["cursor"]})
        self.assertEqual(error.exception.code, ToolCode.INVALID_CURSOR)

    def test_calculator_errors_spend_one_step_and_submit_uses_original_budget(self):
        env = self.make_env(mode="eval")
        for index, (expression, code) in enumerate((
            ("rate(5)", "invalid_expression"), ("(1+2", "invalid_expression"),
            ("4 / (2 - 2)", "division_by_zero"), ("9" * 35, "calculation_limit"),
        ), 1):
            action = {"type": "calculator", "expression": expression}
            self.assertEqual(self.observation(env, action), {
                "remaining_steps": 100 - index, "error": {"code": code},
            })
        self.assertEqual(
            self.observation(env, {"type": "calculator", "expression": "0.1 + 0.2"})["result"],
            "0.3",
        )
        self.assertEqual(
            self.observation(env, {"type": "unrecognized"})["error"]["code"], "invalid_action",
        )
        reply = env.step({"type": "submit", "findings": self.fixture.findings})
        self.assertEqual(reply, ({"remaining_steps": 93, "status": "submitted"}, 0.0, True, False, {}))
        self.assertEqual(env.terminal_result.score.inspection_steps, 6)
        self.assertEqual(env.terminal_result.score.reward, 0.988)
        env.reset(0, {})
        self.assertEqual(
            env.step({"type": "submit", "findings": []}),
            ({"remaining_steps": 99, "status": "submitted"}, 0.0, True, False, {}),
        )
        self.assertEqual(env.terminal_result.score.reward, -1)


class CalculatorTests(unittest.TestCase):
    def test_fixed_decimal_arithmetic(self):
        for text, expected in {
            "  -(2 + .5) * +4 ": "-10",
            "1+2*3": "7",
            "9/3/3": "1",
            "0.1+0.20": "0.3",
            "-0.00": "0",
            "1/3": "0." + "3" * 34,
            "9" * 34: "9" * 34,
            "(" * 16 + "1" + ")" * 16: "1",
            "1+" * 64 + "1": "65",
            " " * 255 + "1": "1",
        }.items():
            with self.subTest(text=text):
                self.assertEqual(calculate(text), expected)

    def test_decimal_limits_and_grammar(self):
        for text in (
            "9" * 35, "(" * 17 + "1" + ")" * 17, "1+" * 65 + "1", " " * 256 + "1",
            ("9" * 34) + "*" + ("9" * 34),
            "1 / " + ("9" * 34) + " / " + ("9" * 34),
        ):
            with self.subTest(text=text):
                with self.assertRaises(OverflowError):
                    calculate(text)
        for text in ("", "1 2", "2(3)", "1e2", "1,2", "NaN", "1**2", "1 /", "9 %% 2"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    calculate(text)
        with self.assertRaises(ZeroDivisionError):
            calculate("1 / -0.0")
