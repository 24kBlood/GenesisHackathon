"""Acceptance tests and a reproducible local baseline smoke comparison."""

import asyncio
import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import genesis
import test_env_contract_and_isolation as contract
from genesis.environment.auditor_env import AuditorEnv
from genesis.environment.docker_policy import _host_environment
from genesis.environment.episode import build_episode
from genesis.environment.jsonl import FRAME_BYTES, JsonlProcess
from genesis.environment.tools import PublicTools
from genesis.evaluation.benchmark import BenchmarkEpisode, LocalPolicyRunner, run_benchmark
from genesis.evaluation.judge import run_episode
from genesis.evaluation.oracle import OraclePolicy
from genesis.evaluation.permutation import TABLES
from genesis.generation.company import SCHEMA_VERSION, generate_episode
from genesis.policies import make_policy
from genesis.policies.worker import main as worker_main, run as run_worker


_SEED = 17


def _case(fixture, *, clean=False, omit_quote=False, extra_rows=0):
    manifest = deepcopy(fixture.manifest)
    truth = deepcopy(fixture.truth)
    tables = {name: deepcopy(getattr(fixture, name)) for name in ("ledger", "invoices", "payments", "fx")}
    if clean:
        issue, = truth["issues"]
        for row in tables["ledger"]:
            if row["row_id"] in issue["affected_ids"]:
                key = "debit_minor" if row["debit_minor"] else "credit_minor"
                row[key] += issue["expected_amount_minor"]
        manifest["episode_id"] += "-clean"
        truth = {"episode_id": manifest["episode_id"], "issues": []}
    if omit_quote:
        tables["fx"].clear()
    for index in range(extra_rows):
        tables["ledger"].append({**tables["ledger"][0], "row_id": f"extra-ledger-{index}"})
    for table, rows in tables.items():
        manifest["record_counts"][table] = len(rows)
    return {
        "task": "Audit the synthetic ledger", "manifest": manifest,
        "policy": fixture.policy, "tables": tables, "truth": truth,
    }


def _make_env(fixture, mode="eval"):
    return AuditorEnv(
        lambda _, config: build_episode(**_case(fixture, **config)), mode=mode, clock=perf_counter,
    )


def _make_oracle(fixture, config):
    data = _case(fixture, **config)
    tools = PublicTools(
        episode_id=data["manifest"]["episode_id"], policy=data["policy"], tables=data["tables"],
    )
    return OraclePolicy(data["truth"], tools.public_evidence_ids)


async def _smoke(fixture):
    episodes = (
        BenchmarkEpisode("fx_discrepancy", 101, {}),
        BenchmarkEpisode("clean", 102, {"clean": True}),
    )
    runners = {
        name: (lambda _, label=name: LocalPolicyRunner(lambda: make_policy(label, seed=_SEED)))
        for name in ("random", "simple_reconciliation", "mark_all")
    }
    runners["oracle"] = lambda case: LocalPolicyRunner(lambda: _make_oracle(fixture, case.config))
    return await run_benchmark(lambda: _make_env(fixture), episodes, runners)


class PolicyAdaptersTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = contract.DataContractTests
        cls.fixture.setUpClass()

    async def run_public(self, name, *, policy_seed=_SEED, **config):
        runner = LocalPolicyRunner(lambda: make_policy(name, seed=policy_seed))
        result = await run_episode(_make_env(self.fixture), seed=986, config=config, runner=runner)
        self.assertIsNone(result.transport_error)
        for action in runner.actions:
            self.fixture.validators["action"].validate(action)
        return result.terminal, runner.actions

    async def test_random_actions_and_scores_repeat_for_the_same_policy_seed(self):
        first, actions = await self.run_public("random")
        second, replay = await self.run_public("random")
        _, another_sequence = await self.run_public("random", policy_seed=_SEED + 1)
        self.assertEqual(actions, replay)
        self.assertNotEqual(actions, another_sequence)
        self.assertEqual(first.score, second.score)
        self.assertEqual(first.matching, second.matching)
        self.assertEqual(first.total_steps, 9)
        self.assertEqual(actions[-1]["type"], "submit")
        self.assertTrue(any(action["type"] in {"get_record", "get_document"} for action in actions))

    async def test_fx_reconciliation_uses_public_pages_and_complete_proof(self):
        result, actions = await self.run_public("simple_reconciliation")
        self.assertEqual([action["table"] for action in actions[:-1]], ["payments", "invoices", "fx", "ledger"])
        self.assertEqual((result.matching.tp, result.matching.fp, result.matching.fn), (1, 0, 0))
        finding, = result.submission["findings"]
        self.assertEqual(finding["claimed_amount_minor"], 10000)
        expected = self.fixture.findings["findings"][0]["evidence_ids"]
        self.assertEqual(set(finding["evidence_ids"]), set(expected))
        self.assertEqual(result.score.reward, 0.992)

    async def test_simple_reconciliation_abstains_on_clean_or_missing_quote(self):
        for case in ({"clean": True}, {"omit_quote": True}):
            with self.subTest(case=case):
                result, _ = await self.run_public("simple_reconciliation", **case)
                self.assertFalse(result.invalid_submission)
                self.assertEqual(result.submission["findings"], [])
        paginated, actions = await self.run_public("simple_reconciliation", extra_rows=100)
        self.assertGreater(sum(action.get("table") == "ledger" for action in actions), 1)
        self.assertEqual(paginated.matching.tp, 1)

    async def test_E17_mark_all_submits_no_more_than_thirty_valid_false_claims(self):
        result, _ = await self.run_public("mark_all", clean=True, extra_rows=40)
        self.assertEqual(result.total_steps, 2)
        self.assertFalse(result.invalid_submission)
        self.assertEqual((result.matching.tp, result.matching.fp, result.matching.fn), (0, 30, 0))
        self.assertEqual((result.score.metrics.precision, result.score.metrics.recall, result.score.metrics.f1), (0, 0, 0))
        self.assertEqual(result.score.reward, -1)

    async def test_benchmark_runs_the_same_ordered_cases_for_every_policy(self):
        first = await _smoke(self.fixture)
        again = await _smoke(self.fixture)
        self.assertEqual(first.diagnostics, again.diagnostics)
        names = ("random", "simple_reconciliation", "mark_all", "oracle")
        self.assertEqual([row.policy for row in first.diagnostics], list(names) * 2)
        self.assertEqual(len(first.wall_time_seconds), len(first.diagnostics))
        self.assertTrue(all(duration >= 0 for duration in first.wall_time_seconds))
        for case in ("fx_discrepancy", "clean"):
            rows = [row for row in first.diagnostics if row.case == case]
            self.assertEqual(len({row.episode_id for row in rows}), 1)
            self.assertTrue(all(row.transport_error is None for row in rows))
        anomaly, clean = [row for row in first.diagnostics if row.policy == "oracle"]
        self.assertEqual((anomaly.tp, anomaly.fp, anomaly.fn, anomaly.reward), (1, 0, 0, 1))
        self.assertEqual((clean.tp, clean.fp, clean.fn, clean.reward), (0, 0, 0, 1))
        self.assertEqual((anomaly.total_steps, clean.total_steps), (1, 1))
        self.assertEqual((anomaly.inspection_steps, clean.inspection_steps), (0, 0))

    def test_oracle_chooses_complete_available_evidence(self):
        truth = deepcopy(self.fixture.truth)
        issue, = truth["issues"]
        existing, = issue["acceptable_evidence_sets"]
        issue["acceptable_evidence_sets"] = [["nonexistent"], existing]
        with self.assertRaisesRegex(ValueError, "complete public evidence"):
            OraclePolicy(truth, ["policy"])
        oracle = OraclePolicy(truth, existing)
        action = oracle.act({"manifest": self.fixture.manifest}, None)
        self.fixture.validators["action"].validate(action)
        [result] = action["findings"]["findings"]
        self.assertEqual(set(result["evidence_ids"]), set(existing))
        with self.assertRaisesRegex(ValueError, "unknown public policy"):
            make_policy("oracle")

    async def test_jsonl_worker_runs_with_only_public_modules_staged(self):
        source = Path(genesis.__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "genesis"
            package.mkdir()
            shutil.copyfile(source / "__init__.py", package / "__init__.py")
            for name in ("domain", "policies"):
                shutil.copytree(source / name, package / name, ignore=shutil.ignore_patterns("__pycache__"))
            environment = package / "environment"
            environment.mkdir()
            shutil.copyfile(source / "environment" / "__init__.py", environment / "__init__.py")
            shutil.copyfile(source / "environment" / "jsonl.py", environment / "jsonl.py")
            self.assertFalse((package / "evaluation").exists())
            process = None

            class PublicOnlyRunner:
                @asynccontextmanager
                async def open(self, deadline):
                    nonlocal process
                    process = await asyncio.create_subprocess_exec(
                        sys.executable, "-I", "-B", "-u", "-c",
                        "import sys; sys.path.insert(0, sys.argv.pop(1)); "
                        "from genesis.policies.worker import main; raise SystemExit(main())",
                        directory, "mark_all",
                        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE, env=_host_environment(), limit=FRAME_BYTES,
                    )
                    channel = JsonlProcess(process)
                    try:
                        yield channel
                    finally:
                        await channel.close()

            result = await run_episode(
                _make_env(self.fixture), seed=986, config={"clean": True}, runner=PublicOnlyRunner(),
            )
            self.assertIsNone(result.transport_error)
            self.assertEqual(result.terminal.matching.fp, len(self.fixture.ledger))
            self.assertEqual(process.returncode, 0)

    def test_worker_rejects_unterminated_or_oversized_frames(self):
        for raw in (b"{}", b" " * FRAME_BYTES + b"\n"):
            with self.subTest(size=len(raw)):
                outgoing = io.BytesIO()
                with self.assertRaisesRegex(ValueError, "invalid policy frame"):
                    run_worker(make_policy("mark_all"), io.BytesIO(raw), outgoing)
                self.assertEqual(outgoing.getvalue(), b"")

    async def test_simple_reconciliation_handles_invoice_date_fx_rule(self):
        episode = generate_episode(20260401, {
            "schema_version": SCHEMA_VERSION,
            "error_counts": {"fx_conversion": 1},
            "fx_date_rule": "invoice_date",
        })
        public = episode.public
        env = AuditorEnv(
            lambda _s, _c: build_episode(
                task="Audit invoice_date FX episode",
                manifest=public.manifest,
                policy=public.policy,
                tables={name: getattr(public, name) for name in TABLES},
                truth=episode.evaluator.truth,
            ),
            mode="eval",
        )
        runner = LocalPolicyRunner(lambda: make_policy("simple_reconciliation"))
        outcome = await run_episode(env, seed=20260401, config={}, runner=runner)
        self.assertIsNone(outcome.transport_error)
        self.assertEqual((outcome.terminal.matching.tp, outcome.terminal.matching.fp, outcome.terminal.matching.fn), (1, 0, 0))
        finding, = outcome.terminal.submission["findings"]
        self.assertEqual(finding["type"], "fx_conversion")
        invoice_doc_ids = {row["doc_id"] for row in public.invoices}
        self.assertTrue(set(finding["evidence_ids"]) & invoice_doc_ids)

    async def test_simple_reconciliation_solves_all_four_types_interactions_distractors_and_pagination(self):
        cases = (
            ("clean_distractors", 20260101, {"schema_version": SCHEMA_VERSION, "error_counts": {}, "include_distractors": True}, 0),
            ("duplicate_invoice", 20260102, {"schema_version": SCHEMA_VERSION, "error_counts": {"duplicate_invoice": 1}}, 1),
            ("payment_allocation", 20260103, {"schema_version": SCHEMA_VERSION, "error_counts": {"payment_allocation": 1}}, 1),
            ("period_cutoff", 20260105, {"schema_version": SCHEMA_VERSION, "error_counts": {"period_cutoff": 1}}, 1),
            ("partial_fx_cutoff", 20260106, {"schema_version": SCHEMA_VERSION, "error_counts": {}, "interaction_scenarios": ["partial_fx_payment_at_cutoff"]}, 2),
            ("duplicate_with_fx", 20260107, {"schema_version": SCHEMA_VERSION, "error_counts": {}, "interaction_scenarios": ["duplicate_invoice_with_fx"]}, 2),
            ("scaled_multi_page_all_four", 20260110, {
                "schema_version": SCHEMA_VERSION,
                "base_currency": "USD",
                "foreign_currency": "JPY",
                "timezone": "UTC",
                "obligation_count": 60,
                "include_distractors": True,
                "error_counts": {
                    "duplicate_invoice": 1,
                    "payment_allocation": 1,
                    "fx_conversion": 1,
                    "period_cutoff": 1,
                },
                "interaction_scenarios": ["partial_fx_payment_at_cutoff", "duplicate_invoice_with_fx"],
            }, 8),
        )
        for label, seed, config, expected_tp in cases:
            with self.subTest(case=label):
                episode = generate_episode(seed, config)
                public = episode.public
                env = AuditorEnv(
                    lambda _s, _c, pub=public, ep=episode: build_episode(
                        task="Audit public records",
                        manifest=pub.manifest,
                        policy=pub.policy,
                        tables={name: getattr(pub, name) for name in TABLES},
                        truth=ep.evaluator.truth,
                    ),
                    mode="eval",
                )
                runner = LocalPolicyRunner(lambda: make_policy("simple_reconciliation"))
                outcome = await run_episode(env, seed=seed, config=config, runner=runner)
                self.assertIsNone(outcome.transport_error)
                for action in runner.actions:
                    self.fixture.validators["action"].validate(action)
                m = outcome.terminal.matching
                self.assertEqual((m.tp, m.fp, m.fn), (expected_tp, 0, 0))
                self.assertEqual(outcome.terminal.score.metrics.f1, 1.0)
                if label == "scaled_multi_page_all_four":
                    self.assertGreater(sum(a.get("table") == "ledger" for a in runner.actions), 1)
                    self.assertGreater(outcome.terminal.score.reward, 0.95)

    def test_worker_cli_exit_codes_and_error_handling(self):
        reset_frame = (json.dumps({
            "type": "reset",
            "observation": {
                "task": "Audit",
                "manifest": self.fixture.manifest,
                "policy": self.fixture.policy,
                "remaining_steps": 100,
            },
            "info": {},
        }) + "\n").encode("utf-8")
        terminal_frame = (json.dumps({
            "type": "step",
            "observation": {"remaining_steps": 99, "status": "submitted"},
            "reward": 0.0,
            "terminated": True,
            "truncated": False,
            "info": {},
        }) + "\n").encode("utf-8")
        # Valid JSONL session exits 0 and emits a valid action frame.
        stdin_ok = io.TextIOWrapper(io.BytesIO(reset_frame + terminal_frame))
        stdout_buf = io.BytesIO()
        stdout_ok = io.TextIOWrapper(stdout_buf)
        saved_in, saved_out = sys.stdin, sys.stdout
        try:
            sys.stdin, sys.stdout = stdin_ok, stdout_ok
            self.assertEqual(worker_main(["random", "--seed", "7"]), 0)
        finally:
            sys.stdin, sys.stdout = saved_in, saved_out
        emitted = json.loads(stdout_buf.getvalue().decode("utf-8"))
        self.fixture.validators["action"].validate(emitted)

        # Malformed JSON frame exits 2 cleanly without traceback.
        stdin_bad = io.TextIOWrapper(io.BytesIO(b"{not-json}\n"))
        stdout_bad = io.TextIOWrapper(io.BytesIO())
        try:
            sys.stdin, sys.stdout = stdin_bad, stdout_bad
            self.assertEqual(worker_main(["simple_reconciliation"]), 2)
        finally:
            sys.stdin, sys.stdout = saved_in, saved_out


if __name__ == "__main__":
    if sys.argv[1:] == ["--smoke"]:
        fixture = contract.DataContractTests
        fixture.setUpClass()
        report = asyncio.run(_smoke(fixture))
        print("case policy TP FP FN P R F1 return steps inspection wall_seconds")
        for row, elapsed in zip(report.diagnostics, report.wall_time_seconds):
            print(f"{row.case} {row.policy} {row.tp} {row.fp} {row.fn} "
                  f"{row.precision:.3f} {row.recall:.3f} {row.f1:.3f} {row.reward:.3f} "
                  f"{row.total_steps} {row.inspection_steps} {elapsed:.6f}")
    else:
        unittest.main()
