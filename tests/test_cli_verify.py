"""Unit tests for genesis.cli verify gate, demo, run-policy dispatch, and packaging."""

from contextlib import contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import tomllib
import unittest
from unittest.mock import patch

import genesis.cli.__main__ as cli_main
import genesis.evaluation as evaluation_pkg
from genesis.evaluation.permutation import permute_episode, validate_links
from genesis.generation.company import SCHEMA_VERSION, generate_episode


_ROOT = Path(__file__).resolve().parents[1]
_SYNTHETIC_MODULES = (
    "test_env_contract_and_isolation",
    "test_public_datasets",
    "test_trajectories",
    "test_reproducibility_and_permutation",
    "test_matrix_synthetic",
    "test_missing_dep_synthetic",
    "test_import_err_synthetic",
    "test_broken_discovery",
)


@contextmanager
def _isolated_test_modules(names=_SYNTHETIC_MODULES):
    saved_modules = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    saved_path = list(sys.path)
    try:
        yield
    finally:
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path


def _write_synthetic_repo(root: Path, *, skip_docker: bool = True, extra_non_matrix: bool = True) -> None:
    tests_dir = root / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    for mod in (
        "test_env_contract_and_isolation",
        "test_public_datasets",
        "test_trajectories",
        "test_reproducibility_and_permutation",
    ):
        (tests_dir / f"{mod}.py").write_text(
            "import unittest\n"
            "class BasicTests(unittest.TestCase):\n"
            "    def test_ok(self):\n"
            "        self.assertTrue(True)\n",
            encoding="utf-8",
        )
    lines = [
        "import unittest",
        "class MatrixTests(unittest.TestCase):",
    ]
    for category in ("C", "E"):
        for number in range(1, 21):
            lines.append(f"    def test_{category}{number:02d}_case(self): self.assertTrue(True)")
    if extra_non_matrix:
        lines.append("    def test_CSV_parsing(self): self.assertTrue(True)")
        lines.append("    def test_EOF_handling(self): self.assertTrue(True)")
    lines.append("class DockerPolicyTests(unittest.TestCase):")
    if skip_docker:
        lines.append("    @unittest.skip('requires docker')")
    lines.append("    def test_smoke(self): self.assertTrue(True)")
    (tests_dir / "test_matrix_synthetic.py").write_text("\n".join(lines) + "\n", encoding="utf-8")


class CliVerifyTests(unittest.TestCase):
    def test_matrix_names_require_numeric_identifier_boundary(self):
        self.assertEqual(cli_main._MATRIX_TEST.match("test_C18_docker")[1], "C18")
        self.assertEqual(cli_main._MATRIX_TEST.match("test_C18")[1], "C18")
        self.assertIsNone(cli_main._MATRIX_TEST.match("test_CSV_export"))
        self.assertIsNone(cli_main._MATRIX_TEST.match("test_E20extra"))

    def test_failed_test_discovery_reports_import_error(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            tests = root / "tests"
            tests.mkdir()
            (tests / "test_broken_discovery.py").write_text(
                "import genesis_missing_test_dependency\n", encoding="utf-8",
            )
            with _isolated_test_modules(), redirect_stdout(StringIO()) as output:
                self.assertEqual(cli_main.verify(root), 1)
        self.assertIn("ModuleNotFoundError", output.getvalue())
        self.assertIn("genesis_missing_test_dependency", output.getvalue())

    def test_discovery_surfaces_module_not_found_without_host_paths(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_synthetic_repo(root)
            (root / "tests" / "test_missing_dep_synthetic.py").write_text(
                "import definitely_nonexistent_genesis_pkg_xyz\n",
                encoding="utf-8",
            )
            with _isolated_test_modules():
                with self.assertRaises(ModuleNotFoundError) as raised:
                    cli_main._test_groups(root)
                self.assertEqual(raised.exception.name, "definitely_nonexistent_genesis_pkg_xyz")
                self.assertIn("definitely_nonexistent_genesis_pkg_xyz", str(raised.exception))
                self.assertNotIn(str(root), str(raised.exception))

            out = StringIO()
            with _isolated_test_modules(), redirect_stdout(out):
                code = cli_main.verify(root)
            self.assertEqual(code, 1)
            output = out.getvalue()
            self.assertIn("FAIL verification setup: ModuleNotFoundError", output)
            self.assertIn("definitely_nonexistent_genesis_pkg_xyz", output)
            self.assertNotIn(str(root), output)

    def test_discovery_surfaces_generic_import_error(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_synthetic_repo(root)
            (root / "tests" / "test_import_err_synthetic.py").write_text(
                "raise ImportError('broken helper')\n",
                encoding="utf-8",
            )
            with _isolated_test_modules():
                with self.assertRaises(ImportError) as raised:
                    cli_main._test_groups(root)
                self.assertIn("test_import_err_synthetic", str(raised.exception))

    def test_non_matrix_csv_and_eof_names_do_not_break_matrix_grouping(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_synthetic_repo(root, skip_docker=True, extra_non_matrix=True)
            with _isolated_test_modules():
                groups = cli_main._test_groups(root)
            self.assertEqual(len(groups["contract-matrix"]), 20)
            self.assertEqual(len(groups["evaluation-matrix"]), 20)
            regression_methods = {case.id().rsplit(".", 1)[-1] for case in groups["runtime-regressions"]}
            self.assertIn("test_CSV_parsing", regression_methods)
            self.assertIn("test_EOF_handling", regression_methods)

    def test_run_group_respects_require_docker_flag(self):
        class DockerPolicyTests(unittest.TestCase):
            @unittest.skip("docker absent")
            def test_smoke(self):
                pass

        case = DockerPolicyTests("test_smoke")
        out_optional = StringIO()
        with redirect_stdout(out_optional):
            ok_optional = cli_main._run_group("docker-policy-smoke", [case], require_docker=False)
        self.assertTrue(ok_optional)
        self.assertIn("SKIP docker-policy-smoke:", out_optional.getvalue())

        out_required = StringIO()
        with redirect_stdout(out_required):
            ok_required = cli_main._run_group("docker-policy-smoke", [case], require_docker=True)
        self.assertFalse(ok_required)
        self.assertIn("FAIL docker-policy-smoke:", out_required.getvalue())

    def test_verify_require_docker_returns_nonzero_when_docker_skipped(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_synthetic_repo(root, skip_docker=True)
            with _isolated_test_modules(), patch.object(cli_main, "_verify_profiles", return_value=True):
                out_ok = StringIO()
                with redirect_stdout(out_ok):
                    self.assertEqual(cli_main.verify(root, require_docker=False), 0)
                self.assertIn("SKIP docker-policy-smoke:", out_ok.getvalue())

                out_fail = StringIO()
                with redirect_stdout(out_fail):
                    self.assertEqual(cli_main.verify(root, require_docker=True), 1)
                self.assertIn("FAIL docker-policy-smoke:", out_fail.getvalue())

    def test_verify_profiles_preserves_safe_diff_path_and_json_decode_error(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "benchmarks").mkdir(parents=True)
            (root / "benchmarks" / "local_eval_profiles.json").write_text("{", encoding="utf-8")
            out_json = StringIO()
            with redirect_stdout(out_json):
                ok_json = cli_main._verify_profiles(root)
            self.assertFalse(ok_json)
            self.assertRegex(out_json.getvalue(), r"^FAIL replay-and-permutation .*JSONDecodeError")

        diff_msg = "clean-01: replay: clean-01: public records differs at $.ledger[0].debit_minor"
        out_diff = StringIO()
        with (
            patch("genesis.evaluation.reproducibility.verify_reproducibility", side_effect=ValueError(diff_msg)),
            redirect_stdout(out_diff),
        ):
            ok_diff = cli_main._verify_profiles(_ROOT)
        self.assertFalse(ok_diff)
        self.assertIn(f"ValueError: {diff_msg}", out_diff.getvalue())

    def test_verify_profiles_reuses_captured_results_without_double_execution(self):
        fake_report = type("Report", (), {"replay_passed": 20, "pair_passed": 20, "total": 20})()
        diagnostics = [
            type("Row", (), {"policy": name, "transport_error": None})()
            for name in ("random", "simple_reconciliation", "mark_all", "oracle")
            for _ in range(20)
        ]
        fake_bench = type("Bench", (), {"diagnostics": diagnostics})()
        fake_summary = {
            "diagnostics": {
                name: {"precision": 0.5, "recall": 0.5, "f1": 0.5, "return": 0.0, "steps": 2.0}
                for name in ("random", "simple_reconciliation", "mark_all", "oracle")
            },
            "wall_time_seconds": {
                name: 0.001 for name in ("random", "simple_reconciliation", "mark_all", "oracle")
            },
        }
        cached = {"reproducibility": (fake_report, 0.05), "benchmark": (fake_bench, 0.04)}
        out = StringIO()
        with (
            patch(
                "genesis.evaluation.reproducibility.verify_reproducibility",
                side_effect=AssertionError("should not rerun verify_reproducibility"),
            ),
            patch(
                "genesis.evaluation.reproducibility.local_benchmark",
                side_effect=AssertionError("should not rerun local_benchmark"),
            ),
            patch("genesis.evaluation.reproducibility.benchmark_summary", return_value=fake_summary),
            redirect_stdout(out),
        ):
            ok = cli_main._verify_profiles(_ROOT, cached=cached)
        self.assertTrue(ok)
        self.assertIn("PASS replay-and-permutation: replay 20/20, pairs 20/20", out.getvalue())
        self.assertIn("PASS policy-diagnostics: 80/80 local runs", out.getvalue())

    def test_main_accepts_verify_with_and_without_all_flag(self):
        with patch.object(cli_main, "verify", return_value=0) as mocked:
            self.assertEqual(cli_main.main(["verify", "--all"]), 0)
            self.assertEqual(cli_main.main(["verify"]), 0)
            self.assertEqual(cli_main.main(["verify", "--all", "--require-docker"]), 0)
        self.assertEqual(mocked.call_count, 3)
        self.assertFalse(mocked.call_args_list[0].kwargs["require_docker"])
        self.assertFalse(mocked.call_args_list[1].kwargs["require_docker"])
        self.assertTrue(mocked.call_args_list[2].kwargs["require_docker"])

    def test_main_dispatches_run_policy_subcommand(self):
        with patch("genesis.cli.run_policy.run_policy", return_value=0) as mocked:
            code = cli_main.main([
                "run-policy", "standard-03", "--image", "genesis-policy:ci",
                "--cpus", "2", "--memory-mib", "1024", "--tmpfs-mib", "64",
            ])
        self.assertEqual(code, 0)
        mocked.assert_called_once()
        root_arg, parsed_args = mocked.call_args.args
        self.assertEqual(root_arg, cli_main._REPOSITORY)
        self.assertEqual(parsed_args.profile, "standard-03")
        self.assertEqual(parsed_args.image, "genesis-policy:ci")
        self.assertEqual((parsed_args.cpus, parsed_args.memory_mib, parsed_args.tmpfs_mib), (2, 1024, 64))

    def test_demo_subcommand_runs_all_four_policies_end_to_end(self):
        for policy in ("simple_reconciliation", "random", "mark_all", "oracle"):
            with self.subTest(policy=policy):
                out = StringIO()
                with redirect_stdout(out):
                    code = cli_main.main(["demo", "--policy", policy, "--seed", "20260304"])
                self.assertEqual(code, 0)
                text = out.getvalue()
                self.assertIn(f"policy={policy}", text)
                self.assertIn("Trace:", text)
                self.assertIn("Diagnostics:", text)
                if policy in ("simple_reconciliation", "oracle"):
                    self.assertIn("F1=1.0000", text)

    def test_packaging_metadata_and_regular_evaluation_package(self):
        pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(pyproject["project"]["name"], "genesis-fintech-env")
        self.assertEqual(pyproject["project"]["requires-python"], ">=3.11")
        self.assertIn("tzdata>=2025.2", pyproject["project"]["dependencies"])
        self.assertEqual(pyproject["project"]["scripts"]["genesis"], "genesis.cli.__main__:main")
        self.assertIsNotNone(getattr(evaluation_pkg, "__file__", None))
        self.assertTrue(Path(evaluation_pkg.__file__).name == "__init__.py")

    def test_permutation_validate_links_accepts_valid_and_rejects_broken_references(self):
        episode = generate_episode(20260304, {"schema_version": SCHEMA_VERSION, "error_counts": {"fx_conversion": 1}})
        opaque = validate_links(episode)
        self.assertIn(episode.public.manifest["episode_id"], opaque)
        moved, _ = permute_episode(episode, salt=74137)
        self.assertEqual(len(validate_links(moved)), len(opaque))
        moved.public.ledger[0]["invoice_id"] = "inv-missing-999"
        with self.assertRaisesRegex(ValueError, "unresolved ledger reference"):
            validate_links(moved)


if __name__ == "__main__":
    unittest.main()
