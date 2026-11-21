from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase, skip
from unittest.mock import patch

from genesis.cli.__main__ import _MATRIX_TEST, _run_group, verify


class VerificationTests(TestCase):
    def test_matrix_names_require_numeric_identifier_boundary(self):
        self.assertEqual(_MATRIX_TEST.match("test_C18_docker")[1], "C18")
        self.assertIsNone(_MATRIX_TEST.match("test_CSV_export"))
        self.assertIsNone(_MATRIX_TEST.match("test_E20extra"))

    def test_failed_test_discovery_reports_import_error(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            tests = root / "tests"
            tests.mkdir()
            (tests / "test_broken_discovery.py").write_text(
                "import genesis_missing_test_dependency\n", encoding="utf-8",
            )
            with patch.object(sys, "path", sys.path.copy()), redirect_stdout(StringIO()) as output:
                self.assertEqual(verify(root), 1)
        self.assertIn("ImportError", output.getvalue())

    def test_require_docker_turns_skip_into_failure(self):
        class SkippedCase(TestCase):
            @skip("fixture")
            def test_docker(self):
                pass

        cases = [SkippedCase("test_docker")]
        with redirect_stdout(StringIO()):
            self.assertTrue(_run_group("docker-policy-smoke", cases, False))
            self.assertFalse(_run_group("docker-policy-smoke", cases, True))
