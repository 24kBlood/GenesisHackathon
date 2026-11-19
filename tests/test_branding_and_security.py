"""Tests for neutral AuditLab branding, GUI HTTP endpoints & security, and custom Agent Skills."""

from __future__ import annotations

import http.client
import json
import re
import threading
import unittest
from pathlib import Path
from typing import Any

from audit_rl.gui import create_server

REPO_ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN_BRANDING_RE = re.compile(r"(hackathon|genesis|жюри|jury)", re.IGNORECASE)

USER_FACING_FILES = (
    "audit_rl/gui.py",
    "audit_rl/rubric.py",
    "audit_rl/cli.py",
    "README.md",
    "DESIGN.md",
    "DEMO.md",
    "LICENSE",
    "pyproject.toml",
    "PROJECT_STATUS.md",
    "ARCHITECTURE.md",
    "IMPLEMENTATION_PLAN.md",
    "DECISIONS.md",
    "CONTRIBUTING.md",
)

CUSTOM_SKILLS = (
    "fintech-sandbox-engineering",
    "synthetic-accounting-data",
    "rl-environment-design",
    "reward-engineering",
    "reproducible-benchmarking",
    "fintech-security",
    "python-testing",
    "github-pr-workflow",
)


class TestBrandingAndSkills(unittest.TestCase):
    def test_zero_hackathon_or_jury_branding_in_user_facing_files(self) -> None:
        for rel_path in USER_FACING_FILES:
            full_path = REPO_ROOT / rel_path
            with self.subTest(file=rel_path):
                self.assertTrue(full_path.is_file(), f"Expected file missing: {rel_path}")
                text = full_path.read_text(encoding="utf-8")
                matches = FORBIDDEN_BRANDING_RE.findall(text)
                self.assertEqual(
                    matches,
                    [],
                    f"Forbidden competition/hackathon branding found in {rel_path}: {matches[:5]}",
                )

    def test_all_eight_custom_skills_exist_with_valid_yaml_frontmatter(self) -> None:
        skills_dir = REPO_ROOT / ".agents" / "skills"
        self.assertTrue(skills_dir.is_dir(), ".agents/skills directory missing")

        for skill_name in CUSTOM_SKILLS:
            skill_md = skills_dir / skill_name / "SKILL.md"
            with self.subTest(skill=skill_name):
                self.assertTrue(skill_md.is_file(), f"Missing {skill_md}")
                text = skill_md.read_text(encoding="utf-8")
                self.assertTrue(
                    text.startswith("---\n"),
                    f"{skill_md} must start with YAML frontmatter '---'",
                )
                parts = text.split("---\n", 2)
                self.assertGreaterEqual(len(parts), 3, f"{skill_md} missing closing frontmatter '---'")
                frontmatter = parts[1]
                body = parts[2].strip()
                self.assertRegex(frontmatter, r"(?m)^name:\s*\S+")
                self.assertRegex(frontmatter, r"(?m)^description:\s*\S+")
                self.assertGreater(len(body), 200, f"{skill_md} body too short / empty")
                self.assertEqual(
                    FORBIDDEN_BRANDING_RE.findall(text),
                    [],
                    f"Forbidden branding found in {skill_md}",
                )


class TestGuiEndpointsAndSecurity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = create_server(host="127.0.0.1", port=0, root=REPO_ROOT, quiet=True)
        cls.host, cls.port = cls.server.server_address[:2]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5.0)

    def _request(
        self,
        method: str,
        path: str,
        body: bytes | dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=30)
        req_headers = dict(headers or {})
        raw_body: bytes | None = None
        if isinstance(body, dict):
            raw_body = json.dumps(body).encode("utf-8")
            req_headers.setdefault("Content-Type", "application/json")
        elif isinstance(body, bytes):
            raw_body = body
        try:
            conn.request(method, path, body=raw_body, headers=req_headers)
            resp = conn.getresponse()
            data = resp.read()
            resp_headers = {k.lower(): v for k, v in resp.getheaders()}
            return resp.status, resp_headers, data
        finally:
            conn.close()

    def test_get_root_html_is_200_offline_and_zero_external_cdns(self) -> None:
        status, headers, data = self._request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers.get("content-type", ""))
        html = data.decode("utf-8")
        self.assertIn("AuditLab", html)
        # Verify zero external CDN or http(s) resource links in HTML
        external_urls = re.findall(r"https?://[^\s\"'<>]+", html)
        self.assertEqual(external_urls, [], f"Found external URLs in offline UI: {external_urls[:5]}")

    def test_get_json_endpoints_return_200_and_valid_json(self) -> None:
        for path in ("/api/overview", "/api/configs", "/api/trajectories", "/api/settings"):
            with self.subTest(path=path):
                status, headers, data = self._request("GET", path)
                self.assertEqual(status, 200, f"{path} returned {status}: {data[:200]!r}")
                self.assertIn("application/json", headers.get("content-type", ""))
                parsed = json.loads(data.decode("utf-8"))
                self.assertIsInstance(parsed, dict)

    def test_workspace_reset_step_and_record_lookup(self) -> None:
        # Reset workspace session on train episode 0
        status, _, data = self._request(
            "POST",
            "/api/workspace/reset",
            {"session_id": "test-sess", "split": "train", "index": 0},
        )
        self.assertEqual(status, 200)
        reset_resp = json.loads(data.decode("utf-8"))
        self.assertEqual(reset_resp["session_id"], "test-sess")
        self.assertEqual(reset_resp["step"], 0)
        self.assertIn("manifest", reset_resp)

        # Step: list invoices
        status, _, data = self._request(
            "POST",
            "/api/workspace/step",
            {
                "session_id": "test-sess",
                "action": {"action": "list_records", "table": "invoices", "cursor": "0", "limit": 5},
            },
        )
        self.assertEqual(status, 200)
        step_resp = json.loads(data.decode("utf-8"))
        self.assertEqual(step_resp["step"], 1)
        self.assertEqual(step_resp["observation"]["status"], "ok")
        first_inv_id = step_resp["observation"]["records"][0]["invoice_id"]

        # Record lookup on that invoice
        status, _, data = self._request(
            "POST",
            "/api/record/lookup",
            {"split": "train", "index": 0, "id": first_inv_id},
        )
        self.assertEqual(status, 200)
        lookup_resp = json.loads(data.decode("utf-8"))
        self.assertTrue(lookup_resp["ok"])
        self.assertEqual(lookup_resp["id"], first_inv_id)

    def test_calculator_endpoint_valid_and_injection_blocked(self) -> None:
        status, _, data = self._request("POST", "/api/calculator", {"expression": "12500 * 1.085"})
        self.assertEqual(status, 200)
        parsed = json.loads(data.decode("utf-8"))
        self.assertTrue(parsed["ok"])
        self.assertEqual(parsed["result"], "13562.5")

        # Unsafe injection expression blocked cleanly
        status, _, data = self._request(
            "POST", "/api/calculator", {"expression": "__import__('os').system('id')"}
        )
        self.assertEqual(status, 200)
        parsed = json.loads(data.decode("utf-8"))
        self.assertFalse(parsed["ok"])
        self.assertIn("error", parsed)

    def test_train_endpoint_returns_200_and_improvement_metrics(self) -> None:
        status, _, data = self._request(
            "POST",
            "/api/train",
            {"seed": 42, "epochs": 1, "save_to_store": False},
        )
        self.assertEqual(status, 200, f"/api/train failed: {data[:300]!r}")
        parsed = json.loads(data.decode("utf-8"))
        self.assertIn("trained_eval", parsed)
        self.assertIn("untrained_eval", parsed)
        self.assertGreater(parsed["trained_eval"]["f1"], parsed["untrained_eval"]["f1"])

    def test_security_path_traversal_malformed_json_and_oversized_body(self) -> None:
        # Path traversal rejected with HTTP 400
        for bad_path in ("/../etc/passwd", "/api/%2e%2e/truth.json", "/api\\secret"):
            with self.subTest(bad_path=bad_path):
                status, _, _ = self._request("GET", bad_path)
                self.assertEqual(status, 400)

        # Malformed JSON rejected with HTTP 400
        status, _, data = self._request(
            "POST",
            "/api/calculator",
            body=b"{not valid json",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(data.decode("utf-8")))

        # Oversized calculator expression (>256 chars) rejected cleanly
        status, _, data = self._request(
            "POST",
            "/api/calculator",
            {"expression": "1+" * 200 + "1"},
        )
        self.assertEqual(status, 200)
        parsed = json.loads(data.decode("utf-8"))
        self.assertFalse(parsed["ok"])
        self.assertIn("error", parsed)

        # Oversized HTTP body (>2 MB Content-Length) rejected immediately with HTTP 400
        conn = http.client.HTTPConnection(self.host, self.port, timeout=10)
        try:
            conn.putrequest("POST", "/api/calculator")
            conn.putheader("Content-Type", "application/json")
            conn.putheader("Content-Length", str(3 * 1024 * 1024))
            conn.endheaders()
            resp = conn.getresponse()
            data = resp.read()
            self.assertEqual(resp.status, 400)
            self.assertIn("error", json.loads(data.decode("utf-8")))
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
