"""Zero-dependency offline local Web GUI for the FinTech Audit RL Environment.

Serves a single-page dark FinTech terminal dashboard over Python stdlib
``http.server.ThreadingHTTPServer`` with JSON API endpoints backed by
``SubprocessAuditEnv``, ``AuditEnv``, and ``safe_calculate``.
"""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from audit_rl.calculator import CalculatorError, safe_calculate
from audit_rl.env import AuditEnv, SubprocessAuditEnv
from audit_rl.generator import generate_episode
from audit_rl.policies import (
    FlagEverythingAttackPolicy,
    OraclePolicy,
    RandomExplorerPolicy,
    SimpleReconciliationPolicy,
)
from audit_rl.schemas import EpisodeConfig, MAX_PAGE_LIMIT

_VALID_SPLITS = ("train", "dev", "eval")
_VALID_POLICIES = (
    "simple_reconciliation",
    "random_explorer",
    "flag_everything_attack",
    "oracle",
)
_VALID_TABLES = ("invoices", "payments", "ledger", "fx")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_all_configs(root: Path) -> dict[str, Any]:
    configs_dir = root / "configs"
    result: dict[str, Any] = {}
    for split in _VALID_SPLITS:
        spec = _load_json(configs_dir / f"{split}_configs.json")
        episodes = []
        for idx, ep in enumerate(spec.get("episodes", [])):
            cfg = ep.get("config", {})
            episodes.append(
                {
                    "index": idx,
                    "seed": int(ep["seed"]),
                    "description": str(ep.get("description", "")),
                    "base_currency": cfg.get("base_currency", "USD"),
                    "num_invoices": cfg.get("num_invoices", 0),
                    "clean": bool(cfg.get("clean", False)),
                    "error_counts": cfg.get("error_counts", {}),
                    "interacting_scenarios": cfg.get("interacting_scenarios", []),
                    "config": cfg,
                }
            )
        result[split] = episodes
    return result


def _resolve_episode_spec(root: Path, payload: dict[str, Any]) -> tuple[int, dict[str, Any], str]:
    """Resolve (seed, config_dict, description) from either {split, index} or {seed, config}."""
    if "seed" in payload and "config" in payload and payload["config"] is not None:
        seed = int(payload["seed"])
        cfg = payload["config"]
        if not isinstance(cfg, dict):
            raise ValueError("'config' must be a JSON object")
        desc = str(payload.get("description", f"Custom episode (seed={seed})"))
        return seed, cfg, desc

    split = str(payload.get("split", "dev"))
    if split not in _VALID_SPLITS:
        raise ValueError(f"Invalid split {split!r}; expected one of {_VALID_SPLITS}")

    raw_index = payload.get("index", 0)
    if isinstance(raw_index, bool) or not isinstance(raw_index, int):
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid episode 'index'; must be an integer") from exc
    else:
        index = raw_index

    spec = _load_json(root / "configs" / f"{split}_configs.json")
    episodes = spec.get("episodes", [])
    if index < 0 or index >= len(episodes):
        raise ValueError(f"Episode index {index} out of range for split {split!r} (0..{len(episodes) - 1})")

    ep_spec = episodes[index]
    return int(ep_spec["seed"]), dict(ep_spec["config"]), str(ep_spec.get("description", ""))


def _summarize_observation(action: dict[str, Any], obs: dict[str, Any]) -> str:
    act_type = action.get("action")
    if obs.get("status") == "error":
        return f"Error: {obs.get('error', 'unknown error')}"
    if act_type == "list_records":
        table = obs.get("table", action.get("table", "?"))
        count = obs.get("count", len(obs.get("records", [])))
        total = obs.get("total_count", "?")
        cursor = obs.get("cursor", "0")
        nxt = obs.get("next_cursor")
        nxt_label = f"next={nxt}" if nxt is not None else "end"
        return f"Returned {count}/{total} rows from {table} (cursor={cursor}, {nxt_label})"
    if act_type == "get_record":
        return f"Fetched {obs.get('table', action.get('table', '?'))} record {obs.get('id', action.get('id', '?'))}"
    if act_type == "get_document":
        doc = obs.get("document", {})
        doc_type = doc.get("doc_type", "document") if isinstance(doc, dict) else "document"
        return f"Fetched {doc_type} {obs.get('doc_id', action.get('doc_id', '?'))}"
    if act_type == "calculator":
        return f"{action.get('expression', '')} = {obs.get('result', '?')}"
    if act_type == "submit":
        findings = action.get("findings", {})
        n_findings = len(findings.get("findings", [])) if isinstance(findings, dict) else 0
        return f"Submitted {n_findings} finding(s) for {obs.get('episode_id', '')}"
    return f"Status: {obs.get('status', 'ok')}"


def _make_policy(policy_name: str, seed: int, cfg_dict: dict[str, Any]) -> Any:
    if policy_name == "simple_reconciliation":
        return SimpleReconciliationPolicy(use_calculator=True)
    if policy_name == "random_explorer":
        return RandomExplorerPolicy(seed=0)
    if policy_name == "flag_everything_attack":
        return FlagEverythingAttackPolicy()
    if policy_name == "oracle":
        # Generate evaluator-side truth out-of-band only for OraclePolicy
        ep = generate_episode(seed=seed, config=EpisodeConfig.from_dict(cfg_dict))
        return OraclePolicy(truth=ep.truth)
    raise ValueError(f"Unknown policy {policy_name!r}; expected one of {_VALID_POLICIES}")


def _run_episode_api(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    seed, cfg_dict, description = _resolve_episode_spec(root, payload)
    policy_name = str(payload.get("policy", "simple_reconciliation"))
    policy = _make_policy(policy_name, seed, cfg_dict)

    t0 = time.perf_counter()
    trajectory: list[dict[str, Any]] = []
    submitted_findings: list[dict[str, Any]] = []

    with SubprocessAuditEnv(mode="eval") as env:
        obs, info = env.reset(seed=seed, config=cfg_dict)
        manifest = dict(obs.get("manifest", {}))
        policy_doc = dict(obs.get("policy", {}))
        policy.reset(obs, info)

        step_num = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = policy.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            step_num += 1
            if action.get("action") == "submit":
                raw_findings = action.get("findings")
                if isinstance(raw_findings, dict) and isinstance(raw_findings.get("findings"), list):
                    submitted_findings = list(raw_findings["findings"])
            trajectory.append(
                {
                    "step": step_num,
                    "action": action,
                    "action_type": str(action.get("action", "unknown")),
                    "observation_summary": _summarize_observation(action, next_obs),
                    "observation": next_obs,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            obs = next_obs
            if step_num > 500:
                break

        report = env.get_evaluator_report()

    wall_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    metrics = dict(report.get("metrics", {}))
    evaluator_report = {
        "episode_id": report.get("episode_id", manifest.get("episode_id", "")),
        "tp": int(metrics.get("tp", report.get("tp", 0))),
        "fp": int(metrics.get("fp", report.get("fp", 0))),
        "fn": int(metrics.get("fn", report.get("fn", 0))),
        "precision": round(float(metrics.get("precision", report.get("precision", 0.0))), 4),
        "recall": round(float(metrics.get("recall", report.get("recall", 0.0))), 4),
        "f1": round(float(metrics.get("f1", report.get("f1", 0.0))), 4),
        "reward": round(float(report.get("reward", 0.0)), 4),
        "steps": int(report.get("steps", step_num)),
        "inspection_steps": int(report.get("inspection_steps", max(0, step_num - 1))),
        "runtime_ms": round(float(report.get("runtime_ms", wall_ms)), 2),
        "wall_runtime_ms": wall_ms,
        "valid_submission": bool(report.get("valid_submission", True)),
        "validation_errors": list(report.get("validation_errors", [])),
        "submitted_count": int(report.get("submitted_count", len(submitted_findings))),
        "truth_count": int(report.get("truth_count", 0)),
        "matched_pairs": list(metrics.get("matched_pairs", [])),
        "unmatched_finding_indices": list(metrics.get("unmatched_finding_indices", [])),
        "missed_issue_ids": list(metrics.get("missed_issue_ids", [])),
    }

    return {
        "seed": seed,
        "description": description,
        "policy_name": policy_name,
        "manifest": manifest,
        "policy": policy_doc,
        "trajectory": trajectory,
        "submitted_findings": submitted_findings,
        "report": evaluator_report,
    }


def _inspect_episode_api(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    seed, cfg_dict, description = _resolve_episode_spec(root, payload)
    table = str(payload.get("table", "invoices"))
    if table not in _VALID_TABLES:
        raise ValueError(f"Unknown table {table!r}; expected one of {_VALID_TABLES}")

    raw_limit = payload.get("limit", 25)
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid 'limit'; must be an integer") from exc
    if limit < 1 or limit > MAX_PAGE_LIMIT:
        raise ValueError(f"Limit must be between 1 and {MAX_PAGE_LIMIT}, got {limit}")

    cursor = payload.get("cursor", "0")
    if cursor is None or cursor == "":
        cursor = "0"
    cursor_str = str(cursor)

    env = AuditEnv(mode="eval")
    reset_obs, _ = env.reset(seed=seed, config=cfg_dict)
    page_obs, _, _, _, _ = env.step(
        {
            "action": "list_records",
            "table": table,
            "cursor": cursor_str,
            "limit": limit,
        }
    )
    if page_obs.get("status") == "error":
        raise ValueError(str(page_obs.get("error", "Failed to list records")))

    return {
        "seed": seed,
        "description": description,
        "episode_id": reset_obs["manifest"].get("episode_id", ""),
        "manifest": reset_obs["manifest"],
        "policy": reset_obs["policy"],
        "table": page_obs.get("table", table),
        "records": page_obs.get("records", []),
        "count": int(page_obs.get("count", 0)),
        "total_count": int(page_obs.get("total_count", 0)),
        "cursor": str(page_obs.get("cursor", cursor_str)),
        "next_cursor": page_obs.get("next_cursor"),
        "limit": limit,
    }


def _calculator_api(payload: dict[str, Any]) -> dict[str, Any]:
    expression = payload.get("expression", "")
    try:
        result = safe_calculate(expression)
        return {"ok": True, "expression": expression, "result": result}
    except CalculatorError as exc:
        return {"ok": False, "expression": expression, "error": str(exc)}


def _baselines_api(root: Path) -> dict[str, Any]:
    # Lazy import to avoid circular import between cli.py and gui.py
    from audit_rl.cli import evaluate_baselines

    eval_cfg = root / "configs" / "eval_configs.json"
    return evaluate_baselines(eval_cfg)


class AuditGUIServer(ThreadingHTTPServer):
    """Threaded HTTP server carrying the repository root path."""

    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_cls: type[BaseHTTPRequestHandler],
        root: Path,
        quiet: bool = False,
    ) -> None:
        self.root = root
        self.quiet = quiet
        super().__init__(server_address, handler_cls)


class AuditGUIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the Audit RL local dashboard and JSON API."""

    server: AuditGUIServer  # type: ignore[assignment]

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        if not getattr(self.server, "quiet", False):
            super().log_message(format, *args)

    def _send_bytes(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, data: Any) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._send_bytes(status, "application/json; charset=utf-8", raw)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        if length > 2 * 1024 * 1024:
            raise ValueError("Request body too large")
        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid JSON body: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("JSON request body must be an object")
        return parsed

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                self._send_bytes(
                    HTTPStatus.OK,
                    "text/html; charset=utf-8",
                    _DASHBOARD_HTML.encode("utf-8"),
                )
                return
            if path == "/api/health":
                self._send_json(HTTPStatus.OK, {"status": "ok"})
                return
            if path == "/api/configs":
                configs = _load_all_configs(self.server.root)
                self._send_json(HTTPStatus.OK, configs)
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint: {path}"})
        except Exception as exc:  # pragma: no cover - defensive
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        try:
            if path == "/api/episode/run":
                result = _run_episode_api(self.server.root, payload)
                self._send_json(HTTPStatus.OK, result)
                return
            if path == "/api/episode/inspect":
                result = _inspect_episode_api(self.server.root, payload)
                self._send_json(HTTPStatus.OK, result)
                return
            if path == "/api/calculator":
                result = _calculator_api(payload)
                self._send_json(HTTPStatus.OK, result)
                return
            if path == "/api/baselines":
                result = _baselines_api(self.server.root)
                self._send_json(HTTPStatus.OK, result)
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint: {path}"})
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - defensive
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})


def create_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    root: Path | None = None,
    quiet: bool = False,
) -> AuditGUIServer:
    """Create and bind an ``AuditGUIServer`` instance."""
    resolved_root = Path(root).resolve() if root is not None else _repo_root()
    return AuditGUIServer((host, port), AuditGUIHandler, root=resolved_root, quiet=quiet)


def run_gui(
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
    root: Path | None = None,
) -> None:
    """Start the local offline Audit RL Web GUI server and block until interrupted."""
    server = create_server(host=host, port=port, root=root, quiet=False)
    actual_host, actual_port = server.server_address[:2]
    url = f"http://{actual_host}:{actual_port}"
    print(f"Audit RL Web GUI running at {url} (100% offline, Press Ctrl+C to stop)")

    if open_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Audit RL Web GUI...")
    finally:
        server.server_close()


_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Audit RL — FinTech Audit Environment Dashboard</title>
<style>
  :root {
    --bg: #080c14;
    --bg-elev: #0f1623;
    --bg-card: #131c2c;
    --bg-hover: #19253a;
    --border: rgba(148, 163, 184, 0.14);
    --border-strong: rgba(148, 163, 184, 0.26);
    --text: #e2e8f0;
    --text-muted: #94a3b8;
    --text-dim: #64748b;
    --accent: #38bdf8;
    --accent-soft: rgba(56, 189, 248, 0.14);
    --emerald: #10b981;
    --emerald-soft: rgba(16, 185, 129, 0.14);
    --amber: #f59e0b;
    --amber-soft: rgba(245, 158, 11, 0.14);
    --rose: #f43f5e;
    --rose-soft: rgba(244, 63, 94, 0.14);
    --violet: #a78bfa;
    --violet-soft: rgba(167, 139, 250, 0.14);
    --font-sans: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    --radius: 10px;
  }

  * { box-sizing: border-box; }
  html, body {
    margin: 0;
    padding: 0;
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-sans);
    font-size: 14px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }

  .app-shell {
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px 24px 48px;
  }

  /* Header */
  .topbar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding-bottom: 18px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .brand-mark {
    width: 36px;
    height: 36px;
    border-radius: 9px;
    background: linear-gradient(135deg, #0ea5e9, #10b981);
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--font-mono);
    font-weight: 800;
    font-size: 15px;
    color: #061018;
    letter-spacing: -0.5px;
    box-shadow: 0 0 24px rgba(14, 165, 233, 0.25);
  }
  .brand-title {
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #f8fafc;
  }
  .brand-sub {
    font-size: 12px;
    color: var(--text-muted);
    font-family: var(--font-mono);
  }
  .status-pills {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }
  .pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-family: var(--font-mono);
    font-weight: 600;
    border: 1px solid var(--border);
    background: var(--bg-elev);
    color: var(--text-muted);
  }
  .pill-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--emerald);
    box-shadow: 0 0 8px var(--emerald);
  }
  .pill-emerald { color: var(--emerald); background: var(--emerald-soft); border-color: rgba(16, 185, 129, 0.3); }
  .pill-sky { color: var(--accent); background: var(--accent-soft); border-color: rgba(56, 189, 248, 0.3); }
  .pill-amber { color: var(--amber); background: var(--amber-soft); border-color: rgba(245, 158, 11, 0.3); }
  .pill-rose { color: var(--rose); background: var(--rose-soft); border-color: rgba(244, 63, 94, 0.3); }
  .pill-violet { color: var(--violet); background: var(--violet-soft); border-color: rgba(167, 139, 250, 0.3); }

  /* Tabs */
  .tabs {
    display: flex;
    gap: 6px;
    margin-bottom: 20px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 10px;
    flex-wrap: wrap;
  }
  .tab-btn {
    background: transparent;
    border: 1px solid transparent;
    color: var(--text-muted);
    padding: 8px 16px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
    font-family: var(--font-sans);
  }
  .tab-btn:hover {
    color: var(--text);
    background: var(--bg-elev);
  }
  .tab-btn.active {
    color: #f8fafc;
    background: var(--bg-card);
    border-color: var(--border-strong);
    box-shadow: inset 0 -2px 0 var(--accent);
  }

  .tab-panel { display: none; }
  .tab-panel.active { display: block; }

  /* Cards & Controls */
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 18px;
    margin-bottom: 16px;
  }
  .card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 12px;
    flex-wrap: wrap;
  }
  .card-title {
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    margin: 0;
  }

  .controls-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: flex-end;
  }
  .field {
    display: flex;
    flex-direction: column;
    gap: 5px;
    min-width: 140px;
  }
  .field.grow { flex: 1 1 280px; }
  label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
  }
  select, input[type="text"], input[type="number"] {
    background: var(--bg-elev);
    border: 1px solid var(--border-strong);
    color: var(--text);
    border-radius: 7px;
    padding: 8px 10px;
    font-size: 13px;
    font-family: var(--font-sans);
    outline: none;
    transition: border-color 0.15s ease;
  }
  select:focus, input:focus { border-color: var(--accent); }
  .mono-input { font-family: var(--font-mono); }

  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 8px 16px;
    border-radius: 7px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    border: 1px solid var(--border-strong);
    background: var(--bg-elev);
    color: var(--text);
    transition: all 0.15s ease;
    font-family: var(--font-sans);
    white-space: nowrap;
  }
  .btn:hover:not(:disabled) { background: var(--bg-hover); border-color: var(--accent); }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-primary {
    background: linear-gradient(135deg, #0284c7, #0ea5e9);
    border-color: rgba(56, 189, 248, 0.5);
    color: #031018;
    font-weight: 700;
  }
  .btn-primary:hover:not(:disabled) {
    background: linear-gradient(135deg, #0ea5e9, #38bdf8);
    border-color: #7dd3fc;
  }
  .btn-sm { padding: 5px 10px; font-size: 12px; }

  /* KPI Grid */
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(165px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
  }
  .kpi {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px 16px;
  }
  .kpi-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    margin-bottom: 4px;
  }
  .kpi-val {
    font-family: var(--font-mono);
    font-size: 24px;
    font-weight: 700;
    color: #f8fafc;
    line-height: 1.2;
  }
  .kpi-sub {
    font-size: 11px;
    font-family: var(--font-mono);
    color: var(--text-dim);
    margin-top: 4px;
  }
  .val-good { color: var(--emerald); }
  .val-warn { color: var(--amber); }
  .val-bad { color: var(--rose); }
  .val-sky { color: var(--accent); }

  /* Metadata strip */
  .meta-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    padding: 10px 14px;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 16px;
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--text-muted);
  }
  .meta-strip strong { color: var(--text); font-weight: 600; }

  /* Tables */
  .table-wrap {
    overflow-x: auto;
    max-width: 100%;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--bg-elev);
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12.5px;
  }
  th, td {
    text-align: left;
    padding: 9px 12px;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
  }
  th {
    background: rgba(15, 23, 42, 0.9);
    color: var(--text-muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 700;
    position: sticky;
    top: 0;
    white-space: nowrap;
  }
  tr:last-child td { border-bottom: none; }
  tbody tr:hover td { background: rgba(56, 189, 248, 0.04); }
  .mono { font-family: var(--font-mono); }
  .num { font-family: var(--font-mono); text-align: right; white-space: nowrap; }
  .col-nowrap { white-space: nowrap; }

  .tag-list { display: flex; flex-wrap: wrap; gap: 4px; }
  .tag {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 5px;
    font-family: var(--font-mono);
    font-size: 11px;
    background: rgba(148, 163, 184, 0.12);
    color: var(--text);
    border: 1px solid var(--border);
  }
  .null-badge {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 5px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-dim);
    background: rgba(100, 116, 139, 0.12);
    border: 1px dashed rgba(148, 163, 184, 0.22);
  }

  /* Line Items & Invoice Refs Cells */
  .col-line-items {
    white-space: normal;
    min-width: 260px;
    max-width: 420px;
  }
  .line-items-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .line-item-card {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 9px;
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  .line-item-top {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 8px;
    flex-wrap: wrap;
  }
  .line-item-desc {
    font-family: var(--font-sans);
    font-weight: 600;
    font-size: 12px;
    color: #f1f5f9;
    word-break: break-word;
  }
  .line-item-date {
    display: inline-block;
    padding: 1px 6px;
    border-radius: 999px;
    font-family: var(--font-mono);
    font-size: 10.5px;
    color: var(--accent);
    background: var(--accent-soft);
    border: 1px solid rgba(56, 189, 248, 0.25);
    white-space: nowrap;
  }
  .line-item-math {
    font-family: var(--font-mono);
    font-size: 11.5px;
    color: var(--text-muted);
    white-space: nowrap;
  }
  .line-item-math strong {
    color: var(--emerald);
    font-weight: 600;
  }

  .col-refs {
    white-space: normal;
    min-width: 180px;
    max-width: 340px;
  }
  .ref-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 2px 8px;
    border-radius: 5px;
    font-family: var(--font-mono);
    font-size: 11.5px;
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.26);
    color: var(--text);
    white-space: nowrap;
  }
  .ref-chip-id {
    color: var(--accent);
    font-weight: 600;
  }
  .ref-chip-amt {
    color: var(--emerald);
  }

  /* Trajectory Accordion */
  .traj-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  details.traj-step {
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }
  details.traj-step[open] { border-color: var(--border-strong); }
  details.traj-step summary {
    list-style: none;
    cursor: pointer;
    padding: 10px 14px;
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    font-size: 13px;
  }
  details.traj-step summary::-webkit-details-marker { display: none; }
  details.traj-step summary:hover { background: rgba(255, 255, 255, 0.02); }
  .step-num {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-dim);
    min-width: 54px;
  }
  .step-summary {
    color: var(--text);
    font-family: var(--font-mono);
    font-size: 12px;
    flex: 1 1 260px;
  }
  .step-body {
    border-top: 1px solid var(--border);
    padding: 12px 14px;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 12px;
    background: #0b101b;
  }
  pre.json-box {
    margin: 0;
    padding: 10px 12px;
    background: #070b12;
    border: 1px solid var(--border);
    border-radius: 6px;
    font-family: var(--font-mono);
    font-size: 11.5px;
    color: #cbd5e1;
    overflow: auto;
    max-height: 280px;
    white-space: pre-wrap;
    word-break: break-word;
  }

  /* Inspector split layout */
  .two-col {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
    gap: 16px;
  }
  .subtabs {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
  }
  .subtab {
    padding: 5px 12px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--bg-elev);
    color: var(--text-muted);
    font-family: var(--font-mono);
    font-size: 12px;
    cursor: pointer;
  }
  .subtab.active {
    color: #031018;
    background: var(--accent);
    border-color: var(--accent);
    font-weight: 700;
  }

  /* Calculator */
  .calc-row {
    display: flex;
    gap: 8px;
    margin-bottom: 10px;
  }
  .calc-row input { flex: 1; }
  .calc-presets {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 12px;
  }
  .preset-chip {
    background: var(--bg-elev);
    border: 1px solid var(--border);
    color: var(--text-muted);
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 11px;
    font-family: var(--font-mono);
    cursor: pointer;
  }
  .preset-chip:hover { border-color: var(--accent); color: var(--text); }
  .calc-output {
    padding: 12px 14px;
    border-radius: 8px;
    background: #070b12;
    border: 1px solid var(--border);
    font-family: var(--font-mono);
    font-size: 14px;
    min-height: 46px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  /* Benchmark bars */
  .bars-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
    gap: 16px;
    margin-top: 16px;
  }
  .bar-row {
    margin-bottom: 12px;
  }
  .bar-meta {
    display: flex;
    justify-content: space-between;
    font-family: var(--font-mono);
    font-size: 12px;
    margin-bottom: 5px;
  }
  .bar-track {
    height: 12px;
    border-radius: 999px;
    background: #090e18;
    border: 1px solid var(--border);
    overflow: hidden;
    position: relative;
  }
  .bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.35s ease;
  }

  .empty-state {
    padding: 28px 16px;
    text-align: center;
    color: var(--text-muted);
    font-size: 13px;
  }
  .error-banner {
    padding: 10px 14px;
    border-radius: 8px;
    background: var(--rose-soft);
    border: 1px solid rgba(244, 63, 94, 0.35);
    color: #fecdd3;
    font-size: 13px;
    margin-bottom: 14px;
    display: none;
  }
  .spinner {
    display: inline-block;
    width: 13px;
    height: 13px;
    border: 2px solid rgba(255,255,255,0.25);
    border-top-color: currentColor;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="app-shell">
  <header class="topbar">
    <div class="brand">
      <div class="brand-mark">AR</div>
      <div>
        <div class="brand-title">Audit RL // FinTech Audit Environment</div>
        <div class="brand-sub">Deterministic Ledger Reconciliation · Subprocess Eval Isolation · Zero-CDN Local UI</div>
      </div>
    </div>
    <div class="status-pills">
      <span class="pill"><span class="pill-dot"></span>100% OFFLINE</span>
      <span class="pill pill-sky" id="cfg-counts-pill">Loading configs…</span>
    </div>
  </header>

  <nav class="tabs" role="tablist">
    <button class="tab-btn active" data-tab="runner">1. Episode Runner &amp; Trajectory</button>
    <button class="tab-btn" data-tab="inspector">2. Data &amp; Tool Inspector</button>
    <button class="tab-btn" data-tab="benchmark">3. 20-Episode Benchmark</button>
  </nav>

  <!-- TAB 1: EPISODE RUNNER -->
  <section id="tab-runner" class="tab-panel active">
    <div class="card">
      <div class="controls-grid">
        <div class="field">
          <label for="run-split">Dataset Split</label>
          <select id="run-split">
            <option value="dev" selected>dev (10)</option>
            <option value="train">train (10)</option>
            <option value="eval">eval (20)</option>
          </select>
        </div>
        <div class="field grow">
          <label for="run-episode">Episode</label>
          <select id="run-episode"></select>
        </div>
        <div class="field">
          <label for="run-policy">Policy</label>
          <select id="run-policy">
            <option value="simple_reconciliation">simple_reconciliation</option>
            <option value="random_explorer">random_explorer</option>
            <option value="flag_everything_attack">flag_everything_attack</option>
            <option value="oracle">oracle (evaluator ceiling)</option>
          </select>
        </div>
        <button id="btn-run-episode" class="btn btn-primary">Run Episode</button>
      </div>
    </div>

    <div id="run-error" class="error-banner"></div>
    <div id="run-results" style="display:none;">
      <div class="meta-strip" id="run-meta"></div>

      <div class="kpi-grid" id="run-kpis"></div>

      <div class="card">
        <div class="card-header">
          <h2 class="card-title">Submitted Findings</h2>
          <span class="pill" id="findings-count-pill">0 findings</span>
        </div>
        <div id="findings-container"></div>
      </div>

      <div class="card">
        <div class="card-header">
          <h2 class="card-title">Step-by-Step Trajectory</h2>
          <div style="display:flex; gap:8px;">
            <button class="btn btn-sm" id="btn-expand-all">Expand All</button>
            <button class="btn btn-sm" id="btn-collapse-all">Collapse All</button>
          </div>
        </div>
        <div class="traj-list" id="trajectory-container"></div>
      </div>
    </div>
    <div id="run-empty" class="card empty-state">
      Select a split, episode, and policy above, then click <strong>Run Episode</strong> to execute inside <code>SubprocessAuditEnv(mode="eval")</code>.
    </div>
  </section>

  <!-- TAB 2: DATA & TOOL INSPECTOR -->
  <section id="tab-inspector" class="tab-panel">
    <div class="card">
      <div class="controls-grid">
        <div class="field">
          <label for="ins-split">Dataset Split</label>
          <select id="ins-split">
            <option value="dev" selected>dev (10)</option>
            <option value="train">train (10)</option>
            <option value="eval">eval (20)</option>
          </select>
        </div>
        <div class="field grow">
          <label for="ins-episode">Episode</label>
          <select id="ins-episode"></select>
        </div>
        <div class="field" style="min-width:100px;">
          <label for="ins-limit">Page Size (&le;50)</label>
          <select id="ins-limit">
            <option value="10">10</option>
            <option value="25" selected>25</option>
            <option value="50">50</option>
          </select>
        </div>
        <button id="btn-ins-load" class="btn btn-primary">Inspect</button>
      </div>
    </div>

    <div id="ins-error" class="error-banner"></div>

    <div class="card">
      <div class="card-header">
        <div class="subtabs" id="ins-table-tabs">
          <button class="subtab active" data-table="invoices">invoices</button>
          <button class="subtab" data-table="payments">payments</button>
          <button class="subtab" data-table="ledger">ledger</button>
          <button class="subtab" data-table="fx">fx</button>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
          <span class="pill" id="ins-page-info">No data loaded</span>
          <button class="btn btn-sm" id="btn-ins-prev" disabled>&larr; Prev</button>
          <button class="btn btn-sm" id="btn-ins-next" disabled>Next &rarr;</button>
        </div>
      </div>
      <div class="table-wrap" id="ins-table-wrap">
        <div class="empty-state">Loading table records via <code>AuditEnv.step({"action": "list_records", ...})</code>…</div>
      </div>
    </div>

    <div class="two-col">
      <div class="card">
        <div class="card-header">
          <h2 class="card-title">Manifest &amp; Policy</h2>
          <span class="pill" id="ins-ep-id-pill">—</span>
        </div>
        <div class="two-col" style="grid-template-columns: 1fr 1fr; gap: 10px;">
          <div>
            <label style="display:block; margin-bottom:6px;">manifest.json</label>
            <pre class="json-box" id="ins-manifest-json">{}</pre>
          </div>
          <div>
            <label style="display:block; margin-bottom:6px;">policy.json</label>
            <pre class="json-box" id="ins-policy-json">{}</pre>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <h2 class="card-title">Safe Decimal Calculator</h2>
          <span class="pill pill-violet">AST · No eval()</span>
        </div>
        <div class="calc-row">
          <input type="text" id="calc-input" class="mono-input" value="round(123450 * 1.0875, 2)" placeholder="e.g. round(12500 * 1.0825, 2) or abs(100 - 142.5)" />
          <button id="btn-calc-eval" class="btn btn-primary">Evaluate</button>
        </div>
        <div class="calc-presets">
          <button class="preset-chip" data-expr="round(123450 * 1.0875, 2)">round(123450 * 1.0875, 2)</button>
          <button class="preset-chip" data-expr="abs(45200 - 41950)">abs(45200 - 41950)</button>
          <button class="preset-chip" data-expr="round(19999 / 100 * 0.9154, 2)">FX conversion</button>
          <button class="preset-chip" data-expr="__import__('os').system('id')">Safety test</button>
        </div>
        <div class="calc-output" id="calc-output">
          <span style="color:var(--text-muted);">Press Evaluate or Enter</span>
        </div>
      </div>
    </div>
  </section>

  <!-- TAB 3: 20-EPISODE BENCHMARK -->
  <section id="tab-benchmark" class="tab-panel">
    <div class="card">
      <div class="card-header" style="margin-bottom:0;">
        <div>
          <h2 class="card-title" style="margin-bottom:4px;">Held-Out 20-Episode Benchmark Suite</h2>
          <div style="font-size:12.5px; color:var(--text-muted);">
            Evaluates all 4 baseline policies across all 20 <code>eval_configs.json</code> episodes inside isolated <code>SubprocessAuditEnv(mode="eval")</code>.
          </div>
        </div>
        <button id="btn-run-baselines" class="btn btn-primary">Run All 4 Baselines (20 Eval Episodes)</button>
      </div>
    </div>

    <div id="bench-error" class="error-banner"></div>
    <div id="bench-results" style="display:none;">
      <div class="card">
        <div class="card-header">
          <h2 class="card-title">Comparative Metrics Table</h2>
          <span class="pill pill-emerald" id="bench-ep-pill">20 eval episodes</span>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Policy</th>
                <th class="num">Precision</th>
                <th class="num">Recall</th>
                <th class="num">F1</th>
                <th class="num">Mean Return</th>
                <th class="num">Avg Steps</th>
                <th class="num">Avg ms</th>
                <th class="num">Total ms</th>
              </tr>
            </thead>
            <tbody id="bench-tbody"></tbody>
          </table>
        </div>
      </div>

      <div class="bars-grid">
        <div class="card">
          <div class="card-header"><h2 class="card-title">Mean F1 Score (0.0 – 1.0)</h2></div>
          <div id="bench-f1-bars"></div>
        </div>
        <div class="card">
          <div class="card-header"><h2 class="card-title">Mean Return (Reward)</h2></div>
          <div id="bench-return-bars"></div>
        </div>
      </div>
    </div>
    <div id="bench-empty" class="card empty-state">
      Click <strong>Run All 4 Baselines (20 Eval Episodes)</strong> to benchmark <code>random_explorer</code>, <code>simple_reconciliation</code>, <code>flag_everything_attack</code>, and <code>oracle</code> live.
    </div>
  </section>
</div>

<script>
(function() {
  const state = {
    configs: { train: [], dev: [], eval: [] },
    inspector: { table: "invoices", cursor: "0", nextCursor: null, cursorStack: [] },
  };

  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

  async function api(url, opts = {}) {
    const res = await fetch(url, {
      headers: { "Content-Type": "application/json" },
      ...opts,
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || `HTTP ${res.status}`);
    }
    return data;
  }

  function showError(id, msg) {
    const el = $(id);
    if (!msg) {
      el.style.display = "none";
      el.textContent = "";
    } else {
      el.style.display = "block";
      el.textContent = msg;
    }
  }

  // Tab switching
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      $("tab-" + btn.dataset.tab).classList.add("active");
    });
  });

  function populateEpisodes(selectId, split) {
    const list = state.configs[split] || [];
    const sel = $(selectId);
    sel.innerHTML = list.map((ep) => {
      const tag = ep.clean ? "clean" : Object.keys(ep.error_counts || {}).join(",") || "multi";
      return `<option value="${ep.index}">#${ep.index} · seed ${ep.seed} · ${esc(ep.description)} [${esc(ep.base_currency)}]</option>`;
    }).join("");
    if (split === "dev" && list.length > 7) {
      sel.value = "7"; // seed 2008 multi-error episode as great default
    }
  }

  // Load configs on startup
  async function init() {
    try {
      const cfgs = await api("/api/configs");
      state.configs = cfgs;
      $("cfg-counts-pill").textContent =
        `train:${cfgs.train.length} · dev:${cfgs.dev.length} · eval:${cfgs.eval.length}`;
      populateEpisodes("run-episode", $("run-split").value);
      populateEpisodes("ins-episode", $("ins-split").value);
      loadInspector(true);
    } catch (err) {
      $("cfg-counts-pill").textContent = "Config load failed";
      showError("run-error", "Failed to load configs: " + err.message);
    }
  }

  $("run-split").addEventListener("change", (e) => populateEpisodes("run-episode", e.target.value));
  $("ins-split").addEventListener("change", (e) => {
    populateEpisodes("ins-episode", e.target.value);
    loadInspector(true);
  });
  $("ins-episode").addEventListener("change", () => loadInspector(true));
  $("ins-limit").addEventListener("change", () => loadInspector(true));

  // TAB 1: Run Episode
  function actionPillClass(act) {
    if (act === "submit") return "pill-emerald";
    if (act === "calculator") return "pill-violet";
    if (act === "get_document" || act === "get_record") return "pill-amber";
    return "pill-sky";
  }

  $("btn-run-episode").addEventListener("click", async () => {
    const btn = $("btn-run-episode");
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> Running…`;
    showError("run-error", "");

    try {
      const payload = {
        split: $("run-split").value,
        index: parseInt($("run-episode").value, 10),
        policy: $("run-policy").value,
      };
      const data = await api("/api/episode/run", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      renderEpisodeRun(data);
    } catch (err) {
      showError("run-error", err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = "Run Episode";
    }
  });

  function renderEpisodeRun(data) {
    $("run-empty").style.display = "none";
    $("run-results").style.display = "block";

    const m = data.manifest || {};
    const p = data.policy || {};
    const r = data.report || {};

    $("run-meta").innerHTML = `
      <span>episode: <strong>${esc(m.episode_id)}</strong></span>
      <span>company: <strong>${esc(m.company_name || "N/A")}</strong></span>
      <span>currency: <strong>${esc(m.base_currency)}</strong></span>
      <span>period: <strong>${esc((m.period_start || "").slice(0, 10))} &rarr; ${esc((m.period_end || "").slice(0, 10))}</strong></span>
      <span>fx_policy: <strong>${esc(p.fx_date_policy)}</strong> (${esc(p.rounding_rule)})</span>
      <span>runtime: <strong>${r.runtime_ms} ms</strong></span>
    `;

    const f1Class = r.f1 >= 0.85 ? "val-good" : (r.f1 >= 0.4 ? "val-warn" : "val-bad");
    const rewClass = r.reward > 0 ? "val-good" : (r.reward === 0 ? "val-warn" : "val-bad");

    $("run-kpis").innerHTML = `
      <div class="kpi">
        <div class="kpi-label">F1 Score</div>
        <div class="kpi-val ${f1Class}">${Number(r.f1).toFixed(4)}</div>
        <div class="kpi-sub">${r.valid_submission ? "valid submission" : "invalid submission"}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Reward</div>
        <div class="kpi-val ${rewClass}">${Number(r.reward).toFixed(4)}</div>
        <div class="kpi-sub">policy: ${esc(data.policy_name)}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Precision</div>
        <div class="kpi-val">${Number(r.precision).toFixed(4)}</div>
        <div class="kpi-sub">submitted: ${r.submitted_count}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Recall</div>
        <div class="kpi-val">${Number(r.recall).toFixed(4)}</div>
        <div class="kpi-sub">truth issues: ${r.truth_count}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Steps</div>
        <div class="kpi-val val-sky">${r.steps}</div>
        <div class="kpi-sub">inspection: ${r.inspection_steps}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">TP / FP / FN</div>
        <div class="kpi-val"><span class="val-good">${r.tp}</span> / <span class="${r.fp > 0 ? "val-warn" : ""}">${r.fp}</span> / <span class="${r.fn > 0 ? "val-bad" : ""}">${r.fn}</span></div>
        <div class="kpi-sub">evaluator mode=eval</div>
      </div>
    `;

    // Findings
    const findings = data.submitted_findings || [];
    $("findings-count-pill").textContent = `${findings.length} finding${findings.length === 1 ? "" : "s"}`;
    if (findings.length === 0) {
      $("findings-container").innerHTML = `<div class="empty-state">No findings submitted (clean books or empty submission).</div>`;
    } else {
      const rows = findings.map((f, i) => `
        <tr>
          <td class="mono">${i + 1}</td>
          <td><span class="pill pill-amber">${esc(f.type)}</span></td>
          <td><div class="tag-list">${(f.affected_ids || []).map((id) => `<span class="tag">${esc(id)}</span>`).join("")}</div></td>
          <td class="num">${esc(f.claimed_amount_minor)} ${esc(f.currency)}</td>
          <td><div class="tag-list">${(f.evidence_ids || []).map((id) => `<span class="tag">${esc(id)}</span>`).join("")}</div></td>
          <td style="color:var(--text-muted); max-width:340px;">${esc(f.explanation || "")}</td>
        </tr>
      `).join("");
      $("findings-container").innerHTML = `
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Type</th>
                <th>Affected IDs</th>
                <th class="num">Amount (minor)</th>
                <th>Evidence IDs</th>
                <th>Explanation</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      `;
    }

    // Trajectory
    const traj = data.trajectory || [];
    $("trajectory-container").innerHTML = traj.map((t) => `
      <details class="traj-step">
        <summary>
          <span class="step-num">STEP ${String(t.step).padStart(2, "0")}</span>
          <span class="pill ${actionPillClass(t.action_type)}">${esc(t.action_type)}</span>
          <span class="step-summary">${esc(t.observation_summary)}</span>
          ${t.terminated ? `<span class="pill pill-emerald">terminated</span>` : ""}
          ${t.truncated ? `<span class="pill pill-rose">truncated</span>` : ""}
        </summary>
        <div class="step-body">
          <div>
            <label style="display:block; margin-bottom:6px;">Action Payload</label>
            <pre class="json-box">${esc(JSON.stringify(t.action, null, 2))}</pre>
          </div>
          <div>
            <label style="display:block; margin-bottom:6px;">Observation Payload</label>
            <pre class="json-box">${esc(JSON.stringify(t.observation, null, 2))}</pre>
          </div>
        </div>
      </details>
    `).join("");
  }

  $("btn-expand-all").addEventListener("click", () => {
    document.querySelectorAll("#trajectory-container details").forEach((d) => { d.open = true; });
  });
  $("btn-collapse-all").addEventListener("click", () => {
    document.querySelectorAll("#trajectory-container details").forEach((d) => { d.open = false; });
  });

  // TAB 2: Data & Tool Inspector
  document.querySelectorAll("#ins-table-tabs .subtab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#ins-table-tabs .subtab").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.inspector.table = btn.dataset.table;
      loadInspector(true);
    });
  });

  $("btn-ins-load").addEventListener("click", () => loadInspector(true));
  $("btn-ins-next").addEventListener("click", () => {
    if (state.inspector.nextCursor !== null) {
      state.inspector.cursorStack.push(state.inspector.cursor);
      state.inspector.cursor = state.inspector.nextCursor;
      loadInspector(false);
    }
  });
  $("btn-ins-prev").addEventListener("click", () => {
    if (state.inspector.cursorStack.length > 0) {
      state.inspector.cursor = state.inspector.cursorStack.pop();
      loadInspector(false);
    }
  });

  async function loadInspector(resetCursor) {
    if (resetCursor) {
      state.inspector.cursor = "0";
      state.inspector.cursorStack = [];
    }
    showError("ins-error", "");
    try {
      const payload = {
        split: $("ins-split").value,
        index: parseInt($("ins-episode").value || "0", 10),
        table: state.inspector.table,
        cursor: state.inspector.cursor,
        limit: parseInt($("ins-limit").value, 10),
      };
      const data = await api("/api/episode/inspect", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      state.inspector.nextCursor = data.next_cursor;
      $("ins-ep-id-pill").textContent = data.episode_id || "—";
      $("ins-manifest-json").textContent = JSON.stringify(data.manifest, null, 2);
      $("ins-policy-json").textContent = JSON.stringify(data.policy, null, 2);

      const start = parseInt(data.cursor, 10) + (data.count > 0 ? 1 : 0);
      const end = parseInt(data.cursor, 10) + data.count;
      $("ins-page-info").textContent = `${data.table}: ${start}–${end} of ${data.total_count}`;
      $("btn-ins-prev").disabled = state.inspector.cursorStack.length === 0;
      $("btn-ins-next").disabled = data.next_cursor === null;

      renderInspectorTable(data.records || []);
    } catch (err) {
      showError("ins-error", err.message);
    }
  }

  const NOWRAP_COLS = new Set([
    "doc_id", "invoice_id", "payment_id", "row_id", "entity_id",
    "issued_at", "due_at", "timestamp", "date",
    "currency", "source_currency", "target_currency",
    "total_minor", "amount_minor", "debit_minor", "credit_minor", "rate"
  ]);

  function renderLineItemsCell(items) {
    if (!Array.isArray(items) || items.length === 0) {
      return `<td class="col-line-items"><span class="null-badge">none</span></td>`;
    }
    const cards = items.map((it) => {
      const desc = esc(it.description ?? "Item");
      const qty = esc(it.quantity ?? 1);
      const unit = esc(it.unit_price_minor ?? 0);
      const amt = esc(it.amount_minor ?? 0);
      const sdate = it.service_date ? `<span class="line-item-date">${esc(it.service_date)}</span>` : "";
      return `
        <div class="line-item-card">
          <div class="line-item-top">
            <span class="line-item-desc">${desc}</span>
            ${sdate}
          </div>
          <div class="line-item-math">${qty} &times; ${unit} = <strong>${amt}</strong></div>
        </div>
      `;
    }).join("");
    return `<td class="col-line-items"><div class="line-items-list">${cards}</div></td>`;
  }

  function renderInvoiceRefsCell(refs) {
    if (!Array.isArray(refs) || refs.length === 0) {
      return `<td class="col-refs"><span class="null-badge">null</span></td>`;
    }
    const chips = refs.map((ref) => {
      if (ref === null || ref === undefined) {
        return `<span class="null-badge">null</span>`;
      }
      if (typeof ref === "string") {
        return `<span class="ref-chip"><span class="ref-chip-id">${esc(ref)}</span></span>`;
      }
      const invId = ref.invoice_id;
      const alloc = ref.allocated_minor ?? ref.amount_minor;
      if (invId === null || invId === undefined || invId === "") {
        return `<span class="null-badge">null${alloc !== undefined ? `: ${esc(alloc)}` : ""}</span>`;
      }
      return `<span class="ref-chip"><span class="ref-chip-id">${esc(invId)}</span>${alloc !== undefined ? `: <span class="ref-chip-amt">${esc(alloc)}</span>` : ""}</span>`;
    }).join("");
    return `<td class="col-refs"><div class="tag-list">${chips}</div></td>`;
  }

  function renderInspectorTable(records) {
    const wrap = $("ins-table-wrap");
    if (!records.length) {
      wrap.innerHTML = `<div class="empty-state">No records in this page.</div>`;
      return;
    }
    const cols = [];
    records.forEach((r) => {
      Object.keys(r).forEach((k) => {
        if (!cols.includes(k)) cols.push(k);
      });
    });
    const thead = `<tr>${cols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>`;
    const tbody = records.map((r) => {
      return `<tr>${cols.map((c) => {
        const v = r[c];
        if (c === "line_items") return renderLineItemsCell(v);
        if (c === "invoice_refs") return renderInvoiceRefsCell(v);
        const nowrap = NOWRAP_COLS.has(c) ? " col-nowrap" : "";
        if (v === null || v === undefined) return `<td class="mono${nowrap}"><span class="null-badge">null</span></td>`;
        if (typeof v === "object") return `<td class="mono" style="white-space:normal; word-break:break-word;">${esc(JSON.stringify(v))}</td>`;
        if (typeof v === "number") return `<td class="num${nowrap}">${esc(v)}</td>`;
        return `<td class="mono${nowrap}">${esc(v)}</td>`;
      }).join("")}</tr>`;
    }).join("");
    wrap.innerHTML = `<table><thead>${thead}</thead><tbody>${tbody}</tbody></table>`;
  }

  // Calculator
  async function evalCalc() {
    const expr = $("calc-input").value;
    try {
      const data = await api("/api/calculator", {
        method: "POST",
        body: JSON.stringify({ expression: expr }),
      });
      if (data.ok) {
        $("calc-output").innerHTML = `
          <span style="color:var(--text-muted);">${esc(data.expression)} =</span>
          <strong class="val-good" style="font-size:18px;">${esc(data.result)}</strong>
        `;
      } else {
        $("calc-output").innerHTML = `<span class="val-bad">Error: ${esc(data.error)}</span>`;
      }
    } catch (err) {
      $("calc-output").innerHTML = `<span class="val-bad">Error: ${esc(err.message)}</span>`;
    }
  }

  $("btn-calc-eval").addEventListener("click", evalCalc);
  $("calc-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") evalCalc();
  });
  document.querySelectorAll(".preset-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      $("calc-input").value = chip.dataset.expr;
      evalCalc();
    });
  });

  // TAB 3: 20-Episode Benchmark
  $("btn-run-baselines").addEventListener("click", async () => {
    const btn = $("btn-run-baselines");
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> Running 20 Eval Episodes…`;
    showError("bench-error", "");

    try {
      const data = await api("/api/baselines", {
        method: "POST",
        body: "{}",
      });
      renderBenchmark(data);
    } catch (err) {
      showError("bench-error", err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = "Run All 4 Baselines (20 Eval Episodes)";
    }
  });

  function renderBenchmark(data) {
    $("bench-empty").style.display = "none";
    $("bench-results").style.display = "block";
    $("bench-ep-pill").textContent = `${data.num_episodes} eval episodes`;

    const summary = data.summary || {};
    const names = Object.keys(summary);
    const colors = {
      oracle: "#a78bfa",
      simple_reconciliation: "#10b981",
      random_explorer: "#38bdf8",
      flag_everything_attack: "#f43f5e",
    };

    $("bench-tbody").innerHTML = names.map((name) => {
      const s = summary[name];
      const retClass = s.return > 0 ? "val-good" : (s.return < 0 ? "val-bad" : "");
      return `
        <tr>
          <td class="mono" style="font-weight:700; color:${colors[name] || "#e2e8f0"};">${esc(name)}</td>
          <td class="num">${Number(s.precision).toFixed(4)}</td>
          <td class="num">${Number(s.recall).toFixed(4)}</td>
          <td class="num" style="font-weight:700;">${Number(s.f1).toFixed(4)}</td>
          <td class="num ${retClass}" style="font-weight:700;">${Number(s.return).toFixed(4)}</td>
          <td class="num">${Number(s.steps).toFixed(2)}</td>
          <td class="num">${Number(s.runtime_ms).toFixed(2)}</td>
          <td class="num">${Number(s.total_runtime_ms).toFixed(2)}</td>
        </tr>
      `;
    }).join("");

    // F1 Bars
    $("bench-f1-bars").innerHTML = names.map((name) => {
      const val = Number(summary[name].f1 || 0);
      const pct = Math.max(0, Math.min(100, val * 100));
      return `
        <div class="bar-row">
          <div class="bar-meta">
            <span>${esc(name)}</span>
            <strong>${val.toFixed(4)}</strong>
          </div>
          <div class="bar-track">
            <div class="bar-fill" style="width:${pct}%; background:${colors[name] || "#38bdf8"};"></div>
          </div>
        </div>
      `;
    }).join("");

    // Return Bars (scale across min..max, highlighting negative bars in rose)
    const returns = names.map((n) => Number(summary[n].return || 0));
    const maxAbs = Math.max(1.0, ...returns.map((v) => Math.abs(v)));
    $("bench-return-bars").innerHTML = names.map((name) => {
      const val = Number(summary[name].return || 0);
      const pct = Math.min(100, (Math.abs(val) / maxAbs) * 100);
      const barColor = val < 0 ? "#f43f5e" : (colors[name] || "#10b981");
      return `
        <div class="bar-row">
          <div class="bar-meta">
            <span>${esc(name)}</span>
            <strong class="${val < 0 ? "val-bad" : "val-good"}">${val >= 0 ? "+" : ""}${val.toFixed(4)}</strong>
          </div>
          <div class="bar-track">
            <div class="bar-fill" style="width:${pct}%; background:${barColor};"></div>
          </div>
        </div>
      `;
    }).join("");
  }

  init();
})();
</script>
</body>
</html>
"""
