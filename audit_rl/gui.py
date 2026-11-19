"""Zero-dependency offline local Web GUI for the FinTech Audit RL Environment.

Serves a rich, single-page editorial dark FinTech dashboard over Python stdlib
``http.server.ThreadingHTTPServer`` with JSON API endpoints backed by
``SubprocessAuditEnv``, ``AuditEnv``, ``safe_calculate``, and ``audit_rl.rubric``.
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


def _classify_episode_category(cfg: dict[str, Any]) -> str:
    if cfg.get("clean"):
        return "clean"
    if cfg.get("interacting_scenarios"):
        return "interacting"
    err_counts = cfg.get("error_counts") or {}
    active_types = [k for k, v in err_counts.items() if int(v or 0) > 0]
    if len(active_types) == 0:
        return "clean"
    if len(active_types) == 1:
        return "single"
    return "multi"


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
                    "category": _classify_episode_category(cfg),
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
        max_steps = int(info.get("max_steps", obs.get("budget_remaining", 100)))
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
                    "budget_remaining": next_obs.get("budget_remaining", max(0, max_steps - step_num)),
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
        "max_steps": max_steps,
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


def _lookup_record_api(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Resolve an evidence or entity ID via AuditEnv get_document / get_record."""
    seed, cfg_dict, _ = _resolve_episode_spec(root, payload)
    raw_id = payload.get("id")
    if not isinstance(raw_id, str) or not raw_id.strip():
        raise ValueError("Missing or empty 'id' to look up")
    rec_id = raw_id.strip()

    env = AuditEnv(mode="eval")
    reset_obs, _ = env.reset(seed=seed, config=cfg_dict)

    attempts: list[tuple[str, dict[str, Any]]] = []
    if rec_id.startswith("DOC-"):
        attempts.append(("document", {"action": "get_document", "doc_id": rec_id}))
    elif rec_id.startswith("ROW-"):
        attempts.append(("ledger", {"action": "get_record", "table": "ledger", "id": rec_id}))
    elif rec_id.startswith("INV-"):
        attempts.append(("invoices", {"action": "get_record", "table": "invoices", "id": rec_id}))
    elif rec_id.startswith("PAY-"):
        attempts.append(("payments", {"action": "get_record", "table": "payments", "id": rec_id}))
    elif ":" in rec_id:
        attempts.append(("fx", {"action": "get_record", "table": "fx", "id": rec_id}))
    else:
        attempts.extend(
            [
                ("document", {"action": "get_document", "doc_id": rec_id}),
                ("invoices", {"action": "get_record", "table": "invoices", "id": rec_id}),
                ("payments", {"action": "get_record", "table": "payments", "id": rec_id}),
                ("ledger", {"action": "get_record", "table": "ledger", "id": rec_id}),
                ("fx", {"action": "get_record", "table": "fx", "id": rec_id}),
            ]
        )

    last_error = f"Record {rec_id!r} not found"
    for source, action in attempts:
        obs, _, _, _, _ = env.step(action)
        if obs.get("status") == "ok":
            data = obs.get("document") if source == "document" else obs.get("record")
            return {
                "ok": True,
                "id": rec_id,
                "episode_id": reset_obs["manifest"].get("episode_id", ""),
                "action": action["action"],
                "source": source,
                "data": data,
            }
        last_error = str(obs.get("error", last_error))

    return {
        "ok": False,
        "id": rec_id,
        "episode_id": reset_obs["manifest"].get("episode_id", ""),
        "error": last_error,
    }


def _calculator_api(payload: dict[str, Any]) -> dict[str, Any]:
    expression = payload.get("expression", "")
    try:
        result = safe_calculate(expression)
        return {"ok": True, "expression": expression, "result": result}
    except CalculatorError as exc:
        return {"ok": False, "expression": expression, "error": str(exc)}


def _build_episode_matrix(root: Path, details: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    spec = _load_json(root / "configs" / "eval_configs.json")
    episodes = spec.get("episodes", [])
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for pol_name, rows in details.items():
        for row in rows:
            s = int(row["seed"])
            by_seed.setdefault(s, {})[pol_name] = {
                "f1": round(float(row.get("f1", 0.0)), 4),
                "reward": round(float(row.get("return", row.get("reward", 0.0))), 4),
                "precision": round(float(row.get("precision", 0.0)), 4),
                "recall": round(float(row.get("recall", 0.0)), 4),
                "steps": int(row.get("steps", 0)),
            }

    matrix: list[dict[str, Any]] = []
    for ep_spec in episodes:
        seed = int(ep_spec["seed"])
        cfg = ep_spec.get("config", {})
        err_counts = cfg.get("error_counts") or {}
        num_issues = sum(int(v or 0) for v in err_counts.values()) + len(cfg.get("interacting_scenarios") or [])
        matrix.append(
            {
                "seed": seed,
                "label": str(ep_spec.get("description", f"seed {seed}")),
                "category": _classify_episode_category(cfg),
                "num_issues": num_issues,
                "results": by_seed.get(seed, {}),
            }
        )
    return matrix


def _baselines_api(root: Path) -> dict[str, Any]:
    # Lazy import to avoid circular import between cli.py and gui.py
    from audit_rl.cli import evaluate_baselines

    eval_cfg = root / "configs" / "eval_configs.json"
    result = evaluate_baselines(eval_cfg)
    if "episode_matrix" not in result:
        result["episode_matrix"] = _build_episode_matrix(root, result.get("details", {}))
    return result


def _verify_api(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    from audit_rl.rubric import run_full_rubric_verification

    run_unittests = bool(payload.get("run_unittests", True))
    try:
        return run_full_rubric_verification(run_unittests=run_unittests, root=root)
    except TypeError:
        return run_full_rubric_verification(run_unittests=run_unittests)


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
            if path == "/api/verify":
                result = _verify_api(self.server.root, payload)
                self._send_json(HTTPStatus.OK, result)
                return
            if path == "/api/episode/run":
                result = _run_episode_api(self.server.root, payload)
                self._send_json(HTTPStatus.OK, result)
                return
            if path == "/api/episode/inspect":
                result = _inspect_episode_api(self.server.root, payload)
                self._send_json(HTTPStatus.OK, result)
                return
            if path == "/api/record/lookup":
                result = _lookup_record_api(self.server.root, payload)
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
<title>Audit RL — FinTech Audit Environment &amp; Rubric Workbench</title>
<style>
  :root {
    --bg: #070b12;
    --bg-elev: #0d1421;
    --bg-card: #111a2b;
    --bg-card-alt: #142034;
    --bg-hover: #1a2840;
    --border: rgba(148, 163, 184, 0.14);
    --border-strong: rgba(148, 163, 184, 0.28);
    --text: #e2e8f0;
    --text-bright: #f8fafc;
    --text-muted: #94a3b8;
    --text-dim: #64748b;

    /* Semantic FinTech Palette */
    --emerald: #10b981;
    --emerald-soft: rgba(16, 185, 129, 0.14);
    --cyan: #06b6d4;
    --cyan-soft: rgba(6, 182, 212, 0.14);
    --sky: #38bdf8;
    --sky-soft: rgba(56, 189, 248, 0.14);
    --amber: #f59e0b;
    --amber-soft: rgba(245, 158, 11, 0.15);
    --rose: #f43f5e;
    --rose-soft: rgba(244, 63, 94, 0.15);
    --violet: #a855f7;
    --violet-soft: rgba(168, 85, 247, 0.15);

    --font-sans: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    --radius: 12px;
  }

  * { box-sizing: border-box; }
  html, body {
    margin: 0;
    padding: 0;
    background:
      radial-gradient(ellipse 80% 420px at 18% -80px, rgba(6, 182, 212, 0.14), transparent 65%),
      radial-gradient(ellipse 65% 360px at 82% -60px, rgba(168, 85, 247, 0.12), transparent 65%),
      radial-gradient(ellipse 50% 260px at 50% 0px, rgba(16, 185, 129, 0.08), transparent 70%),
      var(--bg);
    color: var(--text);
    font-family: var(--font-sans);
    font-size: 14px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
    min-height: 100vh;
  }

  .app-shell {
    max-width: 1440px;
    margin: 0 auto;
    padding: 18px 24px 56px;
  }

  /* Header */
  .topbar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    padding: 14px 18px;
    border: 1px solid var(--border);
    border-radius: 14px;
    background: linear-gradient(180deg, rgba(17, 26, 43, 0.92), rgba(13, 20, 33, 0.88));
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    margin-bottom: 14px;
  }
  .brand { display: flex; align-items: center; gap: 12px; }
  .brand-mark {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: linear-gradient(135deg, #06b6d4 0%, #10b981 55%, #a855f7 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--font-mono);
    font-weight: 800;
    font-size: 15px;
    color: #041016;
    letter-spacing: -0.5px;
    box-shadow: 0 0 26px rgba(6, 182, 212, 0.32);
  }
  .brand-title {
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text-bright);
  }
  .brand-sub {
    font-size: 11.5px;
    color: var(--text-muted);
    font-family: var(--font-mono);
  }
  .status-pills { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }

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
    background: rgba(13, 20, 33, 0.9);
    color: var(--text-muted);
    white-space: nowrap;
  }
  .pill-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--emerald);
    box-shadow: 0 0 8px var(--emerald);
  }
  .pill-emerald { color: var(--emerald); background: var(--emerald-soft); border-color: rgba(16, 185, 129, 0.34); }
  .pill-cyan { color: var(--cyan); background: var(--cyan-soft); border-color: rgba(6, 182, 212, 0.34); }
  .pill-sky { color: var(--sky); background: var(--sky-soft); border-color: rgba(56, 189, 248, 0.34); }
  .pill-amber { color: var(--amber); background: var(--amber-soft); border-color: rgba(245, 158, 11, 0.36); }
  .pill-rose { color: var(--rose); background: var(--rose-soft); border-color: rgba(244, 63, 94, 0.36); }
  .pill-violet { color: var(--violet); background: var(--violet-soft); border-color: rgba(168, 85, 247, 0.36); }

  /* Compact context strip */
  .context-strip {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    padding: 8px 14px;
    margin-bottom: 16px;
    border-radius: 10px;
    background: rgba(13, 20, 33, 0.72);
    border: 1px solid var(--border);
    font-family: var(--font-mono);
    font-size: 11.5px;
    color: var(--text-muted);
  }
  .legend-row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
  .legend-item { display: inline-flex; align-items: center; gap: 5px; font-size: 11px; }
  .swatch { width: 8px; height: 8px; border-radius: 2px; display: inline-block; }

  /* Tabs */
  .tabs {
    display: flex;
    gap: 8px;
    margin-bottom: 18px;
    flex-wrap: wrap;
    background: rgba(13, 20, 33, 0.75);
    border: 1px solid var(--border);
    padding: 6px;
    border-radius: 12px;
  }
  .tab-btn {
    background: transparent;
    border: 1px solid transparent;
    color: var(--text-muted);
    padding: 9px 16px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.16s ease;
    font-family: var(--font-sans);
    display: inline-flex;
    align-items: center;
    gap: 8px;
  }
  .tab-btn:hover { color: var(--text-bright); background: rgba(255, 255, 255, 0.04); }
  .tab-btn.active {
    color: var(--text-bright);
    background: linear-gradient(180deg, rgba(26, 40, 64, 0.95), rgba(17, 26, 43, 0.95));
    border-color: rgba(56, 189, 248, 0.35);
    box-shadow: 0 2px 12px rgba(6, 182, 212, 0.15);
  }
  .tab-num {
    font-family: var(--font-mono);
    font-size: 11px;
    padding: 1px 6px;
    border-radius: 5px;
    background: rgba(148, 163, 184, 0.12);
    color: var(--text-muted);
  }
  .tab-btn.active .tab-num { background: var(--cyan-soft); color: var(--cyan); }

  .tab-panel { display: none; }
  .tab-panel.active { display: block; }

  /* Cards & Controls */
  .card {
    background: linear-gradient(180deg, rgba(17, 26, 43, 0.96), rgba(13, 20, 33, 0.96));
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 18px;
    margin-bottom: 16px;
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.24), inset 0 1px 0 rgba(255, 255, 255, 0.03);
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
    font-size: 12.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--text-muted);
    margin: 0;
  }

  .controls-grid { display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-end; }
  .field { display: flex; flex-direction: column; gap: 5px; min-width: 140px; }
  .field.grow { flex: 1 1 280px; }
  label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
  }
  select, input[type="text"], input[type="number"], input[type="search"] {
    background: #0a101b;
    border: 1px solid var(--border-strong);
    color: var(--text);
    border-radius: 8px;
    padding: 8px 11px;
    font-size: 13px;
    font-family: var(--font-sans);
    outline: none;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
  }
  select:focus, input:focus {
    border-color: var(--cyan);
    box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.14);
  }
  .mono-input { font-family: var(--font-mono); }

  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 8px 15px;
    border-radius: 8px;
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
  .btn:hover:not(:disabled) { background: var(--bg-hover); border-color: var(--cyan); color: var(--text-bright); }
  .btn:disabled { opacity: 0.45; cursor: not-allowed; }
  .btn-primary {
    background: linear-gradient(135deg, #0891b2, #06b6d4 55%, #10b981);
    border-color: rgba(6, 182, 212, 0.6);
    color: #031217;
    font-weight: 700;
    box-shadow: 0 4px 16px rgba(6, 182, 212, 0.25);
  }
  .btn-primary:hover:not(:disabled) {
    filter: brightness(1.08);
    border-color: #67e8f9;
    color: #020c10;
  }
  .btn-sm { padding: 5px 10px; font-size: 12px; }

  /* Tab 1: Rubric Hero & Category Cards */
  .rubric-hero {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 28px;
    padding: 20px 22px;
    background:
      radial-gradient(circle at 12% 50%, rgba(16, 185, 129, 0.12), transparent 55%),
      radial-gradient(circle at 88% 30%, rgba(6, 182, 212, 0.1), transparent 55%),
      linear-gradient(180deg, rgba(17, 26, 43, 0.98), rgba(11, 18, 30, 0.98));
  }
  .score-ring-wrap {
    position: relative;
    width: 148px;
    height: 148px;
    flex-shrink: 0;
  }
  .score-ring-center {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
  }
  .score-ring-val {
    font-family: var(--font-mono);
    font-size: 30px;
    font-weight: 800;
    color: var(--text-bright);
    line-height: 1;
  }
  .score-ring-max {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 4px;
  }
  .rubric-hero-meta { flex: 1 1 320px; }
  .rubric-hero-title {
    font-size: 20px;
    font-weight: 700;
    color: var(--text-bright);
    margin: 0 0 6px;
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }
  .rubric-hero-desc { color: var(--text-muted); font-size: 13px; margin: 0 0 14px; }
  .rubric-stats-row { display: flex; flex-wrap: wrap; gap: 10px; }

  .rubric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
    gap: 14px;
  }
  .cat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-top: 3px solid var(--cyan);
    border-radius: var(--radius);
    padding: 14px 16px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .cat-top { display: flex; align-items: flex-start; justify-content: space-between; gap: 10px; }
  .cat-name { font-weight: 700; font-size: 14.5px; color: var(--text-bright); }
  .cat-sub { font-family: var(--font-mono); font-size: 11.5px; color: var(--text-muted); }
  .cat-score { font-family: var(--font-mono); font-size: 18px; font-weight: 800; }
  .prog-track {
    height: 8px;
    border-radius: 999px;
    background: #080d16;
    border: 1px solid var(--border);
    overflow: hidden;
  }
  .prog-fill { height: 100%; border-radius: 999px; transition: width 0.4s ease; }

  details.cat-checks {
    border-top: 1px solid var(--border);
    padding-top: 8px;
  }
  details.cat-checks summary {
    cursor: pointer;
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--text-muted);
    user-select: none;
  }
  details.cat-checks summary:hover { color: var(--text-bright); }
  .check-list {
    margin-top: 10px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    max-height: 320px;
    overflow-y: auto;
    padding-right: 4px;
  }
  .check-row {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 6px 8px;
    border-radius: 7px;
    background: rgba(7, 11, 18, 0.7);
    border: 1px solid var(--border);
    font-size: 12px;
  }
  .check-badge {
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 4px;
    flex-shrink: 0;
  }
  .check-pass { background: var(--emerald-soft); color: var(--emerald); border: 1px solid rgba(16, 185, 129, 0.35); }
  .check-fail { background: var(--rose-soft); color: var(--rose); border: 1px solid rgba(244, 63, 94, 0.35); }
  .check-name { font-weight: 600; color: var(--text); }
  .check-detail { font-family: var(--font-mono); font-size: 11px; color: var(--text-muted); word-break: break-word; }

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
    border-left: 3px solid var(--border-strong);
    border-radius: 10px;
    padding: 13px 15px;
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
    color: var(--text-bright);
    line-height: 1.2;
  }
  .kpi-sub { font-size: 11px; font-family: var(--font-mono); color: var(--text-dim); margin-top: 4px; }
  .val-good { color: var(--emerald); }
  .val-warn { color: var(--amber); }
  .val-bad { color: var(--rose); }
  .val-cyan { color: var(--cyan); }
  .val-violet { color: var(--violet); }

  /* Metadata strip */
  .meta-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    padding: 10px 14px;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 9px;
    margin-bottom: 16px;
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--text-muted);
  }
  .meta-strip strong { color: var(--text-bright); font-weight: 600; }

  /* Findings Cards & Clickable Evidence Chips */
  .findings-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
    gap: 12px;
  }
  .finding-card {
    background: rgba(10, 16, 27, 0.9);
    border: 1px solid var(--border);
    border-left: 4px solid var(--amber);
    border-radius: 9px;
    padding: 13px 15px;
    display: flex;
    flex-direction: column;
    gap: 9px;
  }
  .finding-top { display: flex; align-items: center; justify-content: space-between; gap: 8px; flex-wrap: wrap; }
  .finding-amount { font-family: var(--font-mono); font-size: 14px; font-weight: 700; color: var(--text-bright); }
  .finding-expl { font-size: 12.5px; color: var(--text-muted); line-height: 1.45; }
  .finding-row-label {
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-dim);
    margin-bottom: 4px;
    font-weight: 700;
  }
  .evidence-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 8px;
    border-radius: 6px;
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    background: rgba(6, 182, 212, 0.12);
    color: #67e8f9;
    border: 1px solid rgba(6, 182, 212, 0.34);
    transition: all 0.14s ease;
  }
  .evidence-chip:hover {
    background: rgba(6, 182, 212, 0.26);
    border-color: #67e8f9;
    transform: translateY(-1px);
  }

  .evidence-drawer {
    margin-top: 14px;
    padding: 14px 16px;
    border-radius: 10px;
    background: #080d17;
    border: 1px solid rgba(6, 182, 212, 0.42);
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.4);
    display: none;
  }
  .evidence-drawer.open { display: block; }

  /* Trajectory Scrubber */
  .scrubber-bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px;
    padding: 12px 14px;
    background: #0a101c;
    border: 1px solid var(--border);
    border-radius: 9px;
    margin-bottom: 12px;
  }
  .scrubber-slider {
    flex: 1 1 200px;
    accent-color: var(--cyan);
    cursor: pointer;
  }
  .budget-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 170px;
    font-family: var(--font-mono);
    font-size: 11.5px;
    color: var(--text-muted);
  }
  .budget-track {
    flex: 1;
    height: 7px;
    border-radius: 999px;
    background: #131e31;
    overflow: hidden;
  }
  .budget-fill { height: 100%; background: linear-gradient(90deg, var(--emerald), var(--cyan)); transition: width 0.2s ease; }

  .timeline-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-bottom: 14px;
  }
  .step-dot-btn {
    padding: 4px 8px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: #0a101b;
    color: var(--text-muted);
    font-family: var(--font-mono);
    font-size: 11px;
    cursor: pointer;
    transition: all 0.12s ease;
  }
  .step-dot-btn:hover { border-color: var(--cyan); color: var(--text-bright); }
  .step-dot-btn.active {
    background: var(--cyan-soft);
    border-color: var(--cyan);
    color: #67e8f9;
    font-weight: 700;
  }
  .step-dot-btn.act-submit { border-color: rgba(16, 185, 129, 0.45); color: var(--emerald); }
  .step-dot-btn.act-calculator { border-color: rgba(168, 85, 247, 0.45); color: var(--violet); }
  .step-dot-btn.act-get_document, .step-dot-btn.act-get_record { border-color: rgba(245, 158, 11, 0.45); color: var(--amber); }

  .step-stage {
    border: 1px solid var(--border-strong);
    border-radius: 10px;
    background: #090e18;
    overflow: hidden;
  }
  .step-stage-head {
    padding: 11px 14px;
    background: rgba(17, 26, 43, 0.85);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }
  .step-stage-body {
    padding: 14px;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 12px;
  }
  pre.json-box {
    margin: 0;
    padding: 10px 12px;
    background: #060910;
    border: 1px solid var(--border);
    border-radius: 7px;
    font-family: var(--font-mono);
    font-size: 11.5px;
    color: #cbd5e1;
    overflow: auto;
    max-height: 300px;
    white-space: pre-wrap;
    word-break: break-word;
  }

  /* Tables & Formatted Cells */
  .table-wrap {
    overflow-x: auto;
    max-width: 100%;
    border: 1px solid var(--border);
    border-radius: 9px;
    background: #0a101b;
  }
  table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  th, td {
    text-align: left;
    padding: 9px 12px;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
  }
  th {
    background: rgba(15, 23, 42, 0.96);
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
  tbody tr.clickable-row { cursor: pointer; }
  tbody tr:hover td { background: rgba(6, 182, 212, 0.06); }
  tbody tr.selected-row td { background: rgba(6, 182, 212, 0.12); }
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
    border: 1px dashed rgba(148, 163, 184, 0.24);
  }

  /* Line Items & Invoice Refs Cells */
  .col-line-items { white-space: normal; min-width: 260px; max-width: 420px; }
  .line-items-list { display: flex; flex-direction: column; gap: 6px; }
  .line-item-card {
    background: rgba(17, 26, 43, 0.82);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 9px;
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  .line-item-top { display: flex; align-items: flex-start; justify-content: space-between; gap: 8px; flex-wrap: wrap; }
  .line-item-desc { font-family: var(--font-sans); font-weight: 600; font-size: 12px; color: #f1f5f9; word-break: break-word; }
  .line-item-date {
    display: inline-block;
    padding: 1px 6px;
    border-radius: 999px;
    font-family: var(--font-mono);
    font-size: 10.5px;
    color: var(--cyan);
    background: var(--cyan-soft);
    border: 1px solid rgba(6, 182, 212, 0.28);
    white-space: nowrap;
  }
  .line-item-math { font-family: var(--font-mono); font-size: 11.5px; color: var(--text-muted); white-space: nowrap; }
  .line-item-math strong { color: var(--emerald); font-weight: 600; }

  .col-refs { white-space: normal; min-width: 180px; max-width: 340px; }
  .ref-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 2px 8px;
    border-radius: 5px;
    font-family: var(--font-mono);
    font-size: 11.5px;
    background: rgba(6, 182, 212, 0.1);
    border: 1px solid rgba(6, 182, 212, 0.28);
    color: var(--text);
    white-space: nowrap;
  }
  .ref-chip-id { color: var(--cyan); font-weight: 600; }
  .ref-chip-amt { color: var(--emerald); }

  /* Inspector split & subtabs */
  .two-col { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }
  .subtabs { display: flex; gap: 6px; flex-wrap: wrap; }
  .subtab {
    padding: 6px 13px;
    border-radius: 7px;
    border: 1px solid var(--border);
    background: #0a101b;
    color: var(--text-muted);
    font-family: var(--font-mono);
    font-size: 12px;
    cursor: pointer;
  }
  .subtab.active {
    color: #031217;
    background: linear-gradient(135deg, #06b6d4, #38bdf8);
    border-color: var(--cyan);
    font-weight: 700;
  }

  /* Calculator */
  .calc-row { display: flex; gap: 8px; margin-bottom: 10px; }
  .calc-row input { flex: 1; }
  .calc-presets { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
  .preset-chip {
    background: #0a101b;
    border: 1px solid var(--border);
    color: var(--text-muted);
    border-radius: 999px;
    padding: 4px 10px;
    font-size: 11px;
    font-family: var(--font-mono);
    cursor: pointer;
  }
  .preset-chip:hover { border-color: var(--violet); color: var(--text-bright); }
  .preset-chip.danger:hover { border-color: var(--rose); color: #fda4af; }
  .calc-output {
    padding: 12px 14px;
    border-radius: 8px;
    background: #060910;
    border: 1px solid var(--border);
    font-family: var(--font-mono);
    font-size: 14px;
    min-height: 48px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
  }

  /* Heatmap cells */
  .hm-cell {
    font-family: var(--font-mono);
    font-size: 11.5px;
    text-align: center;
    border-radius: 6px;
    padding: 5px 8px;
    font-weight: 700;
    white-space: nowrap;
  }
  .hm-sub { display: block; font-size: 10px; font-weight: 500; opacity: 0.85; }

  .empty-state { padding: 30px 16px; text-align: center; color: var(--text-muted); font-size: 13px; }
  .error-banner {
    padding: 10px 14px;
    border-radius: 8px;
    background: var(--rose-soft);
    border: 1px solid rgba(244, 63, 94, 0.4);
    color: #fecdd3;
    font-size: 13px;
    margin-bottom: 14px;
    display: none;
  }
  .spinner {
    display: inline-block;
    width: 13px;
    height: 13px;
    border: 2px solid rgba(255, 255, 255, 0.25);
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
        <div class="brand-title">Audit RL // FinTech Audit Workbench</div>
        <div class="brand-sub">Deterministic Ledger Reconciliation · 100-Point Rubric · Subprocess Eval Isolation</div>
      </div>
    </div>
    <div class="status-pills">
      <span class="pill pill-emerald"><span class="pill-dot"></span>SubprocessAuditEnv · mode=eval</span>
      <span class="pill pill-cyan">Offline Stdlib</span>
      <span class="pill pill-sky" id="cfg-counts-pill">Loading configs…</span>
    </div>
  </header>

  <div class="context-strip">
    <div class="legend-row">
      <span style="color:var(--text-dim); font-weight:700;">ERROR PALETTE:</span>
      <span class="legend-item"><span class="swatch" style="background:var(--amber);"></span>duplicate_invoice</span>
      <span class="legend-item"><span class="swatch" style="background:var(--rose);"></span>payment_allocation_mismatch</span>
      <span class="legend-item"><span class="swatch" style="background:var(--cyan);"></span>fx_conversion_error</span>
      <span class="legend-item"><span class="swatch" style="background:var(--violet);"></span>period_cutoff_error</span>
      <span class="legend-item"><span class="swatch" style="background:var(--emerald);"></span>clean / pass</span>
    </div>
    <div id="active-ep-strip">No episode inspected yet</div>
  </div>

  <nav class="tabs" role="tablist">
    <button class="tab-btn active" data-tab="rubric"><span class="tab-num">01</span> Verification &amp; 100-pt Rubric</button>
    <button class="tab-btn" data-tab="runner"><span class="tab-num">02</span> Episode Runner &amp; Trajectory Replay</button>
    <button class="tab-btn" data-tab="inspector"><span class="tab-num">03</span> Data &amp; Tool Inspector</button>
    <button class="tab-btn" data-tab="benchmark"><span class="tab-num">04</span> 20-Episode Benchmark &amp; Heatmap</button>
  </nav>

  <!-- TAB 1: VERIFICATION & 100-PT RUBRIC -->
  <section id="tab-rubric" class="tab-panel active">
    <div class="card rubric-hero">
      <div class="score-ring-wrap">
        <svg viewBox="0 0 140 140" width="148" height="148">
          <circle cx="70" cy="70" r="58" fill="none" stroke="rgba(148,163,184,0.14)" stroke-width="11"></circle>
          <circle id="rubric-ring-arc" cx="70" cy="70" r="58" fill="none" stroke="#10b981" stroke-width="11"
            stroke-linecap="round" stroke-dasharray="364.42" stroke-dashoffset="364.42"
            transform="rotate(-90 70 70)" style="transition: stroke-dashoffset 0.7s ease, stroke 0.3s ease;"></circle>
        </svg>
        <div class="score-ring-center">
          <div class="score-ring-val" id="rubric-score-val">—</div>
          <div class="score-ring-max" id="rubric-score-max">/ 100 pts</div>
        </div>
      </div>
      <div class="rubric-hero-meta">
        <h2 class="rubric-hero-title">
          Genesis Hackathon 100-Point Rubric Verification
          <span class="pill" id="rubric-pass-pill">Not Run</span>
        </h2>
        <p class="rubric-hero-desc">
          Runs all 5 rubric categories (Contract &amp; Isolation, Data &amp; Ground Truth, Reward Robustness,
          Reproducibility &amp; Generalization, Engineering &amp; Diagnostics), all unit tests, and 20-episode subprocess baselines.
        </p>
        <div style="display:flex; flex-wrap:wrap; gap:10px; align-items:center;">
          <button id="btn-run-verify" class="btn btn-primary">Run Full 100-Point Verification</button>
          <button id="btn-rubric-expand" class="btn btn-sm" style="display:none;">Expand All Checks</button>
          <button id="btn-rubric-collapse" class="btn btn-sm" style="display:none;">Collapse All</button>
          <div class="rubric-stats-row" id="rubric-stats-pills"></div>
        </div>
      </div>
    </div>

    <div id="verify-error" class="error-banner"></div>
    <div id="rubric-categories" class="rubric-grid"></div>
    <div id="rubric-empty" class="card empty-state">
      Click <strong>Run Full 100-Point Verification</strong> to execute <code>audit_rl.rubric.run_full_rubric_verification()</code> and inspect every check across all 5 categories.
    </div>
  </section>

  <!-- TAB 2: EPISODE RUNNER & INTERACTIVE TRAJECTORY REPLAY -->
  <section id="tab-runner" class="tab-panel">
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
          <h2 class="card-title">Submitted Findings (Click Any Evidence ID to Verify)</h2>
          <span class="pill" id="findings-count-pill">0 findings</span>
        </div>
        <div id="findings-container"></div>
        <div id="evidence-drawer" class="evidence-drawer">
          <div class="card-header" style="margin-bottom:10px;">
            <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
              <span class="pill pill-cyan" id="ev-source-pill">lookup</span>
              <strong class="mono" id="ev-id-title" style="color:var(--text-bright);"></strong>
            </div>
            <button class="btn btn-sm" id="btn-close-evidence">Close &times;</button>
          </div>
          <div id="ev-formatted-preview" style="margin-bottom:10px;"></div>
          <pre class="json-box" id="ev-raw-json"></pre>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <h2 class="card-title">Interactive Trajectory Replay</h2>
          <span class="pill pill-sky" id="scrubber-step-pill">Step 0 / 0</span>
        </div>
        <div class="scrubber-bar">
          <button class="btn btn-sm" id="btn-step-prev">&larr; Prev</button>
          <button class="btn btn-sm btn-primary" id="btn-step-play">&#9654; Play</button>
          <button class="btn btn-sm" id="btn-step-next">Next &rarr;</button>
          <input type="range" id="step-slider" class="scrubber-slider" min="0" max="0" value="0" />
          <div class="budget-wrap">
            <span>Budget:</span>
            <div class="budget-track"><div class="budget-fill" id="budget-fill" style="width:100%;"></div></div>
            <strong id="budget-text">100</strong>
          </div>
        </div>
        <div class="timeline-strip" id="timeline-strip"></div>
        <div class="step-stage" id="step-stage"></div>
      </div>
    </div>

    <div id="run-empty" class="card empty-state">
      Select a split, episode, and policy above, then click <strong>Run Episode</strong> to execute inside <code>SubprocessAuditEnv(mode="eval")</code>.
    </div>
  </section>

  <!-- TAB 3: DATA & TOOL INSPECTOR -->
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
        <div class="field" style="min-width:200px;">
          <label for="ins-filter">Filter Current Page</label>
          <input type="search" id="ins-filter" placeholder="Search ID, currency, desc…" />
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
        <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
          <span class="pill" id="ins-page-info">No data loaded</span>
          <button class="btn btn-sm" id="btn-ins-prev" disabled>&larr; Prev</button>
          <button class="btn btn-sm" id="btn-ins-next" disabled>Next &rarr;</button>
        </div>
      </div>
      <div class="table-wrap" id="ins-table-wrap">
        <div class="empty-state">Loading table records via <code>AuditEnv.step({"action": "list_records", ...})</code>…</div>
      </div>
      <div id="row-detail-drawer" class="evidence-drawer">
        <div class="card-header" style="margin-bottom:8px;">
          <div style="display:flex; align-items:center; gap:8px;">
            <span class="pill pill-sky">Selected Row</span>
            <strong class="mono" id="row-detail-title"></strong>
          </div>
          <button class="btn btn-sm" id="btn-close-row-detail">Close &times;</button>
        </div>
        <pre class="json-box" id="row-detail-json"></pre>
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
          <h2 class="card-title">Safe AST Decimal Calculator</h2>
          <span class="pill pill-violet">AST · No eval()</span>
        </div>
        <div class="calc-row">
          <input type="text" id="calc-input" class="mono-input" value="round(123450 * 1.0875, 2)" placeholder="e.g. round(12500 * 1.0825, 2)" />
          <button id="btn-calc-eval" class="btn btn-primary">Evaluate</button>
        </div>
        <div class="calc-presets">
          <button class="preset-chip" data-expr="round(123450 * 1.0875, 2)">round(123450 * 1.0875, 2)</button>
          <button class="preset-chip" data-expr="abs(45200 - 41950)">abs(45200 - 41950)</button>
          <button class="preset-chip" data-expr="round(19999 / 100 * 0.9154, 2)">FX conversion</button>
          <button class="preset-chip danger" data-expr="__import__('os').system('id')">Injection test: __import__('os')</button>
          <button class="preset-chip danger" data-expr="2 ** 100000">Power-DoS test</button>
        </div>
        <div class="calc-output" id="calc-output">
          <span style="color:var(--text-muted);">Press Evaluate or Enter</span>
        </div>
      </div>
    </div>
  </section>

  <!-- TAB 4: 20-EPISODE BENCHMARK & HEATMAP -->
  <section id="tab-benchmark" class="tab-panel">
    <div class="card">
      <div class="card-header" style="margin-bottom:0;">
        <div>
          <h2 class="card-title" style="margin-bottom:4px;">Held-Out 20-Episode Benchmark &amp; Heatmap Matrix</h2>
          <div style="font-size:12.5px; color:var(--text-muted);">
            Evaluates all 4 baseline policies across all 20 <code>eval_configs.json</code> episodes inside isolated <code>SubprocessAuditEnv(mode="eval")</code>.
          </div>
        </div>
        <button id="btn-run-baselines" class="btn btn-primary">Run All 4 Baselines (20 Eval Episodes)</button>
      </div>
    </div>

    <div id="bench-error" class="error-banner"></div>
    <div id="bench-results" style="display:none;">
      <div class="two-col">
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
                  <th class="num">P</th>
                  <th class="num">R</th>
                  <th class="num">F1</th>
                  <th class="num">Return</th>
                  <th class="num">Steps</th>
                  <th class="num">Avg ms</th>
                </tr>
              </thead>
              <tbody id="bench-tbody"></tbody>
            </table>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <h2 class="card-title">F1 vs Mean Return (Grouped SVG Chart)</h2>
            <div class="legend-row">
              <span class="legend-item"><span class="swatch" style="background:var(--cyan);"></span>F1</span>
              <span class="legend-item"><span class="swatch" style="background:var(--emerald);"></span>Return (+)</span>
              <span class="legend-item"><span class="swatch" style="background:var(--rose);"></span>Return (&minus;)</span>
            </div>
          </div>
          <div id="bench-svg-chart"></div>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <h2 class="card-title">20-Episode &times; 4-Policy Heatmap Matrix</h2>
          <span class="pill pill-sky">F1 / Return per held-out episode</span>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Seed</th>
                <th>Category</th>
                <th>Episode Description</th>
                <th class="num">Issues</th>
                <th style="text-align:center;">oracle</th>
                <th style="text-align:center;">simple_reconciliation</th>
                <th style="text-align:center;">random_explorer</th>
                <th style="text-align:center;">flag_everything_attack</th>
              </tr>
            </thead>
            <tbody id="bench-heatmap-tbody"></tbody>
          </table>
        </div>
      </div>
    </div>

    <div id="bench-empty" class="card empty-state">
      Click <strong>Run All 4 Baselines (20 Eval Episodes)</strong> (or run Tab 01 Verification) to populate the grouped chart and 20-episode heatmap matrix.
    </div>
  </section>
</div>

<script>
(function() {
  const state = {
    configs: { train: [], dev: [], eval: [] },
    inspector: { table: "invoices", cursor: "0", nextCursor: null, cursorStack: [], records: [] },
    runner: { split: "dev", index: 7, trajectory: [], currentStep: 0, maxSteps: 100, playTimer: null },
  };

  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

  const ERROR_STYLES = {
    duplicate_invoice: { color: "#f59e0b", pill: "pill-amber" },
    payment_allocation_mismatch: { color: "#f43f5e", pill: "pill-rose" },
    fx_conversion_error: { color: "#06b6d4", pill: "pill-cyan" },
    period_cutoff_error: { color: "#a855f7", pill: "pill-violet" },
  };

  const CATEGORY_ACCENTS = ["#06b6d4", "#10b981", "#f59e0b", "#a855f7", "#38bdf8"];

  async function api(url, opts = {}) {
    const res = await fetch(url, {
      headers: { "Content-Type": "application/json" },
      ...opts,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
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
      return `<option value="${ep.index}">#${ep.index} · seed ${ep.seed} · [${esc(ep.category)}] ${esc(ep.description)} (${esc(ep.base_currency)})</option>`;
    }).join("");
    if (split === "dev" && list.length > 7) sel.value = "7";
  }

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
      showError("verify-error", "Failed to load configs: " + err.message);
    }
  }

  $("run-split").addEventListener("change", (e) => populateEpisodes("run-episode", e.target.value));
  $("ins-split").addEventListener("change", (e) => {
    populateEpisodes("ins-episode", e.target.value);
    loadInspector(true);
  });
  $("ins-episode").addEventListener("change", () => loadInspector(true));
  $("ins-limit").addEventListener("change", () => loadInspector(true));

  // =========================================================================
  // TAB 1: VERIFICATION & 100-PT RUBRIC
  // =========================================================================
  $("btn-run-verify").addEventListener("click", async () => {
    const btn = $("btn-run-verify");
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> Running 100-Point Rubric Suite…`;
    showError("verify-error", "");

    try {
      const data = await api("/api/verify", {
        method: "POST",
        body: JSON.stringify({ run_unittests: true }),
      });
      renderRubricVerification(data);
      // Also populate Tab 4 Benchmark & Heatmap from cached verify results!
      if (data.baselines || data.episode_matrix) {
        renderBenchmarkFromAny(data);
      }
    } catch (err) {
      showError("verify-error", err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = "Run Full 100-Point Verification";
    }
  });

  function renderRubricVerification(data) {
    $("rubric-empty").style.display = "none";
    $("btn-rubric-expand").style.display = "inline-flex";
    $("btn-rubric-collapse").style.display = "inline-flex";

    const score = Number(data.total_score ?? 0);
    const maxScore = Number(data.max_score ?? 100);
    const pct = maxScore > 0 ? Math.max(0, Math.min(1, score / maxScore)) : 0;
    const circ = 364.42;
    const arc = $("rubric-ring-arc");
    arc.style.strokeDashoffset = String(circ * (1 - pct));
    arc.style.stroke = data.overall_pass ? "#10b981" : (pct >= 0.7 ? "#f59e0b" : "#f43f5e");

    $("rubric-score-val").textContent = score.toFixed(1);
    $("rubric-score-max").textContent = `/ ${maxScore.toFixed(0)} pts`;

    const passPill = $("rubric-pass-pill");
    passPill.className = "pill " + (data.overall_pass ? "pill-emerald" : "pill-rose");
    passPill.textContent = data.overall_pass ? "OVERALL PASS" : "FAIL";

    const ut = data.unittest || {};
    const totalChecks = (data.categories || []).reduce((acc, c) => acc + Number(c.total || 0), 0);
    const passedChecks = (data.categories || []).reduce((acc, c) => acc + Number(c.passed || 0), 0);

    $("rubric-stats-pills").innerHTML = `
      <span class="pill pill-emerald">Checks: ${passedChecks}/${totalChecks}</span>
      ${ut.ran !== undefined ? `<span class="pill ${ut.passed ? "pill-cyan" : "pill-rose"}">Unittests: ${ut.ran} run, ${ut.failures || 0} fail</span>` : ""}
      <span class="pill pill-sky">Duration: ${Number(data.duration_s || 0).toFixed(2)}s</span>
    `;

    $("rubric-categories").innerHTML = (data.categories || []).map((cat, idx) => {
      const accent = CATEGORY_ACCENTS[idx % CATEGORY_ACCENTS.length];
      const cPct = cat.max_score > 0 ? Math.min(100, (cat.score / cat.max_score) * 100) : 0;
      const checksHtml = (cat.checks || []).map((ch) => `
        <div class="check-row">
          <span class="check-badge ${ch.passed ? "check-pass" : "check-fail"}">${ch.passed ? "PASS" : "FAIL"}</span>
          <div>
            <div class="check-name">${esc(ch.name)}</div>
            ${ch.detail ? `<div class="check-detail">${esc(ch.detail)}</div>` : ""}
          </div>
        </div>
      `).join("");

      return `
        <div class="cat-card" style="border-top-color:${accent};">
          <div class="cat-top">
            <div>
              <div class="cat-name">${esc(cat.name)}</div>
              <div class="cat-sub">${cat.passed}/${cat.total} checks passed</div>
            </div>
            <div class="cat-score" style="color:${cat.passed === cat.total ? "#10b981" : accent};">
              ${Number(cat.score).toFixed(1)}<span style="font-size:12px; color:var(--text-dim);">/${Number(cat.max_score).toFixed(0)}</span>
            </div>
          </div>
          <div class="prog-track">
            <div class="prog-fill" style="width:${cPct}%; background:${accent};"></div>
          </div>
          <details class="cat-checks">
            <summary>Inspect ${cat.checks ? cat.checks.length : 0} individual checks</summary>
            <div class="check-list">${checksHtml}</div>
          </details>
        </div>
      `;
    }).join("");
  }

  $("btn-rubric-expand").addEventListener("click", () => {
    document.querySelectorAll("#rubric-categories details").forEach((d) => { d.open = true; });
  });
  $("btn-rubric-collapse").addEventListener("click", () => {
    document.querySelectorAll("#rubric-categories details").forEach((d) => { d.open = false; });
  });

  // =========================================================================
  // TAB 2: EPISODE RUNNER & INTERACTIVE TRAJECTORY REPLAY
  // =========================================================================
  function stopReplay() {
    if (state.runner.playTimer) {
      clearInterval(state.runner.playTimer);
      state.runner.playTimer = null;
    }
    $("btn-step-play").innerHTML = "&#9654; Play";
  }

  function actionPillClass(act) {
    if (act === "submit") return "pill-emerald";
    if (act === "calculator") return "pill-violet";
    if (act === "get_document" || act === "get_record") return "pill-amber";
    return "pill-cyan";
  }

  $("btn-run-episode").addEventListener("click", async () => {
    stopReplay();
    const btn = $("btn-run-episode");
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> Running…`;
    showError("run-error", "");
    $("evidence-drawer").classList.remove("open");

    try {
      const split = $("run-split").value;
      const index = parseInt($("run-episode").value, 10);
      const data = await api("/api/episode/run", {
        method: "POST",
        body: JSON.stringify({ split, index, policy: $("run-policy").value }),
      });
      state.runner.split = split;
      state.runner.index = index;
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

    $("active-ep-strip").innerHTML =
      `Active: <strong style="color:var(--text-bright);">${esc(m.episode_id)}</strong> · ${esc(m.company_name || "")} · ${esc(m.base_currency)}`;

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
      <div class="kpi" style="border-left-color:${r.f1 >= 0.85 ? "var(--emerald)" : (r.f1 >= 0.4 ? "var(--amber)" : "var(--rose)")};">
        <div class="kpi-label">F1 Score</div>
        <div class="kpi-val ${f1Class}">${Number(r.f1).toFixed(4)}</div>
        <div class="kpi-sub">${r.valid_submission ? "valid submission" : "invalid submission"}</div>
      </div>
      <div class="kpi" style="border-left-color:${r.reward > 0 ? "var(--emerald)" : "var(--rose)"};">
        <div class="kpi-label">Reward</div>
        <div class="kpi-val ${rewClass}">${Number(r.reward).toFixed(4)}</div>
        <div class="kpi-sub">policy: ${esc(data.policy_name)}</div>
      </div>
      <div class="kpi" style="border-left-color:var(--cyan);">
        <div class="kpi-label">Precision</div>
        <div class="kpi-val">${Number(r.precision).toFixed(4)}</div>
        <div class="kpi-sub">submitted: ${r.submitted_count}</div>
      </div>
      <div class="kpi" style="border-left-color:var(--violet);">
        <div class="kpi-label">Recall</div>
        <div class="kpi-val">${Number(r.recall).toFixed(4)}</div>
        <div class="kpi-sub">truth issues: ${r.truth_count}</div>
      </div>
      <div class="kpi" style="border-left-color:var(--sky);">
        <div class="kpi-label">Steps</div>
        <div class="kpi-val val-cyan">${r.steps}</div>
        <div class="kpi-sub">inspection: ${r.inspection_steps}</div>
      </div>
      <div class="kpi" style="border-left-color:var(--amber);">
        <div class="kpi-label">TP / FP / FN</div>
        <div class="kpi-val"><span class="val-good">${r.tp}</span> / <span class="${r.fp > 0 ? "val-warn" : ""}">${r.fp}</span> / <span class="${r.fn > 0 ? "val-bad" : ""}">${r.fn}</span></div>
        <div class="kpi-sub">evaluator mode=eval</div>
      </div>
    `;

    // Findings Cards
    const findings = data.submitted_findings || [];
    $("findings-count-pill").textContent = `${findings.length} finding${findings.length === 1 ? "" : "s"}`;
    if (findings.length === 0) {
      $("findings-container").innerHTML = `<div class="empty-state">No findings submitted (clean books or empty submission).</div>`;
    } else {
      $("findings-container").innerHTML = `<div class="findings-grid">` + findings.map((f, i) => {
        const style = ERROR_STYLES[f.type] || { color: "#38bdf8", pill: "pill-sky" };
        const affectedHtml = (f.affected_ids || []).map((id) =>
          `<button class="evidence-chip" data-lookup-id="${esc(id)}" title="Inspect ${esc(id)}">${esc(id)}</button>`
        ).join("");
        const evHtml = (f.evidence_ids || []).map((id) =>
          `<button class="evidence-chip" data-lookup-id="${esc(id)}" title="Inspect evidence ${esc(id)}">&#128269; ${esc(id)}</button>`
        ).join("");
        return `
          <div class="finding-card" style="border-left-color:${style.color};">
            <div class="finding-top">
              <div style="display:flex; align-items:center; gap:8px;">
                <span class="mono" style="color:var(--text-dim); font-size:11px;">#${i + 1}</span>
                <span class="pill ${style.pill}">${esc(f.type)}</span>
              </div>
              <span class="finding-amount">${esc(f.claimed_amount_minor)} ${esc(f.currency)}</span>
            </div>
            <div class="finding-expl">${esc(f.explanation || "")}</div>
            <div>
              <div class="finding-row-label">Affected IDs</div>
              <div class="tag-list">${affectedHtml}</div>
            </div>
            <div>
              <div class="finding-row-label">Evidence IDs (Click to Inspect)</div>
              <div class="tag-list">${evHtml}</div>
            </div>
          </div>
        `;
      }).join("") + `</div>`;

      document.querySelectorAll("#findings-container .evidence-chip").forEach((chip) => {
        chip.addEventListener("click", () => lookupEvidence(chip.dataset.lookupId));
      });
    }

    // Trajectory Scrubber
    const traj = data.trajectory || [];
    state.runner.trajectory = traj;
    state.runner.maxSteps = Number(r.max_steps || 100);
    state.runner.currentStep = 0;

    const slider = $("step-slider");
    slider.min = "0";
    slider.max = String(Math.max(0, traj.length - 1));
    slider.value = "0";

    $("timeline-strip").innerHTML = traj.map((t, idx) =>
      `<button class="step-dot-btn act-${esc(t.action_type)}" data-step-idx="${idx}" title="Step ${t.step}: ${esc(t.action_type)}">${String(t.step).padStart(2, "0")}·${esc(t.action_type.replace("_records", "").replace("_document", "").replace("_record", ""))}</button>`
    ).join("");
    document.querySelectorAll("#timeline-strip .step-dot-btn").forEach((b) => {
      b.addEventListener("click", () => {
        stopReplay();
        renderTrajectoryStep(parseInt(b.dataset.stepIdx, 10));
      });
    });

    renderTrajectoryStep(0);
  }

  function renderTrajectoryStep(idx) {
    const traj = state.runner.trajectory;
    if (!traj.length) {
      $("step-stage").innerHTML = `<div class="empty-state">No trajectory steps recorded.</div>`;
      return;
    }
    const clamped = Math.max(0, Math.min(traj.length - 1, idx));
    state.runner.currentStep = clamped;
    $("step-slider").value = String(clamped);

    const t = traj[clamped];
    $("scrubber-step-pill").textContent = `Step ${t.step} / ${traj.length}`;
    $("btn-step-prev").disabled = clamped === 0;
    $("btn-step-next").disabled = clamped === traj.length - 1;

    const budget = Number(t.budget_remaining ?? Math.max(0, state.runner.maxSteps - t.step));
    const budgetPct = Math.max(0, Math.min(100, (budget / state.runner.maxSteps) * 100));
    $("budget-fill").style.width = `${budgetPct}%`;
    $("budget-text").textContent = `${budget}/${state.runner.maxSteps}`;

    document.querySelectorAll("#timeline-strip .step-dot-btn").forEach((b, i) => {
      b.classList.toggle("active", i === clamped);
    });

    $("step-stage").innerHTML = `
      <div class="step-stage-head">
        <span class="mono" style="color:var(--text-dim); font-weight:700;">STEP ${String(t.step).padStart(2, "0")}</span>
        <span class="pill ${actionPillClass(t.action_type)}">${esc(t.action_type)}</span>
        <span class="mono" style="font-size:12.5px; color:var(--text-bright); flex:1;">${esc(t.observation_summary)}</span>
        ${t.terminated ? `<span class="pill pill-emerald">terminated</span>` : ""}
        ${t.truncated ? `<span class="pill pill-rose">truncated</span>` : ""}
      </div>
      <div class="step-stage-body">
        <div>
          <label style="display:block; margin-bottom:6px;">Action Payload</label>
          <pre class="json-box">${esc(JSON.stringify(t.action, null, 2))}</pre>
        </div>
        <div>
          <label style="display:block; margin-bottom:6px;">Observation Payload</label>
          <pre class="json-box">${esc(JSON.stringify(t.observation, null, 2))}</pre>
        </div>
      </div>
    `;
  }

  $("btn-step-prev").addEventListener("click", () => { stopReplay(); renderTrajectoryStep(state.runner.currentStep - 1); });
  $("btn-step-next").addEventListener("click", () => { stopReplay(); renderTrajectoryStep(state.runner.currentStep + 1); });
  $("step-slider").addEventListener("input", (e) => { stopReplay(); renderTrajectoryStep(parseInt(e.target.value, 10)); });
  $("btn-step-play").addEventListener("click", () => {
    if (state.runner.playTimer) {
      stopReplay();
      return;
    }
    if (state.runner.currentStep >= state.runner.trajectory.length - 1) {
      renderTrajectoryStep(0);
    }
    $("btn-step-play").innerHTML = "&#10074;&#10074; Pause";
    state.runner.playTimer = setInterval(() => {
      if (state.runner.currentStep >= state.runner.trajectory.length - 1) {
        stopReplay();
      } else {
        renderTrajectoryStep(state.runner.currentStep + 1);
      }
    }, 650);
  });

  async function lookupEvidence(id) {
    const drawer = $("evidence-drawer");
    drawer.classList.add("open");
    $("ev-id-title").textContent = id;
    $("ev-source-pill").textContent = "loading…";
    $("ev-formatted-preview").innerHTML = "";
    $("ev-raw-json").textContent = "Querying AuditEnv…";

    try {
      const data = await api("/api/record/lookup", {
        method: "POST",
        body: JSON.stringify({ split: state.runner.split, index: state.runner.index, id }),
      });
      if (!data.ok) {
        $("ev-source-pill").className = "pill pill-rose";
        $("ev-source-pill").textContent = "not found";
        $("ev-raw-json").textContent = data.error || "Record not found";
        return;
      }
      $("ev-source-pill").className = "pill pill-emerald";
      $("ev-source-pill").textContent = `${data.action} · ${data.source}`;
      const rec = data.data || {};
      let preview = "";
      if (Array.isArray(rec.line_items)) {
        preview += `<div style="margin-bottom:8px;"><label style="display:block; margin-bottom:4px;">Line Items</label><table><tbody><tr>${renderLineItemsCell(rec.line_items)}</tr></tbody></table></div>`;
      }
      if (Array.isArray(rec.invoice_refs)) {
        preview += `<div style="margin-bottom:8px;"><label style="display:block; margin-bottom:4px;">Invoice Refs</label><table><tbody><tr>${renderInvoiceRefsCell(rec.invoice_refs)}</tr></tbody></table></div>`;
      }
      $("ev-formatted-preview").innerHTML = preview;
      $("ev-raw-json").textContent = JSON.stringify(rec, null, 2);
    } catch (err) {
      $("ev-source-pill").className = "pill pill-rose";
      $("ev-source-pill").textContent = "error";
      $("ev-raw-json").textContent = err.message;
    }
  }
  $("btn-close-evidence").addEventListener("click", () => $("evidence-drawer").classList.remove("open"));

  // =========================================================================
  // TAB 3: DATA & TOOL INSPECTOR
  // =========================================================================
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
  $("ins-filter").addEventListener("input", () => renderFilteredInspectorTable());
  $("btn-close-row-detail").addEventListener("click", () => $("row-detail-drawer").classList.remove("open"));

  async function loadInspector(resetCursor) {
    if (resetCursor) {
      state.inspector.cursor = "0";
      state.inspector.cursorStack = [];
    }
    showError("ins-error", "");
    $("row-detail-drawer").classList.remove("open");
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
      state.inspector.records = data.records || [];
      $("ins-ep-id-pill").textContent = data.episode_id || "—";
      $("ins-manifest-json").textContent = JSON.stringify(data.manifest, null, 2);
      $("ins-policy-json").textContent = JSON.stringify(data.policy, null, 2);

      const m = data.manifest || {};
      $("active-ep-strip").innerHTML =
        `Active: <strong style="color:var(--text-bright);">${esc(m.episode_id)}</strong> · ${esc(m.company_name || "")} · ${esc(m.base_currency)} · ${esc(data.table)} (${data.total_count} rows)`;

      const start = parseInt(data.cursor, 10) + (data.count > 0 ? 1 : 0);
      const end = parseInt(data.cursor, 10) + data.count;
      $("ins-page-info").textContent = `${data.table}: ${start}–${end} of ${data.total_count}`;
      $("btn-ins-prev").disabled = state.inspector.cursorStack.length === 0;
      $("btn-ins-next").disabled = data.next_cursor === null;

      renderFilteredInspectorTable();
    } catch (err) {
      showError("ins-error", err.message);
    }
  }

  function renderFilteredInspectorTable() {
    const q = ($("ins-filter").value || "").trim().toLowerCase();
    const all = state.inspector.records || [];
    const filtered = q ? all.filter((r) => JSON.stringify(r).toLowerCase().includes(q)) : all;
    renderInspectorTable(filtered);
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
      if (ref === null || ref === undefined) return `<span class="null-badge">null</span>`;
      if (typeof ref === "string") return `<span class="ref-chip"><span class="ref-chip-id">${esc(ref)}</span></span>`;
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
      wrap.innerHTML = `<div class="empty-state">No matching records in this page.</div>`;
      return;
    }
    const cols = [];
    records.forEach((r) => {
      Object.keys(r).forEach((k) => { if (!cols.includes(k)) cols.push(k); });
    });
    const thead = `<tr>${cols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>`;
    const tbody = records.map((r, rIdx) => {
      return `<tr class="clickable-row" data-row-idx="${rIdx}">${cols.map((c) => {
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

    wrap.querySelectorAll("tbody tr.clickable-row").forEach((tr) => {
      tr.addEventListener("click", () => {
        wrap.querySelectorAll("tbody tr").forEach((row) => row.classList.remove("selected-row"));
        tr.classList.add("selected-row");
        const rec = records[parseInt(tr.dataset.rowIdx, 10)];
        const label = rec.invoice_id || rec.payment_id || rec.row_id || rec.doc_id || `${rec.date || ""}:${rec.source_currency || ""}:${rec.target_currency || ""}`;
        $("row-detail-title").textContent = label;
        $("row-detail-json").textContent = JSON.stringify(rec, null, 2);
        $("row-detail-drawer").classList.add("open");
      });
    });
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
        $("calc-output").innerHTML = `<span class="val-bad">Rejected: ${esc(data.error)}</span>`;
      }
    } catch (err) {
      $("calc-output").innerHTML = `<span class="val-bad">Error: ${esc(err.message)}</span>`;
    }
  }
  $("btn-calc-eval").addEventListener("click", evalCalc);
  $("calc-input").addEventListener("keydown", (e) => { if (e.key === "Enter") evalCalc(); });
  document.querySelectorAll(".preset-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      $("calc-input").value = chip.dataset.expr;
      evalCalc();
    });
  });

  // =========================================================================
  // TAB 4: 20-EPISODE BENCHMARK & HEATMAP
  // =========================================================================
  $("btn-run-baselines").addEventListener("click", async () => {
    const btn = $("btn-run-baselines");
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> Running 20 Eval Episodes…`;
    showError("bench-error", "");

    try {
      const data = await api("/api/baselines", { method: "POST", body: "{}" });
      renderBenchmarkFromAny(data);
    } catch (err) {
      showError("bench-error", err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = "Run All 4 Baselines (20 Eval Episodes)";
    }
  });

  function normalizeSummary(data) {
    if (data.summary && typeof data.summary === "object") return data.summary;
    const out = {};
    if (Array.isArray(data.baselines)) {
      data.baselines.forEach((b) => {
        const name = b.policy || b.name;
        if (name) {
          out[name] = {
            precision: Number(b.precision ?? b.p ?? 0),
            recall: Number(b.recall ?? b.r ?? 0),
            f1: Number(b.f1 ?? 0),
            return: Number(b.return ?? b.mean_return ?? b.reward ?? 0),
            steps: Number(b.steps ?? b.avg_steps ?? 0),
            runtime_ms: Number(b.runtime_ms ?? b.avg_ms ?? 0),
          };
        }
      });
    } else if (data.baselines && typeof data.baselines === "object") {
      return data.baselines;
    }
    return out;
  }

  function renderBenchmarkFromAny(data) {
    const summary = normalizeSummary(data);
    const names = Object.keys(summary);
    if (!names.length) return;

    $("bench-empty").style.display = "none";
    $("bench-results").style.display = "block";
    const matrix = data.episode_matrix || [];
    $("bench-ep-pill").textContent = `${data.num_episodes || matrix.length || 20} eval episodes`;

    const policyColors = {
      oracle: "#a855f7",
      simple_reconciliation: "#10b981",
      random_explorer: "#06b6d4",
      flag_everything_attack: "#f43f5e",
    };

    $("bench-tbody").innerHTML = names.map((name) => {
      const s = summary[name];
      const retClass = s.return > 0 ? "val-good" : (s.return < 0 ? "val-bad" : "");
      return `
        <tr>
          <td class="mono" style="font-weight:700; color:${policyColors[name] || "#e2e8f0"};">${esc(name)}</td>
          <td class="num">${Number(s.precision).toFixed(4)}</td>
          <td class="num">${Number(s.recall).toFixed(4)}</td>
          <td class="num" style="font-weight:700;">${Number(s.f1).toFixed(4)}</td>
          <td class="num ${retClass}" style="font-weight:700;">${Number(s.return).toFixed(4)}</td>
          <td class="num">${Number(s.steps).toFixed(2)}</td>
          <td class="num">${Number(s.runtime_ms).toFixed(2)}</td>
        </tr>
      `;
    }).join("");

    renderGroupedSvgChart(names, summary);
    renderHeatmapMatrix(matrix);
  }

  function renderGroupedSvgChart(names, summary) {
    const W = 540, H = 230, padL = 44, padR = 16, padT = 18, padB = 46;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;
    const yOf = (val) => padT + ((1.0 - Math.max(-1.0, Math.min(1.0, val))) / 2.0) * plotH;
    const zeroY = yOf(0);
    const groupW = plotW / Math.max(1, names.length);
    const barW = Math.min(34, groupW * 0.28);

    const gridLines = [1.0, 0.5, 0.0, -0.5, -1.0].map((v) => {
      const y = yOf(v).toFixed(1);
      const stroke = v === 0 ? "rgba(148,163,184,0.45)" : "rgba(148,163,184,0.12)";
      return `
        <line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="${stroke}" stroke-width="${v === 0 ? 1.5 : 1}" />
        <text x="${padL - 8}" y="${Number(y) + 4}" text-anchor="end" fill="#64748b" font-size="10" font-family="monospace">${v.toFixed(1)}</text>
      `;
    }).join("");

    const bars = names.map((name, i) => {
      const s = summary[name];
      const f1 = Number(s.f1 || 0);
      const ret = Number(s.return || 0);
      const cx = padL + groupW * i + groupW / 2;

      const f1Top = yOf(f1);
      const f1H = Math.max(1, zeroY - f1Top);

      const retTop = ret >= 0 ? yOf(ret) : zeroY;
      const retH = Math.max(1, Math.abs(yOf(ret) - zeroY));
      const retColor = ret >= 0 ? "#10b981" : "#f43f5e";

      const shortName = name.replace("_reconciliation", "_recon").replace("_everything_attack", "_attack").replace("_explorer", "_expl");

      return `
        <rect x="${(cx - barW - 3).toFixed(1)}" y="${f1Top.toFixed(1)}" width="${barW.toFixed(1)}" height="${f1H.toFixed(1)}" rx="4" fill="#06b6d4"></rect>
        <text x="${(cx - barW / 2 - 3).toFixed(1)}" y="${(f1Top - 4).toFixed(1)}" text-anchor="middle" fill="#67e8f9" font-size="9.5" font-family="monospace">${f1.toFixed(2)}</text>

        <rect x="${(cx + 3).toFixed(1)}" y="${retTop.toFixed(1)}" width="${barW.toFixed(1)}" height="${retH.toFixed(1)}" rx="4" fill="${retColor}"></rect>
        <text x="${(cx + barW / 2 + 3).toFixed(1)}" y="${(ret >= 0 ? retTop - 4 : retTop + retH + 11).toFixed(1)}" text-anchor="middle" fill="${ret >= 0 ? "#6ee7b7" : "#fda4af"}" font-size="9.5" font-family="monospace">${ret.toFixed(2)}</text>

        <text x="${cx.toFixed(1)}" y="${H - 14}" text-anchor="middle" fill="#cbd5e1" font-size="10.5" font-family="monospace">${esc(shortName)}</text>
      `;
    }).join("");

    $("bench-svg-chart").innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" width="100%" style="max-height:240px; display:block;">
        ${gridLines}
        ${bars}
      </svg>
    `;
  }

  function heatmapCellHtml(res) {
    if (!res) return `<td style="text-align:center;"><span class="null-badge">—</span></td>`;
    const f1 = Number(res.f1 ?? 0);
    const rew = Number(res.reward ?? res.return ?? 0);
    const steps = res.steps ?? "?";
    let bg = "rgba(244,63,94,0.22)", fg = "#fda4af", bd = "rgba(244,63,94,0.35)";
    if (f1 >= 0.99 && rew >= 0.8) {
      bg = "rgba(16,185,129,0.24)"; fg = "#6ee7b7"; bd = "rgba(16,185,129,0.42)";
    } else if (f1 >= 0.5) {
      bg = "rgba(245,158,11,0.22)"; fg = "#fcd34d"; bd = "rgba(245,158,11,0.4)";
    } else if (rew >= 0) {
      bg = "rgba(56,189,248,0.16)"; fg = "#7dd3fc"; bd = "rgba(56,189,248,0.3)";
    }
    return `
      <td style="text-align:center;" title="F1=${f1.toFixed(4)} | Reward=${rew.toFixed(4)} | Steps=${steps}">
        <div class="hm-cell" style="background:${bg}; color:${fg}; border:1px solid ${bd};">
          F1 ${f1.toFixed(2)}
          <span class="hm-sub">R ${rew >= 0 ? "+" : ""}${rew.toFixed(2)} · ${steps}s</span>
        </div>
      </td>
    `;
  }

  function catPillClass(cat) {
    if (cat === "clean") return "pill-emerald";
    if (cat === "single") return "pill-cyan";
    if (cat === "multi") return "pill-amber";
    return "pill-violet";
  }

  function renderHeatmapMatrix(matrix) {
    const tbody = $("bench-heatmap-tbody");
    if (!matrix || !matrix.length) {
      tbody.innerHTML = `<tr><td colspan="8" class="empty-state">No per-episode matrix data available.</td></tr>`;
      return;
    }
    const cols = ["oracle", "simple_reconciliation", "random_explorer", "flag_everything_attack"];
    tbody.innerHTML = matrix.map((ep) => {
      const results = ep.results || {};
      return `
        <tr>
          <td class="mono col-nowrap" style="font-weight:700;">${esc(ep.seed)}</td>
          <td class="col-nowrap"><span class="pill ${catPillClass(ep.category)}">${esc(ep.category)}</span></td>
          <td style="color:var(--text-muted); max-width:340px;">${esc(ep.label)}</td>
          <td class="num">${esc(ep.num_issues)}</td>
          ${cols.map((c) => heatmapCellHtml(results[c])).join("")}
        </tr>
      `;
    }).join("");
  }

  init();
})();
</script>
</body>
</html>
"""
