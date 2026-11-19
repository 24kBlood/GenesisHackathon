"""Gymnasium-style financial audit RL environment and subprocess-isolated wrapper."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from audit_rl.calculator import CalculatorError, safe_calculate
from audit_rl.generator import generate_episode
from audit_rl.matcher import evaluate_submission
from audit_rl.schemas import (
    DEFAULT_MAX_STEPS,
    MAX_PAGE_LIMIT,
    EpisodeConfig,
    GeneratedEpisode,
)

_TABLE_ALIASES: dict[str, str] = {
    "ledger": "ledger",
    "invoices": "invoices",
    "payments": "payments",
    "fx": "fx",
    "fx_rates": "fx",
}

_CANONICAL_TABLES: tuple[str, ...] = ("ledger", "invoices", "payments", "fx")

_TASK_PROMPT = (
    "Audit the financial records for the reporting period specified in manifest and policy. "
    "Inspect tables and source documents using the available actions, verify arithmetic "
    "with the calculator tool, and submit all confirmed material findings via the submit action."
)


def _fx_record_key(row: dict[str, Any]) -> str:
    return f"{row.get('date', '')}:{row.get('source_currency', '')}:{row.get('target_currency', '')}"


def _available_actions_spec() -> list[dict[str, Any]]:
    return [
        {
            "action": "list_records",
            "params": ["table", "cursor", "limit"],
            "tables": list(_CANONICAL_TABLES),
            "max_limit": MAX_PAGE_LIMIT,
        },
        {
            "action": "get_record",
            "params": ["table", "id"],
            "tables": list(_CANONICAL_TABLES),
        },
        {
            "action": "get_document",
            "params": ["doc_id"],
        },
        {
            "action": "calculator",
            "params": ["expression"],
        },
        {
            "action": "submit",
            "params": ["findings"],
        },
    ]


class AuditEnv:
    """Deterministic, JSON-native RL environment for financial audit agents."""

    def __init__(
        self,
        mode: str = "train",
        max_steps: int = DEFAULT_MAX_STEPS,
        timeout_seconds: float = 60.0,
    ) -> None:
        if mode not in ("train", "eval"):
            raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError(f"max_steps must be a positive integer, got {max_steps!r}")
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds!r}")

        self.mode = mode
        self.max_steps = max_steps
        self.timeout_seconds = float(timeout_seconds)

        # Private episode state (never exposed in observations or info)
        self._initialized: bool = False
        self._episode_id: str = ""
        self._manifest: dict[str, Any] = {}
        self._policy: dict[str, Any] = {}
        self._tables: dict[str, list[dict[str, Any]]] = {}
        self._table_indices: dict[str, dict[str, dict[str, Any]]] = {}
        self._documents: dict[str, dict[str, Any]] = {}
        self._truth: dict[str, Any] = {}
        self._provenance: dict[str, Any] = {}

        # Episode step & lifecycle state
        self._step_count: int = 0
        self._inspection_steps: int = 0
        self._terminated: bool = False
        self._truncated: bool = False
        self._start_time: float = 0.0
        self._end_time: float | None = None
        self._evaluator_report: dict[str, Any] | None = None

    def reset(
        self,
        seed: int = 0,
        config: EpisodeConfig | dict[str, Any] | None = None,
        episode: GeneratedEpisode | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the environment to a fresh episode, clearing all previous state."""
        if episode is None:
            ep = generate_episode(seed=seed, config=config)
        else:
            ep = episode

        # Deep-copy all open and private records to guarantee full isolation
        self._manifest = copy.deepcopy(ep.manifest)
        self._policy = copy.deepcopy(ep.policy)
        self._episode_id = str(self._manifest.get("episode_id", ""))

        ledger_rows = copy.deepcopy(ep.ledger)
        invoice_rows = copy.deepcopy(ep.invoices)
        payment_rows = copy.deepcopy(ep.payments)
        fx_rows = copy.deepcopy(ep.fx_rates)

        self._tables = {
            "ledger": ledger_rows,
            "invoices": invoice_rows,
            "payments": payment_rows,
            "fx": fx_rows,
        }
        self._table_indices = {
            "ledger": {str(r["row_id"]): r for r in ledger_rows if "row_id" in r},
            "invoices": {str(r["invoice_id"]): r for r in invoice_rows if "invoice_id" in r},
            "payments": {str(r["payment_id"]): r for r in payment_rows if "payment_id" in r},
            "fx": {_fx_record_key(r): r for r in fx_rows},
        }

        documents: dict[str, dict[str, Any]] = {}
        for inv in invoice_rows:
            doc_id = inv.get("doc_id")
            if isinstance(doc_id, str) and doc_id:
                doc_copy = copy.deepcopy(inv)
                doc_copy.setdefault("doc_type", "invoice")
                documents[doc_id] = doc_copy
        for pay in payment_rows:
            doc_id = pay.get("doc_id")
            if isinstance(doc_id, str) and doc_id:
                doc_copy = copy.deepcopy(pay)
                doc_copy.setdefault("doc_type", "payment")
                documents[doc_id] = doc_copy
        self._documents = documents

        self._truth = copy.deepcopy(ep.truth)
        self._provenance = copy.deepcopy(ep.provenance)

        self._step_count = 0
        self._inspection_steps = 0
        self._terminated = False
        self._truncated = False
        self._start_time = time.monotonic()
        self._end_time = None
        self._evaluator_report = None
        self._initialized = True

        obs = {
            "task": _TASK_PROMPT,
            "manifest": copy.deepcopy(self._manifest),
            "policy": copy.deepcopy(self._policy),
            "budget_remaining": self.max_steps,
            "step": 0,
            "available_actions": _available_actions_spec(),
        }
        info = {
            "episode_id": self._episode_id,
            "mode": self.mode,
            "step": 0,
            "max_steps": self.max_steps,
        }
        return obs, info

    def _budget_remaining(self) -> int:
        return max(0, self.max_steps - self._step_count)

    def _error_obs(self, message: str, action_name: str | None = None) -> dict[str, Any]:
        obs: dict[str, Any] = {
            "status": "error",
            "error": message,
            "step": self._step_count,
            "budget_remaining": self._budget_remaining(),
        }
        if action_name is not None:
            obs["action"] = action_name
        return obs

    def _finalize_episode(
        self,
        payload: dict[str, Any] | None,
        terminated: bool,
        truncated: bool,
    ) -> float:
        self._terminated = terminated
        self._truncated = truncated
        self._end_time = time.monotonic()
        runtime_ms = (self._end_time - self._start_time) * 1000.0

        eval_result = evaluate_submission(
            payload=payload,
            truth=self._truth,
            inspection_steps=self._inspection_steps,
            expected_episode_id=self._episode_id,
        )

        metrics = {
            "tp": eval_result["tp"],
            "fp": eval_result["fp"],
            "fn": eval_result["fn"],
            "precision": eval_result["precision"],
            "recall": eval_result["recall"],
            "f1": eval_result["f1"],
            "matched_pairs": eval_result["matched_pairs"],
            "unmatched_finding_indices": eval_result["unmatched_finding_indices"],
            "missed_issue_ids": eval_result["missed_issue_ids"],
        }

        self._evaluator_report = {
            "episode_id": self._episode_id,
            "mode": self.mode,
            "terminated": terminated,
            "truncated": truncated,
            "steps": self._step_count,
            "inspection_steps": self._inspection_steps,
            "runtime_ms": round(runtime_ms, 3),
            "reward": eval_result["reward"],
            "valid_submission": eval_result["valid_submission"],
            "validation_errors": eval_result["validation_errors"],
            "submitted_count": eval_result["submitted_count"],
            "truth_count": eval_result["truth_count"],
            "metrics": metrics,
            **metrics,
        }

        if self.mode == "eval":
            return 0.0
        return float(eval_result["reward"])

    def _handle_list_records(self, action: dict[str, Any]) -> dict[str, Any]:
        raw_table = action.get("table")
        if not isinstance(raw_table, str) or raw_table not in _TABLE_ALIASES:
            return self._error_obs(
                f"Unknown table: {raw_table!r}. Valid tables: {list(_CANONICAL_TABLES)}",
                action_name="list_records",
            )
        table = _TABLE_ALIASES[raw_table]

        raw_limit = action.get("limit", MAX_PAGE_LIMIT)
        if raw_limit is None:
            raw_limit = MAX_PAGE_LIMIT
        if isinstance(raw_limit, bool) or not isinstance(raw_limit, int):
            return self._error_obs(
                f"Invalid limit: must be an integer between 1 and {MAX_PAGE_LIMIT}",
                action_name="list_records",
            )
        if raw_limit < 1 or raw_limit > MAX_PAGE_LIMIT:
            return self._error_obs(
                f"Limit {raw_limit} out of bounds; must be between 1 and {MAX_PAGE_LIMIT}",
                action_name="list_records",
            )

        raw_cursor = action.get("cursor", "0")
        if raw_cursor is None or raw_cursor == "":
            offset = 0
        elif isinstance(raw_cursor, bool):
            return self._error_obs(
                "Invalid cursor: must be a non-negative integer string",
                action_name="list_records",
            )
        elif isinstance(raw_cursor, int):
            if raw_cursor < 0:
                return self._error_obs(
                    "Invalid cursor: must be non-negative",
                    action_name="list_records",
                )
            offset = raw_cursor
        elif isinstance(raw_cursor, str):
            if not raw_cursor.isdigit():
                return self._error_obs(
                    f"Invalid cursor {raw_cursor!r}: must be a non-negative integer string",
                    action_name="list_records",
                )
            offset = int(raw_cursor)
        else:
            return self._error_obs(
                "Invalid cursor: must be a non-negative integer string",
                action_name="list_records",
            )

        rows = self._tables[table]
        page = copy.deepcopy(rows[offset : offset + raw_limit])
        next_offset = offset + len(page)
        next_cursor = str(next_offset) if next_offset < len(rows) else None

        return {
            "status": "ok",
            "action": "list_records",
            "table": table,
            "records": page,
            "count": len(page),
            "total_count": len(rows),
            "cursor": str(offset),
            "next_cursor": next_cursor,
            "step": self._step_count,
            "budget_remaining": self._budget_remaining(),
        }

    def _handle_get_record(self, action: dict[str, Any]) -> dict[str, Any]:
        raw_table = action.get("table")
        if not isinstance(raw_table, str) or raw_table not in _TABLE_ALIASES:
            return self._error_obs(
                f"Unknown table: {raw_table!r}. Valid tables: {list(_CANONICAL_TABLES)}",
                action_name="get_record",
            )
        table = _TABLE_ALIASES[raw_table]

        rec_id = action.get("id")
        if not isinstance(rec_id, str) or not rec_id.strip():
            return self._error_obs(
                "Invalid record id: must be a non-empty string",
                action_name="get_record",
            )

        record = self._table_indices[table].get(rec_id)
        if record is None:
            return self._error_obs(
                f"Record {rec_id!r} not found in table {table!r}",
                action_name="get_record",
            )

        return {
            "status": "ok",
            "action": "get_record",
            "table": table,
            "id": rec_id,
            "record": copy.deepcopy(record),
            "step": self._step_count,
            "budget_remaining": self._budget_remaining(),
        }

    def _handle_get_document(self, action: dict[str, Any]) -> dict[str, Any]:
        doc_id = action.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id.strip():
            return self._error_obs(
                "Invalid doc_id: must be a non-empty string",
                action_name="get_document",
            )

        doc = self._documents.get(doc_id)
        if doc is None:
            return self._error_obs(
                f"Document {doc_id!r} not found",
                action_name="get_document",
            )

        return {
            "status": "ok",
            "action": "get_document",
            "doc_id": doc_id,
            "document": copy.deepcopy(doc),
            "step": self._step_count,
            "budget_remaining": self._budget_remaining(),
        }

    def _handle_calculator(self, action: dict[str, Any]) -> dict[str, Any]:
        expr = action.get("expression")
        try:
            result = safe_calculate(expr)
        except CalculatorError as exc:
            return self._error_obs(f"Calculator error: {exc}", action_name="calculator")

        return {
            "status": "ok",
            "action": "calculator",
            "expression": expr,
            "result": result,
            "step": self._step_count,
            "budget_remaining": self._budget_remaining(),
        }

    def step(
        self,
        action: dict[str, Any],
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Execute one environment step following the Gymnasium 5-tuple contract."""
        if not self._initialized:
            raise RuntimeError("Environment must be reset() before calling step()")

        if self._terminated or self._truncated:
            obs = self._error_obs("Episode has already ended; call reset() to start a new episode.")
            info = {
                "episode_id": self._episode_id,
                "mode": self.mode,
                "step": self._step_count,
                "terminated": self._terminated,
                "truncated": self._truncated,
            }
            return obs, 0.0, self._terminated, self._truncated, info

        # Check wall-clock timeout before executing the step
        elapsed = time.monotonic() - self._start_time
        if elapsed > self.timeout_seconds:
            reward = self._finalize_episode(payload=None, terminated=False, truncated=True)
            obs = self._error_obs("Episode timed out before action could complete.")
            obs["truncated_reason"] = "timeout"
            info = {
                "episode_id": self._episode_id,
                "mode": self.mode,
                "step": self._step_count,
                "terminated": False,
                "truncated": True,
            }
            return obs, reward, False, True, info

        self._step_count += 1

        if not isinstance(action, dict):
            self._inspection_steps += 1
            obs = self._error_obs("Action must be a JSON object (dict)")
        else:
            act_name = action.get("action")
            if act_name == "submit":
                if "findings" not in action or action["findings"] is None:
                    # Explicit submit with missing/null payload is malformed (not a clean truncation)
                    payload: dict[str, Any] = {}
                else:
                    payload = action["findings"]
                reward = self._finalize_episode(payload=payload, terminated=True, truncated=False)
                obs = {
                    "status": "submitted",
                    "action": "submit",
                    "episode_id": self._episode_id,
                    "step": self._step_count,
                    "budget_remaining": self._budget_remaining(),
                }
                info = {
                    "episode_id": self._episode_id,
                    "mode": self.mode,
                    "step": self._step_count,
                    "terminated": True,
                    "truncated": False,
                }
                return obs, reward, True, False, info

            self._inspection_steps += 1
            if act_name == "list_records":
                obs = self._handle_list_records(action)
            elif act_name == "get_record":
                obs = self._handle_get_record(action)
            elif act_name == "get_document":
                obs = self._handle_get_document(action)
            elif act_name == "calculator":
                obs = self._handle_calculator(action)
            else:
                obs = self._error_obs(f"Unknown action: {act_name!r}")

        # Check step budget or wall-clock timeout after non-submit action
        timed_out = (time.monotonic() - self._start_time) > self.timeout_seconds
        out_of_steps = self._step_count >= self.max_steps
        if timed_out or out_of_steps:
            reward = self._finalize_episode(payload=None, terminated=False, truncated=True)
            obs["truncated_reason"] = "timeout" if timed_out else "max_steps_reached"
            info = {
                "episode_id": self._episode_id,
                "mode": self.mode,
                "step": self._step_count,
                "terminated": False,
                "truncated": True,
            }
            return obs, reward, False, True, info

        info = {
            "episode_id": self._episode_id,
            "mode": self.mode,
            "step": self._step_count,
            "terminated": False,
            "truncated": False,
        }
        return obs, 0.0, False, False, info

    def get_evaluator_report(self) -> dict[str, Any]:
        """Return evaluator-only diagnostics after an episode has ended."""
        if self._evaluator_report is None:
            raise RuntimeError("Evaluator report is only available after the episode finishes.")
        return copy.deepcopy(self._evaluator_report)


def _serialize_episode(episode: GeneratedEpisode) -> dict[str, Any]:
    return {
        "manifest": episode.manifest,
        "policy": episode.policy,
        "ledger": episode.ledger,
        "invoices": episode.invoices,
        "payments": episode.payments,
        "fx_rates": episode.fx_rates,
        "truth": episode.truth,
        "provenance": episode.provenance,
    }


class SubprocessAuditEnv:
    """Out-of-process isolated audit environment communicating over JSON-lines stdio.

    Spawns ``python3 -m audit_rl.eval_worker`` in a dedicated child process so that
    ``truth``, ``provenance``, and ``seed`` never enter the agent process's memory.
    """

    def __init__(
        self,
        mode: str = "eval",
        max_steps: int = DEFAULT_MAX_STEPS,
        timeout_seconds: float = 60.0,
    ) -> None:
        if mode not in ("train", "eval"):
            raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}")
        self.mode = mode
        self.max_steps = max_steps
        self.timeout_seconds = float(timeout_seconds)
        self._closed = False

        project_root = str(Path(__file__).resolve().parent.parent)
        env_vars = os.environ.copy()
        existing_pythonpath = env_vars.get("PYTHONPATH", "")
        if existing_pythonpath:
            if project_root not in existing_pythonpath.split(os.pathsep):
                env_vars["PYTHONPATH"] = f"{project_root}{os.pathsep}{existing_pythonpath}"
        else:
            env_vars["PYTHONPATH"] = project_root

        self._proc = subprocess.Popen(
            [sys.executable, "-m", "audit_rl.eval_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env_vars,
            cwd=project_root,
        )

        self._rpc(
            {
                "cmd": "init",
                "mode": self.mode,
                "max_steps": self.max_steps,
                "timeout_seconds": self.timeout_seconds,
            }
        )

    def _rpc(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("SubprocessAuditEnv is closed")
        if self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("Worker stdio pipes are unavailable")

        payload = json.dumps(request, separators=(",", ":"))
        try:
            self._proc.stdin.write(payload + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError as exc:
            stderr_text = ""
            if self._proc.stderr is not None:
                stderr_text = self._proc.stderr.read()
            raise RuntimeError(f"Eval worker pipe closed unexpectedly: {stderr_text}") from exc

        line = self._proc.stdout.readline()
        if not line:
            stderr_text = ""
            if self._proc.stderr is not None:
                stderr_text = self._proc.stderr.read()
            raise RuntimeError(f"Eval worker exited unexpectedly: {stderr_text}")

        response = json.loads(line)
        if not response.get("ok", False):
            err_msg = response.get("error", "Unknown worker error")
            err_type = response.get("error_type", "RuntimeError")
            if err_type == "ValueError":
                raise ValueError(err_msg)
            raise RuntimeError(f"[{err_type}] {err_msg}")
        return response

    def reset(
        self,
        seed: int = 0,
        config: EpisodeConfig | dict[str, Any] | None = None,
        episode: GeneratedEpisode | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the worker environment and return sanitized ``(observation, info)``."""
        cfg_payload: dict[str, Any] | None
        if isinstance(config, EpisodeConfig):
            cfg_payload = config.to_dict()
        elif isinstance(config, dict):
            cfg_payload = copy.deepcopy(config)
        else:
            cfg_payload = None

        req: dict[str, Any] = {
            "cmd": "reset",
            "seed": seed,
            "config": cfg_payload,
        }
        if episode is not None:
            req["episode"] = _serialize_episode(episode)

        resp = self._rpc(req)
        return resp["observation"], resp["info"]

    def step(
        self,
        action: dict[str, Any],
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Send an action to the worker and return ``(obs, reward, terminated, truncated, info)``."""
        resp = self._rpc({"cmd": "step", "action": action})
        return (
            resp["observation"],
            float(resp["reward"]),
            bool(resp["terminated"]),
            bool(resp["truncated"]),
            resp["info"],
        )

    def get_evaluator_report(self) -> dict[str, Any]:
        """Retrieve the evaluator report from the isolated worker after episode end."""
        resp = self._rpc({"cmd": "get_evaluator_report"})
        return resp["report"]

    def close(self) -> None:
        """Gracefully shut down the isolated worker subprocess."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._proc.poll() is None and self._proc.stdin is not None:
                try:
                    self._proc.stdin.write(json.dumps({"cmd": "close"}) + "\n")
                    self._proc.stdin.flush()
                except OSError:
                    pass
                try:
                    self._proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    self._proc.terminate()
                    try:
                        self._proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        self._proc.kill()
                        self._proc.wait()
        finally:
            for stream in (self._proc.stdin, self._proc.stdout, self._proc.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass

    def __enter__(self) -> SubprocessAuditEnv:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
