"""Thread-safe pure-stdlib SQLite persistence for AuditLab experiments and policy versions."""

from __future__ import annotations

from contextlib import contextmanager
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def _default_db_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "platform.db"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class PlatformStore:
    """SQLite experiment and policy-version store (thread-safe via per-call connections)."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path).resolve() if db_path is not None else _default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    policy_name TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    config_version TEXT NOT NULL,
                    split TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    details_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS policy_versions (
                    version_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    policy_name TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    params_json TEXT NOT NULL,
                    train_mean_reward REAL NOT NULL,
                    val_f1 REAL NOT NULL,
                    eval_f1 REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_experiments_kind_created ON experiments(kind, id DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_policy_versions_created ON policy_versions(created_at DESC)"
            )
            conn.commit()

    def record_experiment(
        self,
        kind: str,
        policy_name: str,
        summary: dict[str, Any],
        details: dict[str, Any] | list[Any] | None = None,
        policy_version: str = "1.0.0",
        config_version: str = "1.0.0",
        split: str = "eval",
    ) -> int:
        """Persist an experiment record ('train', 'benchmark', 'episode', or 'verify') and return its row id."""
        created_at = _utc_now_iso()
        summary_json = json.dumps(summary, ensure_ascii=False)
        details_json = json.dumps(details if details is not None else {}, ensure_ascii=False)
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO experiments (
                    created_at, kind, policy_name, policy_version,
                    config_version, split, summary_json, details_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    str(kind),
                    str(policy_name),
                    str(policy_version),
                    str(config_version),
                    str(split),
                    summary_json,
                    details_json,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def list_experiments(self, limit: int = 30, kind: str | None = None) -> list[dict[str, Any]]:
        """List recent experiment records newest-first, optionally filtered by kind."""
        lim = max(1, min(int(limit), 500))
        with self._connect() as conn:
            if kind is not None:
                rows = conn.execute(
                    """
                    SELECT id, created_at, kind, policy_name, policy_version,
                           config_version, split, summary_json, details_json
                    FROM experiments
                    WHERE kind = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (str(kind), lim),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, created_at, kind, policy_name, policy_version,
                           config_version, split, summary_json, details_json
                    FROM experiments
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (lim,),
                ).fetchall()

        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "id": int(row["id"]),
                    "created_at": str(row["created_at"]),
                    "kind": str(row["kind"]),
                    "policy_name": str(row["policy_name"]),
                    "policy_version": str(row["policy_version"]),
                    "config_version": str(row["config_version"]),
                    "split": str(row["split"]),
                    "summary": json.loads(row["summary_json"]),
                    "details": json.loads(row["details_json"]),
                }
            )
        return out

    def save_policy_version(
        self,
        policy_name: str,
        seed: int,
        params: dict[str, Any],
        train_mean_reward: float,
        val_f1: float,
        eval_f1: float,
        version_id: str | None = None,
    ) -> str:
        """Persist a trained policy version and return its version_id."""
        vid = version_id or f"{policy_name}-s{int(seed)}-{uuid.uuid4().hex[:8]}"
        created_at = _utc_now_iso()
        params_json = json.dumps(params, ensure_ascii=False, sort_keys=True)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO policy_versions (
                    version_id, created_at, policy_name, seed,
                    params_json, train_mean_reward, val_f1, eval_f1
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(vid),
                    created_at,
                    str(policy_name),
                    int(seed),
                    params_json,
                    float(train_mean_reward),
                    float(val_f1),
                    float(eval_f1),
                ),
            )
            conn.commit()
        return str(vid)

    def list_policy_versions(self, limit: int = 20) -> list[dict[str, Any]]:
        """List saved policy versions newest-first."""
        lim = max(1, min(int(limit), 200))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT version_id, created_at, policy_name, seed,
                       params_json, train_mean_reward, val_f1, eval_f1
                FROM policy_versions
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
                """,
                (lim,),
            ).fetchall()

        return [
            {
                "version_id": str(row["version_id"]),
                "created_at": str(row["created_at"]),
                "policy_name": str(row["policy_name"]),
                "seed": int(row["seed"]),
                "params": json.loads(row["params_json"]),
                "train_mean_reward": float(row["train_mean_reward"]),
                "val_f1": float(row["val_f1"]),
                "eval_f1": float(row["eval_f1"]),
            }
            for row in rows
        ]

    def get_overview_stats(self) -> dict[str, Any]:
        """Return aggregate counts and latest summaries for the platform dashboard."""
        with self._connect() as conn:
            total_exp = int(conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0])
            by_kind_rows = conn.execute(
                "SELECT kind, COUNT(*) AS cnt FROM experiments GROUP BY kind"
            ).fetchall()
            counts_by_kind = {str(r["kind"]): int(r["cnt"]) for r in by_kind_rows}
            total_versions = int(conn.execute("SELECT COUNT(*) FROM policy_versions").fetchone()[0])

            latest_by_kind: dict[str, dict[str, Any] | None] = {}
            for kind in ("verify", "train", "benchmark", "episode"):
                row = conn.execute(
                    """
                    SELECT id, created_at, kind, policy_name, policy_version,
                           config_version, split, summary_json
                    FROM experiments
                    WHERE kind = ?
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (kind,),
                ).fetchone()
                if row is not None:
                    latest_by_kind[kind] = {
                        "id": int(row["id"]),
                        "created_at": str(row["created_at"]),
                        "kind": str(row["kind"]),
                        "policy_name": str(row["policy_name"]),
                        "policy_version": str(row["policy_version"]),
                        "config_version": str(row["config_version"]),
                        "split": str(row["split"]),
                        "summary": json.loads(row["summary_json"]),
                    }
                else:
                    latest_by_kind[kind] = None

            best_pv = conn.execute(
                """
                SELECT version_id, policy_name, seed, train_mean_reward, val_f1, eval_f1, created_at
                FROM policy_versions
                ORDER BY eval_f1 DESC, val_f1 DESC, rowid DESC
                LIMIT 1
                """
            ).fetchone()

        return {
            "db_path": str(self.db_path),
            "total_experiments": total_exp,
            "counts_by_kind": counts_by_kind,
            "total_policy_versions": total_versions,
            "latest_by_kind": latest_by_kind,
            "best_policy_version": (
                {
                    "version_id": str(best_pv["version_id"]),
                    "policy_name": str(best_pv["policy_name"]),
                    "seed": int(best_pv["seed"]),
                    "train_mean_reward": float(best_pv["train_mean_reward"]),
                    "val_f1": float(best_pv["val_f1"]),
                    "eval_f1": float(best_pv["eval_f1"]),
                    "created_at": str(best_pv["created_at"]),
                }
                if best_pv is not None
                else None
            ),
        }
