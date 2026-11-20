"""Read-only tools over an episode's public records."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from hashlib import blake2s
import json

from .auditor_env import (
    PublicAction, ToolCode, ToolError, _copy_public, _project_policy, _project_record,
)
from .calculator import calculate


DEFAULT_PAGE_SIZE = 50
_TABLE_KEYS = {"ledger": "row_id", "invoices": "invoice_id", "payments": "payment_id", "fx": None}


def _fx_key(row: Mapping) -> str:
    return f"fx:{row['date']}:{row['source_currency']}:{row['target_currency']}"


def _digest(value: object) -> str:
    serialized = json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return blake2s(serialized.encode("ascii"), digest_size=16).hexdigest()


class PublicTools:
    """Index a detached snapshot. Pass this callable to Episode.handle_action."""

    def __init__(
        self, *, episode_id: str, policy: Mapping[str, object],
        tables: Mapping[str, Sequence[Mapping[str, object]]],
    ):
        self._rows: dict[str, list[dict]] = {}
        self._by_id: dict[str, dict[str, dict]] = {}
        self._documents = {"policy": _copy_public(_project_policy(policy))}
        for table, key in _TABLE_KEYS.items():
            rows = [_copy_public(_project_record(row)) for row in tables[table]]
            lookup = {row[key] if key is not None else _fx_key(row): row for row in rows}
            if len(lookup) != len(rows):
                raise ValueError("duplicate public record ID")
            self._rows[table] = rows
            self._by_id[table] = lookup
            if table in ("invoices", "payments"):
                for row in rows:
                    doc_id = row["doc_id"]
                    if doc_id in self._documents:
                        raise ValueError("duplicate public document ID")
                    self._documents[doc_id] = row
        self._scope = _digest([episode_id, self._rows, self._documents["policy"]])
        self._positions: dict[tuple[str, str], int] = {}
        self._public_evidence_ids = frozenset(
            self._documents.keys() | self._by_id["ledger"].keys() | self._by_id["fx"].keys()
        )

    @property
    def public_evidence_ids(self) -> frozenset[str]:
        return self._public_evidence_ids

    def __call__(self, action: PublicAction) -> dict[str, object]:
        """Handle an already validated research action from AuditorEnv."""
        kind = action["type"]
        if kind == "list_records":
            return self._page(action["table"], action.get("cursor"), action.get("limit", DEFAULT_PAGE_SIZE))
        if kind == "get_record":
            return {"record": self._lookup(self._by_id[action["table"]], action["id"])}
        if kind == "get_document":
            return {"document": self._lookup(self._documents, action["doc_id"])}
        if kind == "calculator":
            return {"result": self._calculate(action["expression"])}
        raise ToolError(ToolCode.INVALID_ARGUMENT)

    @staticmethod
    def _lookup(index: dict[str, dict], record_id: str) -> dict:
        if record_id not in index:
            raise ToolError(ToolCode.NOT_FOUND)
        return deepcopy(index[record_id])

    def _page(self, table: str, cursor: str | None, limit: int) -> dict[str, object]:
        rows = self._rows[table]
        offset = 0
        if cursor is not None:
            position = self._positions.get((table, cursor))
            if position is None:
                raise ToolError(ToolCode.INVALID_CURSOR)
            offset = position
        end = offset + limit
        next_cursor = None
        if end < len(rows):
            next_cursor = _digest([self._scope, table, end])
            self._positions[table, next_cursor] = end
        return {"records": deepcopy(rows[offset:end]), "cursor": next_cursor}

    @staticmethod
    def _calculate(expression: str) -> str:
        try:
            return calculate(expression)
        except ZeroDivisionError:
            raise ToolError(ToolCode.DIVISION_BY_ZERO) from None
        except OverflowError:
            raise ToolError(ToolCode.CALCULATION_LIMIT) from None
        except ValueError:
            raise ToolError(ToolCode.INVALID_EXPRESSION) from None
