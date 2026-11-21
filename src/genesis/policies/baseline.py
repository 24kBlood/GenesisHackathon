"""Deterministic research and intentionally broad structured findings."""

from random import Random

from genesis.domain.enums import FindingType

from .protocol import Action, Observation, submit


_TABLES = ("ledger", "invoices", "payments", "fx")
_RECORD_ID = {"ledger": "row_id", "invoices": "invoice_id", "payments": "payment_id"}


class RandomPolicy:
    """Sample a stable action list with a policy-only seed and an eight-step cap."""

    def __init__(self, seed: int):
        self._rng = Random(seed)
        self._researches = 0
        self._records: set[tuple[str, str]] = set()
        self._documents = {"policy"}
        self._episode_id = ""

    def act(self, observation: Observation, previous_action: Action | None) -> Action:
        if previous_action is None:
            self._episode_id = observation["manifest"]["episode_id"]
        elif previous_action["type"] == "list_records":
            table = previous_action["table"]
            for row in observation.get("records", []):
                if table == "fx":
                    record_id = f"fx:{row['date']}:{row['source_currency']}:{row['target_currency']}"
                else:
                    record_id = row[_RECORD_ID[table]]
                self._records.add((table, record_id))
                if "doc_id" in row:
                    self._documents.add(row["doc_id"])
        if self._researches == 8 or observation["remaining_steps"] <= 1:
            return submit(self._episode_id)
        self._researches += 1
        candidates = [
            *({"type": "list_records", "table": table, "limit": 5} for table in _TABLES),
            {"type": "calculator", "expression": "2 + 2"},
            *({"type": "get_record", "table": table, "id": key} for table, key in sorted(self._records)),
            *({"type": "get_document", "doc_id": key} for key in sorted(self._documents)),
        ]
        return self._rng.choice(candidates)


class MarkAllPolicy:
    """Turn the first bounded ledger page into sweeping, schema-valid findings."""

    def __init__(self):
        self._episode_id = ""

    def act(self, observation: Observation, previous_action: Action | None) -> Action:
        if previous_action is None:
            self._episode_id = observation["manifest"]["episode_id"]
            if observation["remaining_steps"] > 1:
                return {"type": "list_records", "table": "ledger", "limit": 30}
        types = tuple(FindingType)
        findings = [
            {
                "type": str(types[index % len(types)]), "affected_ids": [row["row_id"]],
                "claimed_amount_minor": abs(row["debit_minor"] - row["credit_minor"]),
                "currency": row["currency"], "evidence_ids": [row["row_id"]],
                "explanation": "Flagged from a single ledger entry.",
            }
            for index, row in enumerate(observation.get("records", [])[:30])
        ]
        return submit(self._episode_id, findings)
