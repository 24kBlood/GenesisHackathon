"""Trusted QA transform of accounting records, with strict ID correspondence."""

from dataclasses import asdict
from hashlib import sha256
from random import Random

from genesis.generation.company import Episode, EvaluatorEpisode, PublicEpisode


TABLES = ("ledger", "invoices", "payments", "fx")
_REFERENCE_LISTS = {
    "invoice_refs", "affected_ids", "acceptable_evidence_sets", "evidence_ids",
    "invoice_ids", "payment_ids",
}


def evidence_catalog(public: PublicEpisode) -> frozenset[str]:
    return frozenset({
        "policy", *(row["row_id"] for row in public.ledger),
        *(doc["doc_id"] for doc in [*public.invoices, *public.payments]),
        *(f"fx:{row['date']}:{row['source_currency']}:{row['target_currency']}" for row in public.fx),
    })


def _references(value, key=""):
    if isinstance(value, dict):
        for field, item in value.items():
            yield from _references(item, field)
    elif isinstance(value, list):
        for item in value:
            yield from _references(item, key)
    elif value is not None and (key.endswith("_id") or key in _REFERENCE_LISTS):
        yield value


def _definitions(public: PublicEpisode, truth: dict, provenance: dict) -> set[str]:
    # A before/after ledger snapshot can contain an ID removed by a mutation.
    historical_rows = {
        row["row_id"]
        for change in provenance["changes"]
        for state in (change["before"], change["after"])
        for row in (state or {}).get("rows", [])
    }
    return {
        public.manifest["episode_id"], *historical_rows,
        *(row["row_id"] for row in public.ledger),
        *(row["entity_id"] for row in public.ledger),
        *(doc["entity_id"] for doc in public.invoices),
        *(doc["doc_id"] for doc in [*public.invoices, *public.payments]),
        *(doc["invoice_id"] for doc in public.invoices),
        *(doc["payment_id"] for doc in public.payments),
        *(issue["issue_id"] for issue in truth["issues"]),
        *(change["change_id"] for change in provenance["changes"]),
    }


def validate_links(episode: Episode) -> frozenset[str]:
    """Check typed public links, live evidence, and references in private history."""
    public = episode.public
    truth, provenance = episode.evaluator.truth, episode.evaluator.provenance
    if not public.manifest["episode_id"] == truth["episode_id"] == provenance["episode_id"]:
        raise ValueError("episode ID references disagree")
    invoices = {row["invoice_id"] for row in public.invoices}
    payments = {row["payment_id"] for row in public.payments}
    evidence = evidence_catalog(public)
    if len(evidence) != 1 + len(public.ledger) + len(public.invoices) + len(public.payments) + len(public.fx):
        raise ValueError("duplicated public evidence ID")
    if len(invoices) != len(public.invoices) or len(payments) != len(public.payments):
        raise ValueError("duplicated public record ID")
    valid_affected = evidence | invoices | payments
    for row in public.ledger:
        invoice_id = row["invoice_id"]
        payment_id = row["payment_id"]
        if (invoice_id is not None and invoice_id not in invoices) or (
            payment_id is not None and payment_id not in payments
        ):
            raise ValueError("unresolved ledger reference")
    for row in public.payments:
        refund_id = row.get("refund_of_payment_id")
        if not set(row["invoice_refs"] or []) <= invoices or (
            refund_id is not None and refund_id not in payments
        ):
            raise ValueError("unresolved payment reference")
    for issue in truth["issues"]:
        if not set(issue["affected_ids"]) <= valid_affected:
            raise ValueError("unresolved truth affected ID")
        if not issue["acceptable_evidence_sets"] or any(
            not refs or not set(refs) <= evidence for refs in issue["acceptable_evidence_sets"]
        ):
            raise ValueError("unresolved truth evidence reference")
    issue_ids = {issue["issue_id"] for issue in truth["issues"]}
    if len(issue_ids) != len(truth["issues"]) or {change["issue_id"] for change in provenance["changes"]} != issue_ids:
        raise ValueError("truth and provenance issue IDs disagree")
    opaque = _definitions(public, truth, provenance)
    if not set(_references(provenance)) <= opaque | evidence:
        raise ValueError("unresolved provenance reference")
    return frozenset(opaque)


def rewrite_ids(value, mapping):
    """Rewrite ID-bearing fields only. Non-ID descriptions, times and amounts stay exact."""
    def visit(node, key=""):
        if isinstance(node, dict):
            return {field: visit(item, field) for field, item in node.items()}
        if isinstance(node, list):
            return [visit(item, key) for item in node]
        if isinstance(node, str) and (key.endswith("_id") or key in _REFERENCE_LISTS):
            return mapping.get(node, node)
        return node

    return visit(value)


def permute_episode(episode: Episode, *, salt: int, mapping=None) -> tuple[Episode, dict[str, str]]:
    opaque = validate_links(episode)
    if mapping is None:
        mapping = {
            key: "qa-" + sha256(f"{salt}:{key}".encode("utf-8")).hexdigest()[:24]
            for key in sorted(opaque)
        }
    targets = set(mapping.values())
    if (
        set(mapping) != opaque or not all(type(key) is str and key for key in targets)
        or len(targets) != len(opaque) or targets & (opaque | evidence_catalog(episode.public))
    ):
        raise ValueError("ID mapping is not a fresh bijection")
    public = rewrite_ids(asdict(episode.public), mapping)
    order = Random(salt)
    for table in TABLES:
        rows = public[table]
        original = rows.copy()
        order.shuffle(rows)
        if len(rows) > 1 and rows == original:
            rows.append(rows.pop(0))
    moved = Episode(
        PublicEpisode(**public), EvaluatorEpisode(**rewrite_ids(asdict(episode.evaluator), mapping)),
    )
    validate_links(moved)
    return moved, dict(mapping)
