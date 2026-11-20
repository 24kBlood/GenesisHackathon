"""Deterministic v1 episode serialization."""

import csv
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .company import Episode, PublicEpisode

LEDGER_COLUMNS = (
    "row_id", "timestamp", "account", "debit_minor", "credit_minor", "currency",
    "entity_id", "invoice_id", "payment_id",
)
FX_COLUMNS = ("date", "source_currency", "target_currency", "rate")


def _json_line(value: object) -> bytes:
    return (json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"),
    ) + "\n").encode("utf-8")


def _csv(columns: tuple[str, ...], rows) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow("null" if row[key] is None else row[key] for key in columns)
    return stream.getvalue().encode("utf-8")


def _public_files(public: PublicEpisode) -> dict[str, bytes]:
    return {
        "manifest.json": _json_line(public.manifest),
        "policy.json": _json_line(public.policy),
        "ledger.csv": _csv(LEDGER_COLUMNS, public.ledger),
        "invoices.jsonl": b"".join(_json_line(invoice) for invoice in public.invoices),
        "payments.jsonl": b"".join(_json_line(payment) for payment in public.payments),
        "fx.csv": _csv(FX_COLUMNS, public.fx),
    }


def export_episode(episode: Episode, directory: Path | str) -> None:
    """Write UTF-8/LF artifacts to a new episode directory.

    Publish public and evaluator-only files under separate subdirectories.
    The caller chooses the destination; it must not exist and its parent
    directory must exist. An unsuccessful serialization is never published.
    """
    target = Path(directory)
    if target.exists():
        raise FileExistsError(target)
    files = {
        "public": _public_files(episode.public),
        "evaluator-only": {
            "truth.json": _json_line(episode.evaluator.truth),
            "provenance.json": _json_line(episode.evaluator.provenance),
        },
    }
    with TemporaryDirectory(prefix=".genesis-", dir=target.parent) as temporary:
        staged = Path(temporary) / "episode"
        staged.mkdir()
        for projection, documents in files.items():
            destination = staged / projection
            destination.mkdir()
            for name, contents in documents.items():
                (destination / name).write_bytes(contents)
        staged.rename(target)
