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


def public_files(public: PublicEpisode) -> dict[str, bytes]:
    """Serialize only the six public v1 artifacts, in canonical UTF-8/LF."""
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
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    files = {
        "public": public_files(episode.public),
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
            private = projection == "evaluator-only"
            destination.mkdir(mode=0o700 if private else 0o755)
            for name, contents in documents.items():
                path = destination / name
                path.write_bytes(contents)
                if private:
                    path.chmod(0o600)
        staged.rename(target)
