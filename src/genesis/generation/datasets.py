"""Reproduce and verify the public v1 train/dev datasets without network access."""

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

from .company import SCHEMA_VERSION, PublicEpisode, _generation_config, generate_episode
from .export import _json_line, public_files


SPLITS = ("train", "dev")
SPLIT_SIZE = 10
DATASET_VERSION = "1.1.0"
GENERATOR_VERSION = "1.1.0"
_CONFIG_FIELDS = {"split", "dataset_version", "generator_version", "schema_version", "entries"}


@dataclass(frozen=True)
class DatasetEntry:
    split: str
    seed: int
    config: dict


def _unique_fields(pairs: list[tuple[str, object]]) -> dict:
    fields: dict = {}
    for key, value in pairs:
        if key in fields:
            raise ValueError(f"duplicate index field: {key}")
        fields[key] = value
    return fields


def _read_index(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or invalid index file: {path.name}")
    try:
        result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_fields)
    except (UnicodeError, ValueError) as error:
        raise ValueError(f"{path.name}: {error}") from None
    if not isinstance(result, dict) or result.keys() != _CONFIG_FIELDS:
        raise ValueError(f"{path.name}: invalid index shape")
    return result


def load_dataset_entries(config_dir: Path | str) -> tuple[DatasetEntry, ...]:
    """Read versioned public indices in split order and reject invalid entries."""
    base = Path(config_dir)
    if base.is_symlink() or not base.is_dir():
        raise ValueError("missing or invalid config directory")
    entries = []
    seeds: set[int] = set()
    config_splits: dict[bytes, str] = {}
    for split in SPLITS:
        path = base / f"{split}_configs.json"
        index = _read_index(path)
        version = index["dataset_version"], index["generator_version"], index["schema_version"]
        if index["split"] != split or version != (DATASET_VERSION, GENERATOR_VERSION, SCHEMA_VERSION):
            raise ValueError(f"{path.name}: wrong split or unsupported version")
        rows = index["entries"]
        if not isinstance(rows, list) or len(rows) != SPLIT_SIZE:
            raise ValueError(f"{path.name}: expected {SPLIT_SIZE} entries")
        for position, row in enumerate(rows, 1):
            label = f"{path.name} entry {position}"
            if not isinstance(row, dict) or row.keys() != {"seed", "config"} or type(row["seed"]) is not int:
                raise ValueError(f"{label}: expected an integer seed and configuration")
            seed = row["seed"]
            if seed in seeds:
                raise ValueError(f"{label}: duplicate seed {seed}")
            try:
                config = _generation_config(row["config"])
            except ValueError as error:
                raise ValueError(f"{label}: {error}") from None
            if config != row["config"]:
                raise ValueError(f"{label}: use a normalized configuration")
            canonical = _json_line(_without(config, "episode_id"))
            if config_splits.get(canonical, split) != split:
                raise ValueError(f"{label}: configuration is already in the other split")
            config_splits[canonical] = split
            seeds.add(seed)
            entries.append(DatasetEntry(split, seed, config))
    return tuple(entries)


def _digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def _semantic_rows(rows) -> list[bytes]:
    return sorted(_json_line(row) for row in rows)


def _without(row: Mapping, *keys: str) -> dict:
    return {name: value for name, value in row.items() if name not in keys}


def _book_hash(public: PublicEpisode) -> str:
    """Fingerprint substantive rows, independent of opaque IDs and publication order."""
    invoices = {
        row["invoice_id"]: _digest(_json_line(_without(row, "invoice_id", "doc_id", "entity_id")))
        for row in public.invoices
    }
    payments = {
        row["payment_id"]: _digest(_json_line({
            **_without(row, "payment_id", "doc_id", "refund_of_payment_id"),
            "invoice_refs": (
                sorted(invoices[key] for key in row["invoice_refs"]) if row["invoice_refs"] is not None else None
            ),
        }))
        for row in public.payments
    }
    payment_rows = [
        (payments[row["payment_id"]], payments.get(row.get("refund_of_payment_id")))
        for row in public.payments
    ]
    ledger = [
        {
            **_without(row, "row_id", "entity_id"),
            "invoice_id": invoices.get(row["invoice_id"]),
            "payment_id": payments.get(row["payment_id"]),
        }
        for row in public.ledger
    ]
    chunks = [
        [_json_line(public.policy)], _semantic_rows(invoices.values()),
        _semantic_rows(payment_rows), _semantic_rows(ledger), _semantic_rows(public.fx),
    ]
    return _digest(b"\0".join(b"".join(rows) for rows in chunks))


def _index_bytes(entries: tuple[DatasetEntry, ...], split: str) -> bytes:
    return _json_line({
        "split": split, "dataset_version": DATASET_VERSION, "generator_version": GENERATOR_VERSION,
        "schema_version": SCHEMA_VERSION,
        "entries": [{"seed": entry.seed, "config": entry.config} for entry in entries if entry.split == split],
    })


def _materialize(entries: tuple[DatasetEntry, ...]) -> dict[str, bytes]:
    files = {f"configs/{split}_configs.json": _index_bytes(entries, split) for split in SPLITS}
    summary = []
    episode_ids: set[str] = set()
    books: set[str] = set()
    for entry in entries:
        try:
            public = generate_episode(entry.seed, entry.config).public
        except ValueError as error:
            raise ValueError(f"{entry.split} seed {entry.seed}: {error}") from None
        manifest = public.manifest
        episode_id = manifest["episode_id"]
        book_hash = _book_hash(public)
        if episode_id in episode_ids or book_hash in books:
            raise ValueError(f"{entry.split} seed {entry.seed}: duplicate episode or book")
        episode_ids.add(episode_id)
        books.add(book_hash)
        artifacts = public_files(public)
        for filename, contents in artifacts.items():
            files[f"{entry.split}/{episode_id}/{filename}"] = contents
        summary.append({
            "split": entry.split, "episode_id": episode_id,
            "documents": manifest["document_count"], "record_counts": manifest["record_counts"],
            "book_sha256": book_hash,
            "sha256": {filename: _digest(contents) for filename, contents in artifacts.items()},
        })
    files["checksums.json"] = _json_line({
        "dataset_version": DATASET_VERSION, "generator_version": GENERATOR_VERSION,
        "schema_version": SCHEMA_VERSION, "episodes": summary,
    })
    return files


def export_public_dataset(config_dir: Path | str, output_dir: Path | str) -> int:
    """Export the whole public dataset to a new directory and return its episode count."""
    target = Path(output_dir)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"output already exists: {target}")
    entries = load_dataset_entries(config_dir)
    files = _materialize(entries)
    target.mkdir()
    created = [target]
    try:
        for name, contents in files.items():
            path = target / name
            parent = target
            for component in Path(name).parts[:-1]:
                parent /= component
                if parent not in created:
                    parent.mkdir()
                    created.append(parent)
            with path.open("xb") as stream:
                created.append(path)
                stream.write(contents)
    except OSError:
        for path in reversed(created):
            if path.is_dir():
                path.rmdir()
            else:
                path.unlink()
        raise
    return len(entries)


def _check_tree(directory: Path, expected: dict[str, bytes]) -> None:
    expected_dirs = {parent.as_posix() for name in expected for parent in Path(name).parents} - {"."}
    actual_files = set()
    for path in directory.rglob("*"):
        name = path.relative_to(directory).as_posix()
        if path.is_symlink() or (path.is_dir() and name not in expected_dirs):
            raise ValueError(f"unexpected dataset entry: {name}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"unexpected dataset entry: {name}")
        if name == "README.md":
            continue
        if name not in expected:
            raise ValueError(f"unexpected dataset file: {name}")
        if path.read_bytes() != expected[name]:
            raise ValueError(f"dataset content mismatch: {name}")
        actual_files.add(name)
    missing = expected.keys() - actual_files
    if missing:
        raise ValueError(f"missing dataset file: {min(missing)}")


def check_public_dataset(root: Path | str) -> int:
    """Regenerate from public inputs and verify every canonical file and directory."""
    directory = Path(root)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("missing or invalid dataset directory")
    entries = load_dataset_entries(directory / "configs")
    _check_tree(directory, _materialize(entries))
    return len(entries)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="write a new public dataset directory")
    export.add_argument("--configs", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    check = commands.add_parser("check", help="recompute and verify a public dataset directory")
    check.add_argument("--dataset", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            count = export_public_dataset(args.configs, args.output)
        else:
            count = check_public_dataset(args.dataset)
    except (OSError, ValueError) as error:
        parser.exit(1, f"dataset: {error}\n")
    print(f"{args.command}: {count} public episodes ({SPLIT_SIZE} train, {SPLIT_SIZE} dev)")


if __name__ == "__main__":
    main()
