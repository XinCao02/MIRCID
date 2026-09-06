#!/usr/bin/env python3
"""Build the MIRCID dataset inventory, checksums and duplicate audit."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "MIRCID_dataset"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def delimited_shape(path: Path) -> tuple[int | None, int | None]:
    if path.suffix.lower() not in {".csv", ".tsv"}:
        return None, None
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration:
            return 0, 0
        rows = sum(1 for _ in reader)
    return rows, len(header)


def main() -> None:
    paths = sorted(
        path for path in ROOT.rglob("*")
        if path.is_file() and path.name not in {"manifest.tsv", "checksums.sha256"}
    )
    records = []
    by_hash: dict[str, list[str]] = {}
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        digest = sha256(path)
        rows, columns = delimited_shape(path)
        records.append(
            {
                "path": relative,
                "module": relative.split("/", 1)[0],
                "bytes": path.stat().st_size,
                "rows_excluding_header": "" if rows is None else rows,
                "columns": "" if columns is None else columns,
                "sha256": digest,
            }
        )
        by_hash.setdefault(digest, []).append(relative)
    duplicates = {key: value for key, value in by_hash.items() if len(value) > 1}
    if duplicates:
        details = "\n".join(f"{key}: {value}" for key, value in duplicates.items())
        raise SystemExit(f"Byte-identical duplicates detected:\n{details}")
    with (ROOT / "manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(records)
    with (ROOT / "checksums.sha256").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(f"{record['sha256']}  {record['path']}\n")
    print(f"PASS: {len(records)} unique files; no byte-identical duplicates")


if __name__ == "__main__":
    main()

