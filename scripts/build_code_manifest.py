"""Build a deterministic SHA-256 inventory of the public code staging tree."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "manifests" / "code_manifest.tsv"
EXCLUDED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", "outputs"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    files = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != OUTPUT
        and not EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts)
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=["path", "bytes", "sha256"],
        )
        writer.writeheader()
        for path in sorted(files):
            writer.writerow(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    print(f"PASS: indexed {len(files)} code-repository files -> {OUTPUT}")


if __name__ == "__main__":
    main()
