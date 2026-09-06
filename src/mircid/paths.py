"""Shared, relocatable paths for the public code and data repositories."""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(
    os.environ.get("MIRCID_DATA_ROOT", REPO_ROOT.parent / "MIRCID_dataset")
).expanduser().resolve()
WORK_ROOT = Path(
    os.environ.get("MIRCID_WORK_ROOT", REPO_ROOT / "outputs")
).expanduser().resolve()


def require(path: Path) -> Path:
    """Return *path* or fail with an actionable data-root message."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required file is missing: {path}. Set MIRCID_DATA_ROOT to the "
            "downloaded MIRCID_dataset directory."
        )
    return path

