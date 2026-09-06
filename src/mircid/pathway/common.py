from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from mircid.paths import DATA_ROOT, REPO_ROOT, WORK_ROOT

# ROOT is the writable experiment-output root retained for compatibility with
# the original aligned scripts. Input data always come from DATA_ROOT.
ROOT = WORK_ROOT / "pathway"
CONFIG_PATH = REPO_ROOT / "configs" / "pathway_experiment.json"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def normalize_sample_ids(values: Iterable[object]) -> pd.Index:
    idx = pd.Index(values).astype(str).str.strip()
    idx = idx.str.replace("JAK.STAT.", "JAK-STAT.", regex=False)
    idx = idx.str.replace(".E.", ".E-", regex=False)
    idx = idx.str.replace(".S.", ".S-", regex=False)
    idx = idx.str.replace(".GEOD.", "GEOD-", regex=False)
    idx = idx.str.replace(r"GEOD\.(\d+)\.", r"GEOD-\1.", regex=True)
    idx = idx.str.replace(r"(\.E-[A-Z]+)\.(\d+)\.", r"\1-\2.", regex=True)
    idx = idx.str.replace(r"(\.S-[A-Z]+)\.(\d+)\.", r"\1-\2.", regex=True)
    return idx


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def environment_manifest() -> dict:
    packages = {}
    for package in ["numpy", "pandas", "scipy", "sklearn", "torch", "matplotlib", "seaborn"]:
        try:
            module = __import__(package)
            packages[package] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            packages[package] = f"unavailable:{type(exc).__name__}"
    cuda = None
    try:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception:
        pass
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "cuda": cuda,
        "cpu_count": os.cpu_count(),
    }


def load_processed() -> dict[str, np.ndarray]:
    path = DATA_ROOT / "pathway" / "processed" / "pathway_aligned.npz"
    if not path.exists():
        raise FileNotFoundError(f"Run prepare_data.py first: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def split_indices(seed: int, data: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    data = load_processed() if data is None else data
    manifest = pd.read_csv(DATA_ROOT / "pathway" / "splits" / "frozen_20_splits.csv")
    ids = data["sample_ids"].astype(str)
    positions = {sample_id: i for i, sample_id in enumerate(ids)}
    sub = manifest[manifest["split_seed"] == int(seed)]
    if len(sub) != len(ids):
        raise ValueError(f"Incomplete split manifest for seed {seed}")
    result = {}
    for partition in ["train", "validation", "test"]:
        part_ids = sub.loc[sub["partition"] == partition, "sample_id"].astype(str)
        result[partition] = np.asarray([positions[x] for x in part_ids], dtype=np.int64)
    if len(np.unique(np.concatenate(list(result.values())))) != len(ids):
        raise ValueError(f"Split overlap or omission for seed {seed}")
    return result


def build_features(data: dict[str, np.ndarray], feature_space: str) -> np.ndarray:
    keys = load_config()["feature_spaces"][feature_space]
    return np.concatenate([data[key] for key in keys], axis=1).astype(np.float32, copy=False)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def prediction_rows(
    *,
    model: str,
    feature_space: str,
    seed: int,
    partition: str,
    indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_ids: np.ndarray,
    class_names: np.ndarray,
) -> list[dict]:
    return [
        {
            "run_id": load_config()["run_id"],
            "model": model,
            "feature_space": feature_space,
            "split_seed": int(seed),
            "partition": partition,
            "sample_id": str(sample_ids[idx]),
            "true_label": int(truth),
            "pred_label": int(pred),
            "true_pathway": str(class_names[int(truth)]),
            "pred_pathway": str(class_names[int(pred)]),
            "correct": bool(truth == pred),
        }
        for idx, truth, pred in zip(indices, y_true, y_pred)
    ]


def metric_rows(model: str, feature_space: str, seed: int, partition: str, values: dict[str, float]) -> list[dict]:
    return [
        {
            "run_id": load_config()["run_id"],
            "model": model,
            "feature_space": feature_space,
            "split_seed": int(seed),
            "partition": partition,
            "metric": metric,
            "value": value,
        }
        for metric, value in values.items()
    ]


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    compression = "gzip" if path.suffix == ".gz" else None
    frame.to_csv(tmp, index=False, compression=compression)
    tmp.replace(path)


def read_csv_auto(path: Path) -> pd.DataFrame:
    """Read CSVs while repairing the early-run plain-text .gz naming issue."""
    with path.open("rb") as handle:
        is_gzip = handle.read(2) == b"\x1f\x8b"
    return pd.read_csv(path, compression="gzip" if is_gzip else None)
