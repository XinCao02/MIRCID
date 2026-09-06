"""Evaluate inferred TF activities against known TF perturbations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def evaluate(activities: pd.DataFrame, metadata: pd.DataFrame, top_fraction: float) -> pd.DataFrame:
    required = {"sample_id", "perturbed_tf"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"Metadata must contain {sorted(required)}")
    if activities.index.duplicated().any() or metadata["sample_id"].duplicated().any():
        raise ValueError("Sample identifiers must be unique")
    metadata = metadata.set_index("sample_id").loc[activities.index]
    ranks: list[float] = []
    signed_correct: list[float] = []
    for sample_id, row in activities.iterrows():
        tf = str(metadata.loc[sample_id, "perturbed_tf"])
        if tf not in activities.columns:
            ranks.append(np.nan)
            signed_correct.append(np.nan)
            continue
        descending = row.abs().rank(method="average", ascending=False)
        ranks.append(float(descending[tf]))
        if "effect" in metadata.columns:
            effect = str(metadata.loc[sample_id, "effect"]).lower()
            expected = 1 if effect.startswith("act") else -1 if effect.startswith("inh") else 0
            signed_correct.append(float(expected != 0 and np.sign(row[tf]) == expected))
        else:
            signed_correct.append(np.nan)
    n_tfs = activities.shape[1]
    threshold = max(1, int(np.ceil(top_fraction * n_tfs)))
    return pd.DataFrame(
        {
            "sample_id": activities.index.to_numpy(dtype=str),
            "perturbed_tf": metadata["perturbed_tf"].astype(str).to_numpy(),
            "rank": ranks,
            "success": np.asarray(ranks) <= threshold,
            "direction_correct": signed_correct,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("activities", type=Path, help="CSV: samples by inferred TF activities")
    parser.add_argument("metadata", type=Path, help="CSV with sample_id and perturbed_tf")
    parser.add_argument("output", type=Path)
    parser.add_argument("--top-fraction", type=float, default=0.30)
    args = parser.parse_args()
    activities = pd.read_csv(args.activities, index_col=0)
    metadata = pd.read_csv(args.metadata)
    result = evaluate(activities, metadata, args.top_fraction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    summary = result[["success", "direction_correct"]].mean(numeric_only=True)
    print(f"median_rank={result['rank'].median():.3f}")
    print(summary.to_string())


if __name__ == "__main__":
    main()
