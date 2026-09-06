#!/usr/bin/env python3
"""Build a fully paired four-model panel-e source table for Figure 4.5 v3."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
ENHANCEMENT = next(parent for parent in ROOT.parents if parent.name == "enhancement")
ALIGNED = ENHANCEMENT / "aligned" / "PathwayClassificationExp"
FORMAL_MIRONLY = (
    ALIGNED
    / "figures"
    / "enhanced"
    / "figS.1"
    / "control_miRonly"
    / "results"
    / "mironly_metrics.csv"
)
MODELS = ["RF", "MLP", "KAN", "ResNet"]
CONTRAST_GENE = "Gene + HubmiR − Gene"
CONTRAST_MIR = "Gene + HubmiR − HubmiR only"


def main() -> None:
    current = pd.read_csv(DATA / "selected_task_deltas.csv")
    benchmark = pd.read_csv(DATA / "source_fig4_v9_lowRF_selected_test_metrics.csv")
    old_mironly = pd.read_csv(DATA / "source_mironly_metrics.csv")
    formal_mironly = pd.read_csv(FORMAL_MIRONLY)
    new_mironly = pd.read_csv(DATA / "source_mlp_kan_mironly_v3.csv")
    mironly = (
        pd.concat([formal_mironly, old_mironly, new_mironly], ignore_index=True, sort=False)
        .drop_duplicates(["model", "feature_space", "split_seed", "partition", "metric"], keep="last")
    )

    seed_map = {
        model: sorted(
            current.loc[
                current["model"].eq(model)
                & current["contrast"].eq(CONTRAST_GENE)
                & current["metric"].eq("macro_f1"),
                "split_seed",
            ]
            .astype(int)
            .unique()
            .tolist()
        )
        for model in MODELS
    }
    if any(len(seeds) != 7 for seeds in seed_map.values()):
        raise ValueError(f"Expected seven selected seeds per model: {seed_map}")

    paired_frames: list[pd.DataFrame] = []
    for model, seeds in seed_map.items():
        combined = benchmark[
            benchmark["model"].eq(model)
            & benchmark["feature_space"].eq("Gene + HubmiR")
            & benchmark["partition"].eq("test")
            & benchmark["metric"].isin(["accuracy", "macro_f1"])
            & benchmark["split_seed"].isin(seeds)
        ][["model", "split_seed", "metric", "value"]].rename(columns={"value": "value_combined"})
        mir = mironly[
            mironly["model"].eq(model)
            & mironly["feature_space"].eq("HubmiR only")
            & mironly["partition"].eq("test")
            & mironly["metric"].isin(["accuracy", "macro_f1"])
            & mironly["split_seed"].isin(seeds)
        ][["model", "split_seed", "metric", "value"]].rename(columns={"value": "value_mironly"})
        paired = combined.merge(mir, on=["model", "split_seed", "metric"], validate="one_to_one")
        paired["delta"] = paired["value_combined"] - paired["value_mironly"]
        expected_rows = len(seeds) * 2
        if len(paired) != expected_rows:
            raise ValueError(f"Incomplete paired {model} table: expected {expected_rows}, found {len(paired)}")
        paired_frames.append(paired)

    paired_all = pd.concat(paired_frames, ignore_index=True).sort_values(
        ["model", "metric", "split_seed"]
    )
    if len(paired_all) != 56 or not np.isfinite(paired_all["delta"]).all():
        raise ValueError("The four-model paired HubmiR-only contrast table is incomplete")
    paired_all.to_csv(DATA / "paired_combined_minus_mironly_v3.csv", index=False)

    gene_rows = current[current["contrast"].eq(CONTRAST_GENE)].copy()
    mir_rows = paired_all[paired_all["metric"].eq("macro_f1")][
        ["model", "split_seed", "metric", "delta"]
    ].copy()
    mir_rows["contrast"] = CONTRAST_MIR
    mir_rows["selection_status"] = "same selected seeds as the paired Gene + HubmiR contrast"
    task_v3 = pd.concat([gene_rows, mir_rows], ignore_index=True).sort_values(
        ["contrast", "model", "split_seed"]
    )
    expected = {(contrast, model) for contrast in [CONTRAST_GENE, CONTRAST_MIR] for model in MODELS}
    observed = set(zip(task_v3["contrast"], task_v3["model"]))
    if observed != expected or len(task_v3) != 56:
        raise ValueError(f"Incomplete Figure 4.5 v3 task grid: {observed}")
    if not task_v3.groupby(["contrast", "model"]).size().eq(7).all():
        raise ValueError("Every Figure 4.5 v3 panel-e group must contain seven paired seeds")
    task_v3.to_csv(DATA / "selected_task_deltas_v3.csv", index=False)

    summary = (
        task_v3.groupby(["contrast", "model"])["delta"]
        .agg(n="size", mean="mean", sd="std", minimum="min", maximum="max")
        .reset_index()
    )
    summary.to_csv(DATA / "selected_task_deltas_v3_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
