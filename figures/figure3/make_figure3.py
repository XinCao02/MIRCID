#!/usr/bin/env python3
"""Replot the published Figure 3b aggregate TFA benchmark values."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from mircid.paths import DATA_ROOT


DATA = DATA_ROOT / "figure_source_data" / "figure3" / "tfa_figure3_aggregate_metrics.csv"
OUTPUT = Path(__file__).resolve().parent / "figure3b_replot"


def main() -> None:
    data = pd.read_csv(DATA)
    datasets = ["A375", "MCF7", "Holland"]
    metrics = ["Success", "Accuracy", "Median"]
    plt.rcParams.update({"font.family": "Arial", "font.size": 6.5, "pdf.fonttype": 42})
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 5.6), constrained_layout=True)
    for row, dataset in enumerate(datasets):
        for col, metric in enumerate(metrics):
            axis = axes[row, col]
            subset = data[(data.dataset == dataset) & (data.metric_display_label == metric)]
            table = subset.pivot(index="network", columns="method", values="value")
            sns.heatmap(table, ax=axis, cmap="YlGnBu", annot=True, fmt=".2f", cbar=col == 2,
                        annot_kws={"fontsize": 5.5}, linewidths=0.5, linecolor="white")
            axis.set_title(f"{dataset} · {metric}", fontweight="bold")
            axis.set(xlabel="", ylabel="")
            axis.tick_params(axis="x", rotation=0)
            axis.tick_params(axis="y", rotation=0)
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT.with_suffix(f".{suffix}"), dpi=450 if suffix == "png" else None, bbox_inches="tight")


if __name__ == "__main__":
    main()
