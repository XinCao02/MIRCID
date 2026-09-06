#!/usr/bin/env python3
"""Replot the selected Figure 4 pathway-classification distributions."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mircid.paths import DATA_ROOT


DATA = DATA_ROOT / "figure_source_data" / "figure4"
OUTPUT = Path(__file__).resolve().parent / "figure4_replot"
FEATURES = ["Gene", "Gene + TFA", "Gene + HubmiR", "All features"]
COLORS = ["#59A14F", "#E0A45B", "#4E79A7", "#A7A9AC"]


def main() -> None:
    data = pd.read_csv(DATA / "fig4_v9_lowRF_PathClass_selected_test_metrics.csv")
    baseline = pd.read_csv(DATA / "fig4_v9_lowRF_PathClass_progeny_baselines.csv").set_index("metric")
    models = list(dict.fromkeys(data["model"]))
    plt.rcParams.update({"font.family": "Arial", "font.size": 6.5, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for axis, metric, title in zip(axes, ["accuracy", "macro_f1"], ["Accuracy", "Macro-F1"]):
        positions, values, colors = [], [], []
        for model_index, model in enumerate(models):
            for feature_index, feature in enumerate(FEATURES):
                subset = data[(data.model == model) & (data.feature_space == feature) & (data.metric == metric)]
                positions.append(model_index * 5 + feature_index)
                values.append(subset["value"].to_numpy())
                colors.append(COLORS[feature_index])
        boxes = axis.boxplot(values, positions=positions, widths=0.68, patch_artist=True, showfliers=False, whis=(0, 100))
        for patch, color, position, vector in zip(boxes["boxes"], colors, positions, values):
            patch.set(facecolor=color, edgecolor="#555555", linewidth=0.6)
            axis.scatter(position + np.linspace(-0.20, 0.20, len(vector)), vector, s=8,
                         facecolor="white", edgecolor=color, linewidth=0.5, zorder=3)
        axis.axhline(float(baseline.loc[metric, "progeny_mean"]), color="#4B8F57", linestyle="--", linewidth=0.9)
        axis.set_xticks([index * 5 + 1.5 for index in range(len(models))], models)
        axis.set_title(title, loc="left", pad=7, fontweight="bold")
        axis.set_ylim(0.35, 0.90)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Held-out performance")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color, label=label) for color, label in zip(COLORS, FEATURES)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=False)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.18, top=0.82, wspace=0.04)
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT.with_suffix(f".{suffix}"), dpi=450 if suffix == "png" else None, bbox_inches="tight")


if __name__ == "__main__":
    main()
