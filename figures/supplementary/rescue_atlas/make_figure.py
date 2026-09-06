#!/usr/bin/env python3
"""Replot the four-panel supplementary rescue atlas from frozen source tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mircid.paths import DATA_ROOT


HERE = Path(__file__).resolve().parent
DATA = DATA_ROOT / "figure_source_data" / "supplementary" / "rescue_atlas"
OUTPUT = HERE / "figureS_rescue_atlas_replot"


def main() -> None:
    matrix = pd.read_csv(DATA / "panel_a_transition_matrix.csv")
    pathway = pd.read_csv(DATA / "panel_b_pathway_rescue_harm_counts.csv")
    catalogue = pd.read_csv(DATA / "panel_c_selected_rescue_profiles.csv")
    heat = pd.read_csv(DATA / "panel_d_selected_heatmap_source.csv")
    plt.rcParams.update({"font.family": "Arial", "font.size": 6.1, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.2, 7.1), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=[0.82, 1.35], width_ratios=[0.82, 1.18],
                            left=0.08, right=0.98, bottom=0.07, top=0.95, hspace=0.34, wspace=0.30)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]

    # Panel a source is already a tidy 2 × 2 transition table.
    a = matrix.pivot(index="gene_correct", columns="hubmir_correct", values="count")
    sns.heatmap(a, annot=True, fmt="g", cmap="Blues", cbar=False, ax=axes[0], linewidths=1, linecolor="white")
    axes[0].set_title("Prediction-state transitions", loc="left", fontweight="bold")

    x = np.arange(len(pathway))
    rescued_col = next(column for column in pathway if column.lower() == "rescued")
    harmed_col = next(column for column in pathway if column.lower() == "harmed")
    label_col = next(column for column in pathway if "pathway" in column.lower())
    axes[1].bar(x - 0.18, pathway[rescued_col], width=0.36, color="#59A14F", label="Rescued")
    axes[1].bar(x + 0.18, pathway[harmed_col], width=0.36, color="#B64B47", label="Harmed")
    axes[1].set_xticks(x, pathway[label_col], rotation=40, ha="right")
    axes[1].set_ylabel("Test-instance count")
    axes[1].set_title("Rescue and harm across pathways", loc="left", fontweight="bold")
    axes[1].legend(frameon=False)

    axes[2].axis("off")
    shown = catalogue.iloc[: min(12, len(catalogue)), : min(4, catalogue.shape[1])]
    table = axes[2].table(cellText=shown.values, colLabels=shown.columns, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(4.5)
    table.scale(1, 1.22)
    axes[2].set_title("Selected rescue-case catalogue", loc="left", fontweight="bold")

    if {"sample_id", "feature", "value"}.issubset(heat.columns):
        d = heat.pivot(index="sample_id", columns="feature", values="value")
    else:
        value_col = "mean_ecdf_deviation" if "mean_ecdf_deviation" in heat.columns else heat.select_dtypes("number").columns[-1]
        d = heat.pivot(index="sample_id", columns="feature", values=value_col)
    sns.heatmap(d, cmap="vlag", center=0, ax=axes[3], cbar_kws={"label": "Standardized feature value"})
    axes[3].set(xlabel="", ylabel="")
    axes[3].set_title("Representative pathway feature profiles", loc="left", fontweight="bold")
    axes[3].tick_params(axis="x", rotation=45)
    for label, axis in zip("abcd", axes):
        axis.text(-0.12, 1.08, label, transform=axis.transAxes, fontweight="bold", fontsize=8.5)
    for axis in axes[1:2]:
        axis.spines[["top", "right"]].set_visible(False)
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT.with_suffix(f".{suffix}"), dpi=450 if suffix == "png" else None, bbox_inches="tight")


if __name__ == "__main__":
    main()
