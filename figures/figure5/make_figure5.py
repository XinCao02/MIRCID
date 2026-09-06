#!/usr/bin/env python3
"""Rebuild the Figure 5 composite from its frozen mechanism image and heatmap table."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image

from mircid.paths import DATA_ROOT


HERE = Path(__file__).resolve().parent
DATA = DATA_ROOT / "figure_source_data" / "figure5"
OUTPUT = HERE / "figure5_replot"


def main() -> None:
    source = pd.read_csv(DATA / "Figure5_RescueCases_featurewise_normalized_v3_source_data.csv")
    source_image = Image.open(HERE / "upper_mechanism_source.png").convert("RGB")
    pathways = (
        source[["pathway", "pathway_order"]]
        .drop_duplicates()
        .sort_values("pathway_order")["pathway"]
        .tolist()
    )
    plt.rcParams.update({"font.family": "Arial", "font.size": 6.2, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.2, 8.8), facecolor="white")
    grid = fig.add_gridspec(4, 2, height_ratios=[5.0, 0.76, 0.76, 0.76], width_ratios=[1, 0.025],
                            left=0.19, right=0.96, bottom=0.06, top=0.985, hspace=0.30, wspace=0.05)
    upper = fig.add_subplot(grid[0, :])
    upper.imshow(source_image)
    upper.axis("off")

    for row, pathway in enumerate(pathways, start=1):
        axis = fig.add_subplot(grid[row, 0])
        subset = source[source.pathway == pathway].sort_values(["sample_order", "feature_order"])
        samples = subset[["sample_id", "sample_order", "display_label_v3"]].drop_duplicates().sort_values("sample_order")
        features = subset[["feature", "feature_order", "modality"]].drop_duplicates().sort_values("feature_order")
        table = subset.pivot(index="sample_id", columns="feature", values="mean_ecdf_deviation")
        table = table.loc[samples.sample_id, features.feature]
        cbar_axis = fig.add_subplot(grid[row, 1]) if row == 2 else None
        sns.heatmap(table, ax=axis, cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f",
                    annot_kws={"fontsize": 4.3}, linewidths=0.45, linecolor="white",
                    cbar=cbar_axis is not None, cbar_ax=cbar_axis)
        axis.set(xlabel="", ylabel="")
        axis.set_yticklabels(samples.display_label_v3, rotation=0, fontweight="bold")
        axis.set_xticklabels(features.feature, rotation=32, ha="left", rotation_mode="anchor")
        for tick, modality in zip(axis.get_xticklabels(), features.modality):
            tick.set_color("#B33A3A" if modality == "Gene" else "#2D7F4F")
        axis.set_title(f"{pathway} pathway", loc="left", fontweight="bold", pad=5)
        axis.tick_params(length=0, pad=1)
        if cbar_axis is not None:
            cbar_axis.set_ylabel("Within-feature ECDF deviation", fontsize=5.6)
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT.with_suffix(f".{suffix}"), dpi=450 if suffix == "png" else None, bbox_inches="tight")


if __name__ == "__main__":
    main()
