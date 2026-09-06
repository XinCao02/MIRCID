#!/usr/bin/env python3
"""Replot the numerical HubmiRNet benchmark and PCA panels of Figure 2."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from mircid.paths import DATA_ROOT


DATA = DATA_ROOT / "figure_source_data" / "figure2"
OUTPUT = __file__.replace("make_figure2_data_panels.py", "figure2_data_panels")


def main() -> None:
    benchmark = pd.read_csv(DATA / "fig2b_published_values.csv").sort_values("display_order")
    observed = pd.read_csv(DATA / "hubmir414_ground_truth.csv", index_col=0)
    predicted = pd.read_csv(DATA / "hubmir414_predictions.csv", index_col=0)
    # mRNA barcodes retain the vial suffix (for example, ``01A``), whereas the
    # miRNA table stops at the sample-type code (``01``). TCGA patient and
    # sample-type fields are the shared biological identifier here.
    observed.index = observed.index.astype(str).str.slice(0, 15)
    predicted.index = predicted.index.astype(str).str.slice(0, 15)
    if observed.shape != predicted.shape or not observed.index.equals(predicted.index):
        raise ValueError("Observed and predicted HubmiR matrices are not aligned")

    joined = np.vstack([observed.to_numpy(), predicted.to_numpy()])
    embedding = PCA(n_components=2, random_state=0).fit_transform(joined)
    n = len(observed)

    plt.rcParams.update({"font.family": "Arial", "font.size": 7, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)
    colors = ["#B9C8D8"] * len(benchmark)
    colors[-1] = "#4E79A7"
    x = np.arange(len(benchmark))
    axes[0].bar(x, benchmark["pcc_percent"], yerr=benchmark["pcc_error"], color=colors, capsize=2)
    axes[0].set_ylabel("PCC (%)")
    axes[0].set_ylim(65, 91)
    axes[1].bar(x, benchmark["rmse"], yerr=benchmark["rmse_error"], color=colors, capsize=2)
    axes[1].set_ylabel("RMSE")
    axes[1].set_ylim(0.44, 0.78)
    for axis in axes[:2]:
        axis.set_xticks(x, benchmark["model"], rotation=35, ha="right")
        axis.spines[["top", "right"]].set_visible(False)
    axes[2].scatter(embedding[:n, 0], embedding[:n, 1], s=4, alpha=0.35, color="#E39C52", label="Observed")
    axes[2].scatter(embedding[n:, 0], embedding[n:, 1], s=4, alpha=0.35, color="#4E79A7", label="Predicted")
    axes[2].set(xlabel="PC1", ylabel="PC2")
    axes[2].legend(frameon=False, markerscale=2)
    axes[2].spines[["top", "right"]].set_visible(False)
    for label, axis in zip("bcd", axes):
        axis.text(-0.14, 1.06, label, transform=axis.transAxes, fontweight="bold", fontsize=9)
    for suffix in ("png", "pdf"):
        fig.savefig(f"{OUTPUT}.{suffix}", dpi=450 if suffix == "png" else None, bbox_inches="tight")


if __name__ == "__main__":
    main()
