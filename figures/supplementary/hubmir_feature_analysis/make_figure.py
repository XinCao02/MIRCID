#!/usr/bin/env python3
"""Render the active four-panel supplementary HubmiR feature-analysis figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd

import make_supplementary_figures_v2 as base
from mircid.paths import DATA_ROOT


PACKAGE = Path(__file__).resolve().parent
DATA = DATA_ROOT / "figure_source_data" / "supplementary" / "hubmir_feature_analysis"
FIGURES = PACKAGE
OUTPUT = FIGURES / "FigureS_HubmiRFeatureAnalysis"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6.25,
        "axes.titlesize": 7.0,
        "axes.labelsize": 6.6,
        "xtick.labelsize": 5.8,
        "ytick.labelsize": 5.8,
        "legend.fontsize": 5.6,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def replace_header(
    ax: plt.Axes,
    old_letter: str,
    old_title: str,
    new_letter: str,
    new_title: str,
    *,
    title_x: float = 0.065,
) -> None:
    """Replace inherited panel headers while retaining the audited plot body."""
    for artist in list(ax.texts):
        if artist.get_text() in {old_letter, old_title}:
            artist.remove()
    base.panel_header(ax, new_letter, new_title, title_x=title_x)


def save_all(fig: plt.Figure) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".pdf"), facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".svg"), facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(
        OUTPUT.with_suffix(".tiff"),
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def main() -> None:
    base.set_style()
    accuracy = pd.read_csv(DATA / "FigS1a_accuracy_display.csv")
    ae = pd.read_csv(DATA / "FigS1d_autoencoder_learning_curve.csv")
    ridge = pd.read_csv(DATA / "FigS2b_ridge_full20.csv")
    per_mirna = pd.read_csv(DATA / "FigS2c_per_mirna_ridge.csv")

    fig = plt.figure(figsize=(7.2, 4.85), facecolor="white")
    outer = fig.add_gridspec(
        2,
        48,
        left=0.082,
        right=0.988,
        bottom=0.095,
        top=0.885,
        height_ratios=[1.0, 1.0],
        hspace=0.52,
        wspace=0.90,
    )

    ax_a = fig.add_subplot(outer[0, :30])
    ax_b = fig.add_subplot(outer[1, :27])
    ax_c = fig.add_subplot(outer[0, 32:])
    ax_d = fig.add_subplot(outer[1, 30:])

    base.draw_accuracy_panel(ax_a, accuracy)
    ax_a.set_ylim(0.58, 0.86)
    for artist in ax_a.texts:
        if artist.get_text() in {"RF", "ResNet"}:
            artist.set_y(0.915)

    base.draw_ae_panel(ax_b, ae, letter="b")
    base.draw_per_mirna_panel(ax_c, per_mirna)
    replace_header(
        ax_c,
        "c",
        "Per-miRNA recoverability",
        "c",
        "Per-miRNA recoverability",
        title_x=0.105,
    )
    base.draw_ridge_summary_panel(ax_d, ridge)
    replace_header(
        ax_d,
        "b",
        "Cross-validated linear recovery",
        "d",
        "Linear HubmiR recovery",
        title_x=0.085,
    )

    feature_handles = [
        Patch(
            facecolor=base.FEATURE_COLORS[feature],
            edgecolor="#555555",
            linewidth=0.55,
            label=label,
        )
        for feature, label in zip(
            base.FEATURES,
            ["Gene", "+PCA", "+AE", "+RP mean", "+HubmiR"],
        )
    ]
    feature_handles.append(
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            color=base.COLORS["Ink"],
            markersize=3.5,
            label="Mean ± 1 SD",
        )
    )
    fig.legend(
        handles=feature_handles,
        loc="upper center",
        bbox_to_anchor=(0.37, 0.990),
        ncol=6,
        frameon=False,
        handlelength=1.25,
        columnspacing=0.82,
        handletextpad=0.32,
    )
    save_all(fig)
    plt.close(fig)
    print("PASS: rendered the active four-panel supplementary HubmiR figure")


if __name__ == "__main__":
    main()
