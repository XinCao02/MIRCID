#!/usr/bin/env python3
"""Render Figure 4.5 v3 with a complete four-model panel e and aligned headers."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

import make_fig4_5_v2 as v2


# Keep the wrapper independently auditable; the imported base applies the same
# settings again before rendering.
mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6.25,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


base = v2.base
base.OUTPUT = base.ROOT / "figure4_5"
base.SHOW_FOOTNOTE = False


def panel_header(
    ax: plt.Axes,
    letter: str,
    title: str,
    *,
    y: float = 1.075,
    title_x: float = 0.065,
) -> None:
    """Place the panel letter and short description on one visual baseline."""
    ax.text(
        0.0,
        y,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="baseline",
        fontsize=8.6,
        fontweight="bold",
        clip_on=False,
    )
    ax.text(
        title_x,
        y,
        title,
        transform=ax.transAxes,
        ha="left",
        va="baseline",
        fontsize=6.9,
        fontweight="bold",
        clip_on=False,
    )


def remove_inherited_header(ax: plt.Axes, letter: str) -> None:
    """Remove the base script's detached panel letter and left title."""
    for artist in list(ax.texts):
        if artist.get_text() == letter:
            artist.remove()
    ax.set_title("", loc="left")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    s1, cca, cka, ridge, _ = base.load_and_validate()
    task = pd.read_csv(base.DATA / "selected_task_deltas_v3.csv")
    models = ["RF", "MLP", "KAN", "ResNet"]
    contrasts = ["Gene + HubmiR − Gene", "Gene + HubmiR − HubmiR only"]
    expected = {(contrast, model) for contrast in contrasts for model in models}
    observed = set(zip(task["contrast"], task["model"]))
    if observed != expected:
        raise ValueError(f"Incomplete v3 panel-e model/contrast grid: {observed}")
    if len(task) != 56 or not task.groupby(["contrast", "model"]).size().eq(7).all():
        raise ValueError("Panel e must contain seven paired seeds for all eight groups")
    if not task["metric"].eq("macro_f1").all() or not np.isfinite(task["delta"]).all():
        raise ValueError("Panel e contains invalid metric rows")
    return s1, cca, cka, ridge, task


def plot_panel_e_v3(ax: plt.Axes, data: pd.DataFrame) -> None:
    models = ["RF", "MLP", "KAN", "ResNet"]
    contrast_gene = "Gene + HubmiR − Gene"
    contrast_mir = "Gene + HubmiR − HubmiR only"
    rows = [
        *[(contrast_gene, model, float(7 - idx)) for idx, model in enumerate(models)],
        *[(contrast_mir, model, float(2 - idx)) for idx, model in enumerate(models)],
    ]
    family_colors = {contrast_gene: "#4E79A7", contrast_mir: "#59A14F"}

    for contrast, model, y in rows:
        values = (
            data[data["contrast"].eq(contrast) & data["model"].eq(model)]
            .sort_values("split_seed")["delta"]
            .to_numpy(dtype=float)
        )
        if len(values) != 7:
            raise ValueError(f"Expected seven paired values for {contrast}, {model}")
        color = family_colors[contrast]
        ax.plot([values.min(), values.max()], [y, y], color=color, linewidth=1.0, zorder=2)
        y_offsets = np.linspace(-0.12, 0.12, len(values))
        ax.scatter(
            values,
            y + y_offsets,
            s=9.5,
            facecolor="white",
            edgecolor=color,
            linewidth=0.55,
            zorder=3,
        )
        ax.scatter(
            [values.mean()],
            [y],
            marker="D",
            s=19,
            facecolor=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=4,
        )

    ax.axvline(0, color="#202020", linewidth=0.75, zorder=1)
    ax.axhline(3.0, color="#D0D3D8", linewidth=0.75)
    ax.set_xlim(-0.06, 0.22)
    ax.set_xticks([-0.05, 0.00, 0.05, 0.10, 0.15, 0.20])
    ax.set_ylim(-1.65, 7.80)
    ax.set_yticks([row[2] for row in rows], [row[1] for row in rows])
    ax.set_xlabel("Paired Δmacro-F1")
    base.clean_axis(ax, grid_axis="x")
    ax.text(
        0.02,
        0.972,
        "Gene + HubmiR − Gene",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.7,
        fontweight="bold",
        color=family_colors[contrast_gene],
    )
    ax.text(
        0.02,
        0.455,
        "Gene + HubmiR − HubmiR only",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.7,
        fontweight="bold",
        color=family_colors[contrast_mir],
    )


def main() -> None:
    base.set_style()
    s1, cca, cka, ridge, task = load_data()

    fig = plt.figure(figsize=(7.2, 6.05), facecolor="white")
    outer = fig.add_gridspec(
        2,
        48,
        height_ratios=[1.02, 1.0],
        left=0.065,
        right=0.993,
        bottom=0.10,
        top=0.845,
        hspace=0.38,
        wspace=0.95,
    )
    top_left = outer[0, :32].subgridspec(1, 2, wspace=0.04)
    ax_a1 = fig.add_subplot(top_left[0, 0])
    ax_a2 = fig.add_subplot(top_left[0, 1], sharey=ax_a1)
    ax_b = fig.add_subplot(outer[0, 33:])
    ax_c = fig.add_subplot(outer[1, :20])
    ax_d = fig.add_subplot(outer[1, 21:30])
    ax_e = fig.add_subplot(outer[1, 32:])

    base.plot_panel_a(ax_a1, s1, "RF", show_ylabel=True)
    base.plot_panel_a(ax_a2, s1, "ResNet", show_ylabel=False)
    panel_header(ax_a1, "a", "Embedding-control task performance", y=1.075, title_x=0.055)

    base.plot_panel_b(ax_b, cca)
    remove_inherited_header(ax_b, "b")
    panel_header(ax_b, "b", "Regularized CCA", y=1.075, title_x=0.17)

    base.plot_panel_c(ax_c, cka)
    remove_inherited_header(ax_c, "c")
    panel_header(ax_c, "c", "Representation geometry", y=1.075, title_x=0.075)

    base.plot_panel_d(ax_d, ridge)
    remove_inherited_header(ax_d, "d")
    panel_header(ax_d, "d", "Linear HubmiR recovery", y=1.075, title_x=0.17)

    plot_panel_e_v3(ax_e, task)
    panel_header(ax_e, "e", "Task utility across models", y=1.075, title_x=0.095)

    legend_handles = [
        Patch(facecolor=base.FEATURE_COLORS[feature], edgecolor="#555555", linewidth=0.55, label=label)
        for feature, label in zip(
            base.FEATURES,
            ["Gene", "+PCA", "+AE", "+RP", "HubmiR only", "Gene + HubmiR"],
        )
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markersize=3.4,
            markerfacecolor="#202020",
            markeredgecolor="white",
            label="Mean",
        )
    )
    # Center the shared representation key above panel a, detached from the
    # panel letter/header and below the figure-level title.
    fig.legend(
        handles=legend_handles,
        loc="center",
        bbox_to_anchor=(0.355, 0.905),
        ncol=7,
        frameon=False,
        columnspacing=0.85,
        handlelength=1.25,
        handletextpad=0.35,
    )
    fig.suptitle(
        "Fig. 4.5 | HubmiR defines a distinct, partially gene-recoverable regulatory representation",
        x=0.065,
        y=0.982,
        ha="left",
        fontsize=8.65,
        fontweight="bold",
    )

    output = Path(f"{base.OUTPUT}")
    fig.savefig(Path(f"{output}.png"), dpi=600, facecolor="white")
    fig.savefig(Path(f"{output}.pdf"), facecolor="white")
    fig.savefig(Path(f"{output}.svg"), facecolor="white")
    fig.savefig(
        Path(f"{output}.tiff"),
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
