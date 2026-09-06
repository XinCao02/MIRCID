#!/usr/bin/env python3
"""Render the Figure 4.5 v2 candidate with the Figure 4 color language."""

from __future__ import annotations

import make_fig4_5 as base
from matplotlib.patches import Patch


base.OUTPUT = base.ROOT / "Fig4.5_HubmiRFeatureAnalysis_v2"
base.SHOW_FOOTNOTE = False

# Figure 4 anchors: Gene green and Gene + HubmiR blue. Generic embedding
# controls share one desaturated pale-gold family derived from Figure 4's
# orange/TFA accent; line style and marker, rather than a false light-to-dark
# ordering, distinguish PCA, AE and RP.
base.FEATURE_COLORS.update(
    {
        "Gene": "#59A14F",
        "Gene + PCA-414": "#EFD39A",
        "Gene + AE-414": "#EFD39A",
        "Gene + RP-414 (5-seed mean)": "#EFD39A",
        "HubmiR only": "#8DB8D8",
        "Gene + HubmiR": "#4E79A7",
    }
)

base.REP_COLORS.update(
    {
        "HubmiR": "#4E79A7",
        "PCA-414": "#EFD39A",
        "AE-414": "#EFD39A",
        "RP-414 mean": "#EFD39A",
    }
)

# A darker companion keeps the warm controls legible as lines; the existing
# dash patterns and marker shapes carry their identities in colour and grey.
base.LINE_STYLES.update(
    {
        "HubmiR": ("#4E79A7", "-", "o", 1.65),
        "PCA-414": ("#B8893C", "--", "s", 0.90),
        "AE-414": ("#B8893C", ":", "^", 0.90),
        "RP-414 mean": ("#B8893C", "-.", "D", 0.90),
    }
)


# Small final-layout refinements requested for the v2 candidate only.
_plot_panel_b = base.plot_panel_b
_plot_panel_c = base.plot_panel_c
_plot_panel_d = base.plot_panel_d


def plot_panel_b_v2(ax, data) -> None:
    _plot_panel_b(ax, data)
    ax.set_ylabel("Held-out correlation", labelpad=1.5)


def plot_panel_c_v2(ax, data) -> None:
    _plot_panel_c(ax, data)
    ax.set_xticks(range(4), base.REP_LABELS, rotation=0, ha="center")
    ax.legend(
        handles=[
            Patch(facecolor="#EFD39A", edgecolor="#B8893C", label="Linear"),
            Patch(
                facecolor="#FFF7E7",
                edgecolor="#B8893C",
                hatch="////",
                label="RBF",
            ),
        ],
        frameon=False,
        loc="upper left",
        ncol=2,
        handlelength=1.5,
        columnspacing=0.9,
        borderpad=0.1,
    )


def plot_panel_d_v2(ax, data) -> None:
    _plot_panel_d(ax, data)
    ax.set_ylabel("Held-out weighted R²", labelpad=1.5)


base.plot_panel_b = plot_panel_b_v2
base.plot_panel_c = plot_panel_c_v2
base.plot_panel_d = plot_panel_d_v2


if __name__ == "__main__":
    base.main()
