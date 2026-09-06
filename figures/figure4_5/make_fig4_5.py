#!/usr/bin/env python3
"""Generate the Figure 4.5 HubmiR feature-analysis candidate."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from mircid.paths import DATA_ROOT


ROOT = Path(__file__).resolve().parent
DATA = DATA_ROOT / "figure_source_data" / "figure4_5"
OUTPUT = ROOT / "figure4_5"
SHOW_FOOTNOTE = True

# Seven evenly spaced splits from the frozen 20-split sequence, used only for
# the requested compact CKA/Ridge display. The rule is frozen in data/display_splits.csv.
DISPLAY_SPLITS = [7, 28, 49, 70, 91, 112, 140]

FEATURES = [
    "Gene",
    "Gene + PCA-414",
    "Gene + AE-414",
    "Gene + RP-414 (5-seed mean)",
    "HubmiR only",
    "Gene + HubmiR",
]
FEATURE_LABELS = ["Gene", "+PCA", "+AE", "+RP", "HubmiR\nonly", "Gene +\nHubmiR"]
FEATURE_COLORS = {
    "Gene": "#59A14F",
    "Gene + PCA-414": "#C7C9CC",
    "Gene + AE-414": "#C7C9CC",
    "Gene + RP-414 (5-seed mean)": "#C7C9CC",
    "HubmiR only": "#8DB8D8",
    "Gene + HubmiR": "#3F6F9F",
}

REPS = ["HubmiR", "PCA-414", "AE-414", "RP-414 mean"]
REP_LABELS = ["HubmiR", "PCA-414", "AE-414", "RP-414"]
REP_COLORS = {
    "HubmiR": "#4E79A7",
    "PCA-414": "#C7C9CC",
    "AE-414": "#C7C9CC",
    "RP-414 mean": "#C7C9CC",
}
LINE_STYLES = {
    "HubmiR": ("#4E79A7", "-", "o", 1.65),
    "PCA-414": ("#8A8D91", "--", "s", 0.90),
    "AE-414": ("#5F6368", ":", "^", 0.90),
    "RP-414 mean": ("#A7A9AC", "-.", "D", 0.90),
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.25,
            "axes.titlesize": 6.9,
            "axes.labelsize": 6.15,
            "xtick.labelsize": 5.45,
            "ytick.labelsize": 5.55,
            "legend.fontsize": 5.15,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def clean_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color="#D9DDE3", linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str, x: float = -0.15, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        fontweight="bold",
    )


def draw_box_with_points(
    ax: plt.Axes,
    values: np.ndarray,
    position: float,
    color: str,
    width: float = 0.58,
    facecolor: str | None = None,
    hatch: str | None = None,
    point_span: float = 0.18,
) -> None:
    values = np.asarray(values, dtype=float)
    box = ax.boxplot(
        [values],
        positions=[position],
        widths=width,
        patch_artist=True,
        showfliers=False,
        whis=(0, 100),
        medianprops={"color": "#202020", "linewidth": 0.85},
        whiskerprops={"color": "#555555", "linewidth": 0.65},
        capprops={"color": "#555555", "linewidth": 0.65},
    )
    box["boxes"][0].set(
        facecolor=color if facecolor is None else facecolor,
        edgecolor=color if facecolor == "white" else "#4A4A4A",
        linewidth=0.7,
        alpha=0.90,
        hatch=hatch,
    )
    offsets = np.linspace(-point_span, point_span, len(values))
    ax.scatter(
        position + offsets,
        values,
        s=9.5,
        facecolor="white",
        edgecolor=color,
        linewidth=0.55,
        zorder=3,
    )
    ax.errorbar(
        position,
        values.mean(),
        yerr=values.std(ddof=1),
        fmt="none",
        ecolor="#202020",
        elinewidth=0.72,
        capsize=1.8,
        capthick=0.72,
        zorder=3.5,
    )
    ax.scatter(
        [position],
        [values.mean()],
        marker="D",
        s=17,
        facecolor="#202020",
        edgecolor="white",
        linewidth=0.45,
        zorder=4,
    )


def collapse_rp_cca(frame: pd.DataFrame) -> pd.DataFrame:
    direct = frame[~frame["target_representation"].str.startswith("RP-414")].copy()
    rp = (
        frame[frame["target_representation"].str.startswith("RP-414")]
        .groupby(["split_seed", "source_space", "mode"], as_index=False)["test_correlation"]
        .mean()
    )
    rp["target_representation"] = "RP-414 mean"
    return pd.concat([direct, rp], ignore_index=True, sort=False)


def load_and_validate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    s1 = pd.read_csv(DATA / "s1_with_mironly_plot_metrics.csv")
    cka = pd.read_csv(DATA / "representation_geometry_tidy.csv")
    ridge = pd.read_csv(DATA / "ridge_split_metrics.csv")
    task = pd.read_csv(DATA / "selected_task_deltas.csv")
    cca = pd.read_csv(DATA / "cca_modes_heldout.csv")
    cca_summary = pd.read_csv(DATA / "cca_modes_summary.csv")

    s1 = s1[(s1["metric"] == "macro_f1") & s1["feature_space"].isin(FEATURES)].copy()
    assert set(s1["model"]) == {"RF", "ResNet"}
    assert s1.groupby(["model", "feature_space"]).size().eq(7).all()
    assert len(s1) == 84

    cka = cka[
        (cka["source_space"] == "Gene-29045")
        & cka["target_representation"].isin(REPS)
        & cka["method"].isin(["debiased_linear", "debiased_rbf"])
        & cka["split_seed"].isin(DISPLAY_SPLITS)
    ].copy()
    assert cka.groupby(["target_representation", "method"]).size().eq(7).all()
    assert len(cka) == 56

    ridge = ridge[
        (ridge["partition"] == "test")
        & ridge["source_space"].isin(["Gene-29045", "HubmiR-input-977"])
        & ridge["split_seed"].isin(DISPLAY_SPLITS)
    ].copy()
    assert ridge.groupby("source_space").size().eq(7).all()
    assert len(ridge) == 14

    expected_task_groups = {
        ("Gene + HubmiR − Gene", "RF"),
        ("Gene + HubmiR − Gene", "MLP"),
        ("Gene + HubmiR − Gene", "KAN"),
        ("Gene + HubmiR − Gene", "ResNet"),
        ("Gene + HubmiR − HubmiR only", "RF"),
        ("Gene + HubmiR − HubmiR only", "ResNet"),
    }
    assert set(zip(task["contrast"], task["model"])) == expected_task_groups
    assert task.groupby(["contrast", "model"]).size().eq(7).all()
    assert task["metric"].eq("macro_f1").all()
    assert len(task) == 42

    cca = cca[cca["source_space"].eq("Gene-29045")].copy()
    cca = collapse_rp_cca(cca)
    cca = cca[cca["target_representation"].isin(REPS)][
        ["split_seed", "target_representation", "mode", "test_correlation"]
    ]
    assert cca.groupby(["target_representation", "mode"])["split_seed"].nunique().eq(20).all()
    assert len(cca) == 800
    assert cca["test_correlation"].between(-1, 1).all()

    cca_summary = cca_summary[
        cca_summary["target_representation"].isin(REPS)
        & cca_summary["mode"].between(1, 10)
    ].copy()
    assert len(cca_summary) == 40
    assert cca_summary["n_splits"].eq(20).all()
    raw_means = (
        cca.groupby(["target_representation", "mode"], as_index=False)["test_correlation"]
        .mean()
        .rename(columns={"test_correlation": "raw_mean"})
    )
    checked = cca_summary.merge(raw_means, on=["target_representation", "mode"], validate="one_to_one")
    assert np.allclose(checked["mean"], checked["raw_mean"], rtol=0, atol=1e-12)

    return s1, cca_summary, cka, ridge, task


def plot_panel_a(ax: plt.Axes, data: pd.DataFrame, model: str, show_ylabel: bool) -> None:
    subset = data[data["model"] == model]
    for i, feature in enumerate(FEATURES):
        values = subset[subset["feature_space"] == feature]["value"].to_numpy()
        draw_box_with_points(ax, values, i, FEATURE_COLORS[feature], width=0.56, point_span=0.16)
    ax.set_xlim(-0.62, len(FEATURES) - 0.38)
    ax.set_ylim(0.43, 0.80)
    ax.set_yticks([0.45, 0.55, 0.65, 0.75])
    ax.set_xticks(range(len(FEATURES)), FEATURE_LABELS)
    ax.set_title(model, loc="center", fontweight="bold", pad=2)
    if show_ylabel:
        ax.set_ylabel("Held-out macro-F1")
    else:
        ax.tick_params(axis="y", labelleft=False)
    clean_axis(ax)


def plot_panel_b(ax: plt.Axes, data: pd.DataFrame) -> None:
    for representation in REPS:
        summary = data[data["target_representation"] == representation].sort_values("mode")
        x = summary["mode"].to_numpy(dtype=float)
        mean = summary["mean"].to_numpy(dtype=float)
        low = summary["ci95_low"].to_numpy(dtype=float)
        high = summary["ci95_high"].to_numpy(dtype=float)
        color, linestyle, marker, linewidth = LINE_STYLES[representation]
        ax.plot(
            x,
            mean,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=2.8,
            linewidth=linewidth,
            label=representation,
            zorder=3 if representation == "HubmiR" else 2,
        )
        ax.fill_between(
            x,
            low,
            high,
            color=color,
            alpha=0.13 if representation == "HubmiR" else 0.06,
            linewidth=0,
        )
    ax.set_xlim(0.7, 10.3)
    ax.set_ylim(0.68, 1.005)
    ax.set_xticks([1, 3, 5, 7, 9])
    ax.set_yticks([0.70, 0.80, 0.90, 1.00])
    ax.set_xlabel("Canonical mode")
    ax.set_ylabel("Held-out correlation")
    ax.set_title("Regularized CCA", loc="left", fontweight="bold")
    clean_axis(ax, grid_axis="both")
    ax.legend(frameon=False, loc="lower left", ncol=2, columnspacing=0.65, handlelength=1.9)
    panel_label(ax, "b", x=-0.17)


def plot_panel_c(ax: plt.Axes, data: pd.DataFrame) -> None:
    offset = 0.19
    for i, rep in enumerate(REPS):
        for method, dx in [("debiased_linear", -offset), ("debiased_rbf", offset)]:
            values = data[
                (data["target_representation"] == rep) & (data["method"] == method)
            ]["cka"].to_numpy()
            draw_box_with_points(
                ax,
                values,
                i + dx,
                REP_COLORS[rep],
                width=0.30,
                facecolor=None if method == "debiased_linear" else "white",
                hatch=None if method == "debiased_linear" else "////",
                point_span=0.085,
            )
    ax.set_xlim(-0.58, 3.58)
    ax.set_ylim(0.0, 0.90)
    ax.set_yticks([0.0, 0.3, 0.6, 0.9])
    ax.set_xticks(
        range(4),
        REP_LABELS,
        rotation=20,
        rotation_mode="anchor",
        ha="right",
    )
    ax.set_ylabel("CKA with full Gene")
    ax.set_title("Representation geometry", loc="left", fontweight="bold")
    clean_axis(ax)
    ax.legend(
        handles=[
            Patch(facecolor="#9EA4AA", edgecolor="#666666", label="Linear"),
            Patch(facecolor="white", edgecolor="#666666", hatch="////", label="RBF"),
        ],
        frameon=False,
        loc="upper left",
        ncol=2,
        handlelength=1.5,
        columnspacing=0.9,
        borderpad=0.1,
    )
    panel_label(ax, "c", x=-0.12)


def plot_panel_d(ax: plt.Axes, data: pd.DataFrame) -> None:
    sources = ["Gene-29045", "HubmiR-input-977"]
    labels = ["Full Gene", "977-gene\ninput"]
    colors = ["#59A14F", "#8DB8D8"]
    for i, (source, color) in enumerate(zip(sources, colors)):
        values = data[data["source_space"] == source]["variance_weighted_r2"].to_numpy()
        draw_box_with_points(ax, values, i, color, width=0.55, point_span=0.14)
        ax.text(i, 0.215, f"{values.mean():.2f}", ha="center", va="bottom", fontsize=5.25)
    ax.set_xlim(-0.62, 1.62)
    ax.set_ylim(0.20, 0.60)
    ax.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6])
    ax.set_xticks(range(2), labels)
    ax.set_ylabel("Held-out weighted R²")
    ax.set_title("Linear HubmiR recovery", loc="left", fontweight="bold")
    clean_axis(ax)
    panel_label(ax, "d", x=-0.17)


def plot_panel_e(ax: plt.Axes, data: pd.DataFrame) -> None:
    rows = [
        ("Gene + HubmiR − Gene", "RF", 5.0),
        ("Gene + HubmiR − Gene", "MLP", 4.0),
        ("Gene + HubmiR − Gene", "KAN", 3.0),
        ("Gene + HubmiR − Gene", "ResNet", 2.0),
        ("Gene + HubmiR − HubmiR only", "RF", 0.5),
        ("Gene + HubmiR − HubmiR only", "ResNet", -0.5),
    ]
    family_colors = {
        "Gene + HubmiR − Gene": "#4E79A7",
        "Gene + HubmiR − HubmiR only": "#59A14F",
    }
    for contrast, model, y in rows:
        values = (
            data[(data["contrast"] == contrast) & (data["model"] == model)]
            .sort_values("split_seed")["delta"]
            .to_numpy(dtype=float)
        )
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
    ax.axhline(1.25, color="#D0D3D8", linewidth=0.75)
    ax.set_xlim(-0.11, 0.19)
    ax.set_xticks([-0.10, -0.05, 0.00, 0.05, 0.10, 0.15])
    ax.set_ylim(-1.15, 5.75)
    ax.set_yticks([row[2] for row in rows], [row[1] for row in rows])
    ax.set_xlabel("Paired Δmacro-F1")
    ax.set_title("Task utility across models", loc="left", fontweight="bold")
    clean_axis(ax, grid_axis="x")
    ax.text(
        0.02,
        0.975,
        "Gene + HubmiR − Gene Only",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.75,
        fontweight="bold",
        color=family_colors["Gene + HubmiR − Gene"],
    )
    ax.text(
        0.02,
        0.30,
        "Gene + HubmiR − HubmiR only",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.75,
        fontweight="bold",
        color=family_colors["Gene + HubmiR − HubmiR only"],
    )
    panel_label(ax, "e", x=-0.15)


def main() -> None:
    set_style()
    s1, cca, cka, ridge, task = load_and_validate()

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

    plot_panel_a(ax_a1, s1, "RF", show_ylabel=True)
    plot_panel_a(ax_a2, s1, "ResNet", show_ylabel=False)
    panel_label(ax_a1, "a", x=-0.13, y=1.24)
    fig.text(
        0.065,
        0.867,
        "Embedding-control task performance",
        ha="left",
        va="bottom",
        fontsize=7.1,
        fontweight="bold",
    )
    plot_panel_b(ax_b, cca)
    plot_panel_c(ax_c, cka)
    plot_panel_d(ax_d, ridge)
    plot_panel_e(ax_e, task)

    legend_handles = [
        Patch(facecolor=FEATURE_COLORS[f], edgecolor="#555555", linewidth=0.55, label=l)
        for f, l in zip(FEATURES, ["Gene", "+PCA", "+AE", "+RP", "HubmiR only", "Gene + HubmiR"])
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
    fig.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.063, 0.927),
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
    if SHOW_FOOTNOTE:
        fig.text(
            0.065,
            0.032,
            "a,e, n = 7 held-out seeds per group; b, n = 20 frozen splits; c,d display seven evenly spaced frozen splits.\nBoxes show median/IQR and min–max; diamonds and bars show mean ± 1 SD; lines in e show min–max; bands in b show bootstrap 95% CIs.",
            ha="left",
            va="bottom",
            fontsize=5.15,
            color="#4F5256",
        )

    fig.savefig(Path(f"{OUTPUT}.png"), dpi=600, facecolor="white")
    fig.savefig(Path(f"{OUTPUT}.pdf"), facecolor="white")
    fig.savefig(Path(f"{OUTPUT}.svg"), facecolor="white")
    fig.savefig(
        Path(f"{OUTPUT}.tiff"),
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
