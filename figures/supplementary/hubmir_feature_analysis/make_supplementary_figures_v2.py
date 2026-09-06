#!/usr/bin/env python3
"""Render three Nature-style supplementary figures from frozen aligned results."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyBboxPatch
import numpy as np
import pandas as pd

from mircid.paths import DATA_ROOT


PACKAGE = Path(__file__).resolve().parent
DATA = DATA_ROOT / "figure_source_data" / "supplementary" / "hubmir_feature_analysis"
FIGURES = PACKAGE

COLORS = {
    "Gene": "#59A14F",
    "HubmiR": "#4E79A7",
    "HubmiR light": "#8FB7D1",
    "PCA": "#F0D69A",
    "AE": "#E7C47C",
    "RP": "#DAB05D",
    "SVM": "#8B8178",
    "ResNet": "#4E79A7",
    "Residual": "#C76B73",
    "Ink": "#242424",
    "Grid": "#DDE1E6",
    "Muted": "#6A6A6A",
}

FEATURES = [
    "Gene",
    "Gene + PCA-414",
    "Gene + AE-414",
    "Gene + RP-414 (5-seed mean)",
    "Gene + HubmiR",
]
FEATURE_LABELS = ["Gene", "PCA", "AE", "RP", "HubmiR"]
FEATURE_COLORS = {
    "Gene": COLORS["Gene"],
    "Gene + PCA-414": COLORS["PCA"],
    "Gene + AE-414": COLORS["AE"],
    "Gene + RP-414 (5-seed mean)": COLORS["RP"],
    "Gene + HubmiR": COLORS["HubmiR"],
}


def set_style() -> None:
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
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def panel_header(ax: plt.Axes, letter: str, title: str, title_x: float = 0.08) -> None:
    ax.text(
        0.0,
        1.07,
        letter,
        transform=ax.transAxes,
        fontsize=8.6,
        fontweight="bold",
        va="baseline",
        clip_on=False,
    )
    ax.text(
        title_x,
        1.07,
        title,
        transform=ax.transAxes,
        fontsize=7.0,
        fontweight="bold",
        va="baseline",
        clip_on=False,
    )


def clean_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=COLORS["Grid"], linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(COLORS["Ink"])
        ax.spines[spine].set_linewidth(0.75)


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    output = FIGURES / stem
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    fig.savefig(output.with_suffix(".svg"), facecolor="white")
    fig.savefig(output.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(
        output.with_suffix(".tiff"),
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def mean_sd(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    require(len(values) > 1 and np.isfinite(values).all(), "Invalid replicate vector")
    return float(values.mean()), float(values.std(ddof=1))


def draw_accuracy_panel(ax: plt.Axes, data: pd.DataFrame) -> None:
    models = ["RF", "ResNet"]
    width = 0.64
    positions: list[float] = []
    labels: list[str] = []
    for model_index, model in enumerate(models):
        base = model_index * 6.2
        for feature_index, (feature, label) in enumerate(zip(FEATURES, FEATURE_LABELS)):
            values = data[
                data["model"].eq(model) & data["feature_space"].eq(feature)
            ]["value"].to_numpy(dtype=float)
            require(len(values) == 7, f"Expected seven Accuracy values for {model}, {feature}")
            position = base + feature_index
            box = ax.boxplot(
                values,
                positions=[position],
                widths=width,
                patch_artist=True,
                showfliers=False,
                whis=(0, 100),
                medianprops={"color": COLORS["Ink"], "linewidth": 0.95},
                whiskerprops={"color": "#555555", "linewidth": 0.75},
                capprops={"color": "#555555", "linewidth": 0.75},
                boxprops={"edgecolor": "#555555", "linewidth": 0.7},
            )
            box["boxes"][0].set_facecolor(FEATURE_COLORS[feature])
            box["boxes"][0].set_alpha(0.88)
            offsets = np.linspace(-0.13, 0.13, len(values))
            ax.scatter(
                position + offsets,
                values,
                s=10.5,
                facecolor="white",
                edgecolor=FEATURE_COLORS[feature],
                linewidth=0.6,
                zorder=3,
            )
            mean, sd = mean_sd(values)
            ax.errorbar(
                position,
                mean,
                yerr=sd,
                fmt="D",
                ms=3.3,
                color=COLORS["Ink"],
                markeredgecolor="white",
                markeredgewidth=0.45,
                capsize=2.0,
                linewidth=0.75,
                zorder=4,
            )
            positions.append(position)
            labels.append(label)
        ax.text(
            base + 2,
            1.005,
            model,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=6.8,
            fontweight="bold",
        )
    ax.axvline(5.1, color="#C9CDD2", linewidth=0.7)
    ax.set_xticks(positions, labels)
    ax.set_xlim(-0.8, 11.0)
    ax.set_ylim(0.58, 0.84)
    ax.set_ylabel("Held-out accuracy")
    clean_axis(ax)
    panel_header(ax, "a", "Accuracy across embedding controls", title_x=0.055)


def draw_embedding_contrast_panel(ax: plt.Axes, data: pd.DataFrame) -> None:
    comparisons = [
        ("Gene + HubmiR − Gene", "Gene"),
        ("Gene + HubmiR − Gene + PCA-414", "PCA-414"),
        ("Gene + HubmiR − Gene + AE-414", "AE-414"),
        ("Gene + HubmiR − Gene + RP-414 (5-seed mean)", "RP mean"),
    ]
    models = ["SVM", "ResNet"]
    offsets = {"SVM": -0.12, "ResNet": 0.12}
    for model in models:
        sub = data[data["model"].eq(model)].set_index("contrast")
        for index, (contrast, _) in enumerate(comparisons):
            row = sub.loc[contrast]
            y = index + offsets[model]
            lower = float(row["mean_delta"] - row["ci95_low"])
            upper = float(row["ci95_high"] - row["mean_delta"])
            ax.errorbar(
                float(row["mean_delta"]),
                y,
                xerr=np.array([[lower], [upper]]),
                fmt="o" if model == "SVM" else "s",
                color=COLORS[model],
                markerfacecolor=COLORS[model],
                markeredgecolor="white",
                markeredgewidth=0.45,
                markersize=4.2,
                linewidth=1.0,
                capsize=2.0,
                zorder=3,
            )
    ax.axvline(0, color=COLORS["Ink"], linewidth=0.75)
    ax.set_yticks(range(len(comparisons)), [entry[1] for entry in comparisons])
    ax.set_ylim(3.85, -0.45)
    ax.set_xlabel("Paired Δmacro-F1 (HubmiR − comparator)")
    clean_axis(ax, grid_axis="x")
    panel_header(ax, "b", "Full-20 paired contrasts", title_x=0.07)
    handles = [
        Line2D([0], [0], marker="o", color=COLORS["SVM"], linestyle="none", label="SVM"),
        Line2D([0], [0], marker="s", color=COLORS["ResNet"], linestyle="none", label="ResNet"),
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(1.0, -0.015),
        ncol=2,
        handletextpad=0.3,
        columnspacing=0.8,
    )


def draw_rp_panel(ax: plt.Axes, rp: pd.DataFrame, refs: pd.DataFrame) -> None:
    rp_order = [101, 202, 303, 404, 505]
    categories = [f"RP {seed}" for seed in rp_order] + ["Gene", "HubmiR"]
    x = np.arange(len(categories), dtype=float)
    offsets = {"SVM": -0.13, "ResNet": 0.13}
    for model in ["SVM", "ResNet"]:
        means: list[float] = []
        sds: list[float] = []
        for seed in rp_order:
            feature = f"Gene + RP-414 (seed {seed})"
            values = rp[rp["model"].eq(model) & rp["feature_space"].eq(feature)]["value"].to_numpy()
            mean, sd = mean_sd(values)
            means.append(mean)
            sds.append(sd)
        for feature in ["Gene", "Gene + HubmiR"]:
            values = refs[refs["model"].eq(model) & refs["feature_space"].eq(feature)]["value"].to_numpy()
            mean, sd = mean_sd(values)
            means.append(mean)
            sds.append(sd)
        ax.errorbar(
            x + offsets[model],
            means,
            yerr=sds,
            color=COLORS[model],
            marker="o" if model == "SVM" else "s",
            markersize=3.8,
            markeredgecolor="white",
            markeredgewidth=0.4,
            linewidth=0.95,
            capsize=1.8,
            label=model,
        )
    ax.axvline(4.5, color="#C9CDD2", linewidth=0.7)
    ax.set_xticks(x, categories)
    ax.set_ylabel("Mean held-out macro-F1")
    ax.set_ylim(0.30, 0.77)
    clean_axis(ax)
    panel_header(ax, "c", "Random-projection seed variability", title_x=0.055)
    ax.legend(loc="lower right", ncol=2, handletextpad=0.3, columnspacing=0.8)


def draw_ae_panel(ax: plt.Axes, data: pd.DataFrame, letter: str = "d") -> None:
    best = data.loc[data["validation_mse"].idxmin()]
    ax.plot(data["epoch"], data["train_mse"], color=COLORS["SVM"], linewidth=1.15, label="Training")
    ax.plot(
        data["epoch"],
        data["validation_mse"],
        color=COLORS["HubmiR"],
        linewidth=1.45,
        label="Validation",
    )
    ax.scatter(
        [best["epoch"]],
        [best["validation_mse"]],
        marker="D",
        s=24,
        color=COLORS["HubmiR"],
        edgecolor="white",
        linewidth=0.55,
        zorder=4,
    )
    ax.annotate(
        f"Best validation\nepoch {int(best['epoch'])}",
        xy=(best["epoch"], best["validation_mse"]),
        xytext=(38, 0.19),
        arrowprops={"arrowstyle": "-", "color": COLORS["Muted"], "linewidth": 0.65},
        fontsize=5.5,
        color=COLORS["Muted"],
        ha="center",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_xlim(1, int(data["epoch"].max()))
    ax.set_ylim(0.04, 0.49)
    clean_axis(ax)
    panel_header(ax, letter, "Autoencoder optimization", title_x=0.07)
    ax.legend(loc="upper right", ncol=2, handlelength=1.6, columnspacing=0.9)


def figure_s1() -> None:
    accuracy = pd.read_csv(DATA / "FigS1a_accuracy_display.csv")
    contrasts = pd.read_csv(DATA / "FigS1b_embedding_contrasts.csv")
    ae = pd.read_csv(DATA / "FigS1d_autoencoder_learning_curve.csv")

    fig = plt.figure(figsize=(7.2, 4.45), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        left=0.085,
        right=0.985,
        bottom=0.11,
        top=0.84,
        width_ratios=[1.12, 1.0],
        height_ratios=[1.0, 0.72],
        wspace=0.37,
        hspace=0.55,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])
    draw_accuracy_panel(ax_a, accuracy)
    draw_embedding_contrast_panel(ax_b, contrasts)
    draw_ae_panel(ax_c, ae, letter="c")

    legend_handles = [
        Patch(facecolor=FEATURE_COLORS[feature], edgecolor="#555555", linewidth=0.55, label=label)
        for feature, label in zip(FEATURES, ["Gene", "+PCA", "+AE", "+RP mean", "+HubmiR"])
    ]
    legend_handles.append(
        Line2D([0], [0], marker="D", linestyle="none", color=COLORS["Ink"], markersize=3.5, label="Mean ± 1 SD")
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.915),
        ncol=6,
        handlelength=1.25,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    fig.suptitle(
        "Supplementary Fig. Sx | Robustness of matched embedding controls",
        x=0.085,
        y=0.985,
        ha="left",
        fontsize=8.7,
        fontweight="bold",
    )
    save_figure(fig, "FigureSx_EmbeddingControlRobustness_v2")


def draw_box_with_points(
    ax: plt.Axes,
    values: np.ndarray,
    position: float,
    color: str,
    *,
    facecolor: str | None = None,
    hatch: str | None = None,
) -> None:
    values = np.asarray(values, dtype=float)
    require(len(values) == 20 and np.isfinite(values).all(), "Expected 20 finite CKA values")
    box = ax.boxplot(
        values,
        positions=[position],
        widths=0.30,
        patch_artist=True,
        showfliers=False,
        whis=(0, 100),
        medianprops={"color": COLORS["Ink"], "linewidth": 0.85},
        whiskerprops={"color": "#555555", "linewidth": 0.65},
        capprops={"color": "#555555", "linewidth": 0.65},
    )
    box["boxes"][0].set(
        facecolor=color if facecolor is None else facecolor,
        edgecolor=color if facecolor is not None else "#4A4A4A",
        linewidth=0.75,
        alpha=0.92,
        hatch=hatch,
    )
    ax.scatter(
        position + np.linspace(-0.085, 0.085, len(values)),
        values,
        s=8.5,
        facecolor="white",
        edgecolor=color,
        linewidth=0.5,
        zorder=3,
    )
    ax.errorbar(
        position,
        values.mean(),
        yerr=values.std(ddof=1),
        fmt="none",
        ecolor=COLORS["Ink"],
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
        facecolor=COLORS["Ink"],
        edgecolor="white",
        linewidth=0.45,
        zorder=4,
    )


def draw_cka_panel(ax: plt.Axes, data: pd.DataFrame) -> None:
    targets = ["HubmiR", "PCA-414", "AE-414", "RP-414 mean"]
    target_labels = ["HubmiR", "PCA-414", "AE-414", "RP-414 mean"]
    representation_colors = {
        "HubmiR": COLORS["HubmiR"],
        "PCA-414": "#E4BE69",
        "AE-414": "#E4BE69",
        "RP-414 mean": "#E4BE69",
    }
    offset = 0.19
    for target_index, target in enumerate(targets):
        for method, dx in [("debiased_linear", -offset), ("debiased_rbf", offset)]:
            values = data[
                data["target_representation"].eq(target) & data["method"].eq(method)
            ]["cka"].to_numpy()
            color = representation_colors[target]
            draw_box_with_points(
                ax,
                values,
                target_index + dx,
                color,
                facecolor=None if method == "debiased_linear" else "white",
                hatch=None if method == "debiased_linear" else "////",
            )
    ax.set_xticks(range(len(targets)), target_labels)
    ax.set_ylabel("Held-out debiased CKA")
    ax.set_xlim(-0.58, 3.58)
    ax.set_ylim(0, 0.90)
    ax.set_yticks([0.0, 0.3, 0.6, 0.9])
    clean_axis(ax)
    panel_header(ax, "a", "Full-20 representation geometry", title_x=0.055)
    ax.legend(
        handles=[
            Patch(facecolor="#F0D69A", edgecolor="#8A6A2F", label="Linear"),
            Patch(facecolor="white", edgecolor="#8A6A2F", hatch="////", label="RBF"),
        ],
        loc="upper left",
        ncol=2,
        handlelength=1.4,
        columnspacing=0.8,
    )


def draw_ridge_summary_panel(ax: plt.Axes, data: pd.DataFrame) -> None:
    summaries = ["variance_weighted_r2", "uniform_average_r2", "median_per_feature_r2"]
    labels = ["Variance-weighted", "Uniform-average", "Median feature"]
    spaces = [("Gene-29045", "Full Gene", COLORS["Gene"]), ("HubmiR-input-977", "977-gene input", COLORS["HubmiR light"])]
    x = np.arange(len(summaries), dtype=float)
    width = 0.34
    for space_index, (space, label, color) in enumerate(spaces):
        means: list[float] = []
        sds: list[float] = []
        for summary in summaries:
            values = data[
                data["source_space"].eq(space) & data["r2_summary"].eq(summary)
            ]["r2"].to_numpy()
            require(len(values) == 20, f"Missing Ridge values for {space}, {summary}")
            mean, sd = mean_sd(values)
            means.append(mean)
            sds.append(sd)
        ax.bar(
            x + (space_index - 0.5) * width,
            means,
            width,
            yerr=sds,
            capsize=2.0,
            color=color,
            edgecolor="#555555",
            linewidth=0.55,
            alpha=0.90,
            label=label,
            zorder=2,
        )
    ax.axhline(0, color=COLORS["Ink"], linewidth=0.65)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Held-out R²")
    ax.set_ylim(0, 0.58)
    clean_axis(ax)
    panel_header(ax, "b", "Cross-validated linear recovery", title_x=0.065)
    ax.legend(loc="upper right", ncol=2, handlelength=1.3, columnspacing=0.8)


def draw_per_mirna_panel(ax: plt.Axes, data: pd.DataFrame) -> None:
    wide = data.pivot(index="mirna", columns="source_space", values="median_heldout_r2")
    require(not wide.isna().any().any(), "Per-miRNA source-space pairing contains missing values")
    require(len(wide) == 414, "Per-miRNA panel must contain 414 paired miRNAs")
    x = wide["Gene-29045"].to_numpy(dtype=float)
    y = wide["HubmiR-input-977"].to_numpy(dtype=float)
    low = float(min(x.min(), y.min()))
    high = float(max(x.max(), y.max()))
    pad = 0.04 * (high - low)
    limits = (low - pad, high + pad)
    ax.scatter(
        x,
        y,
        s=8,
        color=COLORS["HubmiR"],
        alpha=0.46,
        edgecolors="none",
        rasterized=True,
    )
    ax.plot(limits, limits, linestyle="--", linewidth=0.8, color="#777777")
    ax.axvline(0.5, color="#C8CCD0", linewidth=0.6, linestyle=":")
    ax.axhline(0.5, color="#C8CCD0", linewidth=0.6, linestyle=":")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Full Gene median held-out R²")
    ax.set_ylabel("977-gene input median held-out R²")
    clean_axis(ax)
    fraction_gene = float((x > 0.5).mean())
    fraction_input = float((y > 0.5).mean())
    ax.text(
        0.03,
        0.97,
        f"n = 414 miRNAs\nR² > 0.5: {fraction_gene:.1%} / {fraction_input:.1%}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        color=COLORS["Muted"],
    )
    panel_header(ax, "c", "Per-miRNA recoverability", title_x=0.07)


def draw_residual_panel(ax: plt.Axes, data: pd.DataFrame) -> None:
    order = [
        ("Gene + predicted HubmiR − Gene", "Predicted − Gene"),
        ("Gene + residual HubmiR − Gene", "Residual − Gene"),
        ("Gene + HubmiR − Gene + predicted HubmiR", "Full − predicted"),
    ]
    offsets = {"SVM": -0.12, "ResNet": 0.12}
    for model in ["SVM", "ResNet"]:
        sub = data[data["model"].eq(model)].set_index("contrast")
        for index, (contrast, _) in enumerate(order):
            row = sub.loc[contrast]
            mean = float(row["mean_delta"])
            ax.errorbar(
                mean,
                index + offsets[model],
                xerr=np.array(
                    [[mean - float(row["ci95_low"])], [float(row["ci95_high"]) - mean]]
                ),
                fmt="o" if model == "SVM" else "s",
                markersize=4.2,
                color=COLORS[model],
                markeredgecolor="white",
                markeredgewidth=0.45,
                linewidth=1.0,
                capsize=2.0,
                zorder=3,
            )
    ax.axvline(0, color=COLORS["Ink"], linewidth=0.75)
    ax.set_yticks(range(len(order)), [label for _, label in order])
    ax.set_ylim(2.85, -0.45)
    ax.set_xlabel("Paired Δmacro-F1")
    clean_axis(ax, grid_axis="x")
    panel_header(ax, "d", "Residualized task contribution", title_x=0.07)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color=COLORS["SVM"], linestyle="none", label="SVM"),
            Line2D([0], [0], marker="s", color=COLORS["ResNet"], linestyle="none", label="ResNet"),
        ],
        loc="lower right",
        bbox_to_anchor=(1.0, -0.015),
        ncol=2,
        handletextpad=0.3,
        columnspacing=0.8,
    )


def figure_s2() -> None:
    cka = pd.read_csv(DATA / "FigS2a_cka_full20.csv")
    ridge = pd.read_csv(DATA / "FigS2b_ridge_full20.csv")
    per_mirna = pd.read_csv(DATA / "FigS2c_per_mirna_ridge.csv")
    residual = pd.read_csv(DATA / "FigS2d_residualization_contrasts.csv")

    fig = plt.figure(figsize=(7.2, 5.15), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        left=0.09,
        right=0.985,
        bottom=0.105,
        top=0.90,
        width_ratios=[1.02, 1.0],
        height_ratios=[1.0, 1.05],
        wspace=0.38,
        hspace=0.48,
    )
    draw_cka_panel(fig.add_subplot(grid[0, 0]), cka)
    draw_ridge_summary_panel(fig.add_subplot(grid[0, 1]), ridge)
    draw_per_mirna_panel(fig.add_subplot(grid[1, 0]), per_mirna)
    draw_residual_panel(fig.add_subplot(grid[1, 1]), residual)
    fig.suptitle(
        "Supplementary Fig. Sy | Linear recoverability and residual task utility",
        x=0.09,
        y=0.985,
        ha="left",
        fontsize=8.7,
        fontweight="bold",
    )
    save_figure(fig, "FigureSy_LinearRecoveryResidualUtility_v2")


def figure_s3() -> None:
    runs = pd.read_csv(DATA / "FigS3a_resnet_tuning_runs.csv")
    configs = pd.read_csv(DATA / "FigS3b_resnet_tuning_configs.csv")
    summary = (
        runs.groupby("round_id", as_index=False)
        .agg(
            validation_mean=("validation_macro_f1", "mean"),
            validation_sd=("validation_macro_f1", "std"),
            train_mean=("train_macro_f1", "mean"),
            train_sd=("train_macro_f1", "std"),
            parameter_count=("parameter_count", "first"),
        )
        .sort_values("round_id")
    )
    require(len(summary) == 10, "ResNet tuning figure requires ten rounds")
    frozen_round = int(summary.loc[summary["validation_mean"].idxmax(), "round_id"])
    require(frozen_round == 10, "The highest validation mean must be round 10")
    frozen = configs[configs["round_id"].eq(frozen_round)].iloc[0]

    fig = plt.figure(figsize=(7.2, 2.95), facecolor="white")
    grid = fig.add_gridspec(
        1,
        2,
        left=0.085,
        right=0.985,
        bottom=0.18,
        top=0.83,
        width_ratios=[1.9, 0.85],
        wspace=0.30,
    )
    ax = fig.add_subplot(grid[0, 0])
    rounds = summary["round_id"].to_numpy(dtype=float)
    for mean_col, sd_col, label, color, marker in [
        ("validation_mean", "validation_sd", "Validation macro-F1", COLORS["HubmiR"], "o"),
        ("train_mean", "train_sd", "Training macro-F1", "#A7836A", "s"),
    ]:
        ax.errorbar(
            rounds,
            summary[mean_col],
            yerr=summary[sd_col],
            color=color,
            marker=marker,
            markersize=4.1,
            markeredgecolor="white",
            markeredgewidth=0.4,
            linewidth=1.25,
            capsize=2.0,
            label=label,
        )
    frozen_value = float(summary.loc[summary["round_id"].eq(frozen_round), "validation_mean"].iloc[0])
    ax.scatter(
        [frozen_round],
        [frozen_value],
        s=64,
        facecolors="none",
        edgecolors=COLORS["Gene"],
        linewidth=1.4,
        zorder=5,
        label="Frozen configuration",
    )
    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.7, 10.3)
    ax.set_ylim(0.47, 1.04)
    ax.set_xlabel("Sequential tuning round")
    ax.set_ylabel("Macro-F1 across validation seeds")
    clean_axis(ax)
    panel_header(ax, "a", "Validation-only sequential tuning", title_x=0.055)
    ax.legend(loc="lower right", handlelength=1.7)

    card = fig.add_subplot(grid[0, 1])
    card.axis("off")
    card.add_patch(
        FancyBboxPatch(
            (0.0, 0.02),
            0.98,
            0.94,
            transform=card.transAxes,
            boxstyle="round,pad=0.018,rounding_size=0.018",
            facecolor="#FAF2DF",
            edgecolor="#DAB05D",
            linewidth=0.8,
        )
    )
    panel_header(card, "b", "Frozen configuration", title_x=0.12)
    lines = [
        ("Selection", "Highest mean validation macro-F1"),
        ("Hidden width", f"{int(frozen['hidden'])}"),
        ("Residual blocks", f"{int(frozen['blocks'])}"),
        ("Dropout", f"{float(frozen['dropout']):.2f}"),
        ("Batch size", f"{int(frozen['batch_size'])}"),
        ("Learning rate", f"{float(frozen['learning_rate']):.0e}"),
        ("Weight decay", f"{float(frozen['weight_decay']):.0e}"),
        ("Patience", f"{int(frozen['patience'])} epochs"),
        ("Parameters", f"{int(summary.loc[summary['round_id'].eq(10), 'parameter_count'].iloc[0]) / 1e6:.2f} M"),
    ]
    y = 0.89
    for key, value in lines:
        card.text(0.07, y, key, transform=card.transAxes, ha="left", va="top", fontsize=5.55, color=COLORS["Muted"])
        if key == "Selection":
            card.text(0.07, y - 0.055, value, transform=card.transAxes, ha="left", va="top", fontsize=5.65, fontweight="bold", color=COLORS["Ink"], wrap=True)
            y -= 0.15
        else:
            card.text(0.92, y, value, transform=card.transAxes, ha="right", va="top", fontsize=5.65, fontweight="bold", color=COLORS["Ink"])
            y -= 0.077
    card.text(
        0.07,
        0.065,
        "Test data were not loaded during tuning.",
        transform=card.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.4,
        color=COLORS["Gene"],
        fontweight="bold",
    )

    fig.suptitle(
        "Supplementary Fig. Sz | Validation-only selection of the ResNet configuration",
        x=0.085,
        y=0.985,
        ha="left",
        fontsize=8.7,
        fontweight="bold",
    )
    save_figure(fig, "FigureSz_ResNetValidationTuning")


def main() -> None:
    set_style()
    figure_s1()
    figure_s2()
    print("PASS: rendered active Supplementary Fig. Sx and Sy v2 in PDF/SVG/PNG/TIFF")


if __name__ == "__main__":
    main()
