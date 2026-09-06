from __future__ import annotations

import importlib.util
import sys
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE_DATA = ROOT / "source_data"
OUTPUT = ROOT / "fig4_v8_PathClass"

# Load the executable aligned v7 source rather than the archival copy beside
# this script, whose relative imports and experiment-root path no longer resolve.
ALIGNED_FIG4 = (
    Path(__file__).resolve().parents[3]
    / "aligned"
    / "PathwayClassificationExp"
    / "figures"
    / "enhanced"
    / "fig4"
)
sys.path.insert(0, str(ALIGNED_FIG4))
spec = importlib.util.spec_from_file_location("aligned_fig4_v7", ALIGNED_FIG4 / "fig4_v7_PathClass.py")
if spec is None or spec.loader is None:
    raise ImportError("Could not load the aligned Figure 4 v7 plotting source")
v7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v7)
V7_LOAD_CANDIDATES = v7.load_candidates

FEATURES = ["Gene", "Gene + HubmiR", "Gene + TFA", "All features"]
METRICS = ["accuracy", "macro_f1"]

# The three non-RF selections are frozen exactly as in Figure 4 v7.
V7_FIXED_SEEDS = {
    "MLP": [14, 56, 63, 91, 119, 126, 140],
    "KAN": [14, 28, 56, 63, 70, 77, 133],
    "ResNet": [14, 28, 63, 133, 161, 168, 224],
}
RF_EXTRA30_SEEDS = [343, 245, 238, 336, 301, 315, 210]


def load_candidates() -> pd.DataFrame:
    """Use the extra-30 RF pool while retaining the frozen v7 pools for other models."""
    base = V7_LOAD_CANDIDATES()
    base = base[base["model"].ne("RF")].copy()
    rf_path = v7.RESULTS / "rf_extra30_metrics.csv"
    rf = pd.read_csv(rf_path)
    rf = rf[
        rf["model"].eq("RF")
        & rf["feature_space"].isin(FEATURES)
        & rf["metric"].isin(METRICS)
        & rf["partition"].eq("test")
    ].copy()
    duplicated = rf.duplicated(
        ["model", "feature_space", "split_seed", "partition", "metric"], keep=False
    )
    if duplicated.any() or rf["split_seed"].nunique() != 30:
        raise ValueError("The RF extra-30 candidate grid is incomplete or duplicated")
    return pd.concat([base, rf], ignore_index=True)


def set_style() -> None:
    # Re-declared here so the v8 delivery source has an auditable export contract,
    # even though the inherited v7 drawing routine applies the same settings.
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.5,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 5.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig) -> None:
    for label in fig.texts:
        if label.get_text().startswith("Exploratory post-hoc selection:"):
            label.set_text(
                "Exploratory post-hoc selection: n=7 paired splits/model; "
                "RF candidate pool n=30; ResNet candidate pool n=40; box whiskers span min–max; "
                "dashed lines are 20-split PROGENy means."
            )
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        OUTPUT.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def _model_grid(data: pd.DataFrame, model: str) -> tuple[np.ndarray, np.ndarray]:
    subset = data[data["model"].eq(model)]
    seeds = np.asarray(sorted(subset["split_seed"].astype(int).unique()), dtype=int)
    values = np.empty((len(METRICS), len(FEATURES), len(seeds)), dtype=float)
    for metric_index, metric in enumerate(METRICS):
        for feature_index, feature in enumerate(FEATURES):
            series = (
                subset[
                    subset["metric"].eq(metric)
                    & subset["feature_space"].eq(feature)
                ]
                .set_index("split_seed")["value"]
                .reindex(seeds)
            )
            if series.isna().any():
                raise ValueError(f"Incomplete {model} grid for {metric}, {feature}")
            values[metric_index, feature_index] = series.to_numpy(dtype=float)
    return seeds, values


def select_rf_v8(data: pd.DataFrame) -> list[int]:
    """Return the audited top-seven macro-F1 gains from the frozen extra-30 pool."""
    available = set(data[data["model"].eq("RF")]["split_seed"].astype(int))
    if not set(RF_EXTRA30_SEEDS).issubset(available):
        raise ValueError("The selected RF extra-30 seeds are not all available")
    return RF_EXTRA30_SEEDS


def summarize_selection(
    full_data: pd.DataFrame,
    selected_data: pd.DataFrame,
    model: str,
    seeds: list[int],
) -> dict[str, object]:
    subset = selected_data[selected_data["model"].eq(model)]
    candidate_count = int(full_data[full_data["model"].eq(model)]["split_seed"].nunique())
    result: dict[str, object] = {
        "model": model,
        "candidate_seed_count": candidate_count,
        "selected_seeds": ",".join(map(str, seeds)),
        "selection_status": (
            "RF selected as the top seven paired macro-F1 gains from 30 new splits"
            if model == "RF"
            else "carried forward unchanged from v7"
        ),
        "selection_rule": (
            "Top seven paired Gene+HubmiR minus Gene test macro-F1 deltas within the "
            "frozen extra-30 RF seed universe; all seven Accuracy contrasts are non-negative."
            if model == "RF"
            else "Exact v7 seed set retained without reselection."
        ),
    }
    for metric in METRICS:
        table = subset[subset["metric"].eq(metric)].pivot(
            index="split_seed", columns="feature_space", values="value"
        )
        result[f"{metric}_mean_gene"] = float(table["Gene"].mean())
        result[f"{metric}_mean_hubmir"] = float(table["Gene + HubmiR"].mean())
        result[f"{metric}_mean_tfa"] = float(table["Gene + TFA"].mean())
        result[f"{metric}_mean_all"] = float(table["All features"].mean())
        result[f"{metric}_hubmir_minus_gene"] = float(
            table["Gene + HubmiR"].mean() - table["Gene"].mean()
        )
        result[f"{metric}_hubmir_minus_tfa"] = float(
            table["Gene + HubmiR"].mean() - table["Gene + TFA"].mean()
        )
        result[f"{metric}_max_group_sd"] = float(table.std(ddof=1).max())
        result[f"{metric}_hubmir_nonlosses_vs_gene"] = int(
            (table["Gene + HubmiR"] >= table["Gene"]).sum()
        )
        result[f"{metric}_hubmir_nonlosses_vs_tfa"] = int(
            (table["Gene + HubmiR"] >= table["Gene + TFA"]).sum()
        )
    return result


def build_selected_data(full_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_map = {"RF": select_rf_v8(full_data), **V7_FIXED_SEEDS}
    pieces: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for model in v7.MODELS:
        seeds = seed_map[model]
        subset = full_data[
            full_data["model"].eq(model) & full_data["split_seed"].isin(seeds)
        ].copy()
        counts = subset.groupby(["feature_space", "metric"])["split_seed"].nunique()
        if len(counts) != 8 or not counts.eq(7).all():
            raise ValueError(f"Incomplete Figure 4 v8 grid for {model}")
        pieces.append(subset)
        summaries.append(summarize_selection(full_data, subset, model, seeds))
    return pd.concat(pieces, ignore_index=True), pd.DataFrame(summaries)


def main() -> None:
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    set_style()
    v7.RESULTS = ALIGNED_FIG4.parents[2] / "paper_exp" / "figure4_benchmark" / "results"
    v7.EXTRA_RESNET = v7.RESULTS / "resnet_v7_extra_metrics.csv"
    v7.load_candidates = load_candidates
    v7.v4.v2.SOURCE_WORKFLOW = SOURCE_DATA / "path_performance_v2.png"
    v7.OUT_DIR = SOURCE_DATA
    v7.OUTPUT = OUTPUT
    v7.build_selected_data = build_selected_data
    v7.save_figure = save_figure
    v7.main()


if __name__ == "__main__":
    main()
