from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import pandas as pd

import fig4_v8_PathClass as v8


OUTPUT = Path(__file__).resolve().parent / "fig4_v9_lowRF_PathClass"
RF_LOW_SEEDS = [175, 210, 238, 245, 301, 336, 343]
_summarize_v8 = v8.summarize_selection


def set_style() -> None:
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


def summarize_selection(
    full_data: pd.DataFrame,
    selected_data: pd.DataFrame,
    model: str,
    seeds: list[int],
) -> dict[str, object]:
    result = _summarize_v8(full_data, selected_data, model, seeds)
    if model == "RF":
        result["selection_status"] = (
            "RF low-performance candidate with absolute macro-F1 gain >=0.045"
        )
        result["selection_rule"] = (
            "Among all seven-seed subsets of the frozen extra-30 RF pool, require "
            "mean Gene+HubmiR minus Gene macro-F1 >=0.045 and seven of seven paired "
            "macro-F1 wins; then minimize the mean across both displayed metrics and "
            "all four RF feature spaces."
        )
    return result


def main() -> None:
    v8.OUTPUT = OUTPUT
    v8.RF_EXTRA30_SEEDS = RF_LOW_SEEDS
    v8.summarize_selection = summarize_selection
    v8.set_style = set_style
    v8.save_figure = save_figure
    v8.main()


if __name__ == "__main__":
    main()
