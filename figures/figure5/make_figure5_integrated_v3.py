from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hubmir-fig5-integrated-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "hubmir-fig5-integrated-xdg"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

import run_rescue_case_normalized as base
import run_rescue_case_normalized_v2 as v2
import run_rescue_case_normalized_v3 as v3


WORK_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = WORK_DIR / "figures"
RESULTS_DIR = WORK_DIR / "results"
OUTPUT_STEM = FIGURES_DIR / "Figure5_IntegratedRescue_v3"

ORIGINAL_RELATIVE_PATH = Path(
    "Paper Writing/SubmittedVers/latex_archive/figures/path_case_v2.png"
)
ORIGINAL_CROP_Y = 1240
# Remove only the obsolete tilted feature labels from the central transition
# strip. The submitted schematic and its red/green side annotations remain intact.
ORIGINAL_MASK_BOX = (185, 1135, 1454, ORIGINAL_CROP_Y)

FRAME_RED = "#D43F32"
FRAME_GREEN = "#55A946"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_original_schematic(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        original_size = rgb.size
        array = np.asarray(rgb).copy()
    if ORIGINAL_CROP_Y >= array.shape[0]:
        raise AssertionError("Submitted Figure 5 crop exceeds source image height")
    top = array[:ORIGINAL_CROP_Y].copy()
    x0, y0, x1, y1 = ORIGINAL_MASK_BOX
    if not (0 <= x0 < x1 <= top.shape[1] and 0 <= y0 < y1 <= top.shape[0]):
        raise AssertionError("Submitted Figure 5 transition mask lies outside crop")
    top[y0:y1, x0:x1, :] = 255
    return top, {
        "source_size_px": list(original_size),
        "crop_y_px": ORIGINAL_CROP_Y,
        "cropped_size_px": [int(top.shape[1]), int(top.shape[0])],
        "central_transition_mask_xyxy_px": list(ORIGINAL_MASK_BOX),
    }


def load_selected() -> tuple[Path, dict[str, pd.DataFrame], dict[str, object]]:
    root = base.find_project_root(WORK_DIR)
    inputs = base.load_inputs(base.source_paths(root))
    data = inputs["data"]
    predictions = inputs["predictions"]
    assert isinstance(data, dict)
    assert isinstance(predictions, pd.DataFrame)
    split_audit = base.verify_prediction_splits(data, predictions)
    scored = v2.score_all_rescue_features(inputs)
    selected = v2.select_cases_and_features(scored)
    validation = v2.validate_selection(selected, scored)
    validation["reconstructed_test_ids_match"] = bool(split_audit["exact_match"].all())
    validation["display_labels_match_standalone_v3"] = set(
        selected["selected_cases"]["sample_id"].astype(str)
    ) == set(v3.DISPLAY_LABELS)
    if not all(value for key, value in validation.items() if key != "status"):
        validation["status"] = "FAIL"
    if validation["status"] != "PASS":
        raise AssertionError(f"Integrated Figure 5 validation failed: {validation}")
    return root, selected, validation


def add_feature_frames(
    fig: plt.Figure,
    gene_axes: list[plt.Axes],
    mirna_axes: list[plt.Axes],
) -> None:
    if len(gene_axes) != len(mirna_axes):
        raise AssertionError("Gene and HubmiR axes must be paired")
    for ax_gene, ax_mirna in zip(gene_axes, mirna_axes):
        gene_box = ax_gene.get_position()
        mirna_box = ax_mirna.get_position()
        green = Rectangle(
            (gene_box.x0 - 0.004, gene_box.y0 - 0.004),
            mirna_box.x1 - gene_box.x0 + 0.008,
            gene_box.height + 0.008,
            transform=fig.transFigure,
            fill=False,
            edgecolor=FRAME_GREEN,
            linewidth=1.15,
            linestyle=(0, (6, 4)),
            zorder=20,
            clip_on=False,
        )
        red = Rectangle(
            (gene_box.x0 - 0.0015, gene_box.y0 - 0.0015),
            gene_box.width + 0.003,
            gene_box.height + 0.003,
            transform=fig.transFigure,
            fill=False,
            edgecolor=FRAME_RED,
            linewidth=1.15,
            linestyle=(0, (6, 4)),
            zorder=21,
            clip_on=False,
        )
        fig.add_artist(green)
        fig.add_artist(red)

    top_gene = gene_axes[0].get_position()
    bottom_gene = gene_axes[-1].get_position()
    top_mirna = mirna_axes[0].get_position()
    bottom_mirna = mirna_axes[-1].get_position()
    red_x = top_gene.x0 - 0.012
    green_x = top_mirna.x1 + 0.012
    y0 = min(bottom_gene.y0, bottom_mirna.y0) - 0.004
    y1 = max(top_gene.y1, top_mirna.y1) + 0.004
    red_arrow = FancyArrowPatch(
        (red_x, y0),
        (red_x, y1),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.5,
        linestyle=(0, (6, 5)),
        color=FRAME_RED,
        zorder=25,
        clip_on=False,
    )
    green_arrow = FancyArrowPatch(
        (green_x, y0),
        (green_x, y1),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.5,
        linestyle=(0, (6, 5)),
        color=FRAME_GREEN,
        zorder=25,
        clip_on=False,
    )
    fig.add_artist(red_arrow)
    fig.add_artist(green_arrow)


def make_integrated_figure(
    original_top: np.ndarray,
    selected: dict[str, pd.DataFrame],
) -> None:
    base.set_style()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 5.2,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    cases = selected["selected_cases"]
    genes = selected["selected_genes"]
    mirnas = selected["selected_mirnas"]
    cmap = LinearSegmentedColormap.from_list(
        "integrated_relative_signal", [v3.COLORS["blue"], v3.COLORS["mid"], v3.COLORS["red"]]
    )
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    fig = plt.figure(figsize=(7.2, 8.8), constrained_layout=False)
    ax_top = fig.add_axes([0.02, 0.33, 0.96, 0.67])
    ax_top.imshow(original_top, aspect="auto", interpolation="lanczos")
    ax_top.set_axis_off()

    grid = fig.add_gridspec(
        v2.N_PATHWAYS,
        3,
        height_ratios=[1, 1, 1],
        width_ratios=[2.65, 8.0, 3.0],
        left=0.055,
        right=0.900,
        top=0.300,
        bottom=0.042,
        hspace=0.40,
        wspace=0.090,
    )
    gene_axes: list[plt.Axes] = []
    mirna_axes: list[plt.Axes] = []
    last_image: mpl.image.AxesImage | None = None

    for row, pathway in enumerate(v2.PATHWAY_SPECS):
        pathway_cases = cases.loc[cases["pathway"].eq(pathway)].sort_values("sample_order")
        pathway_genes = genes.loc[genes["pathway"].eq(pathway)].sort_values(
            ["sample_order", "feature_order"]
        )
        pathway_mirnas = mirnas.loc[mirnas["pathway"].eq(pathway)].sort_values(
            ["sample_order", "feature_order"]
        )
        row_ids = pathway_cases["sample_id"].astype(str).tolist()
        gene_features = (
            pathway_genes.drop_duplicates("feature_order")
            .sort_values("feature_order")["gene"]
            .astype(str)
            .tolist()
        )
        mirna_features = (
            pathway_mirnas.drop_duplicates("feature_order")
            .sort_values("feature_order")["mirna"]
            .astype(str)
            .tolist()
        )
        anchors = set(
            pathway_mirnas.loc[pathway_mirnas["literature_anchor"], "mirna"].astype(str)
        )

        ax_label = fig.add_subplot(grid[row, 0])
        ax_gene = fig.add_subplot(grid[row, 1])
        ax_mirna = fig.add_subplot(grid[row, 2])
        gene_axes.append(ax_gene)
        mirna_axes.append(ax_mirna)
        ax_label.set_axis_off()
        ax_label.text(
            0.04,
            1.23,
            f"{pathway} Pathway",
            transform=ax_label.transAxes,
            fontsize=5.9,
            fontweight="bold",
            color=v3.COLORS["ink"],
            va="top",
        )
        for i, case in enumerate(pathway_cases.itertuples(index=False)):
            y_position = 1.0 - (i + 0.5) / v2.N_SAMPLES_PER_PATHWAY
            sample_id = str(case.sample_id)
            ax_label.text(
                0.04,
                y_position,
                v3.DISPLAY_LABELS[sample_id],
                transform=ax_label.transAxes,
                fontsize=4.8,
                color=v3.COLORS["ink"],
                fontweight="medium",
                va="center",
            )

        v3._heatmap(
            ax_gene,
            pathway_genes,
            row_ids,
            gene_features,
            "gene",
            norm,
            cmap,
            v3.COLORS["gene_text"],
        )
        last_image = v3._heatmap(
            ax_mirna,
            pathway_mirnas,
            row_ids,
            mirna_features,
            "mirna",
            norm,
            cmap,
            v3.COLORS["mirna_text"],
            anchors,
        )

    if last_image is None:
        raise AssertionError("No integrated rescue heatmap generated")
    add_feature_frames(fig, gene_axes, mirna_axes)

    cax = fig.add_axes([0.950, 0.060, 0.012, 0.205])
    colorbar = fig.colorbar(last_image, cax=cax, orientation="vertical")
    colorbar.set_label("Training-fold empirical-percentile deviation", fontsize=4.8, labelpad=3)
    colorbar.ax.tick_params(labelsize=4.5, length=1.8, pad=1.2)
    fig.text(
        0.50,
        0.012,
        "Expanded strict-rescue profiles; inferred HubmiRs are predictive recodings and do not establish causal rescue.",
        ha="center",
        va="bottom",
        fontsize=4.5,
        color=v3.COLORS["muted"],
    )

    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=450, facecolor="white", bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
    fig.savefig(
        OUTPUT_STEM.with_suffix(".tiff"),
        dpi=600,
        facecolor="white",
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    root, selected, validation = load_selected()
    original_path = root / ORIGINAL_RELATIVE_PATH
    if not original_path.exists():
        raise FileNotFoundError(original_path)
    original_top, original_meta = prepare_original_schematic(original_path)
    make_integrated_figure(original_top, selected)

    manifest = {
        "version": "integrated_v3",
        "original_figure_sha256": sha256(original_path),
        "original_figure": original_meta,
        "composition": "submitted schematic crop plus redrawn aligned v3 expanded rescue heatmaps",
        "red_frame": "eight Gene columns per pathway",
        "green_frame": "eight Gene plus three inferred-HubmiR columns per pathway",
        "vertical_arrows": "red Gene-only and green Gene-plus-HubmiR visual grammar",
        "heatmap_source_data": "Figure5_RescueCases_featurewise_normalized_v3_source_data.csv",
        "selection_and_values_changed": False,
    }
    base.write_json(manifest, RESULTS_DIR / "Figure5_IntegratedRescue_v3_manifest.json")
    base.write_json(validation, RESULTS_DIR / "Figure5_IntegratedRescue_v3_validation.json")
    print(json.dumps({"validation": validation, "manifest": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
