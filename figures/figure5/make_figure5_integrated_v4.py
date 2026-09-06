from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hubmir-fig5-integrated-v4-mpl")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "hubmir-fig5-integrated-v4-xdg")
)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image

import make_figure5_integrated_v3 as integrated_v3
import run_rescue_case_normalized as base
import run_rescue_case_normalized_v2 as v2
import run_rescue_case_normalized_v3 as v3


WORK_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = WORK_DIR / "figures"
RESULTS_DIR = WORK_DIR / "results"
OUTPUT_STEM = FIGURES_DIR / "Figure5_IntegratedRescue_v4"
RESULT_PREFIX = "Figure5_IntegratedRescue_v4"
VERSION = "integrated_v4"

ORIGINAL_RELATIVE_PATH = Path(
    "Paper Writing/SubmittedVers/latex_archive/figures/path_case_v2.png"
)
ORIGINAL_CROP_Y = 1240
# Mask the obsolete central feature labels while preserving both submitted
# side arrows and the complete right-side "with miRNA Features" annotation.
ORIGINAL_MASK_BOX = (160, 1135, 1410, ORIGINAL_CROP_Y)

FRAME_RED = "#D43F32"
FRAME_GREEN = "#55A946"
FIGSIZE = (7.2, 10.0)
TOP_IMAGE_RECT = (0.030, 0.385, 0.940, 0.615)
PRESERVE_SOURCE_ARROW_PIXELS = False
ADD_ARROW_CONTINUATIONS = True
GREEN_FRAME_XPAD = 0.0035
GREEN_FRAME_YPAD = 0.0035
RED_FRAME_PAD = 0.0015
SAMPLE_FONT_SIZE = 4.8
SAMPLE_FONT_WEIGHT = "medium"
PATHWAY_FONT_SIZE = 5.9
TRAILING_DIRECTION_ARROW = False
TOP_FEATURE_LABEL_PAD = 0.8
OTHER_FEATURE_LABEL_PAD = 0.8
CAX_Y = 0.060
CAX_HEIGHT = 0.245

# Fixed axes align the new heatmaps with the submitted red/green arrow columns.
LABEL_X0 = 0.035
LABEL_WIDTH = 0.108
GENE_X0 = 0.150
GENE_WIDTH = 0.520
MIRNA_X0 = 0.680
MIRNA_WIDTH = 0.215
GROUP_HEIGHT = 0.072
GROUP_BOTTOMS = (0.270, 0.165, 0.060)


def prepare_original_schematic(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        original_size = rgb.size
        array = np.asarray(rgb).copy()
    if ORIGINAL_CROP_Y >= array.shape[0]:
        raise AssertionError("Submitted Figure 5 crop exceeds source image height")
    top = array[:ORIGINAL_CROP_Y].copy()
    unmasked_top = top.copy()
    x0, y0, x1, y1 = ORIGINAL_MASK_BOX
    if not (0 <= x0 < x1 <= top.shape[1] and 0 <= y0 < y1 <= top.shape[0]):
        raise AssertionError("Submitted Figure 5 transition mask lies outside crop")
    top[y0:y1, x0:x1, :] = 255
    if PRESERVE_SOURCE_ARROW_PIXELS:
        source_region = unmasked_top[y0:y1, x0:x1, :]
        red_pixels = (
            (source_region[..., 0] > source_region[..., 1] + 10)
            & (source_region[..., 0] > source_region[..., 2] + 10)
            & (source_region[..., 0] > 135)
        )
        green_pixels = (
            (source_region[..., 1] > source_region[..., 0] + 8)
            & (source_region[..., 1] > source_region[..., 2] + 8)
            & (source_region[..., 1] > 95)
        )
        retained_pixels = red_pixels | green_pixels
        masked_region = top[y0:y1, x0:x1, :]
        masked_region[retained_pixels] = source_region[retained_pixels]
    return top, {
        "source_size_px": list(original_size),
        "crop_y_px": ORIGINAL_CROP_Y,
        "cropped_size_px": [int(top.shape[1]), int(top.shape[0])],
        "central_transition_mask_xyxy_px": list(ORIGINAL_MASK_BOX),
        "side_arrows_preserved": True,
        "source_arrow_pixels_restored_after_mask": PRESERVE_SOURCE_ARROW_PIXELS,
    }


def add_global_frames_and_arrow_continuations(
    fig: plt.Figure,
    gene_axes: list[plt.Axes],
    mirna_axes: list[plt.Axes],
) -> None:
    if len(gene_axes) != v2.N_PATHWAYS or len(mirna_axes) != v2.N_PATHWAYS:
        raise AssertionError("Expected one Gene and one HubmiR axis per pathway")

    gene_boxes = [axis.get_position() for axis in gene_axes]
    mirna_boxes = [axis.get_position() for axis in mirna_axes]
    left = min(box.x0 for box in gene_boxes)
    gene_right = max(box.x1 for box in gene_boxes)
    all_right = max(box.x1 for box in mirna_boxes)
    bottom = min(box.y0 for box in gene_boxes + mirna_boxes)
    top = max(box.y1 for box in gene_boxes + mirna_boxes)
    xpad = GREEN_FRAME_XPAD
    ypad = GREEN_FRAME_YPAD

    green = Rectangle(
        (left - xpad, bottom - ypad),
        all_right - left + 2 * xpad,
        top - bottom + 2 * ypad,
        transform=fig.transFigure,
        fill=False,
        edgecolor=FRAME_GREEN,
        linewidth=1.25,
        linestyle=(0, (7, 4)),
        zorder=20,
        clip_on=False,
    )
    red = Rectangle(
        (left - RED_FRAME_PAD, bottom - RED_FRAME_PAD),
        gene_right - left + 2 * RED_FRAME_PAD,
        top - bottom + 2 * RED_FRAME_PAD,
        transform=fig.transFigure,
        fill=False,
        edgecolor=FRAME_RED,
        linewidth=1.25,
        linestyle=(0, (7, 4)),
        zorder=21,
        clip_on=False,
    )
    fig.add_artist(green)
    fig.add_artist(red)

    if ADD_ARROW_CONTINUATIONS:
        # v4 retains the aligned vector continuation used in that frozen layout.
        for x, color in ((left, FRAME_RED), (all_right, FRAME_GREEN)):
            fig.add_artist(
                Line2D(
                    [x, x],
                    [bottom - 0.006, TOP_IMAGE_RECT[1] + 0.060],
                    transform=fig.transFigure,
                    color=color,
                    linewidth=1.55,
                    linestyle=(0, (7, 5)),
                    solid_capstyle="round",
                    dash_capstyle="round",
                    zorder=24,
                    clip_on=False,
                )
            )


def format_sample_label(label: str) -> str:
    if TRAILING_DIRECTION_ARROW and len(label) >= 3 and label[0] in {"↑", "↓"}:
        return f"{label[2:]} {label[0]}"
    return label


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
        "integrated_relative_signal_v4",
        [v3.COLORS["blue"], v3.COLORS["mid"], v3.COLORS["red"]],
    )
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    fig = plt.figure(figsize=FIGSIZE, constrained_layout=False)
    ax_top = fig.add_axes(TOP_IMAGE_RECT)
    ax_top.imshow(original_top, aspect="auto", interpolation="lanczos")
    ax_top.set_axis_off()

    gene_axes: list[plt.Axes] = []
    mirna_axes: list[plt.Axes] = []
    last_image: mpl.image.AxesImage | None = None

    for pathway_index, (pathway, y0) in enumerate(zip(v2.PATHWAY_SPECS, GROUP_BOTTOMS)):
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

        ax_label = fig.add_axes([LABEL_X0, y0, LABEL_WIDTH, GROUP_HEIGHT])
        ax_gene = fig.add_axes([GENE_X0, y0, GENE_WIDTH, GROUP_HEIGHT])
        ax_mirna = fig.add_axes([MIRNA_X0, y0, MIRNA_WIDTH, GROUP_HEIGHT])
        gene_axes.append(ax_gene)
        mirna_axes.append(ax_mirna)
        ax_label.set_axis_off()
        ax_label.text(
            0.98,
            1.28,
            f"{pathway} Pathway",
            transform=ax_label.transAxes,
            fontsize=PATHWAY_FONT_SIZE,
            fontweight="bold",
            color=v3.COLORS["ink"],
            ha="right",
            va="top",
        )
        for i, case in enumerate(pathway_cases.itertuples(index=False)):
            y_position = 1.0 - (i + 0.5) / v2.N_SAMPLES_PER_PATHWAY
            sample_id = str(case.sample_id)
            ax_label.text(
                0.98,
                y_position,
                format_sample_label(v3.DISPLAY_LABELS[sample_id]),
                transform=ax_label.transAxes,
                fontsize=SAMPLE_FONT_SIZE,
                color=v3.COLORS["ink"],
                fontweight=SAMPLE_FONT_WEIGHT,
                ha="right",
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
            label_rotation=22,
            label_pad=(TOP_FEATURE_LABEL_PAD if pathway_index == 0 else OTHER_FEATURE_LABEL_PAD),
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
            anchor_features=anchors,
            label_rotation=22,
            label_pad=(TOP_FEATURE_LABEL_PAD if pathway_index == 0 else OTHER_FEATURE_LABEL_PAD),
        )

    if last_image is None:
        raise AssertionError("No integrated rescue heatmap generated")
    add_global_frames_and_arrow_continuations(fig, gene_axes, mirna_axes)

    cax = fig.add_axes([0.925, CAX_Y, 0.012, CAX_HEIGHT])
    colorbar = fig.colorbar(last_image, cax=cax, orientation="vertical")
    colorbar.set_label(
        "Training-fold empirical-percentile deviation", fontsize=4.8, labelpad=3
    )
    colorbar.ax.tick_params(labelsize=4.5, length=1.8, pad=1.2)
    fig.text(
        0.50,
        0.014,
        "Expanded strict-rescue profiles; inferred HubmiRs are predictive recodings and do not establish causal rescue.",
        ha="center",
        va="bottom",
        fontsize=4.5,
        color=v3.COLORS["muted"],
    )

    save_kwargs = {"facecolor": "white", "bbox_inches": "tight", "pad_inches": 0.025}
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=450, **save_kwargs)
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), **save_kwargs)
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), **save_kwargs)
    fig.savefig(
        OUTPUT_STEM.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
        **save_kwargs,
    )
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    root, selected, validation = integrated_v3.load_selected()
    original_path = root / ORIGINAL_RELATIVE_PATH
    if not original_path.exists():
        raise FileNotFoundError(original_path)
    original_top, original_meta = prepare_original_schematic(original_path)
    make_integrated_figure(original_top, selected)

    manifest = {
        "version": VERSION,
        "original_figure_sha256": integrated_v3.sha256(original_path),
        "original_figure": original_meta,
        "composition": "submitted schematic plus enlarged aligned v3 rescue heatmaps",
        "red_frame": "one global frame enclosing all eight-Gene panels",
        "green_frame": "one global frame enclosing all Gene-plus-HubmiR panels",
        "vertical_arrows": (
            "submitted arrowheads retained with aligned lower continuations"
            if ADD_ARROW_CONTINUATIONS
            else "submitted arrow pixels retained without manual continuation"
        ),
        "heatmap_source_data": "Figure5_RescueCases_featurewise_normalized_v3_source_data.csv",
        "selection_and_values_changed": False,
        "layout_changes": {
            "heatmaps_enlarged": True,
            "sample_labels_right_aligned_near_heatmaps": True,
            "sample_label_font_size": SAMPLE_FONT_SIZE,
            "sample_label_font_weight": SAMPLE_FONT_WEIGHT,
            "trailing_direction_arrow": TRAILING_DIRECTION_ARROW,
            "pathway_font_size": PATHWAY_FONT_SIZE,
            "feature_label_rotation_degrees": 22,
            "top_feature_label_pad_points": TOP_FEATURE_LABEL_PAD,
            "other_feature_label_pad_points": OTHER_FEATURE_LABEL_PAD,
            "green_frame_xpad": GREEN_FRAME_XPAD,
            "green_frame_ypad": GREEN_FRAME_YPAD,
            "manual_arrow_continuations": ADD_ARROW_CONTINUATIONS,
            "global_frames_per_modality": 1,
        },
    }
    base.write_json(manifest, RESULTS_DIR / f"{RESULT_PREFIX}_manifest.json")
    base.write_json(validation, RESULTS_DIR / f"{RESULT_PREFIX}_validation.json")
    print(json.dumps({"validation": validation, "manifest": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
