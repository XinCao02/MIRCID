#!/usr/bin/env python3
"""Integrate the selected-protein v3 mechanism panel with v3 heatmaps."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import make_figure5_integrated_v3 as integrated_v3
import make_figure5_integrated_v4 as figure


WORK_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = WORK_DIR / "figures"
RESULTS_DIR = WORK_DIR / "results"
UPPER_PATH = FIGURES_DIR / "Figure5_UpperMechanism_v3.png"
UPPER_VALIDATION_PATH = RESULTS_DIR / "Figure5_UpperMechanism_v3_validation.json"
OUTPUT = FIGURES_DIR / "Figure5_IntegratedRescue_v7"


def load_upper() -> tuple[np.ndarray, dict[str, object]]:
    if not UPPER_PATH.exists():
        raise FileNotFoundError(UPPER_PATH)
    with Image.open(UPPER_PATH) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb).copy()
        size = list(rgb.size)
    return array, {
        "path": str(UPPER_PATH.relative_to(WORK_DIR)),
        "sha256": integrated_v3.sha256(UPPER_PATH),
        "size_px": size,
    }


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _, selected, selected_validation = integrated_v3.load_selected()
    upper, upper_meta = load_upper()
    upper_validation = json.loads(UPPER_VALIDATION_PATH.read_text(encoding="utf-8"))
    if upper_validation.get("status") != "PASS":
        raise AssertionError("Upper-panel validation did not pass")

    figure.RESULT_PREFIX = OUTPUT.name
    figure.VERSION = "integrated_v7"
    figure.FIGSIZE = (7.2, 9.2)
    figure.TOP_IMAGE_RECT = (0.020, 0.405, 0.960, 0.548)
    figure.GROUP_BOTTOMS = (0.300, 0.205, 0.110)
    figure.GROUP_HEIGHT = 0.070
    figure.CAX_Y = 0.110
    figure.CAX_HEIGHT = 0.260
    figure.PRESERVE_SOURCE_ARROW_PIXELS = False
    figure.ADD_ARROW_CONTINUATIONS = False
    figure.GREEN_FRAME_XPAD = 0.005
    figure.GREEN_FRAME_YPAD = 0.005
    figure.RED_FRAME_PAD = 0.0015
    figure.TOP_FEATURE_LABEL_PAD = 3.2
    figure.OTHER_FEATURE_LABEL_PAD = 0.8
    figure.SAMPLE_FONT_SIZE = 5.3
    figure.SAMPLE_FONT_WEIGHT = "bold"
    figure.TRAILING_DIRECTION_ARROW = True
    figure.PATHWAY_FONT_SIZE = 6.2

    with tempfile.TemporaryDirectory(prefix="fig5-integrated-v7-") as temp_dir:
        temp_stem = Path(temp_dir) / OUTPUT.name
        figure.OUTPUT_STEM = temp_stem
        figure.make_integrated_figure(upper, selected)
        for suffix in (".png", ".pdf", ".svg", ".tiff"):
            shutil.copy2(temp_stem.with_suffix(suffix), OUTPUT.with_suffix(suffix))

    validation = {
        "status": "PASS",
        "version": "integrated_v7",
        "upper_panel": upper_meta,
        "upper_scientific_validation": upper_validation,
        "gene_occurrences_in_mechanism_panel": upper_validation["gene_occurrences"],
        "unique_gene_symbols_in_mechanism_panel": upper_validation["unique_gene_symbols"],
        "supported_selected_nodes": upper_validation["supported_selected_nodes"],
        "selected_gene_list_below_mrna_removed": True,
        "heatmap_source_data": "Figure5_RescueCases_featurewise_normalized_v3_source_data.csv",
        "selection_and_heatmap_values_changed": False,
        "source_selection_validation": selected_validation,
        "exports": ["png", "pdf", "svg", "tiff"],
    }
    (RESULTS_DIR / "Figure5_IntegratedRescue_v7_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
