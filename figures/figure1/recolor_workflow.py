#!/usr/bin/env python3
"""Create palette-only variants of the original Figure 1.

Only pixels belonging to the four large header fills and four large panel-body
fills are recolored. All foreground artwork, text, arrows, and geometry remain
from the original raster.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "figure1_workflow.png"

# Inclusive panel regions in the original 3352 x 1818 artwork.
PANELS = (
    {"x": (18, 744), "header": (81, 124, 190), "body": (220, 232, 253)},
    {"x": (757, 1782), "header": (112, 168, 82), "body": (231, 238, 225)},
    {"x": (1799, 2557), "header": (131, 101, 170), "body": (234, 220, 230)},
    {"x": (2576, 3338), "header": (213, 141, 61), "body": (250, 241, 224)},
)

# The three options retain the original blue/green/violet/orange identities.
VARIANTS = {
    "Fig1_palette_A_mineral.png": {
        "headers": ("#347DAA", "#609B56", "#8068A0", "#DD852E"),
        "body_strength": (0.135, 0.020),
    },
    "Fig1_palette_B_editorial_muted.png": {
        "headers": ("#2E6588", "#557D5B", "#6F608B", "#B96B40"),
        "body_strength": (0.090, 0.000),
    },
    "Fig1_palette_C_soft_modern.png": {
        "headers": ("#3C7198", "#5B8C70", "#846A96", "#CC7D43"),
        "body_strength": (0.155, 0.035),
    },
}


def hex_rgb(value: str) -> np.ndarray:
    value = value.lstrip("#")
    return np.array([int(value[i : i + 2], 16) for i in (0, 2, 4)], dtype=float)


def mixture_mask(region: np.ndarray, source: tuple[int, int, int], tolerance: float = 4.0):
    """Find a flat fill and its antialiased mixture with white."""
    pix = region.astype(float)
    src = np.array(source, dtype=float)
    white = np.full(3, 255.0)
    direction = src - white
    alpha = np.sum((pix - white) * direction, axis=2) / np.sum(direction * direction)
    alpha_clipped = np.clip(alpha, 0.0, 1.0)
    reconstructed = white + alpha_clipped[..., None] * direction
    residual = np.linalg.norm(pix - reconstructed, axis=2)
    mask = (alpha > 0.035) & (alpha < 1.025) & (residual <= tolerance)
    return mask, alpha_clipped


def recolor_fill(
    canvas: np.ndarray,
    box: tuple[int, int, int, int],
    source: tuple[int, int, int],
    replacement: np.ndarray,
    vertical_fade: Optional[Tuple[float, float]] = None,
) -> None:
    x0, y0, x1, y1 = box
    region = canvas[y0:y1, x0:x1, :3]
    mask, edge_alpha = mixture_mask(region, source)
    white = np.full(3, 255.0)

    if vertical_fade is None:
        target = np.broadcast_to(replacement, region.shape)
    else:
        top_strength, bottom_strength = vertical_fade
        position = np.linspace(0.0, 1.0, region.shape[0], dtype=float)
        # Smoothstep gives a quiet editorial fade without a visible band.
        smooth = position * position * (3.0 - 2.0 * position)
        strength = top_strength + (bottom_strength - top_strength) * smooth
        row_color = white + strength[:, None] * (replacement - white)
        target = np.broadcast_to(row_color[:, None, :], region.shape)

    composited = white + edge_alpha[..., None] * (target - white)
    region[mask] = np.rint(composited[mask]).astype(np.uint8)


def create_variant(output_name: str, spec: dict) -> None:
    image = Image.open(SOURCE).convert("RGBA")
    canvas = np.array(image)
    height, width = canvas.shape[:2]
    if (width, height) != (3352, 1818):
        raise ValueError(f"Unexpected source size: {(width, height)}")

    for panel, header_hex in zip(PANELS, spec["headers"]):
        x0, x1 = panel["x"]
        header_color = hex_rgb(header_hex)
        recolor_fill(canvas, (x0, 20, x1, 385), panel["header"], header_color)
        recolor_fill(
            canvas,
            (x0, 384, x1, 1801),
            panel["body"],
            header_color,
            vertical_fade=spec["body_strength"],
        )

    Image.fromarray(canvas).save(HERE / output_name, optimize=True)


if __name__ == "__main__":
    for filename, variant in VARIANTS.items():
        create_variant(filename, variant)
