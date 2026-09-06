from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[9]
SOURCE = (
    Path(__file__).resolve().parents[1]
    / "figures"
    / "FigureS_RescueAtlas_composite.png"
)
TARGET_DIR = (
    PROJECT_ROOT
    / "Paper Writing"
    / "SubmittedVers"
    / "latex_archive"
    / "figures"
)

# Boundaries were measured on the final 2,159 × 5,334 px composite.
# The gap between panels c and d is split between the two exports; the final
# footnote begins below row 5,285 and is intentionally excluded from panel d.
CROPS = {
    "S_RescueAtlas.png": (0, 0, 2159, 2290),
    "S_RescueAtlas2.png": (0, 2390, 2159, 5250),
}


def main() -> None:
    image = Image.open(SOURCE).convert("RGB")
    if image.size != (2159, 5334):
        raise AssertionError(f"Unexpected source size: {image.size}")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for filename, box in CROPS.items():
        output = TARGET_DIR / filename
        image.crop(box).save(output, format="PNG", optimize=True, dpi=(300, 300))
        with Image.open(output) as check:
            expected = (box[2] - box[0], box[3] - box[1])
            if check.size != expected or check.mode not in {"RGB", "RGBA"}:
                raise AssertionError(f"Invalid export {output}: {check.size}, {check.mode}")


if __name__ == "__main__":
    main()
