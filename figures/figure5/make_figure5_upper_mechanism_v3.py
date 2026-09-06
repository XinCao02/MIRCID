#!/usr/bin/env python3
"""Mechanism-rich Figure 5 upper panel with selected proteins in context.

The 3 x 8 heatmap gene occurrences are represented as protein nodes in their
defensible cellular/function contexts. A solid red node denotes a selected
protein with a supported mechanistic relationship to the depicted pathway;
dashed red nodes are pathway-response associations and do not assert direct
physical regulation. No selected-gene list is placed beneath mRNA/miRNA.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hubmir-fig5-upper-v3-mpl")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "hubmir-fig5-upper-v3-xdg")
)

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

import make_figure5_upper_mechanism_v1 as base


ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
OUTPUT = FIGURES_DIR / "Figure5_UpperMechanism_v3"
COLORS = base.COLORS


FEATURES = {
    "EGFR": ["PCMTD2", "ELF2", "PNP", "NBR1", "POT1", "SYF2", "BSDC1", "KIAA0430"],
    "MAPK": ["PNP", "ETV4", "TCOF1", "CCND1", "PBLD", "EPHA2", "ECH1", "PHLDA2"],
    "Hypoxia": ["GPI", "VKORC1", "PLOD1", "CA9", "TMEM45A", "PDK3", "ALDOC", "ALDOA"],
}

SUPPORTED_SELECTED_NODES = {"EPHA2", "ETV4", "CCND1", "CA9"}

NODE_AUDIT = {
    "PCMTD2": ("cytoplasm", "putative ECS ubiquitin-ligase substrate-recognition component"),
    "ELF2": ("nucleus", "ETS-family transcription factor"),
    "PNP": ("cytoplasm", "purine nucleoside metabolism"),
    "NBR1": ("cytoplasm/autophagosome", "selective-autophagy adaptor"),
    "POT1": ("nucleus/telomere", "shelterin telomere-protection protein"),
    "SYF2": ("nucleus", "spliceosome component"),
    "BSDC1": ("unresolved", "poorly characterized BSD-domain protein"),
    "KIAA0430": ("peroxisome", "MARF1 RNA-stability factor alias"),
    "ETV4": ("nucleus", "ERK-responsive ETS transcription factor"),
    "TCOF1": ("nucleolus", "RNA polymerase I/ribosome-biogenesis regulator"),
    "CCND1": ("nucleus", "cyclin-D1 cell-cycle effector"),
    "PBLD": ("unresolved", "response-associated protein; location not asserted"),
    "EPHA2": ("cell membrane", "receptor tyrosine kinase with RAS–ERK feedback"),
    "ECH1": ("mitochondrion/peroxisome", "fatty-acid beta-oxidation enzyme"),
    "PHLDA2": ("cytoplasm/membrane-associated", "PH-domain-containing regulatory protein"),
    "GPI": ("cytoplasm", "glycolytic enzyme"),
    "VKORC1": ("endoplasmic-reticulum membrane", "vitamin-K epoxide reductase"),
    "PLOD1": ("endoplasmic-reticulum lumen", "collagen lysyl hydroxylase"),
    "CA9": ("cell membrane", "HIF-responsive carbonic anhydrase IX"),
    "TMEM45A": ("membrane/secretory system", "hypoxia-associated transmembrane protein"),
    "PDK3": ("mitochondrial matrix", "pyruvate-dehydrogenase kinase"),
    "ALDOC": ("cytoplasm", "glycolytic aldolase"),
    "ALDOA": ("cytoplasm", "glycolytic aldolase"),
}


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    face: str,
    edge: str,
    fontsize: float = 5.0,
    weight: str = "normal",
    linestyle: str = "-",
    linewidth: float = 0.9,
    text_color: str | None = None,
    zorder: int = 4,
) -> None:
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.004,rounding_size=0.008",
        facecolor=face, edgecolor=edge, linewidth=linewidth,
        linestyle=linestyle, zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize, fontweight=weight,
        color=text_color or COLORS["ink"], linespacing=1.02, zorder=zorder + 1,
    )


def selected_node(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    gene: str,
    role: str,
    *,
    supported: bool | None = None,
) -> None:
    if supported is None:
        supported = gene in SUPPORTED_SELECTED_NODES
    box(
        ax, x, y, w, h, f"{gene}\n{role}",
        face="#FFF7F5",
        edge=COLORS["gene"],
        fontsize=5.0,
        weight="bold",
        linestyle="-" if supported else (0, (3, 2)),
        linewidth=1.15 if supported else 0.85,
        text_color=COLORS["gene"],
    )


def receptor_at(
    ax: plt.Axes,
    x: float,
    y: float,
    label: str,
    color: str,
    *,
    selected: bool = False,
) -> None:
    for dx in (-0.012, 0.012):
        ax.add_patch(
            FancyBboxPatch(
                (x + dx - 0.009, y), 0.018, 0.065,
                boxstyle="round,pad=0.002,rounding_size=0.007",
                facecolor=color, edgecolor=COLORS["gene"] if selected else COLORS["ink"],
                linewidth=1.15 if selected else 0.75, zorder=6,
            )
        )
    ax.text(
        x, y + 0.071, label, ha="center", va="bottom",
        fontsize=5.1, fontweight="bold",
        color=COLORS["gene"] if selected else COLORS["ink"], zorder=7,
    )


def validate_contract() -> dict[str, object]:
    occurrences = [gene for genes in FEATURES.values() for gene in genes]
    missing_audit = sorted(set(occurrences) - set(NODE_AUDIT))
    assert len(occurrences) == 24
    assert len(set(occurrences)) == 23
    assert not missing_audit
    assert SUPPORTED_SELECTED_NODES.issubset(set(occurrences))
    assert all(len(genes) == 8 for genes in FEATURES.values())
    return {
        "status": "PASS",
        "pathways": list(FEATURES),
        "gene_occurrences": len(occurrences),
        "unique_gene_symbols": len(set(occurrences)),
        "supported_selected_nodes": sorted(SUPPORTED_SELECTED_NODES),
        "association_only_occurrences": len(occurrences) - sum(
            gene in SUPPORTED_SELECTED_NODES for gene in occurrences
        ),
        "all_selected_genes_have_explicit_role_and_compartment_audit": True,
        "selected_gene_list_below_mrna_removed": True,
        "edge_semantics": {
            "solid colored edge": "curated canonical signaling/regulation",
            "solid red selected node": "selected heatmap protein with supported pathway relationship",
            "dashed red selected node": "response-associated protein; no direct interaction asserted",
        },
        "node_audit": {
            gene: {"compartment": values[0], "role": values[1]}
            for gene, values in NODE_AUDIT.items()
        },
    }


def draw_figure() -> plt.Figure:
    base.configure_style()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.2,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.25), facecolor="white")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Figure and compartment architecture.
    ax.add_patch(Rectangle((0.018, 0.455), 0.964, 0.355, facecolor=COLORS["cytosol"], edgecolor="none", zorder=0))
    ax.plot([0.018, 0.982], [0.810, 0.810], color=COLORS["membrane"], linewidth=3.0, zorder=1)
    ax.plot([0.018, 0.982], [0.820, 0.820], color="#C8D4DF", linewidth=1.25, zorder=1)
    ax.add_patch(
        FancyBboxPatch(
            (0.030, 0.075), 0.940, 0.395,
            boxstyle="round,pad=0.007,rounding_size=0.055",
            facecolor=COLORS["nucleus"], edgecolor="#CFB978", linewidth=0.9, zorder=0,
        )
    )
    for x in (0.337, 0.676):
        ax.plot([x, x], [0.095, 0.935], color="#CBD1D8", linewidth=0.75, linestyle=(0, (4, 3)), zorder=1)

    ax.text(0.010, 0.988, "a", fontsize=8.7, fontweight="bold", ha="left", va="top")
    ax.text(0.042, 0.987, "Pathway proteins and response-associated features in strict rescue profiles", fontsize=8.0, fontweight="bold", ha="left", va="top", color=COLORS["ink"])
    ax.text(0.178, 0.947, "EGFR pathway", ha="center", va="center", fontsize=6.8, fontweight="bold", color=COLORS["egfr"])
    ax.text(0.507, 0.947, "MAPK pathway", ha="center", va="center", fontsize=6.8, fontweight="bold", color=COLORS["mapk"])
    ax.text(0.832, 0.947, "Hypoxia pathway", ha="center", va="center", fontsize=6.8, fontweight="bold", color=COLORS["hypoxia"])
    ax.text(0.009, 0.875, "EXTRACELLULAR", fontsize=5.3, fontstyle="italic", color=COLORS["muted"], ha="center", va="center", rotation=90, rotation_mode="anchor")
    ax.text(0.009, 0.625, "CYTOPLASM", fontsize=5.3, fontstyle="italic", color=COLORS["muted"], ha="center", va="center", rotation=90, rotation_mode="anchor")
    ax.text(0.045, 0.445, "NUCLEUS", fontsize=5.5, fontstyle="italic", color="#8A763C", ha="left")

    # EGFR column: ligand/receptor and canonical branches.
    box(ax, 0.045, 0.875, 0.073, 0.038, "EGF", face=COLORS["egfr_light"], edge=COLORS["egfr"], fontsize=5.1, weight="bold")
    box(ax, 0.135, 0.875, 0.090, 0.038, "FGF2 / bFGF", face=COLORS["egfr_light"], edge=COLORS["egfr"], fontsize=5.0, weight="bold")
    box(ax, 0.248, 0.875, 0.072, 0.038, "Gefitinib", face=COLORS["gene_light"], edge=COLORS["inhibit"], fontsize=5.0, weight="bold")
    receptor_at(ax, 0.080, 0.770, "EGFR", COLORS["egfr"])
    receptor_at(ax, 0.180, 0.770, "FGFR", "#6A9BB5")
    base.arrow(ax, (0.082, 0.875), (0.080, 0.840), color=COLORS["egfr"])
    base.arrow(ax, (0.180, 0.875), (0.180, 0.840), color="#4D829D")
    base.inhibition(ax, (0.270, 0.875), (0.100, 0.820), bar_length=0.014)
    box(ax, 0.050, 0.705, 0.142, 0.045, "GRB2–SOS1 / FRS2", face=COLORS["egfr_light"], edge=COLORS["egfr"], fontsize=5.0)
    box(ax, 0.215, 0.705, 0.092, 0.045, "RAS–GTP", face=COLORS["mapk_light"], edge=COLORS["mapk"], fontsize=5.0, weight="bold")
    base.arrow(ax, (0.080, 0.770), (0.090, 0.750), color=COLORS["egfr"])
    base.arrow(ax, (0.180, 0.770), (0.165, 0.750), color="#4D829D")
    base.arrow(ax, (0.192, 0.728), (0.215, 0.728), color=COLORS["mapk"])
    box(ax, 0.050, 0.625, 0.132, 0.045, "GAB1–PI3K", face=COLORS["egfr_light"], edge=COLORS["egfr"], fontsize=5.0, weight="bold")
    box(ax, 0.220, 0.625, 0.073, 0.045, "AKT", face=COLORS["egfr_light"], edge=COLORS["egfr"], fontsize=5.0, weight="bold")
    base.arrow(ax, (0.080, 0.770), (0.088, 0.670), color=COLORS["egfr"], connectionstyle="arc3,rad=0.13")
    base.arrow(ax, (0.182, 0.648), (0.220, 0.648), color=COLORS["egfr"])

    # Selected EGFR proteins in their defensible locations; dashed means association only.
    ax.text(0.178, 0.598, "Selected response-associated proteins", fontsize=5.0, color=COLORS["gene"], fontweight="bold", ha="center")
    selected_node(ax, 0.045, 0.535, 0.125, 0.048, "PCMTD2", "cytoplasmic proteostasis")
    selected_node(ax, 0.182, 0.535, 0.125, 0.048, "PNP", "purine metabolism")
    selected_node(ax, 0.045, 0.478, 0.125, 0.048, "NBR1", "autophagy adaptor")
    selected_node(ax, 0.182, 0.478, 0.125, 0.048, "KIAA0430", "MARF1 · peroxisome/RNA")
    selected_node(ax, 0.045, 0.330, 0.125, 0.050, "ELF2", "ETS transcription factor")
    selected_node(ax, 0.182, 0.330, 0.125, 0.050, "POT1", "telomere protection")
    selected_node(ax, 0.045, 0.265, 0.125, 0.050, "SYF2", "spliceosome")
    selected_node(ax, 0.182, 0.265, 0.125, 0.050, "BSDC1", "role/localization unresolved")

    # MAPK column: canonical cascade, inhibitors and EPHA2 feedback.
    box(ax, 0.350, 0.875, 0.078, 0.038, "PLX4720", face=COLORS["gene_light"], edge=COLORS["inhibit"], fontsize=5.0, weight="bold")
    box(ax, 0.438, 0.875, 0.078, 0.038, "PD98059", face=COLORS["gene_light"], edge=COLORS["inhibit"], fontsize=5.0, weight="bold")
    box(ax, 0.526, 0.875, 0.132, 0.038, "MAP2K1 knockdown", face=COLORS["gene_light"], edge=COLORS["inhibit"], fontsize=5.0, weight="bold")
    receptor_at(ax, 0.630, 0.770, "EPHA2", "#F1C0B8", selected=True)
    box(ax, 0.405, 0.728, 0.105, 0.043, "RAS–GTP", face=COLORS["mapk_light"], edge=COLORS["mapk"], fontsize=5.0, weight="bold")
    box(ax, 0.405, 0.662, 0.105, 0.043, "RAF / BRAF", face=COLORS["mapk_light"], edge=COLORS["mapk"], fontsize=5.0, weight="bold")
    box(ax, 0.405, 0.596, 0.105, 0.043, "MEK1/2", face=COLORS["mapk_light"], edge=COLORS["mapk"], fontsize=5.0, weight="bold")
    box(ax, 0.405, 0.530, 0.105, 0.043, "ERK1/2", face=COLORS["mapk_light"], edge=COLORS["mapk"], fontsize=5.0, weight="bold")
    base.arrow(ax, (0.307, 0.728), (0.405, 0.749), color=COLORS["mapk"], connectionstyle="arc3,rad=-0.10")
    base.arrow(ax, (0.458, 0.728), (0.458, 0.705), color=COLORS["mapk"])
    base.arrow(ax, (0.458, 0.662), (0.458, 0.639), color=COLORS["mapk"])
    base.arrow(ax, (0.458, 0.596), (0.458, 0.573), color=COLORS["mapk"])
    base.inhibition(ax, (0.389, 0.875), (0.438, 0.705), bar_length=0.013)
    base.inhibition(ax, (0.477, 0.875), (0.458, 0.639), bar_length=0.013)
    base.inhibition(ax, (0.592, 0.875), (0.482, 0.639), bar_length=0.013)
    base.arrow(ax, (0.610, 0.770), (0.510, 0.749), color=COLORS["gene"], linestyle=(0, (3, 2)), connectionstyle="arc3,rad=-0.18")
    ax.text(0.570, 0.752, "feedback", fontsize=5.0, color=COLORS["gene"], ha="center")

    selected_node(ax, 0.350, 0.535, 0.047, 0.050, "PNP", "purine")
    selected_node(ax, 0.520, 0.535, 0.058, 0.050, "PBLD", "response")
    selected_node(ax, 0.585, 0.535, 0.070, 0.050, "ECH1", "mito/perox")
    selected_node(ax, 0.520, 0.478, 0.135, 0.050, "PHLDA2", "PH-domain protein")
    box(ax, 0.390, 0.392, 0.180, 0.043, "ERK-responsive nuclear program", face=COLORS["mapk_light"], edge=COLORS["mapk"], fontsize=5.0, weight="bold")
    base.arrow(ax, (0.458, 0.530), (0.458, 0.435), color=COLORS["mapk"])
    selected_node(ax, 0.350, 0.315, 0.095, 0.055, "ETV4", "ERK effector TF", supported=True)
    selected_node(ax, 0.455, 0.315, 0.095, 0.055, "CCND1", "cell-cycle effector", supported=True)
    selected_node(ax, 0.560, 0.315, 0.095, 0.055, "TCOF1", "nucleolar rRNA")
    base.arrow(ax, (0.435, 0.392), (0.400, 0.370), color=COLORS["mapk"])
    base.arrow(ax, (0.500, 0.392), (0.502, 0.370), color=COLORS["mapk"])

    # Hypoxia column and selected adaptive proteins.
    box(ax, 0.735, 0.875, 0.155, 0.042, "Low O2 / hypoxia", face=COLORS["hypoxia_light"], edge=COLORS["hypoxia"], fontsize=5.1, weight="bold")
    box(ax, 0.700, 0.720, 0.112, 0.045, "PHD / EGLN", face=COLORS["hypoxia_light"], edge=COLORS["hypoxia"], fontsize=5.0, weight="bold")
    box(ax, 0.835, 0.710, 0.135, 0.065, "VHL-mediated\nHIF-1α degradation\n(normoxia)", face="#ECEDEF", edge=COLORS["inactive"], fontsize=5.0)
    box(ax, 0.730, 0.635, 0.135, 0.048, "Stabilized HIF-1α", face=COLORS["hypoxia_light"], edge=COLORS["hypoxia"], fontsize=5.0, weight="bold")
    base.inhibition(ax, (0.782, 0.875), (0.756, 0.765), color=COLORS["hypoxia"], bar_length=0.014)
    base.arrow(ax, (0.865, 0.875), (0.825, 0.683), color=COLORS["hypoxia"], connectionstyle="arc3,rad=-0.18")
    base.arrow(ax, (0.812, 0.743), (0.835, 0.743), color=COLORS["inactive"])
    selected_node(ax, 0.895, 0.790, 0.075, 0.047, "CA9", "cell-surface pH", supported=True)
    box(ax, 0.700, 0.540, 0.128, 0.057, "GPI · ALDOA · ALDOC\ncytosolic glycolysis", face="#FFF7F5", edge=COLORS["gene"], fontsize=5.0, weight="bold", linestyle=(0, (3, 2)), text_color=COLORS["gene"])
    selected_node(ax, 0.845, 0.540, 0.125, 0.057, "PDK3", "mitochondrial PDH kinase")
    box(ax, 0.700, 0.475, 0.128, 0.057, "VKORC1 · PLOD1\nER metabolism / ECM", face="#FFF7F5", edge=COLORS["gene"], fontsize=5.0, weight="bold", linestyle=(0, (3, 2)), text_color=COLORS["gene"])
    selected_node(ax, 0.845, 0.475, 0.125, 0.057, "TMEM45A", "hypoxic membrane stress")
    box(ax, 0.755, 0.365, 0.165, 0.050, "HIF-1α–ARNT (HIF-1β)", face=COLORS["hypoxia_light"], edge=COLORS["hypoxia"], fontsize=5.0, weight="bold")
    base.arrow(ax, (0.798, 0.635), (0.820, 0.415), color=COLORS["hypoxia"])
    base.arrow(ax, (0.835, 0.415), (0.835, 0.475), color=COLORS["inactive"], linestyle=(0, (3, 3)))
    ax.text(0.865, 0.444, "adaptive response", fontsize=5.0, color=COLORS["hypoxia"], ha="left")

    # Compact DNA/RNA endpoint and immediate transition to the heatmaps.
    box(ax, 0.100, 0.175, 0.800, 0.040, "DNA response elements · transcription factors · RNA polymerase II", face=COLORS["white"], edge="#887B5A", fontsize=5.2, weight="bold")
    base.arrow(ax, (0.505, 0.315), (0.505, 0.215), color=COLORS["mapk"])
    base.arrow(ax, (0.835, 0.365), (0.760, 0.215), color=COLORS["hypoxia"])
    box(ax, 0.240, 0.105, 0.220, 0.045, "mRNA / Gene-expression features", face=COLORS["gene_light"], edge=COLORS["gene"], fontsize=5.0, weight="bold", text_color=COLORS["gene"])
    box(ax, 0.560, 0.105, 0.220, 0.045, "inferred HubmiR features", face=COLORS["mirna_light"], edge=COLORS["mirna"], fontsize=5.0, weight="bold", text_color=COLORS["mirna"])
    base.arrow(ax, (0.350, 0.175), (0.350, 0.150), color=COLORS["gene"])
    base.arrow(ax, (0.650, 0.175), (0.650, 0.150), color=COLORS["mirna"], linestyle=(0, (3, 2)))
    base.arrow(ax, (0.350, 0.105), (0.350, 0.030), color=COLORS["gene"], linewidth=1.4, mutation_scale=10)
    base.arrow(ax, (0.670, 0.105), (0.670, 0.030), color=COLORS["mirna"], linewidth=1.4, mutation_scale=10)
    ax.text(0.350, 0.008, "Gene heatmaps — EGFR · MAPK · Hypoxia", fontsize=5.0, color=COLORS["gene"], fontweight="bold", ha="center", va="bottom")
    ax.text(0.670, 0.008, "HubmiR heatmaps — EGFR · MAPK · Hypoxia", fontsize=5.0, color=COLORS["mirna"], fontweight="bold", ha="center", va="bottom")

    # Line/node legend.
    ax.add_patch(FancyBboxPatch((0.700, 0.279), 0.026, 0.018, boxstyle="round,pad=0.001,rounding_size=0.003", facecolor="#FFF7F5", edgecolor=COLORS["gene"], linewidth=1.1, zorder=5))
    ax.text(0.731, 0.288, "supported selected node", fontsize=5.0, color=COLORS["muted"], va="center")
    ax.add_patch(FancyBboxPatch((0.840, 0.279), 0.026, 0.018, boxstyle="round,pad=0.001,rounding_size=0.003", facecolor="#FFF7F5", edgecolor=COLORS["gene"], linewidth=0.85, linestyle=(0, (3, 2)), zorder=5))
    ax.text(0.871, 0.288, "association-only node", fontsize=5.0, color=COLORS["muted"], va="center")
    ax.plot([0.700, 0.725], [0.255, 0.255], color=COLORS["mapk"], linewidth=1.2)
    ax.text(0.730, 0.255, "curated pathway edge", fontsize=5.0, color=COLORS["muted"], va="center")
    ax.plot([0.830, 0.855], [0.255, 0.255], color=COLORS["inactive"], linewidth=1.0, linestyle=(0, (3, 2)))
    ax.text(0.860, 0.255, "response association", fontsize=5.0, color=COLORS["muted"], va="center")

    return fig


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    validation = validate_contract()
    fig = draw_figure()
    fig.savefig(f"{OUTPUT}.png", dpi=600, facecolor="white")
    fig.savefig(f"{OUTPUT}.pdf", facecolor="white")
    fig.savefig(f"{OUTPUT}.svg", facecolor="white")
    fig.savefig(f"{OUTPUT}.tiff", dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    validation.update({
        "output_stem": OUTPUT.name,
        "canvas_inches": [7.2, 5.25],
        "nominal_width_mm": 182.88,
        "dpi": 600,
        "exports": ["png", "pdf", "svg", "tiff"],
    })
    (RESULTS_DIR / "Figure5_UpperMechanism_v3_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
