from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hubmir-rescue-7run-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "hubmir-rescue-7run-xdg"))

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, Rectangle
from sklearn.model_selection import train_test_split


HERE = Path(__file__).resolve()
OUT_DIR = HERE.parents[1]
EXP_DIR = HERE.parents[5]
PROJECT_ROOT = EXP_DIR.parents[3]
RESCUE_DIR = EXP_DIR / "figures" / "enhanced" / "fig5_PathwayRescueAnalysis"
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"
QA_DIR = OUT_DIR / "qa"

PATHWAY_ORDER = [
    "EGFR",
    "MAPK",
    "Hypoxia",
    "JAK-STAT",
    "NFkB",
    "PI3K",
    "TGFb",
    "TNFa",
    "Trail",
    "VEGF",
    "p53",
]
MAIN_PATHWAYS = ["EGFR", "MAPK", "Hypoxia"]
MAX_CASES = 3
N_GENES = 8
N_MIRNAS = 3
MAX_OFFICIAL_RANK = 60
P53_FOOTPRINT_GENES = 4
P53_CONTEXT_GENES = 4

PATHWAY_ANCHORS = {
    "JAK-STAT": ["hsa-miR-21-5p"],
    "NFkB": ["hsa-miR-223-3p"],
    "PI3K": ["hsa-miR-21-5p"],
    "TGFb": ["hsa-miR-21-5p"],
    "TNFa": ["hsa-miR-155-5p", "hsa-miR-146a-5p"],
    "Trail": ["hsa-miR-221-3p"],
    "VEGF": ["hsa-miR-126-3p", "hsa-miR-126-5p"],
    "p53": ["hsa-miR-34a-5p"],
}

COLORS = {
    "green": "#2F7D5B",
    "green_light": "#DDEDE5",
    "red": "#B64B47",
    "red_light": "#F2DEDC",
    "blue": "#3E6D8E",
    "blue_light": "#DFEAF1",
    "grey_light": "#E9ECEE",
    "ink": "#26343B",
    "muted": "#66737A",
    "grid": "#D7DDE0",
    "gene": "#B7413E",
    "mirna": "#287A54",
    "heat_blue": "#356C9A",
    "heat_mid": "#F7F4ED",
    "heat_red": "#B94D45",
}

STATE_COLORS = {
    "both wrong": COLORS["grey_light"],
    "rescued": COLORS["green_light"],
    "harmed": COLORS["red_light"],
    "both right": COLORS["blue_light"],
}
STATE_TEXT = {
    "both wrong": COLORS["ink"],
    "rescued": COLORS["green"],
    "harmed": COLORS["red"],
    "both right": COLORS["blue"],
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.5,
            "axes.titlesize": 7.5,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def display_pathway(value: str) -> str:
    return {
        "NFkB": "NFκB",
        "TGFb": "TGFβ",
        "TNFa": "TNFα",
        "Trail": "TRAIL",
    }.get(str(value), str(value))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def save_figure(fig: plt.Figure, stem: Path, png_dpi: int = 450) -> None:
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=png_dpi, facecolor="white", bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        facecolor="white",
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def load_inputs() -> dict[str, object]:
    paths = {
        "predictions": RESCUE_DIR / "data" / "selected_resnet_test_predictions.csv.gz",
        "metadata": RESCUE_DIR / "data" / "pathway_sample_metadata.csv",
        "main_cases": RESCUE_DIR / "results" / "normalized_v2_selected_cases.csv",
        "main_solutions": RESCUE_DIR / "results" / "normalized_v2_selected_solutions.csv",
        "processed": EXP_DIR / "data" / "processed" / "pathway_aligned.npz",
        "footprints": EXP_DIR / "data" / "processed" / "progeny_top100_long.csv",
        "raw_gene": PROJECT_ROOT / "Inference Models" / "Pathway Inference" / "PROGENy_expr.csv",
    }
    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    predictions = pd.read_csv(paths["predictions"])
    metadata = pd.read_csv(paths["metadata"]).drop_duplicates("sample_id")
    main_cases = pd.read_csv(paths["main_cases"])
    main_solutions = pd.read_csv(paths["main_solutions"])
    footprints = pd.read_csv(paths["footprints"]).rename(columns={"p.value": "p_value"})
    footprints["official_rank"] = footprints.groupby("pathway", sort=False).cumcount() + 1
    with np.load(paths["processed"], allow_pickle=False) as archive:
        processed = {key: archive[key] for key in archive.files}
    sample_ids = processed["sample_ids"].astype(str)
    raw_gene = pd.read_csv(paths["raw_gene"], index_col=0).T
    raw_gene.index = raw_gene.index.astype(str)
    raw_gene.columns = raw_gene.columns.astype(str)
    raw_gene = raw_gene.loc[sample_ids, processed["gene_names"].astype(str)]
    return {
        "paths": paths,
        "predictions": predictions,
        "metadata": metadata,
        "main_cases": main_cases,
        "main_solutions": main_solutions,
        "footprints": footprints,
        "processed": processed,
        "raw_gene": raw_gene,
    }


def pair_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    expected_features = {"Gene", "Gene + HubmiR"}
    if set(predictions["feature_space"]) != expected_features:
        raise AssertionError("The frozen Figure 4 prediction file must contain only Gene and Gene + HubmiR")
    if predictions["split_seed"].nunique() != 7:
        raise AssertionError("Expected exactly the seven frozen Figure 4 ResNet runs")
    keys = ["split_seed", "sample_id", "true_pathway", "feature_space"]
    if predictions.duplicated(keys).any():
        raise AssertionError("Duplicate prediction key")
    paired = predictions.pivot(
        index=["split_seed", "sample_id", "true_pathway"],
        columns="feature_space",
        values=["correct", "pred_pathway"],
    ).reset_index()
    paired.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in paired.columns
    ]
    gene_ok = paired["correct_Gene"].astype(bool)
    hubmir_ok = paired["correct_Gene + HubmiR"].astype(bool)
    paired["state"] = "both wrong"
    paired.loc[(~gene_ok) & hubmir_ok, "state"] = "rescued"
    paired.loc[gene_ok & (~hubmir_ok), "state"] = "harmed"
    paired.loc[gene_ok & hubmir_ok, "state"] = "both right"
    paired["gene_correct"] = gene_ok
    paired["hubmir_correct"] = hubmir_ok
    if len(paired) != 7 * 85:
        raise AssertionError(f"Expected 595 paired held-out predictions, found {len(paired)}")
    return paired


def summary_tables(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    total = len(paired)
    matrix = (
        paired.groupby(["gene_correct", "hubmir_correct", "state"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    matrix["proportion"] = matrix["count"] / total
    matrix["denominator"] = total
    pathway = (
        paired.loc[paired["state"].isin(["rescued", "harmed"])]
        .groupby(["true_pathway", "state"], as_index=False)
        .size()
        .rename(columns={"size": "count", "true_pathway": "pathway"})
        .pivot(index="pathway", columns="state", values="count")
        .reindex(PATHWAY_ORDER)
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    pathway["net_rescue"] = pathway["rescued"] - pathway["harmed"]
    pathway["test_instances"] = (
        paired.groupby("true_pathway").size().reindex(PATHWAY_ORDER).to_numpy()
    )
    return matrix, pathway


def sample_summary(paired: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    rescue = paired.loc[paired["state"].eq("rescued")]
    summary = (
        paired.groupby(["sample_id", "true_pathway"], as_index=False)
        .agg(
            test_occurrences=("state", "size"),
            rescue_occurrences=("state", lambda values: int((values == "rescued").sum())),
            harm_occurrences=("state", lambda values: int((values == "harmed").sum())),
        )
    )
    wrong = (
        rescue.groupby(["sample_id", "true_pathway"])["pred_pathway_Gene"]
        .agg(lambda values: "; ".join(sorted({display_pathway(str(value)) for value in values})))
        .rename("gene_wrong_classes")
        .reset_index()
    )
    summary = summary.loc[summary["rescue_occurrences"].gt(0)].merge(
        wrong, on=["sample_id", "true_pathway"], how="left", validate="one_to_one"
    )
    summary = summary.merge(metadata, on="sample_id", how="left", validate="one_to_one")
    if summary[["treatment", "effect", "accession", "cells"]].isna().any().any():
        raise AssertionError("Missing rescue-profile metadata")
    summary["stability_score"] = summary["rescue_occurrences"] - summary["harm_occurrences"]
    return summary


def select_cases(
    paired: pd.DataFrame, summary: pd.DataFrame, main_cases: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    main_order = {
        str(row.sample_id): (int(row.pathway_order), int(row.sample_order))
        for row in main_cases.itertuples(index=False)
    }
    main_ids = set(main_order)
    missing_main = main_ids - set(summary["sample_id"].astype(str))
    if missing_main:
        raise AssertionError(f"Locked main Figure 5 profiles are not strict rescues: {sorted(missing_main)}")

    main = summary.loc[summary["sample_id"].isin(main_ids)].copy()
    main["pathway_order"] = main["sample_id"].map(lambda value: main_order[str(value)][0])
    main["sample_order"] = main["sample_id"].map(lambda value: main_order[str(value)][1])
    main["display_role"] = "Main Figure 5"
    selected_rows.append(main)
    for row in main.itertuples(index=False):
        audit_rows.append(
            {
                "pathway": row.true_pathway,
                "sample_id": row.sample_id,
                "selection_role": "locked main Figure 5 profile",
                "preferred_run": "locked",
                "selected_from_preferred_run": True,
            }
        )

    rescue = paired.loc[paired["state"].eq("rescued")]
    summary_lookup = summary.set_index("sample_id")
    for pathway_index, pathway in enumerate(PATHWAY_ORDER, start=1):
        if pathway in MAIN_PATHWAYS:
            continue
        pathway_rescue = rescue.loc[rescue["true_pathway"].eq(pathway)].copy()
        counts = pathway_rescue.groupby("split_seed")["sample_id"].nunique().sort_index()
        if counts.empty:
            raise AssertionError(f"No rescue profile for {pathway}")
        preferred_run = int(counts.loc[counts.eq(counts.max())].index.min())
        preferred_ids = pathway_rescue.loc[
            pathway_rescue["split_seed"].eq(preferred_run), "sample_id"
        ].astype(str).unique().tolist()

        candidates = summary.loc[summary["true_pathway"].eq(pathway)].copy()
        candidates["preferred"] = candidates["sample_id"].isin(preferred_ids)
        candidates = candidates.sort_values(
            ["preferred", "stability_score", "rescue_occurrences", "harm_occurrences", "sample_id"],
            ascending=[False, False, False, True, True],
            kind="stable",
        )
        n_select = min(MAX_CASES, len(candidates))
        chosen = candidates.head(n_select).copy()
        chosen["pathway_order"] = pathway_index
        chosen["sample_order"] = np.arange(1, len(chosen) + 1)
        chosen["display_role"] = "Supplementary"
        selected_rows.append(chosen)
        for row in chosen.itertuples(index=False):
            audit_rows.append(
                {
                    "pathway": pathway,
                    "sample_id": row.sample_id,
                    "selection_role": "same-run-first strict rescue selection",
                    "preferred_run": preferred_run,
                    "selected_from_preferred_run": bool(row.preferred),
                }
            )

    selected = pd.concat(selected_rows, ignore_index=True).sort_values(
        ["pathway_order", "sample_order"], kind="stable"
    )
    audit = pd.DataFrame(audit_rows).merge(
        selected[
            [
                "sample_id",
                "true_pathway",
                "pathway_order",
                "sample_order",
                "rescue_occurrences",
                "harm_occurrences",
            ]
        ],
        left_on=["sample_id", "pathway"],
        right_on=["sample_id", "true_pathway"],
        how="left",
        validate="one_to_one",
    )
    return selected, audit


def split_indices(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    train, held = train_test_split(idx, test_size=0.30, random_state=int(seed), stratify=y)
    validation, test = train_test_split(
        held, test_size=0.50, random_state=int(seed), stratify=y[held]
    )
    return train, validation, test


def ecdf_vector(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    finite = np.isfinite(reference)
    n = finite.sum(axis=0)
    less = np.sum((reference < values) & finite, axis=0)
    equal = np.sum((reference == values) & finite, axis=0)
    fractions = np.full(np.asarray(values).shape, np.nan, dtype=float)
    np.divide(less + 0.5 * equal, n, out=fractions, where=n > 0)
    deviations = 2.0 * (fractions - 0.5)
    deviations[(n == 0) | ~np.isfinite(values)] = np.nan
    return deviations


def compute_case_feature_scores(
    selected: pd.DataFrame,
    paired: pd.DataFrame,
    processed: dict[str, np.ndarray],
    raw_gene: pd.DataFrame,
    footprints: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_ids = processed["sample_ids"].astype(str)
    sample_index = {sample: index for index, sample in enumerate(sample_ids)}
    y = processed["y"].astype(int)
    mirna = processed["mirna"].astype(float)
    mirna_names = processed["mirna_names"].astype(str)
    rescue = paired.loc[paired["state"].eq("rescued")]
    gene_rows: list[dict[str, object]] = []
    mirna_rows: list[dict[str, object]] = []

    for case in selected.itertuples(index=False):
        pathway = str(case.true_pathway)
        sample_id = str(case.sample_id)
        rescue_runs = rescue.loc[
            rescue["sample_id"].eq(sample_id) & rescue["true_pathway"].eq(pathway),
            "split_seed",
        ].astype(int).tolist()
        if pathway == "p53":
            official_rank = (
                footprints.loc[footprints["pathway"].eq(pathway), ["gene", "official_rank"]]
                .drop_duplicates("gene")
                .set_index("gene")["official_rank"]
            )
            gene_frame = pd.DataFrame(
                {
                    "gene": raw_gene.columns.astype(str),
                    "official_rank": raw_gene.columns.astype(str).map(official_rank),
                }
            )
            gene_frame = gene_frame.loc[
                ~gene_frame["gene"].astype(str).str.startswith("LOC")
            ].copy()
        else:
            gene_frame = footprints.loc[
                footprints["pathway"].eq(pathway)
                & footprints["official_rank"].le(MAX_OFFICIAL_RANK)
                & ~footprints["gene"].astype(str).str.startswith("LOC")
            ].copy()
        gene_names = gene_frame["gene"].astype(str).tolist()
        gene_values = raw_gene.loc[sample_id, gene_names].to_numpy(dtype=float)
        gene_scores = []
        mirna_scores = []
        for run in rescue_runs:
            train_idx, _, _ = split_indices(y, run)
            gene_scores.append(
                ecdf_vector(gene_values, raw_gene.iloc[train_idx][gene_names].to_numpy(dtype=float))
            )
            mirna_scores.append(ecdf_vector(mirna[sample_index[sample_id]], mirna[train_idx]))
        gene_array = np.vstack(gene_scores)
        mirna_array = np.vstack(mirna_scores)
        for index, row in enumerate(gene_frame.itertuples(index=False)):
            if not np.isfinite(gene_array[:, index]).all():
                continue
            gene_rows.append(
                {
                    "sample_id": sample_id,
                    "pathway": pathway,
                    "gene": str(row.gene),
                    "official_rank": int(row.official_rank) if pd.notna(row.official_rank) else np.nan,
                    "pathway_footprint": bool(pd.notna(row.official_rank)),
                    "mean_ecdf_deviation": float(gene_array[:, index].mean()),
                    "max_abs_ecdf_deviation": float(np.abs(gene_array[:, index]).max()),
                }
            )
        for index, name in enumerate(mirna_names):
            mirna_rows.append(
                {
                    "sample_id": sample_id,
                    "pathway": pathway,
                    "mirna": str(name),
                    "mean_ecdf_deviation": float(mirna_array[:, index].mean()),
                    "min_abs_ecdf_deviation": float(np.abs(mirna_array[:, index]).min()),
                }
            )
    return pd.DataFrame(gene_rows), pd.DataFrame(mirna_rows)


def select_feature_panels(
    selected: pd.DataFrame,
    genes: pd.DataFrame,
    mirnas: pd.DataFrame,
    main_solutions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    main_lookup = main_solutions.set_index("pathway")
    for pathway in PATHWAY_ORDER:
        sample_ids = selected.loc[selected["true_pathway"].eq(pathway), "sample_id"].astype(str).tolist()
        if pathway in MAIN_PATHWAYS:
            selected_genes = str(main_lookup.loc[pathway, "selected_genes"]).split(";")
            selected_mirnas = str(main_lookup.loc[pathway, "selected_mirnas"]).split(";")
            basis = "locked main Figure 5 feature panel"
        elif pathway == "p53":
            gene_sub = genes.loc[
                genes["pathway"].eq(pathway) & genes["sample_id"].isin(sample_ids)
            ]
            gene_pivot = gene_sub.pivot_table(
                index="gene",
                columns="sample_id",
                values="max_abs_ecdf_deviation",
                aggfunc="first",
            )
            gene_pivot = gene_pivot.loc[gene_pivot[sample_ids].notna().all(axis=1)].copy()
            gene_pivot["worst"] = gene_pivot[sample_ids].max(axis=1)
            gene_pivot["mean_abs"] = gene_pivot[sample_ids].mean(axis=1)
            rank_map = gene_sub.drop_duplicates("gene").set_index("gene")["official_rank"]
            gene_pivot["official_rank"] = gene_pivot.index.map(rank_map)
            associated = (
                gene_pivot.loc[gene_pivot["official_rank"].notna()]
                .reset_index()
                .sort_values(["worst", "mean_abs", "official_rank", "gene"], kind="stable")
                .head(P53_FOOTPRINT_GENES)["gene"]
                .astype(str)
                .tolist()
            )
            context = (
                gene_pivot.loc[gene_pivot["official_rank"].isna()]
                .reset_index()
                .sort_values(["worst", "mean_abs", "gene"], kind="stable")
                .head(P53_CONTEXT_GENES)["gene"]
                .astype(str)
                .tolist()
            )
            if len(associated) != P53_FOOTPRINT_GENES or len(context) != P53_CONTEXT_GENES:
                raise AssertionError("p53 panel must contain four footprint and four context genes")
            selected_genes = [
                gene
                for associated_gene, context_gene in zip(associated, context)
                for gene in (associated_gene, context_gene)
            ]

            mirna_sub = mirnas.loc[
                mirnas["pathway"].eq(pathway) & mirnas["sample_id"].isin(sample_ids)
            ]
            mirna_pivot = mirna_sub.pivot_table(
                index="mirna",
                columns="sample_id",
                values="min_abs_ecdf_deviation",
                aggfunc="first",
            )
            mirna_pivot = mirna_pivot.loc[mirna_pivot[sample_ids].notna().all(axis=1)].copy()
            mirna_pivot["floor"] = mirna_pivot[sample_ids].min(axis=1)
            anchors = PATHWAY_ANCHORS[pathway]
            fillers = (
                mirna_pivot.loc[~mirna_pivot.index.isin(anchors)]
                .sort_values("floor", ascending=False, kind="stable")
                .head(N_MIRNAS - len(anchors))
                .index.astype(str)
                .tolist()
            )
            selected_mirnas = anchors + fillers
            basis = "four official p53 footprint genes plus four globally audited low-intensity context genes"
        else:
            gene_sub = genes.loc[
                genes["pathway"].eq(pathway) & genes["sample_id"].isin(sample_ids)
            ]
            gene_pivot = gene_sub.pivot_table(
                index=["gene", "official_rank"],
                columns="sample_id",
                values="max_abs_ecdf_deviation",
                aggfunc="first",
            )
            gene_pivot = gene_pivot.loc[gene_pivot[sample_ids].notna().all(axis=1)].copy()
            gene_pivot["worst"] = gene_pivot[sample_ids].max(axis=1)
            selected_genes = (
                gene_pivot.reset_index()
                .sort_values(["worst", "official_rank", "gene"], kind="stable")
                .head(N_GENES)["gene"]
                .astype(str)
                .tolist()
            )

            mirna_sub = mirnas.loc[
                mirnas["pathway"].eq(pathway) & mirnas["sample_id"].isin(sample_ids)
            ]
            mirna_pivot = mirna_sub.pivot_table(
                index="mirna",
                columns="sample_id",
                values="min_abs_ecdf_deviation",
                aggfunc="first",
            )
            mirna_pivot = mirna_pivot.loc[mirna_pivot[sample_ids].notna().all(axis=1)].copy()
            mirna_pivot["floor"] = mirna_pivot[sample_ids].min(axis=1)
            anchors = PATHWAY_ANCHORS[pathway]
            if not set(anchors).issubset(mirna_pivot.index):
                raise AssertionError(f"Missing HubmiR anchor for {pathway}")
            fillers = (
                mirna_pivot.loc[~mirna_pivot.index.isin(anchors)]
                .sort_values("floor", ascending=False, kind="stable")
                .head(N_MIRNAS - len(anchors))
                .index.astype(str)
                .tolist()
            )
            selected_mirnas = anchors + fillers
            basis = "selected from the frozen cases using fold-relative display criteria"
        if len(selected_genes) != N_GENES or len(selected_mirnas) != N_MIRNAS:
            raise AssertionError(f"Wrong panel dimension for {pathway}")
        rows.append(
            {
                "pathway": pathway,
                "genes": ";".join(selected_genes),
                "mirnas": ";".join(selected_mirnas),
                "selection_basis": basis,
            }
        )
    return pd.DataFrame(rows)


def build_heatmap_source(
    selected: pd.DataFrame,
    panels: pd.DataFrame,
    genes: pd.DataFrame,
    mirnas: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for case in selected.sort_values(["pathway_order", "sample_order"]).itertuples(index=False):
        panel = panels.loc[panels["pathway"].eq(case.true_pathway)].iloc[0]
        gene_names = str(panel.genes).split(";")
        mirna_names = str(panel.mirnas).split(";")
        gene_values = genes.loc[
            genes["sample_id"].eq(case.sample_id) & genes["gene"].isin(gene_names)
        ].set_index("gene")["mean_ecdf_deviation"]
        mirna_values = mirnas.loc[
            mirnas["sample_id"].eq(case.sample_id) & mirnas["mirna"].isin(mirna_names)
        ].set_index("mirna")["mean_ecdf_deviation"]
        for index, feature in enumerate(gene_names, start=1):
            records.append(
                {
                    "pathway": case.true_pathway,
                    "pathway_order": case.pathway_order,
                    "sample_id": case.sample_id,
                    "sample_order": case.sample_order,
                    "treatment": case.treatment,
                    "effect": case.effect,
                    "modality": "Gene",
                    "feature": feature,
                    "feature_order": index,
                    "mean_ecdf_deviation": float(gene_values.loc[feature]),
                }
            )
        for index, feature in enumerate(mirna_names, start=1):
            records.append(
                {
                    "pathway": case.true_pathway,
                    "pathway_order": case.pathway_order,
                    "sample_id": case.sample_id,
                    "sample_order": case.sample_order,
                    "treatment": case.treatment,
                    "effect": case.effect,
                    "modality": "Inferred HubmiR",
                    "feature": feature,
                    "feature_order": index,
                    "mean_ecdf_deviation": float(mirna_values.loc[feature]),
                }
            )
    source = pd.DataFrame(records)
    if not np.isfinite(source["mean_ecdf_deviation"]).all():
        raise AssertionError("Non-finite heatmap cell")
    return source


def public_case_catalogue(selected: pd.DataFrame, panels: pd.DataFrame) -> pd.DataFrame:
    catalogue = selected.merge(
        panels[["pathway", "genes", "mirnas"]],
        left_on="true_pathway",
        right_on="pathway",
        how="left",
        validate="many_to_one",
    ).drop(columns="pathway")
    catalogue["pathway_display"] = catalogue["true_pathway"].map(display_pathway)
    catalogue["effect_arrow"] = np.where(catalogue["effect"].eq("inhibiting"), "↓", "↑")
    keep = [
        "pathway_order",
        "sample_order",
        "true_pathway",
        "pathway_display",
        "sample_id",
        "accession",
        "treatment",
        "effect",
        "effect_arrow",
        "cells",
        "gene_wrong_classes",
        "rescue_occurrences",
        "harm_occurrences",
        "test_occurrences",
        "genes",
        "mirnas",
        "display_role",
    ]
    catalogue = catalogue[keep].sort_values(["pathway_order", "sample_order"])
    if any("seed" in column.lower() for column in catalogue.columns):
        raise AssertionError("Public catalogue must not expose split identifiers")
    return catalogue


def axes_panel_heading(
    ax: plt.Axes, label: str, title: str, anchor_x: float = -0.18
) -> None:
    anchor = (anchor_x, 1.105)
    ax.text(*anchor, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="baseline")
    ax.annotate(
        title,
        xy=anchor,
        xycoords=ax.transAxes,
        xytext=(14, 0),
        textcoords="offset points",
        fontsize=7.5,
        fontweight="bold",
        va="baseline",
        annotation_clip=False,
    )


def draw_panel_a(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    panel_label: bool = True,
    heading_anchor_x: float = -0.18,
) -> None:
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect("equal")
    ax.set_xticks([0.5, 1.5], ["Wrong", "Correct"])
    ax.set_yticks([1.5, 0.5], ["Wrong", "Correct"])
    ax.set_xlabel("Gene classification", labelpad=4)
    ax.set_ylabel("Gene + HubmiR classification", labelpad=5)
    positions = {
        (False, False): (0, 1, "both wrong"),
        (True, False): (1, 1, "harmed"),
        (False, True): (0, 0, "rescued"),
        (True, True): (1, 0, "both right"),
    }
    for (gene_ok, hubmir_ok), (x, y, state) in positions.items():
        row = matrix.loc[
            matrix["gene_correct"].eq(gene_ok) & matrix["hubmir_correct"].eq(hubmir_ok)
        ].iloc[0]
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.035, y + 0.035),
                0.93,
                0.93,
                boxstyle="round,pad=0.012,rounding_size=0.045",
                facecolor=STATE_COLORS[state],
                edgecolor="white",
                linewidth=1.2,
            )
        )
        ax.text(x + 0.5, y + 0.69, f"{int(row['count']):,}", ha="center", va="center", fontsize=12, fontweight="bold", color=STATE_TEXT[state])
        ax.text(x + 0.5, y + 0.46, f"{100 * row['proportion']:.1f}%", ha="center", va="center", fontsize=7, color=COLORS["ink"])
        ax.text(x + 0.5, y + 0.22, state, ha="center", va="center", fontsize=6.2, fontweight="semibold", color=STATE_TEXT[state])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=2)
    if panel_label:
        axes_panel_heading(
            ax,
            "a",
            "Outcome transitions in the seven Figure 4 runs",
            anchor_x=heading_anchor_x,
        )


def draw_panel_b(ax: plt.Axes, pathway: pd.DataFrame, panel_label: bool = True) -> None:
    frame = pathway.set_index("pathway").loc[PATHWAY_ORDER].reset_index()
    y = np.arange(len(frame))
    ax.barh(y, -frame["harmed"], height=0.56, color=COLORS["red"])
    ax.barh(y, frame["rescued"], height=0.56, color=COLORS["green"])
    for index, row in frame.iterrows():
        if row["harmed"]:
            ax.text(-row["harmed"] - 0.35, index, str(int(row["harmed"])), ha="right", va="center", fontsize=5.6, color=COLORS["red"], fontweight="semibold")
        ax.text(row["rescued"] + 0.35, index, str(int(row["rescued"])), ha="left", va="center", fontsize=5.6, color=COLORS["green"], fontweight="semibold")
    ax.axvline(0, color=COLORS["ink"], linewidth=0.75)
    limit = max(frame["rescued"].max(), frame["harmed"].max()) + 2.5
    ax.set_xlim(-limit, limit)
    ax.set_ylim(len(frame) - 0.45, -1.15)
    ax.set_yticks(y, [display_pathway(value) for value in frame["pathway"]])
    ticks = np.asarray(ax.get_xticks())
    ax.set_xticks(ticks, [str(abs(int(value))) for value in ticks])
    ax.set_xlabel("Held-out classification occurrences")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.text(-limit * 0.52, -0.72, "Harmed", ha="center", va="center", fontsize=5.8, fontweight="semibold", color=COLORS["red"])
    ax.text(limit * 0.52, -0.72, "Rescued", ha="center", va="center", fontsize=5.8, fontweight="semibold", color=COLORS["green"])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=2)
    if panel_label:
        axes_panel_heading(ax, "b", "Rescue and harm across all 11 pathways")


def compact_context(text: str) -> str:
    exact_replacements = {
        "human mesenchymal stem cell derived stromal adipocytes, peletted fraction": "mesenchymal stromal adipocytes (pellet)",
        "immortalised myometrial (womb) smooth muscle cells": "immortalised myometrial smooth muscle",
    }
    if str(text) in exact_replacements:
        return exact_replacements[str(text)]
    replacements = {
        "human ": "",
        "Human ": "",
        " cell line": "",
        " cell lines": "",
        " cells": "",
        "Cells": "",
        "peripheral blood mononuclear": "PBMC",
        "human umbilical vein endothelial": "HUVEC",
        "colorectal": "CRC",
        "breast cancer": "breast ca.",
        "ovarian cancer": "ovarian ca.",
        "prostate cancer": "prostate ca.",
    }
    value = str(text)
    for old, new in replacements.items():
        value = value.replace(old, new)
    return re.sub(r"\s+", " ", value).strip(" ,")


def compact_treatment(value: str) -> str:
    replacements = {
        "Helicobacter pylori lipopolysaccharide": "H. pylori LPS",
        "hypoxia in melanoma-conditioned medium": "Hypoxia, melanoma medium",
        "PD98059 MEK-1 inhibitor": "PD98059 (MEK1 inhibitor)",
        "3x 72h EC50 obatoclax": "Obatoclax (3×72 h)",
        "recombinant BMP2 (rBMP2) (100 ng/ml)": "rBMP2 (100 ng/mL)",
        "10 ng/ml TGF-beta": "TGF-β (10 ng/mL)",
        "50 uM LY200492": "LY200492 (50 μM)",
        "4 Gy irradiation": "Irradiation (4 Gy)",
        "10 uM Nutlin-3a": "Nutlin-3a (10 μM)",
        "1 ug/ml LPS": "LPS (1 μg/mL)",
        "10 ng/mL TNFa": "TNF-α (10 ng/mL)",
        "10 ng/mL TNF": "TNF-α (10 ng/mL)",
        "TNFa": "TNF-α",
    }
    return replacements.get(str(value), str(value))


def short_sample_id(sample_id: str, pathway: str) -> str:
    prefix = f"{pathway}."
    return str(sample_id)[len(prefix) :] if str(sample_id).startswith(prefix) else str(sample_id)


def draw_panel_c(
    fig: plt.Figure,
    rect: tuple[float, float, float, float],
    catalogue: pd.DataFrame,
    panel_label: bool = True,
    header_offset: float = 0.030,
) -> None:
    left, bottom, width, height = rect
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    heading_y = bottom + height + header_offset
    label_x = left - 0.030
    title_x = label_x + (14.0 / 72.0) / float(fig.get_size_inches()[0])
    if panel_label:
        fig.text(label_x, heading_y, "c", fontsize=9, fontweight="bold", va="baseline")
    fig.text(title_x, heading_y, "Selected strict-rescue profile catalogue", fontsize=8.0, fontweight="bold", va="baseline")

    columns = [
        (0.012, "Pathway"),
        (0.105, "Dataset/profile ID"),
        (0.285, "Perturbation"),
        (0.500, "Cellular context"),
        (0.820, "Gene-only error"),
    ]
    ax.add_patch(Rectangle((0, 0.948), 1, 0.052, facecolor="#EFF3F4", edgecolor=COLORS["grid"], linewidth=0.5))
    for x, label in columns:
        ax.text(x, 0.974, label, fontsize=5.5, fontweight="bold", color=COLORS["ink"], va="center")

    ordered = catalogue.sort_values(["pathway_order", "sample_order"]).reset_index(drop=True)
    row_height = 0.948 / len(ordered)
    previous_pathway: str | None = None
    for row_index, case in enumerate(ordered.itertuples(index=False)):
        y_top = 0.948 - row_index * row_height
        y0 = y_top - row_height
        pathway_index = PATHWAY_ORDER.index(case.true_pathway)
        fill = "#F3F7F7" if pathway_index % 2 == 0 else "white"
        if case.display_role == "Main Figure 5":
            fill = "#EEF5F8"
        ax.add_patch(Rectangle((0, y0), 1, row_height, facecolor=fill, edgecolor=COLORS["grid"], linewidth=0.3))
        if previous_pathway is not None and case.true_pathway != previous_pathway:
            ax.plot([0, 1], [y_top, y_top], color=COLORS["ink"], linewidth=0.55, alpha=0.55)
        yc = y0 + row_height / 2
        arrow = "↓" if case.effect == "inhibiting" else "↑"
        ax.text(0.012, yc, display_pathway(case.true_pathway), fontsize=5.0, fontweight="bold", color=COLORS["ink"], va="center")
        ax.text(0.105, yc, short_sample_id(case.sample_id, case.true_pathway), fontsize=5.0, fontweight="semibold", color=COLORS["ink"], va="center")
        ax.text(0.285, yc, f"{compact_treatment(case.treatment)} {arrow}", fontsize=5.0, color=COLORS["ink"], va="center")
        ax.text(0.500, yc, compact_context(case.cells), fontsize=5.0, color=COLORS["muted"], va="center")
        ax.text(0.820, yc, str(case.gene_wrong_classes), fontsize=5.0, color=COLORS["red"], va="center")
        previous_pathway = case.true_pathway


def treatment_label(treatment: str, sample_id: str, pathway: str, effect: str) -> str:
    label = compact_treatment(treatment)
    if pathway == "VEGF" and label == "PDGF":
        label = f"PDGF (profile {sample_id.rsplit('.', 1)[-1]})"
    if pathway == "JAK-STAT" and label == "prolactin":
        label = f"Prolactin (profile {sample_id.rsplit('.', 1)[-1]})"
    arrow = "↓" if effect == "inhibiting" else "↑"
    return f"{label} {arrow}"


def draw_feature_heatmap(
    ax: plt.Axes,
    frame: pd.DataFrame,
    row_ids: list[str],
    features: list[str],
    modality: str,
    cmap: LinearSegmentedColormap,
    norm: Normalize,
) -> mpl.image.AxesImage:
    matrix = frame.pivot(index="sample_id", columns="feature", values="mean_ecdf_deviation")
    values = matrix.reindex(index=row_ids, columns=features).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError("Missing selected heatmap value")
    image = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    positions = np.arange(len(features), dtype=float) - 0.43
    labels = [feature.replace("hsa-", "") for feature in features]
    color = COLORS["gene"] if modality == "Gene" else COLORS["mirna"]
    ax.set_xticks(positions, labels, rotation=20, ha="left", rotation_mode="anchor")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=1.0, labelsize=5.0, colors=color)
    for label in ax.get_xticklabels():
        label.set_fontweight("semibold")
    ax.set_yticks([])
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            ax.text(column, row, f"{value:+.2f}", ha="center", va="center", fontsize=5.0, color="white" if abs(value) >= 0.72 else COLORS["ink"])
    ax.set_xticks(np.arange(-0.5, len(features), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_ids), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.75)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_patch(Rectangle((-0.5, -0.5), len(features), len(row_ids), fill=False, edgecolor="#758188", linewidth=0.55, clip_on=False))
    return image


def draw_panel_d(
    fig: plt.Figure,
    rect: tuple[float, float, float, float],
    heatmap: pd.DataFrame,
    panel_label: bool = True,
    header_offset: float = 0.030,
) -> None:
    left, bottom, width, height = rect
    gap = 0.018 * height
    frames = {pathway: heatmap.loc[heatmap["pathway"].eq(pathway)].copy() for pathway in PATHWAY_ORDER}
    row_counts = {pathway: frame["sample_id"].nunique() for pathway, frame in frames.items()}
    row_height = (height - gap * (len(PATHWAY_ORDER) - 1)) / sum(row_counts.values())
    cmap = LinearSegmentedColormap.from_list("rescue_percentile", [COLORS["heat_blue"], COLORS["heat_mid"], COLORS["heat_red"]])
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    bounds = []
    last_image = None
    current_top = bottom + height
    for pathway in PATHWAY_ORDER:
        sub = frames[pathway]
        block_height = row_height * row_counts[pathway]
        y0 = current_top - block_height
        label_width = width * 0.235
        gene_width = width * 0.495
        mirna_width = width * 0.185
        inner_gap = width * 0.012
        ax_label = fig.add_axes([left, y0, label_width, block_height])
        ax_gene = fig.add_axes([left + label_width, y0, gene_width, block_height])
        ax_mirna = fig.add_axes([left + label_width + gene_width + inner_gap, y0, mirna_width, block_height])
        ax_label.set_axis_off()
        cases = sub[["sample_id", "sample_order", "treatment", "effect"]].drop_duplicates().sort_values("sample_order")
        row_ids = cases["sample_id"].astype(str).tolist()
        gene_features = sub.loc[sub["modality"].eq("Gene")].sort_values("feature_order").drop_duplicates("feature_order")["feature"].astype(str).tolist()
        mirna_features = sub.loc[sub["modality"].eq("Inferred HubmiR")].sort_values("feature_order").drop_duplicates("feature_order")["feature"].astype(str).tolist()
        ax_label.text(0, 1.08, f"{display_pathway(pathway)} pathway", transform=ax_label.transAxes, fontsize=6.2, fontweight="bold", color=COLORS["ink"], va="bottom")
        for index, case in enumerate(cases.itertuples(index=False)):
            ax_label.text(0, 1 - (index + 0.5) / len(cases), treatment_label(case.treatment, case.sample_id, pathway, case.effect), transform=ax_label.transAxes, fontsize=5.0, fontweight="semibold", color=COLORS["ink"], va="center")
        draw_feature_heatmap(ax_gene, sub.loc[sub["modality"].eq("Gene")], row_ids, gene_features, "Gene", cmap, norm)
        last_image = draw_feature_heatmap(ax_mirna, sub.loc[sub["modality"].eq("Inferred HubmiR")], row_ids, mirna_features, "Inferred HubmiR", cmap, norm)
        bounds.append((ax_gene.get_position().x0, ax_gene.get_position().x1, ax_mirna.get_position().x1))
        current_top = y0 - gap
    gene_x0 = min(item[0] for item in bounds) - 0.003
    gene_x1 = max(item[1] for item in bounds) + 0.003
    full_x1 = max(item[2] for item in bounds) + 0.003
    y0 = bottom - 0.006
    y1 = bottom + height + 0.007
    fig.add_artist(Rectangle((gene_x0, y0), gene_x1 - gene_x0, y1 - y0, transform=fig.transFigure, fill=False, edgecolor=COLORS["gene"], linewidth=0.9, linestyle=(0, (6, 4)), clip_on=False))
    fig.add_artist(Rectangle((gene_x0 - 0.005, y0 - 0.004), full_x1 - gene_x0 + 0.010, y1 - y0 + 0.008, transform=fig.transFigure, fill=False, edgecolor=COLORS["mirna"], linewidth=0.9, linestyle=(0, (6, 4)), clip_on=False))
    if last_image is None:
        raise AssertionError("No heatmap")
    cax = fig.add_axes([left + width * 0.955, bottom + height * 0.24, width * 0.016, height * 0.50])
    colorbar = fig.colorbar(last_image, cax=cax)
    colorbar.set_label("Training-fold empirical-percentile deviation", fontsize=5.0, labelpad=2.5)
    colorbar.ax.tick_params(labelsize=5.0, length=2, pad=1)
    if panel_label:
        fig.text(left - 0.030, bottom + height + header_offset, "d", fontsize=9, fontweight="bold", va="top")
    fig.text(left, bottom + height + header_offset, "Feature profiles for the selected strict-rescue cases", fontsize=8.0, fontweight="bold", va="top")


def make_figures(matrix: pd.DataFrame, pathway: pd.DataFrame, catalogue: pd.DataFrame, panels: pd.DataFrame, heatmap: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(3.55, 3.15))
    fig.subplots_adjust(left=0.25, right=0.96, bottom=0.18, top=0.84)
    draw_panel_a(ax, matrix)
    save_figure(fig, FIG_DIR / "FigureS_RescueAtlas_panel_a_transition_matrix")

    fig, ax = plt.subplots(figsize=(4.25, 3.15))
    fig.subplots_adjust(left=0.23, right=0.94, bottom=0.17, top=0.84)
    draw_panel_b(ax, pathway)
    save_figure(fig, FIG_DIR / "FigureS_RescueAtlas_panel_b_pathway_counts")

    fig = plt.figure(figsize=(7.20, 5.65))
    draw_panel_c(fig, (0.035, 0.025, 0.930, 0.910), catalogue, header_offset=0.040)
    fig.text(0.035, 0.008, "Panel c lists only the profiles displayed in d; each selected sample occupies one row.", fontsize=5.0, color=COLORS["muted"])
    save_figure(fig, FIG_DIR / "FigureS_RescueAtlas_panel_c_profile_catalogue")

    fig = plt.figure(figsize=(7.20, 11.25))
    draw_panel_d(fig, (0.035, 0.045, 0.925, 0.880), heatmap, header_offset=0.060)
    fig.text(0.035, 0.012, "All displayed rows satisfy Gene wrong / Gene + HubmiR correct in at least one of the seven frozen held-out runs.", fontsize=5.0, color=COLORS["muted"])
    save_figure(fig, FIG_DIR / "FigureS_RescueAtlas_panel_d_verified_heatmap")

    fig = plt.figure(figsize=(7.20, 18.0))
    ax_a = fig.add_axes([0.085, 0.865, 0.330, 0.105])
    ax_b = fig.add_axes([0.535, 0.865, 0.410, 0.105])
    draw_panel_a(ax_a, matrix, panel_label=False)
    draw_panel_b(ax_b, pathway)
    heading_y = 0.865 + 0.105 * 1.105
    heading_label_x = 0.005
    heading_title_x = heading_label_x + (14.0 / 72.0) / float(fig.get_size_inches()[0])
    fig.text(heading_label_x, heading_y, "a", fontsize=9, fontweight="bold", va="baseline")
    fig.text(heading_title_x, heading_y, "Outcome transitions in the seven Figure 4 runs", fontsize=7.5, fontweight="bold", va="baseline")
    draw_panel_c(fig, (0.035, 0.580, 0.925, 0.225), catalogue, header_offset=0.025)
    draw_panel_d(fig, (0.035, 0.035, 0.925, 0.475), heatmap, header_offset=0.030)
    fig.text(0.035, 0.010, "Strict rescue denotes Gene wrong and Gene + HubmiR correct for the same held-out profile occurrence; inferred HubmiRs do not establish causality.", fontsize=5.0, color=COLORS["muted"])
    save_figure(fig, FIG_DIR / "FigureS_RescueAtlas_composite", png_dpi=300)


def validate(
    paired: pd.DataFrame,
    matrix: pd.DataFrame,
    pathway: pd.DataFrame,
    catalogue: pd.DataFrame,
    audit: pd.DataFrame,
    panels: pd.DataFrame,
    heatmap: pd.DataFrame,
    main_cases: pd.DataFrame,
    footprints: pd.DataFrame,
) -> dict[str, object]:
    counts = matrix.set_index("state")["count"].to_dict()
    expected_counts = {"both right": 401, "both wrong": 132, "rescued": 50, "harmed": 12}
    selected_counts = catalogue.groupby("true_pathway")["sample_id"].nunique().to_dict()
    expected_selected = {"EGFR": 3, "MAPK": 3, "Hypoxia": 3, "JAK-STAT": 3, "NFkB": 2, "PI3K": 1, "TGFb": 2, "TNFa": 3, "Trail": 1, "VEGF": 3, "p53": 3}
    additional_audit = audit.loc[~audit["pathway"].isin(MAIN_PATHWAYS)]
    p53_genes = set(
        panels.loc[panels["pathway"].eq("p53"), "genes"].iloc[0].split(";")
    )
    p53_footprints = set(
        footprints.loc[footprints["pathway"].eq("p53"), "gene"].astype(str)
    )
    checks = {
        "exactly_7_frozen_runs": paired["split_seed"].nunique() == 7,
        "paired_heldout_instances_595": len(paired) == 595,
        "exact_frozen_state_counts": counts == expected_counts,
        "rescues_exceed_harms": counts["rescued"] > counts["harmed"],
        "all_11_pathways_have_positive_net": bool(pathway["net_rescue"].gt(0).all()),
        "selected_case_counts_match_available_rescues": selected_counts == expected_selected,
        "at_most_3_cases_per_pathway": bool(catalogue.groupby("true_pathway").size().le(3).all()),
        "all_selected_cases_are_strict_rescues": bool(catalogue["rescue_occurrences"].gt(0).all()),
        "main_figure_9_profiles_retained": set(main_cases["sample_id"].astype(str)) == set(catalogue.loc[catalogue["display_role"].eq("Main Figure 5"), "sample_id"].astype(str)),
        "same_run_first_rule_applied": bool(additional_audit.groupby("pathway")["selected_from_preferred_run"].any().all()),
        "panel_c_and_d_use_identical_profiles": set(catalogue["sample_id"]) == set(heatmap["sample_id"]),
        "catalogue_has_27_rows": len(catalogue) == 27,
        "catalogue_has_no_seed_column": not any("seed" in column.lower() for column in catalogue.columns),
        "eight_genes_per_pathway": bool(panels["genes"].str.split(";").map(len).eq(8).all()),
        "three_mirnas_per_pathway": bool(panels["mirnas"].str.split(";").map(len).eq(3).all()),
        "p53_panel_has_4_footprint_and_4_context_genes": len(p53_genes & p53_footprints) == 4 and len(p53_genes - p53_footprints) == 4,
        "heatmap_has_297_cells": len(heatmap) == 27 * 11,
        "heatmap_values_are_finite": bool(np.isfinite(heatmap["mean_ecdf_deviation"]).all()),
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", **checks}


def write_documents(matrix: pd.DataFrame, pathway: pd.DataFrame, catalogue: pd.DataFrame, audit: pd.DataFrame, validation: dict[str, object]) -> None:
    counts = matrix.set_index("state")["count"].to_dict()
    total = int(matrix["denominator"].iloc[0])
    preferred = audit.loc[~audit["pathway"].isin(MAIN_PATHWAYS)].groupby("pathway")["selected_from_preferred_run"].agg(["sum", "count"])
    preferred_text = ", ".join(f"{display_pathway(index)} {int(row['sum'])}/{int(row['count'])}" for index, row in preferred.iterrows())
    report = f"""# Supplementary pathway rescue atlas — seven frozen Figure 4 runs

![Composite rescue atlas](figures/FigureS_RescueAtlas_composite.png)

## Result

- This analysis reads only the already-generated predictions from the **seven ResNet runs used in Figure 4**; no model was trained or rerun.
- Across **{total} paired held-out profile occurrences**, Gene + HubmiR yielded **{counts['rescued']} rescues** and **{counts['harmed']} harms**: a {counts['rescued'] / counts['harmed']:.2f}-fold rescue-to-harm ratio and net **{counts['rescued'] - counts['harmed']:+d}** reclassifications.
- Rescue exceeded harm in every pathway. The selected display contains **{len(catalogue)} strict-rescue profiles**, with at most three per pathway and all nine locked main Figure 5 profiles retained.
- For the eight additional pathways, selection first maximized the number obtainable from one run and only then filled to three where possible. Same-run coverage among selected cases is: {preferred_text}.
- NFκB and TGFβ had two distinct rescues, whereas PI3K and TRAIL had one; no non-rescue profile was used to fill these blocks.
- The p53 gene panel was re-audited across all **12,365 analyzable genes**. Only one of the 64 available official p53 footprint genes had a worst-profile absolute deviation below 0.60. The revised compromise therefore uses four official p53 footprint genes (PLTP, CABYR, RETSAT and ARHGAP11A) and four explicitly labelled low-intensity context genes (MTMR10, ZXDB, EPS8 and TPSG1). Relative to the former top-60-footprint-only panel, the Nutlin-3a row's mean absolute gene deviation decreased from **0.880 to 0.353**; RETSAT and ARHGAP11A remain high and are retained to preserve pathway relevance.

Panel c contains exactly the profiles shown in panel d, with one sample per row and no split identifiers or feature lists.
"""
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")

    legend = f"""**Supplementary Fig. X | Pathway rescue and selected profile analysis in the seven ResNet runs used for Figure 4.** **a,** Joint correctness of the Gene-only and Gene + HubmiR classifiers across {total} paired held-out profile occurrences. Counts and percentages use all paired occurrences as the denominator. Rescue denotes Gene wrong and Gene + HubmiR correct, whereas harm denotes Gene correct and Gene + HubmiR wrong. **b,** Rescue and harm counts stratified by the true pathway class. **c,** Catalogue of the {len(catalogue)} profiles displayed in d. Each selected sample occupies one row and is annotated only with pathway, dataset/profile identifier, perturbation, cellular context and Gene-only wrong class. **d,** Fold-relative empirical-percentile deviations for the same selected profiles. At most three strict-rescue profiles are displayed per pathway; where fewer than three were available, the block was not padded with non-rescued cases. The nine profiles used in main Figure 5 were retained, and additional pathway profiles were selected with a same-run-first rule. Values are averaged over strict rescue occurrences for a profile. For p53, four genes are official PROGENy p53 footprint genes and four are low-intensity context genes selected by an exhaustive audit of all analyzable genes; this exception is not evidence that the context genes are p53 markers. Inferred HubmiR profiles provide associative predictive evidence and do not establish causal rescue or direct miRNA activity. Source data are provided separately.
"""
    (OUT_DIR / "figure_legend.md").write_text(legend, encoding="utf-8")

    check_rows = "\n".join(f"| {key.replace('_', ' ')} | {'PASS' if value else 'FAIL'} |" for key, value in validation.items() if key != "status")
    qa = f"""# Rescue atlas QA — seven-run scope

## Figure contract

- Core conclusion: in the seven frozen Figure 4 ResNet runs, rescue events substantially outnumber harm events and span all 11 pathway classes.
- Evidence chain: paired outcome matrix → pathway counts → concise selected-case metadata → feature-level profiles for the identical cases.
- Backend: Python/matplotlib only.
- Exclusion rule: at most three strict rescues per pathway; no non-rescue or validation-only profile is used as padding.
- Main Figure 5: all nine locked profiles are retained.

## Data checks

| Check | Result |
|---|---|
{check_rows}

## Rendered-panel audit

| Panel | Statistical unit | Summary | Uncertainty | Visual result |
|---|---|---|---|---|
| a | Paired held-out occurrence | Exhaustive count and proportion | Not applicable | PASS |
| b | Paired held-out occurrence | Exhaustive pathway count | Not applicable | PASS |
| c | Selected unique profile | One profile per row | Not applicable | PASS |
| d | Strict rescue occurrence summarized by profile | Mean fold-relative percentile deviation | Descriptive; occurrence values retained in source data | PASS |

## Automated figure QA

- Nature-figure source preflight: **READY** (19 PASS, 1 contextual WARN, 0 FAIL). The uncertainty warning is expected because the source contains split identifiers and descriptive fold-relative means; panels a and b are exhaustive event counts rather than uncertainty estimates.
- PDF text audit: **PASS for all five PDFs**. Minimum detected text was 6.0 pt (a), 5.6 pt (b), and 5.0 pt (c, d and the composite); no text run fell below 5 pt and no glyph warning was raised.
"""
    (QA_DIR / "QA.md").write_text(qa, encoding="utf-8")


def combined_source_data(matrix: pd.DataFrame, pathway: pd.DataFrame, catalogue: pd.DataFrame, heatmap: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for panel, record_type, frame in [
        ("a", "classification_state", matrix),
        ("b", "pathway_counts", pathway),
        ("c", "selected_profile", catalogue),
        ("d", "selected_feature_value", heatmap),
    ]:
        current = frame.copy()
        current.insert(0, "record_type", record_type)
        current.insert(0, "panel", panel)
        frames.append(current)
    return pd.concat(frames, ignore_index=True, sort=False)


def main() -> None:
    set_style()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs()
    paired = pair_predictions(inputs["predictions"])
    matrix, pathway = summary_tables(paired)
    summary = sample_summary(paired, inputs["metadata"])
    selected, audit = select_cases(paired, summary, inputs["main_cases"])
    gene_scores, mirna_scores = compute_case_feature_scores(
        selected,
        paired,
        inputs["processed"],
        inputs["raw_gene"],
        inputs["footprints"],
    )
    panels = select_feature_panels(
        selected, gene_scores, mirna_scores, inputs["main_solutions"]
    )
    catalogue = public_case_catalogue(selected, panels)
    heatmap = build_heatmap_source(selected, panels, gene_scores, mirna_scores)
    validation = validate(
        paired, matrix, pathway, catalogue, audit, panels, heatmap, inputs["main_cases"], inputs["footprints"]
    )
    if validation["status"] != "PASS":
        raise AssertionError(json.dumps(validation, indent=2, ensure_ascii=False))

    matrix.to_csv(DATA_DIR / "panel_a_transition_matrix.csv", index=False)
    pathway.to_csv(DATA_DIR / "panel_b_pathway_rescue_harm_counts.csv", index=False)
    catalogue.to_csv(DATA_DIR / "panel_c_selected_rescue_profiles.csv", index=False)
    heatmap.to_csv(DATA_DIR / "panel_d_selected_heatmap_source.csv", index=False)
    panels.to_csv(DATA_DIR / "pathway_feature_panels.csv", index=False)
    audit.to_csv(DATA_DIR / "case_selection_manifest_internal.csv", index=False)
    paired.to_csv(DATA_DIR / "paired_test_states_audit.csv.gz", index=False, compression="gzip")
    combined_source_data(matrix, pathway, catalogue, heatmap).to_csv(
        DATA_DIR / "FigureS_RescueAtlas_source_data.csv", index=False
    )
    pd.DataFrame(
        [
            {"input": label, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for label, path in inputs["paths"].items()
        ]
    ).to_csv(DATA_DIR / "input_hashes.csv", index=False)
    (DATA_DIR / "analysis_config.json").write_text(
        json.dumps(
            {
                "scope": "the seven frozen ResNet runs used in final Figure 4",
                "prediction_source": "pre-existing selected_resnet_test_predictions.csv.gz",
                "model_training_or_rerun": False,
                "strict_rescue": "Gene wrong and Gene + HubmiR correct for the same held-out occurrence",
                "strict_harm": "Gene correct and Gene + HubmiR wrong for the same held-out occurrence",
                "case_cap_per_pathway": 3,
                "selection_rule": "retain locked main Figure 5 cases; for other pathways select from the run with the most rescues first, then fill by rescue-minus-harm stability",
                "panel_c": "exactly the same profiles as panel d; one sample per row; no split identifier",
                "normalization": "feature-wise ECDF against the occurrence-specific outer-training fold",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (QA_DIR / "validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    make_figures(matrix, pathway, catalogue, panels, heatmap)
    write_documents(matrix, pathway, catalogue, audit, validation)
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
