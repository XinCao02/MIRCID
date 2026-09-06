from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT, atomic_write_csv, load_processed, read_csv_auto, write_json


FIGURE_WORK = ROOT / "paper_exp" / "figure5_rescue"
SUPP_WORK = ROOT / "additional_exp" / "5.4_expanded_rescue"


def analyze_model(model: str, predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    test = predictions[predictions["partition"] == "test"].copy()
    gene = test[test["feature_space"] == "Gene"].copy()
    hub = test[test["feature_space"] == "Gene + HubmiR"].copy()
    keys = ["split_seed", "sample_id"]
    merged = gene.merge(hub, on=keys, suffixes=("_gene", "_hub"), validate="one_to_one")
    if len(merged) != 20 * 85:
        raise ValueError(f"Expected 1,700 paired test observations for {model}; got {len(merged)}")
    if not np.array_equal(merged.true_label_gene, merged.true_label_hub):
        raise ValueError("Truth mismatch")
    g, h = merged.correct_gene.astype(bool), merged.correct_hub.astype(bool)
    merged["state"] = np.select(
        [~g & h, g & ~h, g & h], ["rescued", "harmed", "both_correct"], default="both_wrong"
    )
    merged["model"] = model
    merged["true_pathway"] = merged["true_pathway_gene"]
    merged["gene_prediction"] = merged["pred_pathway_gene"]
    merged["hubmir_prediction"] = merged["pred_pathway_hub"]
    state = merged[["model", "split_seed", "sample_id", "true_label_gene", "true_pathway", "gene_prediction", "hubmir_prediction", "correct_gene", "correct_hub", "state"]].rename(columns={"true_label_gene": "true_label"})

    rows = []
    for (seed, pathway), sub in state.groupby(["split_seed", "true_pathway"]):
        counts = sub.state.value_counts()
        rows.append(
            {"model": model, "split_seed": seed, "pathway": pathway, "test_observations": len(sub),
             "rescued": int(counts.get("rescued", 0)), "harmed": int(counts.get("harmed", 0)),
             "both_correct": int(counts.get("both_correct", 0)), "both_wrong": int(counts.get("both_wrong", 0)),
             "rescue_rate": float((sub.state == "rescued").mean()), "harm_rate": float((sub.state == "harmed").mean()),
             "net_reclassification_rate": float(((sub.state == "rescued").astype(int) - (sub.state == "harmed").astype(int)).mean())}
        )
    by_seed_pathway = pd.DataFrame(rows)
    pathway_summary = state.groupby("true_pathway").state.value_counts().unstack(fill_value=0)
    for column in ["rescued", "harmed", "both_correct", "both_wrong"]:
        if column not in pathway_summary: pathway_summary[column] = 0
    pathway_summary["test_observations"] = pathway_summary.sum(axis=1)
    pathway_summary["rescue_rate"] = pathway_summary["rescued"] / pathway_summary["test_observations"]
    pathway_summary["harm_rate"] = pathway_summary["harmed"] / pathway_summary["test_observations"]
    pathway_summary["net_reclassification"] = pathway_summary["rescued"] - pathway_summary["harmed"]
    pathway_summary["net_reclassification_rate"] = pathway_summary["net_reclassification"] / pathway_summary["test_observations"]
    pathway_summary = pathway_summary.reset_index().assign(model=model)

    overall_rows = []
    for seed, sub in state.groupby("split_seed"):
        counts = sub.state.value_counts()
        overall_rows.append(
            {"model": model, "split_seed": seed, "test_observations": len(sub),
             "rescued": int(counts.get("rescued", 0)), "harmed": int(counts.get("harmed", 0)),
             "net_reclassification": int(counts.get("rescued", 0) - counts.get("harmed", 0)),
             "gene_accuracy": float(sub.correct_gene.mean()), "gene_hubmir_accuracy": float(sub.correct_hub.mean()),
             "paired_accuracy_delta": float(sub.correct_hub.mean() - sub.correct_gene.mean())}
        )
    overall = pd.DataFrame(overall_rows)

    stability_rows = []
    for (sample_id, pathway), sub in state.groupby(["sample_id", "true_pathway"]):
        counts = sub.state.value_counts(); n = len(sub)
        rescue = int(counts.get("rescued", 0)); harm = int(counts.get("harmed", 0))
        stable_rescue = n >= 2 and rescue >= 2 and rescue / n >= 0.50 and harm == 0
        stable_harm = n >= 2 and harm >= 2 and harm / n >= 0.50 and rescue == 0
        stability_rows.append(
            {"model": model, "sample_id": sample_id, "pathway": pathway, "test_occurrences": n,
             "rescue_count": rescue, "harm_count": harm, "both_correct_count": int(counts.get("both_correct", 0)),
             "both_wrong_count": int(counts.get("both_wrong", 0)), "rescue_fraction": rescue / n,
             "harm_fraction": harm / n, "stable_rescue": stable_rescue, "stable_harm": stable_harm}
        )
    stability = pd.DataFrame(stability_rows)

    transitions = state.groupby(["true_pathway", "gene_prediction", "hubmir_prediction", "state"]).size().rename("observations").reset_index().assign(model=model)
    return {"states": state, "by_seed_pathway": by_seed_pathway, "pathway": pathway_summary, "overall": overall, "stability": stability, "transitions": transitions}


def main() -> None:
    data = load_processed()
    result_dir = ROOT / "paper_exp" / "figure4_benchmark" / "results"
    sources = {
        "ResNet": result_dir / "resnet_predictions.csv.gz",
        "SVM": result_dir / "svm_predictions.csv.gz",
    }
    combined: dict[str, list[pd.DataFrame]] = {}
    for model, path in sources.items():
        output = analyze_model(model, read_csv_auto(path))
        for name, frame in output.items(): combined.setdefault(name, []).append(frame)
    FIGURE_WORK.joinpath("results").mkdir(parents=True, exist_ok=True)
    SUPP_WORK.joinpath("results").mkdir(parents=True, exist_ok=True)
    filenames = {
        "states": "paired_rescue_states.csv.gz", "by_seed_pathway": "pathway_seed_reclassification.csv",
        "pathway": "pathway_reclassification_summary.csv", "overall": "seed_reclassification_summary.csv",
        "stability": "sample_rescue_stability.csv", "transitions": "class_transition_summary.csv",
    }
    for name, frames in combined.items():
        frame = pd.concat(frames, ignore_index=True)
        atomic_write_csv(frame, FIGURE_WORK / "results" / filenames[name])
        atomic_write_csv(frame, SUPP_WORK / "results" / filenames[name])

    stability = pd.concat(combined["stability"], ignore_index=True)
    meta = pd.DataFrame(
        {"sample_id": data["sample_ids"].astype(str), "accession": data["accession"].astype(str),
         "cells": data["cells"].astype(str), "treatment": data["treatment"].astype(str),
         "effect": data["effect"].astype(str)}
    )
    cases = stability[(stability.stable_rescue) | (stability.stable_harm)].merge(meta, on="sample_id", how="left")
    atomic_write_csv(cases, FIGURE_WORK / "results" / "locked_stable_cases.csv")
    atomic_write_csv(cases, SUPP_WORK / "results" / "locked_stable_cases.csv")

    write_json(
        FIGURE_WORK / "results" / "rescue_manifest.json",
        {"status": "complete", "primary_model": "ResNet", "sensitivity_model": "SVM",
         "definition": {"rescued": "Gene wrong and Gene+HubmiR correct", "harmed": "Gene correct and Gene+HubmiR wrong"},
         "stable_rescue_rule": "test_occurrences>=2, rescue_count>=2, rescue_fraction>=0.50, harm_count=0",
         "stable_harm_rule": "test_occurrences>=2, harm_count>=2, harm_fraction>=0.50, rescue_count=0",
         "case_selection_used_mirna_values_or_literature": False, "models_and_test_splits_frozen_before_analysis": True},
    )
    print(cases.groupby(["model", "pathway", "stable_rescue", "stable_harm"]).size().to_string())


if __name__ == "__main__": main()
