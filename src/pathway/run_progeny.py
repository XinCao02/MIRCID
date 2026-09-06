from __future__ import annotations

import numpy as np
import pandas as pd

from mircid.paths import DATA_ROOT

from .common import ROOT, atomic_write_csv, load_config, load_processed, metric_rows, metrics, split_indices


def main() -> None:
    cfg = load_config()
    data = load_processed()
    weights = pd.read_csv(DATA_ROOT / "pathway" / "processed" / "progeny_top100_matrix.csv").set_index("gene")
    gene_names = data["gene_names"].astype(str)
    gene_pos = {gene: i for i, gene in enumerate(gene_names)}
    common = [gene for gene in weights.index.astype(str) if gene in gene_pos]
    if not common:
        raise ValueError("No overlap between official PROGENy model and gene matrix")
    x = data["gene"][:, [gene_pos[g] for g in common]].astype(np.float64)
    w = weights.loc[common, data["class_names"].astype(str)].to_numpy(dtype=np.float64)
    raw_scores = x @ w

    coverage_rows = []
    for j, pathway in enumerate(data["class_names"].astype(str)):
        nonzero = weights[pathway] != 0
        used = int(sum(g in gene_pos for g in weights.index[nonzero].astype(str)))
        coverage_rows.append({"pathway": pathway, "official_top_genes": int(nonzero.sum()), "genes_present": used, "coverage": used / int(nonzero.sum())})
    coverage = pd.DataFrame(coverage_rows)
    if coverage["genes_present"].min() < 80:
        raise ValueError(f"Insufficient PROGENy footprint coverage:\n{coverage}")
    atomic_write_csv(coverage, ROOT / "paper_exp" / "figure4_benchmark" / "results" / "progeny_gene_coverage.csv")

    metric_records = []
    prediction_records = []
    score_records = []
    y = data["y"]
    sample_ids = data["sample_ids"].astype(str)
    class_names = data["class_names"].astype(str)
    for seed in cfg["split_seeds"]:
        split = split_indices(seed, data)
        train = split["train"]
        mean = raw_scores[train].mean(axis=0)
        std = raw_scores[train].std(axis=0, ddof=0)
        std[std == 0] = 1.0
        z = (raw_scores - mean) / std
        test = split["test"]
        for rule, pred in {
            "direction_agnostic_absmax": np.abs(z[test]).argmax(axis=1),
            "direction_aware_perturbation_recall": (z[test] * data["sign"][test, None]).argmax(axis=1),
        }.items():
            model_name = f"PROGENy ({rule})"
            vals = metrics(y[test], pred)
            metric_records.extend(metric_rows(model_name, "Gene", seed, "test", vals))
            for idx, truth, predicted in zip(test, y[test], pred):
                prediction_records.append(
                    {
                        "run_id": cfg["run_id"], "model": model_name, "feature_space": "Gene",
                        "split_seed": seed, "partition": "test", "sample_id": sample_ids[idx],
                        "true_label": int(truth), "pred_label": int(predicted),
                        "true_pathway": class_names[truth], "pred_pathway": class_names[predicted],
                        "effect_used_for_prediction": bool(rule.startswith("direction_aware")),
                        "correct": bool(truth == predicted),
                    }
                )
        for idx in test:
            for j, pathway in enumerate(class_names):
                score_records.append(
                    {"split_seed": seed, "sample_id": sample_ids[idx], "pathway": pathway,
                     "raw_score": raw_scores[idx, j], "train_standardized_score": z[idx, j]}
                )

    out = ROOT / "paper_exp" / "figure4_benchmark" / "results"
    atomic_write_csv(pd.DataFrame(metric_records), out / "progeny_metrics.csv")
    atomic_write_csv(pd.DataFrame(prediction_records), out / "progeny_predictions.csv")
    atomic_write_csv(pd.DataFrame(score_records), out / "progeny_scores.csv.gz")

    # Exact official package default: full-cohort column scaling. This is kept
    # only as a provenance/recall audit, never mixed into held-out Figure 4.
    full_z = (raw_scores - raw_scores.mean(axis=0)) / raw_scores.std(axis=0, ddof=0)
    full_direction = (full_z * data["sign"][:, None]).argmax(axis=1)
    full_fair = np.abs(full_z).argmax(axis=1)
    pd.DataFrame(
        [
            {"evaluation": "full_cohort_official_direction_aware", **metrics(y, full_direction), "effect_used": True},
            {"evaluation": "full_cohort_direction_agnostic_absmax", **metrics(y, full_fair), "effect_used": False},
        ]
    ).to_csv(out / "progeny_full_cohort_audit.csv", index=False)
    print(coverage.to_string(index=False))
    print(pd.DataFrame(metric_records).query("metric in ['accuracy','macro_f1']").groupby(["model", "metric"]).value.agg(["mean", "std"]))


if __name__ == "__main__":
    main()
