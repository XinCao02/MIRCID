from __future__ import annotations
import os

import math
from pathlib import Path

import numpy as np
import pandas as pd

from mircid.paths import DATA_ROOT

from .common import ROOT, atomic_write_csv, load_processed, read_csv_auto, sha256_file, write_json


FIG4 = ROOT / "paper_exp" / "figure4_benchmark"
EMBED = ROOT / "additional_exp" / "5.1_embedding_controls"
COMP = ROOT / "additional_exp" / "5.2_complementarity"
RESCUE = ROOT / "additional_exp" / "5.4_expanded_rescue"


def bootstrap_mean_ci(values: np.ndarray, seed: int, iterations: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(values), size=(iterations, len(values)))
    means = values[sample].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def exact_signflip(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    n = len(values); observed = abs(float(values.mean())); extreme = 0; total = 1 << n
    bit_positions = np.arange(n, dtype=np.uint64)
    for start in range(0, total, 65536):
        ids = np.arange(start, min(total, start + 65536), dtype=np.uint64)
        signs = 1.0 - 2.0 * ((ids[:, None] >> bit_positions[None, :]) & 1).astype(np.float64)
        null = np.abs((signs @ values) / n)
        extreme += int(np.sum(null >= observed - 1e-15))
    return extreme / total


def holm(values: pd.Series) -> pd.Series:
    raw = values.to_numpy(float); order = np.argsort(raw); ranked = raw[order]
    adjusted_ranked = np.maximum.accumulate(np.minimum(1.0, (len(raw) - np.arange(len(raw))) * ranked))
    adjusted = np.empty_like(raw); adjusted[order] = adjusted_ranked
    return pd.Series(adjusted, index=values.index)


def absolute_summary(metrics: pd.DataFrame, seed_offset: int) -> pd.DataFrame:
    rows = []
    for i, (keys, sub) in enumerate(metrics.groupby(["model", "feature_space", "metric"])):
        vals = sub.sort_values("split_seed").value.to_numpy(float); lo, hi = bootstrap_mean_ci(vals, seed_offset + i)
        rows.append({"model": keys[0], "feature_space": keys[1], "metric": keys[2], "n_seeds": len(vals),
                     "mean": vals.mean(), "sd": vals.std(ddof=1), "ci95_low": lo, "ci95_high": hi,
                     "median": np.median(vals), "minimum": vals.min(), "maximum": vals.max()})
    return pd.DataFrame(rows)


def paired_contrast(metrics: pd.DataFrame, pairs: list[tuple[str, str]], seed_offset: int) -> pd.DataFrame:
    rows = []
    for model in metrics.model.unique():
        subm = metrics[metrics.model == model]
        for metric in subm.metric.unique():
            wide = subm[subm.metric == metric].pivot_table(index="split_seed", columns="feature_space", values="value")
            for j, (numerator, denominator) in enumerate(pairs):
                if numerator not in wide or denominator not in wide: continue
                delta = (wide[numerator] - wide[denominator]).dropna().to_numpy(float)
                if len(delta) == 0: continue
                lo, hi = bootstrap_mean_ci(delta, seed_offset + len(rows) + j)
                sd = delta.std(ddof=1)
                rows.append({"model": model, "metric": metric, "contrast": f"{numerator} − {denominator}",
                             "numerator": numerator, "denominator": denominator, "n_pairs": len(delta),
                             "mean_delta": delta.mean(), "ci95_low": lo, "ci95_high": hi,
                             "paired_effect_dz": delta.mean() / sd if sd > 0 else np.nan,
                             "positive_pairs": int(np.sum(delta > 0)), "negative_pairs": int(np.sum(delta < 0)),
                             "ties": int(np.sum(delta == 0)), "exact_signflip_p": exact_signflip(delta)})
    result = pd.DataFrame(rows)
    if len(result):
        result["holm_p_within_model_metric"] = result.groupby(["model", "metric"], group_keys=False)["exact_signflip_p"].apply(holm)
    return result


def main_metrics() -> pd.DataFrame:
    frames = []
    for name in ["rf", "svm", "mlp", "kan", "resnet", "progeny"]:
        path = FIG4 / "results" / f"{name}_metrics.csv"
        if path.exists(): frames.append(pd.read_csv(path))
    data = pd.concat(frames, ignore_index=True)
    data["model"] = data["model"].replace({"RESNET": "ResNet"})
    data = data[(data.partition == "test") & data.metric.isin(["accuracy", "macro_f1"])].copy()
    data = data[~data.model.str.contains("direction_aware", regex=False)].copy()
    atomic_write_csv(data, FIG4 / "results" / "main_test_metrics.csv")
    summary = absolute_summary(data, 101)
    atomic_write_csv(summary, FIG4 / "results" / "main_absolute_summary.csv")
    contrasts = paired_contrast(
        data[~data.model.str.startswith("PROGENy")],
        [("Gene + HubmiR", "Gene"), ("Gene + TFA", "Gene"), ("All features", "Gene")], 201,
    )
    atomic_write_csv(contrasts, FIG4 / "results" / "main_paired_contrasts.csv")
    return data


def embedding_analysis(main: pd.DataFrame) -> None:
    frames = []
    for model in ["svm", "resnet"]:
        path = EMBED / "results" / f"{model}_control_metrics.csv"
        if path.exists(): frames.append(pd.read_csv(path))
    controls = pd.concat(frames, ignore_index=True)
    controls["model"] = controls.model.replace({"RESNET": "ResNet"})
    controls = controls[(controls.partition == "test") & controls.metric.isin(["accuracy", "macro_f1"])]
    baseline = main[(main.model.isin(["SVM", "ResNet"])) & (main.feature_space.isin(["Gene", "Gene + HubmiR"]))]
    all_metrics = pd.concat([baseline, controls], ignore_index=True)
    rp = all_metrics[all_metrics.feature_space.str.startswith("Gene + RP-414")].copy()
    rp_collapsed = rp.groupby(["model", "split_seed", "partition", "metric"], as_index=False).value.mean()
    rp_collapsed["feature_space"] = "Gene + RP-414 (5-seed mean)"
    collapsed = pd.concat([all_metrics[~all_metrics.feature_space.str.startswith("Gene + RP-414")], rp_collapsed], ignore_index=True)
    atomic_write_csv(all_metrics, EMBED / "results" / "embedding_test_metrics_all_rp_seeds.csv")
    atomic_write_csv(collapsed, EMBED / "results" / "embedding_test_metrics_rp_collapsed.csv")
    atomic_write_csv(absolute_summary(collapsed, 301), EMBED / "results" / "embedding_absolute_summary.csv")
    pairs = [("Gene + HubmiR", "Gene"), ("Gene + HubmiR", "Gene + PCA-414"),
             ("Gene + HubmiR", "Gene + AE-414"), ("Gene + HubmiR", "Gene + RP-414 (5-seed mean)")]
    atomic_write_csv(paired_contrast(collapsed, pairs, 401), EMBED / "results" / "embedding_paired_contrasts.csv")


def complementarity_analysis(main: pd.DataFrame) -> None:
    frames = []
    for model in ["svm", "resnet"]:
        path = COMP / "results" / f"{model}_decomposition_metrics.csv"
        if path.exists(): frames.append(pd.read_csv(path))
    new = pd.concat(frames, ignore_index=True); new["model"] = new.model.replace({"RESNET": "ResNet"})
    new = new[(new.partition == "test") & new.metric.isin(["accuracy", "macro_f1"])]
    baseline = main[(main.model.isin(["SVM", "ResNet"])) & (main.feature_space.isin(["Gene", "Gene + HubmiR"]))]
    all_metrics = pd.concat([baseline, new], ignore_index=True)
    atomic_write_csv(all_metrics, COMP / "results" / "decomposition_test_metrics_all.csv")
    pairs = [("Gene + residual HubmiR", "Gene"), ("Gene + HubmiR", "Gene + predicted HubmiR"),
             ("Gene + predicted HubmiR", "Gene"), ("Gene + HubmiR", "Gene")]
    atomic_write_csv(paired_contrast(all_metrics, pairs, 501), COMP / "results" / "decomposition_paired_contrasts.csv")
    atomic_write_csv(absolute_summary(all_metrics, 601), COMP / "results" / "decomposition_absolute_summary.csv")
    for kind, value_col, groups in [
        ("cka", "cka", ["source_space", "target_representation", "method"]),
        ("cca", "mean_top10_test_correlation", ["source_space", "target_representation"]),
    ]:
        frame = pd.read_csv(COMP / "results" / f"{kind}_heldout.csv")
        generic = frame[frame.target_representation.str.startswith("RP-414")].copy()
        collapse_groups = [x for x in groups if x != "target_representation"] + ["split_seed"]
        generic = generic.groupby(collapse_groups, as_index=False)[value_col].mean(); generic["target_representation"] = "RP-414 mean"
        frame2 = pd.concat([frame[~frame.target_representation.str.startswith("RP-414")], generic], ignore_index=True)
        rows=[]
        for i,(keys,sub) in enumerate(frame2.groupby(groups)):
            vals=sub[value_col].to_numpy(float);lo,hi=bootstrap_mean_ci(vals,701+i)
            row={k:v for k,v in zip(groups,keys if isinstance(keys,tuple) else (keys,))};row.update({"n_splits":len(vals),"mean":vals.mean(),"sd":vals.std(ddof=1),"ci95_low":lo,"ci95_high":hi});rows.append(row)
        atomic_write_csv(pd.DataFrame(rows), COMP / "results" / f"{kind}_summary.csv")
    ridge = pd.read_csv(COMP / "results" / "ridge_predictability.csv"); test=ridge[ridge.partition=="test"]
    rows=[]
    metric_cols=["variance_weighted_r2","uniform_average_r2","median_per_feature_r2","fraction_features_r2_gt_0.5","fraction_features_r2_gt_0.8"]
    for i,(source,sub) in enumerate(test.groupby("source_space")):
        for j,m in enumerate(metric_cols):
            vals=sub[m].to_numpy(float);lo,hi=bootstrap_mean_ci(vals,801+i*10+j);rows.append({"source_space":source,"metric":m,"n_splits":len(vals),"mean":vals.mean(),"sd":vals.std(ddof=1),"ci95_low":lo,"ci95_high":hi})
    atomic_write_csv(pd.DataFrame(rows),COMP/"results"/"ridge_predictability_summary.csv")


def rescue_statistics() -> None:
    by_seed = pd.read_csv(RESCUE / "results" / "seed_reclassification_summary.csv")
    rows=[]
    for i,(model,sub) in enumerate(by_seed.groupby("model")):
        vals=sub.sort_values("split_seed").paired_accuracy_delta.to_numpy(float);lo,hi=bootstrap_mean_ci(vals,901+i)
        rows.append({"model":model,"contrast":"Gene + HubmiR − Gene","metric":"accuracy","n_pairs":len(vals),"mean_delta":vals.mean(),"ci95_low":lo,"ci95_high":hi,"paired_effect_dz":vals.mean()/vals.std(ddof=1),"exact_signflip_p":exact_signflip(vals)})
    atomic_write_csv(pd.DataFrame(rows),RESCUE/"results"/"overall_rescue_statistics.csv")
    pathseed=pd.read_csv(RESCUE/"results"/"pathway_seed_reclassification.csv");rows=[]
    for i,((model,pathway),sub) in enumerate(pathseed.groupby(["model","pathway"])):
        vals=sub.sort_values("split_seed").net_reclassification_rate.to_numpy(float);lo,hi=bootstrap_mean_ci(vals,1001+i)
        rows.append({"model":model,"pathway":pathway,"n_seeds":len(vals),"mean_net_rate":vals.mean(),"ci95_low":lo,"ci95_high":hi,"exact_signflip_p":exact_signflip(vals)})
    out=pd.DataFrame(rows);out["holm_p_within_model"]=out.groupby("model",group_keys=False).exact_signflip_p.apply(holm);atomic_write_csv(out,RESCUE/"results"/"pathway_net_statistics.csv")


def progeny_audit() -> None:
    legacy_root = os.environ.get("MIRCID_LEGACY_ROOT")
    if not legacy_root:
        write_json(
            FIG4 / "results" / "progeny_legacy_score_audit_status.json",
            {"status": "skipped", "reason": "MIRCID_LEGACY_ROOT was not provided"},
        )
        return
    project = Path(legacy_root).expanduser().resolve()
    old_scores_path = project / "Inference Models" / "Pathway Inference" / "ZYY_LR" / "TEST_PROGENy_val_with_labels" / "PROGENy_val_with_labels.csv"
    old_negative_path = old_scores_path.with_name("PROGENy_val_with_labels_negative.csv")
    old_predictions_path = old_scores_path.with_name("final_predictions.csv")
    data=load_processed();weights=pd.read_csv(DATA_ROOT/"pathway/processed/progeny_top100_matrix.csv").set_index("gene");pos={g:i for i,g in enumerate(data["gene_names"].astype(str))};classes=data["class_names"].astype(str);common=[g for g in weights.index if g in pos];raw=data["gene"][:,[pos[g] for g in common]]@weights.loc[common,classes].to_numpy();raw=pd.DataFrame(raw,index=data["sample_ids"].astype(str),columns=classes)
    old=pd.read_csv(old_scores_path,index_col=0);old.index=old.index.str.replace("JAK.STAT.","JAK-STAT.",regex=False);rows=[]
    for pathway in classes:
        old_col="JAK.STAT" if pathway=="JAK-STAT" else pathway;x=raw.loc[old.index,pathway].to_numpy();y=old[old_col].to_numpy();slope,intercept=np.polyfit(x,y,1);residual=y-(slope*x+intercept);rows.append({"pathway":pathway,"pearson_r":np.corrcoef(x,y)[0,1],"affine_slope":slope,"affine_intercept":intercept,"max_abs_affine_residual":np.max(np.abs(residual))})
    atomic_write_csv(pd.DataFrame(rows),FIG4/"results"/"progeny_legacy_score_audit.csv")
    final=pd.read_csv(old_predictions_path).set_index("id");positive=pd.read_csv(old_scores_path,index_col=0);negative=pd.read_csv(old_negative_path,index_col=0);joined=final.join(positive[["label_max"]]).join(negative[["label_negative"]]);expected=joined.apply(lambda r:r.label_max if r.effect=="activating" else r.label_negative,axis=1)
    write_json(FIG4/"results"/"progeny_provenance.json",{"recommended_name":"PROGENy linear pathway activity model (footprint-score baseline)","do_not_call":"Weighted Logistic Regression (WLR)","official_r_package_commit":"cad6be0514c3248b9465e48f1cfd2f6a4c3dfb6f","official_footprints_commit":"e9f7f3f235ce2bfc5380b7e408e12f8791790246","top_genes_per_pathway":100,"target_pathways":classes.tolist(),"legacy_scores_sha256":sha256_file(old_scores_path),"legacy_scores_exact_affine_lineage_max_residual":float(pd.DataFrame(rows).max_abs_affine_residual.max()),"legacy_final_predictions_match_effect_aware_rule_fraction":float((expected==joined.predicted_label).mean()),"legacy_effect_counts":joined.effect.value_counts().to_dict(),"fair_rule":"argmax absolute train-standardized score among 11 target pathways; no effect input","official_recall_rule":"activating: max score; inhibiting: min score; effect input required"})


def main() -> None:
    main = main_metrics(); embedding_analysis(main); complementarity_analysis(main); rescue_statistics(); progeny_audit()
    write_json(ROOT/"validation"/"analysis_manifest.json",{"status":"complete","bootstrap_iterations":10000,"paired_test":"exact two-sided sign-flip","effect_size":"paired Cohen dz","multiple_testing":"Holm within declared model/metric or model/pathway family","statistical_unit":"frozen split seed (n=20)","test_seed_cherry_picking":False})
    print("PASS: all summaries, paired statistics, complementarity and PROGENy audits written")


if __name__=="__main__":main()
