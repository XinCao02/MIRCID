from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .common import ROOT, atomic_write_csv, load_config, load_processed, split_indices, write_json
from .complementarity_core import (
    cross_validated_dual_ridge,
    debiased_cka,
    median_squared_distance,
    predictability_metrics,
    rbf_gram,
    regularized_cca,
    unbiased_hsic,
)


WORK = ROOT / "additional_exp" / "5.2_complementarity"
CONTROL_REPS = ROOT / "additional_exp" / "5.1_embedding_controls" / "artifacts" / "representations"


def target_representations(data: dict, cfg: dict) -> dict[str, np.ndarray]:
    reps = {
        "HubmiR": data["mirna"].astype(np.float32),
        "PCA-414": np.load(CONTROL_REPS / "pca_414.npy").astype(np.float32),
        "AE-414": np.load(CONTROL_REPS / "autoencoder_414.npy").astype(np.float32),
    }
    for seed in cfg["embedding_controls"]["rp_seeds"]:
        reps[f"RP-414 seed {seed}"] = np.load(CONTROL_REPS / f"random_projection_414_seed{seed}.npy").astype(np.float32)
    for name, values in reps.items():
        if values.shape != (565, 414) or not np.isfinite(values).all():
            raise ValueError(f"Bad representation {name}: {values.shape}")
    return reps


def permutation_cka(kernel_x: np.ndarray, kernel_y: np.ndarray, permutations: int, seed: int) -> tuple[float, np.ndarray]:
    observed = debiased_cka(kernel_x, kernel_y)
    k = np.asarray(kernel_x, dtype=np.float64).copy()
    l = np.asarray(kernel_y, dtype=np.float64).copy()
    np.fill_diagonal(k, 0.0); np.fill_diagonal(l, 0.0)
    n = len(k)
    denom = np.sqrt(unbiased_hsic(kernel_x, kernel_x) * unbiased_hsic(kernel_y, kernel_y))
    rng = np.random.default_rng(seed)
    null = []
    ksum, lsum = k.sum(), l.sum()
    krows, lrows = k.sum(axis=1), l.sum(axis=1)
    constant = ksum * lsum / ((n - 1) * (n - 2))
    for start in range(0, permutations, 64):
        batch = min(64, permutations - start)
        orders = np.asarray([rng.permutation(n) for _ in range(batch)])
        lp = l[orders[:, :, None], orders[:, None, :]]
        trace = np.einsum("ij,bij->b", k, lp, optimize=True)
        row = 2.0 * np.einsum("i,bi->b", krows, lrows[orders], optimize=True) / (n - 2)
        hsic = (trace + constant - row) / (n * (n - 3))
        null.extend((hsic / denom).tolist())
    return observed, np.asarray(null)


def pca_scores(train: np.ndarray, test: np.ndarray, components: int, seed: int):
    scaler = StandardScaler()
    train_z = scaler.fit_transform(train).astype(np.float32)
    test_z = scaler.transform(test).astype(np.float32)
    pca = PCA(n_components=components, svd_solver="randomized", iterated_power=5, random_state=seed)
    return pca.fit_transform(train_z).astype(np.float64), pca.transform(test_z).astype(np.float64), float(pca.explained_variance_ratio_.sum())


def run_geometry(force: bool) -> None:
    cfg, data = load_config(), load_processed()
    ccfg = cfg["complementarity"]
    sources = {"Gene-29045": data["gene"], "HubmiR-input-977": data["l1000"]}
    targets = target_representations(data, cfg)
    cka_path = WORK / "results" / "cka_heldout.csv"
    cca_path = WORK / "results" / "cca_heldout.csv"
    mode_path = WORK / "results" / "cca_modes_heldout.csv"
    cka_existing = pd.read_csv(cka_path) if cka_path.exists() and not force else pd.DataFrame()
    cca_existing = pd.read_csv(cca_path) if cca_path.exists() and not force else pd.DataFrame()
    mode_existing = pd.read_csv(mode_path) if mode_path.exists() and not force else pd.DataFrame()
    done_seeds = set(cka_existing.split_seed.unique()) & set(cca_existing.split_seed.unique()) if len(cka_existing) and len(cca_existing) else set()
    cka_rows, cca_rows, mode_rows = [], [], []
    for seed in cfg["split_seeds"]:
        if seed in done_seeds:
            print(f"SKIP geometry seed {seed}"); continue
        split = split_indices(seed, data); train, test = split["train"], split["test"]
        source_pca, target_pca = {}, {}
        for si, (source_name, source) in enumerate(sources.items()):
            scaler = StandardScaler()
            source_train = scaler.fit_transform(source[train]).astype(np.float32)
            source_test = scaler.transform(source[test]).astype(np.float32)
            source_kernels = {
                "debiased_linear": source_test @ source_test.T,
                "debiased_rbf": rbf_gram(source_test, median_squared_distance(source_train)),
            }
            source_pca[source_name] = pca_scores(source[train], source[test], int(ccfg["cca_components"]), 10000 + seed + si)
            for ti, (target_name, target) in enumerate(targets.items()):
                t_scaler = StandardScaler()
                target_train = t_scaler.fit_transform(target[train]).astype(np.float32)
                target_test = t_scaler.transform(target[test]).astype(np.float32)
                target_kernels = {
                    "debiased_linear": target_test @ target_test.T,
                    "debiased_rbf": rbf_gram(target_test, median_squared_distance(target_train)),
                }
                for mi, method in enumerate(["debiased_linear", "debiased_rbf"]):
                    observed, null = permutation_cka(source_kernels[method], target_kernels[method], int(ccfg["cka_permutations"]), 1000000 + seed * 1000 + si * 100 + ti * 10 + mi)
                    cka_rows.append(
                        {"split_seed": seed, "partition": "test", "source_space": source_name,
                         "target_representation": target_name, "method": method, "cka": observed,
                         "null_mean": float(np.nanmean(null)), "null_sd": float(np.nanstd(null, ddof=1)),
                         "permutation_p": float((1 + np.sum(null >= observed)) / (1 + len(null))),
                         "permutations": int(ccfg["cka_permutations"]), "preprocessing_fit": "outer_train"}
                    )
        for ti, (target_name, target) in enumerate(targets.items()):
            target_pca[target_name] = pca_scores(target[train], target[test], int(ccfg["cca_components"]), 20000 + seed + ti)
        for si, source_name in enumerate(sources):
            x_train, x_test, x_var = source_pca[source_name]
            for ti, target_name in enumerate(targets):
                y_train, y_test, y_var = target_pca[target_name]
                result = regularized_cca(x_train, y_train, x_test, y_test, float(ccfg["cca_ridge"]), int(ccfg["cca_modes"]))
                rng = np.random.default_rng(2000000 + seed * 1000 + si * 100 + ti)
                null = []
                for _ in range(int(ccfg["cca_permutations"])):
                    permuted = y_train[rng.permutation(len(y_train))]
                    null_result = regularized_cca(x_train, permuted, x_test, y_test, float(ccfg["cca_ridge"]), int(ccfg["cca_modes"]))
                    null.append(float(np.nanmean(null_result.test_correlations)))
                observed = float(np.nanmean(result.test_correlations))
                cca_rows.append(
                    {"split_seed": seed, "partition": "test", "source_space": source_name,
                     "target_representation": target_name, "pca_components": int(ccfg["cca_components"]),
                     "ridge": float(ccfg["cca_ridge"]), "top1_test_correlation": float(result.test_correlations[0]),
                     "mean_top10_test_correlation": observed, "positive_test_modes": int(np.sum(result.test_correlations > 0)),
                     "null_mean": float(np.mean(null)), "null_sd": float(np.std(null, ddof=1)),
                     "permutation_p": float((1 + np.sum(np.asarray(null) >= observed)) / (1 + len(null))),
                     "permutations": int(ccfg["cca_permutations"]), "source_pca_variance": x_var,
                     "target_pca_variance": y_var, "preprocessing_fit": "outer_train"}
                )
                for mode, (test_corr, train_corr) in enumerate(zip(result.test_correlations, result.train_singular_values), 1):
                    mode_rows.append(
                        {"split_seed": seed, "source_space": source_name, "target_representation": target_name,
                         "mode": mode, "test_correlation": float(test_corr), "train_canonical_correlation": float(train_corr)}
                    )
        cka_frame = pd.concat([cka_existing[cka_existing.split_seed != seed] if len(cka_existing) else pd.DataFrame(), pd.DataFrame(cka_rows)], ignore_index=True)
        cca_frame = pd.concat([cca_existing[cca_existing.split_seed != seed] if len(cca_existing) else pd.DataFrame(), pd.DataFrame(cca_rows)], ignore_index=True)
        mode_frame = pd.concat([mode_existing[mode_existing.split_seed != seed] if len(mode_existing) else pd.DataFrame(), pd.DataFrame(mode_rows)], ignore_index=True)
        atomic_write_csv(cka_frame, cka_path); atomic_write_csv(cca_frame, cca_path); atomic_write_csv(mode_frame, mode_path)
        print(f"DONE geometry seed {seed}", flush=True)
    write_json(WORK / "artifacts" / "geometry_manifest.json", {"status": "complete", "splits": cfg["split_seeds"], "sources": list(sources), "targets": list(targets), "cka_permutations": ccfg["cka_permutations"], "cca_permutations": ccfg["cca_permutations"], "test_rows_used_for_fit": False})


def run_ridge(force: bool) -> None:
    cfg, data = load_config(), load_processed(); ccfg = cfg["complementarity"]
    sources = {"Gene-29045": data["gene"], "HubmiR-input-977": data["l1000"]}
    summary_path = WORK / "results" / "ridge_predictability.csv"
    feature_path = WORK / "results" / "ridge_per_mirna.csv.gz"
    summary_existing = pd.read_csv(summary_path) if summary_path.exists() and not force else pd.DataFrame()
    feature_existing = pd.read_csv(feature_path) if feature_path.exists() and not force else pd.DataFrame()
    done = set(zip(summary_existing.split_seed, summary_existing.source_space)) if len(summary_existing) else set()
    summary_rows, feature_rows = [], []
    decomp_dir = WORK / "artifacts" / "decomposition"; decomp_dir.mkdir(parents=True, exist_ok=True)
    for seed in cfg["split_seeds"]:
        split = split_indices(seed, data)
        for source_name, source in sources.items():
            if (seed, source_name) in done:
                print(f"SKIP ridge seed={seed} source={source_name}"); continue
            result = cross_validated_dual_ridge(
                source[split["train"]], data["mirna"][split["train"]],
                source[split["validation"]], source[split["test"]],
                [float(x) for x in ccfg["ridge_alphas"]], folds=5, seed=seed,
            )
            for partition, truth, predicted in [
                ("train_oof", data["mirna"][split["train"]], result["oof_prediction"]),
                ("validation", data["mirna"][split["validation"]], result["validation_prediction"]),
                ("test", data["mirna"][split["test"]], result["test_prediction"]),
            ]:
                vals, per = predictability_metrics(truth, predicted)
                summary_rows.append({"split_seed": seed, "source_space": source_name, "partition": partition,
                                     "best_alpha": result["best_alpha"], **vals, "alpha_cv_scores": str(result["alpha_scores"])})
                for j, value in enumerate(per):
                    feature_rows.append({"split_seed": seed, "source_space": source_name, "partition": partition,
                                         "mirna": str(data["mirna_names"][j]), "r2": float(value)})
            if source_name == "Gene-29045":
                predicted = np.empty_like(data["mirna"], dtype=np.float64)
                predicted[split["train"]] = result["oof_prediction"]
                predicted[split["validation"]] = result["validation_prediction"]
                predicted[split["test"]] = result["test_prediction"]
                residual = data["mirna"].astype(np.float64) - predicted
                error = float(np.max(np.abs(predicted + residual - data["mirna"])))
                if error >= 1e-6 or len(np.unique(result["fold_assignment"])) != 5:
                    raise ValueError(f"Decomposition validation failed: {error}")
                partition = np.empty(565, dtype="U16")
                for p, idx in split.items(): partition[idx] = p
                np.savez_compressed(decomp_dir / f"split_{seed}.npz", predicted=predicted.astype(np.float32), residual=residual.astype(np.float32), partition=partition, fold_assignment=result["fold_assignment"], max_reconstruction_error=error)
            print(f"DONE ridge seed={seed} source={source_name} alpha={result['best_alpha']}", flush=True)
        atomic_write_csv(pd.concat([summary_existing[summary_existing.split_seed != seed] if len(summary_existing) else pd.DataFrame(), pd.DataFrame(summary_rows)], ignore_index=True), summary_path)
        atomic_write_csv(pd.concat([feature_existing[feature_existing.split_seed != seed] if len(feature_existing) else pd.DataFrame(), pd.DataFrame(feature_rows)], ignore_index=True), feature_path)
    write_json(WORK / "artifacts" / "ridge_manifest.json", {"status": "complete", "splits": cfg["split_seeds"], "sources": list(sources), "alphas": ccfg["ridge_alphas"], "inner_folds": 5, "training_predictions": "exactly-once OOF", "validation_test_predictions": "outer-train fit", "max_allowed_reconstruction_error": 1e-6})


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", required=True, choices=["geometry", "ridge"]); parser.add_argument("--force", action="store_true"); args = parser.parse_args()
    if args.mode == "geometry": run_geometry(args.force)
    else: run_ridge(args.force)


if __name__ == "__main__": main()

