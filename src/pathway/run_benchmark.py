from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .common import ROOT, atomic_write_csv, build_features, load_config, load_processed, metric_rows, metrics, prediction_rows, split_indices, write_json
from .training import train_neural


def fit_classical(name: str, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray, cfg: dict, seed: int):
    if name == "rf":
        model = RandomForestClassifier(
            n_estimators=int(cfg["n_estimators"]), max_features=cfg["max_features"],
            class_weight=cfg["class_weight"], random_state=seed, n_jobs=-1,
        )
        model.fit(x_train, y_train)
        return model.predict(x_train), model.predict(x_val), model.predict(x_test), {
            "trees": len(model.estimators_), "mean_tree_depth": float(np.mean([x.tree_.max_depth for x in model.estimators_]))
        }
    if name == "svm":
        gamma = 1.0 / (x_train.shape[1] * float(np.var(x_train))) if cfg["gamma"] == "scale" else float(cfg["gamma"])
        k_train = rbf_kernel(x_train, x_train, gamma=gamma)
        model = SVC(C=float(cfg["C"]), kernel="precomputed", class_weight=cfg["class_weight"], random_state=seed)
        model.fit(k_train, y_train)
        return model.predict(k_train), model.predict(rbf_kernel(x_val, x_train, gamma=gamma)), model.predict(rbf_kernel(x_test, x_train, gamma=gamma)), {
            "support_vectors": int(model.n_support_.sum()), "gamma": gamma
        }
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["rf", "svm", "mlp", "kan"])
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--feature-spaces", nargs="*")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    data = load_processed()
    seeds = cfg["split_seeds"] if not args.seeds else args.seeds
    spaces = list(cfg["feature_spaces"]) if not args.feature_spaces else args.feature_spaces
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    result_root = ROOT / "paper_exp" / "figure4_benchmark"
    metric_records, prediction_records, run_records = [], [], []
    for feature_space in spaces:
        x = build_features(data, feature_space)
        for seed in seeds:
            run_name = f"{args.model}__{feature_space.replace(' ', '_').replace('+', 'plus')}__seed{seed}"
            run_dir = result_root / "runs" / run_name
            done = run_dir / "run_manifest.json"
            if done.exists() and not args.force:
                existing = json.loads(done.read_text())
                if existing.get("status") == "complete":
                    print(f"SKIP {run_name}")
                    continue
            run_dir.mkdir(parents=True, exist_ok=True)
            started = time.time()
            split = split_indices(seed, data)
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x[split["train"]]).astype(np.float32)
            x_val = scaler.transform(x[split["validation"]]).astype(np.float32)
            x_test = scaler.transform(x[split["test"]]).astype(np.float32)
            y_train, y_val, y_test = (data["y"][split[p]] for p in ["train", "validation", "test"])
            if args.model in {"rf", "svm"}:
                p_train, p_val, p_test, details = fit_classical(args.model, x_train, y_train, x_val, x_test, cfg["models"][args.model], seed)
                evaluations = {"train": p_train, "validation": p_val, "test": p_test}
            else:
                fit = train_neural(
                    name=args.model, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
                    x_test=x_test, y_test=y_test, config=cfg["models"][args.model], seed=seed,
                    artifact_dir=run_dir, device=device, archive_checkpoint=True,
                )
                evaluations = {p: fit[p]["pred"] for p in ["train", "validation", "test"]}
                details = {k: v for k, v in fit.items() if k not in {"train", "validation", "test"}}
            for partition, pred in evaluations.items():
                idx = split[partition]
                truth = data["y"][idx]
                metric_records.extend(metric_rows(args.model.upper(), feature_space, seed, partition, metrics(truth, pred)))
                prediction_records.extend(
                    prediction_rows(model=args.model.upper(), feature_space=feature_space, seed=seed, partition=partition,
                                    indices=idx, y_true=truth, y_pred=pred, sample_ids=data["sample_ids"], class_names=data["class_names"])
                )
            manifest = {
                "status": "complete", "run_name": run_name, "model": args.model, "feature_space": feature_space,
                "seed": seed, "split_manifest": "manifests/splits/frozen_20_splits.csv",
                "scaler_fit_partition": "train", "best_checkpoint_selected_by": "validation_macro_f1" if args.model in {"mlp", "kan"} else None,
                "test_accessed_after_fit": True, "config": cfg["models"][args.model], "details": details,
                "elapsed_seconds": time.time() - started,
            }
            write_json(done, manifest)
            run_records.append(manifest)
            print(f"DONE {run_name}: test macro-F1={metrics(y_test, evaluations['test'])['macro_f1']:.4f}")
    suffix = args.model
    if metric_records:
        atomic_write_csv(pd.DataFrame(metric_records), result_root / "results" / f"{suffix}_metrics.csv")
        atomic_write_csv(pd.DataFrame(prediction_records), result_root / "results" / f"{suffix}_predictions.csv.gz")
        write_json(result_root / "results" / f"{suffix}_run_summary.json", run_records)


if __name__ == "__main__":
    main()

