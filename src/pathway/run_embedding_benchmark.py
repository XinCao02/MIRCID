from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mircid.paths import REPO_ROOT
from sklearn.preprocessing import StandardScaler

from .common import ROOT, atomic_write_csv, load_config, load_processed, metric_rows, metrics, prediction_rows, read_csv_auto, split_indices, write_json
from .run_benchmark import fit_classical
from .training import train_neural


WORK = ROOT / "additional_exp" / "5.1_embedding_controls"
REPS = WORK / "artifacts" / "representations"


def representation_specs(cfg: dict) -> dict[str, Path]:
    result = {
        "Gene + PCA-414": REPS / "pca_414.npy",
        "Gene + AE-414": REPS / "autoencoder_414.npy",
    }
    for seed in cfg["embedding_controls"]["rp_seeds"]:
        result[f"Gene + RP-414 (seed {seed})"] = REPS / f"random_projection_414_seed{seed}.npy"
    return result


def collect(model: str, run_root: Path) -> None:
    metric_files = sorted(run_root.glob("*/metrics.csv"))
    pred_files = sorted(run_root.glob("*/predictions.csv.gz"))
    if metric_files:
        atomic_write_csv(pd.concat([pd.read_csv(x) for x in metric_files], ignore_index=True), WORK / "results" / f"{model}_control_metrics.csv")
    if pred_files:
        atomic_write_csv(pd.concat([read_csv_auto(x) for x in pred_files], ignore_index=True), WORK / "results" / f"{model}_control_predictions.csv.gz")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["rf", "svm", "resnet"])
    parser.add_argument("--representations", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    cfg, data = load_config(), load_processed()
    run_root = WORK / "runs" / args.model
    if args.collect_only:
        collect(args.model, run_root); return
    specs = representation_specs(cfg)
    names = list(specs) if not args.representations else args.representations
    seeds = cfg["split_seeds"] if not args.seeds else args.seeds
    unknown_seeds = sorted(set(seeds) - set(cfg["split_seeds"]))
    if unknown_seeds:
        raise ValueError(
            "Embedding-control runs are restricted to the frozen split manifest; "
            f"unsupported seeds: {unknown_seeds}"
        )
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    selected = json.loads((REPO_ROOT / "configs" / "resnet_selected.json").read_text())["config"]
    for name in names:
        path = specs[name]
        if not path.exists():
            raise FileNotFoundError(path)
        rep = np.load(path).astype(np.float32)
        if rep.shape != (565, 414) or not np.isfinite(rep).all():
            raise ValueError(f"Invalid representation {name}: {rep.shape}")
        x = np.concatenate([data["gene"], rep], axis=1).astype(np.float32)
        for seed in seeds:
            token = name.replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "")
            run_dir = run_root / f"{token}__seed{seed}"
            done = run_dir / "run_manifest.json"
            if done.exists() and json.loads(done.read_text()).get("status") == "complete" and not args.force:
                print(f"SKIP {run_dir.name}"); continue
            split = split_indices(seed, data)
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x[split["train"]]).astype(np.float32)
            x_val = scaler.transform(x[split["validation"]]).astype(np.float32)
            x_test = scaler.transform(x[split["test"]]).astype(np.float32)
            y_train, y_val, y_test = (data["y"][split[p]] for p in ["train", "validation", "test"])
            run_dir.mkdir(parents=True, exist_ok=True)
            if args.model in {"rf", "svm"}:
                p_train, p_val, p_test, details = fit_classical(
                    args.model, x_train, y_train, x_val, x_test,
                    cfg["models"][args.model], seed,
                )
                fit = None
            else:
                fit = train_neural(
                    name="resnet", x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
                    x_test=x_test, y_test=y_test, config=selected, seed=seed, artifact_dir=run_dir,
                    device=device, archive_checkpoint=True,
                )
                p_train, p_val, p_test = (fit[p]["pred"] for p in ["train", "validation", "test"])
                details = {k: v for k, v in fit.items() if k not in {"train", "validation", "test"}}
            metric_records, prediction_records = [], []
            for partition, pred in [("train", p_train), ("validation", p_val), ("test", p_test)]:
                idx = split[partition]
                metric_records.extend(metric_rows(args.model.upper(), name, seed, partition, metrics(data["y"][idx], pred)))
                prediction_records.extend(
                    prediction_rows(model=args.model.upper(), feature_space=name, seed=seed, partition=partition,
                                    indices=idx, y_true=data["y"][idx], y_pred=pred,
                                    sample_ids=data["sample_ids"], class_names=data["class_names"])
                )
            atomic_write_csv(pd.DataFrame(metric_records), run_dir / "metrics.csv")
            atomic_write_csv(pd.DataFrame(prediction_records), run_dir / "predictions.csv.gz")
            write_json(done, {"status": "complete", "model": args.model, "feature_space": name, "seed": seed,
                              "representation_file": str(path.relative_to(ROOT)), "scaler_fit_partition": "train",
                              "split_manifest": "pathway/splits/frozen_20_splits.csv",
                              "test_used_for_selection": False,
                              "config": cfg["models"][args.model] if args.model in {"rf", "svm"} else selected,
                              "details": details})
            print(f"DONE {args.model} {name} seed={seed}: test macro-F1={metrics(y_test, p_test)['macro_f1']:.4f}")
    collect(args.model, run_root)


if __name__ == "__main__":
    main()
