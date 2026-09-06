from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mircid.paths import REPO_ROOT
from sklearn.preprocessing import StandardScaler

from .common import ROOT, atomic_write_csv, build_features, load_config, load_processed, metric_rows, metrics, prediction_rows, read_csv_auto, split_indices, write_json
from .training import train_neural


def candidate_from_args(args: argparse.Namespace, base: dict) -> dict:
    return {
        "hidden": int(args.hidden), "blocks": int(args.blocks), "dropout": float(args.dropout),
        "batch_size": int(args.batch_size or base["batch_size"]),
        "max_epochs": int(args.epochs or base["max_epochs"]),
        "patience": int(args.patience or base["patience"]),
        "learning_rate": float(args.learning_rate), "weight_decay": float(args.weight_decay),
    }


def scaled_parts(x: np.ndarray, data: dict, seed: int):
    split = split_indices(seed, data)
    scaler = StandardScaler()
    parts = {"train": scaler.fit_transform(x[split["train"]]).astype(np.float32)}
    for p in ["validation", "test"]:
        parts[p] = scaler.transform(x[split[p]]).astype(np.float32)
    return split, parts


def tune(args: argparse.Namespace, cfg: dict, data: dict, device: torch.device) -> None:
    if args.round_id is None:
        raise ValueError("--round-id is required for tuning")
    candidate = candidate_from_args(args, cfg["models"]["resnet"])
    x = build_features(data, "Gene + HubmiR")
    round_dir = ROOT / "paper_exp" / "figure4_benchmark" / "runs" / "resnet_tuning" / f"round_{args.round_id:02d}"
    if round_dir.exists() and (round_dir / "summary.csv").exists() and not args.force:
        print(pd.read_csv(round_dir / "summary.csv").to_string(index=False))
        return
    rows = []
    for seed in cfg["resnet_tuning_seeds"]:
        split, parts = scaled_parts(x, data, seed)
        artifact = round_dir / f"seed_{seed}"
        fit = train_neural(
            name="resnet", x_train=parts["train"], y_train=data["y"][split["train"]],
            x_val=parts["validation"], y_val=data["y"][split["validation"]],
            x_test=None, y_test=None, config=candidate, seed=seed, artifact_dir=artifact,
            device=device, archive_checkpoint=False,
        )
        rows.append(
            {"round_id": args.round_id, "seed": seed, "validation_accuracy": fit["validation"]["accuracy"],
             "validation_macro_f1": fit["validation"]["macro_f1"], "validation_loss": fit["validation"]["loss"],
             "train_accuracy": fit["train"]["accuracy"], "train_macro_f1": fit["train"]["macro_f1"],
             "best_epoch": fit["best_epoch"], "epochs_run": fit["epochs_run"],
             "parameter_count": fit["parameter_count"], "elapsed_seconds": fit["elapsed_seconds"]}
        )
    summary = pd.DataFrame(rows)
    atomic_write_csv(summary, round_dir / "summary.csv")
    write_json(
        round_dir / "manifest.json",
        {"status": "complete", "round_id": args.round_id, "feature_space": "Gene + HubmiR",
         "selection_partition": "validation", "test_loaded_into_training_call": False,
         "seeds": cfg["resnet_tuning_seeds"], "config": candidate,
         "mean_validation_macro_f1": float(summary.validation_macro_f1.mean()),
         "mean_validation_accuracy": float(summary.validation_accuracy.mean())},
    )
    print(summary.to_string(index=False))
    print("MEAN", summary[["validation_accuracy", "validation_macro_f1", "train_accuracy", "train_macro_f1"]].mean().to_dict())


def formal(args: argparse.Namespace, cfg: dict, data: dict, device: torch.device) -> None:
    selected_path = REPO_ROOT / "configs" / "resnet_selected.json"
    if not selected_path.exists():
        raise FileNotFoundError("Select and freeze configs/resnet_selected.json after tuning")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    model_cfg = selected["config"]
    spaces = list(cfg["feature_spaces"]) if not args.feature_spaces else args.feature_spaces
    seeds = cfg["split_seeds"] if not args.seeds else args.seeds
    formal_root = ROOT / "paper_exp" / "figure4_benchmark" / "runs" / "resnet_formal"
    for feature_space in spaces:
        x = build_features(data, feature_space)
        for seed in seeds:
            run_dir = formal_root / f"{feature_space.replace(' ', '_').replace('+', 'plus')}__seed{seed}"
            done = run_dir / "run_manifest.json"
            if done.exists() and json.loads(done.read_text()).get("status") == "complete" and not args.force:
                print(f"SKIP {run_dir.name}")
                continue
            split, parts = scaled_parts(x, data, seed)
            fit = train_neural(
                name="resnet", x_train=parts["train"], y_train=data["y"][split["train"]],
                x_val=parts["validation"], y_val=data["y"][split["validation"]],
                x_test=parts["test"], y_test=data["y"][split["test"]], config=model_cfg,
                seed=seed, artifact_dir=run_dir, device=device, archive_checkpoint=True,
            )
            metric_records, pred_records = [], []
            for partition in ["train", "validation", "test"]:
                idx = split[partition]
                pred = fit[partition]["pred"]
                metric_records.extend(metric_rows("ResNet", feature_space, seed, partition, metrics(data["y"][idx], pred)))
                pred_records.extend(
                    prediction_rows(model="ResNet", feature_space=feature_space, seed=seed, partition=partition,
                                    indices=idx, y_true=data["y"][idx], y_pred=pred,
                                    sample_ids=data["sample_ids"], class_names=data["class_names"])
                )
            atomic_write_csv(pd.DataFrame(metric_records), run_dir / "metrics.csv")
            atomic_write_csv(pd.DataFrame(pred_records), run_dir / "predictions.csv.gz")
            write_json(
                done,
                {"status": "complete", "model": "resnet", "feature_space": feature_space, "seed": seed,
                 "config": model_cfg, "selected_from": str(selected_path.relative_to(REPO_ROOT)),
                 "scaler_fit_partition": "train", "best_checkpoint_selected_by": "validation_macro_f1",
                 "checkpoint_reloaded_before_test": True, "test_used_for_tuning": False,
                 "best_epoch": fit["best_epoch"], "epochs_run": fit["epochs_run"],
                 "parameter_count": fit["parameter_count"], "elapsed_seconds": fit["elapsed_seconds"],
                 "device": fit["device"]},
            )
            print(f"DONE {feature_space} seed={seed} test macro-F1={fit['test']['macro_f1']:.4f}")
    collect_formal(formal_root)


def collect_formal(formal_root: Path) -> None:
    metric_files = sorted(formal_root.glob("*/metrics.csv"))
    prediction_files = sorted(formal_root.glob("*/predictions.csv.gz"))
    if metric_files:
        atomic_write_csv(pd.concat([pd.read_csv(x) for x in metric_files], ignore_index=True),
                         ROOT / "paper_exp" / "figure4_benchmark" / "results" / "resnet_metrics.csv")
    if prediction_files:
        atomic_write_csv(pd.concat([read_csv_auto(x) for x in prediction_files], ignore_index=True),
                         ROOT / "paper_exp" / "figure4_benchmark" / "results" / "resnet_predictions.csv.gz")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["tune", "formal", "collect"])
    parser.add_argument("--round-id", type=int)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--feature-spaces", nargs="*")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg, data = load_config(), load_processed()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    if args.mode == "tune":
        tune(args, cfg, data, device)
    elif args.mode == "formal":
        formal(args, cfg, data, device)
    else:
        collect_formal(ROOT / "paper_exp" / "figure4_benchmark" / "runs" / "resnet_formal")


if __name__ == "__main__":
    main()
