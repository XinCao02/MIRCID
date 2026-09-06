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


WORK = ROOT / "additional_exp" / "5.2_complementarity"
DECOMP = WORK / "artifacts" / "decomposition"


def collect(model: str, run_root: Path) -> None:
    metric_files = sorted(run_root.glob("*/metrics.csv")); pred_files = sorted(run_root.glob("*/predictions.csv.gz"))
    if metric_files:
        atomic_write_csv(pd.concat([pd.read_csv(x) for x in metric_files], ignore_index=True), WORK / "results" / f"{model}_decomposition_metrics.csv")
    if pred_files:
        atomic_write_csv(pd.concat([read_csv_auto(x) for x in pred_files], ignore_index=True), WORK / "results" / f"{model}_decomposition_predictions.csv.gz")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--model", required=True, choices=["svm", "resnet"]); parser.add_argument("--seeds", nargs="*", type=int); parser.add_argument("--device", default="auto"); parser.add_argument("--force", action="store_true"); parser.add_argument("--collect-only", action="store_true"); args = parser.parse_args()
    cfg, data = load_config(), load_processed(); seeds = cfg["split_seeds"] if not args.seeds else args.seeds
    run_root = WORK / "runs" / f"{args.model}_decomposition"
    if args.collect_only: collect(args.model, run_root); return
    selected = json.loads((REPO_ROOT / "configs" / "resnet_selected.json").read_text())["config"]
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    for seed in seeds:
        obj = np.load(DECOMP / f"split_{seed}.npz")
        split = split_indices(seed, data)
        expected_partition = np.empty(565, dtype="U16")
        for p, idx in split.items(): expected_partition[idx] = p
        if not np.array_equal(obj["partition"], expected_partition): raise ValueError(f"Partition mismatch seed {seed}")
        for feature_space, rep in [("Gene + predicted HubmiR", obj["predicted"]), ("Gene + residual HubmiR", obj["residual"])]:
            x = np.concatenate([data["gene"], rep.astype(np.float32)], axis=1).astype(np.float32)
            token = feature_space.replace(" ", "_").replace("+", "plus")
            run_dir = run_root / f"{token}__seed{seed}"; done = run_dir / "run_manifest.json"
            if done.exists() and json.loads(done.read_text()).get("status") == "complete" and not args.force:
                print(f"SKIP {run_dir.name}"); continue
            scaler = StandardScaler(); x_train = scaler.fit_transform(x[split["train"]]).astype(np.float32); x_val = scaler.transform(x[split["validation"]]).astype(np.float32); x_test = scaler.transform(x[split["test"]]).astype(np.float32)
            y_train, y_val, y_test = (data["y"][split[p]] for p in ["train", "validation", "test"]); run_dir.mkdir(parents=True, exist_ok=True)
            if args.model == "svm":
                p_train, p_val, p_test, details = fit_classical("svm", x_train, y_train, x_val, x_test, cfg["models"]["svm"], seed)
            else:
                fit = train_neural(name="resnet", x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test, config=selected, seed=seed, artifact_dir=run_dir, device=device, archive_checkpoint=True)
                p_train, p_val, p_test = (fit[p]["pred"] for p in ["train", "validation", "test"]); details = {k:v for k,v in fit.items() if k not in {"train","validation","test"}}
            mrows, prows = [], []
            for partition, pred in [("train",p_train),("validation",p_val),("test",p_test)]:
                idx=split[partition]; mrows.extend(metric_rows(args.model.upper(),feature_space,seed,partition,metrics(data["y"][idx],pred))); prows.extend(prediction_rows(model=args.model.upper(),feature_space=feature_space,seed=seed,partition=partition,indices=idx,y_true=data["y"][idx],y_pred=pred,sample_ids=data["sample_ids"],class_names=data["class_names"]))
            atomic_write_csv(pd.DataFrame(mrows),run_dir/"metrics.csv"); atomic_write_csv(pd.DataFrame(prows),run_dir/"predictions.csv.gz")
            write_json(done,{"status":"complete","model":args.model,"feature_space":feature_space,"seed":seed,"decomposition":str((DECOMP/f"split_{seed}.npz").relative_to(ROOT)),"training_representation":"5-fold OOF","validation_test_representation":"outer-train Ridge","scaler_fit_partition":"train","test_used_for_selection":False,"config":cfg["models"]["svm"] if args.model=="svm" else selected,"details":details})
            print(f"DONE {args.model} {feature_space} seed={seed}: test macro-F1={metrics(y_test,p_test)['macro_f1']:.4f}")
    collect(args.model,run_root)


if __name__ == "__main__": main()
