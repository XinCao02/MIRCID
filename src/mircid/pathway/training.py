from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .common import set_seed
from .networks import model_from_name


def _evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device, loss_fn: nn.Module) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x).to(device))
        loss = float(loss_fn(logits, torch.from_numpy(y).to(device)).item())
        pred = logits.argmax(dim=1).cpu().numpy()
    return {
        "loss": loss,
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "pred": pred,
    }


def train_neural(
    *,
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray | None,
    y_test: np.ndarray | None,
    config: dict,
    seed: int,
    artifact_dir: Path,
    device: torch.device,
    archive_checkpoint: bool = True,
) -> dict:
    set_seed(seed)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model = model_from_name(name, x_train.shape[1], int(max(y_train.max(), y_val.max()) + 1), config).to(device)
    learning_rate = float(config["learning_rate"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=float(config.get("weight_decay", 0.0))
    )
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=int(config["batch_size"]), shuffle=True, generator=generator, num_workers=0,
    )
    best_score = -np.inf
    best_epoch = 0
    best_state = None
    patience = 0
    curve = []
    started = time.time()
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        correct = 0
        train_pred, train_true = [], []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(xb)
            seen += len(xb)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            train_pred.append(pred.detach().cpu().numpy())
            train_true.append(yb.detach().cpu().numpy())
        tr_pred = np.concatenate(train_pred)
        tr_true = np.concatenate(train_true)
        val = _evaluate(model, x_val, y_val, device, loss_fn)
        curve.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / seen,
                "train_accuracy": correct / seen,
                "train_macro_f1": f1_score(tr_true, tr_pred, average="macro", zero_division=0),
                "validation_loss": val["loss"],
                "validation_accuracy": val["accuracy"],
                "validation_macro_f1": val["macro_f1"],
            }
        )
        if val["macro_f1"] > best_score + 1e-8:
            best_score = val["macro_f1"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= int(config["patience"]):
            break
    if best_state is None:
        raise RuntimeError("No best checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    train_eval = _evaluate(model, x_train, y_train, device, loss_fn)
    val_eval = _evaluate(model, x_val, y_val, device, loss_fn)
    test_eval = None if x_test is None else _evaluate(model, x_test, y_test, device, loss_fn)
    pd.DataFrame(curve).to_csv(artifact_dir / "learning_curve.csv", index=False)
    if archive_checkpoint:
        # Half-precision CPU archival keeps all formal checkpoints tractable.
        # Predictions above are always computed from the exact float32 best state.
        archived = {key: value.half() if value.is_floating_point() else value for key, value in best_state.items()}
        torch.save(
            {
                "state_dict_float16_archive": archived,
                "model": name,
                "input_dim": x_train.shape[1],
                "classes": int(max(y_train.max(), y_val.max()) + 1),
                "config": config,
                "seed": seed,
                "best_epoch": best_epoch,
                "best_validation_macro_f1": best_score,
                "prediction_checkpoint_was_float32": True,
            },
            artifact_dir / "best_checkpoint.pt",
        )
    return {
        "train": train_eval,
        "validation": val_eval,
        "test": test_eval,
        "best_epoch": best_epoch,
        "epochs_run": len(curve),
        "best_validation_macro_f1": best_score,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "elapsed_seconds": time.time() - started,
        "device": str(device),
        "learning_rate": learning_rate,
    }
