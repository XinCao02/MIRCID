from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mircid.paths import DATA_ROOT

from .common import ROOT, atomic_write_csv, load_config, load_processed, set_seed, sha256_file, write_json


WORK = ROOT / "additional_exp" / "5.1_embedding_controls"
ARTIFACTS = WORK / "artifacts"
REPS = ARTIFACTS / "representations"


def external_reference_path() -> Path:
    return DATA_ROOT / "hubmir" / "processed" / "TCGA_L1000_mRNA_embedding_reference.csv"


def prepare_linear() -> None:
    cfg = load_config()
    dim = int(cfg["embedding_controls"]["dimension"])
    data = load_processed()
    REPS.mkdir(parents=True, exist_ok=True)
    source = external_reference_path()
    external = pd.read_csv(source, index_col=0)
    mapping = pd.read_csv(DATA_ROOT / "pathway" / "source" / "PROGENy_input_aligned.csv")
    alias_overrides = {"KIFBP": "KIF1BP"}
    selected, rows = [], []
    for _, row in mapping.iterrows():
        candidates = [str(row["name2"]), str(row["name"]), str(row["index"])]
        candidates += [alias_overrides.get(x, "") for x in candidates]
        chosen = next((x for x in candidates if x and x in external.columns), None)
        if chosen is None:
            raise ValueError(f"No external-reference feature for checkpoint input row: {row.to_dict()}")
        selected.append(chosen)
        rows.append({"checkpoint_gene": row["name"], "mapped_gene": row["name2"], "external_column": chosen})
    if len(selected) != 977 or len(set(selected)) != 977:
        raise ValueError("External 977-gene mapping is incomplete or duplicated")
    external_x = external.loc[:, selected].to_numpy(dtype=np.float32)
    if not np.isfinite(external_x).all():
        raise ValueError("Non-finite external reference")
    scaler = StandardScaler()
    external_z = scaler.fit_transform(external_x).astype(np.float32)
    downstream_z = scaler.transform(data["l1000"]).astype(np.float32)
    np.savez_compressed(
        ARTIFACTS / "external_tcga_l1000_standardized.npz",
        x=external_z, sample_ids=external.index.astype(str).to_numpy(dtype="U64"),
    )
    np.save(REPS / "downstream_l1000_standardized.npy", downstream_z)
    np.savez_compressed(ARTIFACTS / "external_scaler.npz", mean=scaler.mean_, scale=scaler.scale_, var=scaler.var_)

    pca = PCA(n_components=dim, svd_solver="randomized", iterated_power=7, random_state=31415)
    pca.fit(external_z)
    pca_rep = pca.transform(downstream_z).astype(np.float32)
    np.save(REPS / "pca_414.npy", pca_rep)
    np.savez_compressed(
        ARTIFACTS / "pca_414_model.npz", components=pca.components_, mean=pca.mean_,
        explained_variance=pca.explained_variance_, explained_variance_ratio=pca.explained_variance_ratio_,
    )
    for seed in cfg["embedding_controls"]["rp_seeds"]:
        rp = GaussianRandomProjection(n_components=dim, random_state=seed)
        rp.fit(external_z)
        np.save(REPS / f"random_projection_414_seed{seed}.npy", rp.transform(downstream_z).astype(np.float32))
        np.save(ARTIFACTS / f"random_projection_matrix_seed{seed}.npy", rp.components_.astype(np.float32))
    atomic_write_csv(pd.DataFrame(rows), ROOT / "manifests" / "features" / "external_tcga_l1000_mapping.csv")
    write_json(
        ARTIFACTS / "linear_control_manifest.json",
        {"status": "complete", "external_reference": str(source.relative_to(DATA_ROOT)),
         "external_reference_sha256": sha256_file(source), "external_samples": len(external),
         "input_dimension": 977, "output_dimension": dim, "external_scaler_only": True,
         "pca_solver": "randomized", "pca_iterated_power": 7, "pca_seed": 31415,
         "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
         "rp_seeds": cfg["embedding_controls"]["rp_seeds"]},
    )
    print(f"PASS: external PCA-{dim} and {len(cfg['embedding_controls']['rp_seeds'])} RP controls prepared")


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int, bottleneck: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, bottleneck))
        self.decoder = nn.Sequential(nn.Linear(bottleneck, hidden), nn.ReLU(), nn.Linear(hidden, input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def train_ae(device_name: str, force: bool) -> None:
    cfg = load_config()["embedding_controls"]
    ae_cfg = cfg["ae"]
    out_rep = REPS / "autoencoder_414.npy"
    if out_rep.exists() and not force:
        print(f"SKIP: {out_rep}")
        return
    set_seed(int(ae_cfg["seed"]))
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    with np.load(ARTIFACTS / "external_tcga_l1000_standardized.npz", allow_pickle=False) as obj:
        x = obj["x"].astype(np.float32)
    train_idx, val_idx = train_test_split(
        np.arange(len(x)), test_size=float(ae_cfg["validation_fraction"]), random_state=int(ae_cfg["seed"])
    )
    model = Autoencoder(x.shape[1], int(ae_cfg["hidden"]), int(cfg["dimension"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(ae_cfg["learning_rate"]), weight_decay=float(ae_cfg["weight_decay"]))
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(int(ae_cfg["seed"]))
    loader = DataLoader(TensorDataset(torch.from_numpy(x[train_idx])), batch_size=int(ae_cfg["batch_size"]), shuffle=True, generator=generator)
    val_tensor = torch.from_numpy(x[val_idx]).to(device)
    best_loss, best_state, best_epoch, wait = np.inf, None, 0, 0
    curve = []
    started = time.time()
    for epoch in range(1, int(ae_cfg["max_epochs"]) + 1):
        model.train(); total = 0.0; count = 0
        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), xb)
            loss.backward(); optimizer.step()
            total += float(loss.item()) * len(xb); count += len(xb)
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_tensor), val_tensor).item())
        curve.append({"epoch": epoch, "train_mse": total / count, "validation_mse": val_loss})
        if val_loss < best_loss - 1e-7:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= int(ae_cfg["patience"]):
            break
    if best_state is None:
        raise RuntimeError("AE failed")
    model.load_state_dict(best_state); model.to(device); model.eval()
    downstream = np.load(REPS / "downstream_l1000_standardized.npy").astype(np.float32)
    with torch.no_grad():
        rep = model.encoder(torch.from_numpy(downstream).to(device)).cpu().numpy().astype(np.float32)
    np.save(out_rep, rep)
    torch.save(
        {"state_dict": best_state, "input_dim": 977, "hidden": ae_cfg["hidden"], "bottleneck": cfg["dimension"],
         "best_epoch": best_epoch, "best_validation_mse": best_loss, "config": ae_cfg},
        ARTIFACTS / "autoencoder_best.pt",
    )
    atomic_write_csv(pd.DataFrame(curve), WORK / "results" / "autoencoder_learning_curve.csv")
    write_json(
        ARTIFACTS / "autoencoder_manifest.json",
        {"status": "complete", "training_samples": len(train_idx), "validation_samples": len(val_idx),
         "best_epoch": best_epoch, "best_validation_mse": best_loss, "elapsed_seconds": time.time() - started,
         "device": str(device), "external_reference_only": True, "config": ae_cfg},
    )
    print(f"PASS: external-reference AE-414 trained; best epoch={best_epoch}, val MSE={best_loss:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["linear", "ae"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.mode == "linear":
        prepare_linear()
    else:
        train_ae(args.device, args.force)


if __name__ == "__main__":
    main()
