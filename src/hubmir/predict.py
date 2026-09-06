"""Command-line inference with a frozen HubmiRNet checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mircid.hubmir.model import HubmiRNet
from mircid.paths import DATA_ROOT, require


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="CSV: samples by 977 L1000 genes")
    parser.add_argument("output", type=Path, help="Output CSV")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DATA_ROOT / "model_artifacts" / "hubmirnet_414_pathway_8192_state_dict.pt",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _state_dict(payload: object) -> dict[str, torch.Tensor]:
    if isinstance(payload, torch.nn.Module):
        return payload.state_dict()
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                return candidate
        if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
            return payload  # type: ignore[return-value]
    raise ValueError("Unsupported checkpoint structure")


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(require(args.input), index_col=0)
    if frame.shape[1] != 977:
        raise ValueError(f"Expected 977 L1000 columns, found {frame.shape[1]}")
    values = frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Input contains non-finite values")

    # Public artifacts are tensor-only state dictionaries. Loading them in
    # weights-only mode avoids executing arbitrary pickle payloads.
    payload = torch.load(require(args.checkpoint), map_location="cpu", weights_only=True)
    state = {
        key: value
        for key, value in _state_dict(payload).items()
        if not key.startswith(("layer4.", "layer5.", "layer6.", "layer7."))
    }
    hidden_size, input_size = state["input_layer.weight"].shape
    output_size = state["output_layer.weight"].shape[0]
    model = HubmiRNet(
        input_size=int(input_size), hidden_size=int(hidden_size),
        output_size=int(output_size), num_blocks=3, dropout=0.0,
    )
    model.load_state_dict(state)
    model.to(args.device).eval()

    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(values), args.batch_size):
            batch = torch.from_numpy(values[start : start + args.batch_size]).to(args.device)
            outputs.append(model(batch).cpu().numpy())
    requested = set(
        require(DATA_ROOT / "hubmir" / "processed" / "hubmir_414.txt").read_text().splitlines()
    )
    target_header = pd.read_csv(
        require(DATA_ROOT / "hubmir" / "processed" / "TCGA_miRNA_train_zscore.csv"),
        nrows=0,
        index_col=0,
    ).columns
    target_names = [name for name in target_header if name in requested]
    if len(target_names) != 414:
        raise ValueError(f"Expected 414 output labels, found {len(target_names)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(np.vstack(outputs), index=frame.index, columns=target_names).to_csv(args.output)


if __name__ == "__main__":
    main()
