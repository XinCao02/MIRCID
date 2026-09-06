"""Convert the historical full-object checkpoint to a portable, pruned state dict."""

from __future__ import annotations

import __main__
import argparse
from pathlib import Path

import torch

from mircid.hubmir.model import HubmiRNet, ResidualBlock, ResNetMLP


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy_checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    __main__.ResNetMLP = ResNetMLP
    __main__.ResidualBlock = ResidualBlock
    legacy = torch.load(args.legacy_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(legacy, torch.nn.Module):
        raise TypeError("Expected a serialized PyTorch module")
    state = {
        key: value
        for key, value in legacy.state_dict().items()
        if not key.startswith(("layer4.", "layer5.", "layer6.", "layer7."))
    }
    hidden_size, input_size = state["input_layer.weight"].shape
    output_size = state["output_layer.weight"].shape[0]
    clean = HubmiRNet(
        input_size=int(input_size),
        hidden_size=int(hidden_size),
        output_size=int(output_size),
        num_blocks=3,
        dropout=float(getattr(legacy.dropout, "p", 0.0)),
    )
    clean.load_state_dict(state)
    legacy.eval()
    clean.eval()
    probe = torch.randn(2, int(input_size))
    with torch.inference_mode():
        error = float((legacy(probe) - clean(probe)).abs().max())
    if error > 1e-6:
        raise AssertionError(f"Conversion mismatch: {error}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": clean.state_dict(), "conversion_max_abs_error": error}, args.output)
    print(f"PASS: max_abs_error={error:.3g}; wrote {args.output}")


if __name__ == "__main__":
    main()
