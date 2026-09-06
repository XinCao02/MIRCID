from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mircid.paths import DATA_ROOT

from .common import ROOT, environment_manifest, load_config, normalize_sample_ids, sha256_file, write_json


SOURCE = DATA_ROOT / "pathway" / "source"


def read_numeric(path: Path, transpose: bool = False) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    if transpose:
        frame = frame.T
    frame.index = normalize_sample_ids(frame.index)
    if frame.index.duplicated().any() or frame.columns.duplicated().any():
        raise ValueError(f"Duplicate IDs in {path}")
    frame = frame.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"Non-finite values in {path}")
    return frame


def main() -> None:
    cfg = load_config()
    gene = read_numeric(SOURCE / "X_0_all.csv")
    tfa = read_numeric(SOURCE / "PROGENy_tf.csv", transpose=True)
    mirna = read_numeric(SOURCE / "PROGENy_mirna_aligned.csv")
    labels = read_numeric(SOURCE / "Y_all.csv")

    meta = pd.read_csv(SOURCE / "PROGENy_meta.csv")
    meta["id"] = normalize_sample_ids(meta["id"])
    if meta["id"].duplicated().any():
        raise ValueError("Duplicate metadata sample IDs")
    meta = meta.set_index("id")

    reference_ids = gene.index
    frames = {"tfa": tfa, "mirna": mirna, "labels": labels, "meta": meta}
    alignment_rows = []
    for name, frame in frames.items():
        missing = reference_ids.difference(frame.index)
        extra = frame.index.difference(reference_ids)
        alignment_rows.append(
            {"table": name, "rows": len(frame), "missing_vs_gene": len(missing), "extra_vs_gene": len(extra)}
        )
        if len(missing) or len(extra):
            raise ValueError(f"Sample mismatch for {name}: missing={list(missing[:5])}, extra={list(extra[:5])}")
        frames[name] = frame.loc[reference_ids]
    tfa, mirna, labels, meta = frames["tfa"], frames["mirna"], frames["labels"], frames["meta"]

    y = labels.iloc[:, 0].to_numpy(dtype=np.int64)
    if sorted(np.unique(y).tolist()) != list(range(11)):
        raise ValueError(f"Expected labels 0..10; got {sorted(np.unique(y))}")
    class_map = meta.groupby(y)["pathway"].agg(lambda x: sorted(set(x)))
    if any(len(x) != 1 for x in class_map):
        raise ValueError("A numeric label maps to multiple pathways")
    class_names = np.asarray([class_map.loc[i][0] for i in range(11)], dtype="U32")
    if not np.array_equal(meta["pathway"].to_numpy(dtype=str), class_names[y]):
        raise ValueError("Metadata pathway and numeric label disagree")

    input_frame = pd.read_csv(SOURCE / "PROGENy_input_aligned.csv")
    required_meta = ["name", "name2", "index"]
    if not set(required_meta).issubset(input_frame.columns):
        raise ValueError("Aligned 977-input metadata columns missing")
    sample_columns = [c for c in input_frame.columns if c not in {"Unnamed: 0", *required_meta}]
    input_ids = normalize_sample_ids(sample_columns)
    if input_ids.duplicated().any() or set(input_ids) != set(reference_ids):
        raise ValueError("Aligned 977-input sample IDs do not exactly match downstream samples")
    renamed = dict(zip(sample_columns, input_ids))
    input_values = input_frame[sample_columns].rename(columns=renamed).loc[:, reference_ids].T
    input_values.columns = input_frame["name"].astype(str)
    if input_values.shape != (565, 977) or not np.isfinite(input_values.to_numpy()).all():
        raise ValueError(f"Bad aligned HubmiR input shape/content: {input_values.shape}")

    (DATA_ROOT / "pathway" / "processed").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA_ROOT / "pathway" / "processed" / "pathway_aligned.npz",
        gene=gene.to_numpy(dtype=np.float32),
        tfa=tfa.to_numpy(dtype=np.float32),
        mirna=mirna.to_numpy(dtype=np.float32),
        l1000=input_values.to_numpy(dtype=np.float32),
        y=y,
        sample_ids=reference_ids.to_numpy(dtype="U128"),
        class_names=class_names,
        gene_names=gene.columns.to_numpy(dtype="U64"),
        tfa_names=tfa.columns.to_numpy(dtype="U64"),
        mirna_names=mirna.columns.to_numpy(dtype="U64"),
        l1000_names=input_values.columns.to_numpy(dtype="U64"),
        effect=meta["effect"].to_numpy(dtype="U16"),
        sign=meta["sign"].to_numpy(dtype=np.int8),
        accession=meta["accession"].fillna("").to_numpy(dtype="U64"),
        cells=meta["cells"].fillna("").to_numpy(dtype="U256"),
        treatment=meta["treatment"].fillna("").to_numpy(dtype="U256"),
    )

    split_rows = []
    all_idx = np.arange(len(y))
    for seed in cfg["split_seeds"]:
        train_idx, held_idx = train_test_split(all_idx, test_size=0.30, random_state=seed, stratify=y)
        val_idx, test_idx = train_test_split(held_idx, test_size=0.50, random_state=seed, stratify=y[held_idx])
        for partition, indices in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
            for idx in indices:
                split_rows.append(
                    {
                        "run_id": cfg["run_id"],
                        "split_seed": seed,
                        "sample_id": reference_ids[idx],
                        "label": int(y[idx]),
                        "pathway": class_names[y[idx]],
                        "partition": partition,
                    }
                )
    splits = pd.DataFrame(split_rows)
    (DATA_ROOT / "pathway" / "splits").mkdir(parents=True, exist_ok=True)
    splits.to_csv(DATA_ROOT / "pathway" / "splits" / "frozen_20_splits.csv", index=False)

    source_manifest = []
    for path in sorted(SOURCE.iterdir()):
        if path.is_file():
            source_manifest.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    (ROOT / "manifests" / "data").mkdir(parents=True, exist_ok=True)
    (ROOT / "manifests" / "features").mkdir(parents=True, exist_ok=True)
    (ROOT / "validation").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(source_manifest).to_csv(ROOT / "manifests" / "data" / "source_files.csv", index=False)
    pd.DataFrame(alignment_rows).to_csv(ROOT / "validation" / "sample_alignment.csv", index=False)
    pd.DataFrame(
        {
            "modality": ["gene", "tfa", "mirna", "l1000", "label"],
            "samples": [565] * 5,
            "features": [gene.shape[1], tfa.shape[1], mirna.shape[1], input_values.shape[1], 1],
        }
    ).to_csv(ROOT / "manifests" / "features" / "dimensions.csv", index=False)
    write_json(ROOT / "manifests" / "environment_local.json", environment_manifest())
    write_json(
        ROOT / "validation" / "data_validation.json",
        {
            "status": "PASS",
            "samples": 565,
            "classes": class_names.tolist(),
            "split_seeds": cfg["split_seeds"],
            "partition_sizes": splits.groupby(["split_seed", "partition"]).size().unstack().to_dict("index"),
            "explicit_sample_id_join": True,
            "finite_numeric_values": True,
            "aligned_mirna_provenance": json.loads((SOURCE / "aligned_mirna_provenance.json").read_text()),
        },
    )
    print("PASS: aligned data and 20 frozen splits prepared")


if __name__ == "__main__":
    main()
