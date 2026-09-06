# MIRCID

This repositry is the code release for the paper "**MIRCID**: Inferred **miR**NAs Drive **C**ross-Task **I**mprovements in **D**rug Mechanistic Modeling". It contains the core HubmiRNet, transcription-factor activity (TFA), pathway-classification, feature-analysis, rescue-analysis and drug mechanism-of-action (MoA) code. Plotting code is isolated in `figures/`; all distributable data and model artifacts live in the sibling `MIRCID_dataset/` archive.

## Repository layout

```text
MIRCID/
├── configs/                 # frozen model and experiment configurations
├── src/mircid/
│   ├── hubmir/              # HubmiRNet model, inference and historical trainers
│   ├── tfa/                 # TFA benchmark utilities and TIGER implementation
│   ├── pathway/             # pathway benchmark, controls, CKA/CCA and rescue
│   └── moa/                 # four connectivity scorers and evaluation driver
├── analysis/                # analysis guide and MoA mixed-effects sensitivity
├── figures/                 # one directory per main/supplementary figure
├── manifests/               # generated release manifests
├── tests/
└── docs/
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
export MIRCID_DATA_ROOT=/path/to/MIRCID_dataset
```

For TIGER, install `pip install -e '.[tfa]'` and configure CmdStan separately. Historical HubmiR scripts have additional optional dependencies documented by their imports and are not required for inference.

`MIRCID_DATA_ROOT` defaults to the sibling directory `../MIRCID_dataset`. Outputs default to `./outputs` and can be redirected with `MIRCID_WORK_ROOT`.

## Core entry points

```bash
# HubmiR inference from a samples × 977-gene CSV
python -m mircid.hubmir.predict input.csv outputs/hubmir.csv

# Pathway benchmark (paired frozen splits)
python -m mircid.pathway.run_benchmark --model svm
python -m mircid.pathway.run_resnet --mode formal
python -m mircid.pathway.run_progeny

# Matched embedding controls and feature complementarity
python -m mircid.pathway.prepare_embedding_controls
python -m mircid.pathway.run_embedding_benchmark --model svm
python -m mircid.pathway.run_complementarity
python -m mircid.pathway.run_decomposition_benchmark --model svm

# Analysis of complete frozen-split outputs
python -m mircid.pathway.analyze_rescue
python -m mircid.pathway.analyze_all
python -m mircid.pathway.validate_all

# Demo-sized MoA run; this is not the full manuscript benchmark
python -m mircid.moa.evaluate_rcsm --head 50 --top-n 2000 --permute-num 10

# Setting-level MoA mixed-effects sensitivity
Rscript analysis/statistics/fit_moa_mixed_effects.R
```

The exact legacy HubmiR training programs are preserved under `src/mircid/hubmir/legacy/`. They document the archived runs but retain notebook-style execution and should not be mistaken for the cleaned inference interface.

## Figure reproduction

Each figure directory contains its plotting entry point, final export and legend where available. Run scripts from the repository root after installing the package. Small plotting tables are stored in `MIRCID_dataset/figure_source_data/`.

- Figure 1: workflow graphic and recolouring source.
- Figure 2: numerical architecture benchmark and observed/predicted PCA replot.
- Figure 3: aggregate TFA heatmaps reconstructed from the published panel values.
- Figure 4: pathway-classification distributions and PROGENy reference.
- Figure 4.5: embedding controls, CCA, CKA, ridge recovery and paired task deltas.
- Figure 5: rescue-case mechanism panel and feature heatmaps.
- Supplementary: HubmiR feature analyses and rescue atlas.

## Reproducibility status

The aligned pathway workflow has the strongest end-to-end provenance: explicit sample-ID alignment, a frozen 20-split manifest, matched preprocessing and prediction-level outputs. The release preserves the following limitations rather than hiding them:

1. The current Figure 4/4.5 final-candidate source tables contain exploratory seven-run subsets selected from larger candidate pools. They must not be described as prespecified or as all available repeats. The 20-split aligned workflow is the appropriate basis for a fully confirmatory rerun.
2. Figure 3 aggregate values were transcribed from the submitted figure; the complete per-sample outputs for every method/network combination were not recovered from the historical workspace.
3. The included MoA matrices are a public demo subset. The full six-cell-line profile matrices underlying the manuscript table were not found locally; `MIRCID_dataset/moa/external_sources/` records the required source datasets.
4. The archived 1,298-output training script does not support the manuscript phrase “only the output dimension was changed”: its observed hyperparameters differ from the 414-output run. This discrepancy must be resolved before release of that claim.
5. Two different 414-output checkpoints were recovered: the aligned pathway provenance uses 8,192 hidden units, whereas the manuscript/training script specifies 4,096. Both were converted losslessly to portable state dictionaries and are kept distinct; the benchmark-to-checkpoint mapping still needs author confirmation.

See [release audit](docs/RELEASE_AUDIT.md) for the actionable checklist.

See also [upstream software](docs/UPSTREAM_SOFTWARE.md) and the [deposition plan](docs/DEPOSITION_PLAN.md).

Rebuild the code inventory with `python scripts/build_code_manifest.py`. The data inventory is rebuilt separately with `python scripts/build_data_manifest.py`.

## Data, licensing and citation

The sibling dataset archive includes a machine-readable manifest and SHA-256 checksums. Several inputs derive from TCGA, GEO, LINCS, CollecTRI, DoRothEA and PROGENy; redistribution rights must be checked before public upload. No blanket data licence has been asserted.

The software licence is intentionally pending author approval. Add an OSI-approved licence before making the GitHub repository public. Update `CITATION.cff` and both repository identifiers after Zenodo deposition.
