# MIRCID

This repositry is the code release for the paper "**MIRCID**: Inferred **miR**NAs Drive **C**ross-Task **I**mprovements in **D**rug Mechanistic Modeling". It contains the core HubmiRNet, transcription-factor activity (TFA), pathway-classification, feature-analysis, rescue-analysis and drug mechanism-of-action (MoA) code. All distributable data and model artifacts live in the sibling `MIRCID_dataset/` archive.

## Paper Abstract
Drug mechanism-of-action (MoA) modeling commonly relies on perturbational transcriptomes, but endpoint gene expression can be noisy and may incompletely represent regulatory state. Here, we present MIRCID, a framework comparing gene expression with inferred transcription factor activity and microRNA expression across pathway classification and similarity-based MoA retrieval. HubmiRNet inferred 414 pan-cancer hub miRNAs from 977 L1000 landmark genes (PCC, 87.72%) and, when expanded to 1,298 outputs, outperformed SiCmiR on its original task (71.21% versus 67.30%). Across RF, MLP, KAN, and ResNet pathway classifiers and four retrieval algorithms, HubmiR augmentation delivered the most consistent gains, whereas TF activity was less consistently beneficial. Dimension-matched controls did not reproduce these gains, and CCA, CKA, and ridge analyses showed that HubmiR preserved gene-derived signal while reorganizing it into a distinct, partially linearly recoverable representation. Rescue analyses further linked improvement to cases with weak gene-level pathway signals. Thus, inferred HubmiRs provide a biologically informed, task-relevant recoding of perturbational transcriptomes for drug-mechanism modeling.

## Repository layout

```text
MIRCID/
├── configs/                 # frozen model and experiment configurations
├── src/
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
- Figure 5: embedding controls, CCA, CKA, ridge recovery and paired task deltas.
- Figure 6: rescue-case mechanism panel and feature heatmaps.
- Supplementary: HubmiR feature analyses and rescue atlas.


## Data, licensing and citation

The sibling dataset archive includes a machine-readable manifest and SHA-256 checksums. Several inputs derive from TCGA, GEO, LINCS, CollecTRI, DoRothEA and PROGENy; redistribution rights must be checked before public upload. No blanket data licence has been asserted.

The software licence is intentionally pending author approval. Add an OSI-approved licence before making the GitHub repository public. Update `CITATION.cff` and both repository identifiers after Zenodo deposition.
