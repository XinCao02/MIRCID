# Release audit

## Ready in this staging package

- Core code is separated from data and figure code.
- The corrected 565-sample pathway matrices are aligned by explicit sample identifier.
- The 20 frozen pathway splits and selected ResNet configuration are present.
- Embedding-control, CKA/CCA/ridge and rescue code is collected with figure source tables.
- One canonical copy of each byte-identical dataset is retained.
- A data manifest and SHA-256 checksums can be rebuilt with `scripts/build_data_manifest.py`.

## Must be resolved before public release

- [ ] Authors approve a software licence.
- [ ] Authors confirm the `CITATION.cff` author order, names, ORCIDs and release metadata.
- [ ] Data owners confirm redistribution rights for processed TCGA/GEO/LINCS and network files.
- [x] Replace both recovered pickled HubmiR model objects with pruned state dictionaries; conversion max absolute error was 0.
- [ ] Confirm which 414-output checkpoint generated the reported reconstruction benchmark (4,096-unit script/checkpoint versus 8,192-unit aligned-pathway checkpoint).
- [ ] Recover the exact 1,298-output run/configuration or revise the manuscript claim.
- [ ] Recover per-sample Figure 3 outputs for all nine TFA configurations, or label the aggregate table as figure-digitized evidence only.
- [ ] Confirm the exact Priori implementation version and parameters used to create the pathway TFA matrix.
- [ ] Recover or rerun the full six-cell-line MoA benchmark; the included pickle is demo-sized.
- [ ] Rerun confirmatory pathway results on all frozen splits rather than presenting post-hoc seven-run subsets as prespecified repeats.
- [ ] Add permanent GitHub and dataset DOI URLs to `CITATION.cff` and both README files.
- [ ] Run a clean-environment reproduction and record package versions and expected output hashes.

## Excluded intentionally

- Historical misaligned PROGENy inputs and downstream results.
- FRoGS experiments.
- Duplicate CSV/RDS/RData serializations where a canonical tabular copy was retained.
- W&B caches, notebook caches, old checkpoints and backup files.
- Mock Figure 2 full-miRNA error bars.
- Manual one-off seed-search/figure variants except where required to disclose final-source provenance.
