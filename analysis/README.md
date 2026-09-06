# Analysis entry points

The reusable pathway analyses are packaged under `src/mircid/pathway/`:

- `analyze_all.py` builds absolute summaries, paired contrasts, embedding-control summaries, complementarity summaries and rescue statistics from complete frozen-split outputs.
- `analyze_rescue.py` creates the prediction-level rescued/harmed classifications and pathway summaries.
- `validate_all.py` checks prediction grids, metric reconstruction and leakage-related manifests.

`statistics/fit_moa_mixed_effects.R` is the retained setting-level MoA sensitivity analysis. It accepts optional input, output and diagnostics paths. Query- or compound-level inference cannot be reproduced until the full manuscript MoA outputs are recovered.

Historical analyses tied to the misaligned pathway input or to absolute private-workspace paths are intentionally excluded from this release.
