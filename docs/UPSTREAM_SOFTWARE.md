# Upstream software and methods

MIRCID keeps author-written analysis code separate from third-party implementations. The following methods named in the manuscript are not re-vendored as complete packages.

| Method/resource | Role in MIRCID | Release treatment |
|---|---|---|
| VIPER | TFA comparator | Install and cite the maintained upstream R/Bioconductor implementation. MIRCID contains benchmark evaluation utilities, not a private fork. |
| Priori | TFA method selected for the pathway TFA representation | The aligned `PROGENy_tf.csv` output is included in the data package. The exact historical executable/package version and parameters still require author confirmation before a complete rerun can be claimed. |
| TIGER | TFA comparator | The recovered Python translation and Stan model are included under `src/mircid/tfa/`; use the `tfa` optional dependency set and a compatible CmdStan installation. |
| CollecTRI and DoRothEA | Prior-knowledge networks | Frozen tabular networks are staged in the data package. Record upstream versions, access dates, citations and licences before public deposition. |
| PROGENy | Fixed pathway footprint-score baseline | The official footprint source and licence notice are preserved under `src/mircid/pathway/vendor/progeny/`; the model data used by the workflow are in the data package. |
| KAN | Historical HubmiR architecture comparator | Exact recovered legacy scripts are retained for provenance. The cleaned downstream KAN-like classifier in `src/mircid/pathway/networks.py` has no external KAN package dependency. |

The release should cite upstream methods rather than imply that their source code was authored in this repository. Vendored files retain their own notices and must not be relicensed under the future MIRCID software licence.
