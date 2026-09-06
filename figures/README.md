# Figure code

Plotting is deliberately separated from model code. Every active entry point reads only from `MIRCID_dataset/figure_source_data/` (or a clearly named frozen image asset) and writes beside its script. The final-candidate exports are retained for visual comparison; files ending in `_replot` are clean-environment checks produced from the staged source tables.

| Directory | Active entry point | Reproduction status |
|---|---|---|
| `figure1/` | `recolor_workflow.py` | Recolours the manually composed workflow; it does not reconstruct the diagram primitives. |
| `figure2/` | `make_figure2_data_panels.py` | Replots the numerical benchmark and PCA panels; the architecture schematic remains a graphic asset. |
| `figure3/` | `make_figure3.py` | Replots aggregate values transcribed from the submitted panel; per-sample source outputs are outstanding. |
| `figure4/` | `make_figure4.py` | Replots paired seven-run final-candidate distributions and the PROGENy reference. |
| `figure4_5/` | `make_figure4_5.py` | Reproduces the five-panel feature-analysis figure. |
| `figure5/` | `make_figure5.py` | Recombines the frozen mechanism graphic with the source-data heatmaps. |
| `supplementary/` | each subdirectory's `make_figure.py` | Replots the staged supplementary source tables. |

Files prefixed `provenance_` preserve the exact historical construction/selection logic. They are retained for audit and may depend on the original private workspace; they are not the recommended public entry points.

