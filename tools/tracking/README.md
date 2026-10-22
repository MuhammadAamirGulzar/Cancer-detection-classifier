# Result tracking with MLflow

One local MLflow store holds the benchmark configurations in one schema, so encoders, aggregations, heads and the five-crop / no-crop arms can be filtered and compared side by side.

- Store: `D:/Aamir Gulzar/KSA_project2/mlflow_clean/` (outside the repository, not in git)
- Build: [build_clean_store.py](build_clean_store.py). The store is rebuilt from scratch from the study tables, so nothing deleted or superseded stays in it.
- Env: `mlflow_tracking` (separate from `exaonepath`; recreate with `conda env create -f tools/tracking/environment.yml`)
- Open: double-click [mlflow_ui_clean.bat](mlflow_ui_clean.bat), then go to http://127.0.0.1:5001

```powershell
& "$env:USERPROFILE\anaconda3\envs\mlflow_tracking\python.exe" tools\tracking\build_clean_store.py --fresh
```

## How the store is organised

- **One experiment per setting**: TCGA-CV, PAIP-IV, PAIP-EV (SurGen-CV and SurGen-EV to be added).
- **One run per configuration**, named `encoder | aggregation | head | arm`.
- **Two arms** (tag `arm`):

  | Arm | Patch features | Aggregations |
  |---|---|---|
  | `fivecrop` (baseline) | five 256 px crops per 512 px patch | Averaging, TITAN, PRISM as published. Caption-14, Caption-15 and Tissue-Type Clustering with **crop-level grouping**: each crop goes to the group of its own label (CONCH v1 zero-shot caption at 224 px; CRC-100K tissue classifier, per crop) |
  | `nocrop` | one whole-patch view per patch | Averaging, Caption-14, Caption-15, Tissue-Type Clustering (patch-level, one view per patch) |

- **Params** (identical on every run): `setting`, `validation`, `train_cohort`, `test_cohort`, `encoder`, `aggregation`, `head`, `crop`, `grouping`, `caption_level`, `ttc_rule`, `precision`, `threshold_rule`, `n_test`, `n_pos`, `features`.
- **Tags**: `role` (`baseline` = fivecrop, `ablation` = nocrop), `study`, `arm`.
- **Metrics**: `auroc` and `bacc` with 95% bootstrap intervals (`*_ci_lo`, `*_ci_hi`; class-stratified, B = 2000). No-crop runs also carry `delta_auroc` / `delta_bacc` (+ interval, p): the paired no-crop − five-crop difference for the same encoder, aggregation and head. Balanced accuracy is taken at each setting's operating point (see `threshold_rule`).

## Inputs

| Table | Rows |
|---|---|
| `experiments/nocrop_vs_fivecrop/results/all/mlflow_table.csv` (`06_mlflow_table.py`) | five-crop Averaging, TITAN, PRISM; all no-crop rows |
| `experiments/crop_grouping/results/mlflow_table.csv` (`05_mlflow_table.py`) | five-crop crop-level Caption-14 / Caption-15 / TTC |
| `experiments/crop_grouping/results/contrasts.csv` (`04_compare.py`) | paired crop-level − no-crop differences, which give the no-crop deltas for the grouped aggregations |

## Useful filters (search box in the runs table)

| Question | Filter |
|---|---|
| The baseline only | `tags.arm = 'fivecrop'` |
| Five-crop vs no-crop for one encoder | `params.encoder = 'UNI2'`, then group by `tags.arm` |
| One aggregation across encoders and heads | `params.aggregation = 'Caption-15'` |

- **Compare**: tick runs, then press *Compare*; the parameter table marks every parameter that differs.
- **Group by** (runs table toolbar): grouping by `params.aggregation`, `params.encoder` or `tags.arm` shows mean, min and max per group.

## Adding results

Run a new study in its own `experiments/<name>/` folder through the runners, write a table in the same schema (see `06_mlflow_table.py`), add it to `build_clean_store.py`, and rebuild with `--fresh`.

The earlier ingest store (`mlflow_store/`, every historical result file) was retired on 2026-10-22. Its run table and the per-run `record.json` copies are kept outside the repository: `KSA_project2/mlflow_store_export_20261022.csv` and `KSA_project2/mlflow_store_records_20261022.zip`.
