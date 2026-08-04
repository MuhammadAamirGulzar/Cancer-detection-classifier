# Running the missing combinations on the server

**Why this exists.** SurGen aggregated features for `H-Optimus-1`, `UNI2` and `ConchV1` live only on the Linux server (`/media/dp-psau/Datum/Aamir/Azfaar`). The cohorts are too large to consolidate onto one machine, so the code travels instead of the data: run the identical pipeline there, copy back only the small result files, and merge.

**16 of 22 SurGen-EV combinations** and the corresponding **SurGen-CV** combinations are waiting on this. Everything else is already complete locally.

---

## 1. What to copy across

Only source and configuration — no data, no models, no results:

```
Cancer-detection-classifier/
  slide_classification/
    config/            paths.py, __init__.py
    runners/           all modules
    eval_patch_features/
    data_layer.py
    utility.py
    kfolds_IDARS_fixed.csv
    surgen_labels.csv
    paip_78_labels.csv
    tcga_full_hparams.json     <- REQUIRED, see step 4
    thresholds_TCGA.json       <- REQUIRED, see step 4
  tools/
```

Roughly a few MB. `git clone` the `dev` branch is the cleanest way to get exactly this.

---

## 2. Point the code at the server's data

Everything resolves from **one** environment variable. No file needs editing:

```bash
export MACHINE=server
```

`slide_classification/config/paths.py` already carries the server roots:

```python
"server": {
    "base":             "/media/dp-psau/Datum/Aamir/Azfaar",
    "tcga_agg":         "dataset/slide_aggregation",
    "paip_agg":         "paip_data/slide_aggregation",
    "paip_labels":      "paip_data/labels",
    "surgen_processed": "surgen_processed",
},
```

> **These values were transcribed from the old hardcoded scripts and have not been verified from a server shell.** Step 3 checks them before anything long-running starts. If a root is wrong, fix it in `paths.py` only — nowhere else hardcodes a path.

---

## 3. Verify what the server can actually see (do this first)

```bash
cd slide_classification
MACHINE=server python config/paths.py
```

This scans the feature tree and writes `available_features.json`, printing every `(cohort, method, model)` combination physically present with its file count. **Never assume — always scan.**

Expect SurGen entries for `H-Optimus-1`, `UNI2` and `ConchV1`. If they are absent, the roots in `paths.py` are wrong; fix them and re-run before continuing.

---

## 4. Two files must come from the local machine, not be regenerated

`tcga_full_hparams.json` and `thresholds_TCGA.json` are derived from **TCGA-CV**, and TCGA-CV has already been run locally on the complete cohort.

**Copy both across. Do not regenerate them on the server.** Regenerating would refit hyperparameters and the decision threshold on whatever TCGA data the server happens to hold, which would silently produce a second, incompatible calibration and break comparability between the two halves of the result set.

The runners will refuse to proceed without them rather than fall back to a default.

---

## 5. Run

```bash
cd slide_classification
MACHINE=server python -m runners.cv_runner --experiment SurGen-CV
MACHINE=server python -m runners.ev_runner --experiment SurGen-EV
```

Optionally, to fill any other gaps the server can cover:

```bash
MACHINE=server python -m runners.cv_runner --experiment TCGA-CV
MACHINE=server python -m runners.iv_runner
MACHINE=server python -m runners.ev_runner --experiment PAIP-EV
```

Notes:

- **Resumable.** Progress is recorded per combination in `progress.json`; an interrupted run skips what already finished. Pass `--force` to redo completed work.
- **Skips, never fails, on missing data.** Anything unavailable is logged to `skipped_combinations.csv` and the run continues.
- **Halts loudly on correctness failures** — a shape mismatch, an unresolvable label, or case-level leakage stops the run rather than being papered over.
- Logs land in `logs/run_<timestamp>.log`.

---

## 6. Copy results back

Only these are needed — small text files, no binaries:

```
slide_classification/SurGen_Results/**/Output/*.json   *.csv   *.xlsx
slide_classification/SurGen_EV_Results/**/Output/*.json   *.csv   *.xlsx
slide_classification/skipped_combinations.csv
slide_classification/available_features.json
```

Model checkpoints (`*.pth`, `*.pkl`) do **not** need transferring — they are large and nothing downstream reads them once the metrics exist.

Put them in a directory alongside the local tree, e.g. `server_results/`.

---

## 7. Merge

```bash
python tools/merge_results.py slide_classification server_results \
    --out slide_classification/best_of_all_exps_metric.xlsx
```

`merge_results.py` keys every record on `(experiment, method, model, variant)`. Each result file is stamped with its machine, timestamp and git commit, so on a duplicate key it keeps the **newer commit** and logs the collision to a `Collisions` sheet rather than overwriting silently. Archived trees are never merged.

Then regenerate the reporting layer on the combined set:

```bash
python tools/build_report.py
python tools/make_radar_plots.py
python tools/write_results_md.py
python tools/compare_to_archive.py
python tools/build_supplementary_results.py
```

The radar plots and RESULTS.md will then show the full matrix instead of the local subset — which is the point of the exercise.

---

## 8. One thing to fix before running the *aggregation* stage there

If the server also needs to **generate** SurGen features rather than just consume existing ones, note that the upstream feature-extraction and aggregation scripts (`Complete_Pipeline/`, `slide_aggregation/`, `data_preprocessing/` — 23 files) still carry hardcoded machine-specific roots. They were deliberately left alone because the classification remediation never executes them (`OPEN_QUESTIONS.md` #4).

They will need either the same `config/paths.py` treatment or explicit CLI arguments before they run cleanly on a second machine. The classification runners above are unaffected.
