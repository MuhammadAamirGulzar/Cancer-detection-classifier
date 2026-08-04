# Phase 1 Report — Correctness fixes

**Date:** 2026-08-04
**Branch:** `dev` (all remediation work; `master`/`main` preserves the pre-remediation state)
**Commits:** `5aad5fe`, `61c2e21`, plus the Task 1.1 sweep commit below
**Environment:** `exaonepath` conda env — Python 3.10.18, torch 2.2.0+cu121 (CUDA on RTX 4090), scikit-learn 1.7.0, pandas 2.3.1

Phase 1 changes *numbers*. Every result produced after this phase supersedes the equivalent result produced before it.

---

## 1. Summary of what changed

| Task | Status | Effect on numbers |
|---|---|---|
| 1.0 Data-layer refactor | Done | **None** — verified bit-identical (this is the point) |
| 1.1 ANN hyperparameters selected on validation | Done | ANN metrics **drop slightly** (removes optimistic bias) |
| 1.2 Save the ANN checkpoint actually selected | Done | PAIP-EV/SurGen-EV change — they were loading the wrong network |
| 1.3 Remove the double softmax | Done | ANN metrics **improve** |
| 1.4 Fail loudly on missing features | Done | Reported N changes 416→413 (TCGA), 78→73 (PAIP) |
| 1.5 Retire stale scripts | Done | None |

---

## 2. Task 1.0 — Data-layer refactor

### What was built

- **`slide_classification/config/paths.py`** — one source of truth for every filesystem root, keyed on the `MACHINE` environment variable (`local` | `server`). Also holds the canonical→on-disk model-name map (SurGen stores `conch1-5`/`virchow2`; results use `Conch1_5`/`Virchow2` — explicit, because case-insensitive path resolution works on Windows and fails on Linux), `expected_dim()`, the TITAN/PRISM combination guards, and `scan_available_features()`.
- **`slide_classification/data_layer.py`** — the single replacement for ~6 duplicated `WSIDataset` classes. `load_cohort()` → `(feats[N,D], labels[N], ids)`; `build_splits()`, `cv_rotation()`, `stratified_holdout()`, `assert_no_leakage()`.
- **`slide_classification/runners/`** — `classifiers.py`, `cv_runner.py`, `results_io.py`, `thresholds.py`, `runlog.py`.

### Inefficiencies fixed (§1c.1)

| # | Problem | Fix |
|---|---|---|
| 1 | Every `.pt` re-read ~20× per (method, model) | Cohort assembled once; splits are index slices |
| 2 | O(n²) label lookup (two linear DataFrame scans per file) | `dict(zip(ids, labels))` built once |
| 3 | `DataLoader(batch_size=4)` immediately undone by `torch.cat` | DataLoader layer deleted |
| 4 | `write_data_in_excel` re-read/rewrote all sheets per call | Accumulate in memory, write workbook once |
| 5 | Redundant re-computation across runs | `.npz` cache keyed on file-count + newest-mtime + name digest |
| 6 | Single-threaded sklearn | Already `n_jobs=-1` in `knn.py` and `r_forest_eval.py` — verified, no change needed |
| 7 | ANN on CPU | Already CUDA-with-CPU-fallback — verified, no change needed |

### Acceptance — metric parity

`tools/parity_check.py`, TCGA / Tissue_Type_Clustering / H-Optimus-1 (D=13824):

```
CHECK A - DATA IDENTITY (old WSIDataset vs new data layer)
  fold1 train n_old=213 n_new=213 ids=OK feats=OK labels=OK max|delta|=0.00e+00
  fold1 val   n_old= 97 n_new= 97 ids=OK feats=OK labels=OK max|delta|=0.00e+00
  fold1 test  n_old=103 n_new=103 ids=OK feats=OK labels=OK max|delta|=0.00e+00
  ... (all 12 splits identical) ...
  RESULT: PASS - data layer is bit-identical

CHECK B - METRIC PARITY (fold 1, identical row order, seed fixed)
  [lin]   max|delta| = 0.000e+00  PASS
  [knn]   max|delta| = 0.000e+00  PASS
  [proto] max|delta| = 0.000e+00  PASS
  [rf]    max|delta| = 0.000e+00  PASS
  [ann]   max|delta| = 0.000e+00  PASS
  RESULT: worst delta across all classifiers = 0.000e+00 (PASS, tolerance 1e-6)
```

Exact bit-level agreement, four orders of magnitude tighter than the required 1e-6.

### Acceptance — benchmark

```
CHECK C - BENCHMARK (data-loading cost, 5 classifiers x 4 folds)
  OLD: 3 WSIDataset builds x 4 folds x 5 classifiers = 60 builds -> 3.3s of loading
  NEW: 1 assembly, reused by every fold and classifier -> 0.1s
  speedup on the data path: 41.5x
  NEW (warm .npz cache): 0.02s -> 162x
```

**41.5×** against the **≥5×** target. Two honest caveats:

1. These are **warm-OS-cache** numbers. The first cold read of 413 files took 5.2s, so cold-cache absolute times are ~15× larger on both sides; the *ratio* is what the refactor controls.
2. This measures the **data path**, which is what §1c.1 diagnosed. Total end-to-end wall-clock also contains classifier training (RF on 13824 features dominates), which the refactor does not change and which caps the whole-pipeline speedup below 41.5×.

### Reproducibility defect found and fixed (not in the work order)

The old train loader used `DataLoader(..., shuffle=True)` with **no seed**. Random Forest's bootstrap sampling depends on row order, so *the old pipeline produced different RF numbers on every run of identical code.* The new data layer uses deterministic sorted row order, and every classifier call is seeded. Previously published RF numbers are therefore not exactly reproducible even from unchanged code — worth knowing before comparing old and new RF results.

### Acceptance — one `WSIDataset`

One definition remains in the active classification path: `data_layer.py`. Still-outstanding copies and their disposition:

| File | Disposition |
|---|---|
| `Slide_Classification.ipynb` cell 6 | **Replaced** — now a 40-line driver over `runners.cv_runner` |
| `Slide_Classification.ipynb` cell 8 | Retired by Task 2.2 (PAIP-CV) |
| `windows_/linux_/slide_classification_surgen.py` | Superseded by `cv_runner` in Task 2.1 |
| `Slide_Classification_TTC_exp.ipynb` | TTC side-experiment; selection fixed, loader untouched |
| `misc/*`, `.ipynb_checkpoints/*` | Already retired / not active code |

---

## 3. Task 1.1 — ANN hyperparameters were selected on the test fold

`eval_ANN` now also returns validation metrics under a `val_` prefix, and every selection site reads them.

### Measured bias

TCGA / Tissue_Type_Clustering / H-Optimus-1, all four folds, seed fixed:

| Fold | Selected by VAL | Selected by TEST | test macro-F1 @val | test macro-F1 @test | Bias |
|---|---|---|---|---|---|
| 1 | 128/64 | 128/64 | 0.7412 | 0.7412 | +0.0000 |
| 2 | 128/64 | 256/128 | 0.7623 | 0.7634 | +0.0011 |
| 3 | 128/64 | 128/128 | 0.6828 | 0.7158 | **+0.0330** |
| 4 | 256/64 | 256/64 | 0.7646 | 0.7646 | +0.0000 |

**Mean optimistic bias under the old rule: +0.0085 test macro-F1.** Folds 2 and 3 genuinely select a different architecture. This is one (method, model) pair — the same bias applies to every ANN number in the previous result set.

### Scope verification (the work order asked for this explicitly)

`lin` (`C=[10]`, `max_iter=[300]`), `knn` (`k=[3]`), `proto` (no grid) and `rf` (single combination) all still have **single-point grids**, so no selection occurs and **their metrics are unbiased**. Confirmed file by file.

One nuance worth recording: `eval_knn` runs an *internal* `GridSearchCV` over 30 parameter combinations — but with cross-validation **on the training data**, scoring balanced accuracy. That is a legitimate nested search, not test-set selection.

### A second form of the bug the work order's grep would have missed

Beyond the documented `eval_metrics['ann_macro_f1'] > ...` pattern, four files select through an indirection:

```python
metric_key = f"{model_type}_macro_f1"     # -> "ann_macro_f1" == TEST metric
...
if best_metrics is None or m[metric_key] > best_metrics[metric_key]:
```

`Slide_Classification_best_k_labels_exp_updated.ipynb` (cells 7, 11, 15, 19), `Slide_Classification_TTC_exp.ipynb` (cell 8), and `slide_classification_surgen_best_k_labels_exp.py` all use this form **with real 2×2 ANN grids** — so they carried the same bias. All patched to:

```python
metric_key = f"val_{model_type}_macro_f1" if model_type == "ann" else f"{model_type}_macro_f1"
```

`Tissue_Type_Combinations_Exp/TTC_combination_experiment.py` uses the same indirection but its ANN grid is single-point (`[256],[64],[500]`), so **no selection occurred and its results are unbiased**. Patched anyway, so widening the grid later cannot silently reintroduce the bug.

---

## 4. Task 1.2 — The saved ANN checkpoint was not the selected one

`torch.save` sat *inside* the caller's grid loop, writing to `fold{f}_trained_ann_model_{input_dim}.pth`. `input_dim` does not vary across the grid, so all four configurations wrote to the same path and the file on disk was the **last** entry tried (`256/128`), not the selected one.

**Consequence:** PAIP-EV ran a different network than the one whose TCGA-CV metrics were printed beside it.

Fixed: `eval_ANN` returns the model; `save_ann_checkpoint()` writes once, after selection.

```
files written : ['fold1_ann_13824_128_64.pth', 'fold1_ann_config.json']
config.json   : input_dim=13824 h1=128 h2=64 selected_on=val_ann_macro_f1=0.738544
state dict    : 0.weight=(128, 13824)  4.weight=(64, 128)
selected cfg  : h1=128 h2=64 input_dim=13824
ASSERT state dict == config.json == selected configuration: PASS
```

`load_ann_checkpoint()` reads the config JSON and keeps state-dict-shape inference as a fallback for legacy checkpoints, warning when it does so.

---

## 5. Task 1.3 — Double softmax

The network ended in `nn.Softmax(dim=1)` and was trained with `nn.CrossEntropyLoss`, which applies log-softmax internally and expects raw logits — softmax applied twice, flattening gradients.

One fold (TCGA / Tissue_Type_Clustering / H-Optimus-1, fold 1, h1=128 h2=64, seed fixed):

| Variant | bacc | AUROC | macro-F1 | acc | confusion matrix |
|---|---|---|---|---|---|
| OLD (Softmax + CrossEntropyLoss) | 0.8148 | 0.8742 | 0.7308 | 0.8252 | `[[73, 15], [3, 12]]` |
| NEW (logits + CrossEntropyLoss) | 0.8205 | 0.8879 | 0.7412 | 0.8350 | `[[74, 14], [3, 12]]` |
| **Δ (new − old)** | **+0.0057** | **+0.0136** | **+0.0103** | **+0.0097** |

Improvement on every metric — the expected direction. The work order's read that "ANN results are probably a floor, not a ceiling" is supported.

**Every ANN checkpoint and every ANN result must be regenerated.** Sequenced with Phase 3 so the ANN is trained once, not twice.

---

## 6. Task 1.4 — Labelled slides with no features

The old loader iterated files on disk and skipped anything not in `fold_ids`; a labelled slide with no feature file silently never appeared. No warning, no count check.

Now reported on load, and written to `missing_slides.csv` in each run's Output folder:

```
[load ] Cohort(tcga/Tissue_Type_Clustering/H-Optimus-1/MSIH: N=413, D=13824,
        classes={0: 353, 1: 60}, missing=3, cached=False)
[WARN] 3 labelled slide(s) have no features -> reported N=413, not 416:
[WARN]   missing features: TCGA-AD-6895_MSIH
[WARN]   missing features: TCGA-AD-6899_nonMSIH
[WARN]   missing features: TCGA-CM-6680_nonMSIH
[INFO] 4 slide(s) have features but no label (harmless, excluded):
       TCGA-D5-7000_nonMSIH, TCGA-EI-6917_nonMSIH, TCGA-F5-6814_nonMSIH, TCGA-T9-A92H_nonMSIH
```

Exactly the 3 slides the work order names. Verified across all three cohorts:

| Cohort | Labels | With features | Class split | Missing |
|---|---|---|---|---|
| TCGA | 416 | **413** | 60 MSI-H / 353 non | the 3 above |
| PAIP | 78 | **73** | 17 MSI-H / 56 non | the 5 the work order names |
| SurGen | 624 usable | **622** | 60 MSI-H / 562 non | `SR386_40X_HE_T086_01`, `SR386_40X_HE_T339_01` |

> **This answers a Task 3.2 question early.** The work order asks to "identify the 2 missing [SurGen slides] and report them" — they are `SR386_40X_HE_T086_01` and `SR386_40X_HE_T339_01`.

### Fold-size clarification for the report

The work order's TCGA fold table (103 / 98 / 112 / 103, MSI-H 15/13/20/13) is the **label-file** view of all 416 slides. After excluding the 3 without features, the **actual modelled** folds are:

| Fold | n (labels) | n (with features) | MSI-H (labels) | MSI-H (with features) |
|---|---|---|---|---|
| 1 | 103 | 103 | 15 | 15 |
| 2 | 98 | **97** | 13 | 13 |
| 3 | 112 | 112 | 20 | 20 |
| 4 | 103 | **101** | 13 | **12** |
| **Total** | 416 | **413** | 61 | **60** |

Both tables are correct; they count different things. The paper must quote the 413 row. (`TCGA-CM-6680` is in fold 2; `TCGA-AD-6895_MSIH` and `TCGA-AD-6899` are both in fold 4, which is why fold 4 loses two slides and the only missing MSI-H case.)

### Undefined-`label` bug

In the `BRAF`/`KRAS`/`TP53` branches, `label` was left **undefined** when `wsi_id` was absent from `folds_df`, raising a `NameError` that the bare `except` swallowed. Labels now resolve through a dict; an unresolvable id is reported, never raised into a silent handler. Latent today (MSIH only), would have bitten in Phase 5.

---

## 7. Task 1.5 — Stale scripts retired

Moved to `slide_classification/misc/` with headers stating what supersedes each:

| Script | Why retired |
|---|---|
| `external_validation_script.py` | Hardcodes ANN `hidden_dim=256, hidden_dim2=64`; saved checkpoints are `256/128`, so it would fail with a shape mismatch. It cannot have produced the published PAIP-EV numbers — `Slide_Classification.ipynb` cell 14 did. |
| `dataloader.py` | Imported nowhere, stale Windows paths, derives labels from filename substrings rather than the label CSV. |
| `plot_surgen_ttc_results.py` | Reads a remote Linux path that does not exist locally, and expects filenames from a different script than the one that produced local `SurGen_Results/`. |

**Acceptance:** no active code path imports any of the three — every remaining reference in the repo is a commented-out line.

---

## 8. Deviations, and what does *not* yet pass

Recorded honestly rather than quietly claimed.

1. **§1c.2 repo-wide hardcoded-path grep does NOT pass.** 34 files carry `D:\Aamir Gulzar\...` or `/media/dp-psau/...`. The 11 in `slide_classification/` are handled (swept, retired, or superseded in Task 2.1/2.2). The **23 upstream** feature-extraction/aggregation files (`Complete_Pipeline/`, `slide_aggregation/`, `data_preprocessing/`) are deferred — they are not re-run by Phases 1–4, and converting untested code that nothing in this run exercises adds risk without benefit. **Trigger for revisiting:** the moment SurGen features for H-Optimus-1 / UNI2 / ConchV1 need generating on the server (anticipated by Task 3.2). See `OPEN_QUESTIONS.md` #4.

2. **CIMP is not binary.** `kfold_CIMP.csv` has four levels (`Non-CIMP` 182, `CRC CIMP-L` 178, `CIMP-H` 54, `GEA CIMP-L` 2). `load_label_map` raises `NotImplementedError` rather than guessing a dichotomisation. Blocks nothing today (Phase 5 deferred), but **needs an owner decision** before CIMP is enabled. See `OPEN_QUESTIONS.md` #5.

3. **Pre-existing, untouched:** `Slide_Classification.ipynb` cell 10 (an ANN/linear hyperparameter diagnostic) has its entire body indented by four spaces and does not parse. Confirmed present at commit `4240546`, before any Phase 1 change — not a regression. Cosmetic; the cell is a scratch diagnostic, not part of any result path.

4. **Results are never overwritten in place.** An early smoke test of the new runner did overwrite two tracked result files in `TCGA_Results/`; they were restored from git immediately and `archive_experiment_tree()` was added, which renames any existing results tree to `<name>_ARCHIVED_<date>` before the first write. Verified working. No pre-correction result file has been lost.

---

## 9. What Phase 2 inherits

- Task 2.1 — implement `runners/surgen_folds.py` (case-level folds, ~554 cases) and retire the three SurGen CV scripts, which resolves their remaining Task 1.1 selection sites and hardcoded paths at the same time.
- Task 2.2 — retire PAIP-CV: archive `PAIP_Results/`, remove the `WSIDataset` + k-fold block from `Slide_Classification.ipynb` cell 8, leave a markdown cell recording the reason.
- Task 2.3 — rename `paip_kfolds_71.csv` → `paip_78_labels.csv` (`data_layer` already prefers the new name and falls back to the old one), and regenerate the malformed `paip_78slides_labels.csv` with a clean `split` column.

## 10. New infrastructure available to later phases

- `thresholds_TCGA.json` + `runners/thresholds.py` — τ_TCGA by Youden's J on pooled out-of-fold TCGA-CV probabilities (§1b), with a guard that each slide appears exactly once.
- `runners/runlog.py` — `logs/run_<timestamp>.log`, resumable `progress.json`, `skipped_combinations.csv`, coverage reporting.
- `runners/results_io.py` — one provenance-stamped JSON per (experiment, method, model, variant), so cross-machine merging is a file copy plus `merge_results.py`.
- `data_layer.assert_no_leakage(...)` — accepts a group key, ready for SurGen case-level checking in Task 2.1.
- `data_layer.stratified_holdout(...)` — ready for the ANN's 15% early-stopping carve-out in Task 3.0a.
