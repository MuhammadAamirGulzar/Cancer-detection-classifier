# Phase 2 Report — Validation protocols

**Date:** 2026-08-04
**Branch:** `dev`
**Commit:** `58c073e`

Phase 2 fixes *what is being measured*. Phase 1 fixed how it was computed; Phase 2 fixes the experimental design underneath it.

---

## 1. Summary

| Task | Status | Effect on numbers |
|---|---|---|
| 2.1 SurGen folds rebuilt at case level | Done | **SurGen-CV drops**: mean BalAcc −0.0252, AUROC −0.0342 |
| 2.2 PAIP-CV retired | Done | PAIP-CV removed from the study entirely |
| 2.3 PAIP label files renamed / regenerated | Done | None (metadata correctness) |

---

## 2. Task 2.1 — SurGen folds were leaking patients

### The bug

`build_runtime_folds` shuffled **slide IDs** and dealt them round-robin. But 70 of SurGen's 554 labelled cases contribute **two slides each**, so sibling slides landed in different folds — the same patient in train and test. Two sections from one tumour are far more alike than two different tumours, so this inflated every SurGen number.

TCGA is immune: 416 rows, 416 unique `Case_ID` — exactly one slide per patient. That is why the same code was safe there and was carried over unchanged.

### Case-ID derivation, verified against the real IDs

The regex strips a trailing numeric section suffix: `SR1482_40X_HE_T004_01` → `SR1482_40X_HE_T004`. Verified rather than assumed:

| Property | Value | Work order expectation |
|---|---|---|
| Usable slides (label 0/1) | 624 | 624 ✓ |
| **Unique cases** | **554** | 554 ✓ |
| Cases with 1 slide | 484 | — |
| Cases with 2 slides | **70** | 70 ✓ |
| Cases with conflicting labels | **0** | "no case has conflicting labels" ✓ |
| Case-level label balance | 505 neg / 49 pos | — |

The function returns the id unchanged when there is no numeric suffix, so an unexpected naming scheme degrades to one-case-per-slide rather than silently merging unrelated slides.

### Leakage actually present in the old scheme

```
Leakage under the OLD slide-level scheme:
  n_multi_slide_cases: 70
  n_split_across_folds: 47        (67.1%)
  leaked_slides: 94
  examples: {'SR1482_40X_HE_T236': [3, 4], 'SR1482_40X_HE_T195': [1, 2],
             'SR1482_40X_HE_T029': [1, 4], 'SR1482_40X_HE_T061': [1, 4]}

Leakage under the NEW case-level scheme:
  n_split_across_folds: 0         (0.0%)
```

(Restricted to the 622 slides that actually have features it is 49/70 = 70.0%. Either way: roughly two cases in three were split, matching the work order's "roughly 3 times in 4" estimate.)

### Acceptance — fold table

```
  SurGen case-level folds (seed=42, K=4)
  fold    slides   cases  MSI-H slides  MSI-H cases
  1          156     140            16           13
  2          151     138            14           12
  3          163     138            15           12
  4          154     138            15           12
  TOTAL      624     554            60           49
  70 case(s) contribute >1 slide; all kept within a single fold.
```

**Per-fold case counts sum to 554** ✓ and the no-case-spans-folds assertion passes ✓ — both Task 2.1 acceptance criteria.

Restricted to the 622 slides that have features, it is 552 cases (156/157/152/153 slides across 139/138/138/137 cases). The 2 missing SurGen slides — `SR386_40X_HE_T086_01`, `SR386_40X_HE_T339_01` — are each the only slide of their case, so removing them removes 2 cases.

### Before / after — the cost of the leak

Both arms run through **identical corrected code** (the Phase 1 fixes are present in both); fold construction is the only variable. 3 (method, model) combinations × 5 classifiers × 4 folds.

| Combination | | lin | knn | proto | rf | ann |
|---|---|---|---|---|---|---|
| **Tissue_Type_Clustering / Conch1_5** | BalAcc old → new | 0.6286 → 0.5835 | 0.5179 → 0.5091 | 0.5795 → 0.5711 | 0.5214 → 0.5026 | 0.7091 → 0.6502 |
| | AUROC old → new | 0.8040 → 0.7666 | 0.6571 → 0.5645 | 0.6604 → 0.6582 | 0.7636 → 0.7329 | 0.7727 → 0.7253 |
| **Averaging / Virchow2** | BalAcc old → new | 0.6638 → 0.6318 | 0.5604 → 0.5169 | 0.6196 → 0.6055 | 0.5420 → 0.4920 | 0.7262 → 0.6620 |
| | AUROC old → new | 0.8624 → 0.8449 | 0.6236 → 0.5551 | 0.6381 → 0.6276 | 0.7656 → 0.7160 | 0.7909 → 0.6909 |
| **Caption_based_aggregation / Conch1_5** | BalAcc old → new | 0.6155 → 0.6153 | 0.5265 → 0.5191 | 0.6033 → 0.5969 | 0.5122 → 0.5086 | 0.6748 → 0.6581 |
| | AUROC old → new | 0.8110 → 0.7817 | 0.5933 → 0.6018 | 0.6781 → 0.6651 | 0.7861 → 0.7597 | 0.7343 → 0.7377 |

**Mean delta (new − old) across all three combinations:**

| Classifier | BalAcc | AUROC | macro-F1 | Acc |
|---|---|---|---|---|
| lin | −0.0258 | −0.0281 | −0.0381 | −0.0126 |
| knn | −0.0199 | −0.0509 | −0.0329 | −0.0043 |
| proto | −0.0096 | −0.0086 | −0.0021 | +0.0015 |
| rf | −0.0242 | −0.0356 | −0.0411 | −0.0069 |
| ann | −0.0466 | −0.0480 | −0.0100 | +0.0098 |
| **ALL** | **−0.0252** | **−0.0342** | **−0.0248** | **−0.0025** |

**All 15 (combination × classifier) balanced-accuracy deltas are negative.** That consistency is what makes this a correction rather than noise: a random fold reshuffle would move numbers in both directions.

**This is the correction working, not a regression.** Every previously published SurGen-CV number is inflated by roughly this much. The ANN is hit hardest (BalAcc −0.0466, AUROC −0.0480), which is consistent with a flexible model exploiting near-duplicate training examples most effectively.

### Files retired

`windows_slide_classification_surgen.py`, `linux_slide_classification_surgen.py`, `slide_classification_surgen.py` → `misc/`, each with a header documenting all three of its defects (leakage, test-set ANN selection, own `WSIDataset` + hardcoded roots). Retiring them also cleared their outstanding Task 1.1 and §1c.2 violations.

`slide_classification_surgen_best_k_labels_exp.py` stays active (it is the SurGen best-k side experiment), but its `build_runtime_folds` now delegates to the case-level builder. Its old version was worse than the CV scripts': it shuffled all slide ids with **no label stratification at all**. Any SurGen best-k result produced before today is leak-inflated and must be regenerated.

---

## 3. Task 2.2 — PAIP-CV retired

**Owner decision, confirmed in the work order.** Only PAIP-IV and PAIP-EV are reported for PAIP.

### Why — recorded because a reviewer comparing against the previous result set will ask

**The fold construction was broken.** Folds were built by *sequential slicing of CSV order* — no shuffle, no stratification. The CSV is ordered `training_data_01 … training_data_47, validation_data_01 …`, so the folds tracked PAIP's own acquisition split:

| Fold | n | MSI-H | from `training_data_*` | from `validation_data_*` |
|---|---|---|---|---|
| 1 | 18 | 4 | 18 | 0 |
| 2 | 18 | 4 | 18 | 0 |
| 3 | 18 | 5 | 6 | 12 |
| 4 | 19 | 4 | 0 | 19 |

Folds 1 and 2 are entirely the provider's training set; fold 4 is entirely its validation set. Class balance came out even **by luck of ordering**, not by design.

**But the deciding argument is that the experiment is strictly dominated.** Even repaired, PAIP-CV would train on ~36 slides and test on 18 with ~4 positives. PAIP-IV trains on **42** and tests on **31**. PAIP-CV trains on less data and tests on smaller sets while answering the same question. An AUROC on 14 negatives and 4 positives cannot separate models — the per-fold confusion matrices already in `best_of_all_exps_metric.xlsx` (e.g. `[[14, 0], [1, 3]]`) show this directly.

### Actions taken

- `PAIP_Results/` → `PAIP_Results_ARCHIVED_20260804/` via `git mv`. **All 484 files intact.** Not deleted — those numbers may already have been circulated.
- `Slide_Classification.ipynb` cell 8 (20,680 characters: its own `WSIDataset`, the sequential fold builder, the CV loop) replaced by a markdown cell carrying the full retirement rationale, plus a stub that raises `NotImplementedError` pointing at the PAIP-IV / PAIP-EV runners.
- **Zero `WSIDataset` definitions now remain in `Slide_Classification.ipynb`.**

**Acceptance:** no active code path builds PAIP folds ✓; `PAIP_Results_ARCHIVED_20260804/` exists and is intact ✓.

### Uncertainty on PAIP

Task 3.1 will report **bootstrap confidence intervals** over the 31 PAIP-IV test slides (1000 resamples, 95% CI on BalAcc and AUROC). That is the correct tool at this sample size — not K-fold.

---

## 4. Task 2.3 — PAIP label files

### Rename

`slide_classification/paip_kfolds_71.csv` held **78 rows** and **no fold column**. It is a label table whose name had it being read as a fold table. Renamed to **`paip_78_labels.csv`**.

A byte-identical duplicate (`md5 380793811c32`) existed at `paip_data/labels/paip_kfolds_71.csv`; renamed to match. All **10 references across 6 notebooks/scripts** updated. No active reference to the old name remains.

### Regeneration

`paip_data/labels/paip_78slides_labels.csv` was **malformed**: its `fold` column mixed fold indices with label strings — `{'nonMSIH': 24, '1': 10, '0': 10, '2': 9, '3': 9, '4': 9, 'MSIH': 7}`. The 31 `validation_data_*` rows carry only three fields, so their label landed in `fold` and `label` was `NaN`.

Regenerated by `tools/regen_paip_labels.py` from the two provider files, with an explicit **`split`** column (`train`/`test`) rather than an overloaded `fold` — PAIP has no folds, it has a provider split. The original is preserved as `paip_78slides_labels_MALFORMED_ARCHIVED_20260804.csv`.

```
checks:
  78 rows                          : 78  OK
  split is 47 train / 31 test      : {'train': 47, 'test': 31}  OK
  19 MSI-H / 59 non-MSI-H          : {'nonMSIH': 59, 'MSIH': 19}  OK
  id suffix agrees with label table: OK
  no duplicate ids                 : 0  OK
  covers every id in label table   : OK

per-split class balance:
  test   MSIH        7      train  MSIH       12
         nonMSIH    24             nonMSIH    35
```

All Task 2.3 acceptance criteria met. The counts reconcile with the work order: of the 47 train ids, 42 have features (10 MSI-H, since `training_data_30_MSIH` and `training_data_42_MSIH` are among the 5 without features); all 31 test ids have features (7 MSI-H).

---

## 5. Findings outside the work order's scope

1. **`Taiga_Paip_inference.ipynb` is malformed JSON** — a stray `}` closes the notebook object at line 51 and the cell array continues after it. **Pre-existing**: confirmed byte-for-byte at commit `4240546`, before any remediation. It cannot be opened by `nbformat`/Jupyter as-is. Handled here by plain-text replacement for the Task 2.3 rename. It is a PAIP inference notebook superseded by the Phase 3 PAIP-EV runner, so it is not on the critical path — but it should be repaired or retired rather than left in this state.

2. **A transient segfault (exit 139)** occurred once while running all 5 classifiers × 3 combinations × 2 fold schemes in a single process. Not reproducible: the identical command succeeded on re-run, and each classifier passes individually. Most likely repeated `joblib` worker-pool creation (`n_jobs=-1` in `GridSearchCV` and `RandomForestClassifier`) alongside a CUDA context on Windows. **Relevant to Phase 3**, which runs long unattended sweeps — the resumable `progress.json` already means a crash costs only the in-flight combination, but if it recurs, bounding `n_jobs` or isolating combinations in subprocesses is the fix.

---

## 6. Protocol facts to state in Methods

Unchanged by Phase 2, but now enforced in one place (`data_layer.cv_rotation`) and worth restating:

- The rotation is test = fold *i*, val = fold *i+1*, train = remaining two → **training uses 50% of the cohort**, not the 75% a reader assumes from "4-fold CV". Owner-confirmed as deliberate.
- `combine_trainval=True` for `lin`/`knn`/`proto`/`rf` merges the validation fold back into training; only the ANN uses it as a true early-stopping set. "Validation fold" means two different things depending on the classifier.
- Folds are 1–4 in the CSVs and reported `Fold1..Fold4`, but checkpoints are saved `fold0..fold3`.

---

## 7. What Phase 3 inherits

- `runners/surgen_folds.py` — case-level folds ready for SurGen-CV and SurGen-EV.
- `runners/thresholds.py` — τ_TCGA machinery, awaiting a full TCGA-CV run to populate `thresholds_TCGA.json`.
- `data_layer.stratified_holdout()` — for the ANN's 15% early-stopping carve-out in Task 3.0a.
- Still to build: `runners/iv_runner.py` (Task 3.1, PAIP-IV + bootstrap CIs) and `runners/ev_runner.py` (Tasks 3.0/3.2, the three EV variants).
- **All results produced before today are superseded.** TCGA-CV and PAIP-EV by the Phase 1 corrections; SurGen-CV additionally by Task 2.1; PAIP-CV is withdrawn outright.
