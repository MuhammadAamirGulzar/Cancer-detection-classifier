# Cohort counts — where every number comes from

**Date:** 2026-08-04

Three questions came up about sample sizes. All three are answered here from the data, with the exact slide IDs, so the numbers can be defended line by line in review.

**Short version:**

| Cohort | Reported here | Why it is not the headline number |
|---|---|---|
| TCGA | **413** | 416 labelled ∩ 417 with features. The pre-remediation pipeline also used 413 — this did not change. |
| PAIP | **73** | 5 of the 78 declared slides were never patched; they do not exist at any pipeline stage. |
| SurGen | **624** (622 modelled) | Our copy is the *complete* 1020-slide dataset. Only 624 slides have a known MMR/MSI status, and MSI is the label. |

---

## 1. TCGA: 413, not 409 or 416

### The arithmetic

| Set | Count |
|---|---|
| Slides in `kfolds_IDARS_fixed.csv` (labelled) | 416 |
| Slides with aggregated features on disk | 417 |
| **Both labelled and featured — what is modelled** | **413** |
| Labelled but no features | 3 |
| Features but no label | 4 |
| Union | 420 |

**Labelled but no features (3)** — excluded because there is nothing to classify:
`TCGA-AD-6895_MSIH`, `TCGA-AD-6899_nonMSIH`, `TCGA-CM-6680_nonMSIH`

**Features but no label (4)** — excluded because there is no ground truth:
`TCGA-D5-7000_nonMSIH`, `TCGA-EI-6917_nonMSIH`, `TCGA-F5-6814_nonMSIH`, `TCGA-T9-A92H_nonMSIH`

### This is not a change introduced by the remediation

The **pre-remediation** results use the same 413, with the identical per-fold split. Read straight off the archived confusion matrices in `TCGA_Results_ARCHIVED_20260804/`:

```
Averaging/Conch1_5      per-fold N = [103, 97, 112, 101]   TOTAL = 413
Averaging/ConchV1       per-fold N = [103, 97, 112, 101]   TOTAL = 413
Averaging/H-Optimus-1   per-fold N = [103, 97, 112, 101]   TOTAL = 413
Averaging/UNI2          per-fold N = [103, 97, 112, 101]   TOTAL = 413
```

The old pipeline *silently* dropped the 3 slides without features and never said so — that silence is precisely what Task 1.4 fixed. It computed on 413 while the surrounding text said 416. **The models did not change; the reported N did.**

### Where 409 might have come from

No file anywhere in the project has 409 rows — every TCGA table is 416 (`kfolds_IDARS_fixed.csv`, `kfold_BRAF/KRAS/TP53/CIMP.csv`, `SR1482_labels.csv`, and the `Analysis_and_Visualization` copy, which is byte-identical to the primary).

The only simple arithmetic that lands on 409 is **413 − 4**, i.e. subtracting the four *unlabelled* slides from the 413. That would be double-counting: those four were never in the 416 to begin with, so they cannot be removed from an intersection that already excludes them.

**I cannot confirm that is the origin, and I am not asserting it.** What can be stated with certainty is that no artifact in this project produces 409, and that both the old and new pipelines computed on 413. If the previous paper quotes 409, the source is outside this codebase — worth locating before the two are cited side by side.

### Fold sizes — two correct tables that count different things

| Fold | n (all 416 labelled) | n (413 modelled) | MSI-H (416) | MSI-H (413) |
|---|---|---|---|---|
| 1 | 103 | 103 | 15 | 15 |
| 2 | 98 | **97** | 13 | 13 |
| 3 | 112 | 112 | 20 | 20 |
| 4 | 103 | **101** | 13 | **12** |
| **Total** | **416** | **413** | **61** | **60** |

`TCGA-CM-6680` is in fold 2; `TCGA-AD-6895_MSIH` and `TCGA-AD-6899` are both in fold 4 — which is why fold 4 loses two slides and the only missing MSI-H case. **Quote the 413 column**; it is what the models saw.

---

## 2. PAIP: 73 of 78

### The 5 that are absent

`training_data_19_nonMSIH` · `training_data_30_MSIH` · `training_data_41_nonMSIH` · `training_data_42_MSIH` · `training_data_46_nonMSIH`

Two are MSI-H, which is why PAIP's positive count is 17 rather than 19.

### They were never patched — this is not an aggregation failure

Traced through every stage of the PAIP tree:

| Stage | Contents | Any of the 5 present? |
|---|---|---|
| `paip_data/patches_metadata/` | **73 entries** | **none** |
| `paip_data/Features/` | 7 model dirs | none |
| `paip_data/slide_h5_files/` | — | none |
| `paip_data/Flat_directory/` | — | none |

They *are* declared in `paip_78slides.csv` and `TrainTest_paip.csv`, and two of them (`training_data_30`, `training_data_42`) appear in `paip_reviewed_slides.csv`. So they were expected, and dropped out at or before **patching** — the source WSIs were either never obtained or failed patch extraction. Nothing downstream could have recovered them.

**Consequence for the provider split:** of the 47 declared training slides, **42** have features (10 MSI-H). All 31 test slides have features (7 MSI-H). PAIP-IV is therefore a **42 / 31** experiment, and the report says so explicitly rather than quoting 47/31.

> If recovering those 5 matters, the fix is upstream: re-obtain the WSIs and re-run patching. It would take PAIP from 73 to 78 and restore 2 MSI-H cases — a ~7% increase in the positive class, which at this cohort size is not negligible.

---

## 3. SurGen: our copy is complete; the label is the limit

This one has the cleanest answer of the three.

### Our data matches the published cohort structure exactly

| Subset | Our slides | Our cases | Published |
|---|---|---|---|
| SR386 | 427 | 427 | 427 WSIs / 427 patients |
| SR1482 | 593 | 416 | 593 WSIs / 416 patients |
| **Total** | **1020** | **843** | **1020 WSIs / 843 patients** |

Every figure matches. **Nothing is missing from our copy of SurGen.**

### The reduction to 624 is entirely about MMR/MSI status availability

| Subset | unknown (−1) | MMR-proficient (0) | MMR-deficient (1) | total | % with known status |
|---|---|---|---|---|---|
| SR386 | 0 | 395 | 32 | 427 | **100%** |
| SR1482 | **396** | 169 | 28 | 593 | **33.2%** |

**All 396 unknowns are in SR1482**, and SR386 is fully annotated. So:

```
624 usable  =  427 (all of SR386)  +  197 (the SR1482 subset with MMR status)
```

This is consistent with how the two subsets are described: SR386 is the clinical/survival cohort with complete annotation, while SR1482 is the metastatic/biomarker cohort of primary tumours alongside metastatic site samples — where MMR status is not available for every slide.

### Why the literature quotes different numbers

The published figures (1020 WSIs, 843 patients, 427, 593) describe the **imaging cohort**. An MSI/MMR classification task can only use slides whose MMR status is known — that is the label. Studies reporting other endpoints will legitimately quote different denominators:

- **survival analysis** → 426–427 cases (SR386, the only subset with 5-year survival)
- **imaging/segmentation** → up to 1020 slides, no label needed
- **MSI/MMR classification (ours)** → **624** slides with known status

None of these contradict each other. They are different questions over the same image set.

### From 624 to the 622 actually modelled

Two slides have a label but no extracted features: `SR386_40X_HE_T086_01` and `SR386_40X_HE_T339_01` (both SR386, both MMR-proficient). Each is the only slide of its case, so the case count drops 554 → 552.

| | slides | cases |
|---|---|---|
| Known MMR status | 624 | 554 |
| **With features — modelled** | **622** | **552** |

### The detail that made case-level folds necessary

| Subset | usable slides | cases | multi-slide cases |
|---|---|---|---|
| SR386 | 427 | 427 | **0** |
| SR1482 | 197 | 127 | **70** |

**Every one of the 70 two-slide cases is in SR1482** — the metastatic subset, which by construction pairs a primary tumour with samples from the same patient's metastatic sites. SR386 is strictly one slide per patient.

This is exactly why Task 2.1 mattered: two slides from one patient (a primary and its liver metastasis) are far more alike than two different patients, and the old slide-level fold assignment split ~67% of those pairs across the train/test boundary. It also explains why TCGA was immune — 416 slides from 416 patients, one each.

---

## 4. What to state in the paper

> TCGA: 416 slides carry MSI labels; 413 have both a label and extracted features and are used throughout. Three labelled slides (`TCGA-AD-6895`, `TCGA-AD-6899`, `TCGA-CM-6680`) lack features and are excluded, as are four featured slides with no label.
>
> PAIP: 78 slides carry labels; 73 have features. Five training-split slides were not available at patch extraction. The provider's split is therefore evaluated as 42 training / 31 test rather than 47 / 31.
>
> SurGen: the full cohort is 1020 WSIs from 843 patients across two subsets (SR386, 427 slides / 427 patients; SR1482, 593 slides / 416 patients). MMR status is available for 624 slides — all of SR386 and 197 of SR1482 — of which 622 have extracted features and are used. Folds are constructed at case level because 70 cases, all within SR1482, contribute two slides each.

Every count above is reproducible from `slide_classification/kfolds_IDARS_fixed.csv`, `paip_data/labels/paip_78_labels.csv` and `slide_classification/surgen_labels.csv` together with the feature directories, and is re-derived on every run by `data_layer.load_cohort`, which names the excluded slides in its output and writes `missing_slides.csv` beside each result.
