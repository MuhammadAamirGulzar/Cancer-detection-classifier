# MSI-H Classification of Colorectal Cancer from Whole-Slide Images

End-to-end pipeline for predicting **microsatellite-instability-high (MSI-H) vs. non-MSI-H** colorectal cancer from H&E whole-slide images (WSIs). It runs from raw slides, through patch extraction and foundation-model embeddings, to slide-level aggregation, five classifier heads, and multi-cohort evaluation. Models are trained on TCGA and validated on two independent cohorts, PAIP and SurGen.

The project compares **how patch embeddings are aggregated into a slide representation** (the main experimental variable) across several pathology foundation models. It also audits the evaluation itself: label provenance, cross-validation leakage, and decision-threshold transfer under domain shift.

> **Start here:** [reports/PROJECT_STATUS.md](reports/PROJECT_STATUS.md) says where the current results live, what is finished and what is left. [reports/EXPERIMENT_LEDGER.md](reports/EXPERIMENT_LEDGER.md) lists every experiment and hyperparameter already tried, with outcomes, so nothing gets repeated.

---

## Highlights

- **Six foundation-model encoders** (CONCH v1, CONCH 1.5, H-Optimus-1, UNI2-h, Virchow2, PRISM) × **four patch-aggregation strategies** (plus the TITAN slide encoder as a baseline) × **five classifier heads** (logistic regression, ANN, kNN, ProtoNet, random forest). The result is one 4-fold CV table plus three external tables.
- **TCGA 4-fold CV (413 slides): best AUROC 0.928** (H-Optimus-1, 15-class caption aggregation, logistic regression). Best balanced accuracy is 0.828 (same encoder/aggregation, ANN).
- **PAIP external validation (TCGA → 78 slides): best AUROC 0.915** (UNI2, averaging, logistic regression). Best balanced accuracy is 0.870 (H-Optimus-1, averaging, ProtoNet).
- **SurGen external validation (TCGA → 622 slides): best AUROC 0.856** (UNI2, 15-class caption aggregation, logistic regression). This was run on the earlier 624-slide label set (see [Status](#11-status-and-limitations)).
- **Evaluation bugs found and fixed without moving the baseline.** kNN scores had collapsed to 2–4 distinct values, and RF/ProtoNet thresholds did not transfer across cohorts. The fixes are layered on as opt-in threshold modes, and the frozen headline metrics are checked to be bit-identical (see [Data integrity](#6-data-integrity-work)).
- **SurGen label set rebuilt from 624 to 991 slides (822 cases, 100 MSI-H)** by reconciling MMR-IHC and PCR-MSI evidence across the SR386 and SR1482 sub-cohorts. A single master CSV is the source of truth, with a verification script that asserts the counts.
- **Leakage fix.** Case-level folds for SurGen (70 two-slide cases in the earlier 622-slide set) removed patient-level leakage that had inflated earlier SurGen-CV numbers.

---

## 1. Cohorts

| Cohort | Role | Slides (label file → modelled) | Cases | MSI-H |
|---|---|---|---|---|
| **TCGA** | Training; 4-fold CV | 416 → **413** | 413 (1 slide per case) | 60 / 413 (**14.5%**) |
| **PAIP** | External validation (EV); provider train/test split (IV) | 78 → **78** | 78 | 19 / 78 (**24.4%**); split 12/47 train, 7/31 test |
| **SurGen** (earlier label set) | External validation; case-grouped CV | 624 → **622** | 552 | 60 / 622 (**9.6%**) |
| **SurGen** (current label set) | Relabelled; **encoding in progress** | 1020 → **991** labelled | **822** | 100 / 991 (**10.1%**) |

Notes:

- TCGA: three labelled slides have no extracted features and are excluded. Four featured slides have no label.
- SurGen sub-cohorts (current label set): SR386 has 425 slides / 425 cases / 32 MSI-H. SR1482 has 566 slides / 397 cases / 68 MSI-H.
- All SurGen results tables in this README were computed on the **622-slide, 60-MSI-H** set. The 991-slide relabel is finished, but the extra slides are not yet patched or encoded (see Status).
- Patch counts:
  - TCGA: 1,377,933 filtered patches across 417 slides (zero-shot patch statistics table).
  - SurGen (622 modelled slides): 11,870,098 tissue patches.
  - PAIP: 275,541 patches across 78 slides (row count of the 14-class caption table, `classification_results_paip.csv`).
  - Combined total: about 13.5 million patches.

---

## 2. Pipeline architecture

```mermaid
flowchart LR
    A["Whole-slide images<br/>TCGA .svs · PAIP .svs · SurGen .czi"] --> B["Tiling<br/>(512 px @ 20x for SurGen)"]
    B --> C["Background and tissue filtering<br/>pixel rules + SVM whiteness classifier"]
    C --> D["Patch embeddings<br/>CONCH v1 / 1.5 · H-Optimus-1 · UNI2-h · Virchow2"]
    D --> E1["Averaging"]
    D --> E2["Caption-based aggregation<br/>CONCH zero-shot, 14 / 15 classes"]
    D --> E3["Tissue-type clustering<br/>9-class CRC-100K classifier"]
    A --> S["Slide encoders<br/>PRISM · TITAN"]
    E1 --> F["Slide representation"]
    E2 --> F
    E3 --> F
    S --> F
    F --> G["Classifier heads<br/>LR · ANN · kNN · ProtoNet · RF"]
    G --> H["Evaluation<br/>TCGA 4-fold CV · PAIP-IV · PAIP-EV · SurGen-CV · SurGen-EV"]
    H --> I["Threshold handling<br/>frozen · corrected · promoted"]
    I --> J["Reporting<br/>xlsx · RESULTS tables · radar plots"]
```

| Stage | Where |
|---|---|
| Tiling, tissue detection, embedding (SurGen, incremental) | [Complete_Pipeline/](Complete_Pipeline/) |
| Slide metadata, patch creation (TCGA/PAIP) | [data_preprocessing/](data_preprocessing/) |
| Background/white-patch SVMs | [data_cleaning/](data_cleaning/) |
| Embedding code | [feature_extraction/](feature_extraction/) |
| Tissue-type classifier (CRC-100K) | [TissueClassifier_CRC100K/](TissueClassifier_CRC100K/) |
| Slide aggregation | [slide_aggregation/](slide_aggregation/) |
| Classifiers, CV/IV/EV runners, threshold modes | [slide_classification/](slide_classification/) |
| Report and plot builders | [tools/](tools/) |

---

## 3. Methods

### Patch extraction and tissue filtering

- **SurGen** ([Complete_Pipeline/](Complete_Pipeline/)): `PATCH_SIZE = 512` at `TARGET_MAG = 20`.
  - Step A-1 computes per-patch RGB statistics.
  - Pixel rules reject near-black patches (black-pixel ratio ≥ 0.20) and near-white patches (white-pixel ratio ≥ 0.90, or mean ≥ 210 with std ≤ 12).
  - Step A-2 passes the remaining patches through an SVM whiteness classifier on `avg_G`, `avg_B` and `std_R`.
  - Step C runs the encoders.
- **TCGA/PAIP** ([data_preprocessing/Patch_creation.ipynb](data_preprocessing/Patch_creation.ipynb)): the notebook switches patch size by magnification (1024 for 40x, otherwise 512). Separate SVMs are trained for TCGA and PAIP background removal ([data_cleaning/](data_cleaning/)).

### Encoders and embedding dimensions

Values are from `MODEL_BASE_DIMS` in [slide_classification/config/paths.py](slide_classification/config/paths.py).

| Encoder | Dim |
|---|---|
| CONCH v1 | 512 |
| CONCH 1.5 | 768 |
| H-Optimus-1 | 1536 |
| UNI2-h | 1536 |
| Virchow2 | 2560 |
| PRISM (slide-level) | 1280 |

TITAN (slide-level, built on CONCH 1.5 patch features) is evaluated as a baseline with a single CONCH 1.5 configuration.

### Slide aggregation

The classifier input is the flattened (rows × dim) matrix (`AGG_MULTIPLIERS` in the same file).

| Strategy | Rows per slide | Idea |
|---|---|---|
| Averaging | 1 | Mean of all patch embeddings |
| Caption-based aggregation | 14 | CONCH zero-shot caption/class assignment of patches into 14 classes; one row per class |
| Caption-based aggregation (15 classes) | 15 | Same, with a 15-class set |
| Tissue-type clustering | 9 | Patches assigned to 9 CRC-100K tissue classes by a trained ANN tissue classifier; one row per class |
| TITAN / PRISM | 1 | Slide-level encoders with no patch aggregation, tabulated separately |

### Classifiers

Grids are in [slide_classification/runners/classifiers.py](slide_classification/runners/classifiers.py).

- **LR:** C = 10, max_iter = 300.
- **kNN:** distance metric selected per configuration (cosine or manhattan), k = 3 in the original grid (changed under the corrected threshold modes, below).
- **ProtoNet:** nearest class prototype.
- **RF:** 500 trees, class weight {0: 1, 1: 10}, prediction threshold 0.3.
- **ANN:** grid over hidden_dim1 ∈ {128, 256, 512} × hidden_dim2 ∈ {64, 128, 256} × max_iter ∈ {500, 1000}. Selection uses validation macro-F1, not test metrics, which was one of the audit fixes.
- **Seeds:** 5 (42–46) for the final external-validation models.

### Cross-validation and external-validation setup

- **TCGA-CV:** 4 folds over 413 slides, each slide tested exactly once, threshold 0.5.
- **PAIP-IV:** the provider's fixed split (47 train / 31 test), with bootstrap 95% CIs.
- **SurGen-CV:** 4 folds built at **case level** (grouped, label-stratified).
- **PAIP-EV and SurGen-EV:** train on **TCGA only**, test on the entire external cohort. Zero external data enter training, hyperparameters or thresholds. Three variants are produced:
  - `tcga_full`: the primary result.
  - `fold_ensemble`: a probability ensemble of the four fold models.
  - `fold_average`: the legacy fold-average convention.
- **Hyperparameters for the final TCGA models** are derived from TCGA-CV alone, by a vote across folds (see `tcga_full_hparams.json`).

### Threshold handling

Implemented in [slide_classification/runners/threshold_modes.py](slide_classification/runners/threshold_modes.py).

| Mode | Behaviour |
|---|---|
| `frozen` (default) | τ_TCGA is fitted by Youden's J on pooled TCGA out-of-fold probabilities and applied unchanged. The strict zero-shot number; the default so every earlier result reproduces exactly. |
| `corrected` | kNN is scored at a larger k with τ refitted on TCGA out-of-fold probabilities **at that k**. Other heads (LR, ANN, ProtoNet, RF) get a rate-matched **quantile threshold**: the target is cut at the (1 − r) quantile, where r is the fraction of TCGA out-of-fold slides flagged positive. |
| `promoted` | The corrected scheme applied to kNN and RF only, with its values taking over the headline fields. The displaced frozen values are kept under `*_frozen` in the same record, and every row carries a `Threshold_scheme` label. |

- **kNN k = 35** is justified from TCGA alone: it maximised TCGA out-of-fold balanced accuracy (0.6971), and 35 × 0.145 ≈ 5 expected positive neighbours.
- **SurGen kNN uses k = 20**, taken from a SurGen k-sweep. That choice is target-informed (see Limitations).
- The quantile rule reads target **scores** but never target **labels**.

---

## 4. Data-integrity work

### SurGen label reconciliation (624 → 991 slides)

The SurGen release has 1020 WSIs. Labels come from two kinds of evidence: **MMR immunohistochemistry** and **PCR-based MSI testing**.

1. An earlier label set used 624 slides (all of SR386 plus 197 SR1482 slides). In SR1482, 396 slides were unlabelled.
2. An audit against the published train/validate/test split found that **270 SR1482 cases were dropped** because MSI testing was "not performed" or ambiguous. 267 of them had a usable MMR-IHC result, and 3 had an ambiguous MSI value ([reports/surgen_drop_reason_audit.csv](reports/surgen_drop_reason_audit.csv), [reports/surgen_msi_split_reconciliation.csv](reports/surgen_msi_split_reconciliation.csv)).
3. The relabel resolves each slide from whichever evidence is available and records the source and the raw evidence text per row ([Complete_Pipeline/surgen_slide_labels.csv](Complete_Pipeline/surgen_slide_labels.csv)). In the 991 included slides, 850 resolve from MMR-IHC (845 structured, plus 5 free-text reports, 2 of them equivocal and treated as negative) and 141 from PCR-MSI.
4. 29 slides are excluded: 27 with no informative MMR or MSI result, and 2 with zero tissue.
5. **Result: 991 slides, 822 cases, 100 MSI-H / 891 non-MSI-H.** [Complete_Pipeline/verify_surgen_labels.py](Complete_Pipeline/verify_surgen_labels.py) asserts 991 / 100 / 891 and fails otherwise. [Complete_Pipeline/regenerate_surgen_labels.py](Complete_Pipeline/regenerate_surgen_labels.py) keeps the two-column file used by the classifier in lock-step with the master CSV (expected 100 / 891 / 29 excluded).
6. The pipeline only looks labels up; no MMR/MSI rule is re-derived in code.

Cohort accounting for the earlier 622-slide run is in [reports/COHORT_COUNTS.md](reports/COHORT_COUNTS.md) and [reports/surgen_cohort_audit.csv](reports/surgen_cohort_audit.csv).

### Case-level folds (leakage fix)

In the earlier SurGen label set, 70 cases contribute two slides (primary and metastatic), and the old slide-level folds split about 67% of those pairs across train/test. Folds are now case-grouped (0 leaked cases). Through identical code, mean balanced accuracy fell 0.025 and AUROC fell 0.034 across 3 combinations × 5 classifiers, with all 15 balanced-accuracy deltas negative. Earlier SurGen-CV numbers were therefore inflated ([reports/PHASE_2_REPORT.md](reports/PHASE_2_REPORT.md)).

### Evaluation-bug diagnosis

| Finding | Evidence |
|---|---|
| **kNN rank collapse.** At k = 3 a kNN score takes only 2–4 distinct values, so thresholds land inside tied blocks and AUROC is depressed. The in-domain grid was also capped at k ≤ 15. | Mean AUROC over 20 aggregation combinations at k = 3 vs. k = 35: PAIP 0.611 → 0.811; SurGen 0.589 → 0.653 at k = 35 and 0.677 at k = 20. |
| **Stale τ.** A τ fitted at k = 3 is meaningless on k = 35 scores, so a fixed-τ run left dead (all-negative) or saturated (all-positive) operating points. | Refitting τ per k removed every dead threshold at k > 3 on the 40-configuration check (PAIP at k = 50: 7 dead + 3 near-dead → 0). |
| **RF/ProtoNet threshold mismatch.** The frozen τ_TCGA does not transfer to a shifted score distribution. ProtoNet's τ sits near 0.50 on scores that barely cross it. | SurGen: 16 of 21 ProtoNet and 7 of 21 RF configurations were dead or near-dead (frozen baseline). The quantile threshold recovered 84% (ProtoNet) and 74% (RF) of the gap to the oracle threshold. |
| **Detector blind spot.** The old health check only caught zero predicted positives, not saturation. | Added saturated and near-dead/near-saturated (1% band) detection. |

**Keeping the baseline reproducible.** `frozen` stays the default, and the corrected schemes are opt-in. Diagnostics wrote only derived copies. The published tree was verified byte-identical by SHA-256 before and after, and 16 headline keys × 315 entries showed zero drift ([experiments/surgen_ev_diagnostic/SUMMARY.md](experiments/surgen_ev_diagnostic/SUMMARY.md), [experiments/threshold_and_k/SUMMARY.md](experiments/threshold_and_k/SUMMARY.md); see the note in Status).

Other audit fixes ([reports/](reports/)): ANN hyperparameters are now selected on validation metrics rather than test metrics; a double-softmax in the ANN training graph was removed; unseeded data shuffling that made RF numbers non-reproducible was removed; and missing slides now fail loudly instead of being dropped silently.

---

## 5. Results

All numbers are from [slide_classification/best_of_all_exps_metric.xlsx](slide_classification/best_of_all_exps_metric.xlsx). Aggregation methods only (TITAN/PRISM are separate rows). External-validation figures use the primary `tcga_full` variant. Single-seed CV means; no confidence intervals except PAIP-IV.

### Best result per cohort

| Experiment | N | Best AUROC | Configuration | Best BalAcc | Configuration |
|---|---|---|---|---|---|
| TCGA-CV | 413 | **0.928** | H-Optimus-1 · Caption-15 · LR | **0.828** | H-Optimus-1 · Caption-15 · ANN |
| PAIP-IV (31 test) | 31 | 0.926 | CONCH 1.5 · Caption-15 · kNN | 0.908 | CONCH 1.5 · Caption-15 / TTC · kNN |
| PAIP-EV | 78 | **0.915** | UNI2 · Averaging · LR | **0.870** | H-Optimus-1 · Averaging · ProtoNet |
| SurGen-CV (case-grouped) | 622 | 0.853 | UNI2 · Caption-15 · RF | 0.733 | UNI2 · Caption-15 · ANN |
| SurGen-EV | 622 | **0.856** | UNI2 · Caption-15 · LR | 0.779 | UNI2 · TTC · RF |

PAIP-IV has only 31 test slides (7 MSI-H), so one slide moves balanced accuracy by about 0.07. Its bootstrap CIs are wide, for example [0.72, 1.00] on the best balanced accuracy.

### Best AUROC per encoder (any aggregation × classifier)

| Encoder | TCGA-CV | PAIP-EV | SurGen-EV |
|---|---|---|---|
| H-Optimus-1 | **0.928** | 0.901 | 0.802 |
| UNI2-h | 0.894 | **0.915** | **0.856** |
| Virchow2 | 0.881 | 0.910 | 0.784 |
| CONCH 1.5 | 0.839 | 0.883 | 0.747 |
| CONCH v1 | 0.817 | 0.864 | 0.739 |

### Mean AUROC by classifier head (TCGA-CV, PAIP-EV and SurGen-EV, aggregation methods)

| Head | TCGA-CV | PAIP-EV | SurGen-EV |
|---|---|---|---|
| LR | 0.850 | 0.872 | 0.709 |
| ANN | 0.845 | 0.850 | 0.714 |
| RF | 0.845 | 0.825 | 0.669 |
| ProtoNet | 0.720 | 0.850 | 0.710 |
| kNN | 0.694 | 0.811 | 0.677 |

PAIP-EV and SurGen-EV kNN/RF rows use the corrected scheme (kNN k = 35 on PAIP and k = 20 on SurGen; RF quantile threshold). The other heads use frozen τ_TCGA. The two external cohorts therefore do not use identical kNN/RF settings and are not directly comparable for those two heads.

### Effect of the threshold correction (`tcga_full`, mean over 20 aggregation combinations)

| Cohort | Head | AUROC before → after | BalAcc before → after |
|---|---|---|---|
| PAIP-EV | kNN (k = 3 → 35, refit τ) | 0.611 → 0.811 | 0.596 → 0.703 |
| PAIP-EV | RF (quantile τ) | 0.825 → 0.825 (unchanged by design) | 0.662 → 0.772 |
| SurGen-EV | kNN (k = 3 → 20, refit τ) | 0.589 → 0.677 | 0.566 → 0.575 |
| SurGen-EV | RF (quantile τ) | 0.669 → 0.669 (unchanged by design) | 0.559 → 0.642 |

### Slide-level encoder baselines

PRISM (TCGA-CV): best AUROC 0.869 (ANN). TITAN on CONCH 1.5 features (TCGA-CV): best AUROC 0.877 (RF). Both are below the best aggregated configuration (0.928). See the `*_TITAN_PRISM` sheets of the workbook.

### Figures

Radar plots (spokes are aggregation × classifier; one line per encoder; radius fixed at 0–1). Generated by [tools/make_radar_plots.py](tools/make_radar_plots.py).

| | AUROC | Balanced accuracy |
|---|---|---|
| TCGA-CV | [radar](Analysis_and_Visualization/Radar_Plots/TCGA-CV/TCGA-CV_AUROC_radar.png) | [radar](Analysis_and_Visualization/Radar_Plots/TCGA-CV/TCGA-CV_BalAcc_radar.png) |
| PAIP-EV (kNN k = 35 and RF corrected) | [radar](Analysis_and_Visualization/Radar_Plots/PAIP-EV/PAIP-EV_AUROC_radar.png) | [radar](Analysis_and_Visualization/Radar_Plots/PAIP-EV/PAIP-EV_BalAcc_radar.png) |
| SurGen-EV (kNN k = 20 and RF corrected) | [radar](Analysis_and_Visualization/Radar_Plots/SurGen-EV/SurGen-EV_AUROC_radar.png) | [radar](Analysis_and_Visualization/Radar_Plots/SurGen-EV/SurGen-EV_BalAcc_radar.png) |

These are the figures drawn from the current workbook (3 Sep 2026). The `PAIP-EV_corrected/` and `SurGen-EV_corrected/` folders hold older plots from a superseded run and should not be cited.

![TCGA-CV AUROC radar](Analysis_and_Visualization/Radar_Plots/TCGA-CV/TCGA-CV_AUROC_radar.png)

---

## 6. Repository structure

```text
Complete_Pipeline/        SurGen end-to-end pipeline: download -> tiling -> tissue filter -> embeddings (per-encoder runners),
                          label master CSV (surgen_slide_labels.csv), label verification/regeneration scripts
data_preprocessing/       TCGA/PAIP slide metadata and patch creation notebooks
data_cleaning/            Background / white-patch SVM training (TCGA, PAIP)
feature_extraction/       Encoder wrappers and dataset modules
TissueClassifier_CRC100K/ 9-class tissue classifiers trained on CRC-100K (per-encoder params and metrics)
slide_aggregation/        Averaging, caption-based (CONCH zero-shot), tissue-type clustering, TITAN
slide_classification/     Classifiers, data layer, CV/IV/EV runners, threshold modes, results trees, metric workbook
  config/paths.py         Machine-keyed data roots, encoder dims, aggregation row counts
  runners/                cv_runner, iv_runner, ev_runner, full_trainer, classifiers, thresholds, threshold_modes
tools/                    Report, workbook, radar-plot, t-SNE and audit builders
reports/                  Phase reports (audit and remediation), cohort accounting, SurGen audit CSVs
Analysis_and_Visualization/  Radar plots, zero-shot and t-SNE shift statistics, classifier comparisons
```

---

## 7. Reproducing

**Requirements.** Python 3.10 (conda). [Complete_Pipeline/requirements.txt](Complete_Pipeline/requirements.txt) lists the pinned stack (PyTorch 2.11, timm, scikit-learn 1.8, pandas, pylibCZIrw for SurGen `.czi` slides). A CUDA GPU is needed for encoding. The foundation-model checkpoints are gated on Hugging Face, so set `HF_TOKEN` (via environment or a local `.env`; never committed). Raw slides and checkpoints are **not** included in this repository.

**Data roots.** All data paths come from [slide_classification/config/paths.py](slide_classification/config/paths.py), selected by the `MACHINE` environment variable (`local` by default). Edit the roots there to point at your own copy of the data.

**1. Encode SurGen slides** (Windows/CZI pipeline; supervised, resumable):

```bash
python Complete_Pipeline/verify_surgen_labels.py          # asserts 991 / 100 / 891
python Complete_Pipeline/run_pipeline_supervisor.py       # runs the v6 batch pipeline
```

> **Warning - these runners overwrite or move results.** Read this before running anything on a copy that holds results you care about.
> - Every runner renames the existing results tree to `<name>_ARCHIVED_<date>` before it writes. Those archive folders are git-ignored, so git will not show the move. Pass `--no-archive` to leave the tree in place.
> - `cv_runner.py --experiment TCGA-CV --force` retrains TCGA and rewrites `thresholds_TCGA.json`, which every external-validation number depends on.
> - SurGen runs skip any encoder whose feature folder is missing instead of failing, so check that all five encoder folders exist under `surgen_data/surgen_processed/` first.

**2. Classification and evaluation** (run from `slide_classification/`):

```bash
python runners/cv_runner.py --experiment TCGA-CV          # or SurGen-CV
python runners/iv_runner.py                               # PAIP provider split
python runners/full_trainer.py                            # final TCGA models, 5 seeds
python runners/ev_runner.py --experiment PAIP-EV          # or SurGen-EV
python runners/ev_runner.py --experiment PAIP-EV --threshold-mode promoted   # corrected kNN/RF
```

Common flags: `--methods`, `--models`, `--classifiers`, `--force`. The `ev_runner` also takes `--variants`, `--seeds` and `--knn-k`.

**3. Reporting:**

```bash
python tools/build_report.py             # writes best_of_all_exps_metric.xlsx
python tools/make_radar_plots.py         # radar figures
python tools/write_results_md.py         # RESULTS.md tables
```

Treat the sequence above as a guide: it reflects the entry points that exist in the code, but a full end-to-end re-run has not been verified from a clean checkout [TBD: confirm on a clean machine].

---

## 8. Tech stack

Python · PyTorch · timm · scikit-learn · pandas / NumPy · pylibCZIrw (CZI reading) · OpenCV · Hugging Face Hub (gated model weights) · matplotlib (reporting) · Jupyter.

Foundation models: CONCH / CONCH 1.5, H-Optimus-1, UNI2-h, Virchow2, PRISM, TITAN.

---

## 9. Status and limitations

- **SurGen is only partly encoded under the new labels.** 991 slides are labelled, but features exist for 622 (11,870,098 patches). The remaining 369 slides are not yet patched or encoded, which means **all SurGen results here are on the earlier 622-slide, 60-MSI-H set** and have not been re-run on the 991-slide labels. SurGen results will change when they are.
- **Published-benchmark match is not verified here.** [TBD: cite the published SurGen benchmark and its label/case counts to confirm 991 / 822 matches it.] The only comparison on file is the earlier-label audit against the published split.
- **The 622-slide results remain valid under the new labels.** None of the 622 modelled slides changed label in the relabel (562 non-MSI-H / 60 MSI-H in both files). The relabel only adds 369 slides, 40 of them MSI-H.
- **No confidence intervals** are reported for CV and EV (single run on TCGA-CV; 5 seeds for final EV models), and PAIP-IV's 31-slide test set is very small.
- **Chosen configurations are optimistic.** The "best" rows above are maxima over 100+ configurations per experiment, selected on the same results they are reported from. Use them as a ranking aid, not as an unbiased performance estimate. The unselected mean-by-head tables are the safer summary.
- **SurGen k = 20 for kNN is target-informed.** It was picked from a sweep run on SurGen itself, unlike k = 35, which is derived from TCGA alone. A target-selected k would not be expected to transfer.
- **The quantile threshold uses unlabelled target scores** (like unsupervised domain adaptation), not labels. It is not a strict zero-shot setting.
- **SurGen performance is modest** and the kNN representation is weak on SurGen. Frozen operating points left many configurations dead or near-dead, which is why the corrected schemes exist.
- **TITAN is evaluated on a single encoder** (CONCH 1.5), and PRISM does not appear in the SurGen result tables. PRISM embeddings now exist for all 622 SurGen slides but have not been classified; five of them were built from slides with missing tiles and need regenerating first.
- **CIMP/BRAF/KRAS/TP53 label files exist, but only the MSI-H task is evaluated.**
- **Repository state.** `RESULTS.md` is generated: its tables match the workbook, but some of its hard-coded prose is out of date (it still says PAIP has 73 slides). Radar PNGs are git-ignored by default; the links resolve locally but not on GitHub unless the images are force-added. Superseded snapshots, old reports and test outputs were moved out of the repository in October 2026 (see [reports/PROJECT_STATUS.md](reports/PROJECT_STATUS.md)).
- **Data and compute.** Raw slides (TCGA, PAIP, SurGen), the embeddings and the model weights are not redistributed. Each cohort has its own access terms.
