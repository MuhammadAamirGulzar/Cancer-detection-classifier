# Project status - MSI-H classification from colorectal whole-slide images

Last updated: 2026-10-05, after a full read-only audit (2026-10-04), a repository cleanup, and the regeneration of the external-validation and PAIP-IV results under one threshold rule.

This page answers three questions: where are the current results, can they be trusted, and what is left to do.
For the history of what has already been tried, see [EXPERIMENT_LEDGER.md](EXPERIMENT_LEDGER.md).

---

## 1. Where the current results are

Everything lives under `slide_classification/`.

| Experiment | Results tree | Slides (MSI-H) | Last run | Contents |
|---|---|---|---|---|
| TCGA-CV | `TCGA_Results/` | 413 (60) | 2026-08-04 | 22 combinations, 4 folds, seed 42 |
| PAIP-IV | `PAIP_IV_Results/` | 47 train / 31 test | 2026-10-05 | 22 combinations, provider split, bootstrap CIs; ANN is `ann_old` |
| PAIP-EV | `TCGA_PAIP_EV_Results/` | 78 (19) | 2026-10-05 | 22 combinations x 3 variants; corrected threshold for all five heads, kNN k = 35 |
| SurGen-CV | `SurGen_Results/` | 622 (60) | 2026-09-03 | 21 combinations, case-level folds, no PRISM |
| SurGen-EV | `SurGen_EV_Results/` | 622 (60) | 2026-10-05 | 21 combinations x 3 variants; same scheme as PAIP-EV |
| TCGA models for EV | `TCGA_FULL_Models/` | 413 | 2026-08-04 | 22 combinations x 5 heads x seeds 42-46 |

- The consolidated workbook is `slide_classification/best_of_all_exps_metric.xlsx`, rebuilt on 2026-10-05 together with `RESULTS.md` and the radar figures.
- Confidence intervals and paired tests are in `experiments/final_analysis/` ([summary](../experiments/final_analysis/SUMMARY.md)).
- `experiments/final_analysis/validate_against_published.py` checks that the analysis and the published external trees agree. It passes for all 215 classifier-combinations.
- The three trees replaced on 2026-10-05 are in the archive under `tier2/slide_classification/*_ARCHIVED_20261005/`; the files git tracks are also at commit `7c66cc72`.

Files the results depend on, which must not be edited or regenerated casually:

- `thresholds_TCGA.json` - the frozen decision thresholds every external-validation number starts from.
- `knn_k35_thresholds.json`, `tcga_full_hparams.json`.
- The label files `kfolds_IDARS_fixed.csv`, `paip_78_labels.csv`, `surgen_labels.csv`.
- The trained models inside each tree (`models/` folders, `TCGA_FULL_Models/`, `final_models/`). They are git-ignored, so they exist only on this disk.
- `cache/` - assembled feature matrices. It is treated as regenerable, but for two weeks it was the only usable copy of the SurGen CONCH 1.5 features (section 6).

## 2. Headline numbers

**Primary external result: the configuration TCGA cross-validation selects** (H-Optimus-1 / Caption-15 / LR, TCGA-CV AUROC 0.928 [0.887, 0.960]).

| Cohort | AUROC [95% CI] | Balanced accuracy [95% CI] | Rank among 100 | Compared with the best configuration |
|---|---|---|---|---|
| PAIP-EV | 0.879 [0.733, 0.984] | 0.799 [0.685, 0.904] | 19 | not distinguishable (gap 0.037 [-0.038, +0.125]) |
| SurGen-EV | 0.740 [0.652, 0.821] | 0.688 [0.621, 0.758] | 32 | clearly lower (gap 0.116 [+0.050, +0.184]) |

**Best of 100 configurations** (optimistic: chosen on the data they are reported on).

| Experiment | Best AUROC [95% CI] | Configuration |
|---|---|---|
| TCGA-CV | 0.928 [0.887, 0.960] | H-Optimus-1 / Caption-15 / LR |
| PAIP-IV | 0.926 [0.774, 1.000] | CONCH 1.5 / Caption-15 / kNN |
| PAIP-EV | 0.915 [0.829, 0.974] | UNI2 / Averaging / LR |
| SurGen-CV | 0.853 [0.799, 0.896] | UNI2 / Caption-15 / RF |
| SurGen-EV | 0.856 [0.800, 0.905] | UNI2 / Caption-15 / LR |

**Main finding.** Mean AUROC over 5 encoders x 5 classifier heads, and the gain of the three semantic aggregations over averaging:

| Experiment | Averaging | Caption-14 | Caption-15 | Tissue-type clustering | Semantic - Averaging [95% CI] | p |
|---|---|---|---|---|---|---|
| TCGA-CV | 0.775 | 0.802 | 0.792 | 0.794 | +0.016 [-0.010, +0.042] | 0.22 |
| PAIP-IV | 0.760 | 0.853 | 0.837 | 0.845 | +0.085 [+0.018, +0.169] | 0.007 |
| PAIP-EV | 0.819 | 0.861 | 0.869 | 0.838 | +0.037 [-0.001, +0.074] | 0.063 |
| SurGen-CV | 0.665 | 0.711 | 0.718 | 0.686 | +0.040 [+0.019, +0.063] | 0.001 |
| SurGen-EV | 0.662 | 0.720 | 0.730 | 0.697 | +0.054 [+0.035, +0.072] | < 0.001 |

Semantic aggregation beats averaging on the external cohorts (on PAIP-EV, Caption-15 alone: +0.050, p = 0.016) but not within TCGA cross-validation. The supportable claim is that it generalises better, not that it fits better.

**SurGen labels.** All SurGen results use the earlier 622-slide set. None of those 622 slides changed label in the 991-slide relabel, so the results stay valid. The relabel adds 369 slides (40 MSI-H); none is encoded yet.

## 3. Weak points

Fixed on 2026-10-05:

- **One threshold rule.** Every head on both external cohorts now uses the corrected scheme, with kNN at k = 35 on both. Before, only kNN and RF were corrected and SurGen kNN ran at a k chosen on SurGen. Dead or saturated operating points went from 38 of 200 to none.
- **Confidence intervals and paired tests** exist for every configuration and for the claims the paper makes.
- **The README leads with the TCGA-selected configuration** and labels the best-of-100 rows as optimistic.
- **PAIP-IV ANN** is the `ann_old` configuration the owner had decided on; the new weights are byte-identical to the frozen ones in `final_models/ann_old/`.
- **One estimand per row.** A corrected row's AUROC and balanced accuracy now describe the same score (the five-seed mean probability, or the k = 35 neighbour fraction).
- **Provenance.** New result files record whether the code was committed and which library versions ran. The external trees were produced from committed code (`e9b92471`) with scikit-learn 1.7.0, the version the TCGA models were trained with.

Still open, ranked by how much a reviewer would care:

1. **The main effect is absent within TCGA** (section 2). Any text that says aggregation "improves MSI prediction" without qualification overstates it.
2. **TITAN beats the aggregation methods on equal features.** On the same CONCH 1.5 patch features TITAN is better than all four aggregation methods by 0.06-0.09 mean AUROC in TCGA-CV, SurGen-CV and SurGen-EV.
3. **The best encoder changes with the cohort**: H-Optimus-1 within TCGA, UNI2 on SurGen. The TCGA-selected configuration therefore transfers poorly to SurGen (0.740).
4. **No learned-pooling baseline.** Attention-based MIL has never been trained on these features.
5. **The quantile threshold is transductive**: it reads the unlabelled score distribution of the target cohort.
6. **PRISM on SurGen cannot be used as is.** The embedding file for `SR386_40X_HE_T237_01` is corrupt, slide `SR386_40X_HE_T241_01` is missing 52% of its tiles, and four more slides miss 6-9 tiles each. Re-encode at least the first two.
7. Smaller items: cross-validation is a single seed; the random forest predicts at 0.30 in cross-validation; the tissue-class subset experiments (TCGA balanced accuracy 0.85-0.87) were selected on test folds with the pre-fix ANN and were never validated; `reports/COHORT_COUNTS.md` still says PAIP has 73 slides; TCGA-CV, SurGen-CV and PAIP-IV results carry older provenance stamps.

## 4. Remaining work

| # | Task | Status |
|---|---|---|
| 1 | One threshold rule on both external cohorts | **done** (published trees regenerated) |
| 2 | Case-level bootstrap CIs and paired tests | **done** (`experiments/final_analysis/`) |
| 3 | TCGA-selected configuration as the primary external result | **done** (README, this page) |
| 4 | PAIP-IV with the `ann_old` ANN protocol | **done** |
| 5 | PRISM on SurGen | blocked until two slides are re-encoded (weak point 6) |
| 6 | Probability ensembles | tested: the 4-fold ensemble gains 0.015-0.018 AUROC on SurGen for LR and RF and nothing on PAIP; not worth pursuing |
| 7 | Redo the top-k tissue-class aggregation with nested cross-validation and take it to external validation | not started, about 1 day, no GPU |
| 8 | Attention-based MIL baseline on the cached patch features | not started, 2-4 days on one GPU |
| 9 | Finish the 991-slide SurGen encoding and re-run SurGen | in progress (owner) |
| 10 | Regenerate the Word deliverables (`tools/build_results_docx.py`, `build_supplementary_results.py`, `build_ev_experiment_report.py`) from the new workbook | not started; `build_ev_experiment_report.py` still has a section arguing for k = 20 |

**SurGen 991 encoding** is being continued separately by the project owner. State on 2026-10-04: 630 of 991 slides screened (tissue filter), 622 with features, last pipeline activity 2026-09-09. Before resuming on this machine:

- Run the screening step with `SURGEN_SKIP_STEP_C=1` and `SURGEN_INCREMENTAL_A1A2=1` set.
- Delete any partial `.czi` left in the download folder first. The pipeline never re-downloads a file that already exists, and a partial file passes a size check.
- The CONCH v1, UNI2, H-Optimus-1 and TITAN scripts still point at server paths and the old 624-slide label rule. The CONCH 1.5, Virchow2 and PRISM scripts already use the 991 labels.
- Encoding the 369 new slides needs roughly 210-240 GB; drive F: had 213 GB free.

**Already tried, do not repeat:** inner-CV threshold fitting on PAIP-IV, kNN distance weighting, further k sweeps on SurGen, PAIP-CV, widening the ANN grid, a quantile threshold for kNN, and the four PAIP-IV ANN protocols. Details are in the ledger.

## 5. Before running anything

- **The runners move results.** Each one renames the existing tree to `<name>_ARCHIVED_<date>` before writing, and those folders are git-ignored. Pass `--no-archive` to keep the tree in place.
- **`cv_runner.py --experiment TCGA-CV --force` rewrites `thresholds_TCGA.json`.**
- **SurGen runs skip missing encoders silently.** Check that `conch1-5`, `conch-v1`, `h-optimus-1`, `uni2-h`, `virchow2` and `prism` all exist under `surgen_data/surgen_processed/`.
- **Never run `git gc --prune`** on this clone. About 2.5 GB of unreachable objects in `.git` include 1,574 file versions that exist nowhere else.

## 6. October 2026 cleanup

The working copy shrank by 34 GB, from about 79 GB to about 45 GB including `.git`. Nothing unique was deleted.

| What | Size | Action |
|---|---|---|
| Byte-identical duplicates (verified by SHA-256 twice) | 9.8 GB | deleted; one copy kept |
| Two extra copies of the CONCH weights | 1.5 GB | replaced by hard links, so the old paths still work |
| `__pycache__` folders | 3.5 MB | deleted |
| Superseded snapshots, old model files, test outputs, old logs | 22.9 GB | moved to the archive |
| Old `.docx` reports, backups and scratch notebooks | 103 MB | removed from git; copies in the archive |

The archive is next to the repository, at `KSA_project2/_archive/Cancer-detection-classifier_2026-10/`:

- `MANIFEST.csv` - one row per action, with the original path. To put something back, move it to that path.
- `tier2/` - moved material, under the same relative paths it had in the repository.
- `from_recycle_bin/` - seven archived result trees, two SurGen reports and the untracked files of `experiments/`, recovered from the Recycle Bin before it was purged.
- `removed_from_git/` - copies of the tracked files whose deletion was committed.
- `audit_2026-10-04/` - the four full audit reports this page summarises.

Also recovered: `surgen_data/surgen_processed/conch1-5` (107 GB, the SurGen CONCH 1.5 patch features), deleted on 2026-09-21 and restored from the Recycle Bin to its original path.

Deliberately left in place:

- `slide_classification/cache/` (1.1 GB) - the next analysis steps read it.
- `data_cleaning/BG_SVM_training/training_data/` (4.9 GB), `feature_extraction/test_data/` (3.8 GB), the CRC-100K feature tensors (3.8 GB) - unique inputs; owner decision.
- `Complete_Pipeline/logs/` from September and `Complete_Pipeline/crash_diagnostics/` - in use by the SurGen encoding.
- The four `*_ARCHIVED_202609*` trees in `slide_classification/` - small, and cited as provenance.
- Outside the repository: `dataset/patch_data` (824 GB) and `paip_data/patch_data` (189 GB) are PNG patch caches. Every encoder has been extracted from them, so they are needed only to add a new encoder. Owner decision.

## 7. Security

Hugging Face tokens were committed to this repository in the past and remain readable in its history. Treat every one of them as compromised and revoke it on huggingface.co (Settings, Access Tokens). The code now reads `HF_TOKEN` from the environment or from a local, git-ignored `.env`.
