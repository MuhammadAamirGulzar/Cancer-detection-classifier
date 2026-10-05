# Project status - MSI-H classification from colorectal whole-slide images

Last updated: 2026-10-05, after a full read-only audit (2026-10-04), a repository cleanup, and the first analysis round (section 4).

This page answers three questions: where are the current results, can they be trusted, and what is left to do.
For the history of what has already been tried, see [EXPERIMENT_LEDGER.md](EXPERIMENT_LEDGER.md).

---

## 1. Where the current results are

Everything lives under `slide_classification/`. The last full run was on 2026-09-03.

| Experiment | Results tree | Slides (MSI-H) | Last run | Contents |
|---|---|---|---|---|
| TCGA-CV | `TCGA_Results/` | 413 (60) | 2026-08-04 | 22 combinations, 4 folds, seed 42 |
| PAIP-IV | `PAIP_IV_Results/` | 47 train / 31 test | 2026-08-31 | 22 combinations, provider split, bootstrap CIs |
| PAIP-EV | `TCGA_PAIP_EV_Results/` | 78 (19) | 2026-09-03 | 22 combinations x 3 variants, kNN k = 35 and RF corrected |
| SurGen-CV | `SurGen_Results/` | 622 (60) | 2026-09-03 | 21 combinations, case-level folds, no PRISM |
| SurGen-EV | `SurGen_EV_Results/` | 622 (60) | 2026-09-03 | 21 combinations x 3 variants, kNN k = 20 and RF corrected |
| TCGA models for EV | `TCGA_FULL_Models/` | 413 | 2026-08-04 | 22 combinations x 5 heads x seeds 42-46 |

The consolidated workbook is `slide_classification/best_of_all_exps_metric.xlsx` (built 2026-09-03).
The audit rebuilt it independently from the 108 result files and compared 24,780 cells: zero mismatches.
Every number in the README matches the workbook.

Files the results depend on, which must not be edited or regenerated casually:

- `thresholds_TCGA.json` - the frozen decision thresholds used by every external-validation number.
- `knn_k35_thresholds.json`, `tcga_full_hparams.json`.
- The label files `kfolds_IDARS_fixed.csv`, `paip_78_labels.csv`, `surgen_labels.csv`.
- The trained models inside each tree (`models/` folders, `TCGA_FULL_Models/`, `final_models/`). They are git-ignored, so they exist only on this disk.
- `cache/` - assembled feature matrices. It is treated as regenerable, but for two weeks it was the only usable copy of the SurGen CONCH 1.5 features (section 6).

## 2. Headline numbers

| Experiment | Best AUROC [95% CI] | Configuration | Same number for the configuration TCGA-CV selects |
|---|---|---|---|
| TCGA-CV | 0.928 [0.89, 0.96] | H-Optimus-1 / Caption-15 / LR | - |
| PAIP-IV | 0.926 | CONCH 1.5 / Caption-15 / kNN | - |
| PAIP-EV | 0.915 [0.84, 0.98] | UNI2 / Averaging / LR | 0.879 |
| SurGen-CV | 0.853 | UNI2 / Caption-15 / RF | - |
| SurGen-EV | 0.856 [0.81, 0.90] | UNI2 / Caption-15 / LR | 0.740 |

The confidence intervals were computed during the audit by bootstrapping the stored per-slide scores. They are not yet in the workbook.

**Main finding.** Mean AUROC over 5 encoders x 5 classifier heads:

| Experiment | Averaging | Caption-14 | Caption-15 | Tissue-type clustering |
|---|---|---|---|---|
| TCGA-CV | 0.775 | 0.802 | 0.792 | 0.794 |
| PAIP-IV | 0.760 | 0.854 | 0.842 | 0.852 |
| PAIP-EV | 0.809 | 0.858 | 0.866 | 0.833 |
| SurGen-CV | 0.666 | 0.711 | 0.718 | 0.686 |
| SurGen-EV | 0.650 | 0.713 | 0.730 | 0.691 |

A semantic aggregation is the best choice in 123 of 125 encoder x head cells across the five experiments. No confidence interval or paired test has been computed for this claim yet.

**SurGen labels.** All SurGen results use the earlier 622-slide set. None of those 622 slides changed label in the 991-slide relabel, so the results stay valid. The relabel adds 369 slides (40 MSI-H); none is encoded yet.

## 3. Weak points to fix before writing up

Ranked by how much a reviewer would care.

1. **The headline numbers are maxima over 100 configurations picked on the test data.** The configuration TCGA-CV itself selects scores 0.879 on PAIP and 0.740 on SurGen. Report that pre-specified configuration (or the mean-by-method table) as the primary result.
2. **No confidence intervals or significance tests** except PAIP-IV. Fold-to-fold AUROC spread on TCGA-CV has a median SD of 0.053, larger than most differences between top rows.
3. **SurGen kNN uses k = 20, chosen from a sweep on SurGen labels.** PAIP uses k = 35, justified from TCGA alone. Use k = 35 on both.
4. **The corrected threshold was applied to kNN and RF only, after seeing external results.** ProtoNet is dead or nearly dead (balanced accuracy 0.50) in 16 of 21 SurGen-EV rows, although the same correction would fix it. Apply one rule to all heads, or to none.
5. **PAIP-IV ANN rows use the `legacy` protocol** (hyperparameters picked on 9 slides, trained on 38 of 47). `final_models/ann_old` records a different final choice, and `RESULTS.md` describes a third. Decide, then regenerate that tree.
6. **Provenance.** The `git_commit` stamp in results from 2026-08-10 to 09-03 names a commit that does not contain the code that ran. The external-validation runs loaded scikit-learn 1.7.0 models in an environment with scikit-learn 1.9.0. Fix one environment before any re-run.
7. **PRISM on SurGen cannot be used as is.** The embedding file for `SR386_40X_HE_T237_01` is corrupt, slide `SR386_40X_HE_T241_01` is missing 52% of its tiles, and four more slides miss 6-9 tiles each (incomplete downloads). Re-encode at least the first two.
8. Smaller items: random-forest predictions in cross-validation are cut at 0.30 although tables say 0.5; the kNN row of the mean-by-head table mixes k = 3, 35 and 20; the tissue-class subset experiments (TCGA balanced accuracy 0.85-0.87) were selected on test folds with the pre-fix ANN and were never validated.

## 4. Remaining work

Items 1-7 need no GPU: they run on stored predictions and cached features.

| # | Task | Status |
|---|---|---|
| 1 | One threshold rule on both external cohorts: k = 35 everywhere, corrected threshold for all five heads | **analysed** in `experiments/final_analysis/`; the published trees, workbook and README still show the old scheme |
| 2 | Case-level bootstrap CIs and paired tests (aggregation vs averaging; encoders; heads) | **done** in `experiments/final_analysis/` |
| 3 | Report the TCGA-selected configuration as primary; keep best-of-100 as exploratory | **computed**; the README still headlines best-of-100 |
| 4 | Regenerate `PAIP_IV_Results/` with the `ann_old` ANN protocol (owner decision) | not started |
| 5 | PRISM on SurGen | blocked: `SR386_40X_HE_T237_01.pt` is corrupt and `SR386_40X_HE_T241_01` is missing 52% of its tiles; both need re-encoding |
| 6 | Probability ensembles | tested: the 4-fold ensemble gains 0.015-0.018 AUROC on SurGen for LR and RF and nothing on PAIP; low priority |
| 7 | Redo the top-k tissue-class aggregation with nested cross-validation and take it to external validation | not started, about 1 day |
| 8 | Attention-based MIL baseline on the cached patch features | not started, 2-4 days on one GPU |
| 9 | Finish the 991-slide SurGen encoding and re-run SurGen | in progress (owner) |

**What the analysis of items 1-3 found** (details and tables in [../experiments/final_analysis/SUMMARY.md](../experiments/final_analysis/SUMMARY.md)):

- Semantic aggregation beats averaging on the external cohorts (SurGen-EV +0.054 mean AUROC, p < 0.001; PAIP-EV +0.050 for Caption-15, p = 0.016) but **not** within TCGA cross-validation (+0.016, p = 0.22). The supportable claim is that it generalises better.
- The uniform threshold rule removes all 38 dead or saturated operating points and raises mean balanced accuracy for every head on both cohorts.
- The configuration TCGA-CV selects (H-Optimus-1 / Caption-15 / LR) scores 0.879 [0.73, 0.98] on PAIP, not distinguishable from the best, and 0.740 [0.65, 0.82] on SurGen, clearly below the best.
- H-Optimus-1 is the best encoder within TCGA; UNI2 is the best on SurGen. On the same CONCH 1.5 patch features the TITAN slide encoder beats all four aggregation methods.

**Next:** make the published trees, workbook, README tables and figures match the uniform rule, and regenerate PAIP-IV with `ann_old`. One choice is open first: whether ANN and RF report the AUROC of the five-seed mean probability (one prediction per slide, what the analysis uses) or the mean of five single-seed AUROCs (what the workbook has now).

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
