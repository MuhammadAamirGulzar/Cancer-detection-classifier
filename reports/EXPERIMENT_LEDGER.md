# Experiment ledger - MSI-H CRC project

Scope: everything that has been tried, with outcome, evidence path and verdict, so that nothing is repeated by accident. Compiled from a read-only audit on 2026-10-04 and updated on 2026-10-05 after the repository cleanup. For the short version (where the results are, what is left to do) see [PROJECT_STATUS.md](PROJECT_STATUS.md).
Repo root = `D:/Aamir Gulzar/KSA_project2/Cancer-detection-classifier` (all relative paths below are relative to it).

Evidence tags
- **VERIFIED** = I opened the file / parsed the table / listed the directory myself during this audit.
- **INFERRED** = taken from a document, a name or a date without an independent check.
- Location codes: **WT** = present in the working tree; **GIT** = at the time of the audit, tracked in git but missing from the working tree; **HIST** = only in older commits (e.g. `e8cbb894`, tip of branch `dev`). Since 2026-10-05 the items marked GIT under `experiments/`, `RESULTS.md`, `OPEN_QUESTIONS.md`, `SERVER_RUN.md` and `Complete_Pipeline/` are back in the working tree. The `.docx` reports, `RESULTS.BACKUP.md`, `RESULTS.PRE_SURGEN_20260827.md`, `CLAUDE_CODE_WORK_ORDER.md` and `TOKENS_TO_REVOKE.md` were removed from the repository; read them with `git show 0423ae40:<path>` or from the local archive folder.

Nothing was modified while the ledger was compiled.

---

## 0. State in one page

| Experiment | Live tree | Last run | N | What is actually in it (VERIFIED from `result_*.json` stamps) |
|---|---|---|---|---|
| TCGA-CV | `slide_classification/TCGA_Results` | 2026-08-04 22:14-22:40 | 413 (60 MSI-H) | 22 combos, wide ANN grid, threshold 0.5 (RF at 0.3 - see S2) |
| TCGA-FULL models | `slide_classification/TCGA_FULL_Models` | 2026-08-04 22:40-22:55 | 413 | 22 combos x 5 heads x seeds 42-46 |
| PAIP-IV | `slide_classification/PAIP_IV_Results` | 2026-08-31 15:54 | 47 train / 31 test | 22 combos; **ANN protocol = `legacy`** (see S1) |
| PAIP-EV | `slide_classification/TCGA_PAIP_EV_Results` | 2026-09-03 12:22 | 78 (19 MSI-H) | `promoted`: kNN k=35 + refit tau, RF quantile; LR/ANN/Proto frozen tau_TCGA |
| SurGen-CV | `slide_classification/SurGen_Results` | 2026-09-03 11:05-11:42 | 622 (60 MSI-H), 552 cases | 21 combos (no PRISM), case-level folds |
| SurGen-EV | `slide_classification/SurGen_EV_Results` | 2026-09-03 13:45 | 622 | `promoted`: kNN **k=20** (chosen on SurGen) + refit tau, RF quantile; others frozen |

Workbook `slide_classification/best_of_all_exps_metric.xlsx` (2026-09-03 13:50) agrees with these trees (VERIFIED: PAIP-IV ANN mean BalAcc over the 20 aggregation rows = 0.6866 = legacy protocol; PAIP-EV kNN 0.7029/0.8114; SurGen-EV kNN 0.5750/0.6770).

Central result (VERIFIED from the workbook, `tcga_full` variant for EV, mean AUROC over 5 encoders x 5 heads):

| Experiment | Averaging | Caption-14 | Caption-15 | TTC | Best single AUROC |
|---|---|---|---|---|---|
| TCGA-CV | 0.775 | 0.802 | 0.792 | 0.794 | 0.928 H-Optimus-1 / Caption-15 / LR |
| PAIP-IV | 0.760 | 0.854 | 0.842 | 0.852 | 0.926 Conch1_5 / Caption-15 / kNN |
| PAIP-EV | 0.809 | 0.858 | 0.866 | 0.833 | 0.915 UNI2 / Averaging / LR |
| SurGen-CV | 0.666 | 0.711 | 0.718 | 0.686 | 0.853 UNI2 / Caption-15 / RF |
| SurGen-EV | 0.650 | 0.713 | 0.730 | 0.691 | 0.856 UNI2 / Caption-15 / LR |

"Best aggregation per encoder x head" (25 cells per experiment): caption-based wins 19/25 (TCGA-CV), 15/25 (PAIP-IV), 24/25 (PAIP-EV), 24/25 (SurGen-CV), 21/25 (SurGen-EV); Averaging wins 1, 0, 1, 0, 0. No significance test has ever been run on these differences (see E-list item N1).

---

## 1. Timeline (VERIFIED from git log, file mtimes and `logs/*.log` unless marked)

| Date | Event |
|---|---|
| 2025-07-01..10 | 8 commits by fahamin5149: CLAM patching, downsampling tests, first background SVM. Last commit before a 13-month gap. |
| 2025-07..08 | Tissue classifier on CRC-100K (ResNet-18, Conch1.5, UNI2, normal vs upscaled); "Titan_Project" prototype; other-target fold files `kfold_BRAF/KRAS/TP53/CIMP.csv` (2025-08-07). |
| 2025-08..11 | Five-crop feature extraction adopted; averaging / TTC / caption aggregation notebooks; H-Optimus-1 added (2025-08-07/08); TCGA + PAIP per-dataset background SVMs (2025-09-30 / 10-01); LR and ANN grid searches (2025-09-11); graph-aggregation prototype (2025-11-13). |
| 2026-02 | RF head added (2026-02-11); CONCH zero-shot validated on CRC-100K (02-17/18); first TCGA->PAIP inference (`TEST_FIRST.md` 02-18). |
| 2026-03 | Virchow2 (03-11) and ConchV1 (03-18) added; ten-crop exploration (03-07); TTC combination experiment v1 (03-27). |
| 2026-04..05 | Few-shot CONCH probes (04-07..28); tier-filtered combination search (04-24, 05-13); fixed global top-k (05-15); pathologist review (05-19/20). |
| 2026-06 | Unified background SVM (06-16/17); combined few-shot v2 (06-18); SurGen best-k on H-Optimus-1 (06-18); 15-class caption scheme (06-26..30). |
| 2026-07 | SurGen v5/v6 pipeline runs (07-08..27); SurGen aggregation scripts. |
| 2026-08-04 | Remediation day: 26 commits, Phases 0-4 (tokens purged, data layer, ANN fixes, case-level folds, PAIP-CV retired, TCGA-FULL, PAIP-IV, EV runners, reporting). Branch `dev`. |
| 2026-08-05 | COHORT_COUNTS.md (last commit until 09-08; everything 08-10..09-03 ran on an uncommitted tree). |
| 2026-08-09/10 | PAIP 5 missing slides recovered (73 -> 78); PAIP-IV and PAIP-EV re-run. |
| 2026-08-10..12 | PAIP-IV ANN experiments 1-4 (4 repeats each), threshold calibration (5-fold and 4-fold). |
| 2026-08-13 | `final_models/ann_old` frozen ("owner decision"). |
| 2026-08-11 | SurGen TITAN (tiled ALiBi patch). 08-16..22: SurGen 15-class caption. 08-21..09-06: SurGen PRISM encoding. |
| 2026-08-27 | SurGen aggregated features for H-Optimus-1/UNI2/ConchV1 transferred from the server (`surgen_agg_transfer_20260827.sha256`); first full SurGen-CV / SurGen-EV (21 combos; two H-Optimus-1 combos at N=613). |
| 2026-08-28..31 | `tools/knn_k_sweep.py`; threshold experiments A-D; `corrected_ev`; full re-run of PAIP-IV (legacy ANN!), PAIP-EV and SurGen-EV in `corrected` mode (08-31). |
| 2026-09-02 | PAIP-EV `promoted` (kNN k=35, RF quantile). |
| 2026-09-03 | H-Optimus-1 613 -> 622 fix; SurGen-CV re-run; SurGen-EV diagnostic (strip + k sweep); SurGen-EV promoted at k=20; reports, radar and t-SNE rebuilt; SurGen cohort / drop-reason / split-reconciliation audits. `progress.json` last updated. |
| 2026-09-05 | SurGen TITAN pickle. |
| 2026-09-07/08 | 991-slide relabel; incremental A-1/A-2; standalone conch1-5 / virchow2 Step C runners (3 commits). |
| 2026-09-09 15:58 | Last pipeline log line: download of `SR1482_40X_HE_T017_02.czi` at 80 %. Nothing after. 630 slides screened of 991. |
| 2026-09-21 | The SurGen `conch1-5` feature tree was deleted from `surgen_data/surgen_processed` (to the Recycle Bin). Restored on 2026-10-05 - see S6. |
| 2026-10-02 | Big commit `19d6d45e` (2060 files; archived Aug-04 trees removed from tracking) + `0423ae40`. 10-03 README.md (untracked). |

---

## 2. A. Experiments

Format: **ID - title** | question | varied / values | cohort | date | outcome (key numbers) | evidence | verdict.

### 2.1 Pre-processing and features (notebook era)

**X01 - Magnification normalisation** | How to get 20x-equivalent 512 px tiles from mixed 20x/40x slides | 40x: CLAM 1024 px tiles resized to 512 (bicubic); 20x: 512 tiles directly; a per-slide `--target_magnification` variant of CLAM feature extraction tested on one slide | TCGA, PAIP | 2025-07 | adopted the resize route; ~1.46 M TCGA patches | `data_preprocessing/Patch_creation.ipynb`, `data_preprocessing/Testing_Files/*`, `feature_extraction/extract_features_fp_40xFix.py`, `feature_extraction/features_20x`, `features_40x_Optimized` (1 slide each) (WT, VERIFIED listing); Supplementary draft s1.1 (GIT) | conclusive-positive (adopted). No 40x-native or 10x features were ever evaluated for MSI.

**X02 - Per-dataset background SVM** | Which RGB statistics separate tissue from white background | all 20 three-feature subsets of {avg,std} x {R,G,B} x kernel {linear, rbf}, C=10, gamma=0.1, 70/30 split (test n=1926) | TCGA (and PAIP separately) | 2025-07 -> 2025-09-30 / 10-01 | best subset 98.75 % (rbf); chosen (avg_G, avg_B, std_R), **linear, unscaled inputs**, 97.35 %, confusion [[1562,25],[26,313]] | `data_cleaning/TCGA_SVM/SVM_model_training.ipynb` cells 6, 9, 11, 30; `data_cleaning/PAIP_SVM/` (PAIP uses `svm_model_rbf.pkl` + scaler) (WT, VERIFIED) | conclusive-positive. Note: `Complete_Pipeline/svm_model.pkl` is byte-identical (md5 13d654c0...) to the TCGA model, so **SurGen is filtered with the TCGA-trained SVM** plus pixel rules; the draft's "SurGen_SVM" / "unified SVM in production" statement is not what the code does (VERIFIED).

**X03 - Unified background SVM (BG_SVM)** | One background model for all cohorts | RBF, C=10, gamma=scale, class_weight=balanced, 6 RGB features, StandardScaler; trained on 2114 CRC-100K BACK + 1343 TCGA background vs 8067 TCGA tissue | TCGA + CRC-100K | 2026-06-16/17 | 98.61 % (INFERRED from draft; pkl files present) | `data_cleaning/BG_SVM_training/` (WT) | partial: used only to build `tcga_clean_no_back.csv` / `paip_clean_no_back.csv`, which feed the 15-class caption path and few-shot inference; not used by Averaging / TTC / Caption-14 nor by the SurGen pipeline.

**X04 - Normal vs upscaled patch input** | Resize 512 -> 224 vs keep 512 for Conch1.5 and UNI2 | 2 x 2 | CRC-100K tissue classifier; TCGA averaging features | 2025-07 | tissue accuracy Conch1.5 89.50 vs 89.50, UNI2 97.12 vs 96.64 (VERIFIED in notebooks); slide-level MSI outcome for the four `1-Averaging_Normal _Upscale` feature sets (417 files each) not found anywhere | `TissueClassifier_CRC100K/Crc100K_Model_Training_Conch_UNI2*.ipynb`, `dataset/slide_aggregation/1-Averaging_Normal _Upscale/` | inconclusive / superseded by five-crop.

**X05 - Five-crop features** | 5 x 256 px crops per 512 px tile (each resized to 224 for H-Optimus-1/Virchow2), 5 x D saved, mean over crops at aggregation | - | all cohorts | 2025-08 -> 2026-03 | adopted for all five patch encoders | `feature_extraction/Feature_Extraction_*fivecrop*.ipynb`, `dataset/Features/*_fivecrop*` (WT, VERIFIED listing) | adopted **without any recorded ablation** against single-crop on MSI.

**X06 - Ten-crop** | TenCrop / custom TenCropNoFlip | - | TCGA (UNI2 at least) | 2026-03-07..11 | crop-coverage visualisation and a UNI2 ten-crop file validator exist; no ten-crop feature tree or result on disk | `feature_extraction/visualize_tencrop.ipynb`, `Analyze_Models.ipynb`, TenCrop branches in the UNI2 / H-Optimus-1 / Virchow2 notebooks | abandoned (no downstream result).

**X07 - ResNet-18 baseline** | ImageNet ResNet-18 as tissue classifier backbone and for an early slide-level ANN | 224 vs 512 input | CRC-100K; early TCGA (418 slides, StratifiedKFold) | 2025-07 | tissue accuracy 70.06 % (224) and 97.55 % (512); slide-level run interrupted (KeyboardInterrupt), no result file | `TissueClassifier_CRC100K/Crc100K_Model_Training_Resnet.ipynb` | abandoned. **The unlabelled row in `TissueClassifier_CRC100K/model_metrics/eval_metrics.csv` (acc 0.7006, AUROC 0.948) is this ResNet-18/224 model, not a production classifier** (VERIFIED: 70.06 % matches).

**X08 - CRC-100K tissue classifier per encoder** | ANN head per encoder | grid h1 {128,224,256} x h2 {64,128} x C {0.1,1,10} x max_iter {500}; selected on a stratified 20 % validation split of NCT-CRC-HE-100K-NONORM; tested on CRC-VAL-HE-7K | CRC-100K | 2025-07 -> 2026-03 | test accuracy Conch1.5 89.50, ConchV1 92.59, H-Optimus-1 96.60, UNI2 97.12, Virchow2 97.30; best config always C=10, h2=128, h1=256 (224 for Conch1.5/ResNet); C=0.1 collapses to 10.4 % | `TissueClassifier_CRC100K/Crc100K_Model_Training_*.ipynb`, `best_parameters/*.json`, `models/*.pth` (WT, VERIFIED) | conclusive-positive (adopted; clean train/val/test protocol).

### 2.2 Caption / tissue assignment

**X09 - CONCH zero-shot validated on CRC-100K** | Is zero-shot caption assignment accurate | 9-class prompts; 14-class prompts mapped to 9 | CRC-100K (100 k patches) | 2026-02-17/18, 2026-05-05..13 | 9-class accuracy 0.748, balanced 0.749; recall TUM 0.441, STR 0.390, NORM 0.525, BACK 0.751. 14-class: 59.1 % correct, 13.6 % correct-subtype, 27.3 % incorrect (NORM 66.7 % wrong, STR 50.2 % wrong) | `slide_aggregation/caption_generation/Results/Conch_Evaluation/crc100k_conch_evaluation.csv`, `Analysis_and_Visualization/Results/CRC100K_ZeroShot_Statistics/patch_class_statistics.csv` (WT, VERIFIED) | conclusive: zero-shot assignment is mediocre, yet it is the basis of both caption aggregations.

**X10 - Few-shot CONCH probe (CRC-100K)** | Few-shot vs zero-shot | 20 or 100 patches per class; nearest-centroid, logistic regression, ZS+ProtoNet ensemble | CRC-100K | 2026-04-07..28 | 180 patches: LR 0.9649 / NC 0.9605 / ensemble 0.9340 vs ZS 0.7484; 900 patches: LR 0.9671. Applied to all 1,377,933 TCGA patches | `.../Results/Few_Shot_Evaluation/{180,900}_trainset/summary.csv`, `classification_results_tcga_fewshot.csv` (WT, VERIFIED) | partial: **never carried into a slide aggregation** (no few-shot aggregation tree exists under `dataset/slide_aggregation`).

**X11 - Combined few-shot v2 (pathologist + CRC top-up)** | 14-class few-shot from 101 pathologist-reviewed TCGA patches + 99 CRC-100K top-ups, cap 20 per class, 160/40 split | LR (C=1, balanced), NC, ZS, ensemble | TCGA patches | 2026-06-17/18 | test (40 patches): ZS 0.750/0.731, NC 0.800/0.731, LR 0.775/0.712, ensemble 0.825/0.808 (acc/BalAcc); applied to 1,351,176 TCGA patches | `FewShot_Combined_Conch.ipynb`, `Results/Few_Shot_Evaluation/Combined/summary.csv`, `classification_results_combined_fewshot_v2.csv` (WT, VERIFIED) | partial / inconclusive (40-patch test set; never aggregated or classified).

**X12 - Pathologist review of patch labels** | Two pathologists label disagreement patches (ZS vs ANN vs few-shot) | ~95 reviewed + 60 blinded patches | CRC-100K / TCGA patches | 2026-05-19/20 | review workbook exists; no summary metric file found | `Analysis_and_Visualization/aggregated_both_Pathologist_Reviews.xlsx`, `complete_pathologists_reviews_file.ipynb`, `Results/classifier_comparison/*.xlsx` (WT, VERIFIED existence) | inconclusive (no quantified outcome on disk).

### 2.3 Aggregation strategies (the main experimental variable)

**X13 - Averaging vs Caption-14 vs Caption-15 vs TTC** | Does semantic pooling beat mean pooling | 4 aggregations x 5 encoders x 5 heads | all five experiments | built 2025-10 -> 2026-06; evaluated 2026-08-04 -> 09-03 | see table in section 0: semantic aggregations beat Averaging on mean AUROC in every experiment (+0.02 to +0.09); a semantic aggregation is the best in 123 of 125 encoder x head cells across the five experiments (caption-based 103, TTC 20), Averaging in 2 | `slide_classification/best_of_all_exps_metric.xlsx` (WT, VERIFIED) | conclusive-positive in direction; **no CI or paired test**; best PAIP-EV AUROC is still Averaging/UNI2/LR 0.915.

**X14 - Caption-14 vs Caption-15** | Does the revised class set help (drop MES, add TIL and BACK) | 14 vs 15 | all | 15-class built 2026-06-26..30 (SurGen 08-16..22) | mean AUROC 14 vs 15: TCGA-CV 0.802 / 0.792; PAIP-IV 0.854 / 0.842; PAIP-EV 0.858 / 0.866; SurGen-CV 0.711 / 0.718; SurGen-EV 0.713 / 0.730 | workbook; `slide_aggregation/caption_generation/config_files/*.yaml` | inconclusive and **confounded**: the 15-class run also uses a different patch list (1,355,579 TCGA patches after the unified BACK SVM vs 1,377,933; PAIP 271,774 vs 275,541) (VERIFIED row counts) and a different prompt set. An `extended_prompts` 15-class config exists; which one produced `Conch_zeroshot_weights_15_classes.pt` was not determined.

**X15 - TITAN and PRISM slide encoders as baselines** | Do slide encoders beat the aggregations | TITAN (Conch1.5 features, 768-d), PRISM (Virchow v1 tiles + Perceiver, 1280-d) | TCGA-CV, PAIP-IV, PAIP-EV; TITAN also SurGen | 2026-08 | best AUROC TITAN / PRISM: TCGA-CV 0.877 / 0.869; PAIP-IV 0.935 / 0.821; PAIP-EV 0.862 / 0.854; SurGen-CV TITAN 0.801; SurGen-EV TITAN 0.764. Both below the best aggregation except PAIP-IV TITAN | workbook `*_TITAN_PRISM` sheets (VERIFIED) | conclusive (baselines). **PRISM on SurGen never run although 622 embeddings exist** (`surgen_data/surgen_processed/prism/features/slide_aggregation/PRISM/prism`, completed 2026-09-06; runners exclude PRISM for SurGen by default).

**X16 - Graph-based aggregation** | GraphSAGE + attentional pooling, kNN graph k=8, positional MLP | - | one slide (1878 patches) | 2025-11-13 | untrained forward pass only; "Set Transformer - not yet implemented" | `slide_aggregation/graph_based/Graph_Based_Aggregation.ipynb` (WT, VERIFIED) | abandoned prototype. No attention-MIL (ABMIL / CLAM / TransMIL) was ever trained - the only occurrences are the vendored CLAM toolbox and this notebook (VERIFIED by grep).

**X17 - TTC combination experiment v1 (background included)** | Do subsets of tissue classes beat all 9 | exhaustive k=3 and k=5 per encoder, all 5 heads | TCGA-CV | 2026-03-27 | best per encoder: H-Optimus-1 mucin+stroma+tumor BalAcc 0.837 / AUROC 0.911 vs all-9 0.805; UNI2 0.817 vs 0.770; Virchow2 0.807 vs 0.739; Conch1_5 0.750 vs 0.717; ConchV1 0.753 vs 0.732 | `Analysis_and_Visualization/TTC_Combinations_Exp/best_results_all_bacc.csv` (WT, VERIFIED) | superseded by X18/X19; pre-remediation.

**X18 - Tier-filtered exhaustive combination search** | Same, with tiers (Strong >= 0.65, Neutral >= 0.58 BalAcc; <= 45 % weak, <= 70 % neutral, >= 1 strong) | 9-class k in {3,5,8}; 14-class k in {3,4,14}; ANN only | TCGA-CV | 2026-04-24, 05-13 | best: H-Optimus-1 9-class k=3 adipose+mucin+tumor 0.870 / 0.932; 14-class k=3 carcinoma+stroma+mucin 0.877 / 0.919; UNI2 0.855 / 0.913; Virchow2 0.810 (9) / 0.849 (14, k=4); Conch1_5 0.784; ConchV1 0.787 | `slide_classification/Tissue_Type_Combinations_Exp/TCGA_TTC_Result/<agg>/<model>/1-MSIH/Output/tissue_combo_summary*.xlsx` (WT, VERIFIED) | **optimistic and stale**: combination and ANN config were chosen on the test folds, with the double-softmax ANN; never regenerated after 2026-08-04.

**X19 - Fixed global top-k ("best-k labels")** | Rank classes by single-class BalAcc averaged over 5 encoders, take top-k | 9-class k in {3,5,7,8}; 14-class k in {3,5,7,9,10,14}; ANN | TCGA-CV | 2026-05-15 | 9-class ranking: tumor 0.790, mucin 0.723, stroma 0.687, debris 0.645, normal 0.608, lymphocyte 0.595, muscle 0.577, adipose 0.533. k=3 (tumor+mucin+stroma): H-Optimus-1 0.852 / AUROC 0.920, UNI2 0.862 / 0.909, Virchow2 0.817, Conch1_5 0.776, ConchV1 0.729; k=8: 0.802 / 0.798 / 0.732 / 0.727 / 0.729. 14-class ranking: PDC 0.761, ADE 0.751, CAR 0.734, MUC 0.712, SIG 0.709; k=3 H-Optimus-1 0.869, k=5 0.865 / AUROC 0.926, k=14 0.833 | `.../TCGA_TTC_Result/*/Global/1-MSIH/*.xlsx` (WT, VERIFIED) | inconclusive-promising: fewer classes look better, but ranking and evaluation use the same test folds and pre-fix ANN. **Never regenerated, never taken to PAIP/SurGen EV.**

**X20 - SurGen best-k (H-Optimus-1)** | Same on SurGen | k in {3,5,8} | SurGen-CV (slide-level folds, leaky) | 2026-06-18 | tumor alone 0.786 / AUROC 0.825; k=3 tumor+debris+mucin 0.755 / 0.801; k=8 0.743 / 0.790 | `slide_classification/misc/Surgen_Results_Outdated/Tissue_Type_Clustering/H-Optimus-1/1-MSIH/Output/` (WT, VERIFIED) | leak-inflated; PHASE_2_REPORT says "must be regenerated" - it never was. Server-side `Surgen_TTC_exp` results are not on this machine.

### 2.4 Classifier heads and their tuning

**X21 - LR C / max_iter grid** | Regularisation strength | C {0.01,0.1,1,10} x max_iter {300,500,1000} | TCGA-CV, caption-14 features of H-Optimus-1 and UNI2 (2025-09-11), Conch1_5 (2026-02-11), TITAN (notebook cell) | - | test-fold BalAcc / AUROC: H-Optimus-1 C=10 0.793 / 0.925 (best BalAcc), C=0.01 0.716 / 0.929; UNI2 C=10 0.728 / **0.877** vs C=0.1 0.722 / **0.902**; Conch1_5 C=10 0.723 / 0.839 vs C=0.01 0.655 / 0.846; TITAN C=1 0.721 / 0.856 vs C=10 0.698 / 0.841; max_iter irrelevant | `slide_classification/misc/Test_Result/*/Output/grid_search_linear_C_maxiter.xlsx`; `Slide_Classification.ipynb` cell 10 output (WT, VERIFIED) | C=10 adopted on test-fold BalAcc. **On AUROC (the declared primary metric) C=10 is the worst or tied in 3 of 4 grids.** Never re-tuned in the corrected pipeline.

**X22 - ANN 45-point grid (notebook)** | h1 {128,256,512} x h2 {64,128,256} x max_iter {100,200,350,500,1000} | - | TCGA-CV, same feature sets as X21 | 2025-09-11, 2026-02-11 | spread across the grid: H-Optimus-1 BalAcc 0.779-0.834; UNI2 0.748-0.821; Conch1_5 0.637-0.730; TITAN 0.675-0.815; no monotone trend in any axis | `misc/Test_Result/*/Output/grid_search_ann_hidden1_hidden2_iter.xlsx` (VERIFIED) | inconclusive (noise; test-fold selection; pre-fix network).

**X23 - RF head and its 0.3 threshold** | RF never predicted MSI-H on PAIP at 0.5 | class_weight {0:1,1:10}, threshold 0.3, 500 trees, min_samples_split 5; alternates left in code: max_depth 6 / min_split 20 / min_leaf 10 / 'balanced'; 1500 trees + threshold tuned on the validation fold (`find_best_threshold`) | TCGA -> PAIP | 2026-02-11 -> 03-04 | adopted hand-set values; no result file compares the alternates | `slide_classification/test_random_forest.py`, `eval_patch_features/r_forest_eval.py` (commented block + defaults) (WT, VERIFIED) | adopted by hand; alternates untested on record.

**X24 - LogReg-balanced / SVM-RBF / RF comparison script** | - | 5-fold StratifiedKFold or GroupKFold | - | 2026-02-11 | placeholder paths `"__"`, empty model dict, no output | `eval_patch_features/r_forest.py` (WT, VERIFIED) | abandoned (never run). **An SVM head and a class-balanced LR head have never been evaluated.**

**X25 - ANN selection bias, checkpoint and double-softmax corrections** | see bugs B1-B3 | - | TCGA / TTC / H-Optimus-1 | 2026-08-04 | selection on test vs validation: +0.0085 macro-F1 mean bias (fold 3 +0.033); softmax fix: BalAcc +0.0057, AUROC +0.0136 (one fold) | `reports/PHASE_1_REPORT.md`, `tools/ann_fixes_check.py` (WT) | conclusive (fixed; INFERRED numbers from the report).

**X26 - ANN grid widening** | Was {128,256} x {64,128} x {500} truncating | wider {128,256,512} x {64,128,256} x {500,1000} | TCGA-CV, 8 combos x 4 folds = 32 pairs, selection on validation in both arms | 2026-08-04 | probe: macro-F1 +0.0196 (Wilcoxon p=0.006), accuracy +0.0208 (p=0.0007), BalAcc +0.0115 (p=0.15), AUROC +0.0037 (p=0.63). Full pipeline: BalAcc +0.0025, AUROC +0.0020 | `slide_classification/ann_grid_probe.csv`, `tools/ann_grid_probe.py`, PHASE_3/4 reports (WT, CSV VERIFIED: 64 rows) | adopted; no significant gain on the primary metric. **max_iter = 1000 was never selected in any of the 88 fold selections** (`tcga_full_hparams.json`, VERIFIED) - that axis is inert. 5 of 22 combos had a 4-way tie across folds.

**X27 - kNN metric grid** | 'minkowski' (p=2) duplicated 'euclidean' | replace by 'manhattan' | TCGA-CV | 2026-08-04 | kNN BalAcc -0.0070, AUROC +0.0001 | `eval_patch_features/knn.py`, PHASE_3 | adopted.

**X28 - PAIP-IV ANN experiments 1-4** | Which ANN protocol on 47 slides | (1) legacy: 18-point grid selected on 9 PAIP slides, trained on 38; (2) `fixed`: TCGA-CV-voted config, refit on 47; (3) `ann_old`: 1 hidden layer 512, dropout 0.5, 1000 iters, patience 10; (4) `ann_edited`: same with dropout 0.7 | PAIP-IV, 22 combos, 4 bit-identical repeats each | 2026-08-10..12 | mean BalAcc / mean AUROC / best: 0.6921 / 0.8233 / 0.7827; 0.7166 / 0.8249 / 0.8244; 0.7321 / 0.8106 / 0.8125; 0.7255 / 0.8190 / 0.8125. ANN is the best head in 0-1 of 22 combos in every variant; legacy selection is worse than the grid median (0.6075 vs 0.6133 test macro-F1) | `slide_classification/repeats_legacy.csv`, `paip_iv_repeats.csv`, `repeats_ann_old.csv`, `repeats_ann_edited.csv` (WT, VERIFIED: means reproduced); `slide_classification/ANN_EXPERIMENTS.md`, `reports/ann_experiments/*` | conclusive-negative for legacy; the three alternatives differ by < 1/4 of a test slide. Owner decision 2026-08-13: `ann_old` (`final_models/ann_old/`). `TCGA_HP_paip_iv_repeats.csv` and `OH_paip_iv_repeats.csv` are earlier exports of experiment 2 with identical values (VERIFIED).

**X29 - PAIP-IV inner-CV threshold (tau_train)** | Fit the cut-point by Youden's J on inner out-of-fold predictions of the 47 training slides | inner folds 5 and 4; all five heads | PAIP-IV | 2026-08-12 | BalAcc change vs 0.5 (5-fold / 4-fold): lin -0.060 / -0.035; proto -0.061 / -0.057; knn -0.028 / -0.019; rf -0.032 / -0.017; ann -0.006 / +0.010. 110 rows (4-fold): 63 worse, 31 better, 16 same. tau ranges: lin 0.003-0.630, ann 0.116-0.987 | `ANN_EXPERIMENTS.md`, `reports/ann_experiments/05_threshold.md` (WT) | conclusive-negative. Do not repeat.

**X30 - Repeat-run determinism** | Are results exact | 4 repeats x 4 ANN configs; seeds 42 vs 43 for lin/knn/proto | PAIP-IV; TCGA-FULL | 2026-08-04, 08-11/12 | bit-identical repeats; lin/knn/proto identical across seeds; ANN and RF vary: mean across-seed SD on PAIP-EV ANN BalAcc 0.048 / AUROC 0.020, RF 0.029 / 0.020 (VERIFIED from live JSONs) | repeats CSVs; `TCGA_PAIP_EV_Results/**/result_*.json` | conclusive. Note the ANN seed SD (0.05 BalAcc) is larger than most between-method differences.

### 2.5 Protocol experiments (remediation)

**X31 - SurGen slide-level vs case-level folds** | Size of the leak | fold builder only | SurGen-CV, 3 combos x 5 heads | 2026-08-04 | 47-49 of 70 two-slide cases were split; mean BalAcc -0.0252, AUROC -0.0342; all 15 BalAcc deltas negative; ANN hit hardest (-0.047 / -0.048) | `reports/PHASE_2_REPORT.md`, `tools/surgen_fold_audit.py`, `runners/surgen_folds.py` | conclusive (case-level adopted).

**X32 - PAIP-CV** | 4-fold CV inside PAIP | - | PAIP (73) | 2026-02/03 | e.g. H-Optimus-1 caption LR 0.867 / 0.942 on 18-slide folds | results only in HIST (`slide_classification/PAIP_Results_ARCHIVED_20260804` at commit `e8cbb894`), `slide_classification/PAIP_best_of_all_exps_metric.xlsx` (WT), draft docx (GIT) | abandoned 2026-08-04: folds were sequential slices of the CSV and the design is dominated by PAIP-IV. Do not revive.

**X33 - EV variants: TCGA-FULL vs 4-fold probability ensemble vs 4-fold average** | Which TCGA artifact transfers best | 3 variants x 5 seeds | PAIP-EV, SurGen-EV | 2026-08-04 (73 slides), final 09-03 | final workbook, mean over 20 combos, BalAcc / AUROC: PAIP-EV ANN 0.7496 / 0.8498 (full) vs **0.7946 / 0.8592** (ensemble) vs 0.7528 / 0.8379 (average); LR 0.7431 / 0.8717 vs 0.7512 / 0.8735 vs 0.7260 / 0.8595. SurGen-EV ANN AUROC 0.7140 vs 0.7294 vs 0.6985; LR 0.7090 vs 0.7267 vs 0.7144; RF 0.6693 vs 0.7180 vs 0.6800 | workbook (VERIFIED), `reports/PHASE_3_REPORT.md` | TCGA-FULL kept as primary by decision; **the fold ensemble is equal or better on AUROC for LR, ANN and RF on both external cohorts.** Caveat: for LR/kNN/Proto/RF the fold models train on train+val (about 310 slides), not about 208 (see S3).

**X34 - Frozen tau_TCGA (Youden's J on pooled TCGA out-of-fold)** | Does a TCGA cut-point transfer | vs 0.5 and RF 0.3 | PAIP-EV, SurGen-EV | 2026-08-04 -> 09-03 | PAIP: rescues RF from zero positives, costs LR / Proto slightly. SurGen: 16 of 21 ProtoNet and 7 of 21 RF configurations dead or near-dead; mean BalAcc Proto 0.523, RF 0.557 | `runners/thresholds.py`, `thresholds_TCGA.json`; `experiments/surgen_ev_diagnostic/SUMMARY.md` (GIT) | partial: principled but fails under shift for ProtoNet / RF / kNN.

**X35 - Experiment A: threshold-health detector** | Catch saturated and near-dead / near-saturated operating points | 1 % band | PAIP, SurGen; 284 rows | ~2026-08-28..31 (INFERRED) | 5 previously "healthy" saturated cases found (e.g. SurGen Caption/ConchV1 k=50: 622/622 positive) | `experiments/threshold_and_k/SUMMARY.md`, `results/A_threshold_health.csv` (GIT, VERIFIED) | adopted (`threshold_modes.health`).

**X36 - Experiment B: refit tau per k for kNN** | - | k in {3,5,10,15,25,35,50} | PAIP, SurGen | same | PAIP k=35: AUROC 0.8114, BalAcc stale 0.596 -> refit 0.703 (oracle 0.770); k=50: 0.8158, 0.586 -> 0.719. SurGen k=25: 0.6675, 0.542 -> 0.563; k=35: 0.6533, 0.534 -> 0.571. Refit removes every dead threshold for k > 3 | `results/B_refit_tau.csv` (GIT, VERIFIED) | conclusive-positive (mandatory when k changes).

**X37 - Experiment C: rate-matched (quantile) threshold, all heads** | Cut the target at its (1-r) quantile, r = TCGA positive rate at tau_TCGA | 5 heads x 40 configs | PAIP, SurGen | same | gap to oracle closed: PAIP rf 71 %, proto 67 %, ann 60 %, lin 46 %, **knn -418 %**; SurGen proto 84 %, ann 75 %, rf 74 %, lin 61 %, knn 18 %. Unhealthy points 38 -> 10 of 200 | `results/C_quantile_threshold.csv` (GIT) | positive for LR/ANN/Proto/RF, negative for kNN. Uses unlabelled target scores (transductive).

**X38 - Experiment D1: kNN distance weighting** | uniform vs distance | - | PAIP, SurGen | same | AUROC +0.0064 (PAIP), +0.0015 (SurGen); ties partly dissolve; exact-zero scores unchanged | `results/D_weighting.csv` (GIT, VERIFIED) | conclusive-negative. Do not repeat.

**X39 - Experiment D2: in-domain k** | Is small k really optimal on TCGA | k in {3,5,7,10,15,25,35,50} on TCGA out-of-fold | TCGA | same | BalAcc at the in-sample Youden point: 0.678, 0.677, 0.640, 0.671, 0.682, 0.687, **0.697**, 0.678. TCGA out-of-fold **AUROC**: 0.694, 0.716, 0.691, 0.735, **0.743**, 0.742, 0.739, 0.728 (VERIFIED from `D_k_selection.csv`). The original grid was capped at k <= 15; fitted artifacts: k=3 in 17/20, k=7 in 2, k=5 in 1; cosine 13, manhattan 7; uniform 20/20 | `results/D_k_selection.csv`, `D_weighting.csv` (GIT) | k=35 adopted "from TCGA alone". The advantage is 0.01-0.02 BalAcc measured at an in-sample threshold; by AUROC k=15-35 are tied. Defensible, not unique.

**X40 - corrected_ev: both corrections for all heads** | kNN k=35 + refit tau; LR/ANN/Proto/RF quantile | - | PAIP-EV, SurGen-EV, 20 combos | ~2026-08-31 | BalAcc frozen -> corrected: PAIP ann 0.755 -> 0.814, knn 0.596 -> 0.703 (AUROC 0.611 -> 0.811), lin 0.743 -> 0.789, proto 0.727 -> 0.794, rf 0.661 -> 0.772. SurGen ann 0.589 -> 0.672, knn 0.565 -> 0.571 (AUROC 0.589 -> 0.653), lin 0.594 -> 0.649, proto 0.523 -> 0.653, rf 0.562 -> 0.642. 38 of 200 configurations lose BalAcc (worst: kNN on SurGen, -0.13) | `experiments/corrected_ev/results/*.csv`, `Results_Corrected_Thresholds.docx` (GIT, VERIFIED) | computed for all five heads; only kNN and RF were later promoted.

**X41 - PAIP-EV promotion** | Publish corrected kNN and RF only | `--threshold-mode promoted` | PAIP-EV | 2026-09-02 17:16, re-run 20:12 and 09-03 12:22 | kNN AUROC 0.6110 -> 0.8114, BalAcc 0.5959 -> 0.7029; RF AUROC unchanged 0.8254, BalAcc 0.6618 -> 0.7724 (VERIFIED from live JSONs) | `slide_classification/TCGA_PAIP_EV_Results`, `paip_ev_knn_rf_before_after_20260902.csv`, `logs/paip_ev_promoted_20260902.log`, archive `TCGA_PAIP_EV_Results_ARCHIVED_20260902_PRE_KNNRF` (WT) | adopted (owner decision).

**X42 - SurGen-EV diagnostic: strip + k sweep x 3 schemes** | Does any k rescue kNN on SurGen; do the schemes rank configurations alike | k in {15,20,25,35,50,75,100} x {frozen, refit, quantile} | SurGen, PAIP | 2026-09-03 | SurGen mean AUROC by k: 0.665, **0.677**, 0.668, 0.653, 0.645, 0.614, 0.550 (PAIP rises monotonically to 0.833 at k=75). Oracle BalAcc ceiling 0.63-0.64. Best scheme changes with k; Spearman between refit and quantile rankings is about -0.2 to +0.16. Part 1: 1,092 unused keys stripped, 315 headline entries unchanged; `status_frozen` (mean-prob) disagrees with the confusion-matrix status in 5 entries | `experiments/surgen_ev_diagnostic/SUMMARY.md` and `results/*` (GIT, VERIFIED) | conclusive: SurGen kNN is a representation limit, "there is no k to pick, only a k to overfit". Document said nothing was promoted.

**X43 - SurGen-EV promotion at k=20** | - | candidate k=35 (`build_candidate.py`) vs k=20 (`build_candidate_k20.py`) | SurGen-EV | 2026-09-03 13:41-13:49 | published: kNN AUROC 0.5894 -> 0.6770, BalAcc 0.5655 -> 0.5750; RF BalAcc 0.5585 -> 0.6422 (VERIFIED from live JSONs) | `slide_classification/SurGen_EV_Results`, `logs/surgen_ev_k20_promotion_20260903.log`, `experiments/surgen_ev_promotion/*` (GIT) | adopted but **methodologically weak: k=20 was selected on SurGen labels** (the same day's diagnostic and `Results_Corrected_Thresholds.docx` both argue against exactly this). k=20 was not even in the TCGA grid of X39.

**X44 - Domain-shift measurement (t-SNE + metrics)** | How far are PAIP / SurGen from TCGA | 20 combos | all | 2026-09-03 | nearest-neighbour distance ratio vs TCGA: PAIP 1.4-4.0, SurGen 1.9-7.1; label silhouette TCGA about 0.00-0.03, PAIP 0.09-0.21, SurGen 0.04-0.08 | `Analysis_and_Visualization/TSNE/tsne_shift_metrics.csv` (WT, VERIFIED), `tools/make_tsne_plots.py` | conclusive (descriptive).

### 2.6 Cohort work

**X45 - PAIP 73 -> 78** | Recover `training_data_19/30/41/42/46` | re-patch, re-extract, re-aggregate | PAIP | 2026-08-09/10 | PAIP-IV train 42 -> 47 (MSI-H 10 -> 12); PAIP-EV N 73 -> 78 (17 -> 19) | `feature_extraction/Feature_Extraction_Missing_PAIP_*.ipynb`, `slide_aggregation/Slide_Aggregation_Missing_PAIP.ipynb`, logs `run_20260810_*` (WT, VERIFIED N in logs) | done; results regenerated. `reports/COHORT_COUNTS.md` and the cohort table inside RESULTS.md still say 73 (stale).

**X46 - SurGen label reconciliation (624 -> 991)** | Why were 396 slides unlabelled | MMR-IHC + PCR-MSI evidence per slide | SurGen | 2026-09-03 -> 09-08 | 270 SR1482 cases had been dropped because MSI was "Not performed" although MMR-IHC existed; new master: 991 slides / 822 cases / 100 MSI-H (850 MMR, 141 MSI); 29 excluded | `Complete_Pipeline/surgen_slide_labels.csv`, `verify_surgen_labels.py`, `reports/surgen_*audit*.csv`, `reports/surgen_msi_split_reconciliation.csv` (WT, VERIFIED counts) | labels done. **I verified that 0 of the 622 featured slides changed label (562 / 60 identical)**; the 369 unfeatured labelled slides add 329 negatives and 40 positives.

**X47 - 991-slide encoding** | Extend features to the 369 new slides | incremental A-1/A-2, standalone conch1-5 / virchow2 Step C | SurGen | 2026-09-08/09 | 630 slides screened (622 + 8); last log line 2026-09-09 15:58 mid-download; every aggregation folder still holds 622 files | `surgen_data/patch_metadata_screened_slides.txt` (630 lines), `Complete_Pipeline/logs/pipeline_20260909.log` (WT, VERIFIED) | partial / stalled.

**X48 - Other targets (BRAF, KRAS, TP53, CIMP)** | - | - | TCGA | files 2025-08-07; CIMP dichotomy decided 2026-08-04 (CIMP-H vs rest: 54 positives, 68.5 % MSI-H) | never run: only `1-MSIH` folders exist anywhere | `slide_classification/kfold_*.csv`, `data_layer.py`, `OPEN_QUESTIONS.md` #5 (GIT) | deferred (Phase 5). SurGen does carry BRAF / KRAS / NRAS (both sub-cohorts) and 5-year survival (SR386): `Complete_Pipeline/SR1482_labels.csv`, `SR386_labels.csv` (GIT, VERIFIED) - this answers the work order's open question.

**X49 - Multi-class slide classifier scaffold** | - | - | - | 2025-08-12 | cell has no outputs | `Slide_Classification.ipynb` cell 12, `eval_patch_features/ann_multiclass.py` | abandoned (never run).

---

## 3. B. Hyperparameters

| # | Component / parameter | Values tried | Selected | Selected on | External data used? | Evidence |
|---|---|---|---|---|---|---|
| H1 | LR C, max_iter | C {0.01,0.1,1,10}, iter {300,500,1000} | C=10, 300, lbfgs, no class weight | TCGA-CV **test-fold** BalAcc, 3-4 feature sets, 2025-09 / 2026-02 (X21) | no | `misc/Test_Result/*/Output/grid_search_linear_C_maxiter.xlsx`; `runners/classifiers.py` GRIDS |
| H2 | LR class weighting / input scaling | none tried | none (raw features) | - | - | `eval_patch_features/logistic.py` |
| H3 | kNN (in-domain) | k {3,5,7,10,15} x metric {cosine, euclidean, manhattan} x weights {uniform, distance}; L2-normalised inputs | per combo by GridSearchCV; TCGA-FULL: k=3 in 17/20, cosine 13 / manhattan 7, uniform 20/20 | nested 5-fold CV on the training rows, balanced accuracy at majority vote | no | `eval_patch_features/knn.py`; `experiments/threshold_and_k/results/D_weighting.csv` (GIT) |
| H4 | kNN k for external validation | {3,5,10,15,25,35,50} on TCGA; {15,20,25,35,50,75,100} on SurGen/PAIP | PAIP-EV k=35; SurGen-EV k=20 | k=35: TCGA out-of-fold BalAcc + prevalence rule (35 x 0.145 = 5). **k=20: SurGen k-sweep** | **k=20 uses SurGen labels - risky** | `runners/threshold_modes.py`, `knn_k35_thresholds.json`, X39, X42, X43 |
| H5 | kNN tau at k=35 / 20 | Youden's J on TCGA out-of-fold at that k | per combo (20 stored for k=35; k=20 refit on the fly) | TCGA | no | `knn_k35_thresholds.json` |
| H6 | ProtoNet | none (class means of L2-normalised features, softmax of negative Euclidean distance) | - | - | - | `eval_patch_features/protonet.py` |
| H7 | RF | 500 trees, depth None, min_split 5, min_leaf 1, class_weight {0:1,1:10}; alternates in code only | as listed | by hand (2026-02), after RF predicted no positives on PAIP EV | **informed by PAIP EV behaviour (INFERRED from `test_random_forest.py`)** | `runners/classifiers.py`, `r_forest_eval.py` |
| H8 | RF threshold | 0.5, 0.3, tau_TCGA, quantile, val-tuned (code only) | internal experiments 0.3 (hard-coded); EV: quantile under `promoted`, tau_TCGA otherwise | 0.3 by hand; tau and r from TCGA out-of-fold; quantile reads target scores | **0.3 PAIP-informed (INFERRED); quantile uses unlabelled target scores** | same; S2 |
| H9 | ANN architecture | h1 {128,256,512} x h2 {64,128,256} (was {128,256} x {64,128}); 1-layer 512 presets | per fold by validation macro-F1; TCGA-FULL by majority vote over 4 folds, ties by mean validation macro-F1 | TCGA validation folds | no | `tcga_full_hparams.json`, `ann_grid_probe.csv`, `runners/hparams.py` |
| H10 | ANN max_iter | {500,1000}; notebook {100,200,350,500,1000} | 500 in 88/88 fold selections | validation | no | `tcga_full_hparams.json` |
| H11 | ANN dropout / depth / patience | 0.3 (2 layers, patience 20); 0.5 and 0.7 (1 layer, patience 10) | 0.3 / 2 layers everywhere; `ann_old` (0.5, 1 layer) declared final for PAIP-IV | collaborator defaults; compared on the **PAIP-IV test set** (X28) | **the choice among the 4 PAIP-IV protocols was made looking at PAIP test metrics** | `eval_patch_features/ann.py`, `runners/iv_runner.py`, `final_models/ann_old/README.md` |
| H12 | ANN optimiser | Adam lr 1e-4, weight decay 1e-4, ReduceLROnPlateau(0.3, 5), full batch, BatchNorm, CE on logits | never varied | - | - | `ann.py` |
| H13 | ANN selection metric | test macro-F1 (bug) -> validation macro-F1 | validation macro-F1 | - | no | `runners/classifiers.py` |
| H14 | ANN early-stop hold-out | 15 % stratified of 413 (TCGA-FULL, 351 + 62); 20 % of 47 (PAIP-IV, 9 slides); validation fold in CV | as listed | fixed | no | `full_trainer.py`, `iv_runner.py` |
| H15 | Seeds | 42 (CV, IV); 42-46 (TCGA-FULL) | - | - | - | `full_trainer.py` |
| H16 | Internal decision threshold | 0.5 (argmax) for LR/kNN/Proto/ANN; RF 0.3 | - | - | - | S2 |
| H17 | External decision threshold | 0.5; tau_TCGA (Youden's J, pooled out-of-fold); refit-at-k; quantile; inner-CV tau_train (PAIP-IV) | frozen tau_TCGA for LR/ANN/Proto; promoted schemes for kNN/RF | TCGA out-of-fold | quantile reads target scores, not labels | `runners/thresholds.py`, `threshold_modes.py`, `thresholds_TCGA.json` |
| H18 | tau_train inner folds | 5, 4 | rejected | PAIP training slides | PAIP train only | X29 |
| H19 | TCGA folds | `kfolds_IDARS_fixed.csv`, 4 folds 103/98/112/103 (413: 103/97/112/101); rotation test=i, val=i+1 | fixed | provenance of the split unknown (name suggests the IDARS paper; INFERRED) | no | `kfolds_IDARS_fixed.csv` |
| H20 | Effective training fraction | ANN trains on 2 folds (about 50 %); LR/kNN/Proto/RF merge train+val (about 75 %) | - | - | - | S3 |
| H21 | SurGen folds | slide-level round-robin (leaky) -> case-level stratified round-robin, seed 42, K=4 | case-level | - | - | `runners/surgen_folds.py` |
| H22 | PAIP split | PAIP-CV sequential folds (retired) -> provider 47 / 31 | provider split | - | - | `iv_runner.py` |
| H23 | EV variants | tcga_full, fold_ensemble, fold_average | tcga_full primary | owner decision before results | no | `ev_runner.py` |
| H24 | Bootstrap | 1000 percentile resamples, slide level, seed 42 (PAIP-IV only) | - | - | - | `iv_runner.py` |
| H25 | Aggregation rows | Averaging 1; TTC 9; caption 14; caption 15; TITAN 1; PRISM 1 | all reported | - | - | `config/paths.py` |
| H26 | Caption assignment | 13 prompt templates x synonyms per class; top-1 caption per patch; mean per class | 14-class (2025-09) and 15-class (2026-06) | prompts "validated by clinical experts" (INFERRED from draft) | no | `caption_generation/config_files/*` |
| H27 | Class subset (top-k) | 9-class k {3,5,7,8}; 14-class k {3,4,5,7,9,10,14}; tiers 0.65 / 0.58 | none adopted (all classes used in the main tables) | TCGA test folds | no | X17-X19 |
| H28 | Tissue classifier head | h1 {128,224,256} x h2 {64,128} x C {0.1,1,10}, 500 iters | C=10, h2=128, h1=256 (224 Conch1.5) | 20 % validation split of CRC-100K | no | X08 |
| H29 | Patch size / magnification | 512 px at 20x (1024 at 40x resized) | fixed | - | - | X01 |
| H30 | Crops | single resize (224 or 512), five-crop 256, ten-crop | five-crop | no MSI ablation | - | X04-X06 |
| H31 | Background filter | TCGA: linear SVM (avg_G, avg_B, std_R); PAIP: RBF SVM + scaler; SurGen: pixel rules (black > 0.20, white > 0.90, mean > 210 with std < 12) + the TCGA SVM; unified RBF SVM for the 15-class path | as listed | hand-labelled patches per cohort | no | X02, X03; v6 pipeline (GIT) |
| H32 | Feature scaling before heads | none for LR/RF/ANN; L2 for kNN/Proto | - | never varied | - | `runners/model_io.py` |
| H33 | Encoders | ResNet-18 (dropped), Conch1.5, UNI2, H-Optimus-1, Virchow2, ConchV1, PRISM, TITAN | 5 patch encoders + 2 slide encoders | - | - | - |
| H34 | Few-shot tissue probe | 20 or 100 per class; LR / NC / ensemble | not adopted | CRC-100K | no | X10, X11 |
| H35 | Health band | 1 % of cohort for near-dead / near-saturated | 1 % | by reasoning | no | `threshold_modes.py` |

Choices that used PAIP or SurGen information (methodologically risky): **H4 (SurGen k=20)**, **H11 (PAIP-IV ANN protocol chosen after looking at PAIP test results)**, H7/H8 (RF class weight and 0.3 threshold set after a PAIP-EV failure, INFERRED), the decision to promote only kNN and RF (made after seeing EV results), and every "best configuration" row (maximum over 100+ configurations on the reported data). The quantile rule (H17) uses unlabelled target scores.

---

## 4. C. Bugs found and fixed

| # | Bug | Measured impact | Fix | Results regenerated? | Evidence |
|---|---|---|---|---|---|
| B1 | ANN hyperparameters selected on the test fold | +0.0085 macro-F1 mean optimistic bias (one combo; fold 3 +0.033) | select on validation macro-F1 | yes (TCGA-CV 08-04; all later runs). **Not regenerated: TTC / best-k results (X17-X20)** | PHASE_1_REPORT, `runners/classifiers.py` |
| B2 | ANN checkpoint written inside the grid loop (last config 256/128 on disk) | PAIP-EV ran a different network than reported | save once after selection + `config.json` | yes | PHASE_1 |
| B3 | Double softmax (Softmax + CrossEntropyLoss) | BalAcc +0.0057, AUROC +0.0136 after fix (one fold) | logits in graph, softmax in `predict_proba` | yes; TTC / best-k not | PHASE_1, `ann.py` |
| B4 | Labelled slides without features silently dropped | reported N wrong (416 vs 413; 78 vs 73; 624 vs 622) | warn + `missing_slides.csv` | n/a (N corrected) | PHASE_1 |
| B5 | Undefined `label` in BRAF/KRAS/TP53 branches (swallowed NameError; later slides inherit the previous label) | latent | dict lookup in `data_layer` | n/a | PHASE_0/1 |
| B6 | Unseeded `DataLoader(shuffle=True)` -> RF not reproducible | RF BalAcc +0.0036 after fix (TCGA-CV) | sorted row order, seeded | yes | PHASE_1/3 |
| B7 | SurGen folds leaked 47-49 of 70 two-slide cases | BalAcc -0.025, AUROC -0.034 after fix | case-level folds | yes (SurGen-CV); SurGen best-k not | PHASE_2 |
| B8 | PAIP-CV folds = sequential CSV slices | experiment invalid | retired | n/a | PHASE_2 |
| B9 | `paip_kfolds_71.csv` misnamed; `paip_78slides_labels.csv` malformed | metadata | renamed / regenerated | n/a | PHASE_2 |
| B10 | kNN grid 'minkowski' duplicated 'euclidean' | kNN BalAcc -0.0070 | 'manhattan' | yes | PHASE_3 |
| B11 | joblib loky worker crash on Windows under CUDA | sweep killed | threading backend | - | `knn.py` |
| B12 | `class_weight` int keys lost in JSON round-trip | TCGA-FULL RF crashed | `_restore_json_types` | - | `hparams.py` |
| B13 | Windows refused archive rename | - | verified-copy fallback | - | PHASE_3 |
| B14 | Old report builder globbed `*_Results` including archives | would have mixed old and new numbers | read stamped JSON, skip `_ARCHIVED_` | - | PHASE_4 |
| B15 | PAIP: 5 slides never patched in the pipeline tree | PAIP 73 -> 78 | recovered from `D:/Aamir Gulzar/dataset/paip_data` | yes (08-10) | X45 |
| B16 | PAIP-IV ANN: 18 configs selected on 9 slides and trained on 38 | selection worse than grid median | `--ann-protocol fixed/full`, presets | **partly undone: the live tree was re-run with the default `legacy` on 08-31** | S1 |
| B17 | kNN rank collapse (k=3 -> 2-4 distinct scores) and grid cap at 15 | AUROC PAIP 0.611 -> 0.811, SurGen 0.589 -> 0.677 | k override + refit tau (EV only, `promoted`) | yes for PAIP-EV / SurGen-EV `tcga_full`; fold variants and internal experiments unchanged | X36-X43 |
| B18 | Stale tau applied at a different k | dead / saturated points | refit per k | yes | X36 |
| B19 | Health check missed saturation | 5 hidden cases | `health()` with near band | yes | X35 |
| B20 | `--knn-k` accepted but the k=35 tau store was still read | wrong tau for k != 35 | gate the lookup on k (2026-09-03) | yes (k=20 promotion ran after) | `ev_runner.py` |
| B21 | `status_frozen` from seed-averaged probabilities vs metrics from per-seed confusion matrices | 5 entries disagree | both reported in the diagnostic copy; published tree not changed | no | X42 |
| B22 | Diagnostic recomputed ANN / RF AUROC from seed-mean probabilities | inflated by about 0.023 (ANN) and 0.033 (RF) vs published mean-of-seeds | promotion script reuses `_promote` | n/a (diagnostic only) | `build_candidate_k20.py` (GIT) |
| B23 | SurGen H-Optimus-1 Averaging / TTC aggregated only 613 slides | two combos on a different denominator | re-aggregated to 622, re-run 09-03 | yes | `verify_transfer.py`, archive `..._PRE_HOPT622` |
| B24 | `ReduceLROnPlateau(verbose=...)` rejected by newer torch | ANN crashed (08-27, 09-03) | kwarg removed 09-03 11:01 | yes (SurGen-CV 11:05) | logs, `ann.py` mtime |
| B25 | Subset re-run archived the whole tree (empty PAIP-IV radar) | missing cells drawn as 0.0 | `--no-archive` flag | yes | `cv_runner.py`, Results_Corrected_Thresholds s5 |
| B26 | 270 SR1482 cases dropped because MSI "Not performed" although MMR-IHC existed | 624 instead of 991 labelled slides | 991 relabel | **no - all SurGen results still on 622** | X46 |
| B27 | Hard-coded Hugging Face tokens (22 files) | security | env variable | purge done 08-04; a token was re-introduced in 3 files and removed again on 2026-10-05 (S4); **revocation still pending** | PHASE_0 |
| B28 | TITAN dense ALiBi bias (22.6 GB for 22.5 k patches) | out of memory | tiled attention, verified equivalent | n/a | `Complete_Pipeline/surgen_titan_process.md` (GIT) |
| B29 | Stale `external_validation_script.py` (wrong ANN shape), dead `dataloader.py`, `test_saved_ann_model` wrong architecture | latent | retired to `misc/` / rewritten | n/a | PHASE_1 |
| B30 | Radar legend built from last panel; title collisions | figures | fixed | yes | PHASE_4 |

Known but **not fixed**: `Taiga_Paip_inference.ipynb` invalid JSON; `Slide_Classification.ipynb` cell 10 does not parse; 23 upstream scripts with hard-coded roots; RF threshold mislabelled (S2).

---

## 5. D. Open items

| # | Item | Source | Status on disk |
|---|---|---|---|
| O1 | Revoke the 3 Hugging Face tokens | `TOKENS_TO_REVOKE.md` (GIT), OPEN_QUESTIONS #3 | **open** - revoke them on Hugging Face (owner action). The hard-coded copies were removed from the code on 2026-10-05 (S4) |
| O2 | Which ANN is the published PAIP-IV configuration | `final_models/ann_old/README.md` vs RESULTS.md vs Supplementary docx | **contradiction** (S1): live tree = legacy |
| O3 | Re-run all SurGen results on the 991-slide labels | README s9 | open; encoding stalled at 630 screened / 622 featured |
| O4 | "[TBD: confirm how many of the 622 changed label]" | README | **resolved by this audit: 0** |
| O5 | "[TBD: cite the published SurGen benchmark ... 991 / 822]" | README | open; reconciliation with the published split exists (`reports/surgen_msi_split_reconciliation.csv`: 549 agree, 1 disagree `SR1482_T412`, 270 locally excluded) but predates the relabel |
| O6 | "[TBD: PAIP total patch count]" | README | resolvable: 275,541 patches in `classification_results_paip.csv` (VERIFIED row count); draft says about 267 k after filtration before the 5 slides were added |
| O7 | "[TBD: confirm on a clean machine]" end-to-end re-run; v6 pipeline absent from working tree | README | open for the clean-machine check; the v6 script was restored to the working tree on 2026-10-05 |
| O8 | Server run for SurGen H-Optimus-1 / UNI2 / ConchV1 | `SERVER_RUN.md` (GIT), PHASE_3 | **resolved differently**: features were transferred 08-27 and everything ran locally (21 combos) |
| O9 | SurGen PRISM classification | PHASE_3 ("no PRISM features") | embeddings now exist (622) - **classification never run** |
| O10 | SurGen-CV `Caption_15` / TITAN coverage | work order Task 3.3 | resolved (in the 21 combos) |
| O11 | 23 upstream scripts with hard-coded paths | OPEN_QUESTIONS #4 | open |
| O12 | CIMP dichotomy | OPEN_QUESTIONS #5 | resolved (CIMP-H vs rest); never run |
| O13 | Phase 5: BRAF / KRAS / TP53 / CIMP | work order | not started; SurGen labels for BRAF / KRAS / NRAS confirmed to exist (GIT only) |
| O14 | Tissue-classifier metric attribution | work order Phase 5, review doc Issue 9 | **resolved by this audit**: the row is ResNet-18/224 |
| O15 | Regenerate TTC / best-k experiments after the ANN and fold fixes | PHASE_1, PHASE_2 | open (files dated 2026-03..06) |
| O16 | `Taiga_Paip_inference.ipynb` malformed | PHASE_2 | open |
| O17 | Two heads / two cohorts use different threshold schemes (PAIP k=35, SurGen k=20) | README s9 | open; README flags it |
| O18 | Corrected values for LR / ANN / ProtoNet exist but are not reported | `experiments/corrected_ev` | open decision |
| O19 | `progress.json` not updated after 2026-09-03; stamps carry commit `e8cbb89` for all 08-10..09-03 runs | - | open (provenance gap, S5) |
| O20 | Empty tissue rows (all-zero) indistinguishable from "absent" (lymphocyte empty in 109 / 417 TCGA slides) | work order known issue 3 | open; never tested (e.g. an indicator feature) |
| O21 | "Published before 2026-08-04" numbers may have been circulated (PAIP-CV, leaky SurGen) | PHASE_2 | unknown what was circulated |
| O22 | Where "409" TCGA slides came from | COHORT_COUNTS | unknown |
| O23 | Fold-file provenance (`kfolds_IDARS_fixed.csv`, "fixed" how?) | - | unknown |
| O24 | `conch1-5` SurGen feature tree missing locally | this audit | **resolved 2026-10-05** (restored, S6) |
| O25 | PRISM SurGen patch-loss audit before use | owner memory note 2026-08-26 | unknown whether the final 622 passed |
| O26 | Stale documents: `reports/COHORT_COUNTS.md` (PAIP 73), RESULTS.md policy text, `Supplementary_Results_UPDATED.docx` (SurGen-EV "partial coverage", "frozen") | - | open |

---

## 6. E. Not yet tried - suggestions, ranked for a project that must finish quickly

All items below are **suggestions**. "Already tried?" states what I searched for.

| Rank | Idea | Already tried? (evidence searched) | Expected benefit | Cost | GPU / server? |
|---|---|---|---|---|---|
| N1 | Uncertainty and significance for every headline number: case-level bootstrap CIs (SurGen has 70 two-slide cases), paired bootstrap or DeLong for "caption vs averaging" and encoder comparisons, on stored predictions | Only PAIP-IV has slide-level bootstrap CIs. One ad hoc SurGen case-level CI is quoted in an owner memory note (0.851 [0.796, 0.897]) with no script or file. grep for DeLong / bootstrap in code: none outside `iv_runner` | Turns best-of-100 point estimates into defensible claims; needed by any reviewer | 0.5-1 day | no (predictions are in `result_*.json` and `oof_predictions_default.csv`) |
| N2 | One uniform operating-point rule on both external cohorts (k=35 on both; quantile for all heads or for none) and report AUROC first | corrected values for all heads already exist (`experiments/corrected_ev`, GIT) | Removes the target-informed k=20 and the selective promotion | hours | no |
| N3 | PRISM on SurGen (CV and EV) | never run; 622 embeddings exist | completes the baseline table | minutes (after the patch-loss audit) | no |
| N4 | Probability ensembling: seeds, folds, encoders, aggregations (late fusion, rank averaging), members chosen on TCGA only | fold ensemble exists and wins (X33); seed-mean probabilities raise SurGen AUROC by 0.02-0.03 (B22); no cross-encoder or cross-aggregation ensemble anywhere (grep Voting / Stacking: none) | likely +0.01-0.03 AUROC externally, less seed noise | hours | no |
| N5 | Tumour-restricted / top-k class aggregation inside the corrected runners, selected by nested CV on TCGA, then PAIP-EV and SurGen-EV | pre-fix only (X17-X20); never external | pre-fix gains of +0.03-0.05 BalAcc on TCGA and SurGen; also shrinks dimensionality | 0.5-1 day (row slicing of existing tensors) | no |
| N6 | Re-tune LR with nested CV on AUROC: C grid, class_weight balanced, StandardScaler or PCA before LR | C grid only on test folds (X21); no class weighting, no scaling, no PCA before heads (grep) | X21 shows up to +0.025 AUROC for UNI2 at smaller C; inputs have up to 38,400 dimensions for about 300 training slides | hours | no |
| N7 | Unsupervised target alignment in feature space (per-cohort standardisation, CORAL) | not tried; only the score-level quantile rule | targets the 2-7x nearest-neighbour distance shift that breaks kNN / ProtoNet | hours | no |
| N8 | Calibration (Platt / isotonic on TCGA out-of-fold; Brier, ECE) and a fixed-sensitivity operating point (specificity at 90-95 % sensitivity) | none (grep Calibrated / isotonic / brier: none); only Youden, quantile and inner-CV thresholds | clinically meaningful rule-out metric; fixes the LR under-confidence (tau 0.006-0.246) | hours | no |
| N9 | Case-level SurGen evaluation and sub-cohort breakdown (SR386 primaries vs SR1482 with metastatic sites; MMR-labelled vs PCR-labelled) | none (grep case-level metrics: none) | explains the weak SurGen transfer; case-level is the clinically correct unit | hours | no |
| N10 | Evaluate on the published SurGen train / validate / test split | split is in `reports/surgen_msi_split_reconciliation.csv`; never used for modelling | comparability with the SurGen benchmark | hours | no |
| N11 | Multi-source training (TCGA + PAIP -> SurGen; TCGA + SurGen-train -> SurGen-test) | none | quantifies how much of the SurGen gap is domain shift | hours | no |
| N12 | Attention-based MIL (ABMIL / CLAM / TransMIL) on cached patch embeddings | never trained (only vendored CLAM code and the X16 prototype) | the standard baseline a reviewer expects; tests the central claim against learned pooling | 2-4 days; patch features exist for TCGA and PAIP (`dataset/Features`, `paip_data/Features`), SurGen patch features only partly local | one GPU for training, no re-encoding |
| N13 | Finish the 991-slide SurGen encoding (at least UNI2 and H-Optimus-1) | stalled at 630 screened | +40 MSI-H positives (60 -> 100), tighter CIs | days to weeks at earlier throughput | yes (GPU; several encoders ran on the server) |
| N14 | Regenerate TTC / best-k and SurGen best-k with the corrected code | open since 08-04 | needed if these figures stay in the paper | 0.5 day | ANN on GPU, small |
| N15 | TITAN / PRISM concatenated or late-fused with aggregation features | none | small; covered by N4 | hours | no |
| N16 | Empty-class handling in TTC / caption tensors (presence indicator, class proportions as extra features) | none | cheap, addresses O20; class proportions alone may carry signal | hours | no |
| N17 | BRAF / KRAS external validation TCGA -> SurGen | labels exist, never run | second task for the paper | 1 day | no |
| N18 | Stain normalisation or stain augmentation | none (`torchstain` only in an old requirements file) | uncertain with foundation models; N7 is the cheap proxy | re-extraction of all features | yes |
| N19 | Single-crop vs five-crop ablation; 40x or multi-scale features | none | low | re-extraction | yes |

---

## 7. Surprises and contradictions between documents and disk

- **S1 - PAIP-IV ANN.** `final_models/ann_old` says ann_old is final (2026-08-13); RESULTS.md (GIT) says the ANN is fitted on all 47 slides with the TCGA-CV configuration (a hard-coded sentence in `tools/write_results_md.py` line 49); `Supplementary_Results_UPDATED.docx` (GIT, 09-03) says legacy is the published configuration; the **live tree, the workbook and the README numbers are legacy** (re-run 2026-08-31 with default flags; stamp `ann_protocol: legacy`; ANN trained on 38 slides). The ann_old source tree `PAIP_IV_Results_ARCHIVED_20260812_3` no longer exists; the 22 weights survive in `final_models/ann_old/weights`.
- **S2 - RF threshold.** In TCGA-CV, SurGen-CV and PAIP-IV the RF confusion matrices are at 0.3 (VERIFIED: predicted positives equal n(prob >= 0.3), e.g. TCGA Averaging/UNI2 25 vs 1 at 0.5), while documents say "threshold 0.5" and `summary_iv.csv` writes `Threshold = 0.5` on the RF row.
- **S3 - Training fraction.** Stamps and reports say "training uses 50 % of the cohort". True only for the ANN. LR, kNN, ProtoNet and RF merge the validation fold (`combine_trainval=True`) and train on about 75 % (about 310 TCGA slides). The "413 vs about 208 slides" explanation of TCGA-FULL is therefore wrong for four of five heads.
- **S4 - Token.** A Hugging Face token had been hard-coded again in three tracked scripts. On 2026-10-05 it was replaced by a lookup of `HF_TOKEN` in the environment or a local `.env`. Every token that was ever committed stays readable in git history, so all of them must be revoked.
- **S5 - Provenance stamps.** Every result produced 2026-08-10..09-03 carries `git_commit e8cbb89`, but `runners/threshold_modes.py`, the ANN protocols and the promotion code were not committed until 2026-10-02. `__pycache__` holds cpython 3.10, 3.11 and 3.12 files, so runs came from at least three environments.
- **S6 - conch1-5 SurGen features.** At the time of the audit `surgen_data/surgen_processed/` held only conch-v1, h-optimus-1, prism, uni2-h and virchow2: the conch1-5 tree (patch features, four aggregations and the TITAN embeddings) had been deleted on 2026-09-21. **Resolved 2026-10-05:** the tree was restored from the Recycle Bin to its original path. Lesson: `slide_classification/cache/` and `surgen_titan_features.pkl` were for two weeks the only usable copies of those matrices, so the cache is not safely "regenerable".
- **S7 - Aug-04 archives.** `PAIP_Results_ARCHIVED_20260804` (the retired PAIP-CV), `TCGA_Results_ARCHIVED_20260804*`, etc. were removed from tracking in `19d6d45e` and are not on disk; they exist only in history (e.g. `e8cbb894`).
- **S8 - k=20.** The same-day diagnostic says "no k is selected, nothing is promoted" and the corrected-thresholds report defends not selecting k on SurGen; the published SurGen-EV table nevertheless uses k=20.
- **S9 - Background SVM for SurGen** is the TCGA linear model (md5 identical), not a SurGen or unified model as the draft states.
- **S10 - 14 vs 15 caption classes** differ in patch list and prompts as well as in class set.

## 8. What I could not check

- The server (`/media/dp-psau/...`): any results that exist only there (e.g. `Surgen_TTC_exp`).
- Whether the 622 PRISM SurGen embeddings all pass the patch-loss audit.
- Which numbers were circulated outside before 2026-08-04.
- Model files were not unpickled; hyperparameters of saved models are taken from JSON side-cars and CSVs. One exception to "listing only": I scanned the bytes of two SVM pickles for class names (no unpickling) to establish that `Complete_Pipeline/svm_model.pkl` is a bare linear SVC.
- The notebooks `Slide_Classification_best_k_labels_exp_updated.ipynb` and `Slide_Classification_TTC_exp.ipynb` were read through their result files and the existing `PIPELINE_AUDIT.md`, not cell by cell.
- Folders outside `KSA_project2` (`D:/Aamir Gulzar/dataset` with IDARS / CAIMAN / baseline features, `existing_approaches`, `WSI_Classification`) were listed at top level only; they look like a separate, earlier comparison of published baselines and were not audited.
