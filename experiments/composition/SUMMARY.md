# Composition-aware aggregation (TCGA-CV, PAIP-IV, PAIP-EV)

Status, 2026-10-19: **complete for TCGA and PAIP**, 5 encoders × Caption-14 / Caption-15 / Tissue-Type Clustering × 5 heads (75 paired configurations per setting and comparison).

**Closed, 2026-10-22 (project owner): a negative result, not continued.** The slide features (`features/`, 1.1 GB) and runs (`runs/`, 15 GB) were deleted and the runs removed from the MLflow store. Code, this summary and `results/` (per-configuration metrics and paired contrasts) are kept; everything can be regenerated with the scripts below.

## Question

The grouping aggregations keep each group's **mean feature** but lose **how much of the slide** each group covers: a group of 3 patches and one of 30,000 patches enter the classifier with equal weight, and an empty group is a row of zeros. MSI-H colorectal cancer is linked to tissue composition (more tumour-infiltrating lymphocytes, mucin, poor differentiation).

- **Does adding each group's share of the slide to the group features help?**
- **How far do the shares get on their own?**

## Design

- **Share.** For each slide and group: the fraction of the slide's grouped units in that group.
  - Caption: patches with a published caption label.
  - TTC: crops after the published labelling rule (top-left; per crop for the 5 gap-filled PAIP slides).
- **Appended block.** √share (Hellinger transform; the block has length 1) × s, where s = the median L2 norm of a non-empty group row in that arm's TCGA slide features (`01_scales.py`, label-free, TCGA only; e.g. UNI2 10.2, Virchow2 34).
  - The composition therefore carries the weight of one typical group row.
  - Raw shares would be ignored next to G × D embedding values: kNN and proto L2-normalise the whole vector, and LR is L2-regularised.
  - One fixed rule, no tuning.
- **Arms** (`02_build_slide_features.py`, `build_slide_features.py --composition-out`):

  | Arm | Classifier input | Paired base |
  |---|---|---|
  | comp_fivecrop | published five-crop group features + shares | published five-crop |
  | comp_nocrop | no-crop group features + shares | no-crop |
  | comp_only | shares alone (14 / 15 / 9 numbers), no embeddings | published five-crop |

- **Only the shares are new.** Every rebuilt slide's group features were compared with the base arm's slide tensor and must match (< 1e-4). This check is repeated after every resume, so a crash between a build and its check cannot leave unverified files.
  - TCGA-3L-AA1B (H-Optimus-1, Caption-14 and TTC) differs by 1.5e-4 (cosine 0.9999998). Its published file predates a 2025-10-08 re-extraction of 1,077 of its patch features, so the published group features are used exactly, with the shares appended.
- **Runs and comparison.** The published runners through the redirect wrapper (`ARM=comp_fivecrop / comp_nocrop / comp_only`; the loader's expected width is G·D + G, or G). Every result file was scanned for crash damage (`03_scan_runs.py`). The same paired bootstrap and operating points as the earlier studies (`04_compare.py`).

## Results

Mean AUROC over 75 configurations; difference = with shares − without [95% CI], p.

| | TCGA-CV (413) | PAIP-IV (31 test) | PAIP-EV (78, external) |
|---|---|---|---|
| **Five-crop + shares** vs five-crop | 0.788 → 0.793, **+0.004** [+0.001, +0.008], p = 0.026 | 0.845 → 0.843, −0.002, p = 0.60 | 0.856 → 0.854, −0.002 [−0.008, +0.003], p = 0.47 |
| **No-crop + shares** vs no-crop | 0.811 → 0.814, +0.003, p = 0.14 | 0.825 → 0.823, −0.002, p = 0.53 | 0.853 → 0.851, −0.002, p = 0.46 |
| **Shares only** vs five-crop | 0.788 → 0.753, −0.035, p = 0.14 | 0.845 → 0.645, **−0.200**, p = 0.001 | 0.856 → 0.740, **−0.116**, p = 0.013 |

Balanced accuracy follows the same pattern: five-crop + shares +0.002 / −0.006 / +0.002, all n.s.

- **By aggregation (five-crop + shares, AUROC):** Caption-15 +0.007 on TCGA-CV (p = 0.014); every other aggregation × setting within ±0.009 and not significant.
- **By encoder:** Conch1_5 +0.010 on TCGA-CV (p = 0.013); externally all encoders within ±0.008.
- **By head:** the small TCGA-CV gain is mostly proto (+0.012 five-crop, +0.008 no-crop) and LR (+0.005 five-crop); kNN gains nothing (−0.000 / −0.001). On PAIP-EV every head is within ±0.008.

Best configurations, AUROC:

| | Five-crop | Five-crop + shares | No-crop | No-crop + shares | Shares only |
|---|---|---|---|---|---|
| TCGA-CV | H-Optimus-1 / Caption-15 / LR 0.928 | same config, 0.925 | H-Optimus-1 / Caption-15 / LR 0.921 | UNI2 / Caption-15 / LR 0.917 | Conch1_5 / Caption-14 / ANN 0.826 |
| PAIP-EV | UNI2 / Caption-14 / ANN 0.912 | H-Optimus-1 / Caption-15 / ANN 0.910 | Virchow2 / Caption-15 / ANN 0.905 | same config, 0.903 | Conch1_5 / Caption-14 / RF 0.804 |

## Reading

- **Appending the composition gives no meaningful benefit.**
  - Within TCGA it adds +0.003 to +0.004 AUROC. That is significant for five-crop (p = 0.026), but tiny and uncorrected across the several comparisons made.
  - Externally (PAIP-EV) it is −0.002 for both arms.
  - The best configurations do not change.
- **Composition carries real MSI-H signal, but the embeddings already contain it.**
  - Shares alone reach AUROC 0.83 within TCGA and 0.80 on external PAIP (best configurations), well above chance. So tissue proportions predict MSI-H.
  - They stay far below the group embeddings, especially externally (−0.12 mean), and adding them to the embeddings changes nothing. The foundation-model group features already encode what the proportions offer.
- **Composition-only generalises worst.** It drops from TCGA-CV to PAIP far more than the embeddings do. Zero-shot / tissue-classifier label distributions likely shift between cohorts (staining, scanner, tissue sampling).
- **For the benchmark,** the published aggregations (group means, no composition) stay the reference. This and the crop-grouping study show that changing how groups are formed or weighted barely moves results; differences between encoders and heads are much larger.

## Limitations

- **One fixed scale for the appended block.** A learned weighting (e.g. attention over groups) could use composition differently. That is a different model class, outside this benchmark's fixed heads.
- **Shares come from hard labels.** Soft (probability-weighted) shares remain possible from the stored crop similarities (`KSA_project2/crop_captions/`).
- **Multiplicity.** Per-configuration p-values are uncorrected; the setting / aggregation / encoder means are the summary results.
- **Composition-only ANN on PAIP-IV is degenerate.** AUROC 0.50 from 15 inputs and 47 training slides.
- **Scope.** One seed; SurGen not yet included.

## Run notes

The workstation froze or crashed repeatedly during this run: Kernel-Power 41 resets on 17 Oct (10:12, 11:24, 13:47) and 19 Oct (11:06, 12:39), a freeze from 17 Oct ~21:53 to 19 Oct, and many 0xC0000005 access violations. The hardware is an i9-13900K on BIOS F22 (Dec 2022). Every step resumed from verified outputs:
- Slide features were re-checked against the reference after each resume.
- Result files were scanned for zero-filled files. One TCGA-FULL combination (comp_fivecrop, Caption-15 / Virchow2) had 3 zero-filled files from the 19 Oct 12:38 crash. It was moved to `damaged/` and regenerated.

## Files

- `01_scales.py`: `scales.json`, scale of the appended block per arm × encoder × aggregation.
- `02_build_slide_features.py`: composition slide features into `features/<arm>/<cohort>/{append,only}/`, verified against the base slide tensors.
- Step 3: `../nocrop_vs_fivecrop/04_run_nocrop_pipeline.py --arm comp_fivecrop|comp_nocrop|comp_only --methods Caption_based_aggregation,Caption_based_aggregation_15_classes,Tissue_Type_Clustering` into `runs/<arm>/`.
- `03_scan_runs.py`: crash-damage scan of `runs/<arm>`.
- `04_compare.py`: writes `results/per_config.csv` and `results/contrasts.csv` (comparisons comp-fivecrop, comp-nocrop, only-fivecrop).
- `05_mlflow_table.py`: writes `results/mlflow_table.csv`.
- **MLflow:** the runs (arms `comp_fivecrop`, `comp_nocrop`, `comp_only`) were removed from the clean store on 2026-10-22; `results/mlflow_table.csv` still holds them in the store's schema.
