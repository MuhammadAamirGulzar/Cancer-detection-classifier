# Final analysis: one threshold rule, confidence intervals, pre-specified configuration

Updated 2026-10-23: **crop-level grouping, TCGA and PAIP.**

- **Crop-level grouping.** Caption-14, Caption-15 and Tissue-Type Clustering now use the five-crop features with every crop grouped by its own label (`experiments/crop_grouping`; owner's decision of 2026-10-22). Their predictions come from `experiments/crop_grouping/runs/cropgroup`, made with the same runners, folds and threshold rule. Averaging, TITAN and PRISM come from the published trees as before.
- **SurGen is left out** until the 991-slide set is encoded. The earlier 622-slide SurGen results used patch-level captions (TTC was already per crop). `02_uncertainty.py --with-surgen` adds them back.
- **The previous patch-level analysis** (2026-10-05, all five experiments) is reproduced exactly by `02_uncertainty.py --grouping patch --with-surgen` and is in the git history (commit `5fd2fa84`). Run on TCGA and PAIP, `--grouping patch` matches it on all 330 configurations and 90 contrasts.

## What was done

1. **One threshold rule for external validation** (`01_uniform_ev_scores.py`). Every head uses the `corrected` scheme of `runners/threshold_modes.py`: kNN at k = 35 with tau refitted on TCGA out-of-fold probabilities, and a rate-matched (quantile) threshold for LR, ANN, ProtoNet and RF. Per-slide scores are written out so that intervals can be computed. The crop-level runs used the same rule through the same function.
2. **Confidence intervals and paired tests** (`02_uncertainty.py`) for TCGA-CV, PAIP-IV and PAIP-EV. The bootstrap is case-level and class-stratified, 2,000 resamples, using the same resampled cases for every configuration so that differences are paired.
3. **The configuration TCGA-CV selects**, evaluated on PAIP, as the unbiased counterpart of "best of 100".

`validate_against_published.py` checks `01_uniform_ev_scores.py` against the published trees. Everything agreed to within rounding (largest probability difference 5e-7), in the environment the TCGA models were trained in (scikit-learn 1.7.0, torch 2.2.0). The crop-level per-configuration AUROCs here agree with `experiments/crop_grouping/results/per_config.csv` to 5e-7.

## Findings

### 1. The uniform rule removes every dead operating point

Under the published frozen thresholds, 38 of 200 external operating points (PAIP and SurGen, 20 aggregation combinations × 5 heads) flagged nobody, everybody, or within 1% of either; under the uniform rule none does (analysis of 2026-10-05). On PAIP, mean balanced accuracy rose for every head: LR 0.743 → 0.789, ANN 0.750 → 0.814, kNN 0.596 → 0.703, ProtoNet 0.727 → 0.794, RF 0.662 → 0.772.

### 2. Semantic aggregation beats averaging on PAIP, but not within TCGA

Mean AUROC over 5 encoders × 5 heads:

| Experiment | Averaging | Caption-14 | Caption-15 | TTC |
|---|---|---|---|---|
| TCGA-CV | 0.773 | 0.792 | 0.790 | 0.775 |
| PAIP-IV | 0.760 | 0.869 | 0.832 | 0.837 |
| PAIP-EV | 0.818 | 0.865 | 0.862 | 0.838 |

Paired difference with 95% interval and two-sided bootstrap p-value:

| Contrast | TCGA-CV | PAIP-IV | PAIP-EV |
|---|---|---|---|
| Caption-14 − Averaging | +0.019 [−0.007, +0.045], p = 0.15 | +0.110 [+0.045, +0.189], p = 0.001 | **+0.046 [+0.003, +0.086], p = 0.041** |
| Caption-15 − Averaging | +0.016 [−0.007, +0.040], p = 0.18 | +0.072 [+0.010, +0.140], p = 0.030 | +0.043 [−0.004, +0.086], p = 0.072 |
| TTC − Averaging | +0.002 [−0.021, +0.027], p = 0.89 | +0.077 [+0.024, +0.144], p = 0.001 | +0.019 [−0.025, +0.062], p = 0.38 |
| Mean of the three − Averaging | +0.012 [−0.010, +0.035], p = 0.26 | +0.086 [+0.031, +0.153], p = 0.001 | +0.036 [−0.007, +0.076], p = 0.094 |

- **Within TCGA** no aggregation is significantly better than averaging.
- **On PAIP-IV** all three are clearly better.
- **On PAIP-EV** Caption-14 is significantly better (+0.046), Caption-15 nearly so (+0.043, p = 0.072), and TTC is not.
  - With patch-level grouping it was the other way round: Caption-15 +0.050 (p = 0.016) and Caption-14 +0.041 (p = 0.041).
  - The size of the effect is unchanged. Only which caption variant crosses p < 0.05 moved.
- **The supportable statement is unchanged:** semantic aggregation generalises better; it does not fit TCGA better.
- **Method pairs.**
  - Caption-14 beats TTC on PAIP-IV (+0.033, p = 0.018) and PAIP-EV (+0.027, p = 0.016).
  - Caption-15 beats TTC on PAIP-EV (+0.024, p = 0.009).
  - Caption-14 beats Caption-15 on PAIP-IV only (+0.037, p = 0.001).

### 3. Encoders, heads and slide encoders

- **Encoder.**
  - H-Optimus-1 is best within TCGA, by 0.059 to 0.103 mean AUROC over every other encoder (all p = 0.001).
  - On PAIP-EV, H-Optimus-1 (0.869), UNI2 (0.859) and Virchow2 (0.851) are not distinguishable; only CONCH v1 is clearly lower (−0.047, p = 0.005).
  - On PAIP-IV, UNI2 has the highest mean (0.867).
- **Head.** LR is best everywhere.
  - Within TCGA, ANN and RF are equivalent to it, while kNN and ProtoNet are 0.17 lower.
  - On PAIP-EV, kNN (−0.054, p = 0.034) and RF (−0.039, p = 0.010) are lower.
- **Slide encoders.**
  - On the same CONCH 1.5 patch features, TITAN beats the mean of the four CONCH 1.5 aggregations within TCGA (by 0.071, p = 0.001), but not on PAIP-IV (0.033, p = 0.19) or PAIP-EV (0.006, p = 0.82).
  - Against all 100 aggregation configurations, TITAN is ahead within TCGA (0.037, p = 0.015) and on PAIP-IV (0.066, p = 0.001), and not distinguishable on PAIP-EV.
  - PRISM is not distinguishable from the aggregation mean anywhere.

### 4. The configuration TCGA-CV selects

TCGA-CV, by the published rule (best mean-of-folds AUROC), picks **H-Optimus-1 / TTC / LR**: mean of folds 0.924, pooled 0.918 [0.877, 0.952]. By pooled out-of-fold AUROC the pick is H-Optimus-1 / Caption-15 / LR (0.921 [0.876, 0.959]).

| Selection | PAIP-EV AUROC | Rank among 100 | Balanced accuracy | Sensitivity / specificity | Gap to the best configuration |
|---|---|---|---|---|---|
| H-Optimus-1 / TTC / LR (published rule) | 0.879 [0.734, 0.987] | 18 | 0.808 [0.697, 0.913] | 0.68 / 0.93 | 0.037 [−0.037, +0.124] |
| H-Optimus-1 / Caption-15 / LR (pooled rule) | 0.876 [0.729, 0.983] | 21 | 0.790 [0.677, 0.896] | 0.63 / 0.95 | 0.039 [−0.037, +0.127] |
| Encoder fixed to UNI2: TTC / LR | 0.874 [0.736, 0.978] | 23 | 0.862 [0.756, 0.948] | 0.84 / 0.88 | 0.041 [−0.028, +0.120] |
| Encoder fixed to Virchow2: Caption-14 / LR | 0.902 [0.773, 0.993] | 5 | 0.853 [0.749, 0.940] | 0.84 / 0.86 | 0.013 [−0.054, +0.087] |

- **The selected configuration is statistically indistinguishable from the best on PAIP**, as it was with patch-level grouping (H-Optimus-1 / Caption-15 / LR, 0.879, rank 19).
- **Rank agreement between TCGA-CV and PAIP-EV is moderate:** Spearman 0.49, with 1 of the TCGA top ten in the PAIP-EV top ten (patch-level: 0.56, 2).
- **There is no agreement with PAIP-IV** (Spearman 0.00), whose 31 test slides cannot rank 100 configurations.

### 5. Best of 100, for reference only

These are maxima over 100 configurations chosen on the same data, so they are optimistic and their intervals ignore the selection.

| Experiment | Best AUROC | Configuration |
|---|---|---|
| TCGA-CV | 0.921 [0.876, 0.959] | H-Optimus-1 / Caption-15 / LR |
| PAIP-IV | 0.970 [0.917, 1.000] | UNI2 / TTC / kNN |
| PAIP-EV | 0.915 [0.829, 0.974] | UNI2 / Averaging / LR |

- **PAIP-IV's 0.970** is the best of 100 on 31 test slides (7 MSI-H). Its mean over configurations did not move, so it is selection noise.
- **Best PAIP-EV balanced accuracy under the uniform rule:** 0.870 [0.766, 0.949] (Virchow2 / Caption-14 / ANN, sensitivity 0.84, specificity 0.90).

## How to read the numbers

- **Pooled vs per-fold AUROC.** For TCGA-CV the intervals are on the pooled out-of-fold AUROC. The published selection rule and the workbook use the mean of the four per-fold AUROCs. `ci_per_config.csv` has both.
- **Seed ensemble vs mean of seeds.** External AUROC for ANN and RF is computed on the mean probability of the five seed models, one prediction per slide. LR, kNN and ProtoNet are deterministic.
- **The quantile threshold is transductive.** It is set from the unlabelled score distribution of the target cohort and held fixed across bootstrap resamples, so the operating-point intervals are conditional on the threshold.
- **No multiplicity correction.** Treat p-values near 0.05 as suggestive. The Caption-14 / Caption-15 swap at p ≈ 0.04–0.07 shows why.

## Files

| File | Content |
|---|---|
| `results/ev_scores_paip.csv`, `results/ev_uniform_summary.csv` | external scores of the published models (Averaging, TITAN, PRISM are used from here; the crop-level grouped rows come from `experiments/crop_grouping/runs/cropgroup/ev/`) |
| `results/ev_scores_surgen.csv` | the earlier 622-slide SurGen scores (patch-level captions); used only with `--with-surgen` |
| `results/ci_per_config.csv` | AUROC with interval for the 330 configurations of TCGA-CV, PAIP-IV and PAIP-EV; operating-point intervals for PAIP-EV |
| `results/contrasts.csv` | the 90 paired contrasts |
| `results/prespecified.csv`, `results/selection_transfer.csv` | TCGA-selected configurations on PAIP; rank agreement |
| `results/ev_uniform_provenance.json` | commit, library versions and rule used for the scores |

Reproduce with the `exaonepath` conda environment:

```
python experiments/final_analysis/01_uniform_ev_scores.py
python experiments/final_analysis/validate_against_published.py
python experiments/final_analysis/02_uncertainty.py                      # crop-level, TCGA + PAIP
python experiments/final_analysis/02_uncertainty.py --grouping patch --with-surgen   # the 2026-10-05 analysis
```
