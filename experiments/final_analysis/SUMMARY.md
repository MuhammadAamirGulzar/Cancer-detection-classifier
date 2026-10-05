# Final analysis: one threshold rule, confidence intervals, pre-specified configuration

Date: 2026-10-05. Nothing in the published result trees was changed by this work; everything here is derived from the saved TCGA models, the cached features and the stored predictions.

## What was done

1. **One threshold rule for external validation** (`01_uniform_ev_scores.py`). The published tables apply the corrected threshold to KNN and RF only, and score SurGen KNN at k = 20, a value picked on SurGen. Here every head on both cohorts uses the `corrected` scheme of `runners/threshold_modes.py`: KNN at k = 35 with tau refitted on TCGA out-of-fold probabilities, and a rate-matched (quantile) threshold for LR, ANN, ProtoNet and RF. Per-slide scores are written out so that intervals can be computed.
2. **Confidence intervals and paired tests** (`02_uncertainty.py`) for all five experiments: case-level, class-stratified bootstrap, 2,000 resamples, the same resampled cases for every configuration so that differences are paired.
3. **The configuration TCGA-CV selects**, evaluated on the external cohorts, as the unbiased counterpart of "best of 100".

`validate_against_published.py` checks the new computation against the published trees. Frozen probabilities, AUROCs and balanced accuracies agree for all 215 classifier-combinations (largest probability difference 5e-7), the already-promoted KNN and RF values are reproduced exactly, and the LR / ANN / ProtoNet / KNN values equal the corrected-mode snapshots kept in the `*_ARCHIVED_*` trees. The run used the environment the TCGA models were trained in (scikit-learn 1.7.0, torch 2.2.0) and still reproduces the published external numbers, so the environment mismatch noted in the audit had no measurable effect.

## Findings

### 1. The uniform rule removes every dead operating point

Mean over the 20 aggregation combinations (`tcga_full` models). "Unhealthy" counts thresholds that flag nobody, everybody, or within 1% of either.

| Cohort | Head | Balanced accuracy, frozen | Balanced accuracy, uniform | Sensitivity | Specificity | Unhealthy, frozen -> uniform |
|---|---|---|---|---|---|---|
| PAIP | LR | 0.743 | 0.789 | 0.67 | 0.90 | 0 -> 0 |
| PAIP | ANN | 0.750 | 0.814 | 0.79 | 0.84 | 0 -> 0 |
| PAIP | KNN | 0.596 | 0.703 | 0.60 | 0.81 | 2 -> 0 |
| PAIP | ProtoNet | 0.727 | 0.794 | 0.72 | 0.87 | 3 -> 0 |
| PAIP | RF | 0.662 | 0.772 | 0.67 | 0.87 | 3 -> 0 |
| SurGen | LR | 0.594 | 0.649 | 0.50 | 0.80 | 4 -> 0 |
| SurGen | ANN | 0.581 | 0.672 | 0.62 | 0.72 | 2 -> 0 |
| SurGen | KNN | 0.565 | 0.571 | 0.85 | 0.30 | 0 -> 0 |
| SurGen | ProtoNet | 0.523 | 0.653 | 0.55 | 0.76 | 16 -> 0 |
| SurGen | RF | 0.558 | 0.642 | 0.51 | 0.77 | 8 -> 0 |

Across both cohorts 38 of 200 operating points were unhealthy under the frozen threshold and none is under the uniform rule. Dropping the SurGen-selected k = 20 costs KNN 0.024 of mean AUROC on SurGen (0.677 -> 0.653). KNN on SurGen stays poor at any k (specificity 0.30): it is a limit of the representation, not of the threshold.

### 2. Semantic aggregation beats averaging on the external cohorts, but not within TCGA

Difference in mean AUROC over 5 encoders x 5 heads, with 95% interval and two-sided bootstrap p-value.

| Contrast | TCGA-CV | PAIP-IV | PAIP-EV | SurGen-CV | SurGen-EV |
|---|---|---|---|---|---|
| Caption-14 - Averaging | +0.024 [-0.005, +0.054], p = 0.11 | +0.094 [+0.024, +0.186], p = 0.004 | +0.041 [+0.002, +0.081], p = 0.041 | +0.046 [+0.022, +0.071], p < 0.001 | +0.058 [+0.033, +0.079], p < 0.001 |
| Caption-15 - Averaging | +0.011 [-0.017, +0.040], p = 0.47 | +0.083 [+0.005, +0.178], p = 0.037 | +0.050 [+0.010, +0.088], p = 0.016 | +0.052 [+0.026, +0.079], p < 0.001 | +0.068 [+0.041, +0.095], p < 0.001 |
| TTC - Averaging | +0.013 [-0.011, +0.036], p = 0.29 | +0.093 [+0.035, +0.163], p < 0.001 | +0.018 [-0.022, +0.059], p = 0.37 | +0.022 [-0.001, +0.046], p = 0.062 | +0.036 [+0.011, +0.060], p = 0.005 |
| Mean of the three - Averaging | +0.016 [-0.010, +0.042], p = 0.22 | +0.090 [+0.025, +0.176], p = 0.003 | +0.037 [-0.001, +0.074], p = 0.063 | +0.040 [+0.019, +0.063], p = 0.001 | +0.054 [+0.035, +0.072], p < 0.001 |

- In TCGA cross-validation no aggregation is significantly better than averaging.
- On SurGen the advantage is clear for all three methods, in both cross-validation and external validation.
- On PAIP external validation the two caption methods are better; tissue-type clustering is not.
- The honest statement of the main result is therefore: **semantic aggregation generalises better; it does not fit TCGA better.**
- Caption-15 and Caption-14 are not distinguishable (the only difference near significance is on the 31-slide PAIP-IV set: -0.011, p = 0.045). Caption-15 is better than tissue-type clustering on PAIP-EV (+0.031, p = 0.001) and SurGen-CV (+0.030, p = 0.006).
- The SurGen-EV gain is concentrated in LR (+0.097), RF (+0.078) and ANN (+0.063); KNN gains nothing (+0.005). By encoder it is largest for H-Optimus-1 (+0.114) and absent for CONCH 1.5 (+0.010).

### 3. Encoders and heads

- **Encoder.** H-Optimus-1 is best within TCGA by 0.055 to 0.121 mean AUROC over every other encoder (all p < 0.001). UNI2 is best on SurGen, in cross-validation and externally, by 0.049 to 0.148 (all p <= 0.001). On PAIP-EV, H-Optimus-1, UNI2 and Virchow2 are not distinguishable. The encoder that wins on TCGA is not the one that transfers best to SurGen.
- **Head.** LR, ANN and RF are equivalent within TCGA; KNN and ProtoNet are 0.16 lower. Externally LR is best on PAIP and ANN on SurGen, with KNN clearly last on both.
- **Slide encoders.** On the same CONCH 1.5 patch features, TITAN is better than the mean of the four aggregation methods in TCGA-CV (by 0.085, p < 0.001), SurGen-CV (0.057, p < 0.001) and SurGen-EV (0.066, p = 0.038). The aggregation methods only overtake TITAN when paired with a stronger patch encoder.
- **Fold ensemble vs the single full-data model.** A small gain on SurGen for LR (+0.018, p = 0.001) and RF (+0.015, p = 0.002); nothing on PAIP.

### 4. The configuration TCGA-CV selects

TCGA-CV picks H-Optimus-1 / Caption-15 / LR (AUROC 0.928 [0.887, 0.960]).

| Cohort | AUROC | Rank among 100 | Balanced accuracy | Sensitivity / specificity | Gap to the best configuration |
|---|---|---|---|---|---|
| PAIP-EV | 0.879 [0.733, 0.984] | 19 | 0.799 [0.685, 0.904] | 0.63 / 0.97 | 0.037 [-0.038, +0.125] |
| SurGen-EV | 0.740 [0.652, 0.821] | 32 | 0.688 [0.621, 0.758] | 0.52 / 0.86 | 0.116 [+0.050, +0.184] |

On PAIP the TCGA-selected configuration is statistically indistinguishable from the best one. On SurGen it is clearly worse. Rank agreement between TCGA-CV and the external cohorts is moderate (Spearman 0.58 for PAIP-EV, 0.50 for SurGen-EV; one or two of the TCGA top ten are in the external top ten).

If the encoder is fixed in advance and only the aggregation and head are chosen on TCGA, UNI2 gives tissue-type clustering / LR: 0.886 [0.757, 0.978] on PAIP and 0.834 [0.768, 0.890] on SurGen (rank 7, not distinguishable from the best).

### 5. Best of 100, for reference only

These are maxima over 100 configurations chosen on the same data, so they are optimistic and their intervals ignore the selection.

| Experiment | Best AUROC | Configuration |
|---|---|---|
| TCGA-CV | 0.928 [0.887, 0.960] | H-Optimus-1 / Caption-15 / LR |
| PAIP-IV | 0.926 [0.774, 1.000] | CONCH 1.5 / Caption-15 / KNN |
| PAIP-EV | 0.915 [0.829, 0.974] | UNI2 / Averaging / LR |
| SurGen-CV | 0.851 [0.799, 0.896] | UNI2 / Caption-15 / RF |
| SurGen-EV | 0.856 [0.800, 0.905] | UNI2 / Caption-15 / LR |

Best balanced accuracy under the uniform rule: PAIP-EV 0.896 [0.800, 0.975] (UNI2 / Caption-14 / ANN, sensitivity 0.84, specificity 0.95); SurGen-EV 0.781 [0.715, 0.842] (UNI2 / Caption-14 / LR, 0.72 / 0.85).

## How to read the numbers

- **Pooled vs per-fold AUROC.** For TCGA-CV and SurGen-CV the intervals are on the pooled out-of-fold AUROC. The workbook reports the mean of the four per-fold AUROCs. `ci_per_config.csv` has both; they differ by a few thousandths (SurGen-CV best: 0.851 pooled, 0.853 mean of folds).
- **Seed ensemble vs mean of seeds.** External AUROC for ANN and RF is computed here on the mean probability of the five seed models, one prediction per slide. The workbook reports the mean of five single-seed AUROCs, which is lower by about 0.01 (PAIP) to 0.03 (SurGen). LR, KNN and ProtoNet are deterministic, so nothing changes for them. `ev_uniform_summary.csv` has both columns.
- **The quantile threshold is transductive.** It is set from the unlabelled score distribution of the target cohort and held fixed across bootstrap resamples, so the operating-point intervals are conditional on the threshold.
- **No multiplicity correction.** The contrasts are the ones the paper's claims rest on, but there are many of them; treat p-values near 0.05 as suggestive.
- **SurGen PRISM is excluded.** Its embeddings exist but are not usable yet: `SR386_40X_HE_T237_01.pt` is not a readable torch file and `SR386_40X_HE_T241_01` was built from a slide missing 52% of its tiles.

## Files

| File | Content |
|---|---|
| `results/ev_scores_paip.csv`, `results/ev_scores_surgen.csv` | one row per slide, combination and head: uniform score and threshold, seed-mean score and frozen threshold, fold-ensemble score |
| `results/ev_uniform_summary.csv` | per combination and head: AUROC under each estimand, frozen and uniform operating points, health status |
| `results/ci_per_config.csv` | AUROC with interval for all 540 configurations of the five experiments; operating-point intervals for the external ones |
| `results/contrasts.csv` | the 150 paired contrasts |
| `results/prespecified.csv`, `results/selection_transfer.csv` | TCGA-selected configurations on the external cohorts; rank agreement |
| `results/ev_uniform_provenance.json` | commit, library versions and rule used for the scores |

Reproduce with the `exaonepath` conda environment:

```
python experiments/final_analysis/01_uniform_ev_scores.py
python experiments/final_analysis/validate_against_published.py
python experiments/final_analysis/02_uncertainty.py
```
