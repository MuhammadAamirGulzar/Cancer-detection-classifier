# Experiment 5 · Threshold calibration

Changes the decision rule rather than the model: a threshold fitted by inner cross-validation on the training slides, for all five classifier heads.

## Configuration

| Parameter | Value |
|---|---|
| ANN configuration | identical to experiment 2 |
| What changed | the decision threshold only |
| Inner CV | stratified 4-fold, over the 47 training slides |
| Criterion | Youden's J on pooled out-of-fold probabilities |
| Fitted on | 47 slides, 12 positives |
| Test slides used in the fit | none |
| Applies to | all five heads, not the ANN alone |

## Method

For each classifier and combination: stratified k-fold inside the 47 training slides, so every slide receives one prediction from a model that did not see it; the 47 out-of-fold probabilities are pooled and a threshold is fitted as the Youden-J optimum; that threshold is applied to the 31 test slides, which took no part in fitting it.

The ANN's early-stopping split is carved from the inner training rows so the inner test fold stays clean, and inner folds run with model_save_path=None so they cannot overwrite the real checkpoints.

It runs for all five heads deliberately. Calibrating the ANN alone while its competitors stayed at 0.5 would have replaced one unfair comparison with a different one - the mirror image of experiment 1, where the ANN alone trained on 38 slides.

0.5 remains the reported operating point. The fitted figures are written to threshold_audit.csv and to three extra summary_iv.csv columns alongside the headline, never instead of it, so nothing in experiments 1 to 4 is affected.

## Results

Means over the 20 aggregation rows per head, 4-fold inner CV.

| Head | BalAcc @ 0.5 | @ tau_train | Δ | median tau |
|---|---:|---:|---:|---:|
| knn | 0.7634 | 0.7442 | -0.0192 | 0.400 |
| lin | 0.7710 | 0.7365 | -0.0345 | 0.050 |
| ann | 0.7146 | 0.7241 | +0.0095 | 0.456 |
| rf | 0.7259 | 0.7094 | -0.0165 | 0.263 |
| proto | 0.7506 | 0.6936 | -0.0570 | 0.494 |

**Four of five heads lose.** The ANN is the only head that gains, by an eighth of one test slide, and still ranks third at tau_train.

Across all 110 rows: **63 worse, 31 better, 16 unchanged.** AUROC is unaffected, being threshold-free.

### The fitted thresholds are not stable

| Head | tau range | median |
|---|---|---:|
| lin | 0.003 – 0.630 | 0.040 |
| knn | 0.200 – 0.667 | 0.367 |
| proto | 0.476 – 0.500 | 0.494 |
| rf | 0.176 – 0.343 | 0.270 |
| ann | 0.116 – 0.987 | 0.427 |

A threshold fitted from 12 positives is dominated by noise. lin fits values as low as 0.003 and the ANN's range spans almost the entire unit interval - neither is a property you would want in a deployed classifier.

### What it does not do

Because balanced accuracy weights 24 negatives and 7 positives equally, a threshold trading sensitivity for specificity could raise the metric while detecting fewer cancers. That is not what happens here:

| False negatives at tau vs 0.5 | rows |
|---|---:|
| More missed positives | 3 |
| Fewer missed positives | 32 |
| Unchanged | 75 |

The procedure fails on accuracy, not on safety - it generally makes models more sensitive, not less.

### Conclusion

A decision threshold fitted by inner cross-validation on 47 training slides does not transfer to the model trained on all 47. It degrades four of five classifier heads, and the fitted values are unstable. The ANN's deficit is in ranking, not operating point - its median AUROC is 0.8244 against kNN's 0.8557 - and no threshold can close that.

## Held constant

| Setting | Value |
|---|---|
| Cohort | PAIP, 78 slides |
| Split | Provider's own: 47 train (12 MSI-H) / 31 test (7 MSI-H) |
| Protocol | Single fixed split, no folds, no averaging |
| Combinations | 22 - 4 aggregation methods x 5 encoders, plus TITAN and PRISM |
| Threshold | 0.5 |
| Uncertainty | Percentile bootstrap, 1000 resamples, 95% CI |
| Seed | 42 |
| Optimiser | Adam, lr=1e-4, weight_decay=1e-4 |
| LR schedule | ReduceLROnPlateau(factor=0.3, patience=5) |
| Loss | CrossEntropyLoss on raw logits |
| Batching | Full-batch - one Adam step per epoch |

The 47/31 split is the recovered cohort. Five training slides (training_data_19/30/41/42/46, two MSI-H) previously had no features; their patches were recovered and features rebuilt, taking the split from 42/31 to the full 47/31 and train MSI-H from 10 to 12.

## Reproducing

```bash
cd slide_classification
python runners/iv_runner.py --ann-protocol fixed --fit-threshold --tau-folds 4 --force
```

One test slide is worth 0.071 balanced accuracy. With 7 positive slides, flipping one moves the metric by 1/14 - larger than most differences reported here.
