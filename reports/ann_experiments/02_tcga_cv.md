# Experiment 2 · TCGA-CV hyperparameters

Configuration taken from the TCGA-CV majority vote; model refit on all 47 training slides.

## Headline

| Metric | Value |
|---|---:|
| Repeats | 4 (bit-identical) |
| Combinations | 22 |
| Mean BalAcc | **0.7166** |
| Median BalAcc | 0.7054 |
| Mean AUROC | **0.8249** |
| Best BalAcc | 0.8244 |
| Best combination | Tissue_Type_Clustering / Conch1_5 |

## Configuration

| Parameter | Value |
|---|---|
| Hidden layers | 2 |
| hidden_dim1 | from TCGA-CV |
| hidden_dim2 | from TCGA-CV |
| max_iter | 500 |
| Dropout | 0.3 |
| Early-stop patience | 20 |
| Grid points evaluated | 1 |
| Selected on | nothing - pinned |
| ANN trains on | 47 |
| Slides held back | 9 (early stop only) |
| Hyperparameters from | TCGA-CV folds |

## Method

Removes both problems identified in experiment 1. The configuration is taken from the majority vote across the four TCGA-CV folds - each validating on ~100 slides rather than 9 - and the model is refit on all 47. TCGA is a separate cohort, so no PAIP information enters the choice.

Eight distinct configurations across the 22 combinations; max_iter is 500 in every one.

## Results by combination

This experiment's ANN beside the four classical heads. Those four are constant across experiments 1-4 - only the ANN changed - so they are sourced once from the run that evaluated all five. Balanced accuracy at threshold 0.5.

| Method | Model | ann | lin | knn | proto | rf | best |
|---|---|---:|---:|---:|---:|---:|---|
| Tissue_Type_Clustering | Conch1_5 | **0.8244** | 0.7440 | 0.9077 | 0.7738 | 0.8363 | knn |
| PRISM | PRISM | **0.7738** | 0.7946 | 0.6726 | 0.7530 | 0.7738 | lin |
| Averaging | Conch1_5 | **0.7619** | 0.8244 | 0.6815 | 0.7113 | 0.7113 | lin |
| Caption_based | UNI2 | **0.7619** | 0.8036 | 0.8244 | 0.8036 | 0.7827 | knn |
| Caption_based_15 | UNI2 | **0.7619** | 0.7321 | 0.8244 | 0.8036 | 0.7827 | knn |
| Averaging | H-Optimus-1 | **0.7530** | 0.7946 | 0.6815 | 0.7530 | 0.7113 | lin |
| Caption_based | H-Optimus-1 | **0.7530** | 0.8155 | 0.8363 | 0.7946 | 0.7530 | knn |
| Caption_based_15 | H-Optimus-1 | **0.7530** | 0.8363 | 0.8363 | 0.7946 | 0.7530 | lin |
| Tissue_Type_Clustering | H-Optimus-1 | **0.7321** | 0.8363 | 0.7946 | 0.7738 | 0.7738 | lin |
| Caption_based | ConchV1 | **0.7113** | 0.7946 | 0.7946 | 0.7321 | 0.7738 | lin |
| Caption_based_15 | ConchV1 | **0.7113** | 0.7946 | 0.8155 | 0.7321 | 0.7530 | knn |
| TITAN | Conch1_5 | **0.6994** | 0.9167 | 0.8155 | 0.7321 | 0.7411 | lin |
| Tissue_Type_Clustering | UNI2 | **0.6994** | 0.7321 | 0.7946 | 0.8036 | 0.7321 | proto |
| Caption_based | Conch1_5 | **0.6905** | 0.7946 | 0.7738 | 0.7321 | 0.7530 | lin |
| Caption_based | Virchow2 | **0.6905** | 0.7946 | 0.7946 | 0.7321 | 0.6280 | lin |
| Tissue_Type_Clustering | ConchV1 | **0.6905** | 0.7232 | 0.7946 | 0.7321 | 0.7321 | knn |
| Averaging | UNI2 | **0.6786** | 0.7530 | 0.6310 | 0.7619 | 0.6994 | proto |
| Averaging | ConchV1 | **0.6696** | 0.6101 | 0.4881 | 0.7113 | 0.5179 | proto |
| Caption_based_15 | Conch1_5 | **0.6696** | 0.8363 | 0.9077 | 0.7321 | 0.7530 | knn |
| Caption_based_15 | Virchow2 | **0.6696** | 0.7946 | 0.7738 | 0.7113 | 0.6905 | lin |
| Averaging | Virchow2 | **0.6607** | 0.6310 | 0.5387 | 0.6905 | 0.6905 | proto |
| Tissue_Type_Clustering | Virchow2 | **0.6488** | 0.7738 | 0.7738 | 0.7321 | 0.6905 | lin |

**The ANN is the best head in 0 of 22 combinations.**

### ANN detail

| Method | Model | BalAcc | AUROC | Confusion matrix |
|---|---|---:|---:|---|
| Tissue_Type_Clustering | Conch1_5 | 0.8244 | 0.9048 | [[19, 5], [1, 6]] |
| PRISM | PRISM | 0.7738 | 0.7976 | [[20, 4], [2, 5]] |
| Averaging | Conch1_5 | 0.7619 | 0.8810 | [[16, 8], [1, 6]] |
| Caption_based | UNI2 | 0.7619 | 0.8750 | [[16, 8], [1, 6]] |
| Caption_based_15 | UNI2 | 0.7619 | 0.8690 | [[16, 8], [1, 6]] |
| Averaging | H-Optimus-1 | 0.7530 | 0.7679 | [[19, 5], [2, 5]] |
| Caption_based | H-Optimus-1 | 0.7530 | 0.8214 | [[19, 5], [2, 5]] |
| Caption_based_15 | H-Optimus-1 | 0.7530 | 0.8095 | [[19, 5], [2, 5]] |
| Tissue_Type_Clustering | H-Optimus-1 | 0.7321 | 0.8333 | [[18, 6], [2, 5]] |
| Caption_based | ConchV1 | 0.7113 | 0.8155 | [[17, 7], [2, 5]] |
| Caption_based_15 | ConchV1 | 0.7113 | 0.7917 | [[17, 7], [2, 5]] |
| TITAN | Conch1_5 | 0.6994 | 0.8810 | [[13, 11], [1, 6]] |
| Tissue_Type_Clustering | UNI2 | 0.6994 | 0.8869 | [[13, 11], [1, 6]] |
| Caption_based | Conch1_5 | 0.6905 | 0.8512 | [[16, 8], [2, 5]] |
| Caption_based | Virchow2 | 0.6905 | 0.8631 | [[16, 8], [2, 5]] |
| Tissue_Type_Clustering | ConchV1 | 0.6905 | 0.8274 | [[16, 8], [2, 5]] |
| Averaging | UNI2 | 0.6786 | 0.8214 | [[12, 12], [1, 6]] |
| Averaging | ConchV1 | 0.6696 | 0.7738 | [[15, 9], [2, 5]] |
| Caption_based_15 | Conch1_5 | 0.6696 | 0.8274 | [[15, 9], [2, 5]] |
| Caption_based_15 | Virchow2 | 0.6696 | 0.8095 | [[15, 9], [2, 5]] |
| Averaging | Virchow2 | 0.6607 | 0.6488 | [[18, 6], [3, 4]] |
| Tissue_Type_Clustering | Virchow2 | 0.6488 | 0.7917 | [[14, 10], [2, 5]] |

## Reading it

Best mean AUROC of the four and the best single row. Since AUROC is threshold-free and the stated primary metric, this is the strongest configuration on the measure the study has committed to.

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
python run_iv_repeats.py --ann-protocol fixed --classifiers ann --out paip_iv_repeats.csv
```

One test slide is worth 0.071 balanced accuracy. With 7 positive slides, flipping one moves the metric by 1/14 - larger than most differences reported here.

### Why four repeats

The pipeline seeds torch and numpy before every fit and draws the carve-out with a seeded stratified split, so repeats at a fixed configuration are expected to be identical. All four runs were bit-identical across balanced accuracy, both CI bounds, AUROC, both CI bounds, accuracy, macro-F1, threshold and confusion matrix. That establishes every figure as exact rather than one draw from a distribution, and rules out GPU nondeterminism - not guaranteed for BatchNorm, and not something the pipeline configures away. It says nothing about how much a result would move under a different random draw; that would require varying the seed.
