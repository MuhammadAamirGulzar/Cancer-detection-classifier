# Experiment 4 · ann_edited

The collaborator's ann_edited.py configuration: identical to ann_old except dropout 0.7.

## Headline

| Metric | Value |
|---|---:|
| Repeats | 4 (bit-identical) |
| Combinations | 22 |
| Mean BalAcc | **0.7255** |
| Median BalAcc | 0.7321 |
| Mean AUROC | **0.8190** |
| Best BalAcc | 0.8125 |
| Best combination | Averaging / Conch1_5 |

## Configuration

| Parameter | Value |
|---|---|
| Hidden layers | 1 |
| hidden_dim1 | 512 |
| hidden_dim2 | - |
| max_iter | 1000 |
| Dropout | 0.7 |
| Early-stop patience | 10 |
| Grid points evaluated | 1 |
| Selected on | nothing - pinned |
| ANN trains on | 47 |
| Slides held back | 0 |
| Hyperparameters from | collaborator script |

## Method

The collaborator's second script, implemented as a preset. It differs from ann_old in exactly one hyperparameter: dropout 0.7 against 0.5. Everything else - one hidden layer of 512, 1000 iterations, patience 10, Adam at 1e-4/1e-4 - is identical.

Unlike ann_old, this script already emits raw logits and applies softmax only in predict_proba, matching the pipeline's own convention.

## Results by combination

This experiment's ANN beside the four classical heads. Those four are constant across experiments 1-4 - only the ANN changed - so they are sourced once from the run that evaluated all five. Balanced accuracy at threshold 0.5.

| Method | Model | ann | lin | knn | proto | rf | best |
|---|---|---:|---:|---:|---:|---:|---|
| Averaging | Conch1_5 | **0.8125** | 0.8244 | 0.6815 | 0.7113 | 0.7113 | lin |
| Caption_based | UNI2 | **0.8036** | 0.8036 | 0.8244 | 0.8036 | 0.7827 | knn |
| Caption_based | H-Optimus-1 | **0.7738** | 0.8155 | 0.8363 | 0.7946 | 0.7530 | knn |
| Caption_based_15 | H-Optimus-1 | **0.7738** | 0.8363 | 0.8363 | 0.7946 | 0.7530 | lin |
| PRISM | PRISM | **0.7738** | 0.7946 | 0.6726 | 0.7530 | 0.7738 | lin |
| Caption_based_15 | UNI2 | **0.7619** | 0.7321 | 0.8244 | 0.8036 | 0.7827 | knn |
| Tissue_Type_Clustering | Conch1_5 | **0.7530** | 0.7440 | 0.9077 | 0.7738 | 0.8363 | knn |
| Tissue_Type_Clustering | ConchV1 | **0.7411** | 0.7232 | 0.7946 | 0.7321 | 0.7321 | knn |
| Tissue_Type_Clustering | UNI2 | **0.7411** | 0.7321 | 0.7946 | 0.8036 | 0.7321 | proto |
| Averaging | H-Optimus-1 | **0.7321** | 0.7946 | 0.6815 | 0.7530 | 0.7113 | lin |
| Caption_based | ConchV1 | **0.7321** | 0.7946 | 0.7946 | 0.7321 | 0.7738 | lin |
| Caption_based_15 | ConchV1 | **0.7321** | 0.7946 | 0.8155 | 0.7321 | 0.7530 | knn |
| TITAN | Conch1_5 | **0.7202** | 0.9167 | 0.8155 | 0.7321 | 0.7411 | lin |
| Caption_based | Conch1_5 | **0.7113** | 0.7946 | 0.7738 | 0.7321 | 0.7530 | lin |
| Caption_based | Virchow2 | **0.7113** | 0.7946 | 0.7946 | 0.7321 | 0.6280 | lin |
| Caption_based_15 | Virchow2 | **0.7113** | 0.7946 | 0.7738 | 0.7113 | 0.6905 | lin |
| Tissue_Type_Clustering | H-Optimus-1 | **0.7113** | 0.8363 | 0.7946 | 0.7738 | 0.7738 | lin |
| Averaging | Virchow2 | **0.6905** | 0.6310 | 0.5387 | 0.6905 | 0.6905 | ann |
| Averaging | UNI2 | **0.6786** | 0.7530 | 0.6310 | 0.7619 | 0.6994 | proto |
| Caption_based_15 | Conch1_5 | **0.6696** | 0.8363 | 0.9077 | 0.7321 | 0.7530 | knn |
| Tissue_Type_Clustering | Virchow2 | **0.6696** | 0.7738 | 0.7738 | 0.7321 | 0.6905 | lin |
| Averaging | ConchV1 | **0.5565** | 0.6101 | 0.4881 | 0.7113 | 0.5179 | proto |

**The ANN is the best head in 1 of 22 combinations.**

### ANN detail

| Method | Model | BalAcc | AUROC | Confusion matrix |
|---|---|---:|---:|---|
| Averaging | Conch1_5 | 0.8125 | 0.8929 | [[15, 9], [0, 7]] |
| Caption_based | UNI2 | 0.8036 | 0.8869 | [[18, 6], [1, 6]] |
| Caption_based | H-Optimus-1 | 0.7738 | 0.8333 | [[20, 4], [2, 5]] |
| Caption_based_15 | H-Optimus-1 | 0.7738 | 0.8095 | [[20, 4], [2, 5]] |
| PRISM | PRISM | 0.7738 | 0.8214 | [[20, 4], [2, 5]] |
| Caption_based_15 | UNI2 | 0.7619 | 0.8810 | [[16, 8], [1, 6]] |
| Tissue_Type_Clustering | Conch1_5 | 0.7530 | 0.7857 | [[19, 5], [2, 5]] |
| Tissue_Type_Clustering | ConchV1 | 0.7411 | 0.8690 | [[15, 9], [1, 6]] |
| Tissue_Type_Clustering | UNI2 | 0.7411 | 0.8690 | [[15, 9], [1, 6]] |
| Averaging | H-Optimus-1 | 0.7321 | 0.7560 | [[18, 6], [2, 5]] |
| Caption_based | ConchV1 | 0.7321 | 0.8155 | [[18, 6], [2, 5]] |
| Caption_based_15 | ConchV1 | 0.7321 | 0.7857 | [[18, 6], [2, 5]] |
| TITAN | Conch1_5 | 0.7202 | 0.8929 | [[14, 10], [1, 6]] |
| Caption_based | Conch1_5 | 0.7113 | 0.8095 | [[17, 7], [2, 5]] |
| Caption_based | Virchow2 | 0.7113 | 0.8274 | [[17, 7], [2, 5]] |
| Caption_based_15 | Virchow2 | 0.7113 | 0.7679 | [[17, 7], [2, 5]] |
| Tissue_Type_Clustering | H-Optimus-1 | 0.7113 | 0.7976 | [[17, 7], [2, 5]] |
| Averaging | Virchow2 | 0.6905 | 0.7202 | [[16, 8], [2, 5]] |
| Averaging | UNI2 | 0.6786 | 0.8512 | [[12, 12], [1, 6]] |
| Caption_based_15 | Conch1_5 | 0.6696 | 0.7857 | [[15, 9], [2, 5]] |
| Tissue_Type_Clustering | Virchow2 | 0.6696 | 0.8214 | [[15, 9], [2, 5]] |
| Averaging | ConchV1 | 0.5565 | 0.7381 | [[13, 11], [3, 4]] |

## Reading it

The clean comparison in this set: against experiment 3 the only difference is dropout, so any difference between them is attributable. Dropout 0.5 gives +0.0066 mean balanced accuracy and -0.0084 mean AUROC against 0.7.

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
python run_iv_repeats.py --ann-protocol full --ann-preset ann_edited --classifiers ann --out repeats_ann_edited.csv
```

One test slide is worth 0.071 balanced accuracy. With 7 positive slides, flipping one moves the metric by 1/14 - larger than most differences reported here.

### Why four repeats

The pipeline seeds torch and numpy before every fit and draws the carve-out with a seeded stratified split, so repeats at a fixed configuration are expected to be identical. All four runs were bit-identical across balanced accuracy, both CI bounds, AUROC, both CI bounds, accuracy, macro-F1, threshold and confusion matrix. That establishes every figure as exact rather than one draw from a distribution, and rules out GPU nondeterminism - not guaranteed for BatchNorm, and not something the pipeline configures away. It says nothing about how much a result would move under a different random draw; that would require varying the seed.
