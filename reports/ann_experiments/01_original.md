# Experiment 1 · Original (legacy)

The pipeline as originally written: an 18-point grid selected on a 9-slide carve-out that is not returned for the final fit.

## Headline

| Metric | Value |
|---|---:|
| Repeats | 4 (bit-identical) |
| Combinations | 22 |
| Mean BalAcc | **0.6921** |
| Median BalAcc | 0.6905 |
| Mean AUROC | **0.8233** |
| Best BalAcc | 0.7827 |
| Best combination | Caption_based_15 / UNI2 |

## Configuration

| Parameter | Value |
|---|---|
| Hidden layers | 2 |
| hidden_dim1 | searched |
| hidden_dim2 | searched |
| max_iter | searched -> 500 |
| Dropout | 0.3 |
| Early-stop patience | 20 |
| Grid points evaluated | 18 |
| Selected on | 9 PAIP slides |
| ANN trains on | 38 of 47 |
| Slides held back | 9 |
| Hyperparameters from | PAIP search |

## Method

The ANN searches an 18-point grid (h1 in {128,256,512} x h2 in {64,128,256} x iter in {500,1000}) and selects on macro-F1 computed over a stratified 20% carve-out - 9 slides, roughly 7 non-MSI-H and 2 MSI-H. The carve-out is not returned for the final fit, so the ANN trains on 38 slides.

Two properties of this design motivated the other experiments. First, lin, knn, proto and rf all merge the carve-out back and train on all 47 (logistic.py:19, knn.py:33, protonet.py:40, r_forest_eval.py:240), so the ANN is the only head in the comparison training on less data.

Second, the selection is not informative at that size. Across the 22 combinations the 18 configurations collapse to a median of 3 distinct validation scores. The configuration chosen averages 0.6075 test macro-F1 against 0.6133 for the grid's median entry - worse than picking at random from its own grid. The grid's best entry averages 0.6623, so 0.055 is the headroom being lost.

## Results by combination

This experiment's ANN beside the four classical heads. Those four are constant across experiments 1-4 - only the ANN changed - so they are sourced once from the run that evaluated all five. Balanced accuracy at threshold 0.5.

| Method | Model | ann | lin | knn | proto | rf | best |
|---|---|---:|---:|---:|---:|---:|---|
| Caption_based_15 | UNI2 | **0.7827** | 0.7321 | 0.8244 | 0.8036 | 0.7827 | knn |
| Averaging | Conch1_5 | **0.7708** | 0.8244 | 0.6815 | 0.7113 | 0.7113 | lin |
| Tissue_Type_Clustering | Conch1_5 | **0.7708** | 0.7440 | 0.9077 | 0.7738 | 0.8363 | knn |
| Caption_based | ConchV1 | **0.7619** | 0.7946 | 0.7946 | 0.7321 | 0.7738 | lin |
| Caption_based | H-Optimus-1 | **0.7530** | 0.8155 | 0.8363 | 0.7946 | 0.7530 | knn |
| PRISM | PRISM | **0.7530** | 0.7946 | 0.6726 | 0.7530 | 0.7738 | lin |
| Caption_based | UNI2 | **0.7411** | 0.8036 | 0.8244 | 0.8036 | 0.7827 | knn |
| TITAN | Conch1_5 | **0.7411** | 0.9167 | 0.8155 | 0.7321 | 0.7411 | lin |
| Caption_based_15 | H-Optimus-1 | **0.7113** | 0.8363 | 0.8363 | 0.7946 | 0.7530 | lin |
| Tissue_Type_Clustering | UNI2 | **0.6994** | 0.7321 | 0.7946 | 0.8036 | 0.7321 | proto |
| Caption_based_15 | Conch1_5 | **0.6905** | 0.8363 | 0.9077 | 0.7321 | 0.7530 | knn |
| Tissue_Type_Clustering | H-Optimus-1 | **0.6905** | 0.8363 | 0.7946 | 0.7738 | 0.7738 | lin |
| Tissue_Type_Clustering | ConchV1 | **0.6786** | 0.7232 | 0.7946 | 0.7321 | 0.7321 | knn |
| Caption_based | Conch1_5 | **0.6696** | 0.7946 | 0.7738 | 0.7321 | 0.7530 | lin |
| Caption_based_15 | ConchV1 | **0.6696** | 0.7946 | 0.8155 | 0.7321 | 0.7530 | knn |
| Averaging | H-Optimus-1 | **0.6488** | 0.7946 | 0.6815 | 0.7530 | 0.7113 | lin |
| Caption_based | Virchow2 | **0.6488** | 0.7946 | 0.7946 | 0.7321 | 0.6280 | lin |
| Caption_based_15 | Virchow2 | **0.6488** | 0.7946 | 0.7738 | 0.7113 | 0.6905 | lin |
| Averaging | UNI2 | **0.6369** | 0.7530 | 0.6310 | 0.7619 | 0.6994 | proto |
| Tissue_Type_Clustering | Virchow2 | **0.6071** | 0.7738 | 0.7738 | 0.7321 | 0.6905 | lin |
| Averaging | Virchow2 | **0.5982** | 0.6310 | 0.5387 | 0.6905 | 0.6905 | proto |
| Averaging | ConchV1 | **0.5536** | 0.6101 | 0.4881 | 0.7113 | 0.5179 | proto |

**The ANN is the best head in 0 of 22 combinations.**

### ANN detail

| Method | Model | BalAcc | AUROC | Confusion matrix |
|---|---|---:|---:|---|
| Caption_based_15 | UNI2 | 0.7827 | 0.8810 | [[17, 7], [1, 6]] |
| Averaging | Conch1_5 | 0.7708 | 0.7798 | [[13, 11], [0, 7]] |
| Tissue_Type_Clustering | Conch1_5 | 0.7708 | 0.8929 | [[13, 11], [0, 7]] |
| Caption_based | ConchV1 | 0.7619 | 0.8512 | [[16, 8], [1, 6]] |
| Caption_based | H-Optimus-1 | 0.7530 | 0.8333 | [[19, 5], [2, 5]] |
| PRISM | PRISM | 0.7530 | 0.7917 | [[19, 5], [2, 5]] |
| Caption_based | UNI2 | 0.7411 | 0.8810 | [[15, 9], [1, 6]] |
| TITAN | Conch1_5 | 0.7411 | 0.8690 | [[15, 9], [1, 6]] |
| Caption_based_15 | H-Optimus-1 | 0.7113 | 0.7976 | [[17, 7], [2, 5]] |
| Tissue_Type_Clustering | UNI2 | 0.6994 | 0.8929 | [[13, 11], [1, 6]] |
| Caption_based_15 | Conch1_5 | 0.6905 | 0.8452 | [[16, 8], [2, 5]] |
| Tissue_Type_Clustering | H-Optimus-1 | 0.6905 | 0.8155 | [[16, 8], [2, 5]] |
| Tissue_Type_Clustering | ConchV1 | 0.6786 | 0.8810 | [[12, 12], [1, 6]] |
| Caption_based | Conch1_5 | 0.6696 | 0.8452 | [[15, 9], [2, 5]] |
| Caption_based_15 | ConchV1 | 0.6696 | 0.8155 | [[15, 9], [2, 5]] |
| Averaging | H-Optimus-1 | 0.6488 | 0.8036 | [[14, 10], [2, 5]] |
| Caption_based | Virchow2 | 0.6488 | 0.8095 | [[14, 10], [2, 5]] |
| Caption_based_15 | Virchow2 | 0.6488 | 0.8095 | [[14, 10], [2, 5]] |
| Averaging | UNI2 | 0.6369 | 0.8571 | [[10, 14], [1, 6]] |
| Tissue_Type_Clustering | Virchow2 | 0.6071 | 0.8274 | [[12, 12], [2, 5]] |
| Averaging | Virchow2 | 0.5982 | 0.6786 | [[15, 9], [3, 4]] |
| Averaging | ConchV1 | 0.5536 | 0.6548 | [[6, 18], [1, 6]] |

## Reading it

This is the baseline every other experiment is measured against. It is last on mean balanced accuracy, and it is the only configuration whose hyperparameters were chosen using PAIP data.

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
python run_iv_repeats.py --classifiers ann --out repeats_legacy.csv
```

One test slide is worth 0.071 balanced accuracy. With 7 positive slides, flipping one moves the metric by 1/14 - larger than most differences reported here.

### Why four repeats

The pipeline seeds torch and numpy before every fit and draws the carve-out with a seeded stratified split, so repeats at a fixed configuration are expected to be identical. All four runs were bit-identical across balanced accuracy, both CI bounds, AUROC, both CI bounds, accuracy, macro-F1, threshold and confusion matrix. That establishes every figure as exact rather than one draw from a distribution, and rules out GPU nondeterminism - not guaranteed for BatchNorm, and not something the pipeline configures away. It says nothing about how much a result would move under a different random draw; that would require varying the seed.
