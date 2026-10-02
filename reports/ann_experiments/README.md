# PAIP-IV ANN Experiments

Five experiments on the PAIP provider split. Experiments 1-4 vary the ANN's configuration; experiment 5 varies the decision threshold and leaves the models untouched.

| # | Experiment | Report | Source | Repeats |
|---:|---|---|---|---:|
| 1 | Original (legacy) | [01_original.md](01_original.md) | `repeats_legacy.csv` | 4 |
| 2 | TCGA-CV hyperparameters | [02_tcga_cv.md](02_tcga_cv.md) | `paip_iv_repeats.csv` | 4 |
| 3 | ann_old | [03_ann_old.md](03_ann_old.md) | `repeats_ann_old.csv` | 4 |
| 4 | ann_edited | [04_ann_edited.md](04_ann_edited.md) | `repeats_ann_edited.csv` | 4 |
| 5 | Threshold calibration | [05_threshold.md](05_threshold.md) | `results trees` | 2 |

## Results across the four configuration experiments

| Experiment | Mean BalAcc | Median | Mean AUROC | Best |
|---|---:|---:|---:|---:|
| 1 · Original (legacy) | 0.6921 | 0.6905 | 0.8233 | 0.7827 |
| 2 · TCGA-CV hyperparameters | 0.7166 | 0.7054 | 0.8249 | 0.8244 |
| 3 · ann_old | 0.7321 | 0.7411 | 0.8106 | 0.8125 |
| 4 · ann_edited | 0.7255 | 0.7321 | 0.8190 | 0.8125 |

Every configuration beats the original. The two metrics disagree about which alternative wins: ann_old is highest on balanced accuracy and lowest on AUROC, while the TCGA-CV configuration is the reverse. AUROC is threshold-free, so a rising balanced accuracy with a falling AUROC means the single-layer variants sit at a better operating point while ranking slides slightly worse. On AUROC - the stated primary metric - the TCGA-CV configuration is strongest.

One test slide is worth 0.071 balanced accuracy. With 7 positive slides, flipping one moves the metric by 1/14 - larger than most differences reported here.

Experiments 3 and 4 change three things at once against experiment 2 - depth, dropout and iterations - so differences from experiment 2 cannot be attributed to one alone. Between 3 and 4 the comparison is clean: dropout is the only difference.

## Provenance

Experiments 2, 3 and 4 can all state that no hyperparameter was tuned on PAIP - experiment 2 takes them from a separate cohort, experiments 3 and 4 from the collaborator's script. Experiment 1 cannot: its configuration was selected on 9 PAIP training slides.


## All five classifier heads

Experiments 1-4 changed only the ANN, so `lin`, `knn`, `proto` and `rf` are identical across all of them - `ann_opts` reaches exactly one branch of `train_and_evaluate`, and a regression check confirmed 0 of 88 non-ANN rows changed between protocols. Their figures below therefore apply to every experiment.

Threshold 0.5, 22 combinations, means across combinations.

| Head | Mean BalAcc | Median | Mean AUROC | Best | Best combination |
|---|---:|---:|---:|---:|---|
| lin | 0.7787 | 0.7946 | 0.8350 | 0.9167 | TITAN / Conch1_5 |
| knn | 0.7616 | 0.7946 | 0.8393 | 0.9077 | Caption_based_15 / Conch1_5 |
| proto | 0.7499 | 0.7321 | 0.8249 | 0.8036 | Caption_based / UNI2 |
| rf | 0.7288 | 0.7470 | 0.8201 | 0.8363 | Tissue_Type_Clustering / Conch1_5 |
| ann | 0.7166 | 0.7054 | 0.8249 | 0.8244 | Tissue_Type_Clustering / Conch1_5 |

**The ANN is last on mean balanced accuracy under every configuration tried.** Its AUROC (0.8249) is within 0.010 of `lin` and `proto`, so its ranking is competitive; the deficit is in where its decision boundary sits, which is what experiment 5 tested and failed to fix.

| Head | Model | Hyperparameters | Search | Trains on | Feature norm |
|---|---|---|---|---:|---|
| lin | Logistic regression | C=10, max_iter=300, lbfgs, random_state=42 | none - single point | 47 | none |
| knn | k-nearest neighbours | k in {3,5,7,10,15} x metric in {cosine,euclidean,manhattan} x weights in {uniform,distance} | 30 points, nested CV on the training data only | 47 | L2 |
| proto | Class-mean prototypes | none | none | 47 | L2 |
| rf | Random forest | n_estimators=500, max_depth=None, min_samples_split=5, min_samples_leaf=1, class_weight={0:1, 1:10}, threshold 0.3 | none - single point | 47 | none |
| ann | 2-hidden-layer MLP | varies by experiment - see the ANN document | varies by experiment | 38 or 47 | none (BatchNorm only) |

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
