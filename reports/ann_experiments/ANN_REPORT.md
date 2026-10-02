# The ANN Head - Configurations and Results

Everything about the ANN across the PAIP-IV experiments: what it was configured with, where each value came from, and what it scored. Read from the run records - saved checkpoint configs and result stamps - not transcribed.

## Results across every experiment

Experiments 1-4 change the ANN's configuration. Experiment 5 changes only the decision threshold - it reuses experiment 2's models, so its AUROC is identical to experiment 2's by construction and only the operating point differs.

| Experiment | Mean BalAcc | Median | Mean AUROC | Best | Best combination |
|---|---:|---:|---:|---:|---|
| 1 · Original (legacy) | 0.6921 | 0.6905 | 0.8233 | 0.7827 | Caption_based_15 / UNI2 |
| 2 · TCGA-CV hyperparameters | 0.7166 | 0.7054 | 0.8249 | 0.8244 | Tissue_Type_Clustering / Conch1_5 |
| 3 · ann_old | 0.7321 | 0.7411 | 0.8106 | 0.8125 | Averaging / Conch1_5 |
| 4 · ann_edited | 0.7255 | 0.7321 | 0.8190 | 0.8125 | Averaging / Conch1_5 |
| 5 · threshold (tau_train) | 0.7247 | 0.7292 | 0.8249 | 0.8155 | Caption_based_15 / H-Optimus-1 |

Every configuration beats the original. The two metrics disagree about which alternative wins - ann_old is highest on balanced accuracy and lowest on AUROC, the TCGA-CV configuration is the reverse. AUROC is threshold-free, so a rising balanced accuracy with a falling AUROC means the single-layer variants sit at a better operating point while ranking slides slightly worse.

One test slide is worth 0.071 balanced accuracy. With 7 positive slides, flipping one moves the metric by 1/14 - larger than most differences reported here.

## Against the other four heads

Threshold 0.5, means over 22 combinations. The four classical heads are constant across experiments 1-4, since only the ANN changed.

| Head | Mean BalAcc | Median | Mean AUROC | Best | Best combination |
|---|---:|---:|---:|---:|---|
| lin | 0.7787 | 0.7946 | 0.8350 | 0.9167 | TITAN / Conch1_5 |
| knn | 0.7616 | 0.7946 | 0.8393 | 0.9077 | Caption_based_15 / Conch1_5 |
| proto | 0.7499 | 0.7321 | 0.8249 | 0.8036 | Caption_based / UNI2 |
| rf | 0.7288 | 0.7470 | 0.8201 | 0.8363 | Tissue_Type_Clustering / Conch1_5 |
| ann | 0.7166 | 0.7054 | 0.8249 | 0.8244 | Tissue_Type_Clustering / Conch1_5 |

**The ANN is last on mean balanced accuracy under every configuration tried.** Its AUROC is within 0.010 of lin and ties proto, so its ranking is competitive - the deficit is in where its decision boundary sits. Threshold calibration was tested against exactly that and failed to fix it (experiment 5).

## ANN balanced accuracy, per combination

| Method | Model | Original | TCGA-CV | ann_old | ann_edited | tau_train |
|---|---|---:|---:|---:|---:|---:|
| Averaging | Conch1_5 | 0.7708 | 0.7619 | 0.8125 | 0.8125 | 0.8125 |
| Averaging | ConchV1 | 0.5536 | 0.6696 | 0.5565 | 0.5565 | 0.7292 |
| Averaging | H-Optimus-1 | 0.6488 | 0.7530 | 0.7530 | 0.7321 | 0.7530 |
| Averaging | UNI2 | 0.6369 | 0.6786 | 0.6994 | 0.6786 | 0.7083 |
| Averaging | Virchow2 | 0.5982 | 0.6607 | 0.6696 | 0.6905 | 0.6190 |
| Caption_based | Conch1_5 | 0.6696 | 0.6905 | 0.7113 | 0.7113 | 0.6905 |
| Caption_based | ConchV1 | 0.7619 | 0.7113 | 0.7530 | 0.7321 | 0.6488 |
| Caption_based | H-Optimus-1 | 0.7530 | 0.7530 | 0.7738 | 0.7738 | 0.7649 |
| Caption_based | UNI2 | 0.7411 | 0.7619 | 0.8036 | 0.8036 | 0.7530 |
| Caption_based | Virchow2 | 0.6488 | 0.6905 | 0.7113 | 0.7113 | 0.6696 |
| Caption_based_15 | Conch1_5 | 0.6905 | 0.6696 | 0.6905 | 0.6696 | 0.7113 |
| Caption_based_15 | ConchV1 | 0.6696 | 0.7113 | 0.7321 | 0.7321 | 0.6071 |
| Caption_based_15 | H-Optimus-1 | 0.7113 | 0.7530 | 0.7738 | 0.7738 | 0.8155 |
| Caption_based_15 | UNI2 | 0.7827 | 0.7619 | 0.7619 | 0.7619 | 0.8036 |
| Caption_based_15 | Virchow2 | 0.6488 | 0.6696 | 0.7113 | 0.7113 | 0.7113 |
| PRISM | PRISM | 0.7530 | 0.7738 | 0.7738 | 0.7738 | 0.7321 |
| TITAN | Conch1_5 | 0.7411 | 0.6994 | 0.7411 | 0.7202 | 0.7292 |
| Tissue_Type_Clustering | Conch1_5 | 0.7708 | 0.8244 | 0.7946 | 0.7530 | 0.8036 |
| Tissue_Type_Clustering | ConchV1 | 0.6786 | 0.6905 | 0.7619 | 0.7411 | 0.7411 |
| Tissue_Type_Clustering | H-Optimus-1 | 0.6905 | 0.7321 | 0.7113 | 0.7113 | 0.7321 |
| Tissue_Type_Clustering | UNI2 | 0.6994 | 0.6994 | 0.7411 | 0.7411 | 0.7292 |
| Tissue_Type_Clustering | Virchow2 | 0.6071 | 0.6488 | 0.6696 | 0.6696 | 0.6786 |

## Architecture and regularisation

| Experiment | Layers | h1 | h2 | max_iter | Dropout | Patience |
|---|---:|---|---|---|---:|---:|
| 1 · Original (legacy) | 2 | searched | searched | searched -> 500 | 0.3 | 20 |
| 2 · TCGA-CV hyperparameters | 2 | from TCGA-CV | from TCGA-CV | 500 | 0.3 | 20 |
| 3 · ann_old | 1 | 512 | - | 1000 | 0.5 | 10 |
| 4 · ann_edited | 1 | 512 | - | 1000 | 0.7 | 10 |

## How each configuration was obtained

| Experiment | Grid points | Selected on | Trains on | Held back | Source |
|---|---:|---|---:|---:|---|
| 1 · Original (legacy) | 18 | 9 PAIP slides | 38 of 47 | 9 | PAIP search |
| 2 · TCGA-CV hyperparameters | 1 | nothing - pinned | 47 | 9 (early stop only) | TCGA-CV folds |
| 3 · ann_old | 1 | nothing - pinned | 47 | 0 | collaborator script |
| 4 · ann_edited | 1 | nothing - pinned | 47 | 0 | collaborator script |

## Fixed in code, identical everywhere

These never varied. They are literals in `eval_patch_features/ann.py` and reach no output file, which is why the harvesting script reads them by introspecting the classifier rather than transcribing them.

| Parameter | Value | Where |
|---|---|---|
| Optimiser | Adam | ann.py fit() |
| lr | 1e-4 | constructor default |
| weight_decay | 1e-4 | constructor default |
| LR schedule | ReduceLROnPlateau(factor=0.3, patience=5) | ann.py fit() |
| Loss | CrossEntropyLoss on raw logits | ann.py __init__ |
| Batching | full-batch, one Adam step per epoch | ann.py fit() |
| Softmax in graph | no - applied only in predict_proba | Task 1.3 correction |
| Seed | 42 | iv_runner.py |

## Per-combination widths under experiment 2

Experiment 2 pins the configuration per (aggregation x encoder) from the TCGA-CV majority vote, so it is the only experiment where the architecture varies across combinations. Eight distinct configurations; `max_iter` is 500 in every one.

| Method | Model | h1 | h2 | max_iter | input dim |
|---|---|---:|---:|---:|---:|
| Averaging | Conch1_5 | 128 | 256 | 500 | 768 |
| Averaging | ConchV1 | 512 | 128 | 500 | 512 |
| Averaging | H-Optimus-1 | 256 | 256 | 500 | 1536 |
| Averaging | UNI2 | 256 | 256 | 500 | 1536 |
| Averaging | Virchow2 | 128 | 64 | 500 | 2560 |
| Caption_based | Conch1_5 | 512 | 64 | 500 | 10752 |
| Caption_based | ConchV1 | 128 | 256 | 500 | 7168 |
| Caption_based | H-Optimus-1 | 512 | 256 | 500 | 21504 |
| Caption_based | UNI2 | 512 | 256 | 500 | 21504 |
| Caption_based | Virchow2 | 128 | 128 | 500 | 35840 |
| Caption_based_15 | Conch1_5 | 512 | 64 | 500 | 11520 |
| Caption_based_15 | ConchV1 | 128 | 64 | 500 | 7680 |
| Caption_based_15 | H-Optimus-1 | 512 | 128 | 500 | 23040 |
| Caption_based_15 | UNI2 | 256 | 128 | 500 | 23040 |
| Caption_based_15 | Virchow2 | 128 | 128 | 500 | 38400 |
| PRISM | PRISM | 512 | 256 | 500 | 1280 |
| TITAN | Conch1_5 | 128 | 256 | 500 | 768 |
| Tissue_Type_Clustering | Conch1_5 | 256 | 128 | 500 | 6912 |
| Tissue_Type_Clustering | ConchV1 | 256 | 128 | 500 | 4608 |
| Tissue_Type_Clustering | H-Optimus-1 | 128 | 128 | 500 | 13824 |
| Tissue_Type_Clustering | UNI2 | 128 | 256 | 500 | 13824 |
| Tissue_Type_Clustering | Virchow2 | 512 | 256 | 500 | 23040 |

Input dimension spans 512 to 38,400 because aggregation is flattened: Averaging gives 1xD, Tissue_Type_Clustering 9xD, Caption_based_15 15xD. The same hidden width therefore means a 1:1 mapping for one combination and a 300:1 bottleneck for another. Measured across the 22 combinations, network size has no relationship with performance (r = +0.042 between log-parameters and balanced accuracy), so this is a presentational inconsistency rather than a performance one.

## Experiment 5 - threshold, not configuration

The threshold experiment reused experiment 2's configuration verbatim. **No ANN hyperparameter differs between them** - only the decision rule changed, from a fixed 0.5 to a threshold fitted by 4-fold inner cross-validation on the 47 training slides.

| Metric | Value |
|---|---:|
| ANN configuration | identical to experiment 2 |
| Threshold at | fitted per combination, not 0.5 |
| tau range | 0.116 - 0.987 |
| Median tau | 0.427 |
| Mean BalAcc at 0.5 | 0.7166 |
| Mean BalAcc at tau_train | 0.7247 |
| Change | +0.0081 |
| Per combination | 12 better, 7 worse, 3 unchanged |
| Mean AUROC | unchanged - threshold-free |

The ANN gains +0.0081 mean balanced accuracy - roughly an eighth of one test slide. It is the only head that gains; the other four all lose, `proto` by 0.057. At tau_train the ANN still ranks third of five.

The ANN's fitted thresholds span almost the entire unit interval (0.116 to 0.987), which is what a threshold estimated from 12 positives looks like. See the experiment 5 report for the full five-head comparison and the stability analysis.
