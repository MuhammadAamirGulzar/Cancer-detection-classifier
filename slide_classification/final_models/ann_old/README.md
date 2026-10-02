# Final ANN Configuration — `ann_old`

The ANN configuration adopted for the PAIP-IV study, its trained weights, and the
evidence behind the choice.

| | |
|---|---|
| **Configuration** | `ann_old` |
| **Networks** | 22 — one per (aggregation method × encoder) |
| **Trained on** | 47 PAIP training slides, nothing held back |
| **Evaluated on** | 31 PAIP test slides, threshold 0.5 |
| **Mean balanced accuracy** | **0.7321** |
| **Mean AUROC** | 0.8106 |
| **Best single result** | 0.8125 — `Averaging / Conch1_5` |
| **Source tree** | `PAIP_IV_Results_ARCHIVED_20260812_3` |

---

## The configuration

Taken verbatim from the collaborator's `ann_old.py`, as that script behaves when
driven through its own `eval_ANN` with nothing overridden.

```
Linear(D, 512) → ReLU → BatchNorm1d(512) → Dropout(0.5) → Linear(512, 2)
```

| Parameter | Value | Source |
|---|---|---|
| Hidden layers | **1** | `ann_old.py` |
| `hidden_dim` | **512** | class default |
| `max_iter` | **1000** | `eval_ANN` default (shadows the class's 100) |
| Dropout | **0.5** | hardcoded literal in the layer stack |
| Early-stop patience | 10 | constructor default |
| Optimiser | Adam, `lr=1e-4`, `weight_decay=1e-4` | constructor defaults |
| LR schedule | `ReduceLROnPlateau(factor=0.3, patience=5)` | hardcoded |
| Loss | `CrossEntropyLoss` on raw logits | — |
| Batching | full-batch, one Adam step per epoch | — |
| Output | raw logits; softmax only in `predict_proba` | Task 1.3 correction |
| Seed | 42 | runner default |

**Neither of the collaborator's scripts contains a grid, a search, or a stored
configuration** — every value is a default argument or a hardcoded literal. 512
and 1000 are simply what you get by calling `eval_ANN` without overriding
anything.

### One thing deliberately not reproduced

`ann_old.py` places `nn.Softmax` inside the training graph while training with
`CrossEntropyLoss`, which applies log-softmax internally. Softmax twice flattens
gradients and underfits the network — the defect the repository's Task 1.3
correction removed.

These networks carry the script's **dropout of 0.5** but emit raw logits like
every other head. Reproducing the double softmax would measure a bug rather than
a configuration.

---

## Why `ann_old`

It gives the **highest mean balanced accuracy of the five ANN configurations
tried**:

| Configuration | mean BalAcc | mean AUROC | best row |
|---|---:|---:|---:|
| Original (legacy) | 0.6921 | 0.8233 | 0.7827 |
| TCGA-CV | 0.7166 | **0.8249** | **0.8244** |
| **`ann_old`** | **0.7321** | **0.8106** | 0.8125 |
| `ann_edited` | 0.7255 | 0.8190 | 0.8125 |
| threshold (tau_train) | 0.7247 | 0.8249 | 0.8155 |

### The trade-off, recorded

`ann_old` has the best mean balanced accuracy **and the worst mean AUROC** of the
five. AUROC is threshold-free; balanced accuracy is read at a fixed 0.5. So the
single-layer network sits at a better operating point while ranking slides
slightly worse than the two-layer TCGA-CV configuration (0.8106 against 0.8249).

If AUROC is treated as the primary metric, TCGA-CV is the stronger configuration
and also produces the best single row. This is stated here so the choice is
visible rather than implicit.

Two further caveats worth carrying into any write-up:

- **One test slide is worth 0.071 balanced accuracy.** With 7 positives, the
  0.0155 spread between `ann_old`, `ann_edited` and TCGA-CV is under a quarter of
  one slide.
- **`ann_old` and `ann_edited` differ in exactly one hyperparameter** — dropout
  0.5 against 0.7. That is the only clean comparison in the set; the difference
  against TCGA-CV confounds depth, dropout and iteration count together.

### Provenance

Every hyperparameter traces to the collaborator's script. **TCGA-CV was not
consulted, and nothing was tuned on PAIP** — not on the 47 training slides and
not on the 31 test slides. That is a stronger claim than the original
configuration could make, where the network was selected on a 9-slide PAIP
subset by a search measured to perform worse than random over its own grid.

---

## Contents

```
final_models/ann_old/
  README.md                          this document
  manifest.json                      per-network record with SHA-256 checksums
  weights/
    <Method>__<Model>.pth            22 trained state dicts
    <Method>__<Model>.json           the config each was trained with
```

`manifest.json` records, for every network: the aggregation method, encoder,
input dimension, hidden width, SHA-256 of the weights file, and the balanced
accuracy, AUROC and confusion matrix it produced.

### Why the weights live here rather than in the results tree

Every `iv_runner.py --force` sweep renames the live tree to
`PAIP_IV_Results_ARCHIVED_<date>`, and the cleanup performed during this study
deleted 16 such trees. A checkpoint reachable only by remembering which archive
it landed in is one sweep from being renamed and a few from being deleted. This
directory is not touched by any sweep.

---

## Results

Per combination, balanced accuracy at threshold 0.5.

| Method | Model | BalAcc | AUROC |
|---|---|---:|---:|
| Averaging | Conch1_5 | **0.8125** | 0.8810 |
| Caption_based | UNI2 | 0.8036 | 0.8452 |
| Tissue_Type_Clustering | Conch1_5 | 0.7946 | 0.8929 |
| Caption_based | H-Optimus-1 | 0.7738 | 0.8393 |
| Caption_based_15 | H-Optimus-1 | 0.7738 | 0.8214 |
| PRISM | PRISM | 0.7738 | 0.8155 |
| Averaging | H-Optimus-1 | 0.7530 | 0.7679 |
| Caption_based | ConchV1 | 0.7530 | 0.8155 |
| Tissue_Type_Clustering | ConchV1 | 0.7619 | 0.8274 |
| Caption_based_15 | ConchV1 | 0.7321 | 0.7917 |
| TITAN | Conch1_5 | 0.7411 | 0.8810 |
| Tissue_Type_Clustering | UNI2 | 0.7411 | 0.8869 |
| Caption_based | Conch1_5 | 0.7113 | 0.8512 |
| Caption_based | Virchow2 | 0.7113 | 0.8631 |
| Caption_based_15 | Virchow2 | 0.7113 | 0.8095 |
| Tissue_Type_Clustering | H-Optimus-1 | 0.7113 | 0.8155 |
| Averaging | UNI2 | 0.6994 | 0.8214 |
| Caption_based_15 | Conch1_5 | 0.6905 | 0.8274 |
| Averaging | Virchow2 | 0.6696 | 0.6488 |
| Tissue_Type_Clustering | Virchow2 | 0.6696 | 0.7917 |
| Averaging | ConchV1 | 0.5565 | 0.7738 |

*(Exact figures for every network are in `manifest.json`.)*

### How it compares to the other classifier heads

| Head | mean BalAcc | mean AUROC |
|---|---:|---:|
| lin | **0.7787** | 0.8350 |
| knn | 0.7616 | **0.8393** |
| proto | 0.7499 | 0.8249 |
| **ann (`ann_old`)** | **0.7321** | 0.8106 |
| rf | 0.7288 | 0.8201 |

Under `ann_old` the ANN places **fourth of five**, ahead of the random forest by
0.0033 — about one-twentieth of a test slide, which is not a meaningful ordering.
On the 22 combinations the ANN is the best-performing head in **one**.

This is the honest position for PAIP-IV. On the external cohort the picture
reverses: with 413 TCGA training slides the ANN is the **best** head of the five
(median balanced accuracy 0.7745 to 0.7968 depending on the variant). The
conclusion supported by both cohorts is that the MLP head needs training sets on
the order of hundreds of slides; on 47 the simpler heads are better calibrated.

---

## Reproducing

```bash
cd slide_classification

# retrain
python runners/iv_runner.py --ann-protocol full --ann-preset ann_old \
    --classifiers ann --force

# or four repeats, collected into one CSV
python run_iv_repeats.py --ann-protocol full --ann-preset ann_old \
    --classifiers ann --out repeats_ann_old.csv

# re-freeze this artefact from a results tree
python ../tools/save_final_ann.py --source <tree>
```

The four repeats already run were **bit-identical** across balanced accuracy,
both CI bounds, AUROC, both CI bounds, accuracy, macro-F1, threshold and
confusion matrix — so the figures above are exact, not averages over runs.

## Loading a network

```python
from eval_patch_features.ann import load_ann_checkpoint

clf, meta = load_ann_checkpoint("final_models/ann_old/weights", fold=0)
probs = clf.predict_proba(feats)      # softmax applied here, not in the graph
```

`load_ann_checkpoint` reads the sibling `.json` and rebuilds the architecture
from it, so a network is never reconstructed from assumed dimensions. Note the
files here are named `<Method>__<Model>.pth` rather than `fold0_*`; pass the
directory and rename, or load the state dict directly and construct
`ANNBinaryClassifier(input_dim=…, hidden_dim1=512, hidden_dim2=None, dropout=0.5)`.
