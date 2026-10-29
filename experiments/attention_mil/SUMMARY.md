# Patch-importance MIL: learning which patches matter (TCGA-CV, PAIP-EV)

Status, 2026-10-29: **complete for all five encoders** (H-Optimus-1, UNI2, Virchow2, CONCH 1.5, CONCH v1), TCGA-CV and PAIP-EV. PAIP-IV is not run (47 training slides are too few for MIL). SurGen is out until it is re-processed in RGB.

## Question

The benchmark builds a slide from its patches with fixed rules (an average, or averages per caption or tissue group), so every patch counts by rule. Not every patch is equally informative for MSI-H.

- **Can a model learn which patches matter, and match the hand-designed aggregations?**
- **Does what it learns make biological sense?**

## Methods

All three use the benchmark's five-crop patch features. Each patch is the mean of its 5 crop vectors (`01_build_bags.py`), so a slide is an N × 1536 matrix. The published Averaging slide feature is exactly the mean of these rows. Zero-filled patch files are left out as the published features left them out (61 in all: H-Optimus-1 22 and UNI2 6 in TCGA, Virchow2 20 and CONCH 1.5 5 in PAIP, CONCH v1 8 in TCGA).

| Method | What it learns per patch | Slide prediction |
|---|---|---|
| **Attention MIL** (gated ABMIL, Ilse et al. 2018) | an importance weight a_i (summing to 1 over the slide) | linear head on the attention-weighted mean |
| **Top-k MIL** | one MSI-H score s_i (small MLP) | mean of the top 5% patch scores |
| **Patch-sum** (control, unlearned) | s_i = sum of the patch vector's values | 9 statistics of the s_i distribution → logistic regression |

**Protocol, as the benchmark:**
- **Settings:** TCGA-CV on the 4 published folds (out-of-fold, BalAcc at 0.5). PAIP-EV trains on all 413 TCGA slides, applies to the 78 PAIP slides, and uses the uniform threshold rule (rate-matched quantile from τ_TCGA).
- **Hyperparameters, fixed in advance and never tuned:** hidden 256, dropout 0.25, Adam lr 2e-4, weight decay 1e-4, 25 epochs, class-weighted BCE, ≤ 4096 random patches per slide per step (all at test time), 5 seeds averaged.
- **Comparison:** the final analysis's paired bootstrap (class-stratified, B = 2000, same seed) against the benchmark configurations of the same encoder (`03_compare.py`).

## Results

AUROC. "Best fixed" is the best of that encoder's 20 benchmark configurations in that setting (chosen after the fact, an optimistic reference). \* = MIL significantly above that encoder's Averaging / LR (p < 0.05).

| Encoder | TCGA-CV Averaging | Best fixed | Attention MIL | Top-k MIL | PAIP-EV Averaging | Best fixed | Attention MIL | Top-k MIL |
|---|---|---|---|---|---|---|---|---|
| H-Optimus-1 | 0.916 | 0.921 | 0.909 | 0.901 | 0.875 | 0.905 | 0.872 | 0.886 |
| UNI2 | 0.866 | 0.894 | 0.869 | 0.881 | 0.915 | 0.915 | 0.895 | **0.930** |
| Virchow2 | 0.858 | 0.869 | 0.851 | 0.844 | 0.891 | 0.902 | 0.902 | 0.879 |
| CONCH 1.5 | 0.754 | 0.850 | 0.825\* | 0.813\* | 0.790 | 0.877 | 0.855 | 0.880 |
| CONCH v1 | 0.756 | 0.839 | 0.837\* | 0.816\* | 0.864 | 0.864 | 0.843 | 0.804 |
| Mean | 0.830 | 0.875 | 0.858 | 0.851 | 0.867 | 0.893 | 0.873 | 0.876 |

Patch-sum + LR (control), TCGA-CV / PAIP-EV: H-Optimus-1 0.582 / 0.735, UNI2 0.621 / 0.691, Virchow2 0.357 / 0.628, CONCH 1.5 0.467 / 0.461, CONCH v1 0.550 / 0.798.

**Paired differences (MIL − reference):**
- **Against the encoder's best configuration:** no MIL run is significantly worse (20 of 20 comparisons p ≥ 0.16). Three PAIP runs equal or exceed it: UNI2 top-k +0.015 (p = 0.41), CONCH 1.5 top-k +0.004 (p = 0.96), Virchow2 attention ±0.000.
- **Against the encoder's Averaging / LR:** within noise for H-Optimus-1, UNI2 and Virchow2; for CONCH 1.5 and CONCH v1 MIL is higher within TCGA, +0.058 to +0.081 (p = 0.008 to 0.039), and on PAIP CONCH 1.5 +0.064 / +0.090 (p = 0.24 / 0.10).
- **UNI2 top-k vs the TCGA-selected H-Optimus-1 / TTC / LR on PAIP-EV:** +0.052 (p = 0.07).
- All-encoder overview (fixed rules with LR, MIL, patch-sum): `07_overview.py` → `results/overview.csv`, `results/fig_overview.png`.

## Explanations

- **Tissue enrichment** (`04_explain.py`, `fig_enrichment.png`): each patch's published CONCH v1 caption, comparing attention mass with area share per class, split by MSI-H and non-MSI-H slides.
  - **Tumour epithelium is favoured:** adenoma-like, poorly differentiated and plain carcinoma get 1.2–4× their area.
  - **Fat, background, muscle, debris and lymphocytes are down-weighted.**
  - **MSI-H-specific:** signet-ring and mucin are attended in MSI-H slides and avoided in non-MSI-H slides, in TCGA and in external PAIP, and with both encoders.

  | | Signet-ring, MSI-H / non-MSI-H | Mucin, MSI-H / non-MSI-H |
  |---|---|---|
  | H-Optimus-1 | 1.5–1.55× / 0.68–0.71× | 1.07–1.15× / 0.53× |
  | UNI2 | 1.9–2.3× / 0.62–0.72× | 1.16–1.65× / 0.51–0.55× |

  Mucinous and signet-ring differentiation are recognised MSI-H features.
  - **Other encoders:** Virchow2, CONCH 1.5 and CONCH v1 up-weight signet-ring and mucin in *both* MSI-H and non-MSI-H slides (1.2–3.7× their area). Signet-ring is still higher in MSI-H in 5 of 6 cohort comparisons; mucin is not consistent. The clean MSI-H-specific contrast is therefore an H-Optimus-1 / UNI2 finding. Lymphocytes are under-attended with every encoder.
- **Attention heatmaps** (`fig_heatmaps.png`, external PAIP slides): the most-attended patches of MSI-H slides are solid, poorly differentiated tumour sheets and mucinous tumour. For a non-MSI-H slide they are conventional gland-forming adenocarcinoma.
- **Top-k evidence maps** (`topk/fig_heatmaps.png`, one colour scale shared across slides): each MSI-H slide shows a distinct high-evidence region, and the non-MSI-H slide none. The first rendering used per-slide scaling, which made a non-MSI-H slide look as "hot" as the MSI-H ones; it was corrected before use.

## Reading

- **Learned patch importance is as accurate as the best hand-designed pipeline:** no significant difference from the best configuration of any of the five encoders. It needs no caption or tissue labels and no aggregation choice, and it gives per-patch explanations that match known MSI-H histology. That makes it a competitive, interpretable alternative and the learned-pooling baseline the benchmark lacked (ledger N12).
- **UNI2 top-k MIL gives the highest external AUROC seen so far (0.930)**, but not significantly above the best benchmark configuration (0.915). It is the best of ten MIL runs, so it is promising, not proven.
- **MIL helps most where averaging is weak.** For the two CONCH encoders, whose plain average is poor within TCGA (0.75), MIL recovers +0.06 to +0.08, close to their caption-grouping results, without any labels.
- **"One number per patch" works only when the number is learned.** The unweighted sum of feature values is near chance within TCGA (0.58–0.62), because embedding dimensions carry no shared scale or meaning.
- **Lymphocytes are under-attended**, although tumour-infiltrating lymphocytes are an MSI-H hallmark. Zero-shot lymphocyte labels are rare (about 3–4% of patches) and possibly noisy; to be checked with the CRC-100K tissue classifier.

## Limitations

- **One fixed hyperparameter set**, no tuning, 5 seeds. MIL is trained class-weighted, which raises its TCGA balanced accuracy at 0.5 compared with the unweighted LR; AUROC is the fair comparison.
- **PAIP-EV has 78 slides (19 MSI-H)**, so intervals are wide. Per-comparison p-values are not corrected for multiplicity.
- **Tissue labels for enrichment are zero-shot captions** (whole patch), not ground truth.

## Files

- `01_build_bags.py`: slide matrices into `features/<cohort>/<model>/` (git-ignored, about 1 GB per encoder).
- `02_train_abmil.py --method abmil|topk`: training and prediction. Writes `results/<model>/<method>/` (out-of-fold and external scores, summary, per-patch attention and scores).
- `05_patchsum_baseline.py`: the unlearned patch-sum control.
- `03_compare.py --method …`: paired bootstrap against the benchmark configurations (`compare.csv`).
- `04_explain.py --method …`: tissue enrichment and heatmaps (`enrichment.csv`, `fig_enrichment.png`, `fig_heatmaps.png`).
- `07_overview.py`: the all-approaches table and figure.
- `run_remaining.sh`: the sequential run for Virchow2, CONCH 1.5 and CONCH v1.
- `06_mlflow_table.py`: rows for the clean MLflow store (arm `mil`, TCGA-CV and PAIP-EV, `delta_*` vs the encoder's Averaging / LR).
