# Patch-importance MIL: learning which patches matter (TCGA-CV, PAIP-EV)

Status, 2026-10-29: **complete for H-Optimus-1 and UNI2**, TCGA-CV and PAIP-EV. PAIP-IV is not run (47 training slides are too few for MIL). SurGen is out until it is re-processed in RGB.

## Question

The benchmark builds a slide from its patches with fixed rules (an average, or averages per caption or tissue group), so every patch counts by rule. Not every patch is equally informative for MSI-H.

- **Can a model learn which patches matter, and match the hand-designed aggregations?**
- **Does what it learns make biological sense?**

## Methods

All three use the benchmark's five-crop patch features. Each patch is the mean of its 5 crop vectors (`01_build_bags.py`), so a slide is an N × 1536 matrix. The published Averaging slide feature is exactly the mean of these rows. Zero-filled patch files are left out as the published features left them out (22 H-Optimus-1, 6 UNI2, all TCGA).

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

AUROC [95% CI]; balanced accuracy in brackets after it.

| | H-Optimus-1 TCGA-CV | H-Optimus-1 PAIP-EV | UNI2 TCGA-CV | UNI2 PAIP-EV |
|---|---|---|---|---|
| Attention MIL | 0.909 [0.858, 0.951] (0.829) | 0.872 [0.732, 0.980] (0.799) | 0.869 [0.796, 0.929] (0.775) | 0.895 [0.751, 0.993] (0.843) |
| Top-k MIL | 0.901 [0.853, 0.940] (0.805) | 0.886 [0.756, 0.982] (0.800) | 0.881 [0.824, 0.928] (0.768) | **0.930** [0.839, 0.992] (0.845) |
| Patch-sum + LR | 0.582 (0.556) | 0.735 (0.574) | 0.621 (0.610) | 0.691 (0.597) |
| Averaging + LR (benchmark) | 0.916 (0.755) | 0.875 (0.843) | 0.866 (0.755) | 0.915 (0.747) |
| Best benchmark config of that encoder | 0.921 (Caption-15 / LR) | 0.905 (TTC / ANN) | 0.894 (TTC / LR) | 0.915 (Averaging / LR) |

**Paired differences (MIL − reference):**
- **Against the encoder's Averaging / LR:** within ±0.021 everywhere (p ≥ 0.27).
- **Against the encoder's best configuration:** −0.012 to −0.032 (p ≥ 0.2) for every MIL run except UNI2 top-k on PAIP, which is +0.015 (p = 0.41).
- **Against the mean of the encoder's 20 configurations:** +0.05 to +0.09 within TCGA (p ≤ 0.009).
- **UNI2 top-k vs the TCGA-selected H-Optimus-1 / TTC / LR on PAIP-EV:** +0.052 (p = 0.07).

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
- **Attention heatmaps** (`fig_heatmaps.png`, external PAIP slides): the most-attended patches of MSI-H slides are solid, poorly differentiated tumour sheets and mucinous tumour. For a non-MSI-H slide they are conventional gland-forming adenocarcinoma.
- **Top-k evidence maps** (`topk/fig_heatmaps.png`, one colour scale shared across slides): each MSI-H slide shows a distinct high-evidence region, and the non-MSI-H slide none. The first rendering used per-slide scaling, which made a non-MSI-H slide look as "hot" as the MSI-H ones; it was corrected before use.

## Reading

- **Learned patch importance is as accurate as the best hand-designed pipeline:** no significant difference from the best configuration of either encoder. It needs no caption or tissue labels and no aggregation choice, and it gives per-patch explanations that match known MSI-H histology. That makes it a competitive, interpretable alternative and the learned-pooling baseline the benchmark lacked (ledger N12).
- **UNI2 top-k MIL gives the highest external AUROC seen so far (0.930)**, but not significantly above the best benchmark configuration (0.915). It is the best of four MIL runs, so it is promising, not proven.
- **"One number per patch" works only when the number is learned.** The unweighted sum of feature values is near chance within TCGA (0.58–0.62), because embedding dimensions carry no shared scale or meaning.
- **Lymphocytes are under-attended**, although tumour-infiltrating lymphocytes are an MSI-H hallmark. Zero-shot lymphocyte labels are rare (about 3–4% of patches) and possibly noisy; to be checked with the CRC-100K tissue classifier.

## Limitations

- **Two encoders only:** H-Optimus-1 (best within TCGA) and UNI2 (best on PAIP); the other three are not run.
- **One fixed hyperparameter set**, no tuning, 5 seeds. MIL is trained class-weighted, which raises its TCGA balanced accuracy at 0.5 compared with the unweighted LR; AUROC is the fair comparison.
- **PAIP-EV has 78 slides (19 MSI-H)**, so intervals are wide. Per-comparison p-values are not corrected for multiplicity.
- **Tissue labels for enrichment are zero-shot captions** (whole patch), not ground truth.

## Files

- `01_build_bags.py`: slide matrices into `features/<cohort>/<model>/` (git-ignored, about 1 GB per encoder).
- `02_train_abmil.py --method abmil|topk`: training and prediction. Writes `results/<model>/<method>/` (out-of-fold and external scores, summary, per-patch attention and scores).
- `05_patchsum_baseline.py`: the unlearned patch-sum control.
- `03_compare.py --method …`: paired bootstrap against the benchmark configurations (`compare.csv`).
- `04_explain.py --method …`: tissue enrichment and heatmaps (`enrichment.csv`, `fig_enrichment.png`, `fig_heatmaps.png`).
- `06_mlflow_table.py`: rows for the clean MLflow store (arm `mil`, TCGA-CV and PAIP-EV, `delta_*` vs the encoder's Averaging / LR).
