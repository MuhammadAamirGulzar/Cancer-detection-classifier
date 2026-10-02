# Threshold and k experiments — results summary

Read-only harness. No pipeline code, artifact or threshold file was modified;
everything new lives under `experiments/threshold_and_k/`. Verified by
timestamp: `thresholds_TCGA.json`, `model_io.py`, `thresholds.py`,
`classifiers.py` and `knn.py` all predate this build.

Run with `python experiments/threshold_and_k/run_all.py`.

---

## A — dead **and** saturated threshold detection

`results/A_threshold_health.csv` — 284 rows, 40 configs, all k, both cohorts.

The old detector only tested `n_pos_pred == 0`. Added `== N` (saturated) plus a
1%-of-cohort band for `near_dead` / `near_saturated`, because a threshold one
slide short of the extreme fails identically to one at it.

Newly caught, all previously reported as healthy:

| cohort | config | k | n_pos_pred | τ_stale | p_min | BalAcc | AUROC |
|---|---|---|---|---|---|---|---|
| surgen | Caption_based / ConchV1 | 50 | **622/622** | 0.0953 | 0.100 | 0.5000 | 0.5902 |
| surgen | Caption_based / ConchV1 | 10 | 621/622 | 0.0953 | 0.000 | 0.4917 | 0.6063 |
| surgen | Caption_based / ConchV1 | 5 | 616/622 | 0.0953 | 0.000 | 0.4869 | 0.5738 |
| surgen | Caption_based / ConchV1 | 35 | 616/622 | 0.0953 | 0.057 | 0.5053 | 0.6193 |
| surgen | Tissue_Type / Conch1_5 | 50 | 619/622 | 0.0935 | 0.080 | 0.5027 | 0.6421 |

The strict `== N` test alone catches only the first row. Both of the cases named
in the brief are now flagged.

## B — refit τ per k

`results/B_refit_tau.csv`, `results/B_refit_thresholds.json` (new file).

Per (config, k): 4-fold TCGA CV at that k → pooled out-of-fold probabilities →
Youden's J → evaluate the target. Exact rather than approximate, because KNN is
lazy: excluding a fold from the reference set *is* training on the rest, which
is what `knn.py:33-35` does. k=35 added from training prevalence only
(35 × 0.145 ≈ 5 expected positive neighbours). No target labels are used.

| cohort | k | AUROC | BalAcc stale | BalAcc refit | oracle | gap stale | gap refit | τ stale | τ refit |
|---|---|---|---|---|---|---|---|---|---|
| paip | 3 | 0.6035 | 0.5890 | 0.5890 | 0.5948 | 0.006 | 0.006 | 0.256 | 0.350 |
| paip | 15 | 0.7566 | 0.6031 | 0.6642 | 0.7338 | 0.131 | 0.070 | 0.256 | 0.183 |
| paip | 35 | 0.8114 | 0.5958 | **0.7029** | 0.7697 | 0.174 | 0.067 | 0.256 | 0.170 |
| paip | 50 | 0.8158 | 0.5855 | **0.7188** | 0.7743 | 0.189 | 0.056 | 0.256 | 0.165 |
| surgen | 25 | **0.6675** | 0.5422 | 0.5627 | 0.6318 | 0.090 | 0.069 | 0.256 | 0.170 |
| surgen | 35 | 0.6533 | 0.5337 | **0.5708** | 0.6211 | 0.087 | 0.050 | 0.256 | 0.170 |
| surgen | 50 | 0.6454 | 0.5262 | 0.5761 | 0.6103 | 0.084 | 0.034 | 0.256 | 0.165 |

**Gap closed at k=50: 71% on PAIP (0.189 → 0.056), 59% on SurGen (0.084 → 0.034).**

Dead/saturated configs, before → after refitting:

```
paip   k=50   7 dead + 3 near_dead   ->  0 dead, 0 saturated  (20/20 ok)
surgen k=50  10 dead + 1 near_dead
              + 1 near_sat + 1 sat   ->  0 dead, 1 near_saturated (19/20 ok)
```

Refitting eliminates every dead threshold at every k above 3. The residual gap
(0.03–0.07) is ranking loss the threshold cannot recover.

## C — quantile threshold, all five classifiers

`results/C_quantile_threshold.csv` — 200 rows (5 classifiers × 40 configs).

`r` = fraction of TCGA out-of-fold slides τ_TCGA flags positive; `τ_q` = the
(1−r) quantile of the target score distribution. Transports the operating
*rate*, not the score value. Uses target scores, never target labels.

| cohort | clf | AUROC | BalAcc @τ | BalAcc @τ_q | oracle | recovered | gap closed |
|---|---|---|---|---|---|---|---|
| paip | rf | 0.8359 | 0.6609 | 0.7724 | 0.8190 | +0.1115 | **71%** |
| paip | proto | 0.8502 | 0.7273 | 0.7940 | 0.8274 | +0.0667 | **67%** |
| paip | ann | 0.8643 | 0.7553 | 0.8139 | 0.8536 | +0.0585 | 60% |
| paip | lin | 0.8717 | 0.7431 | 0.7890 | 0.8428 | +0.0459 | 46% |
| paip | knn | 0.6110 | 0.5959 | 0.5746 | 0.6010 | −0.0213 | **−418%** |
| surgen | proto | 0.7094 | 0.5227 | 0.6525 | 0.6779 | +0.1298 | **84%** |
| surgen | ann | 0.7363 | 0.5886 | 0.6720 | 0.6994 | +0.0834 | 75% |
| surgen | rf | 0.7026 | 0.5618 | 0.6418 | 0.6694 | +0.0800 | 74% |
| surgen | lin | 0.7089 | 0.5944 | 0.6493 | 0.6837 | +0.0548 | 61% |
| surgen | knn | 0.5893 | 0.5653 | 0.5685 | 0.5826 | +0.0032 | 18% |

Unhealthy operating points: **38 → 10 of 200**.

**ProtoNet recovers, decisively** — the largest single gain in the study
(+0.1298 on SurGen, 84% of its gap). RF recovers (71%/74%). LR and ANN gain
too. **One correction, not three patches — for four of the five classifiers.**

**KNN is the exception and must not be treated the same way.** Its score has
2–4 distinct values, so a quantile cannot land at an arbitrary rate; on PAIP it
makes things *worse*. KNN needs the k fix (B) first, then a threshold.

## D — distance weighting, and the k=3 selection

`results/D_weighting.csv`, `results/D_k_selection.csv`.

**Part 1 — weighting changes resolution, not ranking.**

| cohort | AUROC uniform | AUROC distance | Δ | unique uniform | unique distance | zeros uniform | zeros distance |
|---|---|---|---|---|---|---|---|
| paip | 0.6110 | 0.6174 | +0.0064 | 3.3 | 25.2 | 53.6 | 53.6 |
| surgen | 0.5893 | 0.5908 | +0.0015 | 3.85 | 387.8 | 229.8 | 229.8 |

Ties largely dissolve (3.3 → 25.2 distinct values on PAIP) while AUROC moves by
<0.007. It does **not** remove all ties (0 of 40 configs reach `unique == n`)
and leaves the number of exact-zero scores **unchanged**, confirming that a
1/d-weighted vote over an all-negative neighbourhood is still exactly 0.

**Part 2 — the premise is only half right.**

The brief asked me to state that small k is genuinely optimal in-domain. **The
measurement does not support that.** In-domain TCGA out-of-fold BalAcc by k:

```
k         3       5       7      10      15  |     25      35      50
BalAcc  0.6780  0.6766  0.6400  0.6710  0.6821 | 0.6871  0.6971  0.6782
in grid   yes     yes     yes     yes     yes  |   no      no      no
```

The in-domain optimum is **k=35**, not k=3. Per-config optima: k=35 for 9 of 20
configs, k=25 for 3, k=50 for 2 — only **6 of 20 have their optimum at k ≤ 15**.

So the binding constraint is the **grid cap at `knn.py:64`** (`n_neighbors` ∈
[3,5,7,10,15]); k > 15 was unreachable regardless of the scoring rule. Within
the cap, k=3 and k=15 are near-tied in-domain (0.6780 vs 0.6821), so the search
picking k=3 is unremarkable — it is choosing between roughly equivalent options,
not identifying a strong in-domain optimum.

*Caveat:* my criterion (out-of-fold BalAcc at Youden's J) is not the criterion
`GridSearchCV` used (inner-CV `balanced_accuracy` on `predict()`, i.e. a 0.5
majority vote). I did not replicate the search exactly, so this shows the grid
cap is binding but does not fully reconstruct why k=3 beat k=15 inside it.

---

## What is defensible

Using **TCGA-only** information:

1. **Raise the KNN neighbourhood cap.** k=35 is justified two independent ways —
   in-domain TCGA optimum (D2), and training prevalence (35 × 0.145 ≈ 5 expected
   positive neighbours). Neither looks at target data.
2. **Refit τ at whatever k is used** (B). Mandatory, not optional: a k=3-derived
   τ applied to k=35 scores is meaningless, and refitting removes every dead
   threshold.

Using **unlabelled-target** information:

3. **Quantile thresholding (C) for LR, ANN, ProtoNet and RF.** Reads target
   scores, never target labels — the same standing as unsupervised domain
   adaptation. Recovers 46–84% of the threshold gap and cuts unhealthy operating
   points from 38 to 10. Declare it in Methods as target-score-dependent.

**Not defensible, and deliberately not recommended:** k=50 on PAIP (AUROC 0.8158)
looks best but is the target-cohort optimum — selecting it would be test-set
selection. k=35 is recommended instead because it is derivable without touching
PAIP or SurGen. Note the two cohorts disagree on the target-optimal k (PAIP
rises to 50, SurGen peaks at 25), which is itself evidence that any
target-selected k would not transfer.

## Open

- Whether refit-τ plus k=35 also helps the four non-KNN classifiers is untested —
  B is KNN-only, since k has no meaning for the others.
- The exact `GridSearchCV` criterion was not replicated (see D2 caveat).
- Everything here is single-seed (42) for KNN/ProtoNet, which are deterministic;
  RF varies across the five seeds and only seed 42 was used in B and D.
