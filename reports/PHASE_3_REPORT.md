# Phase 3 Report — Missing experiments

**Date:** 2026-08-04
**Branch:** `dev`
**Commit:** `bbe4c0e`

Phase 3 produced the experiments that did not exist and rebuilt the ones that did, all from corrected code. **Every number in this report supersedes its pre-remediation equivalent.**

---

## 1. What now exists

| Experiment | Combinations | N | Status before Phase 3 |
|---|---|---|---|
| TCGA-CV | 22 | 413 | existed (20 combos; TITAN/PRISM were commented out) |
| **PAIP-IV** | 22 | 42 train / 31 test | **did not exist** |
| PAIP-EV | 22 × 3 variants | 73 | existed, single variant, wrong ANN checkpoint |
| SurGen-CV | 6 | 622 | existed, leak-inflated |
| **SurGen-EV** | 6 × 3 variants | 622 | **did not exist** |

TITAN and PRISM are now covered in TCGA-CV, PAIP-IV and PAIP-EV (Task 3.3). SurGen has no TITAN/PRISM features on either machine.

**Runtime:** TCGA-CV 36 min · SurGen-CV 10.3 min · PAIP-IV 10.5 min · TCGA-FULL 17.5 min · PAIP-EV 1.9 min · SurGen-EV 0.7 min.

---

## 2. Task 3.0 — the TCGA-FULL artifact

### Hyperparameters, fixed from TCGA-CV alone

`tcga_full_hparams.json` is derived by majority vote across the 4 TCGA-CV folds, ties broken by best mean **validation** macro-F1. `assert_no_external_data()` enforces the acceptance criterion that no PAIP or SurGen data influences this step — it fails the build if the file so much as mentions either cohort.

Only the ANN has a real grid; the other four are single-point and are recorded explicitly so the file is a complete statement of what the final models were trained with.

### The three variants, side by side

| Variant | Train n | Role |
|---|---|---|
| `tcga_full` | 413 (ANN 351 + 62 early-stop) | **primary** |
| `fold_ensemble` | 4 × ~208, probabilities pooled then thresholded once | robustness check |
| `fold_average` | 4 × ~208, four metric sets averaged | legacy / continuity |

### Δ(TCGA-FULL − legacy) — what doubling the training data bought

Across 110 (combination × classifier) pairs on PAIP-EV:

| Metric | mean Δ | median Δ | improved |
|---|---|---|---|
| BalAcc | **+0.0158** | +0.0105 | 68/110 |
| AUROC | **+0.0046** | +0.0053 | 79/110 |

Training on 413 slides (60 MSI-H) instead of ~208 (~28 MSI-H) is worth roughly **+0.016 balanced accuracy and +0.005 AUROC**. Consistent in direction, modest in size. That is the answer to "how much did the extra ~200 slides buy?", and it is worth stating plainly — it is a smaller gain than the doubling of training data might suggest.

### Unplanned finding: the ensemble is not merely a robustness check

Mean across all 22 combinations, PAIP-EV:

| Classifier | | `fold_average` | `fold_ensemble` | `tcga_full` |
|---|---|---|---|---|
| ann | BalAcc | 0.7489 | **0.7871** | 0.7600 |
| | AUROC | 0.8281 | **0.8432** | 0.8386 |
| lin | BalAcc | 0.7139 | **0.7362** | 0.7344 |
| proto | BalAcc | 0.6830 | 0.7212 | **0.7230** |
| knn | AUROC | 0.6147 | **0.6593** | 0.6103 |
| rf | AUROC | 0.8152 | **0.8276** | 0.8154 |

**The 4-fold probability ensemble beats the single TCGA-FULL model on most classifiers**, despite each of its members training on half the data. The work order designates TCGA-FULL as primary; the evidence says the ensemble earns more than a footnote. Recommended framing for the paper: report TCGA-FULL as primary as planned, and state that the ensemble matched or exceeded it — averaging four models trained on 2 folds each recovers more than training one model on all 4.

### Seed variance

5 seeds (42–46). `lin`, `knn` and `proto` were **verified** bit-identical between seeds 42 and 43, so their SD is exactly 0 by demonstration rather than assumption; seeds 44–46 reuse the verified artifact. Only `ann` and `rf` carry non-zero SD (ANN BalAcc SD ranges ~0.00–0.03 across combinations).

---

## 3. Task 3.1 — PAIP-IV

The provider's own split, run for the first time.

- **Train 42** of the 47 declared ids (5 have no features, 2 of them MSI-H) → 10 MSI-H
- **Test 31**, all with features → 7 MSI-H
- Reported at **threshold 0.5** — an internal experiment, so §1b rule 2 applies
- A stratified 20% of the 42 is carved out for ANN early stopping; the 31 test slides never participate in any tuning decision
- `lin`/`knn`/`proto`/`rf` run with `combine_trainval=True` and therefore train on all 42; only the ANN holds the carve-out back

### Best results, with bootstrap CIs

| Method | Model | Clf | BalAcc | 95% CI | AUROC | 95% CI |
|---|---|---|---|---|---|---|
| Tissue_Type_Clustering | UNI2 | knn | 0.958 | [0.896, 1.000] | 0.964 | [0.905, 1.000] |
| TITAN | Conch1_5 | lin | 0.938 | [0.864, 1.000] | 0.946 | [0.853, 1.000] |
| Tissue_Type_Clustering | Conch1_5 | rf | 0.887 | [0.715, 1.000] | 0.940 | [0.845, 1.000] |

**The confidence intervals are the finding, not the point estimates.** Mean 95% CI width across all 110 rows is **0.363 for AUROC** and **0.369 for BalAcc** — on 31 test slides with 7 positives. Quoting "AUROC 0.964 on PAIP" without the interval would be indefensible; quoting "0.964, 95% CI [0.905, 1.000]" is honest and still a good result.

This is precisely why Task 2.2 replaced PAIP-CV with bootstrap CIs over a single larger split rather than repairing a 4-fold design that would have tested on 18 slides with ~4 positives.

### The ANN is the *worst* classifier here

| Classifier | mean BalAcc | mean AUROC |
|---|---|---|
| lin | **0.7880** | **0.8539** |
| rf | 0.7626 | 0.8527 |
| knn | 0.7584 | 0.8419 |
| proto | 0.7451 | 0.8393 |
| **ann** | **0.6861** | **0.7984** |

With ~34 training slides after the early-stopping carve-out, the ANN overfits and plain logistic regression wins. That is a useful counterweight to TCGA-CV, where the ANN tops the rankings with 213 training slides — and it is an argument the paper should make explicitly, because it shows the ANN's advantage is data-dependent rather than intrinsic.

---

## 4. Task 3.2 — SurGen-EV, and partial coverage by design

6 of 22 combinations ran; 16 were skipped and logged to `skipped_combinations.csv`:

```
ran     (6): Averaging/{Conch1_5,Virchow2}, Caption_based_aggregation/{Conch1_5,Virchow2},
             Tissue_Type_Clustering/{Conch1_5,Virchow2}
skipped (16): everything requiring H-Optimus-1, UNI2, ConchV1, Caption_based_15, TITAN or PRISM
```

This is **not an error**. SurGen aggregated features exist locally only for `conch1-5` and `virchow2`. The identical code run with `MACHINE=server` picks up the rest, and `tools/merge_results.py` combines the two trees, resolving collisions by newer git commit.

**The 2 missing SurGen slides** (Task 3.2 asks for them): `SR386_40X_HE_T086_01` and `SR386_40X_HE_T339_01`. Both are the only slide of their case, so the case count drops 554 → 552 when restricted to slides with features.

---

## 5. The threshold policy in practice (§1b)

τ_TCGA is fitted by Youden's J on pooled out-of-fold TCGA-CV probabilities — each of the 413 slides contributes exactly one prediction, made by a model that did not see it. `thresholds_from_oof` refuses to fit if any slide appears twice.

### What the fitted thresholds reveal

| Classifier | τ range across combinations | Interpretation |
|---|---|---|
| `lin` | **0.006 – 0.246** | Logistic regression trained on 15% positives is systematically under-confident. This *is* the calibration failure behind the `[[56,0],[15,2]]` matrix. |
| `rf` | 0.128 – 0.231 | Consistently **below** the pre-existing hardcoded 0.3, supporting the owner's ruling to replace it. |
| `proto` | 0.498 – 0.505 | ProtoNet's softmax-over-distances probabilities are tightly clustered at 0.5, so its operating point is hypersensitive to tiny threshold shifts. |
| `knn` | quantised (0.14, 0.20, 0.33, 0.67) | KNN probabilities are vote fractions of *k*, so thresholding them is inherently coarse. |
| `ann` | 0.39 – 0.66 | Scattered around 0.5; the best-calibrated of the five. |

### Honest caveat: τ_TCGA does not uniformly help

On one worked PAIP-EV combination (Averaging/Conch1_5, `fold_ensemble`):

| Classifier | @0.5 BalAcc | @τ_TCGA BalAcc | confusion @0.5 → @τ |
|---|---|---|---|
| rf | 0.500 | **0.588** | `[[56,0],[17,0]]` → `[[56,0],[14,3]]` |
| lin | **0.738** | 0.710 | `[[53,3],[8,9]]` → `[[40,16],[5,12]]` |
| proto | **0.760** | 0.714 | `[[39,17],[3,14]]` → `[[47,9],[7,10]]` |

τ_TCGA rescues the random forest — which at both 0.5 **and** its legacy 0.3 predicts *zero* positives — but costs `lin` and `proto` a little. The policy remains correct: you must not tune a threshold on the test cohort, and AUROC (the primary metric) is unaffected either way. But the paper should say that the fixed threshold is a *principled* choice rather than a uniformly optimal one.

Every run also writes `threshold_audit.csv` carrying metrics at τ_TCGA, at 0.5 and — for the random forest — at the legacy 0.3, so the old behaviour stays reproducible.

---

## 6. Comparison against the pre-remediation baseline

`comparison_old_vs_new.xlsx`, built from the archived trees and the corrected result files:

| Experiment | AUROC | BalAcc | MacroF1 | improved / worse (BalAcc) |
|---|---|---|---|---|
| **PAIP-EV** | +0.0056 | **+0.0695** | +0.0835 | 86 / 12 |
| **SurGen-CV** | **−0.0573** | −0.0380 | −0.0399 | 4 / 26 |
| **TCGA-CV** | +0.0002 | −0.0034 | −0.0053 | 23 / 38 (median 0.0000) |

Read together these three rows are the entire remediation:

- **TCGA-CV barely moves.** The corrections that apply to it (ANN selection bias, checkpoint, softmax, KNN grid) largely offset, and `lin`/`proto` are *exactly* unchanged — the sanity check that the corrections touched only what they should.
- **SurGen-CV falls,** because ~67% of two-slide cases had a sibling in both train and test. That drop is the price of correctness and must be presented as such.
- **PAIP-EV rises substantially,** because a fixed out-of-sample threshold replaced an uncalibrated operating point *and* the model now trains on 413 slides instead of ~208.

Per-classifier attribution on TCGA-CV:

| Classifier | AUROC | BalAcc | cause |
|---|---|---|---|
| lin | −0.0000 | −0.0001 | unaffected ✓ |
| proto | 0.0000 | 0.0000 | unaffected ✓ |
| rf | −0.0011 | +0.0036 | deterministic row ordering (bootstrap depends on it) |
| knn | +0.0001 | −0.0070 | metric grid: `manhattan` replacing duplicate `euclidean` |
| **ann** | **+0.0018** | **−0.0136** | test-selection bias removed, partly offset by the softmax fix |

The ANN row is the one to explain in the paper: **operating-point metrics fall because optimistic bias was removed, while AUROC holds up because the double-softmax fix genuinely improved the model.**

---

## 7. Optimisations adopted this phase

### KNN metric grid (owner-requested)

`['cosine', 'euclidean', 'minkowski']` → `['cosine', 'euclidean', 'manhattan']`. sklearn's `minkowski` defaults to `p=2`, which **is** `euclidean`, so a third of every KNN search recomputed a metric it had already evaluated and the grid advertised three distinct metrics while exploring two. Same cost, strictly larger search space, and one fewer thing for a reviewer to pick at.

### Seed reuse for deterministic classifiers (owner-requested)

All 5 classifiers are still reported at all 5 seeds. `lin`, `knn` and `proto` are trained at seeds 42 and 43 and **verified bit-identical** before seeds 44–46 reuse the artifact. This demonstrates the SD ≈ 0 the work order expects instead of assuming it, and the run falls back to training every seed (with a warning) if the check ever fails. TCGA-FULL finished in 17.5 minutes rather than the projected 2–3 hours.

### ANN grid width (evidence-based)

The old grid **truncated**: across the 88 validation-selected configurations, `h1` sat at the maximum in **61%** of cases and `h2` in **65%**. A 32-pair probe measured what widening buys on test, selecting on validation in both arms:

| Metric | mean Δ | win/tie/loss | Wilcoxon p |
|---|---|---|---|
| macro-F1 | +0.0196 | 14/12/6 | **0.006** |
| accuracy | +0.0208 | 16/13/3 | **0.0007** |
| BalAcc | +0.0115 | 13/12/7 | 0.15 |
| **AUROC** | +0.0037 | 12/11/9 | **0.63** |

Adopted as `{128,256,512} × {64,128,256} × {500,1000}`, with the honest framing: **significantly improves macro-F1 and accuracy; does not significantly improve AUROC — the primary metric — or balanced accuracy.** No metric degrades. Widening is only legitimate because Task 1.1 moved selection to validation; under the old test-set selection it would have bought nothing but more bias.

---

## 8. Failures encountered and how the design handled them

1. **joblib worker crash (reproducible).** `GridSearchCV` defaults to the process-based `loky` backend; on Windows the spawned worker re-imports scipy inside a process already holding a CUDA/OpenMP context and dies with `Windows fatal exception: access violation`. It killed the first full sweep at the very first KNN fold. Fixed by running that search under the threading backend — no processes spawned, verified over 12 consecutive fits. Slower (~49s vs ~8s per fold at D=21504) and accepted: finishing an unattended sweep beats finishing fast.

2. **JSON round-trip on `class_weight`.** `{0: 1, 1: 10}` serialises to `{"0": 1, "1": 10}`; sklearn then reports `The classes, [0, 1], are not in class_weight`. TCGA-FULL died on its first random forest. Fixed in `load_hparams`.

   Worth noting **the design worked**: a `ValueError` is classified as a *correctness* failure, so it halted its own stage loudly instead of being downgraded to a skip, while the orchestrator continued to the independent SurGen-CV stage rather than losing the whole run.

3. **Windows refused to rename a results directory** (`PermissionError [WinError 5]`) — a file watcher or indexer holding a handle; both `Path.rename` and `git mv` failed identically across retries. Archiving now falls back to a **verified copy** (file count checked, raises if short) so pre-correction results are still preserved before anything overwrites them.

---

## 9. Carried forward

- **§1c.2's repo-wide hardcoded-path grep still does not pass.** 23 upstream feature-extraction/aggregation files retain machine-specific roots. They are not exercised by Phases 1–4. The trigger to fix them is server-side SurGen feature generation for H-Optimus-1 / UNI2 / ConchV1 — which is exactly what the 16 skipped SurGen-EV combinations are waiting on. `OPEN_QUESTIONS.md` #4.
- **CIMP remains non-binary** (4 levels) and blocks Phase 5 until the owner chooses a dichotomisation. `OPEN_QUESTIONS.md` #5.
- **`Taiga_Paip_inference.ipynb` is malformed JSON**, pre-existing at commit `4240546`. Superseded by the PAIP-EV runner; should be repaired or retired.
