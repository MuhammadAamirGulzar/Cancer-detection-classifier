# SurGen-EV diagnostic — frozen baseline and kNN k sweep

> **DIAGNOSTIC ONLY. Nothing here is a reportable configuration.**
> No k is selected, nothing is promoted, and no published file was written.
> Every output CSV carries a `DIAGNOSTIC` column for the same reason.

Run 3 September 2026; non-kNN coverage completed later the same day after the
two SurGen H-Optimus-1 combinations were re-run at the full 622 slides. All work
lives under `experiments/surgen_ev_diagnostic/`.

## What was and was not touched

| | |
|---|---|
| `slide_classification/SurGen_EV_Results/` | **read-only**, verified byte-identical by SHA-256 before and after |
| `thresholds_TCGA.json` | not opened for writing |
| `knn_k35_thresholds.json` | not modified |
| PAIP-EV tree, artifacts, workbook, docx, plots | untouched |

The brief opens with *"everything under experiments/ — do not touch published
SurGen-EV results"*, so Part 1's stripped record is written **here** as a derived
copy rather than applied to the published tree. Applying it later is a copy;
un-applying it would not have been. See "If you want this applied" below.

---

## Part 1 — the bare frozen baseline

`part1_frozen_baseline.py` → `frozen_baseline/` (21 files) and
`results/part1_surgen_ev_frozen_baseline.csv` (105 rows).

**1,092 unused keys stripped** across 21 combinations × 5 classifiers × 3
variants. **16 headline keys × 315 entries compared: zero drift.** Nothing
published moves, because nothing ever read those columns.

### A discrepancy the strip exposed

`status_frozen` does not describe the operating point the published BalAcc
describes. There are two counts of "slides flagged positive" and they are not
interchangeable:

- **`mean_probs >= tau`** — threshold the seed-*averaged* score vector once.
  This is what `_add_corrected` computed and what `status_frozen` reports.
- **`FP + TP` from `conf_matrix`** — the *mean of five per-seed confusion
  matrices*. This is the path `bacc`, `auroc` and `conf_matrix` take.

Averaging probabilities then thresholding ≠ thresholding then averaging. Five
entries disagree, some dramatically:

| config | head | mean_probs | conf_matrix |
|---|---|---|---|
| Averaging / Virchow2 | ANN | 617 (`near_saturated`) | 553 (`ok`) |
| Averaging / Virchow2 | RF | 6 (`near_dead`) | 14 (`ok`) |
| TTC / ConchV1 | RF | 0 (`dead`) | 2 (`near_dead`) |
| TTC / UNI2 | ANN | 622 (`saturated`) | 618 (`near_saturated`) |
| TITAN / Conch1_5 | ANN | 622 (`saturated`) | 607 (`ok`) |

The baseline reports both, named for what they are. `status` (confusion matrix)
is the one consistent with the reported metrics; `status_meanprob` reproduces the
stored value exactly. Full list in
`results/part1_status_definition_disagreement.csv`.

### Baseline, tcga_full variant

| head | mean AUROC | mean BalAcc | ok | near_dead | dead | near_sat |
|---|---|---|---|---|---|---|
| ANN | 0.7117 | 0.5777 | 20 | 0 | 0 | 1 |
| ProtoNet | 0.7121 | 0.5236 | 5 | 5 | **11** | 0 |
| LR | 0.7078 | 0.5896 | 16 | 4 | 0 | 1 |
| RF | 0.6708 | 0.5565 | 14 | 2 | **5** | 0 |
| kNN | 0.5929 | 0.5699 | 21 | 0 | 0 | 0 |

**16 of 105 configurations sit at a dead or near-dead frozen operating point**,
concentrated in ProtoNet (16 of 21 dead or near-dead — its τ sits at ~0.50 on a
score distribution that barely crosses it) and RF (7 of 21). kNN is the only head
with no pathological threshold, and also the weakest ranker.

---

## Part 2 — k sweep × threshold scheme

`part2_k_sweep.py` → `results/part2_DIAGNOSTIC_knn_k_sweep.csv` (903 rows =
43 combinations × 7 k × 3 schemes) and `part2_DIAGNOSTIC_other_heads.csv` (344,
full coverage).

k ∈ {15, 20, 25, 35, 50, 75, 100}. The two low values bracket the previously
reported peak — a maximum with no measured shoulder below it is an endpoint, not
a maximum.

**The sweep reproduces the earlier one**: k=25 → 0.6677 (previously 0.6675),
k=50 → 0.6453 (previously 0.6454).

### Q1 — Does any k give SurGen kNN a meaningful gain?

**No. It plateaus, and the plateau is the answer.** Mean AUROC over the 20
aggregation combinations:

| k | SurGen | PAIP |
|---|---|---|
| published (artifact k, mostly 3) | 0.5893 | 0.6110 |
| 15 | 0.6649 | 0.7566 |
| **20** | **0.6770** ← peak | 0.7925 |
| 25 | 0.6677 | 0.8030 |
| 35 | 0.6534 | 0.8114 |
| 50 | 0.6453 | 0.8158 |
| 75 | 0.6137 | **0.8330** |
| 100 | 0.5495 | 0.8319 |

The peak is at k=20, not 25 — so bracketing was worth doing — but the whole
usable range k ∈ [15, 50] spans just **0.6453 – 0.6770**. Best case is +0.088
over the published value and then it decays.

Three things make this a representation result rather than a hyperparameter one:

1. **The two cohorts respond to k in opposite directions.** SurGen peaks at k=20
   and collapses to 0.5495 by k=100; PAIP climbs monotonically to 0.8330 at
   k=75. A hyperparameter that was simply mis-set would not invert its own
   gradient between cohorts.
2. **The oracle ceiling is itself low, and falls with k.** The best balanced
   accuracy *any* threshold could reach on SurGen kNN scores is 0.6297 at k=15
   and 0.6368 at k=20 — against 0.7545 for PAIP kNN and 0.68–0.70 for SurGen's
   own other heads. No thresholding rule can rescue scores that do not separate.
3. **No consistent optimum across configurations.** Of 20 SurGen configurations
   the best k is 15 for five, 20 for five, 25 for three, 35 for two and 50 for
   five. There is no k to pick, only a k to overfit.

### Q2 — Do the three schemes rank configurations consistently?

**No. The best scheme depends on the configuration, and (b) ranks differently
from the other two.**

Mean BalAcc, SurGen kNN:

| k | (a) frozen | (b) refit at k | (c) quantile | oracle |
|---|---|---|---|---|
| 15 | 0.5688 | 0.5688 | 0.5843 | 0.6297 |
| 20 | 0.5592 | 0.5750 | **0.5908** | 0.6368 |
| 25 | 0.5422 | 0.5628 | **0.5758** | 0.6320 |
| 35 | 0.5337 | **0.5709** | 0.5608 | 0.6212 |
| 50 | 0.5262 | **0.5760** | 0.5480 | 0.6102 |
| 75 | 0.5131 | **0.5701** | 0.5264 | 0.5936 |
| 100 | 0.5058 | **0.5170** | 0.5042 | 0.5505 |

The winner crosses over between k=25 and k=35 — (c) wins below, (b) above.

- **Per cell**, (b) wins 77 of 140 SurGen cells, (c) 40, (a) 23.
- **Only 3 of 20 SurGen configurations** keep the same winning scheme at every k
  (8 of 20 on PAIP).
- **Rank correlation across the 20 configurations** (Spearman on BalAcc):

  | | a~b | a~c | b~c |
  |---|---|---|---|
  | SurGen k=20 | +0.136 | +0.437 | **−0.167** |
  | SurGen k=35 | +0.028 | +0.591 | **−0.205** |
  | SurGen k=50 | +0.198 | +0.674 | −0.017 |
  | PAIP k=35 | +0.222 | +0.869 | +0.162 |
  | PAIP k=50 | −0.263 | +0.921 | **−0.348** |

  (a) and (c) rank configurations similarly — the quantile moves the cut point
  but inherits the frozen rate. (b) is close to *uncorrelated or
  anti-correlated* with (c). Choosing between them changes which configuration
  looks best, not just how well each scores.

- **Scheme (a) fails outright at scale.** Pooled over all k, SurGen: 49 dead,
  5 near-dead, 5 saturated of 140. (b) leaves 2 dead; (c) leaves 29 dead but
  never saturates. A frozen τ derived at k=3 has no meaning on a k=50 score
  scale — the distribution moved under it.

### Other heads, existing configuration

Schemes (a) and (c) only; scores untouched, so AUROC is identical between them.

**SurGen** — full 20-combination coverage since the H-Optimus-1 re-run of 3 Sep

| head | AUROC | (a) BalAcc | (c) BalAcc | oracle | dead under (a) → (c) |
|---|---|---|---|---|---|
| ANN | 0.7365 | 0.5888 | 0.6720 | 0.6996 | 0 → 0 |
| LR | 0.7090 | 0.5944 | 0.6493 | 0.6837 | 4 → 0 |
| ProtoNet | 0.7096 | 0.5228 | 0.6525 | 0.6779 | **16 → 0** |
| RF | 0.7026 | 0.5618 | 0.6422 | 0.6694 | 8 → 0 |

**PAIP**

| head | AUROC | (a) BalAcc | (c) BalAcc | oracle | dead under (a) → (c) |
|---|---|---|---|---|---|
| ANN | 0.8643 | 0.7553 | 0.8139 | 0.8536 | 0 → 0 |
| LR | 0.8717 | 0.7431 | 0.7890 | 0.8428 | 0 → 0 |
| ProtoNet | 0.8502 | 0.7273 | 0.7940 | 0.8274 | 3 → 0 |
| RF | 0.8359 | 0.6609 | 0.7724 | 0.8190 | 3 → 0 |

The rate-matched scheme lifts every head on both cohorts and clears every dead
threshold, and on SurGen it brings all four within 0.02–0.03 of their own oracle
— so for those heads the frozen τ, not the representation, is the binding
constraint. That is the opposite of the kNN finding, and the contrast is the
useful part.

*(PAIP RF (a) 0.6609 → (c) 0.7724 matches the promoted run's 0.6618 → 0.7724;
the small gap in the "before" figure is the mean_probs vs per-seed-average
distinction documented in Part 1.)*

---

## Coverage gaps

- **Coverage is now complete.** The 8 previously skipped heads (SurGen
  Averaging/H-Optimus-1 and TTC/H-Optimus-1, which stored 613-slide score
  vectors) were fixed on 3 Sep: the aggregated features were confirmed complete
  at 622 `.pt` files and both combinations were re-run through `ev_runner` on
  frozen τ. The non-kNN table now covers all 21 SurGen and 22 PAIP combinations
  (344 rows), and `part2_DIAGNOSTIC_skipped.csv` no longer exists.
- The kNN sweep never had this gap — it re-scores from features, so it was
  always self-consistent at the current cohort size. Its numbers are unchanged
  by the re-run.
- TITAN and PRISM are included in the CSVs but excluded from every mean above,
  which is computed over the 20 aggregation combinations only.

## If you want Part 1 applied to the published tree

```bash
# after taking a stamped backup of SurGen_EV_Results/
cp -r experiments/surgen_ev_diagnostic/frozen_baseline/* \
      slide_classification/SurGen_EV_Results/
python -u tools/build_report.py    # and the rest of the reporting layer
```

Headline values are already proven bit-identical, so the workbook, RESULTS.md,
docx and radar plots would not change — only the `BalAcc_corrected` /
`AUROC_corrected` / `Status_corrected` columns would disappear from the SurGen-EV
sheet. Decide whether you want that before running it.
