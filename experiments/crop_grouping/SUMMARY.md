# Crop-level vs patch-level grouping (TCGA-CV, PAIP-IV, PAIP-EV)

Status, 2026-10-16: **complete for TCGA and PAIP**, 5 encoders × 3 grouping aggregations (Caption-14, Caption-15, Tissue-Type Clustering) × 5 heads (75 paired configurations per setting). SurGen will be added once its processing finishes.

**Decision, 2026-10-22 (project owner):** crop-level grouping **replaces** the patch-level grouping as the five-crop method for Caption-14, Caption-15 and Tissue-Type Clustering. It is the faithful version of the intended method; the published TTC top-left rule was a bug. The clean MLflow store (`tools/tracking/build_clean_store.py`) now holds five-crop = Averaging, TITAN and PRISM as published, plus these crop-level runs, against no-crop. The two give the same results within noise (see below), so the change does not alter any conclusion. The headline analysis (`experiments/final_analysis/`, `reports/PROJECT_STATUS.md`) was regenerated from these runs on 2026-10-23 for TCGA and PAIP (ledger X59).

## Question

The published Caption and TTC aggregations use the five-crop features, but never group the crops individually:

- **Caption-14/15**: one CONCH v1 zero-shot label per patch, from the whole 512 px patch resized to 224 px. All 5 crops of a patch go into that patch's class group.
- **TTC**: every crop takes the tissue label of the patch's top-left crop (TCGA, and 73 of the 78 PAIP slides).

Hypothesis: grouping **each crop by its own label** gives cleaner class groups and better slide features. For example, T1 = {p1c2, p1c4, p2c5}. The slide tensor keeps the same shape (14, 15 or 9 groups × D).

## Design: the grouping is the only difference

- **Crop labels** (`01_label_crops.py`).
  - Every FiveCrop(256) crop was labelled the way the published labels were made: CONCH v1 caption checkpoint, `force_image_size=224`, bicubic resize, `encode_image()` @ the same zero-shot weights, argmax, fp32. Crops are in the same order as the stored feature rows.
  - The whole patch was labelled in the same pass and checked against the published CSVs on every patch: TCGA 1,377,933 / 1,377,933 (14 classes) and 1,355,579 / 1,355,579 (15 classes); PAIP 275,541 / 275,541 and 271,774 / 271,774. Scores agree to about 1e-6.
  - Labels are stored in `KSA_project2/crop_captions/<cohort>/<slide>.npz`, including the full similarity vectors for later soft-assignment work.
- **Slide features** (`02_build_slide_features.py` → `slide_aggregation/build_slide_features.py --caption-level crop --ttc-rule per_crop`).
  - The patch features are the published five-crop features.
  - Each caption variant uses the same patch set as the published labels: only patches that have a label in that CSV.
  - Group mean over crops, zeros for an empty group.
  - With the default `--caption-level patch` the builder still reproduces the published slide features exactly.
- **Runs.** The published runners via the redirect wrapper (`nocrop_runners.py`, `ARM=cropgroup`). Outputs are in `runs/cropgroup/`; the published control files never changed.
- **Comparison** (`04_compare.py`). The same paired, class-stratified bootstrap (B = 2000, same seed) and the same operating points as the no-crop study. The patch-level and no-crop arms here reproduce that study's per-configuration numbers exactly.

## Crops often disagree with their patch's label

| | TCGA | PAIP |
|---|---|---|
| Patches whose 5 crops all get the same caption label (14 / 15 classes) | 25.4% / 25.5% | 36.1% / 33.5% |
| Crops whose own label equals the patch's published label (14 / 15) | 56.9% / 56.2% | 66.1% / 64.7% |

So crop-level grouping moves about 35–45% of crops into a different class group. The slide features change considerably: per-group cosine to the patch-level feature is as low as 0.3 on some slides.

## Results

Mean over configurations; difference = crop-level − patch-level (published) [95% CI], p.

| | TCGA-CV (413) | PAIP-IV (31 test) | PAIP-EV (78, external) |
|---|---|---|---|
| AUROC, all 75 | 0.788 → 0.786, −0.003 [−0.012, +0.008], p = 0.63 | 0.845 → 0.846, +0.001 [−0.019, +0.023], p = 0.92 | 0.856 → 0.855, −0.002 [−0.014, +0.009], p = 0.76 |
| BalAcc, all 75 | 0.673 → 0.672, −0.002, p = 0.66 | 0.770 → 0.767, −0.003, p = 0.71 | 0.787 → 0.786, −0.002, p = 0.78 |

By aggregation (AUROC difference):

| | TCGA-CV | PAIP-IV | PAIP-EV |
|---|---|---|---|
| Caption-14 | −0.006 (p = 0.53) | +0.017 (p = 0.26) | +0.004 (p = 0.58) |
| Caption-15 | +0.007 (p = 0.42) | −0.004 (p = 0.82) | −0.009 (p = 0.29) |
| TTC (per-crop vs top-left) | −0.009 (p = 0.18) | −0.008 (p = 0.29) | −0.000 (p = 0.91) |

By encoder, only one AUROC difference reaches p < 0.05: UNI2 on TCGA-CV, −0.014 (p = 0.017, uncorrected). Crop-level beats patch-level in 36 / 75 (TCGA-CV), 32 / 75 (PAIP-IV) and 33 / 75 (PAIP-EV) configurations, i.e. about half, as expected under no effect.

Best configurations:

| | Patch-level (published) | Crop-level | No-crop |
|---|---|---|---|
| TCGA-CV | H-Optimus-1 / Caption-15 / LR 0.928 | H-Optimus-1 / Caption-15 / LR 0.921 | H-Optimus-1 / Caption-15 / LR 0.921 |
| PAIP-EV | UNI2 / Caption-14 / ANN 0.912 | UNI2 / Caption-15 / LR 0.905 | Virchow2 / Caption-15 / ANN 0.905 |
| PAIP-IV | Conch1_5 / Caption-15 / kNN 0.926 | UNI2 / TTC / kNN 0.970 | H-Optimus-1 / Caption-15 / kNN 0.914 |

The PAIP-IV maximum of 0.970 is the best of 75 configurations on 31 test slides (7 MSI-H). The setting means show no gain, so this is selection noise, not a finding.

Crop-level vs no-crop: on TCGA-CV crop-level grouping is lower (−0.025 AUROC, p = 0.001), the same size as the earlier no-crop − five-crop gain. On PAIP-IV and PAIP-EV the two are not significantly different (+0.021, p = 0.29; +0.002, p = 0.79).

## Reading

- **The hypothesis is not supported.** Grouping crops by their own labels changes which crops go into each group for about 40% of crops, but leaves slide-level discrimination unchanged in every setting. AUROC and BalAcc differences are all within ±0.01 on average, and no aggregation shows a significant effect.
- **Probable reason.** Each group mean averages thousands of crops from the same slide. Moving the 35–45% of crops whose label disagrees mostly reshuffles similar tissue between neighbouring classes, so the information a head can use barely changes. Crop-level zero-shot labels also see less context (256 px instead of 512 px) and are likely noisier, which may cancel any gain from finer grouping.
- **Five-crop question, combined with the no-crop study.** Using the crops more fully does not unlock a benefit from five-crop augmentation. Five-crop with patch-level grouping stays the published choice: it has the best configuration in TCGA-CV and PAIP-EV and is never significantly worse. No-crop is better within TCGA but not externally.

## Limitations

- **Hard labels only.** The stored similarity vectors allow soft (probability-weighted) assignment and composition features (tissue / caption proportions) as follow-ups without re-labelling.
- **Multiplicity.** Per-configuration p-values are not corrected; the setting / aggregation / encoder means are the summary results.
- **Scope.** One seed; SurGen not yet included.

## Data notes

- **Damaged patch PNG.** `TCGA-DM-A1D7_nonMSIH_x47616_y9888_patch01911.png` had a broken data stream.
  - It was re-cut from the raw SVS with the `Patch_creation` procedure; the four neighbouring patches came out pixel-identical.
  - The re-cut patch reproduces the published caption scores exactly (0.629354, 0.595127), so the published features and labels were made from the healthy image.
  - The damaged copy is kept in the session scratch `damaged_backup/`.
- **Zero-filled patch features since the 2025 extraction.**
  - 6 UNI2 files (TCGA-5M-AAT4 ×4, TCGA-CM-4743 ×2) and 22 H-Optimus-1 files (TCGA-5M-AAT4 ×6, TCGA-CM-6168 ×16) are entirely zero bytes (counted directly on 2026-10-29; an earlier count of 56 double-counted log lines).
  - The published slide features left exactly these out: rebuilding without them reproduces the published UNI2 and H-Optimus-1 slide features to ≤ 5e-7.
  - `--skip-zero-filled` leaves them out the same way and logs each one.
- **Crashes during the run.** Three access violations (0xC0000005, faulting module unknown, with a WHEA corrected hardware error the same day) and one unclean shutdown (Kernel-Power 41, 16 Oct). Every step resumed from finished outputs, and all 1,069 files written in the night before the shutdown were checked readable and non-zero.

## Files

- `01_label_crops.py`: crop + whole-patch zero-shot labels, verified against the published CSVs.
- `02_build_slide_features.py`: crop-level Caption-14/15 and per-crop TTC slide features into `features/`, each output checked.
- Step 3: `../nocrop_vs_fivecrop/04_run_nocrop_pipeline.py --arm cropgroup --methods Caption_based_aggregation,Caption_based_aggregation_15_classes,Tissue_Type_Clustering` (TCGA-CV, TCGA-FULL, PAIP-EV, PAIP-IV) into `runs/cropgroup/`.
- `04_compare.py`: writes `results/per_config.csv` and `results/contrasts.csv` (column `comparison`: crop-patch, crop-nocrop).
- `05_mlflow_table.py`: writes `results/mlflow_table.csv`.
- **MLflow:** the clean store (`tools/tracking/build_clean_store.py`), arm `cropgroup`, param `grouping` = patch / crop, with `delta_*` against the published patch-level run.
