# No-crop vs five-crop patch features (TCGA-CV, PAIP-IV, PAIP-EV)

Status, 2026-10-15: **complete for TCGA and PAIP**, 5 encoders × 4 aggregations × 5 heads (100 paired configurations per setting). SurGen will be added once its processing finishes. This closes ledger item N19: five-crop was adopted in X05 without an ablation.

## Question

Every encoder uses five-crop features: five 256 px crops per 512 px patch, each resized to the encoder input and averaged at aggregation. Does a single view of the whole patch (512 px resized to the input size) do as well?

## Design: the crop is the only difference

- **Features.** No-crop patch features come from `feature_extraction/Feature_Extraction_nocrop_clean.ipynb`. Each encoder uses its own five-crop run's model loading, transform and precision: UNI2, CONCH v1 and CONCH v1.5 in fp32; H-Optimus-1 and Virchow2 in fp16 autocast. Recomputing stored five-crop features with that code reproduced them for every encoder in both cohorts (cosine ≥ 0.9999996).
- **Slide features.** `slide_aggregation/build_slide_features.py` reproduces the published Averaging, Caption-14/15 and Tissue-Type Clustering slide features exactly (max difference 0 to 4e-7). The no-crop slide features were built with the same code. A no-crop patch has a single row, so Tissue-Type Clustering classifies that one vector.
- **Five-crop arm.** The published results: `TCGA_Results`, `PAIP_IV_Results`, and `final_analysis/results/ev_scores_paip.csv`. TCGA CONCH v1.5 now uses the corrected 256-crop features (re-extracted 2026-10-13).
- **No-crop arm.** The same published runners, with all writes redirected by `nocrop_runners.py`. Run through the same wrapper on five-crop features, they reproduce the published TCGA-CV, PAIP-EV and PAIP-IV predictions exactly (max difference 1e-16 / 0 / 0). The published control files never changed.
- **Uncertainty.** The `final_analysis/02_uncertainty.py` bootstrap (class-stratified, B = 2000, same seed), paired slide by slide. Balanced accuracy is taken at each setting's published operating point: 0.5 (RF 0.3) for CV and IV, the uniform threshold for EV.

## Results

Mean over configurations; difference = no-crop − five-crop [95% CI], p.

| | TCGA-CV (413 slides) | PAIP-IV (31 test) | PAIP-EV (78, external) |
|---|---|---|---|
| AUROC, all 100 | 0.785 → 0.810, **+0.025** [+0.012, +0.038], p = 0.001 | 0.824 → 0.807, −0.016 [−0.042, +0.004], p = 0.13 | 0.847 → 0.847, +0.000 [−0.012, +0.013], p = 0.96 |
| BalAcc, all 100 | 0.675 → 0.698, **+0.023** [+0.012, +0.034], p = 0.001 | 0.748 → 0.736, −0.012, p = 0.35 | 0.776 → 0.777, +0.001, p = 0.87 |

By encoder (AUROC difference):

| Encoder | TCGA-CV | PAIP-IV | PAIP-EV |
|---|---|---|---|
| UNI2 | **+0.032** (p = 0.014) | −0.038 (p = 0.08) | +0.000 |
| CONCH v1 | **+0.044** (p = 0.001) | −0.019 | +0.005 |
| CONCH v1.5 | **+0.036** (p = 0.001) | −0.020 | +0.021 (p = 0.13) |
| Virchow2 | **+0.041** (p = 0.001) | −0.011 | +0.000 |
| H-Optimus-1 | **−0.026** (p = 0.001) | +0.007 | **−0.025** (p = 0.03) |

By aggregation (AUROC difference): TCGA-CV gains for Averaging (+0.033), Caption-14 (+0.027) and Caption-15 (+0.027), and not significantly for TTC (+0.014, p = 0.08). On PAIP-EV every aggregation is within ±0.012. TTC is the one aggregation that loses on PAIP-IV (−0.040, p = 0.025).

The best configurations are essentially unchanged. TCGA-CV: H-Optimus-1 / Caption-15 / LR, 0.928 five-crop vs 0.921 no-crop. PAIP-EV: best five-crop 0.915 (UNI2 / Averaging / LR) vs best no-crop 0.905 (Virchow2 / Caption-15 / ANN).

## Reading

- **No-crop helps within TCGA, but not externally.** The internal gain (+0.025 AUROC, +0.023 BalAcc) vanishes on PAIP-EV (+0.000). The supportable claim is that crop strategy changes TCGA-CV numbers, not that it generalises.
- **H-Optimus-1 is the exception.** It is the only encoder that loses without crops, and it does so internally (−0.026) and externally (−0.025). The other four gain within TCGA.
- **Five-crop remains the published choice.** It gives the best configuration in every setting, and no-crop gives no external gain.
- **PAIP-IV** has 31 test slides (7 MSI-H), so its intervals are wide. The consistently small negative differences are not significant overall.

## Limitations

- **Two factors change together.** No-crop changes field of view (whole patch vs 256 px crops) and resolution (about 2.3× lower for the 224 px encoders) at the same time.
- **Tissue-Type Clustering.** The published five-crop TTC uses the top-left crop's tissue label for all five crops (TCGA, and 73 of the 78 PAIP slides). The 5 gap-filled PAIP slides use per-crop labels. This was kept "as is" by decision.
- **Multiplicity.** Per-configuration p-values are not corrected for multiplicity; the setting / encoder / aggregation means are the summary results.
- **Scope.** One seed; SurGen not yet included.

## Data incidents during this experiment (resolved)

**Zero-filled files from unexpected shutdowns.** The PC shut down uncleanly about once a day (9, 10, 11, 13 Oct, and three times on 14 Oct; Kernel-Power 41, mostly without a bugcheck). Files written just before a shutdown were left at the right size but full of zeros:
- 60 UNI2, 64 H-Optimus and 64 CONCH v1 PAIP no-crop patch features in `training_data_38_nonMSIH`;
- the Caption-14 / UNI2 no-crop TCGA-FULL models of seeds 43–46.

All were regenerated (`03c_repair_unreadable.py`; regenerated copies of intact files identical to the originals). No zero-filled file remains anywhere in this experiment's outputs.

**Damaged patch PNGs.** Two PNGs were re-cut from the raw slides with the original `Patch_creation.ipynb` procedure (neighbours reproduced pixel-exactly): TCGA `TCGA-AG-3890_..._patch03233` and PAIP `validation_data_26_..._patch00867`.

## Files

- **Scripts** (the first two-encoder Averaging pass, `01`–`03`, was superseded by `05` and removed on 2026-10-22):
  - `03b_build_missing_slides.py`, `03c_repair_unreadable.py`: build the no-crop slide features and repair damaged files.
  - `nocrop_runners.py`: redirect wrapper. `ARM=fivecrop_check` was its validation run, in `runs/fivecrop_check/`.
  - `04_run_nocrop_pipeline.py`: full no-crop pipeline, output in `runs/nocrop/`.
  - `05_compare_all.py`: writes `results/all/per_config.csv` and `contrasts.csv`.
  - `06_mlflow_table.py`: writes `results/all/mlflow_table.csv`.
- **MLflow:** the clean store `D:/Aamir Gulzar/KSA_project2/mlflow_clean` (`tools/tracking/build_clean_store.py`, open with `tools/tracking/mlflow_ui_clean.bat`).
