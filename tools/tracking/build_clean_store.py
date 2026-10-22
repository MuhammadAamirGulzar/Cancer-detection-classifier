"""Build the clean MLflow store: the benchmark configurations, one consistent schema.

Store: D:/Aamir Gulzar/KSA_project2/mlflow_clean   (open with tools/tracking/mlflow_ui_clean.bat, port 5001)

  one MLflow experiment per setting   TCGA-CV, PAIP-IV, PAIP-EV   (SurGen-CV / SurGen-EV to be added)
  one run per configuration           encoder | aggregation | head | arm
  arms
    fivecrop   five-crop patch features (FiveCrop(256) of 512 px patches). Averaging, TITAN and PRISM as published;
               Caption-14 / Caption-15 / Tissue-Type Clustering with CROP-LEVEL grouping: every crop is grouped by
               its own label (CONCH v1 zero-shot caption at 224 px; CRC-100K tissue classifier, per crop)
    nocrop     one whole-patch feature per patch (512 px resized to the encoder input); patch-level grouping
  params (identical on every run)     setting, validation, train_cohort, test_cohort, encoder, aggregation, head,
                                      crop, grouping, caption_level, ttc_rule, precision, threshold_rule, n_test,
                                      n_pos, features
  tags                                role (baseline = fivecrop, ablation = nocrop), study, arm
  metrics                             auroc, auroc_ci_lo/hi, bacc, bacc_ci_lo/hi; nocrop runs also
                                      delta_auroc / delta_bacc (+ ci, p) = paired no-crop - five-crop (same aggregation)

Inputs:
  experiments/nocrop_vs_fivecrop/results/all/mlflow_table.csv   five-crop Averaging / TITAN / PRISM, no-crop (06)
  experiments/crop_grouping/results/mlflow_table.csv            five-crop crop-level Caption-14/15, TTC (05)
  experiments/crop_grouping/results/contrasts.csv               paired crop-level - no-crop differences, which give
                                                                the no-crop deltas for Caption-14/15 and TTC

The store is rebuilt from scratch (--fresh deletes it first), so no deleted or superseded runs remain in it.

  <conda>/envs/mlflow_tracking/python.exe tools/tracking/build_clean_store.py --fresh
"""

import argparse
import math
import os
import shutil
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow
import pandas as pd
from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient

REPO = Path(__file__).resolve().parents[2]
STORE = Path(r"D:/Aamir Gulzar/KSA_project2/mlflow_clean")
EXP = REPO / "experiments"
NOCROP_TABLE = EXP / "nocrop_vs_fivecrop" / "results" / "all" / "mlflow_table.csv"
CROP_TABLE = EXP / "crop_grouping" / "results" / "mlflow_table.csv"
CROP_CONTRASTS = EXP / "crop_grouping" / "results" / "contrasts.csv"
GROUPED = ("Caption-14", "Caption-15", "TTC")
DELTAS = ["delta_auroc", "delta_auroc_ci_lo", "delta_auroc_ci_hi", "delta_auroc_p",
          "delta_bacc", "delta_bacc_ci_lo", "delta_bacc_ci_hi", "delta_bacc_p"]
STUDY = {"fivecrop": "main", "nocrop": "crop_ablation (experiments/nocrop_vs_fivecrop)"}

SETTING = {
    "TCGA-CV": dict(validation="internal (4-fold CV, pooled out-of-fold)", train_cohort="TCGA", test_cohort="TCGA",
                    threshold_rule="0.5 (RF 0.3), as published"),
    "PAIP-IV": dict(validation="internal (provider split 47/31)", train_cohort="PAIP", test_cohort="PAIP",
                    threshold_rule="0.5 (RF 0.3), ANN preset ann_old, as published"),
    "PAIP-EV": dict(validation="external", train_cohort="TCGA", test_cohort="PAIP",
                    threshold_rule="uniform: rate-matched quantile; kNN k=35, tau refit on TCGA OOF"),
}
DESCRIPTION = {
    "TCGA-CV": "MSI-H, TCGA 413 slides (60 MSI-H), 4-fold CV.",
    "PAIP-IV": "MSI-H, PAIP train 47 / test 31 (7 MSI-H).",
    "PAIP-EV": "MSI-H, TCGA-trained models on PAIP 78 slides (19 MSI-H).",
}
NOTE = (" Arms: fivecrop = five-crop features (Caption-14/15 and TTC with crop-level grouping), baseline;"
        " nocrop = one whole-patch feature per patch, with paired deltas to fivecrop.")


def assemble():
    """One row per configuration and arm, with the no-crop deltas against the five-crop rows kept here."""
    nc = pd.read_csv(NOCROP_TABLE)
    crop = pd.read_csv(CROP_TABLE)
    five = pd.concat([nc[(nc.arm == "fivecrop") & ~nc.aggregation.isin(GROUPED)],
                      crop.assign(arm="fivecrop")], ignore_index=True)
    five[DELTAS] = float("nan")

    no = nc[nc.arm == "nocrop"].copy()
    con = pd.read_csv(CROP_CONTRASTS)
    con = con[(con.level == "config") & (con.comparison == "crop-nocrop")].copy()
    parts = con.contrast.str.split(" / ", expand=True)
    con["aggregation"], con["model"], con["classifier"] = parts[0], parts[1], parts[2]
    # crop-level - no-crop -> no-crop - crop-level: negate the difference and its bootstrap interval
    flip = con[["setting", "aggregation", "model", "classifier"]].assign(
        delta_auroc=-con.auroc_diff, delta_auroc_ci_lo=-con.auroc_hi, delta_auroc_ci_hi=-con.auroc_lo,
        delta_auroc_p=con.auroc_p, delta_bacc=-con.bacc_diff, delta_bacc_ci_lo=-con.bacc_hi,
        delta_bacc_ci_hi=-con.bacc_lo, delta_bacc_p=con.bacc_p)
    key = ["setting", "aggregation", "model", "classifier"]
    g = no.aggregation.isin(GROUPED)
    regrouped = no[g].drop(columns=DELTAS).merge(flip, on=key, how="left")
    no = pd.concat([no[~g], regrouped], ignore_index=True)
    assert no.delta_auroc.notna().all(), "every no-crop row needs its paired five-crop delta"

    t = pd.concat([five, no], ignore_index=True)
    assert not t.duplicated(["setting", "arm", "aggregation", "model", "classifier"]).any()
    return t


def params_for(r):
    grouped = r.aggregation in GROUPED
    grouping = "n/a" if not grouped else ("crop" if r.arm == "fivecrop" else "patch (one view per patch)")
    caption = "n/a"
    if r.aggregation.startswith("Caption"):
        caption = ("crop (each crop grouped by its own CONCH v1 zero-shot label at 224 px)" if r.arm == "fivecrop"
                   else "patch (whole-patch CONCH v1 zero-shot label at 224 px)")
    ttc = "n/a"
    if r.aggregation == "TTC":
        ttc = ("per_crop (each crop grouped by its own tissue label)" if r.arm == "fivecrop"
               else "one row per patch (no crops)")
    crop = "own pass (slide encoder)" if r.aggregation in ("TITAN", "PRISM") else r.arm
    feats = "published"
    if r.model == "Conch1_5" and r.arm == "fivecrop" and r.aggregation != "TITAN":
        feats = "Conch_v1_5_fivecrop_256 (TCGA re-extracted at 256 on 2026-10-13)"
    return {"setting": r.setting, **SETTING[r.setting], "encoder": r.model, "aggregation": r.aggregation,
            "head": r.classifier, "crop": crop, "grouping": grouping, "caption_level": caption, "ttc_rule": ttc,
            "precision": r.precision, "n_test": int(r.n_test), "n_pos": int(r.n_pos), "features": feats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="delete the store first and rebuild it from the tables")
    a = ap.parse_args()
    t = assemble()
    if a.fresh and STORE.exists():
        shutil.rmtree(STORE)
    if (STORE / "mlflow.db").exists():
        raise SystemExit(f"{STORE} already holds a store; pass --fresh to rebuild it")
    (STORE / "artifacts").mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{STORE.as_posix()}/mlflow.db")
    client = MlflowClient()

    for setting, g in t.groupby("setting"):
        eid = client.create_experiment(setting, artifact_location=(STORE / "artifacts" / setting).as_uri(),
                                       tags={"mlflow.note.content": DESCRIPTION[setting] + NOTE})
        for _, r in g.iterrows():
            run = client.create_run(eid, run_name=f"{r.model} | {r.aggregation} | {r.classifier} | {r.arm}",
                                    tags={"role": "baseline" if r.arm == "fivecrop" else "ablation",
                                          "study": STUDY[r.arm], "arm": r.arm})
            metrics = []
            for k in ["auroc", "auroc_ci_lo", "auroc_ci_hi", "bacc", "bacc_ci_lo", "bacc_ci_hi"] + DELTAS:
                v = r.get(k)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    metrics.append(Metric(k, float(v), 0, 0))
            # one transaction per run: per-value calls are ~30x slower on the HDD
            client.log_batch(run.info.run_id, metrics=metrics,
                             params=[Param(k, str(v)) for k, v in params_for(r).items()])
            client.set_terminated(run.info.run_id)
        print(f"{setting}: {len(g)} runs (" + ", ".join(f"{a} {(g.arm == a).sum()}" for a in STUDY) + ")")
    print(f"store: {STORE}")


if __name__ == "__main__":
    main()
