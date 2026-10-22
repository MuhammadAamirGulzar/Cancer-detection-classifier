"""Step 4 - the published pipeline on the no-crop slide features, all 5 encoders x 4 aggregations.

Order (each call is its own process; see nocrop_runners.py for what is redirected):
  1. TCGA-CV per combination        -> runs/nocrop/TCGA_Results, runs/nocrop/thresholds_TCGA.json
  2. TCGA-FULL hyperparameters      -> runs/nocrop/tcga_full_hparams.json (derived from step 1 only)
  3. TCGA-FULL models per combination
  4. PAIP-EV per-slide scores (uniform rule) per combination -> runs/nocrop/ev/<method>__<model>/
  5. PAIP-IV per combination (ann_old preset, as published)  -> runs/nocrop/PAIP_IV_Results

A step whose output already exists is skipped, so the script can be rerun after an interruption.
Combinations whose slide features are missing for a cohort are reported and skipped.

  <conda>/envs/exaonepath/python.exe experiments/nocrop_vs_fivecrop/04_run_nocrop_pipeline.py
  ... 04_run_nocrop_pipeline.py --arm cropgroup --methods Caption_based_aggregation,...   (crop-level grouping)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARM_DIRS = {"nocrop": (HERE / "runs" / "nocrop", HERE / "features"),
            "cropgroup": (HERE.parent / "crop_grouping" / "runs" / "cropgroup", HERE.parent / "crop_grouping" / "features"),
            **{arm: (HERE.parent / "composition" / "runs" / arm, HERE.parent / "composition" / "features" / src, variant)
               for arm, src, variant in (("comp_fivecrop", "fivecrop", "append"), ("comp_nocrop", "nocrop", "append"),
                                         ("comp_only", "fivecrop", "only"))}}
ROOT, FEAT = ARM_DIRS["nocrop"]
ARM = "nocrop"
METHODS = ["Averaging", "Caption_based_aggregation", "Caption_based_aggregation_15_classes", "Tissue_Type_Clustering"]
MODELS = ["UNI2", "H-Optimus-1", "ConchV1", "Conch1_5", "Virchow2"]
N_SLIDES = {"tcga": 417, "paip": 78}


def run(*args):
    env = {**os.environ, "ARM": ARM, "PYTHONIOENCODING": "utf-8"}
    p = subprocess.run([sys.executable, "-W", "ignore", str(HERE / "nocrop_runners.py"), *args],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    tail = [l for l in (p.stdout + p.stderr).splitlines() if l.strip()][-3:]
    print(f"  {' '.join(args):70s} exit {p.returncode}  {tail[-1][:90] if tail else ''}", flush=True)
    if p.returncode != 0:
        raise SystemExit("\n".join(tail))


def have(cohort, method, model):
    d = FEAT / cohort / method / model
    if ARM.startswith("comp_"):
        d = FEAT / cohort / ("only" if ARM == "comp_only" else "append") / method / model
    return d.is_dir() and len([f for f in os.listdir(d) if f.endswith(".pt")]) == N_SLIDES[cohort]


def main():
    global ARM, ROOT, FEAT
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARM_DIRS), default="nocrop")
    ap.add_argument("--methods", default=",".join(METHODS))
    a = ap.parse_args()
    ARM, (ROOT, FEAT, *variant) = a.arm, ARM_DIRS[a.arm]
    methods = a.methods.split(",")
    assert all(m in METHODS for m in methods), methods
    combos = [(m, x) for m in methods for x in MODELS]
    ready = [(m, x) for m, x in combos if have("tcga", m, x) and have("paip", m, x)]
    for m, x in combos:
        if (m, x) not in ready:
            print(f"  not ready (slide features incomplete): {m}/{x}")
    tcga_out = lambda m, x: ROOT / "TCGA_Results" / m / x / "1-MSIH" / "Output" / "oof_predictions_default.csv"
    full_out = lambda m, x: ROOT / "TCGA_FULL_Models" / m / x / "1-MSIH" / "training_summary.json"
    ev_out = lambda m, x: ROOT / "ev" / f"{m}__{x}" / "ev_scores_paip.csv"
    iv_out = lambda m, x: ROOT / "PAIP_IV_Results" / m / x / "1-MSIH" / "Output" / "predictions_rf.csv"

    print("1. TCGA-CV")
    for m, x in ready:
        if not tcga_out(m, x).exists():
            run("cv", m, x)
    print("2. TCGA-FULL hyperparameters")
    run("hparams", *methods)
    print("3-5. TCGA-FULL models, PAIP-EV scores, PAIP-IV")
    for m, x in ready:
        if not full_out(m, x).exists():
            run("full", m, x)
        if not ev_out(m, x).exists():
            run("ev", m, x)
        if not iv_out(m, x).exists():
            run("iv", m, x)
    print(f"done: {len(ready)} of {len(combos)} combinations")


if __name__ == "__main__":
    main()
