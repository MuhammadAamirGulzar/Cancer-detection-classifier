"""Crop-level grouping slide features for every encoder, TCGA and PAIP, from the published five-crop patch features.

  Caption_based_aggregation / _15_classes   every crop grouped by its own CONCH v1 zero-shot label
                                            (01_label_crops.py, published labelling protocol)
  Tissue_Type_Clustering                    every crop grouped by its own CRC-100K tissue label (--ttc-rule per_crop)

Averaging does not depend on grouping and is not rebuilt. One pass over each encoder's patch features builds
all three. Entirely zero-filled patch feature files (6 TCGA UNI2 files, empty since the August 2025 extraction)
are left out, exactly as the published slide features left them out (checked: identical to 2e-7). Output: features/<cohort>/<method>/<model>/<slide>.pt, the layout nocrop_runners.py reads with
ARM=cropgroup. Afterwards every output is loaded back and checked (unclean shutdowns leave zero-filled files).

  <conda>/envs/exaonepath/python.exe experiments/crop_grouping/02_build_slide_features.py
"""

import os
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = REPO.parent
FEAT = HERE / "features"
LABELS = BASE / "crop_captions"
COHORT_ROOT = {"tcga": BASE / "dataset", "paip": BASE / "paip_data"}
N_SLIDES = {"tcga": 417, "paip": 78}
METHODS = ["Caption_based_aggregation", "Caption_based_aggregation_15_classes", "Tissue_Type_Clustering"]
FIVECROP = {  # the five-crop patch features behind the published slide features
    "tcga": {"UNI2": "Uni_2h_fivecrop_Updated", "H-Optimus-1": "H_Optimus_1_fivecrop_Updated",
             "ConchV1": "Conch_v1_fivecrop", "Conch1_5": "Conch_v1_5_fivecrop_256", "Virchow2": "Virchow2_fivecrop"},
    "paip": {"UNI2": "Uni_2h_fivecrop_Updated", "H-Optimus-1": "H_Optimus_1_fivecrop_Updated",
             "ConchV1": "Conch_v1_fivecrop_Updated", "Conch1_5": "Conch_v1_5_fivecrop_Updated",
             "Virchow2": "Virchow2_fivecrop_Updated"},
}


def missing(cohort, model):
    """Slides not yet built for every method."""
    slides = {d.name for d in os.scandir(COHORT_ROOT[cohort] / "Features" / FIVECROP[cohort][model]) if d.is_dir()}
    assert len(slides) == N_SLIDES[cohort], (cohort, model, len(slides))
    out = set()
    for m in METHODS:
        d = FEAT / cohort / m / model
        have = {f[:-3] for f in os.listdir(d) if f.endswith(".pt")} if d.is_dir() else set()
        out |= slides - have
    return sorted(out)


def verify(cohort, model):
    bad = []
    for m in METHODS:
        for f in sorted((FEAT / cohort / m / model).glob("*.pt")):
            try:
                t = torch.load(f, map_location="cpu")
                if not torch.isfinite(t).all() or t.abs().sum() == 0:
                    bad.append(f)
            except Exception:
                bad.append(f)
    return bad


def main():
    n_labels = {c: len(list((LABELS / c).glob("*.npz"))) for c in COHORT_ROOT}
    for cohort in COHORT_ROOT:
        if n_labels[cohort] != N_SLIDES[cohort]:
            sys.exit(f"crop labels for {cohort}: {n_labels[cohort]} of {N_SLIDES[cohort]} slides; run 01_label_crops.py first")
    for cohort, models in FIVECROP.items():
        for model, folder in models.items():
            todo = missing(cohort, model)
            if not todo:
                print(f"{cohort} {model}: already built")
            else:
                cmd = [sys.executable, "-W", "ignore", str(REPO / "slide_aggregation" / "build_slide_features.py"),
                       "--cohort", cohort, "--model", model, "--features", str(COHORT_ROOT[cohort] / "Features" / folder),
                       "--out", str(FEAT / cohort), "--methods", ",".join(METHODS), "--ttc-rule", "per_crop",
                       "--caption-level", "crop", "--crop-labels", str(LABELS / cohort), "--skip-zero-filled"]
                if len(todo) < N_SLIDES[cohort]:
                    cmd += ["--slides", ",".join(todo)]
                p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
                lines = [l for l in (p.stdout + p.stderr).splitlines() if l.strip()]
                tail = [l for l in lines if "skipped zero-filled" in l or "UNREADABLE" in l or "  D:" in l] + lines[-2:]
                print(f"{cohort} {model}: exit {p.returncode}", *tail, sep="\n  ", flush=True)
                if p.returncode != 0 or missing(cohort, model):
                    sys.exit(f"{cohort} {model}: build incomplete")
            bad = verify(cohort, model)
            print(f"{cohort} {model}: {'all outputs readable and non-zero' if not bad else f'{len(bad)} BAD: {bad[:3]}'}",
                  flush=True)
            if bad:
                sys.exit(1)


if __name__ == "__main__":
    main()
