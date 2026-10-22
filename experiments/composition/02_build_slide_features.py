"""Composition-aware slide features: the existing group features with each group's share appended.

Arms (5 encoders x Caption-14 / Caption-15 / TTC, TCGA and PAIP):
  fivecrop   published five-crop slide features (patch-level grouping) + sqrt(share) * scale
  nocrop     no-crop slide features (experiments/nocrop_vs_fivecrop/features) + sqrt(share) * scale
  only       sqrt(share) alone, from the five-crop arm (the composition-only reference)

The group features are rebuilt from the patch features and compared, slide by slide, with that arm's existing
slide tensors: they must match (< 1e-4). Published five-crop TTC uses the top-left rule except for 5 gap-filled
PAIP slides built per crop; a TTC slide that only matches under the per-crop rule is rebuilt with it. Any other
mismatch stops the run. Zero-filled patch features are skipped as the published features skipped them.

Output: features/<arm>/<cohort>/{append,only}/<method>/<model>/<slide>.pt, then every file is checked.

  <conda>/envs/exaonepath/python.exe experiments/composition/02_build_slide_features.py
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = REPO.parent
FEAT = HERE / "features"
COHORT_ROOT = {"tcga": BASE / "dataset", "paip": BASE / "paip_data"}
N_SLIDES = {"tcga": 417, "paip": 78}
METHODS = ["Caption_based_aggregation", "Caption_based_aggregation_15_classes", "Tissue_Type_Clustering"]
GROUPS = {"Caption_based_aggregation": 14, "Caption_based_aggregation_15_classes": 15, "Tissue_Type_Clustering": 9}
DIM = {"UNI2": 1536, "H-Optimus-1": 1536, "ConchV1": 512, "Conch1_5": 768, "Virchow2": 2560}
PATCH_FEATURES = {
    "fivecrop": {
        "tcga": {"UNI2": "Uni_2h_fivecrop_Updated", "H-Optimus-1": "H_Optimus_1_fivecrop_Updated",
                 "ConchV1": "Conch_v1_fivecrop", "Conch1_5": "Conch_v1_5_fivecrop_256", "Virchow2": "Virchow2_fivecrop"},
        "paip": {"UNI2": "Uni_2h_fivecrop_Updated", "H-Optimus-1": "H_Optimus_1_fivecrop_Updated",
                 "ConchV1": "Conch_v1_fivecrop_Updated", "Conch1_5": "Conch_v1_5_fivecrop_Updated",
                 "Virchow2": "Virchow2_fivecrop_Updated"}},
    "nocrop": {c: {"UNI2": "UNI2H_nocrop", "H-Optimus-1": "H_Optimus_1_nocrop", "ConchV1": "Conch_v1_nocrop",
                   "Conch1_5": "Conch_v1_5_nocrop", "Virchow2": "Virchow2_nocrop"} for c in COHORT_ROOT},
}
REFERENCE = {"fivecrop": {c: COHORT_ROOT[c] / "slide_aggregation" for c in COHORT_ROOT},
             "nocrop": {c: REPO / "experiments" / "nocrop_vs_fivecrop" / "features" / c for c in COHORT_ROOT}}
SCALES = json.loads((HERE / "scales.json").read_text())


def out_dirs(arm, cohort, model):
    root = FEAT / arm / cohort
    return [root / v / m / model for v in ("append", "only") for m in METHODS]


def readable(f):
    try:
        t = torch.load(f, map_location="cpu")
        return bool(torch.isfinite(t).all()) and t.abs().sum() > 0
    except Exception:
        return False


def todo(arm, cohort, model):
    """Slides with any output missing, unreadable or zero-filled (an unclean shutdown zero-fills recent files)."""
    slides = sorted(d.name for d in os.scandir(COHORT_ROOT[cohort] / "Features" / PATCH_FEATURES[arm][cohort][model])
                    if d.is_dir())
    assert len(slides) == N_SLIDES[cohort], (arm, cohort, model, len(slides))
    return [s for s in slides if not all((d / f"{s}.pt").exists() and readable(d / f"{s}.pt")
                                         for d in out_dirs(arm, cohort, model))]


def complete(arm, cohort, model):
    return not todo(arm, cohort, model)


def builder(arm, cohort, model, methods, rule, slides=None):
    scale = HERE / "scales" / f"{arm}__{model}.json"
    scale.parent.mkdir(exist_ok=True)
    scale.write_text(json.dumps(SCALES[arm][model]))
    cmd = [sys.executable, "-W", "ignore", str(REPO / "slide_aggregation" / "build_slide_features.py"),
           "--cohort", cohort, "--model", model,
           "--features", str(COHORT_ROOT[cohort] / "Features" / PATCH_FEATURES[arm][cohort][model]),
           "--methods", ",".join(methods), "--ttc-rule", rule, "--skip-zero-filled",
           "--check-against", str(REFERENCE[arm][cohort]),
           "--composition-out", str(FEAT / arm / cohort), "--composition-scale", str(scale)]
    if slides:
        cmd += ["--slides", ",".join(slides)]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    lines = [l for l in (p.stdout + p.stderr).splitlines() if l.strip()]
    keep = [l for l in lines if "max abs diff" in l or "MISMATCH" in l or "UNREADABLE" in l or "zero-filled" in l]
    print(f"{arm} {cohort} {model} ({rule}{', ' + str(len(slides)) + ' slides' if slides else ''}): exit {p.returncode}",
          *keep, *(lines[-3:] if p.returncode else []), sep="\n  ", flush=True)
    if p.returncode != 0 or any("UNREADABLE" in l for l in lines):
        sys.exit(f"{arm} {cohort} {model}: builder failed")
    mism = {}
    for l in lines:
        m = re.match(r"\s*MISMATCH (\S+): (.+)", l)
        if m:
            mism[m.group(1)] = m.group(2).split(",")
    return mism


def verify(arm, cohort, model):
    bad = 0
    for d in out_dirs(arm, cohort, model):
        method, variant = d.parent.name, d.parent.parent.name
        G = GROUPS[method]
        want = G if variant == "only" else G * DIM[model] + G
        norm = 1.0 if variant == "only" else SCALES[arm][model][method]     # |sqrt(share)| = 1, times the scale
        for f in d.glob("*.pt"):
            try:
                t = torch.load(f, map_location="cpu")
                ok = (t.numel() == want and bool(torch.isfinite(t).all())
                      and abs(float(t[-G:].norm()) - norm) < 1e-4 * norm)
            except Exception:
                ok = False
            bad += not ok
    return bad


def reference_check(arm, cohort, model):
    """Compare every append file's group features with the reference slide tensor (after a resume too: a crash between
    a build and its end-of-run check leaves readable but unchecked files).

    Returns {method: [slides]} that differ. Tiny numeric differences (cosine > 0.99999, e.g. a reference built from an
    older extraction of the same patches) are fixed in place by taking the reference group features exactly, so the
    arm stays "reference + shares"; they are logged and not returned.
    """
    out = {}
    for m in METHODS:
        G = GROUPS[m]
        for f in sorted((FEAT / arm / cohort / "append" / m / model).glob("*.pt")):
            t = torch.load(f, map_location="cpu")
            ref = torch.load(REFERENCE[arm][cohort] / m / model / f.name, map_location="cpu").float().flatten()
            head = t[:-G]
            if head.numel() == ref.numel() and (head - ref).abs().max() < 1e-4:
                continue
            cos = float(torch.nn.functional.cosine_similarity(head, ref, dim=0)) if head.numel() == ref.numel() else 0.0
            if cos > 0.99999 and not (m == "Tissue_Type_Clustering" and arm == "fivecrop" and cohort == "paip"):
                print(f"  {m} {f.stem}: max diff {(head - ref).abs().max():.1e}, cosine {cos:.7f}; "
                      f"reference group features used", flush=True)
                tmp = f.with_name(f.name + ".tmp")
                torch.save(torch.cat([ref, t[-G:]]), tmp)
                os.replace(tmp, f)
            else:
                out.setdefault(m, []).append(f.stem)
    return out


def main():
    for arm in ("fivecrop", "nocrop"):
        for cohort in ("tcga", "paip"):
            for model in DIM:
                left = todo(arm, cohort, model)
                if not left:
                    print(f"{arm} {cohort} {model}: already built")
                else:
                    part = left if len(left) < N_SLIDES[cohort] else None
                    mism = builder(arm, cohort, model, METHODS, "top_left", part)
                    ttc = mism.pop("Tissue_Type_Clustering", [])
                    if mism:
                        sys.exit(f"group features differ from the reference: {mism}")
                    if ttc:
                        if arm != "fivecrop" or cohort != "paip" or len(ttc) > 5:
                            sys.exit(f"unexpected TTC mismatch: {arm} {cohort} {model} {ttc}")
                        again = builder(arm, cohort, model, ["Tissue_Type_Clustering"], "per_crop", ttc)
                        if again:
                            sys.exit(f"TTC slides match neither rule: {again}")
                    if not complete(arm, cohort, model):
                        sys.exit(f"{arm} {cohort} {model}: outputs incomplete")
                mism = reference_check(arm, cohort, model)
                ttc = mism.pop("Tissue_Type_Clustering", [])
                if mism:
                    sys.exit(f"group features differ from the reference: {arm} {cohort} {model} {mism}")
                if ttc:
                    if arm != "fivecrop" or cohort != "paip" or len(ttc) > 5:
                        sys.exit(f"unexpected TTC mismatch: {arm} {cohort} {model} {ttc}")
                    if builder(arm, cohort, model, ["Tissue_Type_Clustering"], "per_crop", ttc) or                             reference_check(arm, cohort, model):
                        sys.exit(f"TTC slides match neither rule: {arm} {cohort} {model} {ttc}")
                print(f"{arm} {cohort} {model}: group features match the reference on every slide", flush=True)
                bad = verify(arm, cohort, model)
                print(f"{arm} {cohort} {model}: {'all outputs checked' if not bad else f'{bad} BAD files'}", flush=True)
                if bad:
                    sys.exit(1)


if __name__ == "__main__":
    main()
