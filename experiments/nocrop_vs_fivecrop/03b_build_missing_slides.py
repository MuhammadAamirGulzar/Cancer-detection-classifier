"""Build the no-crop slide features that are still missing, and list every unreadable patch-feature file.

For each (cohort, encoder) the slides without all four slide-feature files are passed to
slide_aggregation/build_slide_features.py (--ttc-rule top_left; a no-crop patch has one row, so the rule
makes no difference). The builder skips a slide with an unreadable patch file and lists the file; that list
is written to results/unreadable_<cohort>_<encoder>.txt so the files can be regenerated.

  <conda>/envs/exaonepath/python.exe experiments/nocrop_vs_fivecrop/03b_build_missing_slides.py
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = REPO.parent
FEAT = HERE / "features"
METHODS = ["Averaging", "Caption_based_aggregation", "Caption_based_aggregation_15_classes", "Tissue_Type_Clustering"]
SOURCES = {"UNI2": "UNI2H_nocrop", "H-Optimus-1": "H_Optimus_1_nocrop", "ConchV1": "Conch_v1_nocrop",
           "Conch1_5": "Conch_v1_5_nocrop", "Virchow2": "Virchow2_nocrop"}
COHORT_ROOT = {"tcga": BASE / "dataset", "paip": BASE / "paip_data"}


def missing(cohort, model):
    src = COHORT_ROOT[cohort] / "Features" / SOURCES[model]
    slides = sorted(d.name for d in os.scandir(src) if d.is_dir())
    return [s for s in slides if not all((FEAT / cohort / m / model / f"{s}.pt").exists() for m in METHODS)]


def main():
    (HERE / "results").mkdir(exist_ok=True)
    for cohort in ("paip", "tcga"):
        for model, src in SOURCES.items():
            todo = missing(cohort, model)
            if not todo:
                print(f"{cohort} {model}: complete", flush=True)
                continue
            print(f"{cohort} {model}: building {len(todo)} slide(s)", flush=True)
            p = subprocess.run(
                [sys.executable, "-W", "ignore", str(REPO / "slide_aggregation" / "build_slide_features.py"),
                 "--cohort", cohort, "--model", model,
                 "--features", str(COHORT_ROOT[cohort] / "Features" / src),
                 "--out", str(FEAT / cohort), "--ttc-rule", "top_left", "--slides", ",".join(todo)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            bad = [l.strip().split("  (")[0] for l in p.stdout.splitlines() if l.startswith("  ") and ".pt" in l]
            log = HERE / "results" / f"unreadable_{cohort}_{model}.txt"
            log.write_text("\n".join(bad), encoding="utf-8")
            left = missing(cohort, model)
            print(f"  exit {p.returncode}; unreadable files {len(bad)} (-> {log.name}); "
                  f"slides still missing {len(left)}", flush=True)
            if p.returncode != 0:
                print(p.stderr[-2000:], flush=True)


if __name__ == "__main__":
    main()
