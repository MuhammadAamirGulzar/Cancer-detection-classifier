"""Run the published runners unchanged on the NO-CROP slide features, with every write redirected here.

The five-crop arm of the comparison is the published pipeline itself (TCGA_Results, PAIP_IV_Results,
experiments/final_analysis/results/ev_scores_paip.csv). The no-crop arm must go through exactly the same
code, so instead of re-implementing anything this wrapper imports the runners and redirects, before they
run:

  feature lookup        config.paths.feature_dir      -> features/<cohort>/<method>/<model>
  results trees         config.paths.results_root     -> runs/nocrop/<same relative layout>
  feature-matrix cache  config.paths.cache_dir        -> runs/nocrop/cache
  run log / resume      runlog.PROGRESS_PATH, SKIPPED_PATH, LOG_DIR -> runs/nocrop/
  tau_TCGA              thresholds.THRESHOLDS_PATH    -> runs/nocrop/thresholds_TCGA.json
  TCGA-FULL hparams     hparams.HPARAMS_PATH          -> runs/nocrop/tcga_full_hparams.json
  kNN k=35 taus         threshold_modes.KNN_K35_THRESHOLDS_PATH -> runs/nocrop/knn_k35_thresholds.json
                        (absent: the runner refits tau by the same rule - verified identical to the
                        stored taus on unchanged combinations)

Every call fails if a published control file changed while it ran.

ARM=cropgroup reads the crop-level grouping slide features (experiments/crop_grouping/features) and writes to
experiments/crop_grouping/runs/cropgroup/.

ARM=comp_fivecrop / comp_nocrop / comp_only read the composition-aware slide features of
experiments/composition (group features + sqrt(share) * scale, or the shares alone) and write to
experiments/composition/runs/<ARM>/; config.paths.expected_dim is widened by the G appended values (or set to G).

ARM=fivecrop_check (environment variable) leaves the feature lookup on the PUBLISHED five-crop slide features
and writes to runs/fivecrop_check/: the outputs must then equal the published results, which validates
this wrapper.

Usage (exaonepath env), one combination per process for cv / iv:
  python nocrop_runners.py cv  <method> <model>     TCGA-CV            (cv_runner, as published)
  python nocrop_runners.py hparams [methods...]     tcga_full_hparams  (runners.hparams.build_all; default all 4)
  python nocrop_runners.py full <method> <model>    TCGA-FULL models   (full_trainer)
  python nocrop_runners.py ev   <method> <model>    PAIP-EV per-slide scores, uniform rule
                                                    (final_analysis/01_uniform_ev_scores.evaluate)
  python nocrop_runners.py iv   <method> <model>    PAIP-IV            (iv_runner --ann-protocol full --ann-preset ann_old)
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SC = REPO / "slide_classification"
ARM = os.environ.get("ARM", "nocrop")
COMP = {"comp_fivecrop": ("fivecrop", "append"), "comp_nocrop": ("nocrop", "append"), "comp_only": ("fivecrop", "only")}
assert ARM in ("nocrop", "fivecrop_check", "cropgroup", *COMP), ARM
if ARM == "cropgroup":
    ROOT = REPO / "experiments" / "crop_grouping" / "runs" / ARM
    FEAT = REPO / "experiments" / "crop_grouping" / "features"
elif ARM in COMP:
    ROOT = REPO / "experiments" / "composition" / "runs" / ARM
    FEAT = REPO / "experiments" / "composition" / "features" / COMP[ARM][0]
else:
    ROOT = HERE / "runs" / ARM
    FEAT = HERE / "features"
sys.path.insert(0, str(SC))

from config import paths as P  # noqa: E402

PUBLISHED_CONTROL = [SC / "thresholds_TCGA.json", SC / "tcga_full_hparams.json",
                     SC / "knn_k35_thresholds.json", REPO / "progress.json"]


def _redirect():
    ROOT.mkdir(parents=True, exist_ok=True)
    orig_results_root = P.results_root

    def results_root(*a, **k):
        return ROOT / orig_results_root(*a, **k).relative_to(P.SLIDE_CLS_ROOT)

    def feature_dir(cohort, method, model):
        if ARM in COMP:
            return FEAT / cohort.lower() / COMP[ARM][1] / method / model
        return FEAT / cohort.lower() / method / model

    if ARM in COMP:
        orig_expected_dim = P.expected_dim

        def expected_dim(method, model):
            G = P.AGG_MULTIPLIERS[method]
            return G if ARM == "comp_only" else orig_expected_dim(method, model) + G
        P.expected_dim = expected_dim

    def cache_dir():
        d = ROOT / "cache"
        d.mkdir(parents=True, exist_ok=True)
        return d

    P.results_root, P.cache_dir = results_root, cache_dir
    if ARM != "fivecrop_check":
        P.feature_dir = feature_dir

    from runners import runlog, thresholds, hparams, threshold_modes
    runlog.PROGRESS_PATH = ROOT / "progress.json"
    runlog.SKIPPED_PATH = ROOT / "skipped_combinations.csv"
    runlog.LOG_DIR = ROOT / "logs"
    thresholds.THRESHOLDS_PATH = ROOT / "thresholds_TCGA.json"
    hparams.HPARAMS_PATH = ROOT / "tcga_full_hparams.json"
    threshold_modes.KNN_K35_THRESHOLDS_PATH = ROOT / "knn_k35_thresholds.json"

    # the redirection must hold before anything runs
    assert str(P.results_root("TCGA-CV", "Averaging", "UNI2")).startswith(str(ROOT))
    assert str(P.feature_dir("paip", "Averaging", "UNI2")).startswith(str(FEAT)) == (ARM != "fivecrop_check")


def _stamp():
    return {str(p): (p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else None for p in PUBLISHED_CONTROL}


def main():
    cmd, args = sys.argv[1], sys.argv[2:]
    before = _stamp()
    _redirect()

    if cmd == "cv":
        method, model = args
        from runners import cv_runner
        sys.argv = ["cv_runner.py", "--experiment", "TCGA-CV", "--methods", method, "--models", model,
                    "--force", "--no-archive"]
        cv_runner.main()
    elif cmd == "hparams":
        from runners import hparams
        hparams.build_all(methods=args or ["Averaging", "Caption_based_aggregation",
                                           "Caption_based_aggregation_15_classes", "Tissue_Type_Clustering"])
    elif cmd == "full":
        method, model = args
        from runners import full_trainer
        full_trainer.sweep(methods=[method], models=[model], force=True)
    elif cmd == "ev":
        method, model = args
        spec = importlib.util.spec_from_file_location("ev01", REPO / "experiments" / "final_analysis" / "01_uniform_ev_scores.py")
        ev01 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ev01)
        import pandas as pd
        rows, summary = ev01.evaluate("paip", method, model)
        cols = ["cohort", "method", "model", "classifier", "slide_id", "case_id", "target", "score_uniform",
                "threshold_uniform", "score_seed_mean", "threshold_frozen", "score_fold_ensemble"]
        out = ROOT / "ev" / f"{method}__{model}"
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=cols).to_csv(out / "ev_scores_paip.csv", index=False, float_format="%.10g")
        pd.DataFrame(summary).to_csv(out / "ev_uniform_summary.csv", index=False, float_format="%.10g")
        best = max(summary, key=lambda d: d["auroc_uniform"])
        print(f"PAIP-EV {method}/{model}: best AUROC {best['auroc_uniform']:.4f} ({best['classifier']})")
    elif cmd == "iv":
        method, model = args
        from runners import iv_runner
        sys.argv = ["iv_runner.py", "--methods", method, "--models", model, "--force", "--no-archive",
                    "--ann-protocol", "full", "--ann-preset", "ann_old"]
        iv_runner.main()
    else:
        raise SystemExit(f"unknown command {cmd!r}")

    after = _stamp()
    changed = [p for p in before if before[p] != after[p]]
    if changed:
        raise SystemExit(f"PUBLISHED CONTROL FILE CHANGED: {changed}")


if __name__ == "__main__":
    main()
