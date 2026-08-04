"""Before/after evidence for work order Task 2.1 (SurGen case-level folds).

Runs SurGen-CV twice through **identical corrected code**, changing only how
folds are built:

  OLD  slide-level round-robin  - 47 of 70 multi-slide cases split across folds,
                                  so the same patient appears in train and test
  NEW  case-level round-robin   - whole cases kept together, 0 split

Any metric difference is therefore attributable to the fold construction alone,
not to the Phase 1 corrections (which are present in both arms).

The work order expects SurGen-CV numbers to **drop**: two sections from one
tumour are far more alike than two different tumours, so sibling leakage inflates
every SurGen number. A drop is the correction working, not a regression.

Nothing is written to any results tree - this is measurement only.

Run:  python tools/surgen_fold_audit.py [--combos M/N,M/N] [--classifiers ...]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "slide_classification"))

import data_layer as dl  # noqa: E402
from runners.classifiers import train_and_evaluate  # noqa: E402
from runners.surgen_folds import (  # noqa: E402
    build_case_level_folds, build_slide_level_folds_legacy, case_id_of, quantify_leakage,
)

METRICS = ("bacc", "auroc", "macro_f1", "acc")


def run_scheme(coh, fold_map, classifiers, check_leakage, seed=42):
    """4-fold CV over a given fold map. Returns {classifier: {metric: mean}}."""
    splits = dl.build_splits(coh.ids, fold_map)
    out = {}
    for clf in classifiers:
        per_fold = {m: [] for m in METRICS}
        for f in sorted(splits):
            tr, va, te = dl.cv_rotation(splits, f)
            train_ids = [coh.ids[i] for i in tr] + [coh.ids[i] for i in va]
            test_ids = [coh.ids[i] for i in te]
            if check_leakage:
                # Only the NEW scheme is expected to survive this.
                dl.assert_no_leakage(train_ids, test_ids, case_id_of, label=f"fold{f}")
            metrics, _, _ = train_and_evaluate(
                fold=f - 1,
                train=coh.subset(tr)[:2], valid=coh.subset(va)[:2], test=coh.subset(te)[:2],
                model_type=clf, input_dim=coh.dim, model_save_path=None,
                seed=seed, verbose=False,
            )
            for m in METRICS:
                per_fold[m].append(metrics[f"{clf}_{m}"])
        out[clf] = {m: float(np.mean(v)) for m, v in per_fold.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combos", default="Tissue_Type_Clustering/Conch1_5,Averaging/Virchow2")
    ap.add_argument("--classifiers", default="lin,knn,proto,rf,ann")
    args = ap.parse_args()
    classifiers = args.classifiers.split(",")

    print("=" * 86)
    print("TASK 2.1 - SurGen folds: slide-level (OLD, leaky) vs case-level (NEW)")
    print("=" * 86)

    grand = {c: {m: [] for m in METRICS} for c in classifiers}

    for combo in args.combos.split(","):
        method, model = combo.split("/")
        coh = dl.load_cohort("surgen", method, model, verbose=False)

        old_folds = build_slide_level_folds_legacy(restrict_to=coh.ids)
        new_folds = build_case_level_folds(restrict_to=coh.ids, verbose=False)

        lk_old = quantify_leakage(old_folds)
        lk_new = quantify_leakage(new_folds)

        print(f"\n{'-' * 86}")
        print(f"SurGen / {method} / {model}   N={coh.n}  D={coh.dim}  "
              f"classes={coh.class_counts()}")
        print(f"  OLD folds: {lk_old['n_split_across_folds']}/{lk_old['n_multi_slide_cases']} "
              f"multi-slide cases split across folds ({lk_old['pct_split']:.1f}%), "
              f"{lk_old['leaked_slides']} slides implicated")
        print(f"  NEW folds: {lk_new['n_split_across_folds']}/{lk_new['n_multi_slide_cases']} "
              f"split ({lk_new['pct_split']:.1f}%)")
        print(f"{'-' * 86}")

        t0 = time.perf_counter()
        old = run_scheme(coh, old_folds, classifiers, check_leakage=False)
        new = run_scheme(coh, new_folds, classifiers, check_leakage=True)
        print(f"  (both arms ran in {time.perf_counter() - t0:.0f}s)\n")

        hdr = f"  {'clf':<7}"
        for m in METRICS:
            hdr += f"{m.upper() + ' old':>13}{m.upper() + ' new':>13}{'delta':>9}"
        print(hdr)
        for clf in classifiers:
            line = f"  {clf:<7}"
            for m in METRICS:
                o, n = old[clf][m], new[clf][m]
                line += f"{o:>13.4f}{n:>13.4f}{n - o:>+9.4f}"
                grand[clf][m].append(n - o)
            print(line)

    print(f"\n{'=' * 86}")
    print("MEAN DELTA (new - old) ACROSS ALL COMBINATIONS - negative = inflation removed")
    print("=" * 86)
    print(f"  {'clf':<8}" + "".join(f"{m.upper():>12}" for m in METRICS))
    all_d = {m: [] for m in METRICS}
    for clf in classifiers:
        line = f"  {clf:<8}"
        for m in METRICS:
            d = float(np.mean(grand[clf][m]))
            all_d[m].append(d)
            line += f"{d:>+12.4f}"
        print(line)
    print(f"  {'ALL':<8}" + "".join(f"{np.mean(all_d[m]):>+12.4f}" for m in METRICS))
    print("\nA drop is the correction working: sibling slides from one case were "
          "previously\nsplit across train and test, and two sections of one tumour are far "
          "more alike\nthan two different tumours.")


if __name__ == "__main__":
    main()
