"""Train the TCGA-FULL artifact on all 413 TCGA slides (work order Task 3.0a).

Why this artifact exists
------------------------
PAIP-EV currently applies the 4 TCGA fold-models separately and averages their
four metric sets. But each fold-model was trained on only 2 of 4 folds:

    test fold 1 -> train n=213 (32 MSI-H)   test fold 3 -> train n=200 (28 MSI-H)
    test fold 2 -> train n=204 (27 MSI-H)   test fold 4 -> train n=209 (33 MSI-H)

So the existing external validation demonstrates transfer from models trained on
~208 slides (~28 MSI-H) when 413 slides (60 MSI-H) are available. Training data
and positive count both roughly double under TCGA-FULL.

The owner's decision is that TCGA-FULL is the **primary** EV artifact, with the
4-fold probability ensemble as a robustness check and the legacy 4-fold-average
convention retained as a third reported variant. None is discarded - the
comparison "half the training data vs all of it" is itself a result.

What this module does
---------------------
1. Hyperparameters are **fixed first, from TCGA-CV only** (``runners.hparams``).
   No external data influences them, and nothing is tuned here.
2. One final model per (method, model, classifier) trained on all 413 slides.
   The ANN is the exception: it needs a held-out set for early stopping, so a
   seed-fixed **stratified 15%** is carved out - ~351 train / ~62 early-stop.
   That asymmetry is documented rather than hidden.
3. Repeated over 5 seeds (42-46) for a variance estimate. ``lin`` and ``knn`` are
   effectively deterministic here, so their SD will be ~0 - expected, not a bug.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data_layer as dl
from config import paths as P
from runners import runlog
from runners.classifiers import MODEL_TYPES
from runners.hparams import load_hparams
from runners.model_io import fit_full, save_model

SEEDS = (42, 43, 44, 45, 46)
ANN_HOLDOUT_FRAC = 0.15


def artifact_dir(method: str, model: str, seed: int, task: str = "MSIH") -> Path:
    return P.results_root("TCGA-FULL", method, model, task) / f"seed{seed}"


def train_combination(
    method: str,
    model: str,
    task: str = "MSIH",
    classifiers: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    verbose: bool = True,
) -> dict:
    """Train and persist TCGA-FULL models for one (method, model), all seeds."""
    log = runlog.get_logger()
    classifiers = list(classifiers or MODEL_TYPES)
    seeds = list(seeds or SEEDS)

    coh = dl.load_cohort("tcga", method, model, task=task, verbose=False)
    log.info(f"TCGA-FULL | {method}/{model} | N={coh.n} D={coh.dim} "
             f"classes={coh.class_counts()}")
    if coh.n != 413:
        log.warning(f"  expected 413 TCGA slides with features, got {coh.n}")

    summary = {"method": method, "model": model, "task": task,
               "n_train_total": coh.n, "dim": coh.dim, "seeds": {}}

    for seed in seeds:
        out_dir = artifact_dir(method, model, seed, task)
        out_dir.mkdir(parents=True, exist_ok=True)
        per_seed = {}

        # The ANN's early-stopping carve-out. Stratified and seed-fixed, so it is
        # reproducible; the other classifiers train on every row.
        tr_rows, ho_rows = dl.stratified_holdout(coh.labels, ANN_HOLDOUT_FRAC, seed=seed)
        ann_train = coh.subset(tr_rows)[:2]
        ann_val = coh.subset(ho_rows)[:2]

        for kind in classifiers:
            hp = load_hparams(method, model, kind, task)
            if kind == "ann":
                n_train, n_val = len(tr_rows), len(ho_rows)
                with runlog.timed(f"TCGA-FULL {method}/{model} {kind} seed{seed} "
                                  f"(train {n_train} + early-stop {n_val})", log):
                    clf = fit_full(kind, ann_train[0], ann_train[1], hp,
                                   seed=seed, val=ann_val)
            else:
                n_train, n_val = coh.n, 0
                with runlog.timed(f"TCGA-FULL {method}/{model} {kind} seed{seed} "
                                  f"(train {n_train})", log):
                    clf = fit_full(kind, coh.feats, coh.labels, hp, seed=seed)

            save_model(kind, clf, out_dir, fold=None, extra={
                "classifier": kind, "hparams": hp, "seed": seed,
                "n_train": n_train, "n_early_stop": n_val,
                "cohort": "TCGA", "n_cohort": coh.n,
                "trained_on": "all 413 TCGA slides" if kind != "ann"
                              else f"stratified {1 - ANN_HOLDOUT_FRAC:.0%} of 413",
                **P.run_stamp(),
            })
            per_seed[kind] = {"n_train": n_train, "n_early_stop": n_val, "hparams": hp}

        summary["seeds"][str(seed)] = per_seed
        if verbose:
            log.info(f"  seed {seed}: saved {len(classifiers)} model(s) -> {out_dir}")

    root = P.results_root("TCGA-FULL", method, model, task)
    (root / "training_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def sweep(
    methods: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    task: str = "MSIH",
    classifiers: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    force: bool = False,
) -> None:
    log = runlog.get_logger()
    methods = list(methods or P.AGGREGATION_METHODS)
    models = list(models or P.CANONICAL_MODELS)
    ran, skipped = [], []

    for method in methods:
        for model in models:
            if not P.is_combination_valid(method, model):
                continue
            key = runlog.progress_key("TCGA-FULL", method, model, variant="artifact")
            if runlog.is_done(key, force):
                log.info(f"RESUME skip (already done): {key}")
                ran.append((method, model))
                continue
            if not P.feature_dir("tcga", method, model).is_dir():
                runlog.record_skip("TCGA-FULL", "tcga", method, model,
                                   "features_not_on_this_machine", "")
                skipped.append((method, model))
                continue
            try:
                train_combination(method, model, task, classifiers, seeds)
            except FileNotFoundError as exc:
                runlog.record_skip("TCGA-FULL", "tcga", method, model,
                                   "missing_input", str(exc))
                runlog.mark(key, "skipped", reason=str(exc)[:200])
                skipped.append((method, model))
                continue
            except (AssertionError, ValueError, KeyError, TypeError):
                raise  # correctness failures halt the run
            except Exception as exc:
                runlog.record_skip("TCGA-FULL", "tcga", method, model,
                                   "runtime_failure", f"{type(exc).__name__}: {exc}")
                runlog.mark(key, "failed", reason=f"{type(exc).__name__}: {str(exc)[:200]}")
                skipped.append((method, model))
                continue
            runlog.mark(key, "done")
            ran.append((method, model))

    runlog.coverage_report("TCGA-FULL", ran, skipped, log)


def main():
    ap = argparse.ArgumentParser(description="Train the TCGA-FULL artifact (Task 3.0a)")
    ap.add_argument("--methods", default=",".join(P.AGGREGATION_METHODS))
    ap.add_argument("--models", default=",".join(P.CANONICAL_MODELS))
    ap.add_argument("--classifiers", default=",".join(MODEL_TYPES))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--task", default="MSIH")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    sweep(methods=args.methods.split(","), models=args.models.split(","),
          task=args.task, classifiers=args.classifiers.split(","),
          seeds=[int(s) for s in args.seeds.split(",")], force=args.force)


if __name__ == "__main__":
    main()
