"""K-fold cross-validation runner - the single replacement for the CV loops that
were duplicated across ``Slide_Classification.ipynb`` (cells 6 and 8),
``windows_slide_classification_surgen.py``, ``linux_slide_classification_surgen.py``
and ``slide_classification_surgen.py``.

Serves TCGA-CV and SurGen-CV. PAIP-CV is retired (work order Task 2.2); PAIP uses
the provider's own split via the PAIP-IV runner.

Protocol note (work order "Known issues carried forward" #1 and #2), which the
Methods section must state explicitly:

  * The rotation is test = fold i, val = fold i+1, train = the remaining two, so
    **training uses 50% of the cohort**, not the 75% a reader would assume from
    "4-fold CV". Confirmed deliberate by the owner.
  * ``combine_trainval`` is True for lin/knn/proto/rf, which merge the validation
    fold back into training; only the ANN uses it as a true early-stopping set.
    "Validation fold" therefore means two different things depending on the
    classifier.
  * Folds are 1-4 in the CSVs and reported as Fold1..Fold4, but checkpoints are
    saved fold0..fold3. Consistent, but easy to misread.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

if __package__ in (None, ""):  # allow `python runners/cv_runner.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data_layer as dl
from config import paths as P
from runners import runlog
from runners.classifiers import MODEL_TYPES, train_and_evaluate
from runners.results_io import ResultAccumulator, archive_experiment_tree
from runners.thresholds import thresholds_from_oof, update_threshold_store


def run_cross_validation(
    experiment: str,
    cohort: str,
    method: str,
    model: str,
    task: str = "MSIH",
    model_types: Optional[List[str]] = None,
    fold_map: Optional[Dict[str, int]] = None,
    group_of: Optional[Callable[[str], str]] = None,
    seed: int = 42,
    save_models: bool = True,
    verbose: bool = True,
) -> ResultAccumulator:
    """Run one ``(method, model)`` combination through K-fold CV.

    ``fold_map`` overrides the cohort's own fold assignment - SurGen passes the
    case-level folds built by Task 2.1. ``group_of`` maps a slide id to its
    grouping key so the leakage assertion can check at case level.
    """
    log = runlog.get_logger()
    model_types = list(model_types or MODEL_TYPES)

    coh = dl.load_cohort(cohort, method, model, task=task, verbose=verbose)
    fold_map = fold_map if fold_map is not None else dl.load_fold_map(cohort, task)
    if fold_map is None:
        raise ValueError(
            f"No fold assignment for cohort={cohort}. TCGA ships folds; SurGen "
            f"folds must be built at case level and passed in (Task 2.1)."
        )

    splits = dl.build_splits(coh.ids, fold_map)
    if len(splits) < 3:
        raise ValueError(f"Need at least 3 folds, got {sorted(splits)}")

    out_root = P.results_root(experiment, method, model, task)
    output_dir = out_root / "Output"
    models_dir = out_root / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_models:
        models_dir.mkdir(parents=True, exist_ok=True)

    # Task 1.4: a labelled slide with no features must never vanish silently.
    missing_csv = coh.write_missing_report(output_dir)
    if missing_csv:
        log.warning(f"missing-slide report -> {missing_csv}")

    log.info(f"{experiment} | {method}/{model} | N={coh.n} D={coh.dim} "
             f"classes={coh.class_counts()}")
    for f, rows in splits.items():
        log.info(f"  fold {f}: n={len(rows)} MSI-H={int(coh.labels.numpy()[rows].sum())}")

    acc = ResultAccumulator(experiment, cohort, method, model, task, seed=seed)
    acc.notes = {
        "n_slides": coh.n, "dim": coh.dim,
        "class_counts": coh.class_counts(),
        "missing_ids": coh.missing_ids,
        "unlabelled_ids": coh.unlabelled_ids,
        "fold_sizes": {int(k): int(len(v)) for k, v in splits.items()},
        "protocol": "test=fold i, val=fold i+1, train=remaining (50% train)",
    }

    fold_ids = sorted(splits)
    for clf in model_types:
        for f in fold_ids:
            tr_rows, va_rows, te_rows = dl.cv_rotation(splits, f)

            train_ids = [coh.ids[i] for i in tr_rows]
            val_ids = [coh.ids[i] for i in va_rows]
            test_ids = [coh.ids[i] for i in te_rows]
            # Fail-fast on correctness: leakage must halt the run (1c.3 item 5).
            dl.assert_no_leakage(train_ids + val_ids, test_ids, group_of,
                                 label=f"{experiment} {method}/{model} fold{f}")

            trf, trl, _ = coh.subset(tr_rows)
            vaf, val, _ = coh.subset(va_rows)
            tef, tel, _ = coh.subset(te_rows)

            # Checkpoints are saved fold0..fold3 while folds are reported 1..4.
            ckpt_fold = fold_ids.index(f)
            with runlog.timed(f"{experiment} {method}/{model} {clf} fold{f}", log):
                metrics, dump, selection = train_and_evaluate(
                    fold=ckpt_fold,
                    train=(trf, trl), valid=(vaf, val), test=(tef, tel),
                    model_type=clf, input_dim=coh.dim,
                    model_save_path=str(models_dir) if save_models else None,
                    seed=seed, verbose=False,
                )
            acc.add_fold(clf, f, metrics, dump, test_ids, selection,
                         n_train=len(tr_rows), n_val=len(va_rows))
            log.info(f"    bacc={metrics.get(f'{clf}_bacc', float('nan')):.4f} "
                     f"auroc={metrics.get(f'{clf}_auroc', float('nan')):.4f} "
                     f"cm={metrics.get(f'{clf}_conf_matrix').tolist()}")

    written = acc.write(output_dir)
    log.info(f"wrote: {', '.join(p.name for p in written.values())}")
    return acc


def sweep(
    experiment: str,
    cohort: str,
    methods: List[str],
    models: List[str],
    task: str = "MSIH",
    model_types: Optional[List[str]] = None,
    fold_map: Optional[Dict[str, int]] = None,
    group_of: Optional[Callable[[str], str]] = None,
    seed: int = 42,
    force: bool = False,
    write_thresholds: bool = False,
    archive: bool = True,
) -> None:
    """Sweep ``methods x models``, skipping what is unavailable on this machine.

    Fail-soft on availability, fail-fast on correctness (1c.3 item 5): a missing
    feature directory is logged to ``skipped_combinations.csv`` and the run
    continues; a shape/label/leakage failure propagates and halts.

    ``archive=True`` renames any pre-existing results tree to
    ``<name>_ARCHIVED_<date>`` before the first write, so results produced before
    the Phase 1 corrections are never silently overwritten (rule of engagement #2).
    """
    log = runlog.get_logger()
    ran, skipped = [], []

    if archive:
        moved = archive_experiment_tree(experiment)
        if moved:
            log.warning(f"archived previous {experiment} results -> {moved.name}")

    for method in methods:
        for model in models:
            if not P.is_combination_valid(method, model):
                continue
            key = runlog.progress_key(experiment, method, model, seed=seed)
            if runlog.is_done(key, force):
                log.info(f"RESUME skip (already done): {key}")
                ran.append((method, model))
                continue

            fdir = P.feature_dir(cohort, method, model)
            if not fdir.is_dir():
                runlog.record_skip(experiment, cohort, method, model,
                                   "features_not_on_this_machine", str(fdir))
                runlog.mark(key, "skipped", reason="features_not_on_this_machine")
                skipped.append((method, model))
                continue

            try:
                acc = run_cross_validation(
                    experiment, cohort, method, model, task=task,
                    model_types=model_types, fold_map=fold_map,
                    group_of=group_of, seed=seed,
                )
            except FileNotFoundError as exc:
                runlog.record_skip(experiment, cohort, method, model,
                                   "missing_input", str(exc))
                runlog.mark(key, "skipped", reason=str(exc)[:200])
                skipped.append((method, model))
                continue
            except (AssertionError, ValueError, KeyError, TypeError):
                # Correctness failures - a leaked case, a shape mismatch, an
                # unresolvable label. These must halt the run loudly
                # (work order 1c.3 item 5). Never downgrade one to a skip.
                raise
            except Exception as exc:
                # Infrastructure failure (e.g. a joblib worker dying). One bad
                # combination must not cost a multi-hour unattended sweep; the
                # traceback is already in the log via runlog.timed.
                runlog.record_skip(experiment, cohort, method, model,
                                   "runtime_failure", f"{type(exc).__name__}: {exc}")
                runlog.mark(key, "failed", reason=f"{type(exc).__name__}: {str(exc)[:200]}")
                skipped.append((method, model))
                continue

            if write_thresholds:
                oof = acc.oof_frame()
                if not oof.empty:
                    taus = thresholds_from_oof(oof)
                    path = update_threshold_store(method, model, taus, task)
                    log.info("tau_TCGA (Youden J on pooled out-of-fold): " +
                             ", ".join(f"{c}={v['tau']:.4f}" for c, v in sorted(taus.items())))
                    log.info(f"  -> {path.name}")

            runlog.mark(key, "done",
                        n_slides=acc.notes.get("n_slides"),
                        output=str(P.results_root(experiment, method, model, task) / "Output"))
            ran.append((method, model))

    runlog.coverage_report(experiment, ran, skipped, log)


def main():
    ap = argparse.ArgumentParser(description="K-fold CV runner (TCGA-CV / SurGen-CV)")
    ap.add_argument("--experiment", default="TCGA-CV", choices=["TCGA-CV", "SurGen-CV"])
    ap.add_argument("--methods", default=",".join(P.AGGREGATION_METHODS))
    ap.add_argument("--models", default=",".join(P.CANONICAL_MODELS))
    ap.add_argument("--classifiers", default=",".join(MODEL_TYPES))
    ap.add_argument("--task", default="MSIH")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="re-run completed combinations")
    args = ap.parse_args()

    cohort = "tcga" if args.experiment == "TCGA-CV" else "surgen"
    fold_map, group_of = None, None
    if cohort == "surgen":
        from runners.surgen_folds import build_case_level_folds, case_id_of
        fold_map = build_case_level_folds()
        group_of = case_id_of

    sweep(
        experiment=args.experiment, cohort=cohort,
        methods=args.methods.split(","), models=args.models.split(","),
        task=args.task, model_types=args.classifiers.split(","),
        fold_map=fold_map, group_of=group_of, seed=args.seed, force=args.force,
        write_thresholds=(args.experiment == "TCGA-CV"),
    )


if __name__ == "__main__":
    main()
