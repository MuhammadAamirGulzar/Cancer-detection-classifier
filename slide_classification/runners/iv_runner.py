"""PAIP-IV: internal validation on the provider's own train/test split (Task 3.1).

**IV (internal validation)** - a single train/test split defined by the dataset
provider; no folds, no averaging. This is the right protocol for PAIP because
PAIP *ships* a split; K-fold is what you use when a cohort does not (TCGA,
SurGen). PAIP-CV was retired in Task 2.2 for exactly this reason - it trained on
less data (~36) and tested on smaller sets (18, ~4 positives) while answering the
same question.

The split
---------
    train  paip_47slides.csv         47 ids, of which **42 have features** (10 MSI-H)
    test   paip_31slides_labels.csv  31 ids, **all 31 have features** (7 MSI-H)

The 42/31 counts are not 47/31 and the report must say so. Five training slides
have no features: training_data_19/30/41/42/46 (two of them MSI-H).

A stratified validation subset is carved from the **42 training slides** for ANN
early stopping. The 31 test slides are never touched by any tuning decision.

Threshold
---------
PAIP-IV is an *internal* experiment: train and test distributions are matched, so
it reports at the default 0.5 (work order section 1b rule 2). tau_TCGA applies to
external experiments only.

Uncertainty
-----------
At 31 test slides with 7 positives, bootstrap confidence intervals are the correct
tool - 1000 resamples with replacement, 95% CI on BalAcc and AUROC (Task 2.2).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data_layer as dl
from config import paths as P
from runners import runlog
from runners.classifiers import MODEL_TYPES, train_and_evaluate
from runners.results_io import archive_experiment_tree

N_BOOTSTRAP = 1000
CI_LEVEL = 95
ANN_VAL_FRAC = 0.20   # of the 42 training slides, for ANN early stopping only


def provider_split_ids() -> Tuple[List[str], List[str]]:
    """``(train_ids, test_ids)`` exactly as the provider defines them."""
    train = pd.read_csv(P.paip_split_path("train"))["Fold1"].astype(str).tolist()
    test = pd.read_csv(P.paip_split_path("test"))["WSI_Id"].astype(str).tolist()
    return train, test


def bootstrap_ci(targets: np.ndarray, probs_pos: np.ndarray, preds: np.ndarray,
                 n: int = N_BOOTSTRAP, level: int = CI_LEVEL,
                 seed: int = 42) -> Dict[str, object]:
    """Percentile bootstrap CIs over the test slides.

    Resamples with replacement. Draws that end up single-class are skipped for
    AUROC (undefined) and counted, rather than silently dropped.
    """
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    rng = np.random.default_rng(seed)
    m = len(targets)
    baccs, aurocs, degenerate = [], [], 0
    for _ in range(n):
        idx = rng.integers(0, m, m)
        t = targets[idx]
        if len(np.unique(t)) < 2:
            degenerate += 1
            continue
        baccs.append(balanced_accuracy_score(t, preds[idx]))
        aurocs.append(roc_auc_score(t, probs_pos[idx]))

    lo, hi = (100 - level) / 2, 100 - (100 - level) / 2
    def ci(v):
        return [float(np.percentile(v, lo)), float(np.percentile(v, hi))] if v else [None, None]

    return {
        "n_resamples": n,
        "n_usable": len(baccs),
        "n_degenerate_single_class": degenerate,
        "level": level,
        "bacc_ci": ci(baccs),
        "auroc_ci": ci(aurocs),
        "bacc_boot_mean": float(np.mean(baccs)) if baccs else None,
        "auroc_boot_mean": float(np.mean(aurocs)) if aurocs else None,
    }


def run_paip_iv(method: str, model: str, task: str = "MSIH",
                classifiers: Optional[List[str]] = None, seed: int = 42) -> dict:
    """One (method, model) through the provider's 42/31 split."""
    log = runlog.get_logger()
    classifiers = list(classifiers or MODEL_TYPES)

    coh = dl.load_cohort("paip", method, model, task=task, verbose=True)
    train_ids, test_ids = provider_split_ids()

    tr_all = coh.rows_for(train_ids)     # ids present in BOTH the split and features
    te_rows = coh.rows_for(test_ids)

    n_train_declared, n_test_declared = len(train_ids), len(test_ids)
    log.info(f"PAIP-IV | {method}/{model} | provider split {n_train_declared}/{n_test_declared} "
             f"-> with features {len(tr_all)}/{len(te_rows)}")
    if len(te_rows) != 31:
        raise ValueError(f"PAIP-IV test set must be 31 slides, got {len(te_rows)}")

    dl.assert_no_leakage([coh.ids[i] for i in tr_all], [coh.ids[i] for i in te_rows],
                         label=f"PAIP-IV {method}/{model}")

    # ANN early-stopping subset, carved from the TRAINING slides only.
    sub_tr, sub_val = dl.stratified_holdout(coh.labels[tr_all], ANN_VAL_FRAC, seed=seed)
    tr_rows, va_rows = tr_all[sub_tr], tr_all[sub_val]

    out_root = P.results_root("PAIP-IV", method, model, task)
    out_dir, models_dir = out_root / "Output", out_root / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    coh.write_missing_report(out_dir)

    targets = coh.labels[te_rows].numpy()
    # lin/knn/proto/rf run with combine_trainval=True, so they train on all 42;
    # only the ANN holds the carve-out back, training on 34 and early-stopping on 8.
    log.info(f"  train: {len(tr_all)} effective for lin/knn/proto/rf, "
             f"{len(tr_rows)}+{len(va_rows)} early-stop for ann | test={len(te_rows)} "
             f"| train MSI-H={int(coh.labels[tr_all].sum())} test MSI-H={int(targets.sum())}")

    rows, per_clf = [], {}
    for kind in classifiers:
        with runlog.timed(f"PAIP-IV {method}/{model} {kind}", log):
            metrics, dump, selection = train_and_evaluate(
                fold=0,
                train=coh.subset(tr_rows)[:2], valid=coh.subset(va_rows)[:2],
                test=coh.subset(te_rows)[:2],
                model_type=kind, input_dim=coh.dim,
                model_save_path=str(models_dir), seed=seed, verbose=False,
            )
        probs_pos = np.asarray(dump["probs_all"])[:, 1]
        preds = np.asarray(dump["preds_all"])
        ci = bootstrap_ci(targets, probs_pos, preds, seed=seed)

        entry = {
            "classifier": kind,
            "bacc": float(metrics[f"{kind}_bacc"]),
            "auroc": float(metrics[f"{kind}_auroc"]),
            "acc": float(metrics[f"{kind}_acc"]),
            "macro_f1": float(metrics[f"{kind}_macro_f1"]),
            "weighted_f1": float(metrics[f"{kind}_weighted_f1"]),
            "conf_matrix": metrics[f"{kind}_conf_matrix"].tolist(),
            "threshold": 0.5,
            "n_train": int(len(tr_all)), "n_ann_early_stop": int(len(va_rows)),
            "n_test": int(len(te_rows)),
            "bootstrap": ci,
            "selection": selection,
        }
        per_clf[kind] = entry
        log.info(f"    bacc={entry['bacc']:.4f} "
                 f"[{ci['bacc_ci'][0]:.4f}, {ci['bacc_ci'][1]:.4f}]  "
                 f"auroc={entry['auroc']:.4f} "
                 f"[{ci['auroc_ci'][0]:.4f}, {ci['auroc_ci'][1]:.4f}]  "
                 f"cm={entry['conf_matrix']}")

        rows.append({
            "Experiment": "PAIP-IV", "Method": method, "Model": model,
            "Classifier": kind, "N_train": len(tr_all), "N_test": len(te_rows),
            "BalAcc": entry["bacc"],
            "BalAcc_CI_low": ci["bacc_ci"][0], "BalAcc_CI_high": ci["bacc_ci"][1],
            "AUROC": entry["auroc"],
            "AUROC_CI_low": ci["auroc_ci"][0], "AUROC_CI_high": ci["auroc_ci"][1],
            "Acc": entry["acc"], "MacroF1": entry["macro_f1"],
            "Threshold": 0.5, "ConfMatrix": str(entry["conf_matrix"]),
        })

        # Per-slide predictions, for audit.
        pd.DataFrame({
            "WSI_ID": [coh.ids[i] for i in te_rows],
            "target": targets, "prob_pos": probs_pos, "pred": preds,
        }).to_csv(out_dir / f"predictions_{kind}.csv", index=False)

    payload = {
        "stamp": P.run_stamp(
            experiment="PAIP-IV", cohort="paip", method=method, model=model,
            task=task, n_test=len(te_rows), n_train=len(tr_all),
            n_train_declared=n_train_declared, n_test_declared=n_test_declared,
            protocol="single provider-defined split, no averaging",
            threshold_policy="0.5 (internal experiment, section 1b rule 2)",
            bootstrap=f"{N_BOOTSTRAP} resamples, {CI_LEVEL}% CI",
        ),
        "results": per_clf,
    }
    (out_dir / f"result_PAIP-IV_{method}_{model}_default.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_dir / "summary_iv.csv", index=False)
    log.info(f"  wrote -> {out_dir}")
    return payload


def sweep(methods: List[str], models: List[str], task: str = "MSIH",
          classifiers: Optional[List[str]] = None, seed: int = 42,
          force: bool = False, archive: bool = True) -> None:
    log = runlog.get_logger()
    ran, skipped = [], []
    if archive:
        moved = archive_experiment_tree("PAIP-IV")
        if moved:
            log.warning(f"archived previous PAIP-IV results -> {moved.name}")

    for method in methods:
        for model in models:
            if not P.is_combination_valid(method, model):
                continue
            key = runlog.progress_key("PAIP-IV", method, model, seed=seed)
            if runlog.is_done(key, force):
                log.info(f"RESUME skip (already done): {key}")
                ran.append((method, model))
                continue
            if not P.feature_dir("paip", method, model).is_dir():
                runlog.record_skip("PAIP-IV", "paip", method, model,
                                   "features_not_on_this_machine", "")
                skipped.append((method, model))
                continue
            try:
                run_paip_iv(method, model, task, classifiers, seed)
            except FileNotFoundError as exc:
                runlog.record_skip("PAIP-IV", "paip", method, model, "missing_input", str(exc))
                runlog.mark(key, "skipped", reason=str(exc)[:200])
                skipped.append((method, model))
                continue
            except (AssertionError, ValueError, KeyError, TypeError):
                raise
            except Exception as exc:
                runlog.record_skip("PAIP-IV", "paip", method, model,
                                   "runtime_failure", f"{type(exc).__name__}: {exc}")
                runlog.mark(key, "failed", reason=f"{type(exc).__name__}: {str(exc)[:200]}")
                skipped.append((method, model))
                continue
            runlog.mark(key, "done")
            ran.append((method, model))

    runlog.coverage_report("PAIP-IV", ran, skipped, log)


def main():
    ap = argparse.ArgumentParser(description="PAIP-IV runner (Task 3.1)")
    ap.add_argument("--methods", default=",".join(P.AGGREGATION_METHODS))
    ap.add_argument("--models", default=",".join(P.CANONICAL_MODELS))
    ap.add_argument("--classifiers", default=",".join(MODEL_TYPES))
    ap.add_argument("--task", default="MSIH")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    sweep(args.methods.split(","), args.models.split(","), args.task,
          args.classifiers.split(","), args.seed, args.force)


if __name__ == "__main__":
    main()
