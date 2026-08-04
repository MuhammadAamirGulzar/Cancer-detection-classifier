"""External validation: TCGA-trained models applied to PAIP or SurGen.

Serves **PAIP-EV** and **SurGen-EV** (work order Tasks 3.0, 3.2). Zero
target-cohort data enters training, and no threshold is ever tuned on the target
cohort - doing so would silently convert external validation into internal
validation.

Three variants are reported side by side for every combination, so they can be
compared directly. None is discarded (owner decision):

  | variant            | train n                        | role               |
  |--------------------|--------------------------------|--------------------|
  | ``tcga_full``      | 413 (ANN 351 + 62 early-stop)  | **primary**        |
  | ``fold_ensemble``  | 4 x ~208, pooled probabilities | robustness check   |
  | ``fold_average``   | 4 x ~208, averaged metrics     | legacy / continuity|

``fold_ensemble`` averages the four fold-models' **predicted probabilities** into
one prediction per slide and then applies tau once - one prediction per slide,
not four metric sets averaged. ``fold_average`` is the pre-existing convention:
apply each fold-model separately to the whole external cohort and average the
four metric sets.

Thresholds (section 1b)
-----------------------
Operating-point metrics are reported at **tau_TCGA**, fitted by Youden's J on
pooled out-of-fold TCGA-CV probabilities and applied unchanged. AUROC is the
primary metric everywhere and is threshold-free, so it is unaffected.

The raw 0.5-threshold numbers, and ``RF @ 0.3`` (the pre-existing hardcoded
random-forest threshold), are kept in a per-run audit CSV so the old behaviour
stays reproducible. They are not headline figures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data_layer as dl
from config import paths as P
from eval_patch_features.metrics import get_eval_metrics
from runners import runlog
from runners.classifiers import MODEL_TYPES
from runners.full_trainer import SEEDS, artifact_dir
from runners.model_io import load_model, predict_proba
from runners.results_io import archive_experiment_tree
from runners.thresholds import load_tau

VARIANTS = ("tcga_full", "fold_ensemble", "fold_average")
RF_LEGACY_THRESHOLD = 0.3
N_FOLDS = 4


def _metrics_at(targets: np.ndarray, probs_pos: np.ndarray, tau: float) -> Dict:
    """Metrics for a fixed operating point. AUROC is threshold-free."""
    preds = (np.asarray(probs_pos) >= tau).astype(int)
    two_col = np.column_stack([1.0 - probs_pos, probs_pos])
    m = get_eval_metrics(targets, preds, two_col, prefix="")
    return {
        "acc": float(m["acc"]), "bacc": float(m["bacc"]),
        "macro_f1": float(m["macro_f1"]), "weighted_f1": float(m["weighted_f1"]),
        "auroc": float(m["auroc"]), "conf_matrix": m["conf_matrix"].tolist(),
        "threshold": float(tau),
    }


def _fold_models_dir(method: str, model: str, task: str) -> Path:
    return P.results_root("TCGA-CV", method, model, task) / "models"


def run_external_validation(
    experiment: str,
    target_cohort: str,
    method: str,
    model: str,
    task: str = "MSIH",
    classifiers: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    variants: Optional[List[str]] = None,
) -> dict:
    """Apply TCGA-trained models to an external cohort, all three variants."""
    log = runlog.get_logger()
    classifiers = list(classifiers or MODEL_TYPES)
    seeds = list(seeds or SEEDS)
    variants = list(variants or VARIANTS)

    coh = dl.load_cohort(target_cohort, method, model, task=task, verbose=True)
    out_root = P.results_root(experiment, method, model, task)
    out_dir = out_root / "Output"
    out_dir.mkdir(parents=True, exist_ok=True)
    coh.write_missing_report(out_dir)

    targets = coh.labels.numpy()
    log.info(f"{experiment} | {method}/{model} | test N={coh.n} "
             f"classes={coh.class_counts()} D={coh.dim}")

    # Leakage is structural here: nothing from the target cohort was ever
    # trained on. Assert the cohorts are disjoint anyway - cheap, and it would
    # catch a mis-wired path immediately.
    tcga_ids = set(dl.load_label_map("tcga", task))
    overlap = tcga_ids & set(coh.ids)
    if overlap:
        raise AssertionError(
            f"[LEAKAGE] {len(overlap)} slide(s) appear in both TCGA and "
            f"{target_cohort}: {sorted(overlap)[:5]}")

    results: Dict[str, Dict] = {}
    audit_rows: List[Dict] = []

    for kind in classifiers:
        tau = load_tau(method, model, kind, task)
        per_variant: Dict[str, Dict] = {}

        # ---------------- variant 1: TCGA-FULL, 5 seeds ---------------------
        if "tcga_full" in variants:
            seed_metrics, seed_probs = [], []
            for seed in seeds:
                d = artifact_dir(method, model, seed, task)
                try:
                    clf = load_model(kind, d, fold=None, input_dim=coh.dim)
                except FileNotFoundError as exc:
                    log.warning(f"  {kind} seed{seed}: {exc}")
                    continue
                probs = predict_proba(kind, clf, coh.feats)[:, 1]
                seed_probs.append(probs)
                seed_metrics.append(_metrics_at(targets, probs, tau))
                audit_rows += _audit(experiment, method, model, kind, "tcga_full",
                                     targets, probs, tau, seed=seed)
            if seed_metrics:
                per_variant["tcga_full"] = _aggregate_seeds(seed_metrics, seeds[:len(seed_metrics)])
                per_variant["tcga_full"]["mean_probs"] = np.mean(seed_probs, axis=0).tolist()

        # ------- variants 2 & 3: the 4 TCGA-CV fold models -------------------
        if "fold_ensemble" in variants or "fold_average" in variants:
            mdir = _fold_models_dir(method, model, task)
            fold_probs, fold_metrics = [], []
            for f in range(N_FOLDS):
                try:
                    clf = load_model(kind, mdir, fold=f, input_dim=coh.dim)
                except FileNotFoundError as exc:
                    log.warning(f"  {kind} fold{f}: {exc}")
                    continue
                probs = predict_proba(kind, clf, coh.feats)[:, 1]
                fold_probs.append(probs)
                fold_metrics.append(_metrics_at(targets, probs, tau))

            if fold_probs and "fold_ensemble" in variants:
                # One prediction per slide: average probabilities, THEN threshold.
                pooled = np.mean(fold_probs, axis=0)
                per_variant["fold_ensemble"] = _metrics_at(targets, pooled, tau)
                per_variant["fold_ensemble"]["n_folds_used"] = len(fold_probs)
                audit_rows += _audit(experiment, method, model, kind,
                                     "fold_ensemble", targets, pooled, tau)

            if fold_metrics and "fold_average" in variants:
                # Legacy convention: average the four metric sets.
                per_variant["fold_average"] = _average_metric_sets(fold_metrics)
                per_variant["fold_average"]["n_folds_used"] = len(fold_metrics)
                for f, (probs, _m) in enumerate(zip(fold_probs, fold_metrics)):
                    audit_rows += _audit(experiment, method, model, kind,
                                         "fold_average", targets, probs, tau, fold=f)

        if per_variant:
            results[kind] = per_variant
            _log_variants(log, kind, tau, per_variant)

    payload = {
        "stamp": P.run_stamp(experiment=experiment, cohort=target_cohort,
                             method=method, model=model, task=task,
                             n_test=coh.n, class_counts=coh.class_counts(),
                             variants=variants, seeds=seeds,
                             threshold_policy="tau_TCGA, Youden J on pooled "
                                              "out-of-fold TCGA-CV probabilities"),
        "results": results,
    }
    (out_dir / f"result_{experiment}_{method}_{model}_ev.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    if audit_rows:
        pd.DataFrame(audit_rows).to_csv(out_dir / "threshold_audit.csv", index=False)

    _write_summary(out_dir, experiment, method, model, coh.n, results)
    log.info(f"  wrote -> {out_dir}")
    return payload


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _audit(experiment, method, model, kind, variant, targets, probs, tau,
           seed=None, fold=None) -> List[Dict]:
    """Rows for threshold_audit.csv: tau_TCGA, 0.5, and RF's legacy 0.3."""
    rows = []
    points = [("tau_TCGA", tau), ("0.5", 0.5)]
    if kind == "rf":
        # Owner ruling: tau_TCGA replaces the hardcoded 0.3 for external
        # experiments, but 0.3 stays in the audit so the old behaviour is
        # reproducible.
        points.append(("rf_legacy_0.3", RF_LEGACY_THRESHOLD))
    for label, t in points:
        m = _metrics_at(targets, probs, t)
        rows.append({
            "experiment": experiment, "method": method, "model": model,
            "classifier": kind, "variant": variant, "seed": seed, "fold": fold,
            "threshold_label": label, "threshold": t,
            "bacc": m["bacc"], "auroc": m["auroc"], "acc": m["acc"],
            "macro_f1": m["macro_f1"], "conf_matrix": str(m["conf_matrix"]),
        })
    return rows


def _aggregate_seeds(metric_sets: List[Dict], seeds: List[int]) -> Dict:
    """mean +/- SD across seeds. lin/knn are deterministic here, so SD ~0."""
    out: Dict = {"n_seeds": len(metric_sets), "seeds": list(seeds)}
    for k in ("acc", "bacc", "macro_f1", "weighted_f1", "auroc"):
        vals = [m[k] for m in metric_sets]
        out[k] = float(np.mean(vals))
        out[f"{k}_sd"] = float(np.std(vals, ddof=0))
    cms = np.array([m["conf_matrix"] for m in metric_sets], dtype=float)
    out["conf_matrix"] = np.mean(cms, axis=0).round().astype(int).tolist()
    out["threshold"] = metric_sets[0]["threshold"]
    return out


def _average_metric_sets(metric_sets: List[Dict]) -> Dict:
    out: Dict = {}
    for k in ("acc", "bacc", "macro_f1", "weighted_f1", "auroc"):
        vals = [m[k] for m in metric_sets]
        out[k] = float(np.mean(vals))
        out[f"{k}_sd"] = float(np.std(vals, ddof=0))
    cms = np.array([m["conf_matrix"] for m in metric_sets], dtype=float)
    out["conf_matrix"] = np.mean(cms, axis=0).round().astype(int).tolist()
    out["threshold"] = metric_sets[0]["threshold"]
    return out


def _log_variants(log, kind, tau, per_variant):
    log.info(f"  [{kind}] tau_TCGA={tau:.4f}")
    for v in VARIANTS:
        m = per_variant.get(v)
        if not m:
            continue
        sd = f" +/-{m.get('bacc_sd', 0):.4f}" if "bacc_sd" in m else ""
        log.info(f"      {v:<14} bacc={m['bacc']:.4f}{sd}  auroc={m['auroc']:.4f}  "
                 f"cm={m['conf_matrix']}")


def _write_summary(out_dir: Path, experiment, method, model, n_test, results):
    """Flat summary CSV, one row per (classifier, variant), plus delta columns."""
    rows = []
    for kind, per_variant in results.items():
        base = per_variant.get("fold_average")
        for v in VARIANTS:
            m = per_variant.get(v)
            if not m:
                continue
            row = {
                "Experiment": experiment, "Method": method, "Model": model,
                "Classifier": kind, "Variant": v, "N_test": n_test,
                "BalAcc": m["bacc"], "AUROC": m["auroc"], "Acc": m["acc"],
                "MacroF1": m["macro_f1"], "Threshold": m["threshold"],
                "BalAcc_sd": m.get("bacc_sd"), "AUROC_sd": m.get("auroc_sd"),
                "ConfMatrix": str(m["conf_matrix"]),
            }
            # "How much did doubling the training data buy?" - the work order
            # asks for this delta explicitly.
            if base and v == "tcga_full":
                row["d_BalAcc_vs_legacy"] = m["bacc"] - base["bacc"]
                row["d_AUROC_vs_legacy"] = m["auroc"] - base["auroc"]
            rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "summary_ev.csv", index=False)


# --------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------


def sweep(experiment: str, target_cohort: str, methods: List[str], models: List[str],
          task: str = "MSIH", classifiers: Optional[List[str]] = None,
          seeds: Optional[List[int]] = None, variants: Optional[List[str]] = None,
          force: bool = False, archive: bool = True) -> None:
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
            key = runlog.progress_key(experiment, method, model, variant="ev")
            if runlog.is_done(key, force):
                log.info(f"RESUME skip (already done): {key}")
                ran.append((method, model))
                continue
            if not P.feature_dir(target_cohort, method, model).is_dir():
                runlog.record_skip(experiment, target_cohort, method, model,
                                   "features_not_on_this_machine",
                                   str(P.feature_dir(target_cohort, method, model)))
                skipped.append((method, model))
                continue
            try:
                run_external_validation(experiment, target_cohort, method, model,
                                        task, classifiers, seeds, variants)
            except FileNotFoundError as exc:
                runlog.record_skip(experiment, target_cohort, method, model,
                                   "missing_input", str(exc))
                runlog.mark(key, "skipped", reason=str(exc)[:200])
                skipped.append((method, model))
                continue
            except (AssertionError, ValueError, TypeError):
                raise
            except Exception as exc:
                runlog.record_skip(experiment, target_cohort, method, model,
                                   "runtime_failure", f"{type(exc).__name__}: {exc}")
                runlog.mark(key, "failed", reason=f"{type(exc).__name__}: {str(exc)[:200]}")
                skipped.append((method, model))
                continue
            runlog.mark(key, "done")
            ran.append((method, model))

    runlog.coverage_report(experiment, ran, skipped, log)


def main():
    ap = argparse.ArgumentParser(description="External validation (PAIP-EV / SurGen-EV)")
    ap.add_argument("--experiment", default="PAIP-EV", choices=["PAIP-EV", "SurGen-EV"])
    ap.add_argument("--methods", default=",".join(P.AGGREGATION_METHODS))
    ap.add_argument("--models", default=",".join(P.CANONICAL_MODELS))
    ap.add_argument("--classifiers", default=",".join(MODEL_TYPES))
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--task", default="MSIH")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cohort = "paip" if args.experiment == "PAIP-EV" else "surgen"
    sweep(experiment=args.experiment, target_cohort=cohort,
          methods=args.methods.split(","), models=args.models.split(","),
          task=args.task, classifiers=args.classifiers.split(","),
          seeds=[int(s) for s in args.seeds.split(",")],
          variants=args.variants.split(","), force=args.force)


if __name__ == "__main__":
    main()
