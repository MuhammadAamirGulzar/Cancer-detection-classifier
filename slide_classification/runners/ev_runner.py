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

``--threshold-mode=promoted`` (PAIP-EV, 2026-09-02)
---------------------------------------------------
Owner decision: for **PAIP-EV only**, KNN and RF report under the corrected
scheme, and those become the published numbers.

  * **KNN** - scored at k=35 with ``weights='uniform'``, thresholded at a tau
    refitted on TCGA out-of-fold probabilities *at that k*. The taus are read
    from ``knn_k35_thresholds.json``, not recomputed. k changes the scores, so
    KNN's AUROC moves too; it is the only head whose AUROC moves.
  * **RF** - rate-matched (quantile) threshold. Ranking is untouched, so AUROC is
    unchanged and only the operating point moves.
  * **LR, ANN, ProtoNet** - frozen tau_TCGA, exactly as before, and *not*
    evaluated under the corrected scheme at all. Their numbers are unchanged.

So one PAIP-EV table now carries two threshold schemes. Every row therefore
states its own ``Threshold_scheme``, and the stamp's ``threshold_policy`` names
both. SurGen-EV is untouched and stays entirely frozen.
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
from runners.model_io import load_model, predict_proba, _prep
from runners import threshold_modes as tm
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


def _add_corrected(entry: Dict, kind: str, method: str, model: str, task: str,
                   coh, targets: np.ndarray, tau_frozen: float,
                   knn_k: int, log) -> None:
    """Attach the corrected operating point *alongside* the frozen one.

    Adds ``bacc_corrected`` / ``threshold_corrected`` / ``status_*`` and, for
    KNN, ``auroc_corrected`` (its scores change because k changes). The frozen
    keys are left exactly as they were, so both numbers travel together and the
    strict zero-shot result is never overwritten.

    All logic lives in runners.threshold_modes - the same module the validated
    experiment imports, so the two paths cannot drift.
    """
    probs = np.asarray(entry["mean_probs"], dtype=float)
    n = len(targets)
    entry["status_frozen"] = tm.health(
        int((probs >= tau_frozen).sum()), n)

    oof = tm.load_tcga_oof(method, model, task)
    g = oof[oof.classifier == kind] if not oof.empty else oof

    if kind == "knn":
        # Re-score at the validated k using the saved artifact's own reference
        # set and metric, then threshold at a tau refitted on TCGA OOF at k.
        try:
            art = load_model("knn", artifact_dir(method, model, SEEDS[0], task),
                             fold=None, input_dim=coh.dim)
            tc = dl.load_cohort("tcga", method, model, task=task, verbose=False)
            fmap = dl.load_fold_map("tcga", task) or {}
            folds = np.array([fmap.get(i, -1) for i in tc.ids])
            X_tcga = _prep("knn", tc.feats).numpy()
            X_tgt = _prep("knn", coh.feats).numpy()
            # Reuse the stored, already-validated tau - but ONLY when knn_k is
            # the k that store was fitted at. knn_k35_thresholds.json holds
            # taus fitted at k=35; applying that tau to scores computed at a
            # different k pairs a threshold with a score scale it was never
            # fitted to - silently, since knn_tau_from_store has no way to know
            # what k the caller actually wants. [FIX 2026-09-03] --knn-k on the
            # CLI was accepted but not honoured for exactly this reason: any
            # k != 35 still read the k=35 store here. Gate the lookup on k
            # matching, so every other k always refits.
            if knn_k == tm.KNN_K_VALIDATED:
                tau_c, tau_src = tm.knn_tau_from_store(method, model, task)
            else:
                tau_c, tau_src = None, None
            if tau_c is None:
                tau_c = tm.knn_tau_at_k(art, X_tcga, tc.labels.numpy(), folds, knn_k)
                tau_src = ("refit_at_k%d (no stored entry)" % knn_k
                           if knn_k == tm.KNN_K_VALIDATED else
                           "refit_at_k%d (store only covers k=%d)"
                           % (knn_k, tm.KNN_K_VALIDATED))
                log.warning(
                    f"  {kind}: no stored k={knn_k} tau for {method}/{model}; "
                    f"derived one by the same rule (Youden J on TCGA out-of-fold "
                    f"at k={knn_k}) -> {tau_c:.6f}")
            probs_c = tm.knn_probs_at_k(art, X_tgt, knn_k)
        except Exception as exc:                       # never break the sweep
            log.warning(f"  {kind}: corrected mode unavailable ({exc})")
            return
        entry["auroc_corrected"] = float(_metrics_at(targets, probs_c, tau_c)["auroc"])
        entry["knn_k_corrected"] = int(knn_k)
        entry["knn_tau_source"] = tau_src
    else:
        if g.empty:
            log.warning(f"  {kind}: no TCGA out-of-fold table; "
                        f"corrected threshold skipped")
            return
        probs_c = probs
        tau_c = tm.corrected_threshold(kind, tau_frozen, g.prob_pos.values, probs_c)

    m = _metrics_at(targets, probs_c, tau_c)
    npp = int((probs_c >= tau_c).sum())
    entry.update({
        "threshold_mode": "corrected",
        "threshold_corrected": float(tau_c),
        "bacc_corrected": float(m["bacc"]),
        "acc_corrected": float(m["acc"]),
        "macro_f1_corrected": float(m["macro_f1"]),
        "weighted_f1_corrected": float(m["weighted_f1"]),
        "conf_matrix_corrected": m["conf_matrix"],
        "n_pos_pred_corrected": npp,
        "status_corrected": tm.health(npp, n),
        "corrected_scheme": ("refit_tau_k%d" % knn_k if kind == "knn"
                             else "quantile_rate_matched"),
    })


#: Metric fields the promotion swaps. ``weighted_f1`` is included so a promoted
#: row has no field still quoting the frozen operating point.
_PROMOTED_METRICS = ("bacc", "acc", "macro_f1", "weighted_f1", "conf_matrix")


def _promoted_policy(promote: List[str], classifiers: List[str], knn_k: int) -> str:
    """One sentence naming both schemes and which heads use each."""
    frozen = [c for c in classifiers if c not in promote]
    parts = []
    if "knn" in promote:
        parts.append(f"knn: k={knn_k} scores, tau refit on TCGA out-of-fold at "
                     f"that k (knn_k35_thresholds.json)")
    rate = [c for c in promote if c != "knn"]
    if rate:
        parts.append(f"{'/'.join(rate)}: rate-matched (quantile) threshold, "
                     f"r from tau_TCGA on TCGA out-of-fold")
    if frozen:
        parts.append(f"{'/'.join(frozen)}: frozen tau_TCGA, Youden J on pooled "
                     f"out-of-fold TCGA-CV probabilities (unchanged)")
    return "TWO SCHEMES IN ONE TABLE - " + "; ".join(parts)


def _promote(entry: Dict, kind: str, log) -> bool:
    """Make the corrected operating point this row's **headline**. Returns ok.

    ``_add_corrected`` deliberately writes beside the frozen keys so published
    numbers never move. Promotion is the one place that decision is reversed, and
    it is reversed on purpose: for the promoted heads the corrected scheme *is*
    the reported result.

    Nothing is discarded. Every displaced value is kept under ``*_frozen`` in the
    same record, so the pre-promotion number is still readable without going back
    to the archive.

    Two things are deliberately **not** carried over:

    * ``auroc`` moves only when the scheme changed the scores. The quantile rule
      moves the cut point, not the ranking, so RF's AUROC is identical either way
      and copying ``auroc_corrected`` over it would imply a change that did not
      happen. KNN's k really does change the scores, so its AUROC does move.
    * the across-seed SDs describe five models scored at the frozen tau. The
      corrected point is computed once, from the seed-averaged probabilities, so
      it has no SD of its own. Inventing one, or leaving the frozen one in place
      under a corrected headline, would both be wrong - they are preserved as
      ``*_sd_frozen`` and the headline SD is cleared.

    Returns False when ``_add_corrected`` produced nothing (a missing artifact,
    say). The caller must treat that as fatal: a row left silently frozen inside
    a table captioned "corrected" is exactly the error this whole change is
    meant to remove.
    """
    if "bacc_corrected" not in entry:
        log.error(f"  [{kind}] PROMOTION FAILED - no corrected values were "
                  f"computed for this row; it would stay frozen while the table "
                  f"reports it as corrected.")
        return False

    for key in _PROMOTED_METRICS:
        entry[f"{key}_frozen"] = entry[key]
        entry[key] = entry[f"{key}_corrected"]

    entry["auroc_frozen"] = entry["auroc"]
    if "auroc_corrected" in entry:                # KNN only: k changed the scores
        entry["auroc"] = entry["auroc_corrected"]

    entry["threshold_frozen"] = entry["threshold"]
    entry["threshold"] = entry["threshold_corrected"]

    for key in ("bacc", "acc", "macro_f1", "weighted_f1", "auroc"):
        sd = f"{key}_sd"
        if sd in entry:
            entry[f"{key}_sd_frozen"] = entry[sd]
            entry[sd] = None

    entry["threshold_scheme"] = entry.get("corrected_scheme", "corrected")
    entry["promoted"] = True
    entry["status"] = entry.get("status_corrected")
    log.info(f"  [{kind}] PROMOTED to headline ({entry['threshold_scheme']}): "
             f"bacc {entry['bacc_frozen']:.4f} -> {entry['bacc']:.4f}, "
             f"auroc {entry['auroc_frozen']:.4f} -> {entry['auroc']:.4f}, "
             f"tau {entry['threshold_frozen']:.6f} -> {entry['threshold']:.6f}")
    return True


def run_external_validation(
    experiment: str,
    target_cohort: str,
    method: str,
    model: str,
    task: str = "MSIH",
    classifiers: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    variants: Optional[List[str]] = None,
    threshold_mode: str = tm.DEFAULT_MODE,
    knn_k: int = tm.KNN_K_VALIDATED,
    promote_classifiers: Optional[List[str]] = None,
) -> dict:
    """Apply TCGA-trained models to an external cohort, all three variants.

    ``threshold_mode='frozen'`` (default) is unchanged behaviour. ``'corrected'``
    ADDS a second operating point per classifier; it never replaces the frozen
    one, so published numbers stay reproducible either way. ``'promoted'``
    applies the corrected scheme to ``promote_classifiers`` only and makes it
    their headline - the one mode that moves published numbers, by design.
    """
    log = runlog.get_logger()
    classifiers = list(classifiers or MODEL_TYPES)
    seeds = list(seeds or SEEDS)
    variants = list(variants or VARIANTS)
    promote = list(promote_classifiers if promote_classifiers is not None
                   else tm.PROMOTED_CLASSIFIERS)
    if threshold_mode == "promoted":
        unknown = [c for c in promote if c not in MODEL_TYPES]
        if unknown:
            raise ValueError(f"cannot promote unknown classifier(s) {unknown}")
        log.warning(
            f"threshold-mode=promoted: {promote} take the corrected scheme as "
            f"their HEADLINE metrics; "
            f"{[c for c in classifiers if c not in promote]} keep the frozen tau "
            f"and get no corrected columns. This moves published numbers.")

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
                # `corrected` adds columns; it never replaces the frozen ones,
                # so an unflagged run is byte-identical to the published result.
                if threshold_mode == "corrected":
                    _add_corrected(per_variant["tcga_full"], kind, method, model,
                                   task, coh, targets, tau, knn_k, log)
                # `promoted` touches only the listed heads. Everything else falls
                # through untouched - not "corrected but reported frozen", simply
                # never evaluated under the corrected scheme, so no stale
                # corrected column can be mistaken for a live one.
                elif threshold_mode == "promoted" and kind in promote:
                    _add_corrected(per_variant["tcga_full"], kind, method, model,
                                   task, coh, targets, tau, knn_k, log)
                    if not _promote(per_variant["tcga_full"], kind, log):
                        raise RuntimeError(
                            f"{experiment} {method}/{model}: {kind} was selected "
                            f"for promotion but no corrected values could be "
                            f"computed. Refusing to write a row that would be "
                            f"frozen in a table captioned corrected.")

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
                             threshold_mode=threshold_mode,
                             knn_k_corrected=(
                                 knn_k if threshold_mode in ("corrected", "promoted")
                                 else None),
                             promoted_classifiers=(
                                 promote if threshold_mode == "promoted" else None),
                             # Under `promoted` one policy no longer describes the
                             # table, so the stamp states both and who gets which.
                             threshold_policy=(
                                 _promoted_policy(promote, classifiers, knn_k)
                                 if threshold_mode == "promoted" else
                                 "tau_TCGA, Youden J on pooled "
                                 "out-of-fold TCGA-CV probabilities")),
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
        # A promoted row has no across-seed SD: its operating point is computed
        # once from the seed-averaged probabilities. `is not None` rather than
        # `in`, because the key is present and deliberately null.
        sd = (f" +/-{m['bacc_sd']:.4f}" if m.get("bacc_sd") is not None else "")
        scheme = f"  [{m['threshold_scheme']}]" if m.get("promoted") else ""
        log.info(f"      {v:<14} bacc={m['bacc']:.4f}{sd}  auroc={m['auroc']:.4f}  "
                 f"cm={m['conf_matrix']}{scheme}")


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
                # Which scheme produced THIS row. Under `promoted` a single table
                # holds both, so a bare number is not self-describing any more.
                "Threshold_scheme": m.get("threshold_scheme", "frozen_tau_TCGA"),
            }
            if m.get("promoted"):
                row.update({
                    "BalAcc_frozen_prev": m.get("bacc_frozen"),
                    "AUROC_frozen_prev": m.get("auroc_frozen"),
                    "Threshold_frozen_prev": m.get("threshold_frozen"),
                    "Status": m.get("status"),
                })
            # "How much did doubling the training data buy?" - the work order
            # asks for this delta explicitly.
            if base and v == "tcga_full":
                row["d_BalAcc_vs_legacy"] = m["bacc"] - base["bacc"]
                row["d_AUROC_vs_legacy"] = m["auroc"] - base["auroc"]
            # Corrected-mode columns sit BESIDE the frozen ones. Absent in a
            # default run, so summary_ev.csv keeps its published shape.
            if "bacc_corrected" in m:
                row.update({
                    "BalAcc_frozen": m["bacc"],
                    "BalAcc_corrected": m["bacc_corrected"],
                    "AUROC_corrected": m.get("auroc_corrected", m["auroc"]),
                    "Threshold_corrected": m["threshold_corrected"],
                    "Status_frozen": m.get("status_frozen"),
                    "Status_corrected": m.get("status_corrected"),
                    "N_pos_pred_corrected": m.get("n_pos_pred_corrected"),
                    "Corrected_scheme": m.get("corrected_scheme"),
                })
            rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "summary_ev.csv", index=False)


# --------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------


def sweep(experiment: str, target_cohort: str, methods: List[str], models: List[str],
          task: str = "MSIH", classifiers: Optional[List[str]] = None,
          seeds: Optional[List[int]] = None, variants: Optional[List[str]] = None,
          force: bool = False, archive: bool = True,
          threshold_mode: str = tm.DEFAULT_MODE,
          knn_k: int = tm.KNN_K_VALIDATED,
          promote_classifiers: Optional[List[str]] = None) -> None:
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
                                        task, classifiers, seeds, variants,
                                        threshold_mode=threshold_mode, knn_k=knn_k,
                                        promote_classifiers=promote_classifiers)
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
    ap.add_argument("--methods", default=None)
    ap.add_argument("--models", default=",".join(P.CANONICAL_MODELS))
    ap.add_argument("--classifiers", default=",".join(MODEL_TYPES))
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--task", default="MSIH")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--threshold-mode", default=tm.DEFAULT_MODE, choices=list(tm.MODES),
                    help="frozen (default, unchanged); corrected (adds a second "
                         "operating point beside the frozen one for every head); "
                         "promoted (corrected scheme for --promote-classifiers "
                         "ONLY, as their headline - this moves published numbers)")
    ap.add_argument("--knn-k", type=int, default=tm.KNN_K_VALIDATED,
                    help="neighbourhood size used by corrected/promoted; "
                         "the frozen path always uses the grid-searched k")
    ap.add_argument("--promote-classifiers", default=",".join(tm.PROMOTED_CLASSIFIERS),
                    help="comma list; only meaningful with "
                         "--threshold-mode=promoted (default: knn,rf)")
    ap.add_argument("--no-archive", action="store_true",
                    help="skip the automatic _ARCHIVED_ copy of the previous "
                         "tree; use only when a stamped backup already exists")
    args = ap.parse_args()

    methods = (args.methods.split(",") if args.methods else [
        method for method in P.AGGREGATION_METHODS
        if not (args.experiment == "SurGen-EV" and method == "PRISM")
    ])

    cohort = "paip" if args.experiment == "PAIP-EV" else "surgen"
    sweep(experiment=args.experiment, target_cohort=cohort,
          methods=methods, models=args.models.split(","),
          task=args.task, classifiers=args.classifiers.split(","),
          seeds=[int(s) for s in args.seeds.split(",")],
          variants=args.variants.split(","), force=args.force,
          archive=not args.no_archive,
          threshold_mode=args.threshold_mode, knn_k=args.knn_k,
          promote_classifiers=[c for c in args.promote_classifiers.split(",") if c])


if __name__ == "__main__":
    main()
