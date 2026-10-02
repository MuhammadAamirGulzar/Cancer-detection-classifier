"""PAIP-IV: internal validation on the provider's own train/test split (Task 3.1).

**IV (internal validation)** - a single train/test split defined by the dataset
provider; no folds, no averaging. This is the right protocol for PAIP because
PAIP *ships* a split; K-fold is what you use when a cohort does not (TCGA,
SurGen). PAIP-CV was retired in Task 2.2 for exactly this reason - it trained on
less data (~36) and tested on smaller sets (18, ~4 positives) while answering the
same question.

The split
---------
    train  paip_47slides.csv         47 ids, **all 47 have features** (12 MSI-H)
    test   paip_31slides_labels.csv  31 ids, **all 31 have features** (7 MSI-H)

[UPDATED 2026-08-10] This used to read 42 of 47, because training_data_19/30/41/
42/46 had no features. Their patches were recovered from an older tree, the
features and slide aggregations were rebuilt, and the split is now the full 47/31
the provider defines - N_train=47 in every summary_iv.csv confirms it. Train
MSI-H went 10 -> 12.

A stratified validation subset is carved from the 47 training slides for ANN
early stopping. The 31 test slides are never touched by any tuning decision.

ANN protocol (``--ann-protocol``)
---------------------------------
That carve-out is 9 slides, and the ANN is the only classifier that both searches
18 configurations on it and trains on less data because of it. Measured over the
22 combinations: the 18 configurations collapse to a median of 3 distinct
validation macro-F1 values, and the one validation picks averages 0.6075 test
macro-F1 against 0.6133 for the grid's median entry - the search is worse than
random over its own grid. Meanwhile lin/knn/proto/rf merge the 9 slides back and
train on all 47 while the ANN trains on 38.

  ``legacy`` (default)  search 18 points on the 9 slides, train on 38.
                        Reproduces the published numbers exactly.
  ``pinned``            no search: use the TCGA-CV-derived configuration from
                        ``runners.hparams``. TCGA is a different cohort, so no
                        PAIP information enters the choice.
  ``refit``             search as in ``legacy``, then refit the winner on all 47.
  ``fixed``             ``pinned`` + ``refit``. Puts the ANN on the same footing
                        as the other four classifiers.
  ``full``              ``fixed`` without the discarded selection fit: no slide
                        is held back for tuning at all. Numerically identical to
                        ``fixed`` - the refit re-seeds - but honest about the
                        carve-out being unused. ``n_ann_early_stop`` reports 0.

The protocol is recorded in the result JSON and in ``summary_iv.csv`` so every
row says which one produced it.

Threshold
---------
PAIP-IV is an *internal* experiment: train and test distributions are matched, so
it reports at the default 0.5 (work order section 1b rule 2). tau_TCGA applies to
external experiments only. **0.5 remains the headline and is not replaced.**

``--fit-threshold`` additionally fits tau_train by 5-fold stratified inner CV
over the 47 TRAINING slides: each slide gets one prediction from a model that did
not see it, the predictions are pooled, and the threshold is the Youden-J optimum
over that pooled set. The 31 test slides take no part in it. Results at
tau_train are written to ``threshold_audit.csv`` and to three extra columns in
``summary_iv.csv`` alongside the 0.5 figures - never instead of them.

It runs for EVERY classifier, not only the ANN. Calibrating one head and not its
competitors would replace one unfair comparison with a different one. It costs
5x the training time.

Why this exists: at threshold 0.5 the five heads have identical false-negative
counts (median 2 of 7) and the entire BalAcc spread comes from false positives -
median 3 of 24 for lin against 8 for the ANN, while their AUROCs differ by 0.009.
The gap is an operating point, not a ranking.

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
ANN_VAL_FRAC = 0.20   # of the 47 training slides -> 9 slides, early stopping only

#: See the module docstring. ``legacy`` is the default and changes nothing.
ANN_PROTOCOLS = ("legacy", "pinned", "refit", "fixed", "full")

#: Configurations taken verbatim from the collaborator's standalone ANN scripts,
#: as they behave when driven through their own ``eval_ANN``: hidden_dim=512
#: (class default) and max_iter=1000 (the eval_ANN default). Neither script has a
#: grid or any stored configuration - every value is a default argument or a
#: hardcoded literal. The two differ only in dropout. ``ann_old`` additionally
#: puts nn.Softmax inside the training graph while using CrossEntropyLoss, which
#: applies log-softmax again - that is a defect, not a hyperparameter, and is not
#: reproduced here. Only its dropout is.
ANN_PRESETS: Dict[str, dict] = {
    "ann_edited": {"arch": "shallow", "dropout": 0.7,
                   "hidden_dim1": 512, "hidden_dim2": None, "max_iter": 1000},
    "ann_old":    {"arch": "shallow", "dropout": 0.5,
                   "hidden_dim1": 512, "hidden_dim2": None, "max_iter": 1000},
}


def _ann_protocol_options(protocol: str, method: str, model: str, task: str,
                          max_iter: Optional[int] = None, arch: str = "deep",
                          dropout: Optional[float] = None,
                          preset: Optional[str] = None) -> dict:
    """Keyword arguments for ``train_and_evaluate`` for one protocol.

    ``pinned``/``fixed``/``full`` read the TCGA-CV-derived configuration. If
    TCGA-CV has not been run there is nothing to pin to, and silently falling
    back to the 18-point search would make the protocol label a lie - so it
    raises.

    ``max_iter`` overrides the iteration count. It only applies to the pinned
    protocols: under ``legacy``/``refit`` the count is one of the searched
    dimensions, and overriding it there would halve the grid while still
    reporting an 18-point search. Note that ``full`` holds nothing back, so a
    value passed here is an external decision, not one the run derived - which
    is why it is written into every run stamp.
    """
    if protocol not in ANN_PROTOCOLS:
        raise ValueError(f"Unknown --ann-protocol {protocol!r}; expected one of {ANN_PROTOCOLS}")

    hparams = None
    if preset is not None:
        # The preset IS the fixed configuration, so TCGA-CV is not consulted.
        if preset not in ANN_PRESETS:
            raise ValueError(f"Unknown --ann-preset {preset!r}; "
                             f"expected one of {tuple(ANN_PRESETS)}")
        if protocol == "legacy":
            raise ValueError(
                "--ann-preset fixes the configuration, so it cannot be combined "
                "with --ann-protocol legacy, which searches for one. Use pinned, "
                "fixed or full.")
        spec = ANN_PRESETS[preset]
        hparams = {k: spec[k] for k in ("hidden_dim1", "hidden_dim2", "max_iter")}
        arch = spec["arch"]
        dropout = spec["dropout"] if dropout is None else dropout
    elif protocol in ("pinned", "fixed", "full"):
        from runners.hparams import load_hparams
        hparams = load_hparams(method, model, "ann", task)

    if max_iter is not None:
        if hparams is None:
            raise ValueError(
                f"--ann-max-iter needs a pinned protocol (pinned/fixed/full); "
                f"under {protocol!r} the iteration count is part of the search grid.")
        if max_iter < 1:
            raise ValueError(f"--ann-max-iter must be >= 1, got {max_iter}")
        hparams = {**hparams, "max_iter": int(max_iter)}

    if arch not in ("deep", "shallow"):
        raise ValueError(f"--ann-arch must be 'deep' or 'shallow', got {arch!r}")
    if dropout is not None and not (0.0 <= dropout < 1.0):
        raise ValueError(f"--ann-dropout must be in [0, 1), got {dropout}")

    return {"ann_hparams": hparams,
            "ann_refit_on_trainval": protocol in ("refit", "fixed", "full"),
            "ann_skip_selection_fit": protocol == "full",
            "ann_arch": arch,
            "ann_dropout": dropout}


#: Inner cross-validation used to fit a decision threshold on the TRAINING
#: slides. Configurable via --tau-folds.
#:
#: Note which direction helps. The measured failure of tau_train is that the
#: inner models train on less data than the model the threshold is applied to,
#: so their probabilities are compressed and Youden's J picks too low a cut. At
#: k folds the inner models see (k-1)/k x 47 slides:
#:
#:     k=4   35.3 slides   gap to the final model: 11.7
#:     k=5   37.6 slides   gap:  9.4   (default)
#:     k=10  42.3 slides   gap:  4.7
#:     k=47  46.0 slides   gap:  1.0   (leave-one-out)
#:
#: FEWER folds widens the gap and should make the mismatch worse; more folds
#: narrows it. The pooled fit always uses all 47 predictions regardless of k.
TAU_TRAIN_FOLDS = 5


def _fit_threshold_on_train(coh, tr_all, kind, input_dim, seed, ann_opts,
                            n_splits: int = TAU_TRAIN_FOLDS):
    # n_splits is capped at the positive count - StratifiedKFold cannot make more
    # folds than the rarest class has members.
    """Fit a decision threshold using ONLY the training slides.

    Runs a stratified k-fold inside the 47 training slides. Each slide gets one
    prediction from a model that did not see it, the predictions are pooled, and
    the threshold is the Youden-J optimum over that pooled set. The 31 test
    slides take no part in any of it - which is the whole point, since a
    threshold chosen by checking the test set is not a result, it is a fit.

    Applied to EVERY classifier, not just the ANN. Calibrating one head and not
    the others would replace one unfair comparison with a different one.

    Returns ``(tau, info)`` or ``(None, info)`` if the fit is not possible.
    """
    from sklearn.model_selection import StratifiedKFold
    from runners.thresholds import youden_threshold

    y = coh.labels[tr_all].numpy()
    if len(np.unique(y)) < 2 or y.sum() < n_splits:
        return None, {"fitted": False,
                      "reason": f"too few positives ({int(y.sum())}) for "
                                f"{n_splits}-fold stratified inner CV"}

    oof = np.full(len(tr_all), np.nan, dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for inner_tr, inner_te in skf.split(np.zeros(len(tr_all)), y):
        rows_tr, rows_te = tr_all[inner_tr], tr_all[inner_te]
        # The ANN still needs an early-stopping split; carve it from the inner
        # training rows so the inner test fold stays untouched.
        sub_tr, sub_va = dl.stratified_holdout(coh.labels[rows_tr], ANN_VAL_FRAC, seed=seed)
        _m, dump, _sel = train_and_evaluate(
            fold=0,
            train=coh.subset(rows_tr[sub_tr])[:2],
            valid=coh.subset(rows_tr[sub_va])[:2],
            test=coh.subset(rows_te)[:2],
            model_type=kind, input_dim=input_dim,
            model_save_path=None,        # inner folds must not touch the checkpoints
            seed=seed, verbose=False,
            **(ann_opts if kind == "ann" else {}),
        )
        oof[inner_te] = np.asarray(dump["probs_all"])[:, 1]

    if np.isnan(oof).any():
        return None, {"fitted": False, "reason": "inner CV left slides unpredicted"}

    fit = youden_threshold(y, oof)
    return float(fit["tau"]), {"fitted": True, "n_splits": n_splits,
                               "n_train": int(len(tr_all)),
                               "n_pos": int(y.sum()), **fit}


def _threshold_audit(method: str, model: str, task: str, kind: str,
                     targets: np.ndarray, probs_pos: np.ndarray,
                     tau_train: Optional[float] = None) -> List[dict]:
    """Test metrics at 0.5 and at tau_TCGA, for the audit CSV only.

    PAIP-IV reports at 0.5 and that does not change - it is an internal
    experiment (section 1b rule 2). But the ANN's balanced accuracy at 0.5 is
    dominated by where its probabilities sit rather than how well it ranks
    slides, and this file is the evidence. tau_TCGA is fitted on TCGA
    out-of-fold probabilities, so reading it here leaks nothing from PAIP.
    """
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix
    from runners.thresholds import load_tau

    points = [("0.5", 0.5)]
    try:
        points.append(("tau_TCGA", load_tau(method, model, kind, task)))
    except (FileNotFoundError, KeyError):
        pass  # TCGA-CV not run for this combination; the 0.5 row still stands
    if tau_train is not None:
        points.append(("tau_train", tau_train))

    out = []
    for label, tau in points:
        preds = (probs_pos >= tau).astype(int)
        out.append({
            "experiment": "PAIP-IV", "method": method, "model": model,
            "classifier": kind, "threshold_label": label, "threshold": float(tau),
            "bacc": float(balanced_accuracy_score(targets, preds)),
            "auroc": float(roc_auc_score(targets, probs_pos)),
            "conf_matrix": str(confusion_matrix(targets, preds).tolist()),
        })
    return out


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
                classifiers: Optional[List[str]] = None, seed: int = 42,
                ann_protocol: str = "legacy", ann_max_iter: Optional[int] = None,
                ann_arch: str = "deep", ann_dropout: Optional[float] = None,
                ann_preset: Optional[str] = None,
                fit_threshold: bool = False,
                tau_folds: int = TAU_TRAIN_FOLDS) -> dict:
    """One (method, model) through the provider's 47/31 split."""
    log = runlog.get_logger()
    classifiers = list(classifiers or MODEL_TYPES)
    ann_opts = _ann_protocol_options(ann_protocol, method, model, task,
                                     ann_max_iter, ann_arch, ann_dropout, ann_preset)

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
    # lin/knn/proto/rf merge the carve-out back (combine_trainval=True), so they
    # train on all of tr_all. The ANN only does so under refit/fixed/full.
    ann_n = len(tr_all) if ann_opts["ann_refit_on_trainval"] else len(tr_rows)
    ann_es = 0 if ann_opts["ann_skip_selection_fit"] else len(va_rows)
    log.info(f"  train: {len(tr_all)} for lin/knn/proto/rf, "
             f"{ann_n} for ann (protocol={ann_protocol}, {ann_es} held back) "
             f"| test={len(te_rows)} "
             f"| train MSI-H={int(coh.labels[tr_all].sum())} test MSI-H={int(targets.sum())}")

    rows, per_clf, audit_rows = [], {}, []
    for kind in classifiers:
        with runlog.timed(f"PAIP-IV {method}/{model} {kind}", log):
            metrics, dump, selection = train_and_evaluate(
                fold=0,
                train=coh.subset(tr_rows)[:2], valid=coh.subset(va_rows)[:2],
                test=coh.subset(te_rows)[:2],
                model_type=kind, input_dim=coh.dim,
                model_save_path=str(models_dir), seed=seed, verbose=False,
                **(ann_opts if kind == "ann" else {}),
            )
        probs_pos = np.asarray(dump["probs_all"])[:, 1]
        preds = np.asarray(dump["preds_all"])
        ci = bootstrap_ci(targets, probs_pos, preds, seed=seed)

        # Threshold fitted on the training slides only. Reported alongside the
        # 0.5 headline, never replacing it - PAIP-IV's policy is 0.5 (section 1b
        # rule 2) and that stays the published operating point.
        tau_train, tau_info = (None, {"fitted": False, "reason": "not requested"})
        if fit_threshold:
            with runlog.timed(f"  tau_train {kind} ({tau_folds}-fold inner CV)", log):
                tau_train, tau_info = _fit_threshold_on_train(
                    coh, tr_all, kind, coh.dim, seed,
                    ann_opts if kind == "ann" else {}, n_splits=tau_folds)
            if tau_train is not None:
                from sklearn.metrics import balanced_accuracy_score, confusion_matrix
                tp = (probs_pos >= tau_train).astype(int)
                log.info(f"    tau_train={tau_train:.4f} -> bacc "
                         f"{balanced_accuracy_score(targets, tp):.4f} "
                         f"cm={confusion_matrix(targets, tp).tolist()}")

        entry = {
            "classifier": kind,
            "bacc": float(metrics[f"{kind}_bacc"]),
            "auroc": float(metrics[f"{kind}_auroc"]),
            "acc": float(metrics[f"{kind}_acc"]),
            "macro_f1": float(metrics[f"{kind}_macro_f1"]),
            "weighted_f1": float(metrics[f"{kind}_weighted_f1"]),
            "conf_matrix": metrics[f"{kind}_conf_matrix"].tolist(),
            "threshold": 0.5,
            "n_train": int(len(tr_all)),
            # Under "full" nothing is held back: the carve-out is computed but
            # never used, so reporting 9 here would misstate the protocol.
            "n_ann_early_stop": 0 if (kind == "ann" and ann_protocol == "full")
                                else int(len(va_rows)),
            "n_test": int(len(te_rows)),
            # How many slides this classifier actually fitted on. lin/knn/proto/rf
            # merge the carve-out back; the ANN only does under refit/fixed/full.
            "n_train_effective": int(
                len(tr_rows) if (kind == "ann" and not ann_opts["ann_refit_on_trainval"])
                else len(tr_all)),
            "ann_protocol": ann_protocol if kind == "ann" else None,
            "bootstrap": ci,
            "selection": selection,
            "tau_train": tau_train,
            "tau_train_fit": tau_info,
        }
        if tau_train is not None:
            from sklearn.metrics import (balanced_accuracy_score as _bas,
                                         confusion_matrix as _cm,
                                         f1_score as _f1, accuracy_score as _acc)
            _p = (probs_pos >= tau_train).astype(int)
            entry["at_tau_train"] = {
                "threshold": float(tau_train),
                "bacc": float(_bas(targets, _p)),
                "acc": float(_acc(targets, _p)),
                "macro_f1": float(_f1(targets, _p, average="macro")),
                "conf_matrix": _cm(targets, _p).tolist(),
            }
        per_clf[kind] = entry
        audit_rows += _threshold_audit(method, model, task, kind, targets, probs_pos,
                                       tau_train)
        log.info(f"    bacc={entry['bacc']:.4f} "
                 f"[{ci['bacc_ci'][0]:.4f}, {ci['bacc_ci'][1]:.4f}]  "
                 f"auroc={entry['auroc']:.4f} "
                 f"[{ci['auroc_ci'][0]:.4f}, {ci['auroc_ci'][1]:.4f}]  "
                 f"cm={entry['conf_matrix']}")

        rows.append({
            "Experiment": "PAIP-IV", "Method": method, "Model": model,
            "Classifier": kind, "N_train": len(tr_all), "N_test": len(te_rows),
            "N_train_effective": entry["n_train_effective"],
            "AnnProtocol": entry["ann_protocol"] or "",
            "BalAcc": entry["bacc"],
            "BalAcc_CI_low": ci["bacc_ci"][0], "BalAcc_CI_high": ci["bacc_ci"][1],
            "AUROC": entry["auroc"],
            "AUROC_CI_low": ci["auroc_ci"][0], "AUROC_CI_high": ci["auroc_ci"][1],
            "Acc": entry["acc"], "MacroF1": entry["macro_f1"],
            "Threshold": 0.5, "ConfMatrix": str(entry["conf_matrix"]),
            "TauTrain": tau_train if tau_train is not None else "",
            "BalAcc_at_TauTrain": (entry["at_tau_train"]["bacc"]
                                   if tau_train is not None else ""),
            "ConfMatrix_at_TauTrain": (str(entry["at_tau_train"]["conf_matrix"])
                                       if tau_train is not None else ""),
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
            ann_protocol=ann_protocol,
            ann_preset=ann_preset,
            ann_hparam_source=("collaborator script preset" if ann_preset else
                               "TCGA-CV majority vote (runners.hparams)"
                               if ann_opts["ann_hparams"] else
                               "searched on the PAIP early-stop carve-out"),
            ann_trained_on=f"{ann_n} of {len(tr_all)} training slides",
            ann_held_back_for_tuning=ann_es,
            ann_arch=ann_opts["ann_arch"],
            ann_n_hidden_layers=1 if ann_opts["ann_arch"] == "shallow" else 2,
            ann_dropout=ann_opts["ann_dropout"],
            ann_hidden_dim1=(ann_opts["ann_hparams"] or {}).get("hidden_dim1"),
            ann_max_iter=(ann_opts["ann_hparams"] or {}).get("max_iter"),
            ann_max_iter_overridden=ann_max_iter is not None,
        ),
        "results": per_clf,
    }
    (out_dir / f"result_PAIP-IV_{method}_{model}_default.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_dir / "summary_iv.csv", index=False)
    if audit_rows:
        pd.DataFrame(audit_rows).to_csv(out_dir / "threshold_audit.csv", index=False)
    log.info(f"  wrote -> {out_dir}")
    return payload


def sweep(methods: List[str], models: List[str], task: str = "MSIH",
          classifiers: Optional[List[str]] = None, seed: int = 42,
          force: bool = False, archive: bool = True,
          ann_protocol: str = "legacy", ann_max_iter: Optional[int] = None,
          ann_arch: str = "deep", ann_dropout: Optional[float] = None,
          ann_preset: Optional[str] = None, fit_threshold: bool = False,
          tau_folds: int = TAU_TRAIN_FOLDS) -> None:
    log = runlog.get_logger()
    ran, skipped = [], []
    if ann_protocol != "legacy":
        log.warning(f"ANN protocol: {ann_protocol} (not the published 'legacy'). "
                    f"lin/knn/proto/rf are unaffected.")
    if ann_preset is not None:
        log.warning(f"ANN configuration from preset {ann_preset!r} "
                    f"({ANN_PRESETS[ann_preset]}) - TCGA-CV hyperparameters not used.")
    if ann_max_iter is not None:
        log.warning(f"ANN max_iter overridden to {ann_max_iter}. Not chosen from "
                    f"held-out data - record the basis.")
    if archive:
        moved = archive_experiment_tree("PAIP-IV")
        if moved:
            log.warning(f"archived previous PAIP-IV results -> {moved.name}")

    for method in methods:
        for model in models:
            if not P.is_combination_valid(method, model):
                continue
            # "legacy" keeps the historical key so existing progress entries
            # still match and a default rerun resumes rather than restarts.
            key = runlog.progress_key(
                "PAIP-IV", method, model, seed=seed,
                **({} if ann_protocol == "legacy" else {"variant": ann_protocol}))
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
                run_paip_iv(method, model, task, classifiers, seed, ann_protocol,
                            ann_max_iter, ann_arch, ann_dropout, ann_preset,
                            fit_threshold, tau_folds)
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
    ap.add_argument("--no-archive", action="store_true",
                    help="do not move the previous PAIP-IV tree aside first")
    ap.add_argument("--ann-protocol", default="legacy", choices=ANN_PROTOCOLS,
                    help="legacy: 18-point search on 9 slides, train on 38 "
                         "(reproduces published numbers). pinned: TCGA-CV "
                         "configuration, no search. refit: search then refit on "
                         "all 47. fixed: pinned + refit. full: train once on all "
                         "47, nothing held back (same numbers as fixed).")
    ap.add_argument("--ann-max-iter", type=int, default=None,
                    help="override the iteration count; requires a pinned protocol")
    ap.add_argument("--ann-arch", default="deep", choices=("deep", "shallow"),
                    help="deep (default): two hidden layers, dropout 0.3. "
                         "shallow: one hidden layer, dropout 0.7")
    ap.add_argument("--ann-dropout", type=float, default=None,
                    help="override the dropout rate")
    ap.add_argument("--fit-threshold", action="store_true",
                    help=f"fit a decision threshold by {TAU_TRAIN_FOLDS}-fold inner "
                         f"CV on the TRAINING slides only, for every classifier. "
                         f"Reported alongside 0.5, which stays the headline. "
                         f"Costs {TAU_TRAIN_FOLDS}x the training time.")
    ap.add_argument("--tau-folds", type=int, default=TAU_TRAIN_FOLDS,
                    help=f"inner CV folds for --fit-threshold (default "
                         f"{TAU_TRAIN_FOLDS}). Fewer folds means the inner models "
                         f"train on less data and diverge further from the model "
                         f"the threshold is applied to.")
    ap.add_argument("--ann-preset", default=None, choices=tuple(ANN_PRESETS),
                    help="use a configuration taken verbatim from the standalone "
                         "ANN scripts instead of the TCGA-CV-derived one")
    args = ap.parse_args()
    sweep(args.methods.split(","), args.models.split(","), args.task,
          args.classifiers.split(","), args.seed, args.force,
          archive=not args.no_archive, ann_protocol=args.ann_protocol,
          ann_max_iter=args.ann_max_iter, ann_arch=args.ann_arch,
          ann_dropout=args.ann_dropout, ann_preset=args.ann_preset,
          fit_threshold=args.fit_threshold, tau_folds=args.tau_folds)


if __name__ == "__main__":
    main()
