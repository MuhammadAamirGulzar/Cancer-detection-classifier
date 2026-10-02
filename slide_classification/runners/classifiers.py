"""Classifier dispatch: one place where every classifier is configured and run.

Replaces the ``train_and_evaluate`` function that was copy-pasted into six
drivers. Takes plain tensors (no ``DataLoader``, work order 1c.1 item 3) and
returns metrics, a prediction dump, and - for the ANN - the selected model.

Hyperparameter grids
--------------------
``lin``, ``knn``, ``proto`` and ``rf`` have single-point grids, so no selection
occurs and their test metrics are unbiased. Verified across every driver during
Task 1.1; the grids are declared here so that stays auditable.

``knn`` is the one to be careful about: ``eval_knn`` runs an *internal*
``GridSearchCV`` over 30 parameter combinations, but it does so with
cross-validation **on the training data**, scoring balanced accuracy. That is a
legitimate nested search, not test-set selection.

``ann`` has a real 2x2 grid and is selected on **validation** macro-F1
(Task 1.1). Selecting on the test set was the pre-existing bug.

Small-cohort ANN protocol
-------------------------
Selecting 18 configurations on a validation set of 9 slides - which is what
PAIP-IV has - does not work. Measured over the 22 PAIP-IV combinations, the 18
configurations collapse to a median of 3 distinct validation macro-F1 values,
and the configuration validation picks averages **0.6075** test macro-F1 against
**0.6133** for the grid's median entry. The search performs slightly worse than
choosing at random from its own grid.

``ann_hparams`` and ``ann_refit_on_trainval`` let a caller opt out of that:

* ``ann_hparams`` pins the configuration (no search at all), normally to the one
  ``runners.hparams`` derived by majority vote across the 4 TCGA-CV folds, each
  validating on ~100 slides rather than 9. TCGA is a different cohort, so no
  PAIP information enters the choice.
* ``ann_refit_on_trainval`` refits the chosen configuration on train+validation
  with ``combine_trainval=True``. ``lin``/``knn``/``proto``/``rf`` already merge
  the validation split back (see ``logistic.py:19``, ``knn.py:33``,
  ``protonet.py:40``, ``r_forest_eval.py:240``), so without this the ANN is the
  only classifier in the table training on less data than its peers.
* ``ann_skip_selection_fit`` drops the pre-fit that a pinned+refit run would
  otherwise train and immediately discard, so no slide is held back at all.

All default to off, so existing callers reproduce published numbers exactly.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from eval_patch_features.logistic import eval_linear
from eval_patch_features.knn import eval_knn
from eval_patch_features.protonet import eval_protonet
from eval_patch_features.r_forest_eval import eval_r_forest
from eval_patch_features.ann import eval_ANN, save_ann_checkpoint

MODEL_TYPES = ("lin", "ann", "knn", "proto", "rf")

#: Single-point grids - preserved exactly as the notebooks had them so results
#: stay comparable. Changing any of these changes every downstream number.
GRIDS: Dict[str, Dict[str, list]] = {
    "lin": {"C": [10], "max_iter": [300]},
    "knn": {"n_neighbors": [3]},
    "proto": {},
    "rf": {
        "n_estimators": [500],
        "max_depth": [None],
        "min_samples_split": [5],
        "min_samples_leaf": [1],
        "class_weight": [{0: 1, 1: 10}],
        "prediction_threshold": [0.3],
    },
    # [WIDENED 2026-08-04] Was {128,256} x {64,128} x {500}. Two independent
    # reasons, both recorded in slide_classification/ann_grid_probe.csv:
    #
    #  * the old grid TRUNCATED. Across the 88 validation-selected configurations
    #    from the corrected TCGA-CV run, h1 sat at the grid maximum in 61% of
    #    cases and h2 in 65% - the classic sign the search space is too small.
    #  * a 32-pair probe (8 combinations x 4 folds, selecting on validation in
    #    both arms) measured what widening buys on TEST:
    #        macro-F1 +0.0196 (p=0.006)   accuracy +0.0208 (p=0.0007)
    #        BalAcc   +0.0115 (p=0.15)    AUROC    +0.0037 (p=0.63)
    #    So it significantly improves macro-F1 and accuracy, and does NOT
    #    significantly improve AUROC - the primary metric - or balanced accuracy.
    #    No metric degrades on average. Reported that way, not as a headline win.
    #
    # This is only legitimate because Task 1.1 moved selection to VALIDATION.
    # Under the old test-set selection a wider grid would have bought nothing but
    # more optimistic bias.
    "ann": {"hidden_dim1": [128, 256, 512], "hidden_dim2": [64, 128, 256],
            "max_iter": [500, 1000]},
}

Split = Tuple[torch.Tensor, torch.Tensor]  # (feats, labels)


def train_and_evaluate(
    fold: int,
    train: Split,
    valid: Split,
    test: Split,
    model_type: str,
    input_dim: int,
    model_save_path: Optional[str] = None,
    seed: int = 42,
    verbose: bool = False,
    ann_hparams: Optional[Dict[str, Any]] = None,
    ann_refit_on_trainval: bool = False,
    ann_skip_selection_fit: bool = False,
    ann_arch: str = "deep",
    ann_dropout: Optional[float] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[dict]]:
    """Fit one classifier on one split. Returns ``(metrics, dump, selection)``.

    ``selection`` is ``None`` for single-point grids and, for the ANN, records
    which configuration was chosen and on what score.

    The ``ann_*`` arguments are ignored by every classifier except the ANN; see
    the module docstring for what they are for.
    """
    trf, trl = train
    vaf, val = valid
    tef, tel = test

    torch.manual_seed(seed)
    np.random.seed(seed)

    if model_type == "lin":
        g = GRIDS["lin"]
        metrics, dump = eval_linear(
            fold=fold, train_feats=trf, train_labels=trl,
            valid_feats=vaf, valid_labels=val, test_feats=tef, test_labels=tel,
            max_iter=g["max_iter"][0], C=g["C"][0],
            save_path=model_save_path, verbose=verbose,
        )
        return metrics, dump, None

    if model_type == "knn":
        metrics, dump = eval_knn(
            fold=fold, train_feats=trf, train_labels=trl,
            val_feats=vaf, val_labels=val, test_feats=tef, test_labels=tel,
            n_neighbors=GRIDS["knn"]["n_neighbors"][0], normalize_feats=True,
            model_save_path=model_save_path, verbose=verbose,
        )
        return metrics, dump, None

    if model_type == "proto":
        metrics, dump = eval_protonet(
            fold=fold, train_feats=trf, train_labels=trl,
            val_feats=vaf, val_labels=val, test_feats=tef, test_labels=tel,
            normalize_feats=True, model_save_path=model_save_path,
        )
        return metrics, dump, None

    if model_type == "rf":
        g = GRIDS["rf"]
        metrics, dump = eval_r_forest(
            fold=fold, train_feats=trf, train_labels=trl,
            valid_feats=vaf, valid_labels=val, test_feats=tef, test_labels=tel,
            n_estimators=g["n_estimators"][0], max_depth=g["max_depth"][0],
            min_samples_split=g["min_samples_split"][0],
            min_samples_leaf=g["min_samples_leaf"][0],
            class_weight=g["class_weight"][0],
            prediction_threshold=g["prediction_threshold"][0],
            save_path=model_save_path, verbose=verbose,
        )
        return metrics, dump, None

    if model_type == "ann":
        return _run_ann_grid(fold, train, valid, test, input_dim,
                             model_save_path, seed, verbose,
                             fixed_hparams=ann_hparams,
                             refit_on_trainval=ann_refit_on_trainval,
                             skip_selection_fit=ann_skip_selection_fit,
                             arch=ann_arch, dropout=ann_dropout)

    raise ValueError(f"Unsupported model type: {model_type!r}. Expected one of {MODEL_TYPES}")


def _run_ann_grid(fold, train, valid, test, input_dim, model_save_path, seed, verbose,
                  fixed_hparams=None, refit_on_trainval=False, skip_selection_fit=False,
                  arch="deep", dropout=None):
    """ANN grid search, selected on VALIDATION macro-F1.

    [CORRECTION 2026-08-04, work order Task 1.1] The selection criterion was
    ``eval_metrics['ann_macro_f1']``, which ``eval_ANN`` computes on the TEST
    set - the best of four configurations was chosen using test performance and
    then reported as a test result. It is now ``val_ann_macro_f1``.

    [CORRECTION 2026-08-04, work order Task 1.2] The checkpoint is written once,
    here, after selection - not inside the loop, where all four configurations
    used to overwrite the same file.

    ``fixed_hparams`` reduces the grid to a single point, so no selection occurs.
    ``refit_on_trainval`` then retrains that configuration on train+validation,
    which is what the four non-ANN classifiers already do. ``skip_selection_fit``
    drops the pre-fit entirely - with the configuration already fixed and a refit
    coming, that first model is trained and immediately discarded, so skipping it
    means no slide is held back for tuning. The refit re-seeds before building
    its network, so skipping the pre-fit cannot perturb it: that path is
    numerically identical to fixed_hparams + refit_on_trainval, only cheaper.
    """
    trf, trl = train
    vaf, val = valid
    tef, tel = test

    if arch not in ("deep", "shallow"):
        raise ValueError(f"ann arch must be 'deep' or 'shallow', got {arch!r}")

    if fixed_hparams:
        # hidden_dim2 may legitimately be absent or None - the single-hidden-layer
        # network has no second width, and int(None) would raise.
        _h2 = fixed_hparams.get("hidden_dim2")
        combos = [(int(fixed_hparams["hidden_dim1"]),
                   None if _h2 is None else int(_h2),
                   int(fixed_hparams["max_iter"]))]
    else:
        g = GRIDS["ann"]
        combos = list(product(g["hidden_dim1"], g["hidden_dim2"], g["max_iter"]))

    if arch == "shallow":
        # One hidden layer, so hidden_dim2 is not a parameter. Collapse that axis
        # and drop the duplicates it leaves, or a searching protocol would train
        # the same network three times over.
        seen, collapsed = set(), []
        for h1, _h2, mi in combos:
            if (h1, mi) not in seen:
                seen.add((h1, mi))
                collapsed.append((h1, None, mi))
        combos = collapsed

    if skip_selection_fit and not (fixed_hparams and refit_on_trainval):
        raise ValueError(
            "skip_selection_fit needs both fixed_hparams and refit_on_trainval: "
            "without a fixed configuration there is nothing to train, and without "
            "the refit no model would be produced at all.")

    best = None
    trace: List[dict] = []

    if skip_selection_fit:
        # Nothing to select between. Record the configuration and let the refit
        # below train the one and only model, on the full training split.
        h1, h2, max_iter = combos[0]
        best = {"score": None, "h1": h1, "h2": h2, "max_iter": max_iter,
                "metrics": None, "dump": None}

    for h1, h2, max_iter in ([] if skip_selection_fit else combos):
        torch.manual_seed(seed)
        np.random.seed(seed)
        metrics, dump = eval_ANN(
            fold=fold, train_feats=trf, train_labels=trl,
            valid_feats=vaf, valid_labels=val, test_feats=tef, test_labels=tel,
            input_dim=input_dim, hidden_dim=h1, hidden_dim2=h2,
            max_iter=max_iter, verbose=verbose, dropout=dropout,
        )
        score = metrics.get("val_ann_macro_f1")
        if score is None:
            raise RuntimeError(
                "eval_ANN did not return validation metrics. ANN hyperparameters "
                "must not be selected on the test set (work order Task 1.1)."
            )
        trace.append({"hidden_dim1": h1, "hidden_dim2": h2, "max_iter": max_iter,
                      "val_macro_f1": float(score),
                      "test_macro_f1": float(metrics["ann_macro_f1"])})
        if best is None or score > best["score"]:
            best = {"score": float(score), "h1": h1, "h2": h2, "max_iter": max_iter,
                    "metrics": metrics, "dump": dump}

    selection = {
        "selected": {"hidden_dim1": best["h1"], "hidden_dim2": best["h2"],
                     "max_iter": best["max_iter"]},
        "selection_metric": None if skip_selection_fit else "val_ann_macro_f1",
        "selection_value": best["score"],
        "grid_trace": trace,
        "n_grid_points": 0 if skip_selection_fit else len(combos),
        "hparam_source": (
            "fixed by caller; no selection fit, no slides held back for tuning"
            if skip_selection_fit else
            "fixed by caller (no search)" if fixed_hparams else
            f"searched {len(combos)} points on val_ann_macro_f1"),
        "refit_on_trainval": bool(refit_on_trainval),
        "selection_fit_skipped": bool(skip_selection_fit),
        "arch": arch,
        "n_hidden_layers": 1 if arch == "shallow" else 2,
        "dropout": dropout,   # None means the per-depth default was used
    }

    if refit_on_trainval:
        # Retrain the chosen configuration on train+validation. combine_trainval
        # concatenates the two inside ANNBinaryClassifier.fit and disables early
        # stopping, so the run goes the full max_iter on all the training data -
        # the same deal lin/knn/proto/rf get.
        torch.manual_seed(seed)
        np.random.seed(seed)
        metrics, dump = eval_ANN(
            fold=fold, train_feats=trf, train_labels=trl,
            valid_feats=vaf, valid_labels=val, test_feats=tef, test_labels=tel,
            input_dim=input_dim, hidden_dim=best["h1"], hidden_dim2=best["h2"],
            max_iter=best["max_iter"], combine_trainval=True, verbose=verbose,
            dropout=dropout,
        )
        selection["refit_n_train"] = int(trf.shape[0] + vaf.shape[0])
        # There is no "before" when the selection fit was skipped - this IS the
        # only model that was ever trained.
        selection["test_macro_f1_before_refit"] = (
            None if best["metrics"] is None else float(best["metrics"]["ann_macro_f1"]))
        selection["test_macro_f1_after_refit"] = float(metrics["ann_macro_f1"])
        best = {**best, "metrics": metrics, "dump": dump}

    if model_save_path is not None:
        save_ann_checkpoint(
            model_save_path, fold, best["dump"]["classifier"],
            selection_metric="val_ann_macro_f1", selection_value=best["score"],
        )

    if verbose:
        print(f"  Fold {fold} | ANN h1={best['h1']} h2={best['h2']} "
              f"max_iter={best['max_iter']} | {selection['hparam_source']}"
              + (f" | refit on {selection['refit_n_train']}" if refit_on_trainval else ""))

    # The model object is not JSON-serialisable and is not needed downstream.
    dump = {k: v for k, v in best["dump"].items() if k != "classifier"}
    return best["metrics"], dump, selection
