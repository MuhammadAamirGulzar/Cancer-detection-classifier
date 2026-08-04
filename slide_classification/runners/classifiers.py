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
    "ann": {"hidden_dim1": [128, 256], "hidden_dim2": [64, 128], "max_iter": [500]},
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
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[dict]]:
    """Fit one classifier on one split. Returns ``(metrics, dump, selection)``.

    ``selection`` is ``None`` for single-point grids and, for the ANN, records
    which configuration was chosen and on what score.
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
                             model_save_path, seed, verbose)

    raise ValueError(f"Unsupported model type: {model_type!r}. Expected one of {MODEL_TYPES}")


def _run_ann_grid(fold, train, valid, test, input_dim, model_save_path, seed, verbose):
    """ANN grid search, selected on VALIDATION macro-F1.

    [CORRECTION 2026-08-04, work order Task 1.1] The selection criterion was
    ``eval_metrics['ann_macro_f1']``, which ``eval_ANN`` computes on the TEST
    set - the best of four configurations was chosen using test performance and
    then reported as a test result. It is now ``val_ann_macro_f1``.

    [CORRECTION 2026-08-04, work order Task 1.2] The checkpoint is written once,
    here, after selection - not inside the loop, where all four configurations
    used to overwrite the same file.
    """
    trf, trl = train
    vaf, val = valid
    tef, tel = test
    g = GRIDS["ann"]

    best = None
    trace: List[dict] = []

    for h1, h2, max_iter in product(g["hidden_dim1"], g["hidden_dim2"], g["max_iter"]):
        torch.manual_seed(seed)
        np.random.seed(seed)
        metrics, dump = eval_ANN(
            fold=fold, train_feats=trf, train_labels=trl,
            valid_feats=vaf, valid_labels=val, test_feats=tef, test_labels=tel,
            input_dim=input_dim, hidden_dim=h1, hidden_dim2=h2,
            max_iter=max_iter, verbose=verbose,
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
        "selection_metric": "val_ann_macro_f1",
        "selection_value": best["score"],
        "grid_trace": trace,
    }

    if model_save_path is not None:
        save_ann_checkpoint(
            model_save_path, fold, best["dump"]["classifier"],
            selection_metric="val_ann_macro_f1", selection_value=best["score"],
        )

    if verbose:
        print(f"  Fold {fold} | ANN selected h1={best['h1']} h2={best['h2']} "
              f"on val_macro_f1={best['score']:.4f}")

    # The model object is not JSON-serialisable and is not needed downstream.
    dump = {k: v for k, v in best["dump"].items() if k != "classifier"}
    return best["metrics"], dump, selection
