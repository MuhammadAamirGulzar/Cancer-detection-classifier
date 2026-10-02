"""Shared read-only helpers for the threshold/k experiments.

Nothing in this package writes outside ``experiments/threshold_and_k/``.
Existing artifacts, ``thresholds_TCGA.json`` and every results tree are opened
read-only, so all published numbers stay reproducible.

Why the KNN re-scoring is exact
-------------------------------
KNN is a lazy learner: the "model" is the training matrix plus (k, metric,
weights). The saved ``KNeighborsClassifier`` carries the full TCGA matrix in
``_fit_X`` / ``_y``, so re-scoring at another k needs no retraining - it is the
same estimator read at a different neighbourhood size.

The same property makes the 4-fold out-of-fold recomputation in Experiment B
exact rather than an approximation: excluding a test fold from the reference set
*is* training on the other folds. This matches the pipeline, where ``eval_knn``
merges train+validation (``knn.py:33-35``), so a fold model's reference set is
every slide outside its test fold.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

EXP_DIR = Path(__file__).resolve().parent
RESULTS = EXP_DIR / "results"
REPO_ROOT = EXP_DIR.parents[1]
SLIDE_CLS = REPO_ROOT / "slide_classification"

sys.path.insert(0, str(SLIDE_CLS))
warnings.filterwarnings("ignore")

import data_layer as dl                                    # noqa: E402
from config import paths as P                              # noqa: E402
from runners.full_trainer import artifact_dir              # noqa: E402
from runners.model_io import load_model, predict_proba, _prep  # noqa: E402
from runners.thresholds import load_tau                    # noqa: E402

from sklearn.metrics import roc_curve                      # noqa: E402
from sklearn.metrics.pairwise import (                     # noqa: E402
    cosine_distances, euclidean_distances, manhattan_distances,
)

METHODS = ["Averaging", "Caption_based_aggregation",
           "Caption_based_aggregation_15_classes", "Tissue_Type_Clustering"]
MODELS = ["UNI2", "H-Optimus-1", "Virchow2", "Conch1_5", "ConchV1"]
CLASSIFIERS = ["lin", "ann", "knn", "proto", "rf"]
TARGETS = ["paip", "surgen"]

#: k=35 comes from the TRAINING prevalence only: 35 x 0.145 ~ 5 expected
#: positive neighbours, the smallest neighbourhood that can resolve a positive
#: minority without relying on any target-cohort information.
K_GRID = [3, 5, 10, 15, 25, 35, 50]

DISTANCE = {"cosine": cosine_distances,
            "euclidean": euclidean_distances,
            "manhattan": manhattan_distances}

RESULTS.mkdir(parents=True, exist_ok=True)

_cache: Dict[tuple, object] = {}


# ---------------------------------------------------------------- basic pieces

def youden(y: np.ndarray, p: np.ndarray) -> float:
    """Youden's J cut-point - the same rule as runners/thresholds.py."""
    fpr, tpr, thr = roc_curve(y, p)
    tau = float(thr[int(np.argmax(tpr - fpr))])
    if not np.isfinite(tau):                      # roc_curve's leading +inf
        tau = float(np.nextafter(p.max(), 1.0))
    return tau


def oracle_bacc(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max((tpr + (1 - fpr)) / 2))


#: A threshold one slide short of the extreme fails identically to one at it -
#: surgen/Tissue_Type/Conch1_5 flags 619 of 622 and is not "healthy". The band
#: is 1% of the cohort, so it scales with N rather than hard-coding a count.
NEAR_BAND = 0.01


def health(n_pos_pred: int, n: int) -> str:
    """Classify the operating point.

    The original sweep only detected ``dead``. ``saturated`` is the mirror
    failure - every slide flagged positive - and produces the same BalAcc ~0.50
    while looking like a working threshold. The ``near_`` states catch the same
    pathology a slide or two short of the extreme.
    """
    edge = max(1, int(round(NEAR_BAND * n)))
    if n_pos_pred == 0:
        return "dead"
    if n_pos_pred == n:
        return "saturated"
    if n_pos_pred <= edge:
        return "near_dead"
    if n_pos_pred >= n - edge:
        return "near_saturated"
    return "ok"


# ---------------------------------------------------------------- data access

def knn_artifact(method: str, model: str, seed: int = 42):
    key = ("art", method, model, seed)
    if key not in _cache:
        _cache[key] = load_model("knn", artifact_dir(method, model, seed, "MSIH"),
                                 fold=None)
    return _cache[key]


def cohort(name: str, method: str, model: str):
    key = ("coh", name, method, model)
    if key not in _cache:
        _cache[key] = dl.load_cohort(name, method, model, task="MSIH", verbose=False)
    return _cache[key]


def distances(method: str, model: str, target: Optional[str], metric: str):
    """Distance matrix. ``target=None`` gives TCGA-vs-TCGA (for out-of-fold)."""
    key = ("D", method, model, target, metric)
    if key not in _cache:
        tc = cohort("tcga", method, model)
        Xtr = _prep("knn", tc.feats).numpy()
        fn = DISTANCE.get(metric, cosine_distances)
        Xte = Xtr if target is None else _prep("knn", cohort(target, method, model).feats).numpy()
        _cache[key] = fn(Xte, Xtr)
    return _cache[key]


def knn_probs(order: np.ndarray, y_ref: np.ndarray, k: int,
              weights: str = "uniform", D: Optional[np.ndarray] = None,
              rows: Optional[np.ndarray] = None) -> np.ndarray:
    """Positive-class fraction over the k nearest reference points."""
    idx = order[:, :k]
    if weights == "uniform":
        return y_ref[idx].mean(axis=1)
    # distance weighting: w = 1/d, with exact matches taking the row alone
    d = np.take_along_axis(D if rows is None else D[rows], idx, axis=1)
    out = np.empty(len(idx), dtype=float)
    for i in range(len(idx)):
        di, yi = d[i], y_ref[idx[i]]
        zero = di <= 0
        if zero.any():
            out[i] = yi[zero].mean()
        else:
            w = 1.0 / di
            out[i] = float((w * yi).sum() / w.sum())
    return out


def tcga_fold_map() -> Dict[str, int]:
    key = ("folds",)
    if key not in _cache:
        _cache[key] = dl.load_fold_map("tcga", "MSIH")
    return _cache[key]


def tcga_oof_knn(method: str, model: str, k: int, metric: str,
                 weights: str = "uniform") -> Tuple[np.ndarray, np.ndarray]:
    """Out-of-fold TCGA probabilities at neighbourhood size ``k``.

    For each fold, the reference set is every slide outside that fold - exactly
    what a fold model's KNN would hold (``knn.py:33-35`` merges train+val).
    """
    tc = cohort("tcga", method, model)
    y = tc.labels.numpy()
    fm = tcga_fold_map()
    folds = np.array([fm.get(i, -1) for i in tc.ids])
    D = distances(method, model, None, metric).copy()
    np.fill_diagonal(D, np.inf)                    # never be your own neighbour

    probs = np.full(len(y), np.nan)
    for f in sorted(set(folds[folds > 0])):
        te = np.where(folds == f)[0]
        ref = np.where(folds != f)[0]
        sub = D[np.ix_(te, ref)]
        order = np.argsort(sub, axis=1)
        kk = min(k, len(ref))
        if weights == "uniform":
            probs[te] = y[ref][order[:, :kk]].mean(axis=1)
        else:
            probs[te] = knn_probs(order, y[ref], kk, "distance", D=sub)
    ok = ~np.isnan(probs)
    return y[ok], probs[ok]


def target_knn_probs(method: str, model: str, target: str, k: int,
                     metric: str, weights: str = "uniform") -> np.ndarray:
    """Target-cohort scores using the FULL 413-slide TCGA reference set.

    Mirrors the ``tcga_full`` EV variant, which is what the results tables use.
    """
    tc = cohort("tcga", method, model)
    y_ref = tc.labels.numpy()
    D = distances(method, model, target, metric)
    order = np.argsort(D, axis=1)
    return knn_probs(order, y_ref, min(k, D.shape[1]), weights, D=D)


def valid_pairs() -> Iterable[Tuple[str, str]]:
    for m in METHODS:
        for mo in MODELS:
            if P.is_combination_valid(m, mo):
                yield m, mo
