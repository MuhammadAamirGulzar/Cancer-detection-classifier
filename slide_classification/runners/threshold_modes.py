"""Threshold modes for external validation, and the validated KNN k override.

Promoted from ``experiments/corrected_ev/`` after experiments A-D. This module
is the single implementation - ``runners.ev_runner`` and the experiment script
both import it, so the promoted path and the validated reference cannot drift.

Three modes
-----------
``frozen`` (default, unchanged behaviour)
    tau_TCGA, fitted by Youden's J on pooled out-of-fold TCGA-CV probabilities
    and applied unmodified. This is the strict zero-shot number and stays the
    default so every published result reproduces exactly.

``corrected``
    KNN     - score at ``KNN_K_VALIDATED`` neighbours and threshold at a tau
              refitted on TCGA out-of-fold probabilities *at that k*. A stale
              k=3 tau applied to k=35 scores is meaningless: PAIP
              Averaging/UNI2 has tau=0.3333 while p_max at k=50 is 0.20.
    others  - rate-matched (quantile) threshold: r is the fraction of TCGA
              out-of-fold slides tau_TCGA flags positive; the target is cut at
              its own (1 - r) quantile.

``promoted``
    The corrected scheme applied to ``PROMOTED_CLASSIFIERS`` **only**, with its
    values taking over the headline metric fields. The remaining heads are not
    merely left at the frozen operating point - they are not evaluated under the
    corrected scheme at all, so no ``*_corrected`` column is written for them and
    a reader cannot mistake a leftover column for a live number.

    This mode therefore changes published figures by design, which ``corrected``
    never does. The frozen values it displaces are kept under ``*_frozen`` in the
    same record, so nothing is lost and the change is legible in the file itself.

    Two threshold schemes then coexist in one table. Anything rendering that
    table has to say which row uses which - ``ev_runner`` writes a per-row
    ``Threshold_scheme`` for exactly that purpose.

Why KNN is excluded from the quantile rule
------------------------------------------
Experiment C: the quantile made KNN *worse* (PAIP 0.5959 -> 0.5746) and
saturated 9 configs. A KNN score at small k takes 2-4 distinct values, so there
is no cut point that achieves an arbitrary rate - the mass sits in tied blocks.

Why k=35
--------
Two independent TCGA-only justifications, neither touching target data:
  * it maximised TCGA out-of-fold balanced accuracy (0.6971, best of every k
    tested - experiment D2);
  * prevalence rule: 35 x 0.145 ~ 5 expected positive neighbours, the smallest
    neighbourhood that can resolve a positive minority.
``weights='uniform'`` is kept - distance weighting bought only +0.006 AUROC
(experiment D1) and does not remove all ties. The per-config metric selected by
the original GridSearchCV (cosine / manhattan) is read from the saved artifact.

No target labels
----------------
k, tau and r are all derived from TCGA. The quantile reads target *scores* but
never target *labels*; target labels enter only the reported metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics.pairwise import (
    cosine_distances, euclidean_distances, manhattan_distances,
)

MODES = ("frozen", "corrected", "promoted")
DEFAULT_MODE = "frozen"

#: Under ``promoted`` these classifiers - and only these - are scored and
#: thresholded by the corrected scheme, and those values become the **headline**
#: metrics. Every other head keeps the frozen tau untouched and gets no
#: corrected columns at all. See ``runners.ev_runner`` for the promotion itself.
PROMOTED_CLASSIFIERS = ("knn", "rf")

#: Validated neighbourhood size; see module docstring for the two derivations.
KNN_K_VALIDATED = 35
KNN_WEIGHTS = "uniform"

#: Taus fitted once at k=35 on TCGA-CV out-of-fold probabilities and validated in
#: ``experiments/corrected_ev``. Byte-identical copy of that experiment's file,
#: promoted here so the pipeline does not read out of an experiments folder.
#: Read, never recomputed - see :func:`knn_tau_from_store`.
KNN_K35_THRESHOLDS_PATH = (
    Path(__file__).resolve().parents[1] / "knn_k35_thresholds.json")

#: Classifiers that take the rate-matched threshold under ``corrected``.
QUANTILE_CLASSIFIERS = ("lin", "ann", "proto", "rf")

#: A threshold one slide short of an extreme fails identically to one at it.
NEAR_BAND = 0.01

_DISTANCE = {"cosine": cosine_distances,
             "euclidean": euclidean_distances,
             "manhattan": manhattan_distances}


def health(n_pos_pred: int, n: int) -> str:
    """``ok`` / ``dead`` / ``saturated`` (+ ``near_`` variants).

    A dead or saturated operating point yields BalAcc ~0.50 while looking like
    a working threshold; it must be reported as such, not as a bare number.
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


def youden(y: np.ndarray, p: np.ndarray) -> float:
    """Youden's J cut-point - same rule as runners.thresholds."""
    fpr, tpr, thr = roc_curve(y, p)
    tau = float(thr[int(np.argmax(tpr - fpr))])
    if not np.isfinite(tau):
        tau = float(np.nextafter(float(np.max(p)), 1.0))
    return tau


def quantile_threshold(tcga_oof_probs: Sequence[float], tau_tcga: float,
                       target_probs: Sequence[float]) -> float:
    """Rate-matched cut point. Uses target scores, never target labels."""
    o = np.asarray(tcga_oof_probs, dtype=float)
    t = np.asarray(target_probs, dtype=float)
    r = float((o >= tau_tcga).mean())
    if not 0.0 < r < 1.0:
        return float(tau_tcga)
    return float(np.quantile(t, 1.0 - r))


# --------------------------------------------------------------------------
# KNN at an overridden k, using the saved artifact's reference set and metric
# --------------------------------------------------------------------------


def _order(knn_model, feats_prepped: np.ndarray) -> np.ndarray:
    fn = _DISTANCE.get(getattr(knn_model, "metric", "cosine"), cosine_distances)
    return np.argsort(fn(feats_prepped, knn_model._fit_X), axis=1)


def knn_probs_at_k(knn_model, feats_prepped: np.ndarray, k: int) -> np.ndarray:
    """Positive-class fraction over the k nearest reference points.

    KNN is lazy: the saved estimator carries the whole training matrix, so this
    is the same model read at a different neighbourhood size, not a refit.
    """
    y_ref = knn_model._y
    k = min(int(k), knn_model._fit_X.shape[0])
    return y_ref[_order(knn_model, feats_prepped)[:, :k]].mean(axis=1)


def knn_oof_probs_at_k(knn_model, feats_prepped: np.ndarray,
                       folds: np.ndarray, k: int) -> np.ndarray:
    """TCGA out-of-fold probabilities at ``k``.

    For each fold the reference set is every slide outside it - exactly a fold
    model's KNN, since ``eval_knn`` merges train+validation (knn.py:33-35).
    """
    y_ref = np.asarray(knn_model._y)
    fn = _DISTANCE.get(getattr(knn_model, "metric", "cosine"), cosine_distances)
    D = fn(feats_prepped, knn_model._fit_X)
    np.fill_diagonal(D, np.inf)                  # never your own neighbour
    out = np.full(len(y_ref), np.nan)
    for f in sorted({int(v) for v in folds if v > 0}):
        te = np.where(folds == f)[0]
        ref = np.where(folds != f)[0]
        sub = D[np.ix_(te, ref)]
        kk = min(int(k), len(ref))
        out[te] = y_ref[ref][np.argsort(sub, axis=1)[:, :kk]].mean(axis=1)
    return out


def knn_tau_at_k(knn_model, feats_prepped: np.ndarray, labels: np.ndarray,
                 folds: np.ndarray, k: int) -> float:
    """Refit tau: Youden's J on TCGA out-of-fold probabilities at ``k``."""
    p = knn_oof_probs_at_k(knn_model, feats_prepped, folds, k)
    ok = ~np.isnan(p)
    return youden(np.asarray(labels)[ok], p[ok])


def corrected_threshold(kind: str, tau_frozen: float,
                        tcga_oof_probs: Optional[Sequence[float]],
                        target_probs: Sequence[float],
                        knn_tau: Optional[float] = None) -> float:
    """The ``corrected``-mode cut point for one classifier."""
    if kind == "knn":
        if knn_tau is None:
            raise ValueError("corrected mode needs a refit tau for knn")
        return float(knn_tau)
    if kind in QUANTILE_CLASSIFIERS and tcga_oof_probs is not None:
        return quantile_threshold(tcga_oof_probs, tau_frozen, target_probs)
    return float(tau_frozen)


def knn_tau_from_store(method: str, model: str, task: str = "MSIH",
                       path: Optional[Path] = None) -> Tuple[Optional[float], str]:
    """The stored k=35 tau for one combination, as ``(tau, source)``.

    These values were fitted once, by Youden's J on TCGA-CV out-of-fold
    probabilities at k=35, and validated in ``experiments/corrected_ev``.
    Re-deriving them on every run would burn the whole TCGA distance matrix per
    combination to land on the same number, and any drift in the surrounding
    code would silently move a *published* threshold. So they are read, not
    recomputed - the file is the record.

    ``thresholds_TCGA.json`` is a different file and is never touched: the frozen
    taus it holds must keep reproducing the frozen numbers.

    Returns ``(None, 'absent')`` when the combination has no stored entry, which
    is not an error - the caller decides whether to derive one and must say so.
    """
    p = Path(path) if path else KNN_K35_THRESHOLDS_PATH
    if not p.exists():
        return None, "absent"
    store = json.loads(p.read_text(encoding="utf-8")).get("thresholds", {})
    rec = store.get(f"{method}|{model}|{task}")
    if rec is None:
        return None, "absent"
    k = int(rec.get("k", KNN_K_VALIDATED))
    if k != KNN_K_VALIDATED:
        raise ValueError(
            f"{p.name} holds tau for k={k} but the pipeline runs k="
            f"{KNN_K_VALIDATED}. A tau fitted at one k does not transfer to "
            f"another - the score scale changes with the neighbourhood size.")
    return float(rec["tau"]), p.name


def load_tcga_oof(method: str, model: str, task: str = "MSIH") -> pd.DataFrame:
    """Published TCGA-CV out-of-fold table; empty frame when absent."""
    from config import paths as P
    p = (P.results_root("TCGA-CV", method, model, task) / "Output"
         / "oof_predictions_default.csv")
    return pd.read_csv(p) if p.exists() else pd.DataFrame()
