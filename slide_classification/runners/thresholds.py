"""Decision-threshold policy (work order section 1b).

The problem: on PAIP-EV logistic regression produced ``[[56, 0], [15, 2]]`` -
non-MSI-H for almost every slide - despite AUROC 0.862. The ranking transferred;
the operating point did not. A TCGA-calibrated cut-point applied to a cohort with
different prevalence collapses balanced accuracy. Left unaddressed, a reviewer
reads it as the method failing to generalise.

Policy:

  1. AUROC is primary everywhere - it is threshold-free.
  2. Internal experiments (TCGA-CV, PAIP-IV, SurGen-CV) report at 0.5.
  3. External experiments (PAIP-EV, SurGen-EV) report at tau_TCGA, fixed on TCGA
     and applied unchanged. A threshold is never tuned on PAIP or SurGen - that
     would silently convert external validation into internal validation.
  4. tau_TCGA = argmax_tau (TPR(tau) - FPR(tau)) - Youden's J - over the pooled
     **out-of-fold** TCGA-CV probabilities, so each of the 413 slides contributes
     one prediction made by a model that did not see it.

The random forest's pre-existing hardcoded ``prediction_threshold = 0.3`` is
replaced by tau_TCGA for external experiments (owner ruling); ``class_weight``
is unchanged, and ``RF @ 0.3`` is still recorded in the audit CSV so the old
behaviour stays reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from config import paths as P

THRESHOLDS_PATH = P.SLIDE_CLS_ROOT / "thresholds_TCGA.json"


def youden_threshold(targets, probs_pos) -> Dict[str, float]:
    """Youden's J optimal cut-point and the operating point it sits at."""
    targets = np.asarray(targets).astype(int)
    probs_pos = np.asarray(probs_pos, dtype=float)
    if len(np.unique(targets)) < 2:
        raise ValueError("Need both classes present to fit a threshold")
    fpr, tpr, thr = roc_curve(targets, probs_pos)
    j = tpr - fpr
    k = int(np.argmax(j))
    tau = float(thr[k])
    # roc_curve's first threshold is +inf by construction; clamp to a usable value.
    if not np.isfinite(tau):
        tau = float(np.nextafter(probs_pos.max(), 1.0))
    return {"tau": tau, "youden_j": float(j[k]), "tpr": float(tpr[k]),
            "fpr": float(fpr[k]), "n": int(len(targets)),
            "n_pos": int(targets.sum())}


def thresholds_from_oof(oof: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """One tau per classifier, from a pooled out-of-fold prediction table."""
    out: Dict[str, Dict[str, float]] = {}
    for clf, grp in oof.groupby("classifier"):
        # Each slide must appear exactly once - that is what makes this
        # out-of-sample. Guard against a fold loop that double-counted.
        dupes = grp["WSI_ID"].duplicated().sum()
        if dupes:
            raise ValueError(
                f"{clf}: {dupes} slide(s) appear more than once in the out-of-fold "
                f"table; tau must be fitted on one prediction per slide."
            )
        out[clf] = youden_threshold(grp["target"].values, grp["prob_pos"].values)
    return out


def update_threshold_store(method: str, model: str, taus: Dict[str, Dict[str, float]],
                           task: str = "MSIH", path: Optional[Path] = None) -> Path:
    """Persist tau per (method, model, classifier) into ``thresholds_TCGA.json``."""
    path = Path(path) if path else THRESHOLDS_PATH
    store = {}
    if path.exists():
        try:
            store = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[WARN] {path.name} unreadable; rewriting")
    store.setdefault("_meta", {}).update(P.run_stamp(
        policy="Youden J on pooled out-of-fold TCGA-CV probabilities",
        source_experiment="TCGA-CV",
    ))
    for clf, info in taus.items():
        store.setdefault(f"{method}|{model}|{task}", {})[clf] = info
    path.write_text(json.dumps(store, indent=2), encoding="utf-8")
    return path


def load_tau(method: str, model: str, classifier: str, task: str = "MSIH",
             path: Optional[Path] = None) -> float:
    """Look up tau_TCGA. Raises if absent - never silently fall back to 0.5."""
    path = Path(path) if path else THRESHOLDS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run TCGA-CV first - tau_TCGA is derived from its "
            f"out-of-fold predictions (work order section 1b)."
        )
    store = json.loads(path.read_text(encoding="utf-8"))
    key = f"{method}|{model}|{task}"
    try:
        return float(store[key][classifier]["tau"])
    except KeyError as exc:
        raise KeyError(
            f"No tau_TCGA for {key} / {classifier}. Available: "
            f"{sorted(store.get(key, {}))}"
        ) from exc


def apply_threshold(probs_pos, tau: float) -> np.ndarray:
    """Binarise positive-class probabilities at a fixed threshold."""
    return (np.asarray(probs_pos, dtype=float) >= tau).astype(int)
