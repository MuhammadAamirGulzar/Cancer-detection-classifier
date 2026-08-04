"""Uniform fit / save / load / predict across the five classifiers.

Needed by Task 3.0: external validation applies a *trained* model to a cohort it
has never seen, so training and inference must agree exactly on preprocessing.
They did not before - each ``eval_*`` function applied its own normalisation
inline, and the external-validation script had to re-derive it by hand (and got
the ANN architecture wrong, which is why Task 1.5 retired it).

Preprocessing, per classifier - this is the part that must not drift:

  lin    raw features                      (LogisticRegression on raw input)
  rf     raw features
  ann    raw features, model emits logits, softmax applied at predict time
  knn    L2-normalised features            (normalize_feats=True at train time)
  proto  L2-normalised features, prototypes are class means of normalised train

Getting ``knn``/``proto`` normalisation wrong at inference is silent: it produces
plausible-looking but meaningless probabilities.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from torch.nn.functional import normalize

from eval_patch_features.ann import (
    ANNBinaryClassifier, save_ann_checkpoint, load_ann_checkpoint,
)

#: Classifiers whose features must be L2-normalised at both train and predict.
NORMALISED = {"knn", "proto"}

_FILENAMES = {
    "lin": "logistic_regression.pkl",
    "knn": "knn_model.pkl",
    "proto": "protonet_model.pkl",
    "rf": "random_forest.pkl",
}


def _prep(kind: str, feats: torch.Tensor) -> torch.Tensor:
    return normalize(feats, dim=-1, p=2) if kind in NORMALISED else feats


# --------------------------------------------------------------------------
# Fitting on a full cohort (no held-out fold) - Task 3.0a
# --------------------------------------------------------------------------


def fit_full(
    kind: str,
    feats: torch.Tensor,
    labels: torch.Tensor,
    hparams: Dict[str, Any],
    seed: int = 42,
    val: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    verbose: bool = False,
):
    """Train one classifier on an entire cohort.

    ``val`` is required for the ANN only, which needs a held-out set for early
    stopping (Task 3.0a step 2 carves a stratified 15% for exactly this). The
    other four train on every row.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = _prep(kind, feats)

    if kind == "lin":
        clf = LogisticRegression(C=hparams.get("C", 10),
                                 max_iter=hparams.get("max_iter", 300),
                                 solver="lbfgs", random_state=seed, verbose=0)
        clf.fit(X.numpy(), labels.numpy())
        return clf

    if kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=hparams.get("n_estimators", 500),
            max_depth=hparams.get("max_depth", None),
            min_samples_split=hparams.get("min_samples_split", 5),
            min_samples_leaf=hparams.get("min_samples_leaf", 1),
            class_weight=hparams.get("class_weight", {0: 1, 1: 10}),
            bootstrap=True, oob_score=True, random_state=seed, n_jobs=-1,
        )
        clf.fit(X.numpy(), labels.numpy())
        return clf

    if kind == "knn":
        # Same nested GridSearchCV the fold runs use - cross-validated on the
        # training data, so no external information enters.
        # Threading backend for the same reason as eval_knn: loky workers crash
        # re-importing scipy on Windows under an active torch/CUDA context.
        grid = {"n_neighbors": [3, 5, 7, 10, 15],
                "metric": ["cosine", "euclidean", "minkowski"],
                "weights": ["uniform", "distance"]}
        gs = GridSearchCV(KNeighborsClassifier(), grid, n_jobs=-1,
                          scoring="balanced_accuracy")
        with joblib.parallel_backend("threading"):
            gs.fit(X.numpy(), labels.numpy())
        return gs.best_estimator_

    if kind == "proto":
        y = labels.numpy()
        class_ids = sorted(np.unique(y))
        prototypes = torch.stack([X[labels == c].mean(dim=0) for c in class_ids])
        return {"prototypes": prototypes, "labels_proto": torch.tensor(class_ids)}

    if kind == "ann":
        if val is None:
            raise ValueError(
                "The ANN needs a held-out set for early stopping. Task 3.0a "
                "carves a stratified 15% from the cohort for this - pass it as `val`."
            )
        clf = ANNBinaryClassifier(
            input_dim=int(feats.shape[1]),
            hidden_dim1=hparams.get("hidden_dim1", 128),
            hidden_dim2=hparams.get("hidden_dim2", 64),
            max_iter=hparams.get("max_iter", 500),
            verbose=verbose,
        )
        clf.fit(feats, labels, val[0], val[1], combine_trainval=False)
        return clf

    raise ValueError(f"Unknown classifier {kind!r}")


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def save_model(kind: str, model, out_dir: os.PathLike | str, fold: int = None,
               extra: Optional[dict] = None) -> Path:
    """Persist a trained model. ``fold=None`` means a full-cohort artifact."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "full_" if fold is None else f"fold{fold}_"

    if kind == "ann":
        path = Path(save_ann_checkpoint(
            str(out_dir), fold if fold is not None else "full", model,
            selection_metric=(extra or {}).get("selection_metric", ""),
            selection_value=(extra or {}).get("selection_value"),
        ))
    else:
        path = out_dir / f"{prefix}{_FILENAMES[kind]}"
        joblib.dump(model, path)

    if extra:
        (out_dir / f"{prefix}{kind}_meta.json").write_text(
            json.dumps(extra, indent=2, default=str), encoding="utf-8")
    return path


def load_model(kind: str, out_dir: os.PathLike | str, fold: int = None,
               input_dim: int = None):
    """Load a model saved by :func:`save_model` (or by the fold runners)."""
    out_dir = Path(out_dir)
    if kind == "ann":
        clf, _meta = load_ann_checkpoint(
            str(out_dir), fold if fold is not None else "full", input_dim=input_dim)
        return clf
    prefix = "full_" if fold is None else f"fold{fold}_"
    path = out_dir / f"{prefix}{_FILENAMES[kind]}"
    if not path.exists():
        raise FileNotFoundError(f"No saved {kind} model at {path}")
    return joblib.load(path)


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------


def predict_proba(kind: str, model, feats: torch.Tensor) -> np.ndarray:
    """``(N, 2)`` class probabilities, with each classifier's own preprocessing."""
    X = _prep(kind, feats)

    if kind in ("lin", "knn", "rf"):
        return np.asarray(model.predict_proba(X.numpy()), dtype=float)

    if kind == "proto":
        prototypes = model["prototypes"]
        d = (X[:, None] - prototypes[None, :]).norm(dim=-1, p=2)
        return torch.softmax(-d, dim=1).cpu().numpy().astype(float)

    if kind == "ann":
        return model.predict_proba(X).cpu().numpy().astype(float)

    raise ValueError(f"Unknown classifier {kind!r}")
