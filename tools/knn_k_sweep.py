"""Re-score a saved KNN model at different values of k. Read-only.

Why this exists
---------------
The KNN artifacts were selected by GridSearchCV on TCGA, which usually lands on
k=3. On an external cohort that produces a score with only 2-4 distinct values
({0, 1/3, 2/3, 1}), which crushes AUROC toward 0.5 - and where the score takes
exactly two values, AUROC and balanced accuracy become numerically identical,
because ROC then has a single interior vertex and its area is (sens + spec) / 2.

This script re-scores the **same saved model** at other k, using the training
matrix already embedded in the artifact. Nothing is refitted and nothing is
written back, so it is safe to run against production artifacts.

Two numbers are reported per k, and they move in opposite directions:

  AUROC    ranking quality - improves as k grows and the score gets finer
  BalAcc   measured at tau_TCGA, which was fitted at the ORIGINAL k. Raising k
           shifts the score scale toward the training prevalence (~0.15), so a
           tau sitting at a k=3 quantisation level (0.333, 0.286, ...) can end
           up above the whole distribution and fire on nothing.

A BalAcc of exactly 0.5000 with `n_pos_pred = 0` means the threshold is dead,
not that the model is uninformative - read AUROC in that case.

Caveat: picking k by looking at the target-cohort AUROC printed here is
selection on the test set. Use this to diagnose, not to choose a k to report.
For a reportable number, tau must also be refitted on TCGA out-of-fold
probabilities at the new k.

Run:
  python tools/knn_k_sweep.py
  python tools/knn_k_sweep.py --cohort surgen --ks 3,10,25,50,100
  python tools/knn_k_sweep.py --method Averaging --model UNI2 --out sweep.csv
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics.pairwise import (
    cosine_distances, euclidean_distances, manhattan_distances,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SLIDE_CLS = REPO_ROOT / "slide_classification"
sys.path.insert(0, str(SLIDE_CLS))
warnings.filterwarnings("ignore")

import data_layer as dl  # noqa: E402
from config import paths as P  # noqa: E402
from runners.full_trainer import artifact_dir  # noqa: E402
from runners.model_io import load_model, _prep  # noqa: E402
from runners.thresholds import load_tau  # noqa: E402

METHODS = ["Averaging", "Caption_based_aggregation",
           "Caption_based_aggregation_15_classes", "Tissue_Type_Clustering"]
MODELS = ["UNI2", "H-Optimus-1", "Virchow2", "Conch1_5", "ConchV1"]

#: The artifact records which metric GridSearchCV chose; re-score with the same
#: one, or the neighbourhoods are not the ones the model actually uses.
DISTANCE = {"cosine": cosine_distances,
            "euclidean": euclidean_distances,
            "manhattan": manhattan_distances}


def sweep_one(cohort: str, method: str, model: str, ks, seed: int = 42):
    """One (method, model): re-score the saved KNN at each k. Returns rows."""
    clf = load_model("knn", artifact_dir(method, model, seed, "MSIH"), fold=None)
    coh = dl.load_cohort(cohort, method, model, task="MSIH", verbose=False)
    y = coh.labels.numpy()
    tau = load_tau(method, model, "knn", "MSIH")

    X_train, y_train = clf._fit_X, clf._y      # embedded TCGA reference set
    X_test = _prep("knn", coh.feats).numpy()   # same L2 norm as training

    dist = DISTANCE.get(clf.metric, cosine_distances)
    order = np.argsort(dist(X_test, X_train), axis=1)

    rows = []
    for k in sorted({int(clf.n_neighbors), *ks}):
        if k > X_train.shape[0]:
            continue
        probs = y_train[order[:, :k]].mean(axis=1)   # uniform vote, as fitted
        pred = (probs >= tau).astype(int)
        fpr, tpr, _ = roc_curve(y, probs)
        rows.append(dict(
            cohort=cohort, method=method, model=model, k=k,
            selected=(k == int(clf.n_neighbors)), metric=clf.metric,
            n_unique=len(np.unique(probs)),
            AUROC=roc_auc_score(y, probs),
            BalAcc_at_tau=balanced_accuracy_score(y, pred),
            BalAcc_best=float(np.max((tpr + (1 - fpr)) / 2)),
            tau=tau, n_pos_pred=int(pred.sum()),
            p_min=float(probs.min()), p_max=float(probs.max()),
        ))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cohort", default="paip,surgen",
                    help="comma list: paip, surgen, tcga")
    ap.add_argument("--ks", default="3,5,10,15,25,50",
                    help="k values to try; the model's own k is always included")
    ap.add_argument("--method", default=",".join(METHODS))
    ap.add_argument("--model", default=",".join(MODELS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None, help="optional CSV path")
    args = ap.parse_args()

    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    rows = []
    for cohort in [c.strip() for c in args.cohort.split(",") if c.strip()]:
        for method in [m.strip() for m in args.method.split(",") if m.strip()]:
            for model in [m.strip() for m in args.model.split(",") if m.strip()]:
                if not P.is_combination_valid(method, model):
                    continue
                try:
                    rows += sweep_one(cohort, method, model, ks, args.seed)
                except FileNotFoundError:
                    continue
                except Exception as exc:                      # keep the sweep going
                    print(f"  [skip] {cohort}/{method}/{model}: "
                          f"{type(exc).__name__}: {exc}")

    if not rows:
        raise SystemExit("nothing scored - check --method/--model names")

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    show = ["k", "n_unique", "AUROC", "BalAcc_at_tau", "BalAcc_best",
            "n_pos_pred", "p_min", "p_max"]
    for (cohort, method, model), g in df.groupby(["cohort", "method", "model"],
                                                 sort=False):
        sel = g[g.selected].k.iloc[0]
        print(f"\n=== {cohort} | {method} | {model} "
              f"(selected k={sel}, metric={g.metric.iloc[0]}, tau={g.tau.iloc[0]:.4f})")
        out = g[show].copy().round(4)
        out["k"] = [f"{k}*" if k == sel else str(k) for k in g.k]   # * = as fitted
        print(out.to_string(index=False))
        dead = g[g.n_pos_pred == 0].k.tolist()
        if dead:
            print(f"    threshold fires on nothing at k={dead} "
                  f"-> BalAcc 0.5000 is a dead threshold, read AUROC instead")

    print(f"\n=== mean over {df.groupby(['cohort','method','model']).ngroups} "
          f"config(s), by k ===")
    agg = df.groupby(["cohort", "k"]).agg(
        AUROC=("AUROC", "mean"), BalAcc_at_tau=("BalAcc_at_tau", "mean"),
        BalAcc_best=("BalAcc_best", "mean"), n_unique=("n_unique", "mean"),
        dead=("n_pos_pred", lambda s: int((s == 0).sum())), n=("AUROC", "size"))
    print(agg.round(4).to_string())

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
