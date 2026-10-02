"""EXPERIMENT D - distance weighting, and why the search landed on k=3.

Part 1 - uniform vs distance weighting at the selected k.
    The grid searches ``weights in {uniform, distance}`` (knn.py:64) yet selects
    uniform in every one of the 20 configs. Distance weighting turns the vote
    into a continuous 1/d-weighted average, which should remove ties entirely.
    The question is whether it also moves AUROC or only its resolution.

    Note it cannot rescue an all-negative neighbourhood: if every one of the k
    neighbours is negative the weighted numerator is zero regardless of the
    weights, so the score is still exactly 0.

Part 2 - the model-selection generalisation gap.
    ``GridSearchCV(..., scoring='balanced_accuracy')`` optimises k on TCGA,
    where slides sit ~0.12 cosine from their nearest neighbour. Out of domain
    that distance is 0.39 (SurGen) to 0.50 (PAIP), so a neighbourhood tuned for
    a dense in-domain manifold is far too small. This measures in-domain
    out-of-fold BalAcc per k to show small k genuinely IS optimal on TCGA - the
    selection is behaving correctly and still generalises badly.

    The grid is also capped at 15 (knn.py:64), so k=25/35/50 were never
    reachable regardless.

Output: results/D_weighting.csv, results/D_k_selection.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from common import (K_GRID, RESULTS, TARGETS, cohort, distances, knn_artifact,
                    knn_probs, oracle_bacc, target_knn_probs, tcga_oof_knn,
                    valid_pairs, youden)

GRID_CAP = 15   # knn.py:64 -> n_neighbors in [3, 5, 7, 10, 15]


def part1_weighting() -> pd.DataFrame:
    rows = []
    for method, model in valid_pairs():
        try:
            art = knn_artifact(method, model)
        except FileNotFoundError:
            continue
        metric, k = art.metric, int(art.n_neighbors)
        for target in TARGETS:
            try:
                y = cohort(target, method, model).labels.numpy()
            except Exception:
                continue
            rec = dict(cohort=target, method=method, model=model,
                       k=k, metric=metric,
                       fitted_weights=art.weights, n=len(y))
            for w in ("uniform", "distance"):
                p = target_knn_probs(method, model, target, k, metric, weights=w)
                rec[f"AUROC_{w}"] = round(roc_auc_score(y, p), 4)
                rec[f"n_unique_{w}"] = int(len(np.unique(p)))
                rec[f"BalAccbest_{w}"] = round(oracle_bacc(y, p), 4)
                rec[f"n_zero_{w}"] = int((p == 0).sum())
            rec["dAUROC"] = round(rec["AUROC_distance"] - rec["AUROC_uniform"], 4)
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "D_weighting.csv", index=False)

    print("=== EXPERIMENT D part 1 - uniform vs distance weighting (at selected k) ===")
    agg = df.groupby("cohort").agg(
        n=("k", "size"),
        AUROC_uniform=("AUROC_uniform", "mean"),
        AUROC_distance=("AUROC_distance", "mean"),
        dAUROC=("dAUROC", "mean"),
        unique_uniform=("n_unique_uniform", "mean"),
        unique_distance=("n_unique_distance", "mean"),
        zeros_uniform=("n_zero_uniform", "mean"),
        zeros_distance=("n_zero_distance", "mean")).round(4)
    print(agg.to_string())
    print(f"\n  configs where distance weighting changes AUROC by >0.01: "
          f"{int((df.dAUROC.abs() > 0.01).sum())} of {len(df)}")
    print(f"  configs where it removes ALL ties (unique == n): "
          f"{int((df.n_unique_distance == df.n).sum())} of {len(df)}")
    print(f"  slides still scoring exactly 0 under distance weighting "
          f"(all-negative neighbourhood): mean {df.n_zero_distance.mean():.1f} "
          f"vs {df.n_zero_uniform.mean():.1f} uniform")
    return df


def part2_selection() -> pd.DataFrame:
    """In-domain (TCGA out-of-fold) BalAcc per k - what the search optimised."""
    rows = []
    for method, model in valid_pairs():
        try:
            art = knn_artifact(method, model)
        except FileNotFoundError:
            continue
        metric = art.metric
        for k in sorted({int(art.n_neighbors), *K_GRID}):
            y, p = tcga_oof_knn(method, model, k, metric)
            tau = youden(y, p)
            rows.append(dict(
                method=method, model=model, k=k, metric=metric,
                in_grid=(k <= GRID_CAP),
                tcga_oof_AUROC=round(roc_auc_score(y, p), 4),
                tcga_oof_BalAcc=round(balanced_accuracy_score(y, (p >= tau).astype(int)), 4),
                tcga_oof_BalAcc_best=round(oracle_bacc(y, p), 4),
                selected=(k == int(art.n_neighbors))))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "D_k_selection.csv", index=False)

    print("\n=== EXPERIMENT D part 2 - what the search saw on TCGA (in-domain) ===")
    agg = df.groupby("k").agg(
        in_grid=("in_grid", "first"),
        tcga_oof_AUROC=("tcga_oof_AUROC", "mean"),
        tcga_oof_BalAcc_best=("tcga_oof_BalAcc_best", "mean")).round(4)
    print(agg.to_string())
    best = df.loc[df.groupby(["method", "model"]).tcga_oof_BalAcc_best.idxmax()]
    print(f"\n  in-domain optimal k (by TCGA OOF BalAcc), counted over "
          f"{len(best)} configs:")
    print("   ", dict(best.k.value_counts().sort_index()))
    print(f"  of those, k <= grid cap ({GRID_CAP}): "
          f"{int((best.k <= GRID_CAP).sum())} of {len(best)}")
    return df


if __name__ == "__main__":
    part1_weighting()
    part2_selection()
    print(f"\nwrote {RESULTS/'D_weighting.csv'} and {RESULTS/'D_k_selection.csv'}")
