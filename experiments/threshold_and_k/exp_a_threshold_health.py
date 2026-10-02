"""EXPERIMENT A - dead AND saturated threshold detection.

The sweep in tools/knn_k_sweep.py only flags ``n_pos_pred == 0``. That misses
the mirror failure: a threshold below the whole score range flags *every* slide
positive, which also pins balanced accuracy near 0.50 while looking healthy.
Two known cases at k=50 on SurGen:

    Caption_based/ConchV1   tau=0.0953, p_min=0.10  ->  622/622 positive
    Tissue_Type/Conch1_5                            ->  619/622 positive

This reports both counts separately, for every config, every k, both cohorts,
at the stale (k=3-derived) tau. Read-only.

Output: results/A_threshold_health.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from common import (K_GRID, RESULTS, TARGETS, cohort, health, knn_artifact,
                    load_tau, oracle_bacc, target_knn_probs, valid_pairs)


def main() -> pd.DataFrame:
    rows = []
    for method, model in valid_pairs():
        try:
            art = knn_artifact(method, model)
        except FileNotFoundError:
            continue
        metric, k_sel = art.metric, int(art.n_neighbors)
        tau_stale = load_tau(method, model, "knn", "MSIH")

        for target in TARGETS:
            try:
                coh = cohort(target, method, model)
            except Exception:
                continue
            y = coh.labels.numpy()
            n = len(y)
            for k in sorted({k_sel, *K_GRID}):
                p = target_knn_probs(method, model, target, k, metric)
                pred = (p >= tau_stale).astype(int)
                npp = int(pred.sum())
                rows.append(dict(
                    cohort=target, method=method, model=model, k=k,
                    selected_k=(k == k_sel), metric=metric,
                    n=n, n_pos_pred=npp, frac_pos_pred=round(npp / n, 4),
                    status=health(npp, n),
                    tau_stale=round(tau_stale, 6),
                    p_min=round(float(p.min()), 6), p_max=round(float(p.max()), 6),
                    n_unique=int(len(np.unique(p))),
                    AUROC=round(roc_auc_score(y, p), 4),
                    BalAcc_at_tau=round(balanced_accuracy_score(y, pred), 4),
                    BalAcc_best=round(oracle_bacc(y, p), 4),
                ))
    df = pd.DataFrame(rows)
    out = RESULTS / "A_threshold_health.csv"
    df.to_csv(out, index=False)

    print("=== EXPERIMENT A - threshold health at the stale tau ===")
    print(f"{len(df)} rows over {df.groupby(['cohort','method','model']).ngroups} configs\n")
    tab = (df.groupby(["cohort", "k"]).status
             .value_counts().unstack(fill_value=0)
             .reindex(columns=["ok", "dead", "near_dead", "near_saturated", "saturated"], fill_value=0))
    print(tab.to_string())
    print("\n--- saturated / near-saturated (all missed by the old detector) ---")
    sat = df[df.status.isin(["saturated", "near_saturated"])]
    if len(sat):
        print(sat[["cohort", "method", "model", "k", "n_pos_pred", "n",
                   "tau_stale", "p_min", "BalAcc_at_tau", "AUROC"]]
              .to_string(index=False))
    else:
        print("none")
    print(f"\nwrote {out}")
    return df


if __name__ == "__main__":
    main()
