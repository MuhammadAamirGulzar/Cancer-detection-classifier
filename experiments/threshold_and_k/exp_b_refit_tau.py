"""EXPERIMENT B - refit tau at each k, the column the sweep was missing.

The published tau was fitted on out-of-fold TCGA probabilities produced at the
GridSearchCV-selected k (usually 3). Re-scoring at another k changes the score
scale, so that tau no longer describes the distribution it is applied to:
PAIP Averaging/UNI2 has tau=0.3333 while p_max at k=50 is 0.20. Every
``BalAcc_at_tau`` in that sweep is therefore uninterpretable.

This recomputes tau properly, per k:

  1. 4-fold TCGA cross-validation at that k. For each fold the reference set is
     every slide outside it - exactly a fold model's KNN, since ``knn.py:33-35``
     merges train+validation. KNN is lazy, so this is exact, not approximate.
  2. Youden's J on the pooled out-of-fold probabilities - the same rule as
     runners/thresholds.py:58-72.
  3. Evaluate the target cohort at that refit tau, scoring with the full
     413-slide reference set to mirror the ``tcga_full`` EV variant.

No target-cohort labels enter the threshold at any point.

Refit thresholds are written to results/B_refit_thresholds.json - a NEW file.
thresholds_TCGA.json is opened read-only and never modified.

Output: results/B_refit_tau.csv, results/B_refit_thresholds.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from common import (K_GRID, RESULTS, TARGETS, cohort, health, knn_artifact,
                    load_tau, oracle_bacc, target_knn_probs, tcga_oof_knn,
                    valid_pairs, youden)


def main() -> pd.DataFrame:
    rows, store = [], {}
    for method, model in valid_pairs():
        try:
            art = knn_artifact(method, model)
        except FileNotFoundError:
            continue
        metric, k_sel = art.metric, int(art.n_neighbors)
        tau_stale = load_tau(method, model, "knn", "MSIH")
        key = f"{method}|{model}|MSIH"

        for k in sorted({k_sel, *K_GRID}):
            # --- 1+2: refit tau on TCGA out-of-fold probabilities at this k
            y_oof, p_oof = tcga_oof_knn(method, model, k, metric)
            tau_refit = youden(y_oof, p_oof)
            store.setdefault(key, {})[str(k)] = dict(
                tau=tau_refit, k=k, metric=metric,
                fitted_on="TCGA-CV out-of-fold, 4 folds",
                n=int(len(y_oof)), n_pos=int(y_oof.sum()),
                oof_auroc=round(float(roc_auc_score(y_oof, p_oof)), 4))

            # --- 3: evaluate every target at both thresholds
            for target in TARGETS:
                try:
                    coh = cohort(target, method, model)
                except Exception:
                    continue
                y = coh.labels.numpy()
                n = len(y)
                p = target_knn_probs(method, model, target, k, metric)
                pred_s = (p >= tau_stale).astype(int)
                pred_r = (p >= tau_refit).astype(int)
                rows.append(dict(
                    cohort=target, method=method, model=model, k=k,
                    selected_k=(k == k_sel), metric=metric, n=n,
                    n_unique=int(len(np.unique(p))),
                    AUROC=round(roc_auc_score(y, p), 4),
                    BalAcc_stale=round(balanced_accuracy_score(y, pred_s), 4),
                    BalAcc_refit=round(balanced_accuracy_score(y, pred_r), 4),
                    BalAcc_best=round(oracle_bacc(y, p), 4),
                    tau_stale=round(tau_stale, 6), tau_refit=round(tau_refit, 6),
                    n_pos_pred_stale=int(pred_s.sum()),
                    n_pos_pred_refit=int(pred_r.sum()),
                    status_stale=health(int(pred_s.sum()), n),
                    status_refit=health(int(pred_r.sum()), n),
                ))

    df = pd.DataFrame(rows)
    df["gap_stale"] = (df.BalAcc_best - df.BalAcc_stale).round(4)
    df["gap_refit"] = (df.BalAcc_best - df.BalAcc_refit).round(4)
    df["recovered"] = (df.BalAcc_refit - df.BalAcc_stale).round(4)

    out = RESULTS / "B_refit_tau.csv"
    df.to_csv(out, index=False)
    tj = RESULTS / "B_refit_thresholds.json"
    tj.write_text(json.dumps({
        "_meta": {"policy": "Youden J on pooled TCGA-CV out-of-fold "
                            "probabilities, recomputed per k",
                  "source_experiment": "TCGA-CV", "classifier": "knn",
                  "note": "experimental; thresholds_TCGA.json is unmodified"},
        "thresholds": store}, indent=2), encoding="utf-8")

    print("=== EXPERIMENT B - stale vs refit tau ===")
    agg = df.groupby(["cohort", "k"]).agg(
        AUROC=("AUROC", "mean"),
        BalAcc_stale=("BalAcc_stale", "mean"),
        BalAcc_refit=("BalAcc_refit", "mean"),
        BalAcc_best=("BalAcc_best", "mean"),
        gap_stale=("gap_stale", "mean"), gap_refit=("gap_refit", "mean"),
        recovered=("recovered", "mean"),
        tau_stale=("tau_stale", "mean"), tau_refit=("tau_refit", "mean"),
        n_unique=("n_unique", "mean")).round(4)
    print(agg.to_string())

    print("\n--- how many configs remain dead/saturated after refitting? ---")
    for col, lab in [("status_stale", "stale tau"), ("status_refit", "refit tau")]:
        t = (df.groupby(["cohort", "k"])[col].value_counts().unstack(fill_value=0))
        print(f"\n[{lab}]")
        print(t.to_string())

    print(f"\nwrote {out}")
    print(f"wrote {tj}  (NEW file; thresholds_TCGA.json untouched)")
    return df


if __name__ == "__main__":
    main()
