"""EXPERIMENT C - quantile threshold, all five classifiers.

Independent of k, and applies to every classifier rather than KNN alone.

  r      = the fraction of TCGA slides that tau_TCGA flags positive, measured on
           the pooled out-of-fold probabilities tau was fitted on.
  tau_q  = the (1 - r) quantile of the TARGET score distribution.

This transports the *operating rate* rather than the *score value*. It reads
target scores but never target labels, so it stays legitimate for external
validation - the same standing as any unsupervised domain adaptation.

Motivation: the failure is shared. RF, ProtoNet and KNN all produce scores on a
scale the TCGA-fitted cut-point no longer describes - ProtoNet spans just
0.489-0.500 on PAIP and is dead in 11 of 20 SurGen configs. If all three respond
to tau_q, that is one correction, not three separate patches.

Read-only: target scores come from the stored ``mean_probs`` of the ``tcga_full``
EV variant (the variant the results tables use), TCGA probabilities from the
published out-of-fold CSVs.

Output: results/C_quantile_threshold.csv
"""

from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from common import (CLASSIFIERS, P, RESULTS, cohort, health, oracle_bacc,
                    valid_pairs)

EV_DIR = {"paip": "TCGA_PAIP_EV_Results", "surgen": "SurGen_EV_Results"}


def tcga_oof(method: str, model: str) -> pd.DataFrame:
    p = P.results_root("TCGA-CV", method, model, "MSIH") / "Output" / "oof_predictions_default.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def main() -> pd.DataFrame:
    rows = []
    for method, model in valid_pairs():
        oof = tcga_oof(method, model)
        if oof.empty:
            continue
        for target, ev in EV_DIR.items():
            fs = glob.glob(str(P.SLIDE_CLS_ROOT / ev / method / model /
                               "1-MSIH" / "Output" / "result_*.json"))
            if not fs:
                continue
            res = json.load(open(fs[0]))["results"]
            try:
                y = cohort(target, method, model).labels.numpy()
            except Exception:
                continue
            for clf in CLASSIFIERS:
                v = res.get(clf, {}).get("tcga_full")
                g = oof[oof.classifier == clf]
                if not v or "mean_probs" not in v or g.empty:
                    continue
                p = np.array(v["mean_probs"])
                if len(p) != len(y):
                    continue
                tau = float(v["threshold"])

                # operating RATE on TCGA, transported to the target
                r = float((g.prob_pos.values >= tau).mean())
                tau_q = float(np.quantile(p, 1 - r)) if 0 < r < 1 else tau

                pred_t = (p >= tau).astype(int)
                pred_q = (p >= tau_q).astype(int)
                n = len(y)
                rows.append(dict(
                    cohort=target, method=method, model=model, clf=clf, n=n,
                    AUROC=round(roc_auc_score(y, p), 4),
                    BalAcc_tau=round(balanced_accuracy_score(y, pred_t), 4),
                    BalAcc_quantile=round(balanced_accuracy_score(y, pred_q), 4),
                    BalAcc_best=round(oracle_bacc(y, p), 4),
                    r_tcga=round(r, 4), tau=round(tau, 6), tau_q=round(tau_q, 6),
                    n_pos_tau=int(pred_t.sum()), n_pos_quantile=int(pred_q.sum()),
                    status_tau=health(int(pred_t.sum()), n),
                    status_quantile=health(int(pred_q.sum()), n),
                    p_min=round(float(p.min()), 6), p_max=round(float(p.max()), 6),
                ))

    df = pd.DataFrame(rows)
    df["recovered"] = (df.BalAcc_quantile - df.BalAcc_tau).round(4)
    df["gap_tau"] = (df.BalAcc_best - df.BalAcc_tau).round(4)
    df["gap_quantile"] = (df.BalAcc_best - df.BalAcc_quantile).round(4)
    out = RESULTS / "C_quantile_threshold.csv"
    df.to_csv(out, index=False)

    print("=== EXPERIMENT C - tau_TCGA vs quantile threshold, by classifier ===")
    agg = df.groupby(["cohort", "clf"]).agg(
        n=("AUROC", "size"), AUROC=("AUROC", "mean"),
        BalAcc_tau=("BalAcc_tau", "mean"),
        BalAcc_quantile=("BalAcc_quantile", "mean"),
        BalAcc_best=("BalAcc_best", "mean"),
        recovered=("recovered", "mean"),
        gap_tau=("gap_tau", "mean"), gap_quantile=("gap_quantile", "mean"),
    ).round(4)
    agg["closed_%"] = (100 * (1 - agg.gap_quantile / agg.gap_tau)).round(0)
    print(agg.to_string())

    print("\n--- dead / saturated configs before and after ---")
    for col, lab in [("status_tau", "tau_TCGA"), ("status_quantile", "quantile")]:
        bad = df[~df[col].isin(["ok"])]
        print(f"  {lab:10s} unhealthy: {len(bad):3d} of {len(df)}   "
              + "  ".join(f"{k}={v}" for k, v in bad[col].value_counts().items()))

    print(f"\nwrote {out}")
    return df


if __name__ == "__main__":
    main()
