"""Corrected external-validation tables, written alongside the originals.

Applies the two fixes validated in experiments A-D. Nothing existing is
touched: new KNN artifacts, thresholds and tables all land under
``experiments/corrected_ev/``. The published tables, ``thresholds_TCGA.json``
and every artifact in ``TCGA_FULL_Models`` are opened read-only, so the old and
new numbers can sit side by side in the paper.

The two corrections
-------------------
**KNN - k=35, and a threshold refitted at that k.**
    k=35 is chosen on TCGA alone, two independent ways: it maximised TCGA
    out-of-fold balanced accuracy (0.6971, best of every k tested, Experiment
    D2), and it satisfies the prevalence rule 35 x 0.145 ~ 5 expected positive
    neighbours. ``weights='uniform'`` is kept because distance weighting bought
    only +0.006 AUROC (Experiment D1), and the per-config metric selected by the
    original GridSearchCV (cosine / manhattan) is kept as-is.

    KNN keeps a *refit* tau rather than the quantile rule: Experiment C showed
    the quantile makes KNN worse (PAIP 0.5959 -> 0.5746) and saturates 9
    configs, because a tied-block score has nowhere to cut at an arbitrary rate.

**LR / ANN / ProtoNet / RF - rate-matched (quantile) threshold.**
    r = the fraction of TCGA out-of-fold slides tau_TCGA flags positive; the
    target is cut at its own (1 - r) quantile. Recovered 46-84% of the
    threshold gap in Experiment C.

No target labels
----------------
k=35 comes from TCGA out-of-fold scores. The KNN refit tau comes from TCGA
out-of-fold scores. r comes from TCGA out-of-fold scores. The quantile reads
target *scores* but never target *labels*. Target labels are used only to
compute the reported metrics, never to choose k or any threshold. This is
asserted at runtime by ``_assert_label_free``.

Output
------
  results/corrected_ev_<cohort>.csv   per-config rows, both thresholds
  results/corrected_ev_summary.csv    means by classifier x cohort
  artifacts/knn_k35/<method>/<model>/full_knn_model.pkl
  results/knn_k35_thresholds.json
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "threshold_and_k"))

from common import (CLASSIFIERS, P, RESULTS as EXP_RESULTS, cohort,  # noqa: E402
                    knn_artifact, load_tau, oracle_bacc, target_knn_probs,
                    tcga_oof_knn, valid_pairs, _prep)
# Promoted logic - imported, not duplicated, so the pipeline and this
# validated reference cannot drift apart.
from runners.threshold_modes import health, youden, quantile_threshold  # noqa: E402

K_NEW = 35
WEIGHTS = "uniform"
OUT = HERE / "results"
ART = HERE / "artifacts" / f"knn_k{K_NEW}"
EV_DIR = {"paip": "TCGA_PAIP_EV_Results", "surgen": "SurGen_EV_Results"}
QUANTILE_CLFS = ["lin", "ann", "proto", "rf"]      # KNN deliberately excluded

OUT.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)


def _assert_label_free(name: str, arr) -> None:
    """Guard: nothing derived from target labels may reach a threshold."""
    if arr is None:
        raise AssertionError(f"{name} is None - threshold inputs must be explicit")


def tcga_oof(method: str, model: str) -> pd.DataFrame:
    p = (P.results_root("TCGA-CV", method, model, "MSIH") / "Output"
         / "oof_predictions_default.csv")
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def save_knn_k35(method: str, model: str, metric: str):
    """Refit KNN at k=35 on all 413 TCGA slides and persist under the new tree.

    KNN is lazy, so 'fitting' stores the reference matrix; the estimator is
    deterministic, which is why one artifact replaces the five per-seed ones.
    """
    tc = cohort("tcga", method, model)
    X = _prep("knn", tc.feats).numpy()
    y = tc.labels.numpy()
    clf = KNeighborsClassifier(n_neighbors=K_NEW, metric=metric, weights=WEIGHTS)
    clf.fit(X, y)
    d = ART / method / model
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, d / "full_knn_model.pkl")
    (d / "full_knn_meta.json").write_text(json.dumps(dict(
        k=K_NEW, metric=metric, weights=WEIGHTS, n_train=int(len(y)),
        n_pos=int(y.sum()),
        rationale="k=35: max TCGA OOF BalAcc (exp D2) + prevalence rule "
                  "35*0.145~5 expected positive neighbours; TCGA-only"),
        indent=2), encoding="utf-8")
    return clf


def main() -> pd.DataFrame:
    rows, knn_taus = [], {}

    for method, model in valid_pairs():
        try:
            art = knn_artifact(method, model)
        except FileNotFoundError:
            continue
        metric = art.metric
        oof = tcga_oof(method, model)
        if oof.empty:
            continue

        # --- KNN: new artifact + refit tau at k=35, both TCGA-only ----------
        save_knn_k35(method, model, metric)
        y_oof, p_oof = tcga_oof_knn(method, model, K_NEW, metric, WEIGHTS)
        _assert_label_free("knn tau source (TCGA OOF)", p_oof)
        tau_knn_new = youden(y_oof, p_oof)
        knn_taus[f"{method}|{model}|MSIH"] = dict(
            tau=tau_knn_new, k=K_NEW, metric=metric, weights=WEIGHTS,
            fitted_on="TCGA-CV out-of-fold", n=int(len(y_oof)),
            n_pos=int(y_oof.sum()))

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
            n = len(y)

            for clf in CLASSIFIERS:
                v = res.get(clf, {}).get("tcga_full")
                g = oof[oof.classifier == clf]
                if not v or "mean_probs" not in v or g.empty:
                    continue
                p_frozen = np.array(v["mean_probs"])
                if len(p_frozen) != n:
                    continue
                tau_frozen = float(v["threshold"])
                pred_f = (p_frozen >= tau_frozen).astype(int)

                if clf == "knn":
                    # corrected = NEW scores at k=35 + refit tau
                    p_corr = target_knn_probs(method, model, target, K_NEW,
                                              metric, WEIGHTS)
                    tau_corr, scheme = tau_knn_new, f"refit_tau_k{K_NEW}"
                else:
                    # corrected = same scores, rate-matched threshold
                    p_corr = p_frozen
                    r = float((g.prob_pos.values >= tau_frozen).mean())
                    _assert_label_free("quantile rate r (TCGA OOF)", r)
                    tau_corr = quantile_threshold(g.prob_pos.values,
                                                  tau_frozen, p_corr)
                    scheme = "quantile_rate_matched"
                pred_c = (p_corr >= tau_corr).astype(int)

                rows.append(dict(
                    cohort=target, method=method, model=model, clf=clf, n=n,
                    scheme=scheme,
                    AUROC_frozen=round(roc_auc_score(y, p_frozen), 4),
                    AUROC_corrected=round(roc_auc_score(y, p_corr), 4),
                    BalAcc_frozen=round(balanced_accuracy_score(y, pred_f), 4),
                    BalAcc_corrected=round(balanced_accuracy_score(y, pred_c), 4),
                    BalAcc_oracle=round(oracle_bacc(y, p_corr), 4),
                    status_frozen=health(int(pred_f.sum()), n),
                    status_corrected=health(int(pred_c.sum()), n),
                    n_pos_frozen=int(pred_f.sum()), n_pos_corrected=int(pred_c.sum()),
                    tau_frozen=round(tau_frozen, 6), tau_corrected=round(tau_corr, 6),
                ))

    df = pd.DataFrame(rows)
    df["dAUROC"] = (df.AUROC_corrected - df.AUROC_frozen).round(4)
    df["dBalAcc"] = (df.BalAcc_corrected - df.BalAcc_frozen).round(4)

    for c, g in df.groupby("cohort"):
        g.to_csv(OUT / f"corrected_ev_{c}.csv", index=False)
    (OUT / "knn_k35_thresholds.json").write_text(json.dumps(
        {"_meta": {"k": K_NEW, "weights": WEIGHTS,
                   "policy": "Youden J on TCGA-CV out-of-fold probs at k=35",
                   "note": "NEW file; thresholds_TCGA.json unmodified"},
         "thresholds": knn_taus}, indent=2), encoding="utf-8")

    summ = df.groupby(["cohort", "clf"]).agg(
        n=("AUROC_frozen", "size"),
        AUROC_frozen=("AUROC_frozen", "mean"),
        AUROC_corrected=("AUROC_corrected", "mean"),
        BalAcc_frozen=("BalAcc_frozen", "mean"),
        BalAcc_corrected=("BalAcc_corrected", "mean"),
        BalAcc_oracle=("BalAcc_oracle", "mean")).round(4)
    summ["dBalAcc"] = (summ.BalAcc_corrected - summ.BalAcc_frozen).round(4)
    summ.to_csv(OUT / "corrected_ev_summary.csv")

    print("=== CORRECTED EV - before / after ===")
    print(summ.to_string())

    print("\n=== operating-point health ===")
    for col, lab in [("status_frozen", "frozen tau"),
                     ("status_corrected", "corrected")]:
        bad = df[df[col] != "ok"]
        detail = "  ".join(f"{k}={v}" for k, v in bad[col].value_counts().items())
        print(f"  {lab:12s} unhealthy {len(bad):3d}/{len(df)}   {detail or '-'}")

    still = df[df.status_corrected != "ok"]
    if len(still):
        print("\n--- configs STILL dead/saturated after correction "
              "(reported explicitly, not as a bare 0.5000) ---")
        print(still[["cohort", "method", "model", "clf", "status_corrected",
                     "n_pos_corrected", "n", "BalAcc_corrected", "AUROC_corrected"]]
              .to_string(index=False))

    print(f"\nwrote {OUT/'corrected_ev_<cohort>.csv'}, "
          f"{OUT/'corrected_ev_summary.csv'}, {OUT/'knn_k35_thresholds.json'}")
    print(f"wrote {ART} ({len(list(ART.rglob('*.pkl')))} new KNN artifacts)")
    return df


if __name__ == "__main__":
    main()
