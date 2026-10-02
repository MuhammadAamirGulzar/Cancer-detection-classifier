"""PART 2 - DIAGNOSTIC ONLY. kNN neighbourhood sweep x threshold scheme.

=============================================================================
 THIS IS NOT A REPORTABLE CONFIGURATION. Nothing here is promoted, no k is
 selected, and no published file is written. Every output CSV carries a
 DIAGNOSTIC column so a row cannot be lifted into a results table by accident.
=============================================================================

The k grid brackets the peak rather than starting at it. An earlier sweep put
SurGen kNN's best AUROC at k=25 (0.6675), already declining by k=50 (0.6454), so
{15, 20} are included below the reported peak and {75, 100} well above it. A
maximum with no measured shoulder on either side is not a maximum, it is an
endpoint.

Three schemes, all TCGA-derived
-------------------------------
(a) ``frozen``    tau_TCGA exactly as published - the value Youden's J picked on
                  TCGA out-of-fold probabilities at the artifact's own searched
                  k (3 in most configs). Applied unchanged to scores computed at
                  the sweep's k, which is the point: it shows what the published
                  threshold does once the score scale moves under it.
(b) ``refit``     Youden's J re-run on TCGA out-of-fold probabilities computed at
                  the SAME k. The internally consistent choice.
(c) ``quantile``  Rate-matched. r = the fraction of TCGA out-of-fold slides that
                  tau_TCGA flags positive AT THIS k; the target is then cut at
                  its own (1-r) quantile. r is reported so a degenerate rate is
                  visible rather than inferred.

None of the three reads target labels. Target labels enter only the reported
metrics and the oracle column.

``BalAcc_oracle``
-----------------
The best balanced accuracy any threshold could achieve on the target, found by
maximising over the ROC. It USES TARGET LABELS and is therefore not attainable
by any honest rule - it is the ceiling, included so a scheme's shortfall can be
read as "bad threshold" versus "bad scores". A scheme sitting near the oracle has
a thresholding story; one far below it has a representation story.

Everything is computed on ONE score vector
------------------------------------------
Scores are the seed-averaged probabilities (``mean_probs`` for the non-kNN heads,
read from the published record; the re-scored vector for kNN). All three schemes
threshold that same vector, which is what makes them comparable.

Note this is NOT the path the published BalAcc takes: ``_aggregate_seeds``
averages five per-seed confusion matrices instead. Scheme (a)'s BalAcc here can
therefore differ slightly from the published figure, so both are reported -
``bacc`` (computed here) and ``bacc_published`` (from the record). Part 1
documents the same distinction.

Run:  python experiments/surgen_ev_diagnostic/part2_k_sweep.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = HERE / "results"
sys.path.insert(0, str(REPO / "experiments" / "threshold_and_k"))
sys.path.insert(0, str(REPO / "slide_classification"))

from common import (                                   # noqa: E402
    P, cohort, knn_artifact, distances, tcga_fold_map, load_tau,
    youden, oracle_bacc, health,
)
from runners import threshold_modes as tm              # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402

K_GRID = [15, 20, 25, 35, 50, 75, 100]
EV_TREE = {"paip": "TCGA_PAIP_EV_Results", "surgen": "SurGen_EV_Results"}
EV_GLOB = {"paip": "result_PAIP-EV_*_ev.json", "surgen": "result_SurGen-EV_*_ev.json"}
OTHER_HEADS = ("lin", "ann", "proto", "rf")


def combos(target: str):
    """(method, model, published results dict) for every combination on disk."""
    root = REPO / "slide_classification" / EV_TREE[target]
    for f in sorted(root.rglob(EV_GLOB[target])):
        payload = json.loads(f.read_text(encoding="utf-8"))
        yield payload["stamp"]["method"], payload["stamp"]["model"], payload["results"]


def metrics_at(y, p, tau):
    pred = (p >= tau).astype(int)
    return dict(auroc=round(float(roc_auc_score(y, p)), 4),
                bacc=round(float(balanced_accuracy_score(y, pred)), 4),
                n_pos_pred=int(pred.sum()),
                status=health(int(pred.sum()), len(y)))


def knn_scores_all_k(method, model, target, ks):
    """Target scores and TCGA out-of-fold scores at every k, distances computed once.

    kNN is lazy: the artifact carries the whole TCGA matrix, so every k is the
    same estimator read at a different neighbourhood size. Sorting once and
    slicing is exact, not an approximation.
    """
    art = knn_artifact(method, model)
    metric = getattr(art, "metric", "cosine")
    tc = cohort("tcga", method, model)
    y_ref = tc.labels.numpy()

    D_t = distances(method, model, target, metric)
    order_t = np.argsort(D_t, axis=1)
    tgt = {k: y_ref[order_t[:, :min(k, D_t.shape[1])]].mean(axis=1) for k in ks}

    D_o = distances(method, model, None, metric).copy()
    np.fill_diagonal(D_o, np.inf)
    fm = tcga_fold_map()
    folds = np.array([fm.get(i, -1) for i in tc.ids])
    oof = {k: np.full(len(y_ref), np.nan) for k in ks}
    for f in sorted(set(folds[folds > 0])):
        te = np.where(folds == f)[0]
        ref = np.where(folds != f)[0]
        o = np.argsort(D_o[np.ix_(te, ref)], axis=1)
        for k in ks:
            oof[k][te] = y_ref[ref][o[:, :min(k, len(ref))]].mean(axis=1)
    return metric, tgt, {k: (y_ref, oof[k]) for k in ks}


def quantile_tau(oof_p, tau_frozen, tgt_p):
    """Rate-matched cut point, plus the rate itself so a degenerate r is visible."""
    r = float((np.asarray(oof_p) >= tau_frozen).mean())
    if not 0.0 < r < 1.0:
        return float(tau_frozen), r, True
    return float(np.quantile(np.asarray(tgt_p), 1.0 - r)), r, False


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    knn_rows, other_rows, skipped = [], [], []

    for target in ("surgen", "paip"):
        for method, model, published in combos(target):
            coh = cohort(target, method, model)
            y = coh.labels.numpy()

            # ---------------------------------------------- kNN, swept over k
            try:
                tau_frozen = load_tau(method, model, "knn", "MSIH")
                metric, tgt, oof = knn_scores_all_k(method, model, target, K_GRID)
            except Exception as exc:
                print(f"  [skip knn] {target}/{method}/{model}: "
                      f"{type(exc).__name__}: {exc}")
                continue

            for k in K_GRID:
                p = tgt[k]
                y_oof, p_oof = oof[k]
                ok = ~np.isnan(p_oof)
                base = dict(DIAGNOSTIC="not-a-reportable-config", cohort=target,
                            method=method, model=model, clf="knn", k=k,
                            metric=metric, weights="uniform",
                            n_distinct_scores=int(np.unique(p).size),
                            n_target=len(y),
                            bacc_oracle=round(oracle_bacc(y, p), 4))
                schemes = {
                    "a_frozen":   (float(tau_frozen), np.nan, False),
                    "b_refit_k":  (youden(y_oof[ok], p_oof[ok]), np.nan, False),
                    "c_quantile": quantile_tau(p_oof[ok], float(tau_frozen), p),
                }
                for name, (tau, r, degen) in schemes.items():
                    knn_rows.append({**base, "scheme": name, "tau": round(float(tau), 6),
                                     "quantile_r": (None if np.isnan(r) else round(r, 4)),
                                     "quantile_degenerate": bool(degen),
                                     **metrics_at(y, p, tau)})

            # ------------------------- other heads, existing config, (a) and (c)
            oof_tbl = tm.load_tcga_oof(method, model, "MSIH")
            for clf in OTHER_HEADS:
                rec = published.get(clf, {}).get("tcga_full")
                if not rec or "mean_probs" not in rec:
                    continue
                p = np.asarray(rec["mean_probs"], dtype=float)
                # Some SurGen combinations were run before features existed for
                # every slide, so the stored score vector is shorter than the
                # cohort loads today. Which slides were dropped is not recoverable
                # from the record, and guessing an alignment would silently pair
                # labels with the wrong scores. Skip and say so.
                if len(p) != len(y):
                    skipped.append(dict(cohort=target, method=method, model=model,
                                        clf=clf, n_cohort_now=len(y),
                                        n_in_record=len(p),
                                        reason="stored mean_probs shorter than cohort"))
                    continue
                # PAIP-EV promoted kNN and RF on 2 Sep, so for those heads
                # ``threshold`` is ALREADY the corrected cut point. Reading it as
                # scheme (a) would label a rate-matched threshold "frozen" and
                # collapse (a) onto (c). The displaced frozen value is kept in the
                # record as ``threshold_frozen``; use it, so scheme (a) means the
                # same thing in both cohorts.
                promoted = bool(rec.get("promoted"))
                tau_f = float(rec.get("threshold_frozen", rec["threshold"]))
                g = oof_tbl[oof_tbl.classifier == clf] if not oof_tbl.empty else oof_tbl
                base = dict(DIAGNOSTIC="not-a-reportable-config", cohort=target,
                            method=method, model=model, clf=clf, k=None,
                            n_distinct_scores=int(np.unique(p).size), n_target=len(y),
                            bacc_oracle=round(oracle_bacc(y, p), 4),
                            published_is_promoted=promoted,
                            bacc_published=round(float(
                                rec.get("bacc_frozen", rec["bacc"])), 4),
                            auroc_published=round(float(
                                rec.get("auroc_frozen", rec["auroc"])), 4),
                            bacc_headline_now=round(float(rec["bacc"]), 4))
                other_rows.append({**base, "scheme": "a_frozen",
                                   "tau": round(tau_f, 6), "quantile_r": None,
                                   "quantile_degenerate": False,
                                   **metrics_at(y, p, tau_f)})
                if g is not None and not g.empty:
                    tau_q, r, degen = quantile_tau(g.prob_pos.values, tau_f, p)
                    other_rows.append({**base, "scheme": "c_quantile",
                                       "tau": round(tau_q, 6), "quantile_r": round(r, 4),
                                       "quantile_degenerate": bool(degen),
                                       **metrics_at(y, p, tau_q)})
            print(f"  done {target:<7} {method}/{model}")

    kdf = pd.DataFrame(knn_rows)
    odf = pd.DataFrame(other_rows)
    kdf.to_csv(RESULTS / "part2_DIAGNOSTIC_knn_k_sweep.csv", index=False)
    odf.to_csv(RESULTS / "part2_DIAGNOSTIC_other_heads.csv", index=False)
    if skipped:
        sk = pd.DataFrame(skipped)
        sk.to_csv(RESULTS / "part2_DIAGNOSTIC_skipped.csv", index=False)
        print(f"\nSKIPPED {len(sk)} head(s) - stored score vector shorter than "
              f"the cohort as it loads today:")
        print(sk.groupby(["cohort","method","model","n_cohort_now","n_in_record"])
                .size().rename("heads").reset_index().to_string(index=False))
    print(f"\nwrote part2_DIAGNOSTIC_knn_k_sweep.csv   {len(kdf)} rows")
    print(f"wrote part2_DIAGNOSTIC_other_heads.csv   {len(odf)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
