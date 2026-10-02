"""SurGen-EV PROMOTED candidate at k=20 - DIAGNOSTIC, published tree untouched.

Answers one question rigorously: if SurGen-EV's kNN used k=20 with a refit tau
(the diagnostic sweep's scheme (b), at the k that maximised mean AUROC) instead
of k=35, and RF kept its rate-matched quantile threshold, would the result beat
what is currently published?

Why this is a SEPARATE script from build_candidate.py, not a --k flag on it
-----------------------------------------------------------------------------
``runners.threshold_modes.knn_tau_from_store`` always returns the tau stored in
``knn_k35_thresholds.json`` - which was fitted AT k=35 - regardless of what k a
caller asks it to score at. Naively passing k=20 through
``ev_runner._add_corrected`` would silently apply the k=35 tau to k=20-scored
probabilities: a tau fitted on one score scale, applied to a different one.
That is exactly the failure mode the whole "why 3 schemes" investigation was
about, and it would not raise - it would just produce a wrong number quietly.

So the kNN branch here is NOT ev_runner._add_corrected. It calls the same two
primitives that function calls internally (``tm.knn_tau_at_k``,
``tm.knn_probs_at_k``), forces the refit path unconditionally (never touches
the k=35 store), and is otherwise identical. RF needs no such change -
``ev_runner._add_corrected``'s RF branch has no k dependency at all, so it is
reused completely unmodified.

A discrepancy this script exists BECAUSE of
---------------------------------------------
The earlier k-sweep diagnostic (part2_k_sweep.py) recomputed AUROC/BalAcc for
ANN and RF as ``roc_auc_score(y, mean_probs)`` / a single threshold-and-score
pass. The PUBLISHED value for a stochastic head (ANN, RF - both have
auroc_sd > 0) is the MEAN OF 5 PER-SEED AUROCs (``_aggregate_seeds``), which is
NOT the same quantity - averaging probabilities across seeds and then scoring
systematically outperforms averaging 5 individually-noisier per-seed scores.
Measured gap on this cohort: RF AUROC inflated by ~0.033 mean, ANN by ~0.023
mean, some individual configs by >0.07. That is why the diagnostic's "existing
config" RF/ANN table cannot be trusted against the published numbers.

This script avoids the whole class of bug by reusing ``ev_runner._promote``,
which for RF leaves ``entry["auroc"]`` COMPLETELY UNTOUCHED (only kNN gets
``auroc_corrected`` at all) - so RF's AUROC here is provably the exact
published value, not a recomputation.

Run:  python experiments/surgen_ev_promotion/build_candidate_k20.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SLIDE_CLS = REPO / "slide_classification"
SRC = SLIDE_CLS / "SurGen_EV_Results"
DST = HERE / "candidate_k20"
RESULTS = HERE / "results"

sys.path.insert(0, str(SLIDE_CLS))
import data_layer as dl                                # noqa: E402
from runners import ev_runner as ev                     # noqa: E402
from runners import runlog                              # noqa: E402
from runners import threshold_modes as tm               # noqa: E402
from runners.thresholds import load_tau                 # noqa: E402
from runners.model_io import load_model, _prep          # noqa: E402
from runners.full_trainer import SEEDS, artifact_dir     # noqa: E402

K = 20
PROMOTE = ("knn", "rf")
CLFS = ("lin", "ann", "knn", "proto", "rf")
VARIANTS = ("tcga_full", "fold_ensemble", "fold_average")


def n_pos_pred(cm) -> int:
    return int(cm[0][1]) + int(cm[1][1])


def n_total(cm) -> int:
    return int(cm[0][0]) + int(cm[0][1]) + int(cm[1][0]) + int(cm[1][1])


def add_corrected_knn_refit_at_k(entry, method, model, task, coh, targets,
                                 tau_frozen, k, log):
    """Same as ev_runner._add_corrected's kNN branch, but ALWAYS refits at k -
    never consults knn_k35_thresholds.json, which is fitted at a different k."""
    probs = np.asarray(entry["mean_probs"], dtype=float)
    entry["status_frozen"] = tm.health(int((probs >= tau_frozen).sum()), len(targets))
    try:
        art = load_model("knn", artifact_dir(method, model, SEEDS[0], task),
                         fold=None, input_dim=coh.dim)
        tc = dl.load_cohort("tcga", method, model, task=task, verbose=False)
        fmap = dl.load_fold_map("tcga", task) or {}
        folds = np.array([fmap.get(i, -1) for i in tc.ids])
        X_tcga = _prep("knn", tc.feats).numpy()
        X_tgt = _prep("knn", coh.feats).numpy()
        tau_c = tm.knn_tau_at_k(art, X_tcga, tc.labels.numpy(), folds, k)
        probs_c = tm.knn_probs_at_k(art, X_tgt, k)
    except Exception as exc:
        log.warning(f"  knn: k={k} refit unavailable for {method}/{model} ({exc})")
        return
    m = ev._metrics_at(targets, probs_c, tau_c)
    npp = int((probs_c >= tau_c).sum())
    entry.update({
        "threshold_mode": "corrected", "threshold_corrected": float(tau_c),
        "bacc_corrected": float(m["bacc"]), "acc_corrected": float(m["acc"]),
        "macro_f1_corrected": float(m["macro_f1"]),
        "weighted_f1_corrected": float(m["weighted_f1"]),
        "conf_matrix_corrected": m["conf_matrix"], "n_pos_pred_corrected": npp,
        "status_corrected": tm.health(npp, len(targets)),
        "corrected_scheme": "refit_tau_k%d" % k,
        "auroc_corrected": float(m["auroc"]),
        "knn_k_corrected": int(k), "knn_tau_source": "refit_at_k%d (forced)" % k,
    })


def main() -> int:
    if not SRC.is_dir():
        sys.exit(f"no SurGen-EV tree at {SRC}")
    log = runlog.get_logger()
    DST.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    files = sorted(SRC.rglob("result_SurGen-EV_*_ev.json"))
    print(f"source: {SRC.name}  ({len(files)} combinations, opened read-only)")
    print(f"kNN forced to refit-tau-at-k={K} (never reads knn_k35_thresholds.json)")

    rows = []
    for f in files:
        published = json.loads(f.read_text(encoding="utf-8"))
        stamp, results = published["stamp"], published["results"]
        method, model = stamp["method"], stamp["model"]

        coh = dl.load_cohort("surgen", method, model, task="MSIH", verbose=False)
        targets = coh.labels.numpy()

        out_results = {}
        for clf in CLFS:
            per_variant = results.get(clf, {})
            new_pv = {}
            for var in VARIANTS:
                m = per_variant.get(var)
                if not isinstance(m, dict) or "bacc" not in m:
                    new_pv[var] = m
                    continue
                entry = copy.deepcopy(m)
                if var == "tcga_full" and clf in PROMOTE:
                    tau_frozen = load_tau(method, model, clf, "MSIH")
                    if clf == "knn":
                        add_corrected_knn_refit_at_k(entry, method, model, "MSIH",
                                                     coh, targets, tau_frozen, K, log)
                    else:  # rf - identical to the k=35 candidate, no k dependency
                        ev._add_corrected(entry, clf, method, model, "MSIH", coh,
                                          targets, tau_frozen, tm.KNN_K_VALIDATED, log)
                    ev._promote(entry, clf, log)
                new_pv[var] = entry
            out_results[clf] = new_pv

            tf = new_pv.get("tcga_full")
            if tf:
                rows.append(dict(
                    method=method, model=model, clf=clf,
                    auroc=round(tf["auroc"], 4), bacc=round(tf["bacc"], 4),
                    threshold=round(tf["threshold"], 6),
                    threshold_scheme=tf.get("threshold_scheme", "frozen_tau_TCGA"),
                    status=tm.health(n_pos_pred(tf["conf_matrix"]),
                                     n_total(tf["conf_matrix"])),
                    promoted=bool(tf.get("promoted")),
                    auroc_frozen_published=(round(tf["auroc_frozen"], 4)
                                            if tf.get("promoted") else round(tf["auroc"], 4)),
                    bacc_frozen_published=(round(tf["bacc_frozen"], 4)
                                           if tf.get("promoted") else round(tf["bacc"], 4)),
                ))

        new_stamp = dict(stamp)
        new_stamp["threshold_mode"] = "promoted"
        new_stamp["promoted_classifiers"] = list(PROMOTE)
        new_stamp["knn_k_corrected"] = K
        new_stamp["derived_from"] = f"{SRC.name}/{f.relative_to(SRC).as_posix()}"
        new_stamp["derivation"] = (
            f"DIAGNOSTIC candidate at k={K}, built under experiments/"
            f"surgen_ev_promotion/ via ev_runner._add_corrected/_promote (RF) and "
            f"a forced refit-at-k={K} (kNN, bypassing the k=35-fitted "
            f"knn_k35_thresholds.json). Published SurGen_EV_Results/ untouched.")

        out = DST / f.relative_to(SRC)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"stamp": new_stamp, "results": out_results},
                                  indent=2, default=str), encoding="utf-8")

    df = pd.DataFrame(rows).sort_values(["clf", "method", "model"])
    df.to_csv(RESULTS / f"surgen_ev_promoted_candidate_k{K}.csv", index=False)
    print(f"\nwrote {DST.relative_to(REPO)}/  ({len(files)} files)")
    print(f"wrote results/surgen_ev_promoted_candidate_k{K}.csv  ({len(df)} rows)")

    agg = df[~df.method.isin(["TITAN"])]
    print(f"\n=== k={K} candidate vs TRUE published, mean over 20 agg combos ===")
    for clf in PROMOTE:
        s = agg[agg.clf == clf]
        print(f"  {clf:<4} published: AUROC {s.auroc_frozen_published.mean():.4f}  "
              f"BalAcc {s.bacc_frozen_published.mean():.4f}   |   "
              f"candidate: AUROC {s.auroc.mean():.4f}  BalAcc {s.bacc.mean():.4f}   "
              f"| dead={s.status.str.contains('dead').sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
