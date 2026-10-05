"""Per-slide external-validation scores under ONE threshold rule for every head.

Why this exists
---------------
The published external-validation tables (``--threshold-mode promoted``) apply
the corrected scheme to KNN and RF only, and SurGen-EV scores KNN at k=20, a
value picked from a sweep on SurGen itself. This script evaluates the same saved
TCGA models under a single rule on both cohorts:

  * KNN                - k = 35 (``threshold_modes.KNN_K_VALIDATED``), tau refitted
                         on TCGA out-of-fold probabilities at that k
  * LR, ANN, ProtoNet, RF - rate-matched (quantile) threshold

It is the ``corrected`` scheme of ``runners/threshold_modes.py`` for all five
heads. Nothing here is new method: every number comes from the functions the
runner itself calls. What is new is that the per-slide scores are written out,
which the result JSONs do not do for the corrected KNN, so that confidence
intervals and paired tests can be computed (``02_uncertainty.py``).

No target label is used to choose k, tau or r. The quantile rule reads target
*scores* (it is transductive); target labels enter the reported metrics only.

Read-only with respect to the published trees: this script loads models and
features and writes only into ``experiments/final_analysis/results/``.

Run (environment the TCGA models were trained in: scikit-learn 1.7.0):
    <conda>/envs/exaonepath/python.exe experiments/final_analysis/01_uniform_ev_scores.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "slide_classification"))

import data_layer as dl                                    # noqa: E402
from config import paths as P                              # noqa: E402
from runners import threshold_modes as tm                  # noqa: E402
from runners.classifiers import MODEL_TYPES                # noqa: E402
from runners.ev_runner import _metrics_at, N_FOLDS         # noqa: E402
from runners.full_trainer import SEEDS, artifact_dir       # noqa: E402
from runners.model_io import load_model, predict_proba, _prep  # noqa: E402
from runners.surgen_folds import case_id_of                # noqa: E402
from runners.thresholds import load_tau                    # noqa: E402

OUT = HERE / "results"
KNN_K = tm.KNN_K_VALIDATED
COHORTS = {"paip": "PAIP-EV", "surgen": "SurGen-EV"}


def _case(cohort: str, slide_id: str) -> str:
    """Resampling unit. SurGen has two-slide cases; PAIP is one slide per case."""
    return case_id_of(slide_id) if cohort == "surgen" else slide_id


def evaluate(cohort: str, method: str, model: str, task: str = "MSIH"):
    coh = dl.load_cohort(cohort, method, model, task=task, verbose=False)
    y = coh.labels.numpy().astype(int)
    n = len(y)
    slide_rows, summary_rows = [], []

    tc = dl.load_cohort("tcga", method, model, task=task, verbose=False)
    fmap = dl.load_fold_map("tcga", task) or {}
    folds = np.array([fmap.get(i, -1) for i in tc.ids])
    oof = tm.load_tcga_oof(method, model, task)

    for kind in MODEL_TYPES:
        tau_frozen = load_tau(method, model, kind, task)

        # --- the published tcga_full path: 5 seed models -----------------------
        seed_probs = [predict_proba(kind, load_model(kind, artifact_dir(method, model, s, task),
                                                     fold=None, input_dim=coh.dim), coh.feats)[:, 1]
                      for s in SEEDS]
        mean_probs = np.mean(seed_probs, axis=0)
        frozen = [_metrics_at(y, p, tau_frozen) for p in seed_probs]

        # --- 4 fold models pooled (the published fold_ensemble variant) --------
        mdir = P.results_root("TCGA-CV", method, model, task) / "models"
        fold_probs = np.mean([predict_proba(kind, load_model(kind, mdir, fold=f, input_dim=coh.dim),
                                            coh.feats)[:, 1] for f in range(N_FOLDS)], axis=0)

        # --- the uniform rule ---------------------------------------------------
        if kind == "knn":
            art = load_model("knn", artifact_dir(method, model, SEEDS[0], task),
                             fold=None, input_dim=coh.dim)
            tau_u, tau_src = tm.knn_tau_from_store(method, model, task)
            if tau_u is None:              # TITAN / PRISM have no stored entry
                tau_u = tm.knn_tau_at_k(art, _prep("knn", tc.feats).numpy(),
                                        tc.labels.numpy(), folds, KNN_K)
                tau_src = "refit_at_k%d (no stored entry)" % KNN_K
            score_u = tm.knn_probs_at_k(art, _prep("knn", coh.feats).numpy(), KNN_K)
            scheme = "refit_tau_k%d" % KNN_K
            knn_metric = getattr(art, "metric", "")
        else:
            g = oof[oof.classifier == kind]
            tau_u = tm.corrected_threshold(kind, tau_frozen, g.prob_pos.values, mean_probs)
            tau_src = "quantile of target scores at the TCGA positive-call rate"
            score_u = mean_probs
            scheme = "quantile_rate_matched"
            knn_metric = ""

        m_u = _metrics_at(y, score_u, tau_u)
        npp = int((score_u >= tau_u).sum())
        tn, fp = m_u["conf_matrix"][0]
        fn, tp = m_u["conf_matrix"][1]

        summary_rows.append({
            "cohort": cohort, "experiment": COHORTS[cohort], "method": method, "model": model,
            "classifier": kind, "n": n, "n_pos": int(y.sum()),
            # threshold-free
            "auroc_mean_of_seeds": float(np.mean([m["auroc"] for m in frozen])),
            "auroc_seed_ensemble": float(_metrics_at(y, mean_probs, tau_frozen)["auroc"]),
            "auroc_uniform": m_u["auroc"],          # = seed ensemble, or KNN at k=35
            "auroc_fold_ensemble": float(_metrics_at(y, fold_probs, tau_frozen)["auroc"]),
            # operating points
            "bacc_frozen_mean_of_seeds": float(np.mean([m["bacc"] for m in frozen])),
            "bacc_uniform": m_u["bacc"], "sens_uniform": tp / max(tp + fn, 1),
            "spec_uniform": tn / max(tn + fp, 1), "acc_uniform": m_u["acc"],
            "macro_f1_uniform": m_u["macro_f1"],
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "threshold_frozen": float(tau_frozen), "threshold_uniform": float(tau_u),
            "n_pos_pred_frozen": int((mean_probs >= tau_frozen).sum()),
            "n_pos_pred_uniform": npp,
            "status_frozen": tm.health(int((mean_probs >= tau_frozen).sum()), n),
            "status_uniform": tm.health(npp, n),
            "scheme": scheme, "threshold_source": tau_src, "knn_metric": knn_metric,
        })
        for i, sid in enumerate(coh.ids):
            slide_rows.append((cohort, method, model, kind, sid, _case(cohort, sid), int(y[i]),
                               float(score_u[i]), float(tau_u), float(mean_probs[i]),
                               float(tau_frozen), float(fold_probs[i])))
    return slide_rows, summary_rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=UserWarning)
    cols = ["cohort", "method", "model", "classifier", "slide_id", "case_id", "target",
            "score_uniform", "threshold_uniform", "score_seed_mean", "threshold_frozen",
            "score_fold_ensemble"]
    all_summary = []
    for cohort in COHORTS:
        rows = []
        for method in P.AGGREGATION_METHODS:
            for model in P.CANONICAL_MODELS:
                if not P.is_combination_valid(method, model):
                    continue
                if cohort == "surgen" and method == "PRISM":
                    # Same exclusion as ev_runner. The 622 SurGen PRISM embeddings
                    # exist but are not usable yet: SR386_40X_HE_T237_01.pt is
                    # corrupt (not a readable torch file) and SR386_40X_HE_T241_01
                    # was built from a slide missing 52% of its tiles.
                    print("  skip: surgen PRISM (2 slides need re-encoding first)")
                    continue
                if not P.feature_dir(cohort, method, model).is_dir():
                    print("  skip (no features on this machine): %s %s/%s" % (cohort, method, model))
                    continue
                try:
                    r, s = evaluate(cohort, method, model)
                except FileNotFoundError as exc:
                    print("  skip (missing model): %s %s/%s - %s" % (cohort, method, model, exc))
                    continue
                rows += r
                all_summary += s
                best = max(s, key=lambda d: d["auroc_uniform"])
                print("%-7s %-38s %-12s n=%d  best AUROC %.4f (%s)" % (
                    cohort, method, model, s[0]["n"], best["auroc_uniform"], best["classifier"]))
        df = pd.DataFrame(rows, columns=cols)
        df.to_csv(OUT / ("ev_scores_%s.csv" % cohort), index=False, float_format="%.10g")
        print("wrote %s: %d rows, %d combinations" % (
            "ev_scores_%s.csv" % cohort, len(df), df.groupby(["method", "model"]).ngroups))
    pd.DataFrame(all_summary).to_csv(OUT / "ev_uniform_summary.csv", index=False, float_format="%.10g")
    (OUT / "ev_uniform_provenance.json").write_text(json.dumps(P.run_stamp(
        script="experiments/final_analysis/01_uniform_ev_scores.py", knn_k=KNN_K,
        rule="knn: k=%d + tau refit on TCGA out-of-fold at that k; lin/ann/proto/rf: "
             "rate-matched quantile threshold" % KNN_K,
        seeds=list(SEEDS), python=sys.version.split()[0],
        sklearn=__import__("sklearn").__version__, torch=__import__("torch").__version__,
        numpy=np.__version__), indent=2, default=str), encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
