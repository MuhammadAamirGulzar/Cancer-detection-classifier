"""Check that 01_uniform_ev_scores.py reproduces the published numbers wherever the two must agree.

Three independent comparisons:
  A. frozen path  - seed-mean probabilities, mean-of-seeds AUROC and frozen balanced accuracy
                    against the published result JSONs (all heads, both cohorts).
  B. promoted heads - the uniform rule equals what is already published for PAIP KNN (k=35)
                    and for RF on both cohorts.
  C. other heads  - LR / ANN / ProtoNet quantile values, and SurGen KNN at k=35, against the
                    corrected-mode snapshots kept in the *_ARCHIVED_* trees.

Reads only. Exit code 1 if any comparison exceeds its tolerance.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SC = HERE.parents[1] / "slide_classification"
TREES = {"paip": ("TCGA_PAIP_EV_Results", "PAIP-EV"), "surgen": ("SurGen_EV_Results", "SurGen-EV")}
ARCH = {"paip": "TCGA_PAIP_EV_Results_ARCHIVED_20260902_PRE_KNNRF",
        "surgen": "SurGen_EV_Results_ARCHIVED_20260903_PRE_STRIP"}
TOL_PROB, TOL_METRIC = 1e-4, 1e-3

summ = pd.read_csv(HERE / "results" / "ev_uniform_summary.csv")
fail = []


def load_tree(tree, exp):
    out = {}
    for f in glob.glob(str(SC / tree / "**" / ("result_%s_*_ev.json" % exp)), recursive=True):
        d = json.load(open(f, encoding="utf-8"))
        out[(d["stamp"]["method"], d["stamp"]["model"])] = d
    return out


for cohort, (tree, exp) in TREES.items():
    pub = load_tree(tree, exp)
    arc = load_tree(ARCH[cohort], exp)
    sc = pd.read_csv(HERE / "results" / ("ev_scores_%s.csv" % cohort))
    s = summ[summ.cohort == cohort]
    dprob, dauc, dbacc, n_a = [], [], [], 0
    b_rows, c_rows = [], []
    for (method, model), g in s.groupby(["method", "model"]):
        if (method, model) not in pub:
            fail.append("%s %s/%s: no published result to compare" % (cohort, method, model)); continue
        res = pub[(method, model)]["results"]
        for _, r in g.iterrows():
            k = r.classifier
            e = res[k]["tcga_full"]
            mine = sc[(sc.method == method) & (sc.model == model) & (sc.classifier == k)]
            # A. frozen path
            dprob.append(float(np.max(np.abs(mine.score_seed_mean.values - np.asarray(e["mean_probs"])))))
            dauc.append(abs(r.auroc_mean_of_seeds - e.get("auroc_frozen", e["auroc"])))
            dbacc.append(abs(r.bacc_frozen_mean_of_seeds - e.get("bacc_frozen", e["bacc"])))
            n_a += 1
            # B. heads already promoted in the published tree
            if e.get("promoted"):
                same_k = (k != "knn") or (e.get("knn_k_corrected") == 35)
                if same_k:
                    b_rows.append((k, abs(r.bacc_uniform - e["bacc"]), abs(r.auroc_uniform - e["auroc"]) if k == "knn" else 0.0,
                                   abs(r.threshold_uniform - e["threshold"])))
            # C. corrected-mode snapshot
            a = arc.get((method, model), {}).get("results", {}).get(k, {}).get("tcga_full")
            if a and "bacc_corrected" in a and arc[(method, model)]["stamp"].get("n_test") == r.n:
                if k == "knn" and a.get("knn_k_corrected") != 35:
                    continue
                c_rows.append((k, abs(r.bacc_uniform - a["bacc_corrected"]),
                               abs(r.auroc_uniform - a["auroc_corrected"]) if k == "knn" else 0.0,
                               abs(r.threshold_uniform - a["threshold_corrected"])))
    print("== %s ==" % cohort)
    print("A. frozen path, %d classifier-combinations: max |d prob| %.2e, max |d AUROC| %.2e, max |d BalAcc| %.2e"
          % (n_a, max(dprob), max(dauc), max(dbacc)))
    if max(dprob) > TOL_PROB or max(dauc) > TOL_METRIC:
        fail.append("%s: frozen path differs from the published tree" % cohort)
    if max(dbacc) > 0.02:
        fail.append("%s: frozen balanced accuracy differs by more than 0.02" % cohort)
    for name, rows in (("B. already-promoted heads vs published", b_rows), ("C. corrected-mode snapshot", c_rows)):
        if not rows:
            print("%s: nothing comparable" % name); continue
        t = pd.DataFrame(rows, columns=["head", "d_bacc", "d_auroc", "d_thr"])
        agg = t.groupby("head").agg(n=("d_bacc", "size"), max_d_bacc=("d_bacc", "max"),
                                    max_d_auroc=("d_auroc", "max"), max_d_thr=("d_thr", "max"))
        print(name); print(agg.to_string(float_format=lambda v: "%.2e" % v))
        if (t.d_bacc > 0.02).any() or (t.d_auroc > TOL_METRIC).any():
            fail.append("%s: %s exceeds tolerance" % (cohort, name))
    print()

print("RESULT:", "ALL COMPARISONS WITHIN TOLERANCE" if not fail else "FAILED")
for f in fail:
    print("  !!", f)
sys.exit(1 if fail else 0)
