"""Literal "one number per patch" baseline: the sum of each patch vector's values.

Each patch becomes s_i = sum_d x_id (a fixed, unlearned projection onto the all-ones direction). A slide is then
summarised by the distribution of its patch sums (mean, standard deviation and the 5/10/25/50/75/90/95th
percentiles: 9 numbers), standardised on the training slides, and classified by logistic regression
(class-balanced). Same protocol as 02_train_abmil.py: TCGA 4 published folds; PAIP-EV trained on all TCGA with the
uniform threshold rule. Contrast with --method topk, where the one number per patch is LEARNED.

Writes results/<model>/patchsum/{tcga_oof.csv, paip_ev.csv, summary.json} and the patch sums per slide
(results/<model>/patchsum/scores/{tcga,paip}/<slide>.npy) for heatmaps.

  <conda>/envs/exaonepath/python.exe experiments/attention_mil/05_patchsum_baseline.py --model H-Optimus-1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
SC = HERE.parents[1] / "slide_classification"
sys.path.insert(0, str(SC))
from runners.threshold_modes import quantile_threshold  # noqa: E402
from runners.thresholds import youden_threshold  # noqa: E402

Q = [5, 10, 25, 50, 75, 90, 95]


def slide_features(cohort, model, ids, out):
    (out / "scores" / cohort).mkdir(parents=True, exist_ok=True)
    rows = []
    for s in ids:
        x = torch.load(HERE / "features" / cohort / model / f"{s}.pt")["x"].float()
        v = x.sum(1).numpy()
        np.save(out / "scores" / cohort / f"{s}.npy", v.astype(np.float32))
        rows.append([v.mean(), v.std(), *np.percentile(v, Q)])
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    out = HERE / "results" / a.model / "patchsum"
    out.mkdir(parents=True, exist_ok=True)
    k = pd.read_csv(SC / "kfolds_IDARS_fixed.csv")
    k = k[[(HERE / "features" / "tcga" / a.model / f"{s}.pt").exists() for s in k.WSI_Id]]
    t_ids, t_y, t_fold = list(k.WSI_Id), k.label_id.astype(int).to_numpy(), k.fold.to_numpy()
    pl = pd.read_csv(SC / "paip_78_labels.csv")
    p_ids, p_y = list(pl.WSI_Id), (pl.label == "MSIH").astype(int).to_numpy()
    Xt, Xp = slide_features("tcga", a.model, t_ids, out), slide_features("paip", a.model, p_ids, out)
    clf = lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))

    score = np.zeros(len(t_ids))
    for f in sorted(set(t_fold)):
        tr, te = t_fold != f, t_fold == f
        score[te] = clf().fit(Xt[tr], t_y[tr]).predict_proba(Xt[te])[:, 1]
    pd.DataFrame({"slide_id": t_ids, "fold": t_fold, "target": t_y, "score": score}).to_csv(out / "tcga_oof.csv", index=False)
    tau = float(youden_threshold(t_y, score)["tau"])
    ev = clf().fit(Xt, t_y).predict_proba(Xp)[:, 1]
    thr = quantile_threshold(score, tau, ev)
    pd.DataFrame({"slide_id": p_ids, "target": p_y, "score": ev, "threshold": thr}).to_csv(out / "paip_ev.csv", index=False)
    bacc = lambda y, pr: 0.5 * ((pr[y == 1]).mean() + (1 - pr[y == 0]).mean())
    summ = {"model": a.model, "method": "patchsum",
            "tcga_cv_auroc_pooled": roc_auc_score(t_y, score),
            "tcga_cv_auroc_mean_fold": float(np.mean([roc_auc_score(t_y[t_fold == f], score[t_fold == f]) for f in sorted(set(t_fold))])),
            "tcga_cv_bacc_at_0.5": float(bacc(t_y, (score >= 0.5).astype(float))),
            "paip_ev_auroc": roc_auc_score(p_y, ev), "paip_ev_bacc": float(bacc(p_y, (ev >= thr).astype(float)))}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
