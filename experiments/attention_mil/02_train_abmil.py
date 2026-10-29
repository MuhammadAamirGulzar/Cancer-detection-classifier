"""Patch-importance MIL on the benchmark's patch features, benchmark protocol. Two methods (--method):

  abmil  Gated attention MIL (Ilse et al. 2018). Patch vector (D) -> Linear(D, 256) + ReLU + Dropout -> h;
         attention a_i = softmax_i( w . (tanh(V h_i) * sigmoid(U h_i)) ); slide z = sum_i a_i h_i -> Linear(256, 1).
         Every patch gets a learned importance a_i (sums to 1 over the slide).
  topk   Instance-level top-k MIL. Every patch gets ONE learned score s_i (MSI-H evidence) from
         Linear(D, 256) + ReLU + Dropout + Linear(256, 1); the slide logit is the mean of its top 5% patch scores.
         The "importance" saved per patch is 1/k on the top-k patches (0 elsewhere); the raw scores are saved too.

Hyperparameters are fixed in advance and never tuned on test data: Adam lr 2e-4, weight decay 1e-4, 25 epochs,
dropout 0.25, class-weighted BCE, up to 4096 randomly sampled patches per slide per step (all patches at test time),
5 seeds (42-46) averaged, as the ANN / RF heads are.

  TCGA-CV   the published 4 folds (kfolds_IDARS_fixed.csv): out-of-fold probabilities, BalAcc at 0.5
  PAIP-EV   trained on all TCGA slides, applied to the 78 PAIP slides; threshold = the uniform rule
            (rate-matched quantile from tau_TCGA = Youden on the pooled out-of-fold probabilities)

Writes results/<model>/<method>/{tcga_oof.csv, paip_ev.csv, summary.json}, per-slide importance
results/<model>/<method>/attention/{tcga,paip}/<slide>.npy (seed-mean; TCGA from the fold model that held it out)
and, for topk, the raw patch scores in .../scores/{tcga,paip}/<slide>.npy.

  <conda>/envs/exaonepath/python.exe experiments/attention_mil/02_train_abmil.py --model H-Optimus-1 --method abmil
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SC = REPO / "slide_classification"
sys.path.insert(0, str(SC))
from runners.threshold_modes import quantile_threshold  # noqa: E402
from runners.thresholds import youden_threshold  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
HP = dict(hidden=256, attn=128, dropout=0.25, lr=2e-4, wd=1e-4, epochs=25, max_patches=4096, topk_frac=0.05)


class GatedABMIL(nn.Module):
    def __init__(self, d, hidden=256, attn=128, dropout=0.25):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.v = nn.Sequential(nn.Linear(hidden, attn), nn.Tanh(), nn.Dropout(dropout))
        self.u = nn.Sequential(nn.Linear(hidden, attn), nn.Sigmoid(), nn.Dropout(dropout))
        self.w = nn.Linear(attn, 1)
        self.cls = nn.Linear(hidden, 1)

    def forward(self, x):                                    # x (N, d) -> slide logit, importance (N,), None
        h = self.fc(x)
        a = torch.softmax(self.w(self.v(h) * self.u(h)).squeeze(1), dim=0)
        return self.cls((a.unsqueeze(1) * h).sum(0)).squeeze(), a, None


class TopKMIL(nn.Module):
    def __init__(self, d, hidden=256, dropout=0.25, frac=0.05):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.frac = frac

    def forward(self, x):                                    # x (N, d) -> slide logit, importance (N,), scores (N,)
        s = self.score(x).squeeze(1)
        k = max(1, math.ceil(self.frac * len(s)))
        top = torch.topk(s, k).indices
        w = torch.zeros_like(s)
        w[top] = 1.0 / k
        return s[top].mean(), w, s


def make(method, d):
    if method == "abmil":
        return GatedABMIL(d, HP["hidden"], HP["attn"], HP["dropout"])
    return TopKMIL(d, HP["hidden"], HP["dropout"], HP["topk_frac"])


def load_bags(cohort, model, ids, dev):
    root = HERE / "features" / cohort / model
    return {s: torch.load(root / f"{s}.pt")["x"].to(dev) for s in ids}


def train(bags, ids, y, seed, dev, method="abmil"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = next(iter(bags.values())).shape[1]
    net = make(method, d).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=HP["lr"], weight_decay=HP["wd"])
    pos = float(sum(y)); neg = len(y) - pos
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos, device=dev))
    rng = np.random.default_rng(seed)
    for _ in range(HP["epochs"]):
        net.train()
        for i in rng.permutation(len(ids)):
            x = bags[ids[i]]
            if len(x) > HP["max_patches"]:
                x = x[torch.from_numpy(rng.choice(len(x), HP["max_patches"], replace=False)).to(dev)]
            logit, _, _ = net(x.float())
            loss = lossf(logit, torch.tensor(float(y[i]), device=dev))
            opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


@torch.no_grad()
def predict(net, bags, ids):
    p, att, sc = [], {}, {}
    for s in ids:
        logit, a, raw = net(bags[s].float())
        p.append(torch.sigmoid(logit).item())
        att[s] = a.cpu().numpy()
        if raw is not None:
            sc[s] = raw.cpu().numpy()
    return np.array(p), att, sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--method", choices=["abmil", "topk"], default="abmil")
    a = ap.parse_args()
    dev = torch.device("cuda")
    out = HERE / "results" / a.model / a.method
    for c in ("tcga", "paip"):
        (out / "attention" / c).mkdir(parents=True, exist_ok=True)
        if a.method == "topk":
            (out / "scores" / c).mkdir(parents=True, exist_ok=True)

    k = pd.read_csv(SC / "kfolds_IDARS_fixed.csv")
    k = k[[(HERE / "features" / "tcga" / a.model / f"{s}.pt").exists() for s in k.WSI_Id]]
    t_ids, t_y, t_fold = list(k.WSI_Id), k.label_id.astype(int).to_numpy(), k.fold.to_numpy()
    pl = pd.read_csv(SC / "paip_78_labels.csv")
    p_ids, p_y = list(pl.WSI_Id), (pl.label == "MSIH").astype(int).to_numpy()
    print(f"{a.model} / {a.method}: TCGA {len(t_ids)} slides ({t_y.sum()} MSI-H), PAIP {len(p_ids)} ({p_y.sum()} MSI-H)", flush=True)
    tb = load_bags("tcga", a.model, t_ids, dev)
    pb = load_bags("paip", a.model, p_ids, dev)

    def keep(store, att, sc):
        for s, w in att.items():
            store[0].setdefault(s, []).append(w)
        for s, v in sc.items():
            store[1].setdefault(s, []).append(v)

    def save(store, cohort):
        for s, ws in store[0].items():
            np.save(out / "attention" / cohort / f"{s}.npy", np.mean(ws, axis=0).astype(np.float32))
        for s, vs in store[1].items():
            np.save(out / "scores" / cohort / f"{s}.npy", np.mean(vs, axis=0).astype(np.float32))

    # ---- TCGA-CV: out-of-fold, 5 seeds averaged
    oof = np.zeros((len(SEEDS), len(t_ids)))
    t_store = ({}, {})
    for f in sorted(set(t_fold)):
        tr = np.where(t_fold != f)[0]; te = np.where(t_fold == f)[0]
        for si, seed in enumerate(SEEDS):
            net = train(tb, [t_ids[i] for i in tr], t_y[tr], seed, dev, a.method)
            p, att, sc = predict(net, tb, [t_ids[i] for i in te])
            oof[si, te] = p
            keep(t_store, att, sc)
        print(f"  fold {f}: AUROC {roc_auc_score(t_y[te], oof[:, te].mean(0)):.4f}", flush=True)
    score = oof.mean(0)
    pd.DataFrame({"slide_id": t_ids, "fold": t_fold, "target": t_y, "score": score,
                  **{f"score_s{s}": oof[i] for i, s in enumerate(SEEDS)}}).to_csv(out / "tcga_oof.csv", index=False)
    save(t_store, "tcga")
    tau = float(youden_threshold(t_y, score)["tau"])

    # ---- PAIP-EV: full TCGA models, 5 seeds averaged, uniform threshold rule
    ev = np.zeros((len(SEEDS), len(p_ids)))
    p_store = ({}, {})
    for si, seed in enumerate(SEEDS):
        net = train(tb, t_ids, t_y, seed, dev, a.method)
        p, att, sc = predict(net, pb, p_ids)
        ev[si] = p
        keep(p_store, att, sc)
    ev_score = ev.mean(0)
    thr = quantile_threshold(score, tau, ev_score)
    pd.DataFrame({"slide_id": p_ids, "target": p_y, "score": ev_score, "threshold": thr,
                  **{f"score_s{s}": ev[i] for i, s in enumerate(SEEDS)}}).to_csv(out / "paip_ev.csv", index=False)
    save(p_store, "paip")

    bacc = lambda y, pr: 0.5 * ((pr[y == 1]).mean() + (1 - pr[y == 0]).mean())
    summ = {"model": a.model, "method": a.method, "hp": HP, "seeds": SEEDS,
            "tcga_cv_auroc_pooled": roc_auc_score(t_y, score),
            "tcga_cv_auroc_mean_fold": float(np.mean([roc_auc_score(t_y[t_fold == f], score[t_fold == f]) for f in sorted(set(t_fold))])),
            "tcga_cv_bacc_at_0.5": float(bacc(t_y, (score >= 0.5).astype(float))),
            "tau_tcga": tau,
            "paip_ev_auroc": roc_auc_score(p_y, ev_score), "paip_ev_threshold": thr,
            "paip_ev_bacc": float(bacc(p_y, (ev_score >= thr).astype(float)))}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()
