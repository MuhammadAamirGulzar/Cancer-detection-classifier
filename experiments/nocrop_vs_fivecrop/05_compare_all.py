"""Step 5 - no-crop vs five-crop, TCGA-CV, PAIP-IV and PAIP-EV, 5 encoders x 4 aggregations x 5 heads.

Five-crop arm = the published results:
  TCGA-CV  slide_classification/TCGA_Results/**/oof_predictions_default.csv   (pooled out-of-fold)
  PAIP-IV  slide_classification/PAIP_IV_Results/**/predictions_<head>.csv     (31 provider test slides)
  PAIP-EV  experiments/final_analysis/results/ev_scores_paip.csv              (uniform rule = published scheme)
No-crop arm = the same files from runs/nocrop/ (04_run_nocrop_pipeline.py).

Per setting, both arms are paired slide by slide and bootstrapped together with the final_analysis
bootstrap (02_uncertainty.py, imported: class-stratified, B = 2000, same seed, one slide per case).

Metrics, both on the same per-slide predictions:
  AUROC   threshold-free
  BalAcc  at each setting's published operating point: TCGA-CV and PAIP-IV the head's own prediction
          (0.5; RF 0.3), PAIP-EV the uniform threshold (rate-matched quantile, kNN refit at k = 35)

Writes results/all/: per_config.csv (both arms, with intervals), contrasts.csv (paired no-crop - five-crop
differences: per configuration, per encoder, per aggregation, per head, all 20 configurations).
"""

from __future__ import annotations

import glob
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SC = REPO / "slide_classification"
NOCROP = HERE / "runs" / "nocrop"
OUT = HERE / "results" / "all"

_spec = importlib.util.spec_from_file_location("unc", REPO / "experiments" / "final_analysis" / "02_uncertainty.py")
U = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(U)

METHODS = ["Averaging", "Caption_based_aggregation", "Caption_based_aggregation_15_classes", "Tissue_Type_Clustering"]
SHORT = {"Averaging": "Averaging", "Caption_based_aggregation": "Caption-14",
         "Caption_based_aggregation_15_classes": "Caption-15", "Tissue_Type_Clustering": "TTC"}
MODELS = ["UNI2", "H-Optimus-1", "ConchV1", "Conch1_5", "Virchow2"]
HEADS = U.HEADS


def _combo(path: Path, root: Path):
    parts = path.relative_to(root).parts
    if parts[1] == "PRISM":                      # slide encoder: <tree>/PRISM/1-MSIH/Output/...
        return "PRISM", "PRISM"
    return parts[1], parts[2]


def _files(root: Path, tree: str, name: str):
    return (glob.glob(str(root / tree / "*" / "*" / "1-MSIH" / "Output" / name))
            + glob.glob(str(root / tree / "PRISM" / "1-MSIH" / "Output" / name)))


def load_cv(root: Path):
    frames = []
    for f in _files(root, "TCGA_Results", "oof_predictions_default.csv"):
        m, x = _combo(Path(f), root)
        if m in METHODS and x in MODELS:
            d = pd.read_csv(f)
            frames.append(pd.DataFrame({"method": m, "model": x, "classifier": d.classifier, "slide_id": d.WSI_ID,
                                        "target": d.target, "score": d.prob_pos, "pred": d["pred_at_0.5"]}))
    return pd.concat(frames, ignore_index=True)


def load_iv(root: Path):
    frames = []
    for f in _files(root, "PAIP_IV_Results", "predictions_*.csv"):
        m, x = _combo(Path(f), root)
        if m in METHODS and x in MODELS:
            d = pd.read_csv(f)
            frames.append(pd.DataFrame({"method": m, "model": x, "classifier": Path(f).stem.replace("predictions_", ""),
                                        "slide_id": d.WSI_ID, "target": d.target, "score": d.prob_pos, "pred": d.pred}))
    return pd.concat(frames, ignore_index=True)


def load_ev(files):
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = d[d.method.isin(METHODS) & d.model.isin(MODELS)]
    return pd.DataFrame({"method": d.method, "model": d.model, "classifier": d.classifier, "slide_id": d.slide_id,
                         "target": d.target, "score": d.score_uniform,
                         "pred": (d.score_uniform >= d.threshold_uniform).astype(int)})


def bacc_cols(P, y):
    """Balanced accuracy of every column of a 0/1 prediction matrix."""
    pos, neg = y == 1, y == 0
    return 0.5 * (P[pos].mean(axis=0) + (1 - P[neg]).mean(axis=0))


def compare(setting, five, nocrop, arms=("fivecrop", "nocrop")):
    """Paired comparison of two arms; contrasts are arms[1] - arms[0]."""
    A0, A1 = arms
    five = five.assign(arm=A0)
    nocrop = nocrop.assign(arm=A1)
    both = pd.concat([five, nocrop], ignore_index=True)
    keys = ["arm", "method", "model", "classifier"]
    S = both.pivot_table(index="slide_id", columns=keys, values="score", aggfunc="first")
    Pm = both.pivot_table(index="slide_id", columns=keys, values="pred", aggfunc="first")
    # keep only configurations present in BOTH arms and slides present everywhere
    cfg = sorted({c[1:] for c in S.columns if (A0,) + c[1:] in S.columns and (A1,) + c[1:] in S.columns})
    cols = [(a,) + c for a in (A0, A1) for c in cfg]
    S, Pm = S[cols].dropna(), Pm[cols].loc[S[cols].dropna().index]
    y = both.drop_duplicates("slide_id").set_index("slide_id").target.reindex(S.index).astype(int).values
    Sv, Pv = S.values, Pm.values.astype(float)
    ix = {c: j for j, c in enumerate(cols)}

    point_a, point_b = U.auroc_cols(Sv, y), bacc_cols(Pv, y)
    rng = np.random.default_rng(U.SEED)
    gen = U.boot_indices(y, np.array(list(S.index)), rng)
    A = np.empty((U.B, len(cols)))
    Bm = np.empty_like(A)
    for b in range(U.B):
        idx = next(gen)
        A[b], Bm[b] = U.auroc_cols(Sv[idx], y[idx]), bacc_cols(Pv[idx], y[idx])

    rows = []
    for c in cols:
        j = ix[c]
        lo, hi = U.pct(A[:, j]); blo, bhi = U.pct(Bm[:, j])
        rows.append({"setting": setting, "arm": c[0], "method": SHORT[c[1]], "model": c[2], "classifier": c[3],
                     "n": len(y), "n_pos": int(y.sum()), "auroc": point_a[j], "auroc_lo": lo, "auroc_hi": hi,
                     "bacc": point_b[j], "bacc_lo": blo, "bacc_hi": bhi})

    con = []

    def add(level, label, sel):
        if not sel:
            return
        a = [ix[(A1,) + c] for c in sel]
        b = [ix[(A0,) + c] for c in sel]
        r = {"setting": setting, "level": level, "contrast": label, "n_configs": len(sel)}
        for name, pt, M in (("auroc", point_a, A), ("bacc", point_b, Bm)):
            d = M[:, a].mean(axis=1) - M[:, b].mean(axis=1)
            lo, hi = U.pct(d)
            r.update({f"{name}_{A0}": pt[b].mean(), f"{name}_{A1}": pt[a].mean(),
                      f"{name}_diff": pt[a].mean() - pt[b].mean(), f"{name}_lo": lo, f"{name}_hi": hi,
                      f"{name}_p": min(1.0, U.p_two_sided(d))})
        con.append(r)

    for c in cfg:
        add("config", f"{SHORT[c[0]]} / {c[1]} / {c[2]}", [c])
    for x in MODELS:
        add("encoder", x, [c for c in cfg if c[1] == x])
    for m in METHODS:
        add("aggregation", SHORT[m], [c for c in cfg if c[0] == m])
    for h in HEADS:
        add("head", h, [c for c in cfg if c[2] == h])
    add("all", "all configurations", cfg)
    return pd.DataFrame(rows), pd.DataFrame(con)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    settings = {
        "TCGA-CV": (load_cv(SC), load_cv(NOCROP)),
        "PAIP-IV": (load_iv(SC), load_iv(NOCROP)),
        "PAIP-EV": (load_ev([REPO / "experiments" / "final_analysis" / "results" / "ev_scores_paip.csv"]),
                    load_ev(glob.glob(str(NOCROP / "ev" / "*" / "ev_scores_paip.csv")))),
    }
    per, con = [], []
    for s, (five, nocrop) in settings.items():
        p, c = compare(s, five, nocrop)
        per.append(p)
        con.append(c)
        n_cfg = (c.level == "config").sum()
        print(f"{s}: {n_cfg} configurations paired, n = {p.n.iloc[0]} slides")
    per, con = pd.concat(per), pd.concat(con)
    per.to_csv(OUT / "per_config.csv", index=False)
    con.to_csv(OUT / "contrasts.csv", index=False)

    pd.set_option("display.width", 220)
    f3 = lambda v: f"{v:+.3f}" if isinstance(v, float) else v
    for level in ("all", "encoder", "aggregation", "head"):
        t = con[con.level == level]
        print(f"\n== no-crop - five-crop, by {level} (mean over configurations) ==")
        show = t.assign(AUROC=t.apply(lambda r: f"{r.auroc_fivecrop:.3f} -> {r.auroc_nocrop:.3f}  {r.auroc_diff:+.3f} [{r.auroc_lo:+.3f}, {r.auroc_hi:+.3f}] p={r.auroc_p:.3f}", axis=1),
                        BalAcc=t.apply(lambda r: f"{r.bacc_fivecrop:.3f} -> {r.bacc_nocrop:.3f}  {r.bacc_diff:+.3f} [{r.bacc_lo:+.3f}, {r.bacc_hi:+.3f}] p={r.bacc_p:.3f}", axis=1))
        print(show.pivot_table(index="contrast", columns="setting", values="AUROC", aggfunc="first").to_string())
        print(show.pivot_table(index="contrast", columns="setting", values="BalAcc", aggfunc="first").to_string())


if __name__ == "__main__":
    main()
