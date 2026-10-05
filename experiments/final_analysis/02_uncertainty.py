"""Confidence intervals and paired tests for all five experiments.

Inputs (all per-slide, one score per slide and configuration):
  PAIP-EV, SurGen-EV   results/ev_scores_<cohort>.csv   (uniform rule, from 01_uniform_ev_scores.py)
  TCGA-CV, SurGen-CV   <tree>/**/oof_predictions_default.csv   (pooled out-of-fold predictions)
  PAIP-IV              <tree>/**/predictions_<head>.csv         (31 provider test slides)

Method
------
Case-level, class-stratified bootstrap, B = 2000, one fixed seed. The SAME resampled
cases are used for every configuration of an experiment, so differences between
configurations are paired. SurGen has cases with two slides; a resampled case brings
all its slides. TCGA, PAIP: one slide per case.

Per resample the AUROC of every configuration is computed at once from ranks
(Mann-Whitney, average ranks for ties - identical to sklearn's roc_auc_score).

Reported:
  ci_per_config.csv   AUROC with 95% percentile interval for every configuration; for the two
                      external experiments also balanced accuracy, sensitivity and specificity at
                      the uniform threshold (the threshold is held fixed across resamples).
  contrasts.csv       paired differences of MEAN AUROC between groups of configurations
                      (aggregation method vs Averaging; encoders; heads), with interval and a
                      two-sided bootstrap p-value. These test the claims the paper makes.
  prespecified.csv    the configuration TCGA-CV selects, evaluated on the external cohorts.
  selection_transfer.csv  how well TCGA-CV ranking predicts external ranking.

Notes that matter when reading the numbers
  * TCGA-CV / SurGen-CV AUROC here is the POOLED out-of-fold AUROC. The workbook reports the mean
    of the four per-fold AUROCs; both are given (auroc, auroc_mean_fold).
  * External AUROC for ANN and RF is the AUROC of the 5-seed mean probability (one prediction per
    slide). The workbook reports the mean of five single-seed AUROCs; ev_uniform_summary.csv has
    both. LR, KNN and ProtoNet are deterministic, so the two coincide for them.
  * Intervals are not corrected for multiplicity. 'Best of 100' rows are optimistic by selection
    and their intervals do not account for that; prespecified.csv is the unbiased counterpart.

Run:  <conda>/envs/exaonepath/python.exe experiments/final_analysis/02_uncertainty.py
"""

from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
SC = HERE.parents[1] / "slide_classification"
OUT = HERE / "results"
B = 2000
SEED = 20261005
HEADS = ["lin", "ann", "knn", "proto", "rf"]
AGG = ["Averaging", "Caption_based_aggregation", "Caption_based_aggregation_15_classes", "Tissue_Type_Clustering"]
SHORT = {"Averaging": "Averaging", "Caption_based_aggregation": "Caption-14",
         "Caption_based_aggregation_15_classes": "Caption-15", "Tissue_Type_Clustering": "TTC",
         "TITAN": "TITAN", "PRISM": "PRISM"}
ENC = ["H-Optimus-1", "UNI2", "Virchow2", "Conch1_5", "ConchV1"]
_CASE = re.compile(r"^(.*)_\d+$")


def surgen_case(s):
    m = _CASE.match(s)
    return m.group(1) if m else s


# ---------------------------------------------------------------- loading
def _combo_from_path(p: Path, tree: str):
    parts = p.relative_to(SC / tree).parts          # (method, model, 1-MSIH, Output, file) or (PRISM, 1-MSIH, ...)
    return (parts[0], parts[0]) if parts[1].startswith("1-") else (parts[0], parts[1])


def load_oof(tree: str) -> pd.DataFrame:
    frames = []
    for f in glob.glob(str(SC / tree / "**" / "oof_predictions_default.csv"), recursive=True):
        method, model = _combo_from_path(Path(f), tree)
        d = pd.read_csv(f)
        d["method"], d["model"] = method, model
        frames.append(d.rename(columns={"WSI_ID": "slide_id", "prob_pos": "score"}))
    return pd.concat(frames, ignore_index=True)


def load_paip_iv() -> pd.DataFrame:
    frames = []
    for f in glob.glob(str(SC / "PAIP_IV_Results" / "**" / "predictions_*.csv"), recursive=True):
        p = Path(f)
        method, model = _combo_from_path(p, "PAIP_IV_Results")
        d = pd.read_csv(f)
        d["method"], d["model"], d["classifier"] = method, model, p.stem.replace("predictions_", "")
        frames.append(d.rename(columns={"WSI_ID": "slide_id", "prob_pos": "score"}))
    return pd.concat(frames, ignore_index=True)


def load_ev(cohort: str) -> pd.DataFrame:
    return pd.read_csv(OUT / ("ev_scores_%s.csv" % cohort))


# ---------------------------------------------------------------- core
def to_matrix(df: pd.DataFrame, value: str):
    """slides x configurations matrix, plus labels. Fails loudly on any gap."""
    w = df.pivot_table(index="slide_id", columns=["method", "model", "classifier"], values=value, aggfunc="first")
    if w.isna().any().any():
        raise ValueError("configurations do not share one slide set")
    y = df.drop_duplicates("slide_id").set_index("slide_id")["target"].reindex(w.index).astype(int)
    return w, y


def auroc_cols(S: np.ndarray, y: np.ndarray) -> np.ndarray:
    r = rankdata(S, axis=0)
    n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y == 1].sum(axis=0) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def boot_indices(y: np.ndarray, cases: np.ndarray, rng):
    """Class-stratified resample of CASES; returns slide indices (with multiplicity)."""
    uc, inv = np.unique(cases, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(uc))]
    lab = np.array([y[m].max() for m in members])
    pos, neg = np.where(lab == 1)[0], np.where(lab == 0)[0]
    while True:
        pick = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        yield np.concatenate([members[i] for i in pick])


def pct(a, axis=0):
    return np.percentile(a, 2.5, axis=axis), np.percentile(a, 97.5, axis=axis)


def p_two_sided(d: np.ndarray) -> float:
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(max(p, 1.0 / len(d)))


def run_experiment(name: str, df: pd.DataFrame, case_fn, score_col="score", thr_col=None, extra_col=None):
    W, y = to_matrix(df, score_col)
    yv = y.values
    cases = np.array([case_fn(s) for s in W.index])
    S = W.values
    cols = list(W.columns)
    point = auroc_cols(S, yv)

    have_thr = thr_col is not None
    if have_thr:
        T = df.pivot_table(index="slide_id", columns=["method", "model", "classifier"], values=thr_col,
                           aggfunc="first").reindex(index=W.index, columns=W.columns).values
        PRED = (S >= T).astype(float)
    if extra_col:
        E = to_matrix(df, extra_col)[0].reindex(index=W.index, columns=W.columns).values
        e_point = auroc_cols(E, yv)

    rng = np.random.default_rng(SEED)
    gen = boot_indices(yv, cases, rng)
    A = np.empty((B, S.shape[1]))
    if have_thr:
        SE, SP = np.empty_like(A), np.empty_like(A)
    if extra_col:
        AE = np.empty_like(A)
    for b in range(B):
        idx = next(gen)
        yb = yv[idx]
        A[b] = auroc_cols(S[idx], yb)
        if have_thr:
            pb = PRED[idx]
            SE[b] = pb[yb == 1].mean(axis=0)
            SP[b] = 1.0 - pb[yb == 0].mean(axis=0)
        if extra_col:
            AE[b] = auroc_cols(E[idx], yb)

    lo, hi = pct(A)
    rows = []
    for j, (method, model, kind) in enumerate(cols):
        r = {"experiment": name, "method": method, "model": model, "classifier": kind,
             "n_slides": len(yv), "n_cases": len(np.unique(cases)), "n_pos_slides": int(yv.sum()),
             "auroc": point[j], "auroc_lo": lo[j], "auroc_hi": hi[j]}
        if have_thr:
            se = PRED[yv == 1, j].mean(); sp = 1 - PRED[yv == 0, j].mean()
            bl, bh = pct((SE[:, j] + SP[:, j]) / 2)
            sl, sh = pct(SE[:, j]); pl, ph = pct(SP[:, j])
            r.update({"bacc": (se + sp) / 2, "bacc_lo": bl, "bacc_hi": bh, "sens": se, "sens_lo": sl, "sens_hi": sh,
                      "spec": sp, "spec_lo": pl, "spec_hi": ph, "threshold": T[0, j],
                      "n_pos_pred": int(PRED[:, j].sum())})
        if extra_col:
            el, eh = pct(AE[:, j])
            r.update({"auroc_fold_ensemble": e_point[j], "auroc_fold_ensemble_lo": el, "auroc_fold_ensemble_hi": eh})
        rows.append(r)
    return pd.DataFrame(rows), cols, point, A, (e_point, AE) if extra_col else None


def contrasts(name, cols, point, A, extra=None):
    """Paired differences of mean AUROC between groups of the 100 aggregation configurations."""
    ix = {c: j for j, c in enumerate(cols)}
    out = []

    def mean_of(sel):
        j = [ix[c] for c in sel]
        return point[j].mean(), A[:, j].mean(axis=1)

    def add(family, label, sel_a, sel_b, note=""):
        if not sel_a or not sel_b:
            return
        pa, ba = mean_of(sel_a); pb, bb = mean_of(sel_b)
        d = ba - bb
        lo, hi = pct(d)
        out.append({"experiment": name, "family": family, "contrast": label, "mean_a": pa, "mean_b": pb,
                    "difference": pa - pb, "lo": lo, "hi": hi, "p_boot": p_two_sided(d),
                    "n_configs_a": len(sel_a), "n_configs_b": len(sel_b), "note": note})

    def cells(methods=AGG, encs=ENC, heads=HEADS):
        return [(m, e, h) for m in methods for e in encs for h in heads if (m, e, h) in ix]

    # 1. aggregation method vs Averaging (the paper's main claim)
    for m in AGG[1:]:
        add("method_vs_averaging", "%s - Averaging" % SHORT[m], cells([m]), cells(["Averaging"]), "all 5 encoders x 5 heads")
    add("method_vs_averaging", "mean of the three semantic methods - Averaging", cells(AGG[1:]), cells(["Averaging"]),
        "all 5 encoders x 5 heads")
    for h in HEADS:
        add("method_vs_averaging_by_head", "semantic - Averaging | head=%s" % h, cells(AGG[1:], heads=[h]),
            cells(["Averaging"], heads=[h]), "5 encoders")
    for e in ENC:
        add("method_vs_averaging_by_encoder", "semantic - Averaging | encoder=%s" % e, cells(AGG[1:], encs=[e]),
            cells(["Averaging"], encs=[e]), "5 heads")
    add("method_pairs", "Caption-15 - Caption-14", cells([AGG[2]]), cells([AGG[1]]))
    add("method_pairs", "Caption-15 - TTC", cells([AGG[2]]), cells([AGG[3]]))
    add("method_pairs", "Caption-14 - TTC", cells([AGG[1]]), cells([AGG[3]]))

    # 2. encoders (mean over 4 methods x 5 heads), each against the best by point estimate
    em = {e: mean_of(cells(encs=[e]))[0] for e in ENC if cells(encs=[e])}
    best_e = max(em, key=em.get)
    for e in ENC:
        if e != best_e and e in em:
            add("encoder_vs_best", "%s - %s" % (best_e, e), cells(encs=[best_e]), cells(encs=[e]), "4 methods x 5 heads")

    # 3. heads (mean over 4 methods x 5 encoders), each against the best
    hm = {h: mean_of(cells(heads=[h]))[0] for h in HEADS}
    best_h = max(hm, key=hm.get)
    for h in HEADS:
        if h != best_h:
            add("head_vs_best", "%s - %s" % (best_h, h), cells(heads=[best_h]), cells(heads=[h]), "4 methods x 5 encoders")

    # 4. slide encoders vs the best aggregated configuration built on comparable patch features
    for base in (("TITAN", "Conch1_5"), ("PRISM", "PRISM")):
        slide_cfg = [(base[0], base[1], h) for h in HEADS if (base[0], base[1], h) in ix]
        if slide_cfg:
            add("slide_encoder", "mean of 100 aggregation configs - %s (mean over heads)" % base[0], cells(), slide_cfg)
            if base[0] == "TITAN":
                add("slide_encoder", "Conch1_5 aggregations (4 methods x 5 heads) - TITAN (5 heads)",
                    cells(encs=["Conch1_5"]), slide_cfg, "same CONCH 1.5 patch features")

    # 5. fold ensemble vs the single full-data model (external experiments only)
    if extra is not None:
        e_point, AE = extra
        j = [ix[c] for c in cells()]
        d = AE[:, j].mean(axis=1) - A[:, j].mean(axis=1)
        lo, hi = pct(d)
        out.append({"experiment": name, "family": "ensemble", "contrast": "4-fold ensemble - TCGA-full model (mean AUROC)",
                    "mean_a": e_point[j].mean(), "mean_b": point[j].mean(), "difference": e_point[j].mean() - point[j].mean(),
                    "lo": lo, "hi": hi, "p_boot": p_two_sided(d), "n_configs_a": len(j), "n_configs_b": len(j),
                    "note": "KNN: fold models at their grid-searched k vs full model at k=35"})
        for h in HEADS:
            j = [ix[c] for c in cells(heads=[h])]
            d = AE[:, j].mean(axis=1) - A[:, j].mean(axis=1)
            lo, hi = pct(d)
            out.append({"experiment": name, "family": "ensemble_by_head", "contrast": "4-fold ensemble - TCGA-full | head=%s" % h,
                        "mean_a": e_point[j].mean(), "mean_b": point[j].mean(), "difference": e_point[j].mean() - point[j].mean(),
                        "lo": lo, "hi": hi, "p_boot": p_two_sided(d), "n_configs_a": len(j), "n_configs_b": len(j), "note": ""})
    return pd.DataFrame(out)


def mean_fold_auroc(df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score
    g = df.groupby(["method", "model", "classifier", "fold"]).apply(
        lambda d: roc_auc_score(d.target, d.score), include_groups=False).rename("a").reset_index()
    return g.groupby(["method", "model", "classifier"]).a.mean().rename("auroc_mean_fold").reset_index()


def main():
    ident = lambda s: s
    tcga = load_oof("TCGA_Results")
    surcv = load_oof("SurGen_Results")
    paipiv = load_paip_iv()
    paipev, surev = load_ev("paip"), load_ev("surgen")

    specs = [("TCGA-CV", tcga, ident, "score", None, None),
             ("PAIP-IV", paipiv, ident, "score", None, None),
             ("PAIP-EV", paipev, ident, "score_uniform", "threshold_uniform", "score_fold_ensemble"),
             ("SurGen-CV", surcv, surgen_case, "score", None, None),
             ("SurGen-EV", surev, surgen_case, "score_uniform", "threshold_uniform", "score_fold_ensemble")]
    ci, con, keep = [], [], {}
    for name, df, case_fn, sc, th, ex in specs:
        t, cols, point, A, extra = run_experiment(name, df, case_fn, sc, th, ex)
        if name in ("TCGA-CV", "SurGen-CV"):
            t = t.merge(mean_fold_auroc(df), on=["method", "model", "classifier"], how="left")
        ci.append(t)
        con.append(contrasts(name, cols, point, A, extra))
        keep[name] = (cols, point, A)
        agg = t[t.method.isin(AGG)]
        b = agg.loc[agg.auroc.idxmax()]
        print("%-10s n=%d slides / %d cases / %d positive slides | best of %d: %.3f [%.3f, %.3f] %s / %s / %s" % (
            name, b.n_slides, b.n_cases, b.n_pos_slides, len(agg), b.auroc, b.auroc_lo, b.auroc_hi,
            SHORT[b.method], b.model, b.classifier))
    ci = pd.concat(ci, ignore_index=True)
    con = pd.concat(con, ignore_index=True)
    ci.to_csv(OUT / "ci_per_config.csv", index=False, float_format="%.6g")
    con.to_csv(OUT / "contrasts.csv", index=False, float_format="%.6g")

    # ---------------- the configuration TCGA-CV selects, taken to the external cohorts
    t = ci[(ci.experiment == "TCGA-CV") & ci.method.isin(AGG)]
    rules = {"best TCGA-CV AUROC (mean of folds, as published)": t.loc[t.auroc_mean_fold.idxmax()],
             "best TCGA-CV AUROC (pooled out-of-fold)": t.loc[t.auroc.idxmax()]}
    for e in ENC:
        te = t[t.model == e]
        rules["best TCGA-CV AUROC within encoder %s" % e] = te.loc[te.auroc_mean_fold.idxmax()]
    pres = []
    for rule, sel in rules.items():
        key = (sel.method, sel.model, sel.classifier)
        row = {"selection_rule": rule, "method": SHORT[sel.method], "model": sel.model, "classifier": sel.classifier,
               "tcga_cv_auroc_mean_fold": sel.auroc_mean_fold, "tcga_cv_auroc_pooled": sel.auroc,
               "tcga_cv_lo": sel.auroc_lo, "tcga_cv_hi": sel.auroc_hi}
        for ev in ("PAIP-EV", "SurGen-EV"):
            e_all = ci[(ci.experiment == ev) & ci.method.isin(AGG)].reset_index(drop=True)
            r = e_all[(e_all.method == key[0]) & (e_all.model == key[1]) & (e_all.classifier == key[2])].iloc[0]
            tag = ev.replace("-", "_").lower()
            row.update({tag + "_auroc": r.auroc, tag + "_auroc_lo": r.auroc_lo, tag + "_auroc_hi": r.auroc_hi,
                        tag + "_rank_of_100": int((e_all.auroc > r.auroc).sum()) + 1,
                        tag + "_bacc": r.bacc, tag + "_bacc_lo": r.bacc_lo, tag + "_bacc_hi": r.bacc_hi,
                        tag + "_sens": r.sens, tag + "_spec": r.spec})
            # paired difference to the best external configuration (same resamples)
            cols, point, A = keep[ev]
            jx = {c: j for j, c in enumerate(cols)}
            best = e_all.loc[e_all.auroc.idxmax()]
            d = A[:, jx[(best.method, best.model, best.classifier)]] - A[:, jx[key]]
            lo, hi = pct(d)
            row.update({tag + "_gap_to_best": best.auroc - r.auroc, tag + "_gap_lo": lo, tag + "_gap_hi": hi})
        pres.append(row)
    pd.DataFrame(pres).to_csv(OUT / "prespecified.csv", index=False, float_format="%.6g")

    # ---------------- does TCGA-CV ranking transfer?
    tr = []
    base = t.set_index(["method", "model", "classifier"])
    for other in ("PAIP-IV", "PAIP-EV", "SurGen-CV", "SurGen-EV"):
        o = ci[(ci.experiment == other) & ci.method.isin(AGG)].set_index(["method", "model", "classifier"])
        j = base.join(o[["auroc"]], rsuffix="_other", how="inner")
        rho = spearmanr(j.auroc_mean_fold, j.auroc_other).correlation
        top10 = set(j.auroc_mean_fold.nlargest(10).index) & set(j.auroc_other.nlargest(10).index)
        tr.append({"reference": "TCGA-CV (mean of folds)", "other": other, "n_configs": len(j), "spearman": rho,
                   "top10_overlap": len(top10)})
    pd.DataFrame(tr).to_csv(OUT / "selection_transfer.csv", index=False, float_format="%.4g")
    print("wrote ci_per_config.csv (%d rows), contrasts.csv (%d rows), prespecified.csv, selection_transfer.csv"
          % (len(ci), len(con)))


if __name__ == "__main__":
    main()
