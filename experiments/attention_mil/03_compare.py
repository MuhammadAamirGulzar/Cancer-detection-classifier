"""ABMIL vs the benchmark configurations of the same encoder, TCGA-CV and PAIP-EV, paired bootstrap.

Benchmark per-slide predictions come from experiments/final_analysis/02_uncertainty.py (crop-level grouping, same
loaders) and the bootstrap is its own (class-stratified, B = 2000, same seed), so every configuration and ABMIL
are scored on the same resampled slides.

Writes results/<model>/<method>/compare.csv.

  <conda>/envs/exaonepath/python.exe experiments/attention_mil/03_compare.py --model H-Optimus-1 --method abmil
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
spec = importlib.util.spec_from_file_location("unc", REPO / "experiments" / "final_analysis" / "02_uncertainty.py")
U = importlib.util.module_from_spec(spec)
spec.loader.exec_module(U)
U.GROUPING = "crop"
SHORT = {"Averaging": "Averaging", "Caption_based_aggregation": "Caption-14",
         "Caption_based_aggregation_15_classes": "Caption-15", "Tissue_Type_Clustering": "TTC"}


def matrix(df, value):
    w = df.pivot_table(index="slide_id", columns=["method", "model", "classifier"], values=value, aggfunc="first")
    assert not w.isna().any().any(), "configurations do not share one slide set"
    y = df.drop_duplicates("slide_id").set_index("slide_id").target.reindex(w.index).astype(int).to_numpy()
    return w, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--method", choices=["abmil", "topk", "patchsum"], default="abmil")
    a = ap.parse_args()
    res = HERE / "results" / a.model / a.method
    abmil = ({"abmil": "ABMIL", "topk": "TopK-MIL", "patchsum": "PatchSum-LR"}[a.method], a.model, a.method)

    t = U.load_oof("TCGA_Results")
    t = t[t.method.isin(U.AGG)][["method", "model", "classifier", "slide_id", "target", "score"]]
    m = pd.read_csv(res / "tcga_oof.csv").assign(method=abmil[0], model=abmil[1], classifier=abmil[2])
    t = pd.concat([t, m[["method", "model", "classifier", "slide_id", "target", "score"]]], ignore_index=True)

    e = U.load_ev("paip")
    e = e[e.method.isin(U.AGG)][["method", "model", "classifier", "slide_id", "target", "score_uniform", "threshold_uniform"]]
    m = pd.read_csv(res / "paip_ev.csv").assign(method=abmil[0], model=abmil[1], classifier=abmil[2])
    m = m.rename(columns={"score": "score_uniform", "threshold": "threshold_uniform"})
    e = pd.concat([e, m[e.columns]], ignore_index=True)

    rows = []
    for setting, df, sc, th in (("TCGA-CV", t, "score", None), ("PAIP-EV", e, "score_uniform", "threshold_uniform")):
        W, y = matrix(df, sc)
        cols = list(W.columns)
        ix = {c: j for j, c in enumerate(cols)}
        S = W.to_numpy()
        if th:
            T = matrix(df, th)[0].reindex(index=W.index, columns=W.columns).to_numpy()
            PRED = (S >= T).astype(float)
        else:
            PRED = (S >= 0.5).astype(float)
        bacc = lambda P, yy: 0.5 * (P[yy == 1].mean(0) + (1 - P[yy == 0]).mean(0))
        A0, B0 = U.auroc_cols(S, y), bacc(PRED, y)
        gen = U.boot_indices(y, np.array(list(W.index)), np.random.default_rng(U.SEED))
        A = np.empty((U.B, len(cols))); Bb = np.empty_like(A)
        for b in range(U.B):
            i = next(gen)
            A[b], Bb[b] = U.auroc_cols(S[i], y[i]), bacc(PRED[i], y[i])

        enc = [c for c in cols if c[1] == a.model and c[0] in U.AGG]
        best = max(enc, key=lambda c: A0[ix[c]])
        groups = {
            abmil[0]: [abmil],
            "TCGA-selected: H-Optimus-1 / TTC / LR": [("Tissue_Type_Clustering", "H-Optimus-1", "lin")],
            f"{a.model} / Averaging / LR": [("Averaging", a.model, "lin")],
            f"best {a.model} configuration ({SHORT[best[0]]} / {best[2]})": [best],
            f"mean of the 20 {a.model} configurations": enc,
            "mean of all 100 configurations": [c for c in cols if c[0] in U.AGG],
        }
        jab = ix[abmil]
        for name, sel in groups.items():
            j = [ix[c] for c in sel if c in ix]
            if not j:
                continue
            pa, pb = A0[j].mean(), B0[j].mean()
            lo, hi = U.pct(A[:, j].mean(1))
            blo, bhi = U.pct(Bb[:, j].mean(1))
            r = {"setting": setting, "comparison": name, "n_configs": len(j), "auroc": pa, "auroc_lo": lo, "auroc_hi": hi,
                 "bacc": pb, "bacc_lo": blo, "bacc_hi": bhi}
            if name != abmil[0]:
                d = A[:, jab] - A[:, j].mean(1)
                dl, dh = U.pct(d)
                r.update({"abmil_minus_this": A0[jab] - pa, "diff_lo": dl, "diff_hi": dh, "p": min(1.0, U.p_two_sided(d))})
            rows.append(r)
        print(f"{setting}: {len(y)} slides, {len(cols)} configurations", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(res / "compare.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 60)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
