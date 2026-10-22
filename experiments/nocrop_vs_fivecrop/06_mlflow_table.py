"""Step 6 - one tidy table of every configuration to put into the clean MLflow store.

Rows: setting (TCGA-CV, PAIP-IV, PAIP-EV) x arm (fivecrop = published, nocrop = this experiment) x
aggregation x encoder x head. The five-crop arm also carries the two slide encoders (TITAN on CONCH v1.5,
PRISM), which have no no-crop counterpart.

Metrics per row, from the same per-slide predictions and bootstrap as 05_compare_all.py:
  auroc, auroc_ci_lo, auroc_ci_hi      pooled out-of-fold (CV) / test set (IV) / external cohort (EV)
  bacc,  bacc_ci_lo,  bacc_ci_hi       at the setting's published operating point
No-crop rows also get the paired difference to the five-crop row (delta_*, from results/all/contrasts.csv).

Writes results/all/mlflow_table.csv. Run after 05_compare_all.py.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("c5", HERE / "05_compare_all.py")
c5 = importlib.util.module_from_spec(spec)
sys.modules["c5"] = c5
spec.loader.exec_module(c5)
U = c5.U

ALL_METHODS = c5.METHODS + ["TITAN", "PRISM"]
SHORT = {**c5.SHORT, "TITAN": "TITAN", "PRISM": "PRISM"}
PRECISION = {"UNI2": "fp32", "H-Optimus-1": "fp16-autocast", "ConchV1": "fp32", "Conch1_5": "fp32",
             "Virchow2": "fp16-autocast"}


def single_arm(df):
    """AUROC / BalAcc with intervals for every configuration of one arm."""
    keys = ["method", "model", "classifier"]
    S = df.pivot_table(index="slide_id", columns=keys, values="score", aggfunc="first").dropna()
    Pm = df.pivot_table(index="slide_id", columns=keys, values="pred", aggfunc="first").loc[S.index, S.columns]
    y = df.drop_duplicates("slide_id").set_index("slide_id").target.reindex(S.index).astype(int).values
    Sv, Pv = S.values, Pm.values.astype(float)
    pa, pb = U.auroc_cols(Sv, y), c5.bacc_cols(Pv, y)
    gen = U.boot_indices(y, np.array(list(S.index)), np.random.default_rng(U.SEED))
    A, Bm = np.empty((U.B, Sv.shape[1])), np.empty((U.B, Sv.shape[1]))
    for b in range(U.B):
        i = next(gen)
        A[b], Bm[b] = U.auroc_cols(Sv[i], y[i]), c5.bacc_cols(Pv[i], y[i])
    alo, ahi = U.pct(A)
    blo, bhi = U.pct(Bm)
    return pd.DataFrame([{"method": c[0], "model": c[1], "classifier": c[2], "n_test": len(y), "n_pos": int(y.sum()),
                          "auroc": pa[j], "auroc_ci_lo": alo[j], "auroc_ci_hi": ahi[j],
                          "bacc": pb[j], "bacc_ci_lo": blo[j], "bacc_ci_hi": bhi[j]}
                         for j, c in enumerate(S.columns)])


def main():
    c5.METHODS[:] = ALL_METHODS                                  # loaders keep TITAN / PRISM too
    c5.MODELS[:] = c5.MODELS + ["PRISM"]
    ev_pub = c5.REPO / "experiments" / "final_analysis" / "results" / "ev_scores_paip.csv"
    arms = {
        ("TCGA-CV", "fivecrop"): c5.load_cv(c5.SC), ("TCGA-CV", "nocrop"): c5.load_cv(c5.NOCROP),
        ("PAIP-IV", "fivecrop"): c5.load_iv(c5.SC), ("PAIP-IV", "nocrop"): c5.load_iv(c5.NOCROP),
        ("PAIP-EV", "fivecrop"): c5.load_ev([ev_pub]),
        ("PAIP-EV", "nocrop"): c5.load_ev(sorted(c5.NOCROP.glob("ev/*/ev_scores_paip.csv"))),
    }
    rows = []
    for (setting, arm), df in arms.items():
        t = single_arm(df).assign(setting=setting, arm=arm)
        rows.append(t)
        print(f"{setting:8s} {arm:8s} {len(t):3d} configurations, n_test {t.n_test.iloc[0]}")
    t = pd.concat(rows, ignore_index=True)
    t["aggregation"] = t.method.map(SHORT)
    t["precision"] = t.model.map(PRECISION).fillna("n/a")

    con = pd.read_csv(c5.OUT / "contrasts.csv")
    con = con[con.level == "config"].copy()
    parts = con.contrast.str.split(" / ", expand=True)
    con["aggregation"], con["model"], con["classifier"] = parts[0], parts[1], parts[2]
    d = con[["setting", "aggregation", "model", "classifier", "auroc_diff", "auroc_lo", "auroc_hi", "auroc_p",
             "bacc_diff", "bacc_lo", "bacc_hi", "bacc_p"]].rename(columns={
        "auroc_diff": "delta_auroc", "auroc_lo": "delta_auroc_ci_lo", "auroc_hi": "delta_auroc_ci_hi",
        "auroc_p": "delta_auroc_p", "bacc_diff": "delta_bacc", "bacc_lo": "delta_bacc_ci_lo",
        "bacc_hi": "delta_bacc_ci_hi", "bacc_p": "delta_bacc_p"}).assign(arm="nocrop")
    t = t.merge(d, on=["setting", "arm", "aggregation", "model", "classifier"], how="left")
    t.to_csv(c5.OUT / "mlflow_table.csv", index=False)
    print(f"wrote {c5.OUT / 'mlflow_table.csv'}: {len(t)} rows")


if __name__ == "__main__":
    main()
