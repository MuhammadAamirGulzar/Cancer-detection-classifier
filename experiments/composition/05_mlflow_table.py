"""Composition rows for the clean MLflow store, in the schema of nocrop_vs_fivecrop/06_mlflow_table.py.

Rows: setting x arm (comp_fivecrop, comp_nocrop, comp_only) x Caption-14 / Caption-15 / TTC x encoder x head.
Metrics from the same per-slide predictions and bootstrap (06_mlflow_table.single_arm); delta_* = paired
difference to the arm's base (results/contrasts.csv): comp_fivecrop and comp_only vs published five-crop,
comp_nocrop vs no-crop.

Writes results/mlflow_table.csv. Run after 04_compare.py; tools/tracking/build_clean_store.py reads it.
"""

import glob
import importlib.util
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
spec = importlib.util.spec_from_file_location("t06", REPO / "experiments" / "nocrop_vs_fivecrop" / "06_mlflow_table.py")
t06 = importlib.util.module_from_spec(spec)
sys.modules["t06"] = t06
spec.loader.exec_module(t06)
c5 = t06.c5
COMPARISON = {"comp_fivecrop": "comp-fivecrop", "comp_nocrop": "comp-nocrop", "comp_only": "only-fivecrop"}


def main():
    rows = []
    for arm in COMPARISON:
        root = HERE / "runs" / arm
        loaded = {"TCGA-CV": c5.load_cv(root), "PAIP-IV": c5.load_iv(root),
                  "PAIP-EV": c5.load_ev(sorted(glob.glob(str(root / "ev" / "*" / "ev_scores_paip.csv"))))}
        for setting, df in loaded.items():
            t = t06.single_arm(df).assign(setting=setting, arm=arm)
            rows.append(t)
            print(f"{setting:8s} {arm:14s} {len(t):3d} configurations, n_test {t.n_test.iloc[0]}")
    t = pd.concat(rows, ignore_index=True)
    t["aggregation"] = t.method.map(t06.SHORT)
    t["precision"] = t.model.map(t06.PRECISION)

    con = pd.read_csv(HERE / "results" / "contrasts.csv")
    con = con[con.level == "config"].copy()
    parts = con.contrast.str.split(" / ", expand=True)
    con["aggregation"], con["model"], con["classifier"] = parts[0], parts[1], parts[2]
    con["arm"] = con.new
    d = con[["setting", "arm", "aggregation", "model", "classifier", "auroc_diff", "auroc_lo", "auroc_hi", "auroc_p",
             "bacc_diff", "bacc_lo", "bacc_hi", "bacc_p"]].rename(columns={
        "auroc_diff": "delta_auroc", "auroc_lo": "delta_auroc_ci_lo", "auroc_hi": "delta_auroc_ci_hi",
        "auroc_p": "delta_auroc_p", "bacc_diff": "delta_bacc", "bacc_lo": "delta_bacc_ci_lo",
        "bacc_hi": "delta_bacc_ci_hi", "bacc_p": "delta_bacc_p"})
    t = t.merge(d, on=["setting", "arm", "aggregation", "model", "classifier"], how="left")
    assert t.delta_auroc.notna().all(), "every composition row must have its paired base"
    t.to_csv(HERE / "results" / "mlflow_table.csv", index=False)
    print(f"wrote {HERE / 'results' / 'mlflow_table.csv'}: {len(t)} rows")


if __name__ == "__main__":
    main()
