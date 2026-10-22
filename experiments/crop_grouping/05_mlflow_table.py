"""Crop-level grouping rows for the clean MLflow store, in the schema of nocrop_vs_fivecrop/06_mlflow_table.py.

Rows: setting (TCGA-CV, PAIP-IV, PAIP-EV) x Caption-14 / Caption-15 / TTC x encoder x head, arm = cropgroup.
Metrics from the same per-slide predictions and bootstrap (06_mlflow_table.single_arm); delta_* = paired
crop-level - published patch-level grouping (results/contrasts.csv, comparison crop-patch).

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
CROP = HERE / "runs" / "cropgroup"


def main():
    arms = {"TCGA-CV": c5.load_cv(CROP), "PAIP-IV": c5.load_iv(CROP),
            "PAIP-EV": c5.load_ev(sorted(glob.glob(str(CROP / "ev" / "*" / "ev_scores_paip.csv"))))}
    rows = []
    for setting, df in arms.items():
        t = t06.single_arm(df).assign(setting=setting, arm="cropgroup")
        rows.append(t)
        print(f"{setting:8s} cropgroup {len(t):3d} configurations, n_test {t.n_test.iloc[0]}")
    t = pd.concat(rows, ignore_index=True)
    t["aggregation"] = t.method.map(t06.SHORT)
    t["precision"] = t.model.map(t06.PRECISION)

    con = pd.read_csv(HERE / "results" / "contrasts.csv")
    con = con[(con.level == "config") & (con.comparison == "crop-patch")].copy()
    parts = con.contrast.str.split(" / ", expand=True)
    con["aggregation"], con["model"], con["classifier"] = parts[0], parts[1], parts[2]
    d = con[["setting", "aggregation", "model", "classifier", "auroc_diff", "auroc_lo", "auroc_hi", "auroc_p",
             "bacc_diff", "bacc_lo", "bacc_hi", "bacc_p"]].rename(columns={
        "auroc_diff": "delta_auroc", "auroc_lo": "delta_auroc_ci_lo", "auroc_hi": "delta_auroc_ci_hi",
        "auroc_p": "delta_auroc_p", "bacc_diff": "delta_bacc", "bacc_lo": "delta_bacc_ci_lo",
        "bacc_hi": "delta_bacc_ci_hi", "bacc_p": "delta_bacc_p"})
    t = t.merge(d, on=["setting", "aggregation", "model", "classifier"], how="left")
    assert t.delta_auroc.notna().all(), "every crop-level row must have its patch-level pair"
    t.to_csv(HERE / "results" / "mlflow_table.csv", index=False)
    print(f"wrote {HERE / 'results' / 'mlflow_table.csv'}: {len(t)} rows")


if __name__ == "__main__":
    main()
