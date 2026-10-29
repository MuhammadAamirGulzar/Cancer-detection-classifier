"""Patch-importance MIL rows for the clean MLflow store (schema of nocrop_vs_fivecrop/06_mlflow_table.py).

One row per encoder x method x setting (TCGA-CV, PAIP-EV), from results/<model>/<method>/compare.csv: the method's
own AUROC / balanced accuracy with bootstrap intervals, and delta_* = paired method - <encoder> / Averaging / LR.

Writes results/mlflow_table.csv; tools/tracking/build_clean_store.py reads it.
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
NAME = {"abmil": ("ABMIL", "attention"), "topk": ("TopK-MIL", "top-k"), "patchsum": ("PatchSum", "lin")}
N = {"TCGA-CV": (413, 60), "PAIP-EV": (78, 19)}


def main():
    rows = []
    for f in sorted((HERE / "results").glob("*/*/compare.csv")):
        model, method = f.parent.parent.name, f.parent.name
        c = pd.read_csv(f)
        agg, head = NAME[method]
        for setting, g in c.groupby("setting"):
            own = g.iloc[0]
            ref = g[g.comparison == f"{model} / Averaging / LR"].iloc[0]
            rows.append({"method": agg, "model": model, "classifier": head, "n_test": N[setting][0], "n_pos": N[setting][1],
                         "auroc": own.auroc, "auroc_ci_lo": own.auroc_lo, "auroc_ci_hi": own.auroc_hi,
                         "bacc": own.bacc, "bacc_ci_lo": own.bacc_lo, "bacc_ci_hi": own.bacc_hi,
                         "setting": setting, "arm": "mil", "aggregation": agg, "precision": "as five-crop features",
                         "delta_auroc": ref.abmil_minus_this, "delta_auroc_ci_lo": ref.diff_lo,
                         "delta_auroc_ci_hi": ref.diff_hi, "delta_auroc_p": ref.p})
    t = pd.DataFrame(rows)
    t.to_csv(HERE / "results" / "mlflow_table.csv", index=False)
    print(t[["setting", "model", "aggregation", "auroc", "bacc", "delta_auroc", "delta_auroc_p"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
