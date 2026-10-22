"""Crop-level vs patch-level grouping (and vs no-crop), TCGA-CV, PAIP-IV and PAIP-EV.

Arms (same runners, same thresholds rule, same per-slide pairing):
  patch   published five-crop results: one caption label per patch (whole patch), TTC with the top-left crop's label
  crop    runs/cropgroup (03 step): every crop grouped by its own caption / tissue label
  nocrop  experiments/nocrop_vs_fivecrop/runs/nocrop: one whole-patch feature per patch

Aggregations: Caption-14, Caption-15, Tissue-Type Clustering (Averaging does not depend on grouping).
Bootstrap and metrics come from nocrop_vs_fivecrop/05_compare_all.py (final_analysis 02_uncertainty: class-stratified,
B = 2000, same seed; AUROC, and BalAcc at each setting's published operating point).

Writes results/per_config.csv and results/contrasts.csv (column `comparison`: crop-patch, crop-nocrop).

  <conda>/envs/exaonepath/python.exe experiments/crop_grouping/04_compare.py
"""

import glob
import importlib.util
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SC = REPO / "slide_classification"
CROP = HERE / "runs" / "cropgroup"
NOCROP = REPO / "experiments" / "nocrop_vs_fivecrop" / "runs" / "nocrop"
OUT = HERE / "results"

_spec = importlib.util.spec_from_file_location("cmp05", REPO / "experiments" / "nocrop_vs_fivecrop" / "05_compare_all.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

GROUPING = ["Caption_based_aggregation", "Caption_based_aggregation_15_classes", "Tissue_Type_Clustering"]


def only(df):
    return df[df.method.isin(GROUPING)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    arms = {
        "TCGA-CV": {"patch": C.load_cv(SC), "crop": C.load_cv(CROP), "nocrop": C.load_cv(NOCROP)},
        "PAIP-IV": {"patch": C.load_iv(SC), "crop": C.load_iv(CROP), "nocrop": C.load_iv(NOCROP)},
        "PAIP-EV": {"patch": C.load_ev([REPO / "experiments" / "final_analysis" / "results" / "ev_scores_paip.csv"]),
                    "crop": C.load_ev(glob.glob(str(CROP / "ev" / "*" / "ev_scores_paip.csv"))),
                    "nocrop": C.load_ev(glob.glob(str(NOCROP / "ev" / "*" / "ev_scores_paip.csv")))},
    }
    per, con = [], []
    for setting, a in arms.items():
        for base in ("patch", "nocrop"):
            p, c = C.compare(setting, only(a[base]), only(a["crop"]), arms=(base, "crop"))
            label = f"crop-{base}"
            per.append(p.assign(comparison=label))
            con.append(c.assign(comparison=label).rename(columns=lambda k: k.replace(f"_{base}", "_base")))
            print(f"{setting} {label}: {(c.level == 'config').sum()} configurations paired, n = {p.n.iloc[0]} slides")
    per = pd.concat(per).drop_duplicates(["setting", "arm", "method", "model", "classifier"])
    con = pd.concat(con)
    per.to_csv(OUT / "per_config.csv", index=False)
    con.to_csv(OUT / "contrasts.csv", index=False)

    pd.set_option("display.width", 240)
    for comp in ("crop-patch", "crop-nocrop"):
        for level in ("all", "aggregation", "encoder", "head"):
            t = con[(con.comparison == comp) & (con.level == level)]
            fmt = lambda r, m: (f"{r[f'{m}_base']:.3f} -> {r[f'{m}_crop']:.3f}  {r[f'{m}_diff']:+.3f} "
                                f"[{r[f'{m}_lo']:+.3f}, {r[f'{m}_hi']:+.3f}] p={r[f'{m}_p']:.3f}")
            t = t.assign(AUROC=t.apply(fmt, axis=1, m="auroc"), BalAcc=t.apply(fmt, axis=1, m="bacc"))
            print(f"\n== {comp}, by {level} (mean over configurations) ==")
            print(t.pivot_table(index="contrast", columns="setting", values="AUROC", aggfunc="first").to_string())
            print(t.pivot_table(index="contrast", columns="setting", values="BalAcc", aggfunc="first").to_string())


if __name__ == "__main__":
    main()
