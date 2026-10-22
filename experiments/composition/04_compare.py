"""Composition-aware aggregation vs the same aggregation without shares, TCGA-CV, PAIP-IV and PAIP-EV.

Paired comparisons (new - base), Caption-14 / Caption-15 / TTC x 5 encoders x 5 heads:
  comp-fivecrop   comp_fivecrop (published five-crop group features + shares)  vs  published five-crop
  comp-nocrop     comp_nocrop   (no-crop group features + shares)              vs  no-crop
  only-fivecrop   comp_only     (shares alone, no embeddings)                  vs  published five-crop
Same bootstrap and operating points as nocrop_vs_fivecrop/05_compare_all.py (imported).

Writes results/per_config.csv and results/contrasts.csv (column `comparison`).

  <conda>/envs/exaonepath/python.exe experiments/composition/04_compare.py
"""

import glob
import importlib.util
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SC = REPO / "slide_classification"
RUNS = HERE / "runs"
NOCROP = REPO / "experiments" / "nocrop_vs_fivecrop" / "runs" / "nocrop"
OUT = HERE / "results"

_spec = importlib.util.spec_from_file_location("cmp05", REPO / "experiments" / "nocrop_vs_fivecrop" / "05_compare_all.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

GROUPING = ["Caption_based_aggregation", "Caption_based_aggregation_15_classes", "Tissue_Type_Clustering"]
COMPARISONS = [("comp-fivecrop", "fivecrop", "comp_fivecrop"), ("comp-nocrop", "nocrop", "comp_nocrop"),
               ("only-fivecrop", "fivecrop", "comp_only")]


def load(setting, arm):
    root = {"fivecrop": SC, "nocrop": NOCROP}.get(arm, RUNS / arm)
    if setting == "TCGA-CV":
        d = C.load_cv(root)
    elif setting == "PAIP-IV":
        d = C.load_iv(root)
    elif arm == "fivecrop":
        d = C.load_ev([REPO / "experiments" / "final_analysis" / "results" / "ev_scores_paip.csv"])
    else:
        d = C.load_ev(sorted(glob.glob(str(root / "ev" / "*" / "ev_scores_paip.csv"))))
    return d[d.method.isin(GROUPING)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per, con = [], []
    for setting in ("TCGA-CV", "PAIP-IV", "PAIP-EV"):
        cache = {}
        for label, base, new in COMPARISONS:
            for a in (base, new):
                if a not in cache:
                    cache[a] = load(setting, a)
            p, c = C.compare(setting, cache[base], cache[new], arms=(base, new))
            ren = {f"{m}_{a}": f"{m}_{tag}" for m in ("auroc", "bacc") for a, tag in ((new, "new"), (base, "base"))}
            per.append(p.assign(comparison=label))
            con.append(c.rename(columns=ren).assign(comparison=label, base=base, new=new))
            print(f"{setting} {label}: {(c.level == 'config').sum()} configurations paired, n = {p.n.iloc[0]} slides")
    per = pd.concat(per).drop_duplicates(["setting", "arm", "method", "model", "classifier"])
    con = pd.concat(con)
    per.to_csv(OUT / "per_config.csv", index=False)
    con.to_csv(OUT / "contrasts.csv", index=False)

    pd.set_option("display.width", 240)
    fmt = lambda r, m: (f"{r[f'{m}_base']:.3f} -> {r[f'{m}_new']:.3f}  {r[f'{m}_diff']:+.3f} "
                        f"[{r[f'{m}_lo']:+.3f}, {r[f'{m}_hi']:+.3f}] p={r[f'{m}_p']:.3f}")
    for label, _, _ in COMPARISONS:
        for level in ("all", "aggregation", "encoder", "head"):
            t = con[(con.comparison == label) & (con.level == level)]
            t = t.assign(AUROC=t.apply(fmt, axis=1, m="auroc"), BalAcc=t.apply(fmt, axis=1, m="bacc"))
            print(f"\n== {label}, by {level} (mean over configurations) ==")
            print(t.pivot_table(index="contrast", columns="setting", values="AUROC", aggfunc="first").to_string())
            print(t.pivot_table(index="contrast", columns="setting", values="BalAcc", aggfunc="first").to_string())


if __name__ == "__main__":
    main()
