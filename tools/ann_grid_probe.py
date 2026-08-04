"""Quantify what widening the ANN hyperparameter grid buys (evidence, not a guess).

Why this is a legitimate change now, and was not before
------------------------------------------------------
Under the pre-remediation code the ANN grid was selected on ``ann_macro_f1``,
computed on the **test** set. Widening that grid would simply have bought more
optimistic bias - more configurations to pick the luckiest from. Since Task 1.1,
selection is on **validation** macro-F1, so a wider search can only help by
finding a genuinely better model; the test set still never participates.

The evidence that the current grid truncates
--------------------------------------------
Across 22 combinations x 4 folds = 88 validation-selected configurations from the
corrected TCGA-CV run:

    h1 = 256 (the grid maximum) chosen in 54/88  (61%)
    h2 = 128 (the grid maximum) chosen in 57/88  (65%)

An optimum pressed against the upper boundary in two thirds of cases is the
classic sign the search space is too small.

This script measures, per combination, what the wider grid actually returns -
selecting on validation in both arms, and reporting the test metrics of whatever
each arm selected. A wider grid is only worth adopting if the *test* metric of
the *validation-selected* model improves.

Run:  python tools/ann_grid_probe.py [--combos M/N,M/N] [--folds 1,2,3,4]
"""

from __future__ import annotations

import argparse
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "slide_classification"))

import data_layer as dl  # noqa: E402
from eval_patch_features.ann import eval_ANN  # noqa: E402

CURRENT = {"hidden_dim1": [128, 256], "hidden_dim2": [64, 128], "max_iter": [500]}
WIDER = {"hidden_dim1": [128, 256, 512], "hidden_dim2": [64, 128, 256],
         "max_iter": [500, 1000]}
SEED = 42


def run_grid(grid, coh, splits, fold):
    """Select on VALIDATION macro-F1; report the selected model's TEST metrics."""
    tr, va, te = dl.cv_rotation(splits, fold)
    trf, trl, _ = coh.subset(tr)
    vaf, val, _ = coh.subset(va)
    tef, tel, _ = coh.subset(te)

    best = None
    n = 0
    for h1, h2, mi in product(grid["hidden_dim1"], grid["hidden_dim2"], grid["max_iter"]):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        m, _ = eval_ANN(fold=fold, train_feats=trf, train_labels=trl,
                        valid_feats=vaf, valid_labels=val,
                        test_feats=tef, test_labels=tel,
                        input_dim=coh.dim, hidden_dim=h1, hidden_dim2=h2,
                        max_iter=mi, verbose=False)
        n += 1
        if best is None or m["val_ann_macro_f1"] > best["val"]:
            best = {"val": m["val_ann_macro_f1"], "h1": h1, "h2": h2, "max_iter": mi,
                    "bacc": m["ann_bacc"], "auroc": m["ann_auroc"],
                    "macro_f1": m["ann_macro_f1"], "acc": m["ann_acc"]}
    best["n_configs"] = n
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combos",
                    default="Tissue_Type_Clustering/H-Optimus-1,"
                            "Caption_based_aggregation/Conch1_5,"
                            "Averaging/UNI2")
    ap.add_argument("--folds", default="1,2,3,4")
    args = ap.parse_args()
    folds = [int(f) for f in args.folds.split(",")]

    rows = []
    for combo in args.combos.split(","):
        method, model = combo.split("/")
        coh = dl.load_cohort("tcga", method, model, verbose=False)
        splits = dl.build_splits(coh.ids, dl.load_fold_map("tcga"))
        print(f"\n{'=' * 96}\n{method}/{model}  (D={coh.dim})\n{'=' * 96}")
        print(f"  {'fold':<6}{'grid':<9}{'selected':<18}{'val_F1':>9}"
              f"{'test_bacc':>11}{'test_auroc':>12}{'test_F1':>10}{'secs':>8}")
        for fold in folds:
            for name, grid in (("current", CURRENT), ("wider", WIDER)):
                t0 = time.perf_counter()
                b = run_grid(grid, coh, splits, fold)
                dt = time.perf_counter() - t0
                print(f"  {fold:<6}{name:<9}"
                      f"{f'{b[chr(104)+chr(49)]}/{b[chr(104)+chr(50)]}/{b[chr(109)+chr(97)+chr(120)+chr(95)+chr(105)+chr(116)+chr(101)+chr(114)]}':<18}"
                      f"{b['val']:>9.4f}{b['bacc']:>11.4f}{b['auroc']:>12.4f}"
                      f"{b['macro_f1']:>10.4f}{dt:>8.1f}")
                rows.append({"Method": method, "Model": model, "fold": fold,
                             "grid": name, "n_configs": b["n_configs"],
                             "h1": b["h1"], "h2": b["h2"], "max_iter": b["max_iter"],
                             "val_macro_f1": b["val"], "test_bacc": b["bacc"],
                             "test_auroc": b["auroc"], "test_macro_f1": b["macro_f1"],
                             "test_acc": b["acc"], "seconds": dt})

    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["Method", "Model", "fold"], columns="grid",
                         values=["test_bacc", "test_auroc", "test_macro_f1", "seconds"])
    print(f"\n{'=' * 96}\nMEAN ACROSS ALL PROBED (combination, fold) PAIRS\n{'=' * 96}")
    agg = df.groupby("grid")[["val_macro_f1", "test_bacc", "test_auroc",
                              "test_macro_f1", "test_acc", "seconds"]].mean()
    print(agg.round(4).to_string())
    if {"current", "wider"} <= set(agg.index):
        d = agg.loc["wider"] - agg.loc["current"]
        print("\ndelta (wider - current):")
        print(d.round(4).to_string())
        print(f"\ncost multiplier: {agg.loc['wider','seconds'] / agg.loc['current','seconds']:.1f}x")
        print("\nAdopt only if the TEST metrics of the VALIDATION-selected model improve.")
        print("The test set never participates in selection in either arm.")

    out = REPO_ROOT / "slide_classification" / "ann_grid_probe.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
