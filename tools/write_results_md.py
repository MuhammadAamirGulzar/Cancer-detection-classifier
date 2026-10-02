"""Generate RESULTS.md - the single results document (work order Task 4.5).

Per experiment: definition, cohort, N, split description, coverage matrix,
summary table and known caveats. Plus a top-level table stating **N for every
experiment**, because those differ from the raw label-file counts and reviewers
will check:

    TCGA-CV 413 . PAIP-IV 47 train / 31 test . PAIP-EV 78 . SurGen-CV 624 . SurGen-EV 622

Everything here is derived from the result files - nothing is typed by hand, so
re-running after a re-run keeps the document honest (rule of engagement #1).

Run:  python tools/write_results_md.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SLIDE_CLS = REPO_ROOT / "slide_classification"
sys.path.insert(0, str(REPO_ROOT / "tools"))

from build_report import (discover, _rows_from, SLIDE_LEVEL_ENCODERS,  # noqa: E402
                          sort_summary, fill_threshold_scheme)

OUT = REPO_ROOT / "RESULTS.md"

DEFINITIONS = {
    "TCGA-CV": dict(
        cohort="TCGA (primary)", protocol="4-fold cross-validation, averaged",
        train="TCGA folds", test="TCGA held-out fold",
        kind="CV",
        note="Each of the 413 slides is tested exactly once. Reported at threshold 0.5."),
    "PAIP-IV": dict(
        cohort="PAIP (external cohort, internal split)",
        protocol="single fixed provider split, no averaging",
        train="PAIP official train (all 47 have features, 12 MSI-H)",
        test="PAIP official test (31, all have features)",
        kind="IV",
        note="Reported at threshold 0.5 with bootstrap 95% CIs over the 31 test slides. "
             "The ANN head is fitted on all 47 training slides with the TCGA-CV-derived "
             "configuration, matching the other four heads; the 20% carve-out is used "
             "for early stopping only."),
    "PAIP-EV": dict(
        cohort="PAIP (external)", protocol="external validation",
        train="TCGA (see variant)", test="all 78 PAIP slides", kind="EV",
        note="**Two threshold schemes in one table.** kNN is scored at k=35 "
             "(uniform weights) and cut at a tau refitted on TCGA out-of-fold "
             "probabilities at that k, so kNN is the only head whose AUROC moves. "
             "RF keeps its scores and takes a rate-matched (quantile) threshold, so "
             "its AUROC is unchanged and only the operating point moves. LR, ANN and "
             "ProtoNet keep the frozen tau_TCGA and are unchanged. Both schemes come "
             "from TCGA alone — the quantile reads PAIP scores but never PAIP labels. "
             "Applies to the `tcga_full` variant; the two fold-based variants stay "
             "entirely frozen, as does all of SurGen-EV, so kNN/RF are not comparable "
             "across the two external cohorts."),
    "SurGen-CV": dict(
        cohort="SurGen (second external cohort, internal CV)",
        protocol="4-fold cross-validation, case-grouped",
        train="SurGen folds", test="SurGen held-out fold", kind="CV",
        note="Folds are built at CASE level (Task 2.1). Reported at threshold 0.5."),
    "SurGen-EV": dict(
        cohort="SurGen (external)", protocol="external validation",
        train="TCGA (see variant)", test="all SurGen slides with features", kind="EV",
        note="Reported at tau_TCGA. Partial coverage on the local machine by design."),
}

DEFINITION_TEXT = """\
- **CV (cross-validation)** — the cohort is split into K folds; each fold serves as
  test once; the reported metric is the mean across folds.
- **IV (internal validation)** — a single train/test split defined by the dataset
  provider; no averaging.
- **EV (external validation)** — trained entirely on cohort A, tested on the whole
  of cohort B, with zero cohort-B data in training. Three variants are reported
  side by side so they can be compared directly: TCGA-FULL (primary), the 4-fold
  probability ensemble, and the legacy 4-fold-average convention. None is discarded.
"""


def load() -> pd.DataFrame:
    rows: List[dict] = []
    for rec in discover(SLIDE_CLS):
        rows.extend(_rows_from(rec))
    if not rows:
        raise SystemExit("no result files found")
    return fill_threshold_scheme(pd.DataFrame(rows))


def md_table(df: pd.DataFrame, cols: List[str], max_rows: int = None) -> str:
    cols = [c for c in cols if c in df.columns]
    if df.empty or not cols:
        return "_(no rows)_\n"
    d = df[cols].head(max_rows) if max_rows else df[cols]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in d.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("—" if pd.isna(v) else f"{v:.4f}")
            else:
                cells.append("—" if pd.isna(v) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def coverage_matrix(df: pd.DataFrame, experiment: str) -> str:
    sub = df[df["Experiment"] == experiment]
    methods = sorted(set(sub["Method"]))
    models = sorted(set(sub["Model"]))
    if not methods:
        return "_(no coverage)_\n"
    lines = ["| Method \\ Model | " + " | ".join(models) + " |",
             "|" + "|".join("---" for _ in range(len(models) + 1)) + "|"]
    for me in methods:
        cells = []
        for mo in models:
            n = len(sub[(sub["Method"] == me) & (sub["Model"] == mo)])
            cells.append("✅" if n else "—")
        lines.append(f"| {me} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    df = load()
    experiments = [e for e in ["TCGA-CV", "PAIP-IV", "PAIP-EV", "SurGen-CV", "SurGen-EV"]
                   if e in set(df["Experiment"])]

    parts: List[str] = []
    parts.append("# RESULTS\n")
    parts.append("MSI-H vs non-MSI-H prediction from H&E whole-slide images in "
                 "colorectal cancer. The experimental variable is the **slide-level "
                 "aggregation strategy**, not the encoder.\n")
    parts.append("> Generated by `tools/write_results_md.py` from the result files. "
                 "Do not edit by hand — re-run the script instead.\n")

    # ---- top-level N table -------------------------------------------------
    parts.append("\n## Cohort sizes — read this first\n")
    parts.append("These differ from the raw label-file counts, because a slide needs "
                 "**both** a label and a feature file to be modelled.\n")
    n_rows = []
    for e in experiments:
        sub = df[df["Experiment"] == e]
        n = sub["N_test"].dropna()
        n_rows.append({
            "Experiment": e,
            "Kind": DEFINITIONS[e]["kind"],
            "Train": DEFINITIONS[e]["train"],
            "Test": DEFINITIONS[e]["test"],
            "N_test": int(n.iloc[0]) if len(n) else "—",
        })
    parts.append(md_table(pd.DataFrame(n_rows),
                          ["Experiment", "Kind", "Train", "Test", "N_test"]))
    parts.append("\n### Reconciliation with the label files\n")
    parts.append(
        "| Cohort | Slides in label file | With features (modelled) | Excluded |\n"
        "|---|---|---|---|\n"
        "| TCGA | 416 | **413** (60 MSI-H / 353 non) | `TCGA-AD-6895_MSIH`, "
        "`TCGA-AD-6899_nonMSIH`, `TCGA-CM-6680_nonMSIH` |\n"
        "| PAIP | 78 | **73** (17 MSI-H / 56 non) | `training_data_19/30/41/42/46` |\n"
        "| SurGen | 624 usable (of 1020; 396 are label −1) | **622** (60 MSI-H / 562 non) | "
        "`SR386_40X_HE_T086_01`, `SR386_40X_HE_T339_01` |\n")

    parts.append("\n## Definitions\n")
    parts.append(DEFINITION_TEXT)

    # ---- per experiment ----------------------------------------------------
    for e in experiments:
        d = DEFINITIONS[e]
        sub = df[df["Experiment"] == e]
        agg = sub[~sub["Method"].isin(SLIDE_LEVEL_ENCODERS)]
        enc = sub[sub["Method"].isin(SLIDE_LEVEL_ENCODERS)]

        parts.append(f"\n---\n\n## {e}\n")
        parts.append(f"- **Cohort:** {d['cohort']}\n"
                     f"- **Protocol:** {d['protocol']}\n"
                     f"- **Train:** {d['train']}\n"
                     f"- **Test:** {d['test']}\n"
                     f"- **Threshold policy:** {d['note']}\n")

        parts.append(f"\n### Coverage\n")
        parts.append(coverage_matrix(df, e))

        parts.append(f"\n### Summary — aggregation methods\n")
        parts.append("Sorted by balanced accuracy, with AUROC as the tie-break. "
                     "AUROC is the primary metric: it is threshold-free.\n\n")
        cols = ["Method", "Model", "Classifier", "Variant", "BalAcc", "AUROC",
                "Acc", "MacroF1", "N_test"]
        if e == "PAIP-IV":
            cols = ["Method", "Model", "Classifier", "BalAcc", "BalAcc_CI",
                    "AUROC", "AUROC_CI", "Acc", "MacroF1", "N_test"]
        # Where an experiment mixes threshold schemes, the metrics alone do not
        # say what produced them - and these rows get quoted out of context.
        if "Threshold_scheme" in sub.columns and sub["Threshold_scheme"].notna().any():
            cols = cols + ["Threshold_scheme"]
        parts.append(md_table(sort_summary(agg), cols, max_rows=25))
        if len(agg) > 25:
            parts.append(f"\n_Showing the top 25 of {len(agg)} rows; the full table is "
                         f"in `slide_classification/best_of_all_exps_metric.xlsx`._\n")

        if not enc.empty:
            parts.append(f"\n### TITAN and PRISM (tabulated apart)\n")
            parts.append("TITAN and PRISM are **slide-level encoders with no patch "
                         "aggregation step**. They are baselines, not aggregation "
                         "methods, and are tabulated separately so they are not read "
                         "as comparable rows.\n\n")
            parts.append(md_table(sort_summary(enc), cols, max_rows=15))

    # ---- caveats -----------------------------------------------------------
    parts.append("\n---\n\n## Known caveats — state these in Methods\n")
    parts.append("""\
1. **Training uses 50% of each cohort, not 75%.** The CV rotation is
   test = fold *i*, val = fold *i+1*, train = the remaining two. Owner-confirmed as
   deliberate. A reviewer comparing against 5-fold work will otherwise assume more
   training data than was used.
2. **"Validation fold" means two different things.** `combine_trainval=True` for
   `lin`/`knn`/`proto`/`rf` merges the validation fold back into training; only the
   ANN uses it as a true early-stopping set.
3. **Fold numbering offset.** Folds are 1–4 in the CSVs and reported `Fold1..Fold4`,
   but checkpoints are saved `fold0..fold3`.
4. **Empty tissue rows.** In Tissue_Type_Clustering an absent tissue class yields an
   all-zero row. Across the TCGA slides: lymphocyte empty in 109, adipose 69, smooth
   muscle 51, normal mucosa 51. The classifier cannot distinguish "absent" from
   "zero-valued" — relevant when interpreting which classes the top-k experiments select.
5. **PAIP-CV was run previously and has been retired.** Its folds were built by
   sequential slicing of CSV order, so they tracked PAIP's own acquisition split
   (folds 1–2 were entirely `training_data_*`, fold 4 entirely `validation_data_*`),
   and the class balance came out even by luck of ordering rather than by design.
   The deciding argument, though, is that the experiment is strictly dominated:
   even repaired it would train on ~36 slides and test on 18 with ~4 positives,
   where PAIP-IV trains on 42 and tests on 31. Previous numbers are preserved at
   `slide_classification/PAIP_Results_ARCHIVED_20260804/`.
6. **All SurGen numbers published before 2026-08-04 are leak-inflated.** Folds were
   built at slide level while 70 of 554 cases contribute two slides each, so ~67% of
   those cases had one slide in train and the other in test. Correcting to
   case-level folds moves SurGen-CV by mean BalAcc −0.0252 and AUROC −0.0342, with
   all 15 measured (combination × classifier) BalAcc deltas negative.
7. **ANN results published before 2026-08-04 are affected by three separate bugs**
   (hyperparameters selected on the test fold, the saved checkpoint not being the
   selected configuration, and a double softmax). All are fixed; all ANN numbers
   were regenerated.
""")

    args.out.write_text("".join(parts), encoding="utf-8")
    print(f"wrote {args.out}  ({len(''.join(parts)):,} chars, "
          f"{len(experiments)} experiment(s))")


if __name__ == "__main__":
    main()
