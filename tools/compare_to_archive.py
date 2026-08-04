"""Before/after comparison: pre-remediation results vs the corrected re-run.

This is the evidence base for the supplementary report. Every number published
before 2026-08-04 is superseded, and a reviewer comparing the two result sets
will ask *which* correction moved *which* number and by how much. This produces
that table directly from both result trees - the archived one and the live one -
rather than from anybody's recollection.

It reads two different formats and reconciles them:

  OLD  ``cross_valid_avg_eval_metrics.xlsx`` - one row per ``<clf>_<metric>``,
       one column per fold plus ``AvgFolds``. Written by the pre-remediation
       notebook pipeline.
  NEW  ``result_<experiment>_<method>_<model>_<variant>.json`` - the
       provenance-stamped record written by the corrected runners.

Which corrections apply to which experiment is attributed per row, so the table
explains itself:

  TCGA-CV    ANN hyperparameters were selected on the test fold (Task 1.1); the
             saved ANN checkpoint was not the selected configuration (1.2); the
             ANN had a double softmax (1.3); KNN's metric grid explored two
             distinct metrics while claiming three.
  SurGen-CV  all of the above, plus slide-level folds that split ~67% of
             two-slide cases across train and test (Task 2.1).
  PAIP-EV    all of the above, plus a fixed operating point replacing an
             uncalibrated 0.5 / hardcoded 0.3 (section 1b).

Run:  python tools/compare_to_archive.py [--out FILE.xlsx]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SLIDE_CLS = REPO_ROOT / "slide_classification"
sys.path.insert(0, str(REPO_ROOT / "tools"))

from build_report import discover, _rows_from  # noqa: E402

METRICS = ["bacc", "auroc", "acc", "macro_f1"]
METRIC_LABEL = {"bacc": "BalAcc", "auroc": "AUROC", "acc": "Acc",
                "macro_f1": "MacroF1"}

#: Archived tree -> the canonical experiment whose new results supersede it.
ARCHIVE_MAP = {
    "TCGA_Results_ARCHIVED_": "TCGA-CV",
    "SurGen_Results_ARCHIVED_": "SurGen-CV",
    "TCGA_PAIP_EV_Results_ARCHIVED_": "PAIP-EV",
    "PAIP_Results_ARCHIVED_": "PAIP-CV (retired)",
}

CORRECTIONS = {
    "TCGA-CV": ("ANN hparams selected on test fold (1.1); wrong ANN checkpoint "
                "saved (1.2); ANN double softmax (1.3); KNN metric grid had "
                "euclidean twice"),
    "SurGen-CV": ("all TCGA-CV corrections, plus slide-level folds leaking ~67% "
                  "of two-slide cases across train/test (2.1)"),
    "PAIP-EV": ("all TCGA-CV corrections, plus tau_TCGA replacing an "
                "uncalibrated 0.5 / hardcoded RF 0.3 (1b), plus TCGA-FULL "
                "training on 413 rather than ~208 slides (3.0)"),
    "PAIP-CV (retired)": "experiment withdrawn from the study (Task 2.2)",
}


def parse_old_xlsx(path: Path) -> List[dict]:
    """Rows from a pre-remediation ``*_eval_metrics.xlsx``."""
    try:
        df = pd.read_excel(path, sheet_name=0)
    except Exception as exc:
        print(f"[WARN] unreadable {path}: {exc}")
        return []
    if "Metric" not in df.columns:
        return []
    val_col = "AvgFolds" if "AvgFolds" in df.columns else df.columns[-1]

    out = []
    for _, r in df.iterrows():
        m = re.match(r"^([a-z]+)_(acc|bacc|macro_f1|weighted_f1|auroc)$",
                     str(r["Metric"]).strip())
        if not m:
            continue
        clf, metric = m.group(1), m.group(2)
        try:
            val = float(r[val_col])
        except (TypeError, ValueError):
            continue
        out.append({"Classifier": clf, "metric": metric, "old": val})
    return out


def _baseline_trees(root: Path) -> Dict[str, Path]:
    """Pick the true pre-remediation archive per experiment.

    A tree may be archived more than once - re-running an experiment archives
    whatever was there, so an intermediate corrected run can end up as
    ``*_ARCHIVED_<date>_2``. The baseline is the *first* archive, whose name ends
    in the bare date with no ``_N`` suffix. Later archives are intermediate
    snapshots and are reported, not compared against, because "previous results"
    in the supplementary report means what was published before the remediation -
    not an interim state.
    """
    by_exp: Dict[str, List[Path]] = {}
    for tree in sorted(root.iterdir()):
        if not tree.is_dir():
            continue
        exp = next((v for k, v in ARCHIVE_MAP.items() if tree.name.startswith(k)), None)
        if exp:
            by_exp.setdefault(exp, []).append(tree)

    baselines: Dict[str, Path] = {}
    for exp, trees in by_exp.items():
        bare = [t for t in trees if re.search(r"_ARCHIVED_\d{8}$", t.name)]
        baselines[exp] = (bare or sorted(trees))[0]
        extra = [t.name for t in trees if t != baselines[exp]]
        if extra:
            print(f"[info] {exp}: baseline = {baselines[exp].name}; "
                  f"intermediate archive(s) not compared: {extra}")
    return baselines


def scan_archives(root: Path = SLIDE_CLS) -> pd.DataFrame:
    """Every metric in each experiment's pre-remediation baseline tree."""
    rows = []
    for exp, tree in _baseline_trees(root).items():
        for xlsx in tree.rglob("*eval_metrics.xlsx"):
            parts = xlsx.relative_to(tree).parts
            # <method>/<model>/<task>/Output/... or PRISM/<task>/Output/...
            if parts[0] == "PRISM":
                method, model = "PRISM", "PRISM"
            elif len(parts) >= 2:
                method, model = parts[0], parts[1]
            else:
                continue
            for rec in parse_old_xlsx(xlsx):
                rows.append({"Experiment": exp, "Method": method, "Model": model,
                             "ArchiveTree": tree.name, **rec})
    return pd.DataFrame(rows)


def scan_new(root: Path = SLIDE_CLS) -> pd.DataFrame:
    """Every metric in the corrected result files, long-form."""
    recs = []
    for rec in discover(root):
        recs.extend(_rows_from(rec))
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs)
    keep = df["Variant"].isin(["default", "tcga_full"]) if "Variant" in df else True
    df = df[keep] if not isinstance(keep, bool) else df

    long = []
    for _, r in df.iterrows():
        for m in METRICS:
            col = METRIC_LABEL[m]
            if col in df.columns and pd.notna(r.get(col)):
                long.append({"Experiment": r["Experiment"], "Method": r["Method"],
                             "Model": r["Model"], "Classifier": r["Classifier"],
                             "metric": m, "new": float(r[col]),
                             "Variant": r.get("Variant", "default"),
                             "N_test": r.get("N_test")})
    return pd.DataFrame(long)


def build(out_path: Optional[Path] = None) -> pd.DataFrame:
    old = scan_archives()
    new = scan_new()
    if old.empty:
        raise SystemExit("no archived results found - nothing to compare against")
    if new.empty:
        raise SystemExit("no corrected results found - run the experiments first")

    merged = old.merge(new, on=["Experiment", "Method", "Model", "Classifier", "metric"],
                       how="outer", indicator=True)
    merged["delta"] = merged["new"] - merged["old"]
    merged["Metric"] = merged["metric"].map(METRIC_LABEL)
    merged["Corrections"] = merged["Experiment"].map(CORRECTIONS)
    merged["Status"] = merged["_merge"].map({
        "both": "compared", "left_only": "old only (not re-run / retired)",
        "right_only": "new only (no prior result)"})

    cols = ["Experiment", "Method", "Model", "Classifier", "Metric",
            "old", "new", "delta", "Status", "Variant", "N_test",
            "ArchiveTree", "Corrections"]
    merged = merged[[c for c in cols if c in merged.columns]]
    merged = merged.sort_values(["Experiment", "Metric", "delta"],
                                ascending=[True, True, True])

    # Per-(experiment, metric) roll-up: the headline of the supplementary report.
    both = merged[merged["Status"] == "compared"]
    summary = (both.groupby(["Experiment", "Metric"])
               .agg(n=("delta", "size"), mean_delta=("delta", "mean"),
                    median_delta=("delta", "median"),
                    min_delta=("delta", "min"), max_delta=("delta", "max"),
                    n_improved=("delta", lambda s: int((s > 0).sum())),
                    n_worse=("delta", lambda s: int((s < 0).sum())))
               .reset_index())

    by_clf = (both.groupby(["Experiment", "Metric", "Classifier"])
              .agg(n=("delta", "size"), mean_delta=("delta", "mean"))
              .reset_index())

    out_path = out_path or (SLIDE_CLS / "comparison_old_vs_new.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl", mode="w") as xw:
        summary.to_excel(xw, sheet_name="Summary", index=False)
        by_clf.to_excel(xw, sheet_name="ByClassifier", index=False)
        merged.to_excel(xw, sheet_name="Detail", index=False)
    print(f"wrote {out_path}")

    print("\nMean change (new - old), per experiment and metric:")
    print(summary.to_string(index=False))
    unmatched = merged[merged["Status"] != "compared"]
    if len(unmatched):
        print(f"\n{len(unmatched)} row(s) could not be paired "
              f"({unmatched['Status'].value_counts().to_dict()})")
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=SLIDE_CLS / "comparison_old_vs_new.xlsx")
    args = ap.parse_args()
    build(args.out)


if __name__ == "__main__":
    main()
