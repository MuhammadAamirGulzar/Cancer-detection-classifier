"""Consolidated reporting layer (work order Phase 4, Tasks 4.1-4.3).

Reads the provenance-stamped ``result_*.json`` files every runner writes and
produces ``best_of_all_exps_metric.xlsx`` with **one sheet per canonical
experiment**:

    TCGA-CV . PAIP-IV . PAIP-EV . SurGen-CV . SurGen-EV

Five sheets. The old ``PAIP`` sheet is gone with PAIP-CV (Task 2.2), and archived
``*_ARCHIVED_*`` trees are never read - the old builder globbed ``*_Results``,
which would now match them and silently mix pre- and post-correction numbers into
one sheet.

Each sheet carries three blocks:

  ``Summary``      one row per (aggregation method x foundation model x
                   classifier). **Sort order (Task 4.2): descending by BalAcc and
                   AUROC jointly, AUROC winning ties.** Implemented as sort by
                   AUROC descending, then a *stable* sort by BalAcc descending -
                   which yields BalAcc primary with AUROC as the tie-break.
  ``TITAN_PRISM``  TITAN and PRISM tabulated apart (Task 4.3). They are
                   slide-level encoders with no patch-aggregation step - they are
                   baselines, not aggregation methods, and putting them in the
                   same table invites a reviewer to read them as comparable rows.
  ``Detail``       every column carried through, including per-variant EV rows,
                   confidence intervals, thresholds and provenance.

Run:  python tools/build_report.py [--out FILE.xlsx]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SLIDE_CLS = REPO_ROOT / "slide_classification"
sys.path.insert(0, str(SLIDE_CLS))

CANONICAL_EXPERIMENTS = ["TCGA-CV", "PAIP-IV", "PAIP-EV", "SurGen-CV", "SurGen-EV"]

#: Slide-level encoders - baselines, not aggregation strategies (Task 4.3).
SLIDE_LEVEL_ENCODERS = {"TITAN", "PRISM"}

SUMMARY_COLUMNS = ["Method", "Model", "Classifier", "Variant",
                   "BalAcc", "AUROC", "Acc", "MacroF1", "N_test",
                   # Which threshold scheme produced this row. PAIP-EV now mixes
                   # two of them (KNN k=35 + refit tau, RF rate-matched, the rest
                   # frozen tau_TCGA), so a bare metric is no longer
                   # self-describing. Sits beside the metrics, not at the far
                   # right, because it qualifies them.
                   "Threshold_scheme",
                   # Corrected-mode columns. Every sheet filters this list with
                   # `if c in agg.columns`, so on a frozen run they are absent
                   # and the workbook is identical to before.
                   "BalAcc_corrected", "AUROC_corrected", "Status_corrected"]


def discover(root: Path = SLIDE_CLS) -> List[dict]:
    """Every non-archived result JSON under the results trees."""
    out = []
    for path in sorted(root.rglob("result_*.json")):
        if "_ARCHIVED_" in str(path):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[WARN] unreadable {path}: {exc}")
            continue
        out.append({"path": path, "payload": payload,
                    "stamp": payload.get("stamp", {})})
    return out


def _rows_from(record: dict) -> List[dict]:
    """Flatten one result file into summary rows."""
    payload, stamp = record["payload"], record["stamp"]
    exp = stamp.get("experiment", "?")
    base = {
        "Experiment": exp,
        "Method": stamp.get("method", "?"),
        "Model": stamp.get("model", "?"),
        "N_test": stamp.get("n_test"),
        "Machine": stamp.get("machine"),
        "GitCommit": stamp.get("git_commit"),
        "Timestamp": stamp.get("timestamp"),
    }
    rows: List[dict] = []

    # CV-style: a precomputed per-classifier summary block.
    if "summary" in payload:
        for s in payload["summary"]:
            rows.append({**base, "Classifier": s.get("Classifier"), "Variant": "default",
                         "BalAcc": s.get("bacc"), "AUROC": s.get("auroc"),
                         "Acc": s.get("acc"), "MacroF1": s.get("macro_f1"),
                         "WeightedF1": s.get("weighted_f1"),
                         "BalAcc_sd": s.get("bacc_sd"), "AUROC_sd": s.get("auroc_sd"),
                         "N_test": s.get("N_test", base["N_test"]),
                         "ConfMatrix": s.get("conf_matrix"), "Threshold": 0.5})
        return rows

    results = payload.get("results", {})
    for clf, val in results.items():
        if not isinstance(val, dict):
            continue
        # A record may be flat (IV-style), nested by variant (EV-style), or -
        # as PAIP-IV is - flat *and* carrying an extra operating point under
        # ``at_tau_train``. Handle the two parts independently: treating the
        # third case as purely nested drops the flat row, and with it AUROC,
        # which a threshold-specific block does not restate.
        nested = {k: v for k, v in val.items()
                  if isinstance(v, dict) and "bacc" in v}
        is_flat = "bacc" in val

        if is_flat:
            row = {**base, "Classifier": clf, "Variant": "default", **_metric_cols(val)}
            boot = val.get("bootstrap") or {}
            if boot.get("bacc_ci"):
                row["BalAcc_CI"] = f"[{boot['bacc_ci'][0]:.4f}, {boot['bacc_ci'][1]:.4f}]"
            if boot.get("auroc_ci"):
                row["AUROC_CI"] = f"[{boot['auroc_ci'][0]:.4f}, {boot['auroc_ci'][1]:.4f}]"
            row["N_train"] = val.get("n_train")
            rows.append(row)

        for variant, m in nested.items():
            cols = _metric_cols(m)
            # AUROC is threshold-free, so an operating-point block stores it
            # once on the parent rather than repeating it per threshold.
            if is_flat:
                for key, src in (("AUROC", "auroc"), ("AUROC_sd", "auroc_sd"),
                                 ("WeightedF1", "weighted_f1")):
                    if cols.get(key) is None:
                        cols[key] = val.get(src)
            rows.append({**base, "Classifier": clf, "Variant": variant, **cols})
    return rows


def _metric_cols(m: dict) -> dict:
    """Frozen metrics, plus corrected ones only when the run produced them.

    ``ev_runner --threshold-mode corrected`` writes ``bacc_corrected`` etc.
    BESIDE the frozen keys; a frozen run writes none of them, so the extra
    columns never materialise and the report is byte-for-byte what it was.
    """
    extra = {}
    if m.get("promoted"):
        # A promoted row's headline IS the corrected number, so repeating it in a
        # *_corrected column would imply an alternative reading that no longer
        # exists. What a reader needs instead is the scheme and what the figure
        # displaced, so the change is legible from the workbook alone.
        extra = {
            "Threshold_scheme": m.get("threshold_scheme"),
            "Status": m.get("status"),
            "BalAcc_prev_frozen": m.get("bacc_frozen"),
            "AUROC_prev_frozen": m.get("auroc_frozen"),
            "Threshold_prev_frozen": m.get("threshold_frozen"),
            "Status_prev_frozen": m.get("status_frozen"),
            "N_pos_pred": m.get("n_pos_pred_corrected"),
            "KNN_k": m.get("knn_k_corrected"),
            "KNN_tau_source": m.get("knn_tau_source"),
        }
    elif "bacc_corrected" in m:
        extra = {
            "BalAcc_corrected": m.get("bacc_corrected"),
            # only KNN changes its scores (k changes); others keep frozen AUROC
            "AUROC_corrected": m.get("auroc_corrected", m.get("auroc")),
            "Acc_corrected": m.get("acc_corrected"),
            "MacroF1_corrected": m.get("macro_f1_corrected"),
            "Threshold_corrected": m.get("threshold_corrected"),
            "Status_frozen": m.get("status_frozen"),
            "Status_corrected": m.get("status_corrected"),
            "N_pos_pred_corrected": m.get("n_pos_pred_corrected"),
            "Corrected_scheme": m.get("corrected_scheme"),
            "ConfMatrix_corrected": (str(m.get("conf_matrix_corrected"))
                                     if m.get("conf_matrix_corrected") else None),
        }
    return {**extra, **{
        "BalAcc": m.get("bacc"), "AUROC": m.get("auroc"), "Acc": m.get("acc"),
        "MacroF1": m.get("macro_f1"), "WeightedF1": m.get("weighted_f1"),
        "BalAcc_sd": m.get("bacc_sd"), "AUROC_sd": m.get("auroc_sd"),
        "Threshold": m.get("threshold"), "N_seeds": m.get("n_seeds"),
        "N_folds_used": m.get("n_folds_used"),
        "ConfMatrix": str(m.get("conf_matrix")) if m.get("conf_matrix") else None,
    }}


def fill_threshold_scheme(df: pd.DataFrame) -> pd.DataFrame:
    """Give every row in a mixed-scheme experiment an explicit scheme label.

    Only promoted rows carry ``Threshold_scheme``; a frozen row carries nothing,
    because for four of the five experiments there is only one scheme and naming
    it on every row would be noise. But inside an experiment that mixes schemes,
    a blank cell is the ambiguous case - the reader cannot tell "frozen" from
    "nobody recorded it". So the blanks are filled per experiment, and only where
    at least one row was promoted; experiments with a single scheme never grow
    the column and keep their previous shape exactly.

    Modifies and returns ``df``. Safe to call on a frame that has no such column.
    """
    if "Threshold_scheme" not in df.columns or "Experiment" not in df.columns:
        return df
    mixed = [e for e, g in df.groupby("Experiment")
             if g["Threshold_scheme"].notna().any()]
    m = df["Experiment"].isin(mixed)
    df.loc[m, "Threshold_scheme"] = df.loc[m, "Threshold_scheme"].fillna("frozen_tau_TCGA")
    return df


def sort_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Task 4.2 sort: BalAcc primary, AUROC as tie-break.

    Sort by AUROC descending first, then a **stable** sort by BalAcc descending.
    The stable second pass preserves the AUROC ordering within equal BalAcc, which
    is exactly "descending by BalAcc and AUROC jointly, AUROC wins on a tie".
    """
    if df.empty:
        return df
    df = df.sort_values("AUROC", ascending=False, kind="mergesort")
    df = df.sort_values("BalAcc", ascending=False, kind="mergesort")
    return df.reset_index(drop=True)


def build(out_path: Optional[Path] = None, root: Path = SLIDE_CLS) -> Dict[str, pd.DataFrame]:
    records = discover(root)
    print(f"discovered {len(records)} result file(s) (archived trees excluded)")

    per_exp: Dict[str, List[dict]] = defaultdict(list)
    for rec in records:
        for row in _rows_from(rec):
            per_exp[row["Experiment"]].append(row)

    out_path = out_path or (SLIDE_CLS / "best_of_all_exps_metric.xlsx")
    sheets: Dict[str, pd.DataFrame] = {}

    order = [e for e in CANONICAL_EXPERIMENTS if e in per_exp]
    order += [e for e in sorted(per_exp) if e not in CANONICAL_EXPERIMENTS]

    with pd.ExcelWriter(out_path, engine="openpyxl", mode="w") as xw:
        for exp in order:
            df = fill_threshold_scheme(pd.DataFrame(per_exp[exp]))
            agg = df[~df["Method"].isin(SLIDE_LEVEL_ENCODERS)]
            enc = df[df["Method"].isin(SLIDE_LEVEL_ENCODERS)]

            summary = sort_summary(agg)[
                [c for c in SUMMARY_COLUMNS if c in agg.columns]]
            summary.to_excel(xw, sheet_name=exp[:31], index=False, startrow=0)

            if not enc.empty:
                tp = sort_summary(enc)[[c for c in SUMMARY_COLUMNS if c in enc.columns]]
                tp.to_excel(xw, sheet_name=f"{exp}_TITAN_PRISM"[:31], index=False)

            sort_summary(df).to_excel(xw, sheet_name=f"{exp}_Detail"[:31], index=False)
            sheets[exp] = summary
            print(f"  {exp:<12} {len(summary):>4} aggregation rows, "
                  f"{len(enc):>3} TITAN/PRISM rows")

        cov = pd.DataFrame([{
            "Experiment": r["stamp"].get("experiment"),
            "Method": r["stamp"].get("method"), "Model": r["stamp"].get("model"),
            "N_test": r["stamp"].get("n_test"), "Machine": r["stamp"].get("machine"),
            "GitCommit": r["stamp"].get("git_commit"),
            "Timestamp": r["stamp"].get("timestamp"),
        } for r in records])
        cov.to_excel(xw, sheet_name="Coverage", index=False)

    print(f"wrote {out_path}")
    return sheets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=SLIDE_CLS / "best_of_all_exps_metric.xlsx")
    ap.add_argument("--root", type=Path, default=SLIDE_CLS)
    args = ap.parse_args()
    build(args.out, args.root)


if __name__ == "__main__":
    main()
