"""SurGen-EV kNN k-sweep report - trend charts + docx, DIAGNOSTIC ONLY.

=============================================================================
 DIAGNOSTIC. No k is selected here. Nothing is promoted, nothing published
 changes. The SurGen-EV promotion candidate at experiments/surgen_ev_promotion/
 keeps its fixed k=35 regardless of what this report shows.
=============================================================================

Reruns nothing - reads the CSVs report_knn_k_surgen.py already produced from
the 3 Sep sweep. Configuration reported:

  * kNN     - swept over k in {15,20,25,35,50,75,100}, all three threshold
              schemes shown (frozen tau_TCGA, refit-tau-at-k, rate-matched
              quantile), so the reader can see how each behaves as k moves.
  * RF      - existing model config, rate-matched quantile threshold only
              (the scheme it would use if promoted).
  * LR/ANN/ProtoNet - existing model config, frozen tau_TCGA only (unchanged
              from the published record either way).

Why a line chart, not a radar
------------------------------
The project's radar (tools/make_radar_plots.py) plots FIXED spokes (aggregation
x classifier) with foundation model as the series - it has no axis for a swept
continuous parameter. Forcing k into that shape (e.g. k as the spoke) would
produce 20 spokes with no natural ordering and lose the one thing this report
needs to show: a trend. A line chart with k on the x-axis is what the data is.

Run:  python experiments/surgen_ev_diagnostic/build_k_sweep_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = HERE / "results"
OUT_DOCX = REPO / "SurGen_EV_KNN_K_Sweep_Report.docx"

K_GRID = [15, 20, 25, 35, 50, 75, 100]
SCHEME_LABEL = {"frozen_tau_TCGA": "Frozen τ_TCGA",
                "refit_tau_at_k": "Refit τ at k",
                "rate_matched_quantile": "Rate-matched quantile"}
SCHEME_COLOR = {"frozen_tau_TCGA": "#6b7280", "refit_tau_at_k": "#1f77b4",
                "rate_matched_quantile": "#d62728"}
METHOD_LABEL = {"Averaging": "Averaging", "Caption_based_aggregation": "Caption-based",
                "Caption_based_aggregation_15_classes": "Caption-based (15-class)",
                "Tissue_Type_Clustering": "Tissue-type clustering", "TITAN": "TITAN"}

RGB_HEAD = RGBColor(0x1F, 0x2A, 0x44)
RGB_MUTED = RGBColor(0x5A, 0x5F, 0x70)
RGB_DIAG = RGBColor(0xA1, 0x2D, 0x4C)


# --------------------------------------------------------------------- charts

def build_charts(summary: pd.DataFrame) -> tuple[Path, Path]:
    p1 = HERE / "knn_k_sweep_auroc.png"
    p2 = HERE / "knn_k_sweep_balacc.png"

    frozen = summary[summary.scheme == "frozen_tau_TCGA"].sort_values("k")

    # ---- AUROC vs k (scheme-independent - one line) + oracle ceiling --------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(frozen.k, frozen.auroc_mean, marker="o", color="#111827",
            linewidth=2.2, label="AUROC (all schemes - threshold-free)")
    ax.fill_between(frozen.k, frozen.auroc_min, frozen.auroc_max,
                    color="#111827", alpha=0.08, label="range across 20 combinations")
    ax.plot(frozen.k, frozen.oracle_mean, marker="^", linestyle="--",
            color="#9333ea", linewidth=1.8, label="oracle ceiling (uses target labels)")
    ax.axhline(0.5, color="#d1d5db", linewidth=1, linestyle=":")
    ax.set_xlabel("k (kNN neighbourhood size)")
    ax.set_ylabel("AUROC")
    ax.set_title("SurGen-EV kNN - AUROC vs k\nmean over 20 aggregation combinations "
                 "- DIAGNOSTIC, no k selected", fontsize=11)
    ax.set_xticks(K_GRID)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(p1)
    plt.close(fig)

    # ---- BalAcc vs k, three schemes + oracle ---------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for scheme in ("frozen_tau_TCGA", "refit_tau_at_k", "rate_matched_quantile"):
        s = summary[summary.scheme == scheme].sort_values("k")
        ax.plot(s.k, s.bacc_mean, marker="o", color=SCHEME_COLOR[scheme],
                linewidth=2.2, label=SCHEME_LABEL[scheme])
    ax.plot(frozen.k, frozen.oracle_mean, marker="^", linestyle="--",
            color="#9333ea", linewidth=1.8, label="oracle ceiling (uses target labels)")
    ax.axhline(0.5, color="#d1d5db", linewidth=1, linestyle=":")
    ax.set_xlabel("k (kNN neighbourhood size)")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("SurGen-EV kNN - Balanced accuracy vs k, by threshold scheme\n"
                 "mean over 20 aggregation combinations - DIAGNOSTIC, no k selected",
                 fontsize=11)
    ax.set_xticks(K_GRID)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(p2)
    plt.close(fig)

    return p1, p2


# ----------------------------------------------------------------- docx helpers

def style_header_row(table):
    for cell in table.rows[0].cells:
        for p in cell.paragraphs:
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(9)
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:fill"), "1F2A44")
        tcPr.append(shd)


def add_table(doc, rows, header, widths=None, small=True):
    table = doc.add_table(rows=1, cols=len(header))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Light Grid Accent 1"
    for i, h in enumerate(header):
        table.rows[0].cells[i].text = h
    style_header_row(table)
    for r in rows:
        cells = table.add_row().cells
        for i, v in enumerate(r):
            cells[i].text = str(v)
            if small:
                for p in cells[i].paragraphs:
                    for run in p.runs:
                        run.font.size = Pt(8.5)
    if widths:
        for row in table.rows:
            for cell, w in zip(row.cells, widths):
                cell.width = Inches(w)
    return table


def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGB_HEAD
    return h


def add_body(doc, text, italic=False, size=10, color=None, bold=False):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.italic = italic
    run.bold = bold
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = color
    return p


# ---------------------------------------------------------------------- build

def main() -> None:
    summary = pd.read_csv(RESULTS / "surgen_knn_k_report_summary.csv")
    detail = pd.read_csv(RESULTS / "surgen_knn_k_report_detail.csv")
    other = pd.read_csv(RESULTS / "surgen_other_heads_at_policy_scheme.csv")

    chart_auroc, chart_bacc = build_charts(summary)

    doc = Document()
    doc.styles["Normal"].font.size = Pt(10)
    doc.styles["Normal"].font.name = "Calibri"

    title = doc.add_heading("SurGen-EV — kNN k-Sweep Report", level=0)
    for r in title.runs:
        r.font.color.rgb = RGB_HEAD
    add_body(doc, "DIAGNOSTIC ONLY — no k has been selected, nothing is "
                  "promoted, and no published file has changed. Built from "
                  "experiments/surgen_ev_diagnostic/, reading the sweep computed "
                  "on 3 Sep 2026.", bold=True, color=RGB_DIAG)

    # -------------------------------------------------------------- overview
    add_heading(doc, "1. Overview", level=1)
    add_body(doc, "This report asks one question: as kNN's neighbourhood size k "
                  "moves away from the value each artifact's own grid search "
                  "picked on TCGA (mostly k=3), does SurGen-EV's kNN performance "
                  "improve meaningfully, and does the best threshold scheme "
                  "depend on k? Nothing here changes what is published; the "
                  "SurGen-EV tree remains entirely on the frozen τ_TCGA "
                  "operating point for all five heads, and the promotion "
                  "candidate keeps k=35.")
    add_table(doc, [
        ["Cohort", "SurGen, 622 slides (60 MSI-H / 562 non-MSI-H)"],
        ["k values swept", ", ".join(str(k) for k in K_GRID) + " (kNN only)"],
        ["Aggregation combinations", "20 (4 methods × 5 encoders)"],
        ["Scores", "re-scored from the saved TCGA-FULL kNN artifact at each k "
                   "(lazy learner - no retraining needed)"],
        ["Source", "experiments/surgen_ev_diagnostic/results/"
                   "part2_DIAGNOSTIC_knn_k_sweep.csv"],
    ], header=["Field", "Value"], widths=[1.7, 4.8])

    # ----------------------------------------------------------- configuration
    add_heading(doc, "2. Configuration", level=1)
    add_body(doc, "kNN is the only head swept. The other four keep their "
                  "existing model configuration and are reported at the ONE "
                  "threshold scheme each would use if promoted — not both "
                  "schemes side by side — since that choice is already fixed "
                  "policy (matching PAIP-EV's 2 Sep promotion).")
    add_table(doc, [
        ["k-Nearest Neighbours", "swept: 15, 20, 25, 35, 50, 75, 100",
         "all three: frozen τ_TCGA / refit-τ-at-k / rate-matched quantile"],
        ["Random Forest", "existing (n_estimators=500, class_weight={0:1,1:10})",
         "rate-matched quantile only"],
        ["Logistic Regression", "existing (C=10, max_iter=300)", "frozen τ_TCGA only"],
        ["Neural Network", "existing (per-combination, TCGA-CV majority vote)",
         "frozen τ_TCGA only"],
        ["ProtoNet", "existing (no hyperparameters)", "frozen τ_TCGA only"],
    ], header=["Classifier", "Model configuration", "Threshold scheme(s) reported"],
              widths=[1.3, 2.6, 2.6])

    # ---------------------------------------------------------------- charts
    add_heading(doc, "3. Trend across k", level=1)
    doc.add_picture(str(chart_auroc), width=Inches(6.2))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_body(doc, "Figure 1. AUROC is threshold-free, so all three schemes trace "
                  "the same line; the shaded band is the range across the 20 "
                  "combinations. Peaks at k=20 (0.6770) and falls to 0.5495 by "
                  "k=100. The oracle ceiling — the best balanced accuracy any "
                  "threshold could reach using the target's own labels — caps "
                  "around 0.63-0.64 across the whole range, which is why no "
                  "scheme below closes the gap: the scores themselves do not "
                  "separate the classes much better than that, at any k tested.",
             italic=True, size=8.5, color=RGB_MUTED)
    doc.add_paragraph()
    doc.add_picture(str(chart_bacc), width=Inches(6.2))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_body(doc, "Figure 2. The three schemes cross over: rate-matched quantile "
                  "wins at k≤25, refit-τ-at-k wins at k≥35. The published "
                  "frozen scheme (grey) is worst at every k once k moves away "
                  "from the value it was fitted at, because a τ derived at k=3 "
                  "has no meaning once the score scale changes at a different k.",
             italic=True, size=8.5, color=RGB_MUTED)

    # ---------------------------------------------------------- per-k summary
    doc.add_page_break()
    add_heading(doc, "4. Per-k summary (mean over 20 combinations)", level=1)
    piv = summary.pivot_table(index="k", columns="scheme",
                              values=["bacc_mean", "n_dead"])
    rows = []
    for k in K_GRID:
        auroc = summary[(summary.k == k) & (summary.scheme == "frozen_tau_TCGA")].auroc_mean.iloc[0]
        oracle = summary[(summary.k == k) & (summary.scheme == "frozen_tau_TCGA")].oracle_mean.iloc[0]
        bf = piv.loc[k, ("bacc_mean", "frozen_tau_TCGA")]
        br = piv.loc[k, ("bacc_mean", "refit_tau_at_k")]
        bq = piv.loc[k, ("bacc_mean", "rate_matched_quantile")]
        df_ = piv.loc[k, ("n_dead", "frozen_tau_TCGA")]
        dr = piv.loc[k, ("n_dead", "refit_tau_at_k")]
        dq = piv.loc[k, ("n_dead", "rate_matched_quantile")]
        rows.append([k, f"{auroc:.4f}", f"{oracle:.4f}",
                    f"{bf:.4f}", f"{br:.4f}", f"{bq:.4f}",
                    f"{int(df_)}", f"{int(dr)}", f"{int(dq)}"])
    add_table(doc, rows,
              header=["k", "AUROC", "Oracle", "BalAcc\nfrozen", "BalAcc\nrefit",
                      "BalAcc\nquantile", "Dead\nfrozen", "Dead\nrefit", "Dead\nquantile"],
              widths=[0.5, 0.7, 0.7, 0.75, 0.75, 0.8, 0.6, 0.6, 0.7])

    # --------------------------------------------------------- per-k detail
    doc.add_page_break()
    add_heading(doc, "5. kNN detail by combination, per k", level=1)
    add_body(doc, "AUROC is identical across schemes (threshold-free) and shown "
                  "once; the three BalAcc columns share the same score vector, "
                  "so the difference between them is entirely the threshold.")
    for k in K_GRID:
        add_heading(doc, f"5.{K_GRID.index(k)+1}  k = {k}", level=2)
        dk = detail[detail.k == k]
        combos = sorted(set(zip(dk.method, dk.model)))
        rows = []
        for method, model in combos:
            sub = dk[(dk.method == method) & (dk.model == model)]
            auroc = sub.auroc.iloc[0]
            oracle = sub.bacc_oracle.iloc[0]
            bf = sub[sub.scheme == "frozen_tau_TCGA"].bacc.iloc[0]
            br = sub[sub.scheme == "refit_tau_at_k"].bacc.iloc[0]
            bq = sub[sub.scheme == "rate_matched_quantile"].bacc.iloc[0]
            st = sub[sub.scheme == "frozen_tau_TCGA"].status.iloc[0]
            rows.append([METHOD_LABEL.get(method, method), model, f"{auroc:.4f}",
                        f"{bf:.4f}", f"{br:.4f}", f"{bq:.4f}", f"{oracle:.4f}", st])
        rows.sort(key=lambda r: -float(r[2]))
        add_table(doc, rows,
                  header=["Aggregation", "Encoder", "AUROC", "BalAcc\n(frozen)",
                          "BalAcc\n(refit)", "BalAcc\n(quantile)", "Oracle",
                          "Status\n(frozen)"],
                  widths=[1.35, 0.8, 0.65, 0.75, 0.7, 0.8, 0.65, 0.75])
        doc.add_paragraph()

    # --------------------------------------------------- other heads, policy scheme
    doc.add_page_break()
    add_heading(doc, "6. Other four heads, at their policy scheme", level=1)
    add_body(doc, "Not swept — existing model configuration, reported once "
                  "each, for comparison against the kNN sweep above.")
    for clf, label, scheme_lbl in (("ann", "Neural Network", "frozen"),
                                    ("lin", "Logistic Regression", "frozen"),
                                    ("proto", "ProtoNet", "frozen"),
                                    ("rf", "Random Forest", "rate-matched quantile")):
        add_heading(doc, f"6.{['ann','lin','proto','rf'].index(clf)+1}  "
                        f"{label} ({scheme_lbl})", level=2)
        s = other[other.clf == clf].sort_values("auroc", ascending=False)
        rows = [[METHOD_LABEL.get(m, m), mo, f"{a:.4f}", f"{b:.4f}", f"{o:.4f}",
                f"{t:.6f}", st]
               for m, mo, a, b, o, t, st in
               zip(s.method, s.model, s.auroc, s.bacc, s.bacc_oracle, s.tau, s.status)]
        add_table(doc, rows,
                  header=["Aggregation", "Encoder", "AUROC", "BalAcc", "Oracle",
                          "τ", "Status"],
                  widths=[1.5, 0.9, 0.7, 0.7, 0.7, 0.8, 0.8])
        doc.add_paragraph()

    doc.add_page_break()
    add_heading(doc, "Provenance", level=1)
    add_table(doc, [
        ["Status", "DIAGNOSTIC — no k selected, nothing promoted"],
        ["Source data", "experiments/surgen_ev_diagnostic/results/"
                        "part2_DIAGNOSTIC_knn_k_sweep.csv, "
                        "surgen_other_heads_at_policy_scheme.csv"],
        ["Published SurGen-EV tree", "unaffected — all five heads remain on "
                                     "frozen τ_TCGA"],
        ["SurGen-EV promotion candidate", "unaffected — still fixed at k=35 "
                                          "(experiments/surgen_ev_promotion/)"],
    ], header=["Field", "Value"], widths=[1.8, 4.7])

    OUT_DOCX.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT_DOCX))
    print(f"wrote {OUT_DOCX}  ({OUT_DOCX.stat().st_size/1024:.0f} KB)")
    print(f"wrote {chart_auroc}")
    print(f"wrote {chart_bacc}")


if __name__ == "__main__":
    main()
