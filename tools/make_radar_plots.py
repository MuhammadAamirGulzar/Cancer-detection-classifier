"""Radar plots, one figure per (experiment x metric).

Design source
-------------
This reproduces the radar layout the project already uses in
``Analysis_and_Visualization/Analysis.ipynb`` (the "Below script will generate
radar plots from the best of all exps metrics excel file" cell), rather than the
faceted small-multiple design that previously lived here. Owner decision: the
notebook figure is the one that goes in the paper, so the tool emits the same
thing.

The layout, unchanged from the notebook:

  * spokes  = aggregation method x classifier, grouped so each aggregation owns
              a contiguous wedge of the circle;
  * series  = the five foundation models, overlaid as coloured lines;
  * wedges  = a pastel background per aggregation, matched to the spoke label
              boxes so a reader can tell at a glance which block they are in;
  * radius  = fixed 0.0-1.0 with numbers printed along every spoke, so figures
              from different experiments stay directly comparable.

What changed relative to the notebook
-------------------------------------
1. **Data source.** The notebook parsed the *old* workbook shape - a ``Metric``
   column holding ``lin_auroc_Averaging`` strings and one ``<Model>_N`` column
   per encoder. ``build_report.py`` now emits a tidy frame
   (``Experiment, Method, Model, Classifier, Variant, AUROC, BalAcc, ...``), so
   the extraction here reads that instead. Pointing the notebook cell at the
   current workbook would silently plot zeros everywhere - every ``Metric``
   lookup misses.
2. **Four aggregations, not three.** ``Caption_based_aggregation_15_classes``
   was commented out in the notebook; it has results now, so it is included by
   default. ``--methods`` overrides.
3. **Variant selection.** EV experiments carry three variants. Only the primary
   (``tcga_full``) is plotted, otherwise every model is drawn three times.
4. **N in the title.** Kept from the previous tool - reviewers check it, and the
   SurGen H-Optimus-1 rows genuinely differ (613 vs 622).

Run:  python tools/make_radar_plots.py [--metrics AUROC,BalAcc] [--archive]
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SLIDE_CLS = REPO_ROOT / "slide_classification"
PLOTS_ROOT = REPO_ROOT / "Analysis_and_Visualization" / "Radar_Plots"
sys.path.insert(0, str(REPO_ROOT / "tools"))

from build_report import (discover, _rows_from, SLIDE_LEVEL_ENCODERS,  # noqa: E402
                          fill_threshold_scheme)

# ---------------------------------------------------------------- notebook config

AGG_METHODS = ["Caption_based_aggregation",
               "Caption_based_aggregation_15_classes",
               "Averaging",
               "Tissue_Type_Clustering"]

AGG_ABBREV = {"Caption_based_aggregation": "CBA",
              "Caption_based_aggregation_15_classes": "CBA15",
              "Averaging": "AVG",
              "Tissue_Type_Clustering": "TTC"}

AGGREGATION_COLORS = {"Caption_based_aggregation": "#d0e0ff",
                      "Caption_based_aggregation_15_classes": "#fff0b3",
                      "Averaging": "#ffe0e0",
                      "Tissue_Type_Clustering": "#c0ffc0"}

FOUNDATION_MODELS = ["H-Optimus-1", "Conch1_5", "UNI2", "Virchow2", "ConchV1"]

MODEL_DISPLAY_NAMES = {"H-Optimus-1": "H-Optimus-1", "Conch1_5": "CONCH1.5",
                       "UNI2": "UNI2", "Virchow2": "Virchow2", "ConchV1": "CONCH"}

MODEL_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#6b3f00"]

CLASSIFIERS = ["lin", "ann", "knn", "proto", "rf"]
CLASSIFIER_LABEL_MAP = {"lin": "Linear", "ann": "ANN", "knn": "KNN",
                        "proto": "Proto", "rf": "RF"}

# The notebook called AUROC "AUC"; accept both so existing habits keep working.
METRIC_ALIASES = {"AUC": "AUROC", "AUROC": "AUROC", "BalAcc": "BalAcc",
                  "Acc": "Acc", "MacroF1": "MacroF1", "WF1": "WeightedF1",
                  "WeightedF1": "WeightedF1"}

# Primary variant per experiment - everything else is a robustness/legacy row.
# thresholds.py rule 2: internal experiments report at 0.5, the flat "default"
# row. PAIP-IV's `at_tau_train` is a supplementary operating point, and it
# stores no AUROC of its own (AUROC is threshold-free).
PRIMARY_VARIANT = {"TCGA-CV": "default", "SurGen-CV": "default",
                   "PAIP-IV": "default", "PAIP-EV": "tcga_full",
                   "SurGen-EV": "tcga_full"}

EXPERIMENT_ORDER = ["TCGA-CV", "PAIP-IV", "PAIP-EV", "SurGen-CV", "SurGen-EV"]


def close_circle(values: List[float]) -> List[float]:
    return list(values) + [values[0]]


def describe_n(sub: pd.DataFrame) -> str:
    """Cohort size for the title, naming any combination that differs.

    A bare "613/622" was read as "N is 613", which is wrong - it is one figure
    covering many combinations. State the cohort size, then name the exceptions,
    so nobody has to guess which reading is intended.
    """
    counts = (sub.groupby(["Method", "Model"])["N_test"].first().dropna().astype(int))
    if counts.empty:
        return "?"
    main = int(counts.mode().iloc[0])
    odd = counts[counts != main]
    if odd.empty:
        return str(main)
    parts = [f"{AGG_ABBREV.get(m, m)}/{mo}: {n}" for (m, mo), n in odd.items()]
    return f"{main}  (except {'; '.join(parts)})"


def load_table() -> pd.DataFrame:
    rows: List[dict] = []
    for rec in discover(SLIDE_CLS):
        rows.extend(_rows_from(rec))
    if not rows:
        raise SystemExit("no result files found - run the experiments first")
    return fill_threshold_scheme(pd.DataFrame(rows))


def extract_values(df: pd.DataFrame, methods: List[str], metric: str):
    """Spoke labels, their aggregation group, and one series per model.

    Missing cells become 0.0 - the notebook's convention, so a gap reads
    visibly as a gap rather than as an interpolated line.
    """
    axis_labels: List[str] = []
    agg_groups: List[str] = []
    models_data: Dict[str, List[float]] = {m: [] for m in FOUNDATION_MODELS}

    for agg in methods:
        short = AGG_ABBREV.get(agg, agg)
        for clf in CLASSIFIERS:
            axis_labels.append(
                f"{CLASSIFIER_LABEL_MAP.get(clf, clf)}-{metric}\n({short})")
            agg_groups.append(agg)
            for fm in FOUNDATION_MODELS:
                sel = df[(df["Method"] == agg) & (df["Model"] == fm) &
                         (df["Classifier"] == clf)]
                val = 0.0
                if not sel.empty and metric in sel.columns:
                    v = sel[metric].iloc[0]
                    val = round(float(v), 4) if pd.notna(v) else 0.0
                models_data[fm].append(val)
    return axis_labels, agg_groups, models_data


def describe_schemes(sub: pd.DataFrame) -> str:
    """A caption line when one panel mixes threshold schemes, else ''.

    A radar with KNN scored at k=35 next to LR at the frozen tau is not
    comparing like with like, and the spoke that moved is the one a reader will
    ask about first. Saying so on the figure costs one line; leaving it to the
    surrounding prose means the image travels without it.
    """
    if "Threshold_scheme" not in sub.columns:
        return ""
    per_clf = sub.groupby("Classifier")["Threshold_scheme"].agg(set)
    schemes = {s for v in per_clf for s in v}
    if len(schemes) < 2:
        return ""
    def label(s):
        m = re.fullmatch(r"refit_tau_k(\d+)", s)
        return f"k={m.group(1)} + TCGA-refit tau" if m else s
    named = {"quantile_rate_matched": "rate-matched tau"}
    parts = [f"{clf.upper()} {named.get(s, label(s))}"
             for clf, v in sorted(per_clf.items()) for s in v
             if s != "frozen_tau_TCGA"]
    frozen = sorted(c.upper() for c, v in per_clf.items()
                    if v == {"frozen_tau_TCGA"})
    return (f"Two threshold schemes: {', '.join(parts)}; "
            f"{'/'.join(frozen)} at the frozen TCGA tau")


def make_figure(df: pd.DataFrame, experiment: str, metric: str,
                methods: List[str], out_dir: Path,
                value_set: str = "frozen") -> Optional[Path]:
    sub = df[df["Experiment"] == experiment]
    variant = PRIMARY_VARIANT.get(experiment)
    if variant and variant in set(sub["Variant"]):
        sub = sub[sub["Variant"] == variant]
    # TITAN and PRISM are slide-level encoders and are their own aggregation
    # "method" - they have no place on a foundation-model radar.
    sub = sub[~sub["Method"].isin(SLIDE_LEVEL_ENCODERS)]
    if sub.empty:
        return None

    # `corrected` reads the columns ev_runner writes under
    # --threshold-mode=corrected. They are absent on a frozen run, so the
    # figure is skipped rather than silently drawn from the frozen numbers.
    col = metric if value_set == "frozen" else f"{metric}_corrected"
    if col not in sub.columns or sub[col].notna().sum() == 0:
        if value_set != "frozen":
            print(f"  [skip] {experiment}/{metric}: no {col} "
                  f"(run ev_runner --threshold-mode corrected first)")
            return None
        col = metric

    axis_labels, agg_groups, models_data = extract_values(sub, methods, col)
    if not axis_labels:
        return None

    n_txt = describe_n(sub)
    scheme_note = describe_schemes(sub)

    max_r, label_position = 1.0, 1.13
    n_spokes = len(axis_labels)
    angle_step = 2 * np.pi / n_spokes
    angles = np.linspace(0, 2 * np.pi, n_spokes, endpoint=False) + angle_step / 2
    angles_closed = np.concatenate((angles, [angles[0]]))

    fig = plt.figure(figsize=(24, 24))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    group_span = 2 * np.pi / len(methods)
    for i, agg in enumerate(methods):
        t0, t1 = i * group_span, (i + 1) * group_span
        arc = np.linspace(t0, t1, 200)
        ax.fill(np.concatenate(([t0], arc, [t1])),
                np.concatenate(([0], np.full(len(arc), max_r), [0])),
                color=AGGREGATION_COLORS.get(agg, "#f0f0f0"), alpha=0.4, zorder=0)

    for (fm, values), color in zip(models_data.items(), MODEL_COLORS):
        stats = close_circle(values)
        ax.plot(angles_closed, stats, color=color,
                label=MODEL_DISPLAY_NAMES.get(fm, fm),
                linewidth=3, marker="o", markersize=6, zorder=5)
        ax.fill(angles_closed, stats, color=color, alpha=0.06, zorder=4)

    ax.set_xticks(angles)
    ax.set_xticklabels([])
    for angle, label, agg in zip(angles, axis_labels, agg_groups):
        ax.text(angle, label_position, label, ha="center", va="center",
                fontsize=14, fontweight="bold", rotation=0, clip_on=False,
                bbox=dict(facecolor=AGGREGATION_COLORS.get(agg, "#f0f0f0"),
                          edgecolor="black", boxstyle="square,pad=0.5",
                          linewidth=1.5))

    ax.set_yticklabels([])
    ax.set_ylim(0.0, max_r)
    r_ticks = np.arange(0.50, max_r, 0.10)
    ax.set_yticks(r_ticks)
    # The notebook used fontsize 24 for 15 spokes; 20 spokes need a smaller face
    # or the numbers collide with their neighbours.
    tick_fs = 24 if n_spokes <= 15 else 15
    for angle in angles:
        for r in r_ticks:
            ax.text(angle, r, f"{r:.2f}", ha="center", va="center",
                    fontsize=tick_fs, color="black", zorder=6, clip_on=False)

    # Legends sit in the top corners; the title goes on a band above them, so a
    # long N note (SurGen carries two exceptions) can never run underneath.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.995),
               title="Models", fontsize=18, title_fontsize=20, borderpad=1.2,
               framealpha=0.95, edgecolor="black", fancybox=True, shadow=True)

    agg_handles = [mpatches.Patch(facecolor=AGGREGATION_COLORS.get(a, "#f0f0f0"),
                                  edgecolor="black",
                                  label=f"{AGG_ABBREV.get(a, a)} = {a}")
                   for a in methods]
    fig.legend(handles=agg_handles, loc="upper left", bbox_to_anchor=(0.02, 0.995),
               title="Aggregation Methods (Key)", fontsize=16, title_fontsize=18,
               borderpad=1.2, framealpha=0.95, edgecolor="black",
               fancybox=True, shadow=True)

    tag = "" if value_set == "frozen" else "  [corrected thresholds]"
    fig.text(0.5, 1.055, f"Radar Plot - {experiment} | Metric: {metric}{tag}",
             ha="center", va="bottom", fontsize=30, fontweight="bold")
    fig.text(0.5, 1.030, f"N = {n_txt}", ha="center", va="bottom", fontsize=19)
    if scheme_note:
        fig.text(0.5, 1.010, scheme_note, ha="center", va="bottom",
                 fontsize=15, style="italic", color="#444444")

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if value_set == "frozen" else "_corrected"
    path = out_dir / f"{experiment}_{metric}_radar{suffix}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.5)
    plt.close(fig)
    return path


def archive_old_plots() -> None:
    stamp = date.today().strftime("%Y%m%d")
    for name in ("3_agg_methods", "4_agg_methods"):
        src = PLOTS_ROOT / name
        if not src.is_dir():
            continue
        dst = PLOTS_ROOT / f"{name}_ARCHIVED_{stamp}"
        if dst.exists():
            continue
        try:
            src.rename(dst)
        except OSError:
            shutil.copytree(src, dst)
        print(f"[ARCHIVE] Radar_Plots/{name} -> {dst.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="AUROC,BalAcc",
                    help="comma list; AUC is accepted as an alias for AUROC")
    ap.add_argument("--methods", default=",".join(AGG_METHODS),
                    help="aggregation methods, in spoke order")
    ap.add_argument("--experiments", default="",
                    help="comma list; default is every experiment with results")
    ap.add_argument("--values", default="frozen", choices=["frozen", "corrected", "both"],
                    help="which numbers to plot; corrected goes to a separate "
                         "<exp>_corrected/ folder so both sets can be kept")
    ap.add_argument("--archive", action="store_true",
                    help="archive the pre-correction 3_agg_methods/4_agg_methods folders")
    args = ap.parse_args()

    if args.archive:
        archive_old_plots()

    df = load_table()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    wanted = [e.strip() for e in args.experiments.split(",") if e.strip()]
    experiments = wanted or [e for e in EXPERIMENT_ORDER
                             if e in set(df["Experiment"])]
    print(f"experiments with results: {experiments}")

    written = 0
    for exp in experiments:
        for raw in args.metrics.split(","):
            raw = raw.strip()
            metric = METRIC_ALIASES.get(raw, raw)
            if metric not in df.columns:
                print(f"  [skip] unknown metric {raw!r}")
                continue
            for value_set in (["frozen", "corrected"] if args.values == "both"
                              else [args.values]):
                out = (PLOTS_ROOT / exp if value_set == "frozen"
                       else PLOTS_ROOT / f"{exp}_corrected")
                path = make_figure(df, exp, metric, methods, out, value_set)
                if path:
                    print(f"  wrote {path.relative_to(REPO_ROOT)}")
                    written += 1
    print(f"{written} figure(s) written under {PLOTS_ROOT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
