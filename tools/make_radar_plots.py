"""Radar plots, one figure per (experiment x metric) (work order Task 4.4).

Design decisions, and why
-------------------------
**Faceted by foundation model, not overlaid.** Five foundation models drawn on one
radar means five series visible simultaneously - the "all pairs" case. No
5-colour subset of the palette clears the colourblind-separation floors in dark
mode (checked exhaustively: 11 of 56 five-subsets pass in light, **none** of those
11 passes in dark). So the models become small multiples and the *aggregation
methods* - the actual experimental variable - become the series.

**Four series, validated.** The aggregation methods use blue / yellow / magenta /
green, which passes every check in both modes on the all-pairs list:

    light  CVD dE 13.0 worst pair, normal-vision dE 19.6   -> ALL CHECKS PASS
    dark   CVD dE  6.9 worst pair, normal-vision dE 19.3   -> ALL CHECKS PASS

Two obligations come with that result and are discharged here:
  * the dark-mode CVD warning (green vs yellow, dE 6.9) sits in the 6-8 band,
    which is legal *only* with secondary encoding - so every series also carries a
    distinct line style and marker shape, never colour alone;
  * the light-mode contrast warning on yellow and magenta triggers the relief
    rule - a legend is always present and the same numbers exist as a table in
    best_of_all_exps_metric.xlsx.

Spokes are the five classifiers. Every title carries the canonical experiment
name and the test-set N, which is the Task 4.4 acceptance criterion.

Run:  python tools/make_radar_plots.py [--metric AUROC|BalAcc] [--archive]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SLIDE_CLS = REPO_ROOT / "slide_classification"
PLOTS_ROOT = REPO_ROOT / "Analysis_and_Visualization" / "Radar_Plots"
sys.path.insert(0, str(REPO_ROOT / "tools"))

from build_report import discover, _rows_from, SLIDE_LEVEL_ENCODERS  # noqa: E402

CLASSIFIERS = ["lin", "ann", "knn", "proto", "rf"]
CLASSIFIER_LABELS = {"lin": "Logistic", "ann": "ANN", "knn": "KNN",
                     "proto": "ProtoNet", "rf": "RandomForest"}
AGG_METHODS = ["Averaging", "Caption_based_aggregation",
               "Caption_based_aggregation_15_classes", "Tissue_Type_Clustering"]
AGG_LABELS = {"Averaging": "Averaging",
              "Caption_based_aggregation": "Caption-based (14)",
              "Caption_based_aggregation_15_classes": "Caption-based (15)",
              "Tissue_Type_Clustering": "Tissue-type clustering"}
MODELS = ["H-Optimus-1", "Conch1_5", "UNI2", "Virchow2", "ConchV1"]

# Validated categorical slots (see module docstring for the validator output).
SERIES_LIGHT = ["#2a78d6", "#eda100", "#e87ba4", "#008300"]
SERIES_DARK = ["#3987e5", "#c98500", "#d55181", "#008300"]
# Secondary encoding - required by the dark-mode CVD warning, and good practice
# regardless: identity is never carried by colour alone.
SERIES_STYLE = ["-", "--", "-.", ":"]
SERIES_MARKER = ["o", "s", "^", "D"]

THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e",
              "grid": "#d8d7d2", "series": SERIES_LIGHT},
    "dark": {"surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7",
             "grid": "#3a3a38", "series": SERIES_DARK},
}


def load_table() -> pd.DataFrame:
    rows: List[dict] = []
    for rec in discover(SLIDE_CLS):
        rows.extend(_rows_from(rec))
    if not rows:
        raise SystemExit("no result files found - run the Phase 3 experiments first")
    df = pd.DataFrame(rows)
    # For EV experiments keep the primary variant only; the other two are
    # robustness/legacy rows and would triple-plot the same model.
    if "Variant" in df.columns:
        keep = df["Variant"].isin(["default", "tcga_full"])
        # fall back to fold_average where tcga_full is not available yet
        for (exp, me, mo, clf), grp in df.groupby(["Experiment", "Method", "Model", "Classifier"]):
            if not keep[grp.index].any() and len(grp):
                keep[grp.index[0]] = True
        df = df[keep]
    return df


def _panel(ax, sub: pd.DataFrame, metric: str, theme: dict, model: str,
           show_legend: bool) -> None:
    angles = np.linspace(0, 2 * np.pi, len(CLASSIFIERS), endpoint=False).tolist()
    angles += angles[:1]

    ax.set_facecolor(theme["surface"])
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([CLASSIFIER_LABELS[c] for c in CLASSIFIERS],
                       color=theme["text"], fontsize=8)
    ax.set_ylim(0.4, 1.0)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.5", "", "0.7", "", "0.9", ""],
                       color=theme["muted"], fontsize=6.5)
    # Recessive grid.
    ax.grid(color=theme["grid"], linewidth=0.6, alpha=0.9)
    ax.spines["polar"].set_color(theme["grid"])
    ax.spines["polar"].set_linewidth(0.6)

    for i, method in enumerate(AGG_METHODS):
        vals = []
        for clf in CLASSIFIERS:
            row = sub[(sub["Method"] == method) & (sub["Classifier"] == clf)]
            vals.append(float(row[metric].iloc[0]) if len(row) and pd.notna(
                row[metric].iloc[0]) else np.nan)
        if all(np.isnan(v) for v in vals):
            continue
        vals += vals[:1]
        ax.plot(angles, vals, color=theme["series"][i], linewidth=2.0,
                linestyle=SERIES_STYLE[i], marker=SERIES_MARKER[i], markersize=4.5,
                markeredgecolor=theme["surface"], markeredgewidth=0.8,
                label=AGG_LABELS[method], zorder=3 - i * 0.1)

    ax.set_title(model, color=theme["text"], fontsize=10, pad=18, fontweight="600")


def make_figure(df: pd.DataFrame, experiment: str, metric: str, theme_name: str,
                out_dir: Path) -> Optional[Path]:
    theme = THEMES[theme_name]
    sub = df[(df["Experiment"] == experiment) &
             (~df["Method"].isin(SLIDE_LEVEL_ENCODERS))]
    if sub.empty:
        return None

    models = [m for m in MODELS if m in set(sub["Model"])]
    if not models:
        return None

    n = len(models)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 4.4 * nrows),
                             subplot_kw={"polar": True})
    fig.patch.set_facecolor(theme["surface"])
    axes = np.atleast_1d(axes).ravel()

    n_test = sub["N_test"].dropna()
    n_test = int(n_test.iloc[0]) if len(n_test) else "?"

    for j, model in enumerate(models):
        _panel(axes[j], sub[sub["Model"] == model], metric, theme, model,
               show_legend=False)
    for j in range(len(models), len(axes)):
        axes[j].set_visible(False)

    metric_label = {"AUROC": "AUROC", "BalAcc": "Balanced accuracy"}.get(metric, metric)
    # Task 4.4 acceptance: canonical experiment name AND test-set N in every title.
    fig.suptitle(f"{experiment} — {metric_label} by aggregation method  (N = {n_test})",
                 color=theme["text"], fontsize=13, fontweight="600", y=0.99)

    # Figure-level legend built from explicit proxies, not from one panel's
    # artists: a panel missing a method (partial coverage) would otherwise emit a
    # short or mislabeled legend. Identity is carried by colour AND line style AND
    # marker, so the dark-mode CVD warning is discharged.
    present = set(sub["Method"])
    handles = [Line2D([0], [0], color=theme["series"][i], linestyle=SERIES_STYLE[i],
                      marker=SERIES_MARKER[i], markersize=5, linewidth=2.0,
                      markeredgecolor=theme["surface"], markeredgewidth=0.8,
                      label=AGG_LABELS[m])
               for i, m in enumerate(AGG_METHODS) if m in present]
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)),
                   frameon=False, fontsize=9, labelcolor=theme["text"],
                   bbox_to_anchor=(0.5, -0.005))

    # Generous vertical room: polar spoke labels sit outside the axes and will
    # collide with the next row's panel title otherwise.
    fig.subplots_adjust(hspace=0.30, wspace=0.32,
                        top=0.90, bottom=0.10 if nrows > 1 else 0.14)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{experiment}_{metric}_{theme_name}.png"
    fig.savefig(path, dpi=200, facecolor=theme["surface"], bbox_inches="tight",
                pad_inches=0.3)
    plt.close(fig)
    return path


def archive_old_plots() -> None:
    """Task 4.4: archive 3_agg_methods / 4_agg_methods - they predate every
    Phase 1 and Phase 2 correction."""
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
    ap.add_argument("--metrics", default="AUROC,BalAcc")
    ap.add_argument("--themes", default="light,dark")
    ap.add_argument("--archive", action="store_true",
                    help="archive the pre-correction 3_agg_methods/4_agg_methods folders")
    args = ap.parse_args()

    if args.archive:
        archive_old_plots()

    df = load_table()
    experiments = [e for e in ["TCGA-CV", "PAIP-IV", "PAIP-EV", "SurGen-CV", "SurGen-EV"]
                   if e in set(df["Experiment"])]
    print(f"experiments with results: {experiments}")

    written = 0
    for exp in experiments:
        for metric in args.metrics.split(","):
            for theme in args.themes.split(","):
                p = make_figure(df, exp, metric, theme, PLOTS_ROOT / exp)
                if p:
                    print(f"  wrote {p.relative_to(REPO_ROOT)}")
                    written += 1
    print(f"{written} figure(s) written under {PLOTS_ROOT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
