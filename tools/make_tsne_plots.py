"""t-SNE of the slide-level feature space, per foundation model.

Purpose
-------
The transfer results said the encoders differ in how far the target cohorts sit
from TCGA (normalised centroid distance: UNI2 1.27, H-Optimus-1 1.42, Virchow2
1.57, Conch1_5 1.87, ConchV1 2.03) and that target slides land 3-4x further from
their nearest TCGA neighbour than TCGA slides sit from each other. This renders
that directly, and writes the numbers alongside so the picture can be checked
rather than just admired.

Two panels per foundation model:

  left   coloured by COHORT   - is the target cohort a separate island, or does
                               it overlap TCGA? An island is domain shift, and
                               predicts the threshold-transfer failure.
  right  coloured by LABEL    - are MSI-H slides separable at all in this space?
                               If they are not, no classifier downstream can be.

Read-only: opens the aggregated .pt features and writes only into
``Analysis_and_Visualization/TSNE/``.

Method notes
------------
* Features are L2-normalised first, matching what the distance-based heads see
  (``model_io._prep``), then PCA to ``--pca`` dims before t-SNE. That is the
  standard pipeline - t-SNE on raw 20k-dim vectors is dominated by noise.
* Cohorts are embedded TOGETHER in one t-SNE, which is the only way the relative
  positions mean anything. Embedding each cohort separately and putting the
  panels side by side would say nothing about overlap.
* t-SNE preserves local neighbourhoods, not global distances - cluster sizes and
  inter-cluster gaps are not to scale. The companion CSV carries the metrics that
  ARE quantitative; read the plot for structure and the CSV for magnitude.

Run:
  python tools/make_tsne_plots.py
  python tools/make_tsne_plots.py --methods Averaging,Tissue_Type_Clustering
  python tools/make_tsne_plots.py --models UNI2 --perplexity 50
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from torch.nn.functional import normalize

REPO_ROOT = Path(__file__).resolve().parents[1]
SLIDE_CLS = REPO_ROOT / "slide_classification"
OUT_ROOT = REPO_ROOT / "Analysis_and_Visualization" / "TSNE"
sys.path.insert(0, str(SLIDE_CLS))
warnings.filterwarnings("ignore")

import data_layer as dl                    # noqa: E402
from config import paths as P              # noqa: E402

COHORTS = ["tcga", "paip", "surgen"]
COHORT_LABEL = {"tcga": "TCGA (train)", "paip": "PAIP", "surgen": "SurGen"}

#: Identity is never carried by colour alone - each cohort also gets its own
#: marker, so the figure survives greyscale printing and colour-vision deficiency.
COHORT_STYLE = {
    "tcga":   dict(color="#2a78d6", marker="o", label="TCGA (train)"),
    "paip":   dict(color="#e08c00", marker="^", label="PAIP"),
    "surgen": dict(color="#c0399f", marker="s", label="SurGen"),
}
LABEL_STYLE = {
    0: dict(color="#9aa0a6", marker="o", label="non-MSI-H"),
    1: dict(color="#c0392b", marker="D", label="MSI-H"),
}

MODELS = ["UNI2", "H-Optimus-1", "Virchow2", "Conch1_5", "ConchV1"]


def load_cohort_feats(cohort: str, method: str, model: str):
    coh = dl.load_cohort(cohort, method, model, task="MSIH", verbose=False)
    X = normalize(coh.feats, dim=-1, p=2).numpy()
    return X, coh.labels.numpy()


def shift_metrics(X: np.ndarray, cohort_of: np.ndarray) -> Dict[str, float]:
    """The numbers the plot is a picture of."""
    out: Dict[str, float] = {}
    tc = X[cohort_of == "tcga"]
    d_tt = cosine_distances(tc, tc)
    np.fill_diagonal(d_tt, np.inf)
    out["tcga_nn_dist"] = float(d_tt.min(axis=1).mean())
    for c in ("paip", "surgen"):
        tgt = X[cohort_of == c]
        if not len(tgt):
            continue
        out[f"{c}_nn_dist"] = float(cosine_distances(tgt, tc).min(axis=1).mean())
        out[f"{c}_nn_ratio"] = round(out[f"{c}_nn_dist"] / out["tcga_nn_dist"], 3)
        sd = np.sqrt((tc.var(0) + tgt.var(0)) / 2) + 1e-9
        out[f"{c}_centroid_d"] = round(
            float(np.linalg.norm((tc.mean(0) - tgt.mean(0)) / sd) / np.sqrt(X.shape[1])), 3)
    return out


def panel(ax, emb, keys, style_map, title, theme_text="#0b0b0b",
          centroids: bool = True):
    """One scatter panel with real, readable axes.

    t-SNE axes carry no units and are not reproducible across seeds, so they are
    usually hidden. They are drawn here because they make the figure readable -
    you can cite a position, compare extents, and see how far apart two clusters
    sit - but the label says "arbitrary units" and the caption repeats it, so no
    one reads a coordinate as a measurement.

    ``set_aspect('equal')`` matters more than it looks: with independently scaled
    axes the SAME separation renders differently along x and y, so a reader
    comparing distances by eye is comparing the axis scaling as much as the data.
    """
    for key, st in style_map.items():
        m = keys == key
        if not m.any():
            continue
        ax.scatter(emb[m, 0], emb[m, 1], s=14, alpha=0.65,
                   c=st["color"], marker=st["marker"], linewidths=0,
                   label=f"{st['label']}  (n={int(m.sum())})", zorder=3)
        if centroids:
            cx, cy = emb[m, 0].mean(), emb[m, 1].mean()
            ax.scatter([cx], [cy], s=190, marker="X", c=st["color"],
                       edgecolors="#fcfcfb", linewidths=1.8, zorder=5)
            ax.annotate(st["label"].split(" ")[0], (cx, cy),
                        textcoords="offset points", xytext=(0, 13),
                        ha="center", fontsize=8.5, fontweight="bold",
                        color=st["color"], zorder=6,
                        path_effects=[pe.withStroke(linewidth=2.6,
                                                    foreground="#fcfcfb")])

    ax.set_title(title, fontsize=11, color=theme_text, pad=8)
    ax.set_xlabel("t-SNE dimension 1  (arbitrary units)", fontsize=8.5,
                  color="#52514e")
    ax.set_ylabel("t-SNE dimension 2  (arbitrary units)", fontsize=8.5,
                  color="#52514e")
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(labelsize=7.5, colors="#52514e", length=3, width=0.8)
    ax.grid(True, color="#e6e5e0", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color("#d8d7d2"); sp.set_linewidth(0.8)
    ax.legend(loc="best", fontsize=8, framealpha=0.92, borderpad=0.8)


def run_one(method: str, model: str, args) -> Dict[str, float]:
    Xs, ys, cs = [], [], []
    for c in COHORTS:
        try:
            X, y = load_cohort_feats(c, method, model)
        except Exception as exc:
            print(f"    [skip] {c}: {type(exc).__name__}")
            continue
        Xs.append(X); ys.append(y); cs.append(np.full(len(y), c))
    if len(Xs) < 2:
        print(f"    [skip] {method}/{model}: need >=2 cohorts")
        return {}

    X = np.vstack(Xs); y = np.concatenate(ys); cohort_of = np.concatenate(cs)
    metrics = shift_metrics(X, cohort_of)

    n_pca = min(args.pca, X.shape[0] - 1, X.shape[1])
    Xp = PCA(n_components=n_pca, random_state=args.seed).fit_transform(X)
    emb = TSNE(n_components=2, perplexity=args.perplexity, init="pca",
               learning_rate="auto", random_state=args.seed,
               max_iter=args.iters).fit_transform(Xp)

    # separability of the classes, measured in the SAME space the plot shows
    for c in COHORTS:
        m = cohort_of == c
        if m.sum() > 10 and len(np.unique(y[m])) > 1:
            metrics[f"{c}_label_silhouette"] = round(
                float(silhouette_score(Xp[m], y[m])), 4)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.0))
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
    panel(axes[0], emb, cohort_of, COHORT_STYLE, "Coloured by cohort")
    panel(axes[1], emb, y, LABEL_STYLE, "Coloured by MSI-H label",
      centroids=False)   # classes span all clusters; a mean is not a centre

    sub = "  ".join(f"{c}->TCGA nn x{metrics.get(f'{c}_nn_ratio','?')}"
                    for c in ("paip", "surgen") if f"{c}_nn_ratio" in metrics)
    fig.subplots_adjust(top=0.80)
    fig.suptitle(f"t-SNE - {model} | {method} | D={X.shape[1]}  (n={len(y)})",
                 fontsize=14, fontweight="bold", y=0.985)
    fig.text(0.5, 0.930, sub, ha="center", fontsize=9.5, color="#2b2a28")
    fig.text(0.5, 0.885,
             "Axes are t-SNE dimensions in arbitrary units; equal aspect, so "
             "on-screen distance is comparable within a panel.\n"
             "t-SNE preserves local neighbourhoods, not global scale - read "
             "magnitudes from tsne_shift_metrics.csv.   X = cohort centroid "
             "(left panel only).",
             ha="center", va="top", fontsize=8, color="#6b6a66", linespacing=1.5)

    out_dir = OUT_ROOT / method
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"tsne_{method}_{model}.png"
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    wrote {path.relative_to(REPO_ROOT)}")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="Averaging",
                    help="comma list; Averaging is the base embedding "
                         "(no aggregation confound)")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--pca", type=int, default=50,
                    help="PCA dims before t-SNE")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        for model in [m.strip() for m in args.models.split(",") if m.strip()]:
            if not P.is_combination_valid(method, model):
                continue
            print(f"  {method} / {model}")
            met = run_one(method, model, args)
            if met:
                rows.append({"method": method, "model": model, **met})

    if rows:
        df = pd.DataFrame(rows)
        csv = OUT_ROOT / "tsne_shift_metrics.csv"
        df.to_csv(csv, index=False)
        print(f"\nwrote {csv.relative_to(REPO_ROOT)}")
        cols = [c for c in ("model", "method", "tcga_nn_dist", "paip_nn_ratio",
                            "surgen_nn_ratio", "paip_centroid_d",
                            "surgen_centroid_d", "tcga_label_silhouette")
                if c in df.columns]
        print("\n=== domain shift, most transferable first ===")
        key = "surgen_centroid_d" if "surgen_centroid_d" in df else cols[-1]
        print(df[cols].sort_values(key).to_string(index=False))
    else:
        print("nothing plotted")


if __name__ == "__main__":
    main()
