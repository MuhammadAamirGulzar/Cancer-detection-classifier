"""What does ABMIL attend to? Attention enrichment by tissue type, and attention heatmaps.

Enrichment: each patch carries its published CONCH v1 zero-shot caption (15 classes, whole patch). Per class c and
slide group (MSI-H / non-MSI-H): enrichment = mean over slides of the attention mass on c / mean share of patches
in c. 1 = attended in proportion to its area; > 1 = the model looks there more than its area suggests. TCGA uses
the out-of-fold attention (models that never saw the slide), PAIP the external models.

Heatmaps: PAIP slides (external), patch positions coloured by relative attention N * a_i (1 = average patch), with
each slide's four most-attended patches below.

Writes results/<model>/<method>/{enrichment.csv, fig_enrichment.png, fig_heatmaps.png}.

  <conda>/envs/exaonepath/python.exe experiments/attention_mil/04_explain.py --model H-Optimus-1
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = REPO.parent
CAP = REPO / "slide_aggregation" / "caption_generation" / "Results"
CSV = {"tcga": "classification_results_tcga_15_classes.csv", "paip": "classification_results_paip_15_classes.csv"}
PATCHES = {"tcga": BASE / "dataset" / "patch_data", "paip": BASE / "paip_data" / "patch_data"}
NAMES = {"ADI": "adipose", "DEB": "debris", "TIL": "tumour-infiltrating lymphocytes", "PLC": "plasma cells",
         "LYA": "lymphoid aggregate", "LYM": "lymphocytes", "MUS": "muscle", "MUC": "mucin", "NORM": "normal mucosa",
         "ADE": "adenoma", "STR": "stroma", "CAR": "carcinoma", "PDC": "poorly diff. carcinoma",
         "SIG": "signet-ring carcinoma", "BACK": "background"}
INK, MUTED, SURFACE, GRID = "#1F2A44", "#5B6475", "#FBFBF8", "#E4E1D8"
C_MSI, C_NON = "#C2582E", "#2366B5"
ATTN = LinearSegmentedColormap.from_list("attn", ["#F3E9E0", "#E8A27C", "#C2582E", "#7A2E12"])
RES = None
XY = re.compile(r"_x(\d+)_y(\d+)_")


def bag_names(cohort, model, slide):
    return torch.load(HERE / "features" / cohort / model / f"{slide}.pt")["names"]


def enrichment(cohort, model, scores):
    lab = pd.read_csv(CAP / CSV[cohort], usecols=["patch_name", "label"])
    lab = dict(zip(lab.patch_name.str.replace(".png", "", regex=False), lab.label))
    rows = []
    for s, y in zip(scores.slide_id, scores.target):
        a = np.load(RES / "attention" / cohort / f"{s}.npy")
        names = bag_names(cohort, model, s)
        c = np.array([lab.get(n, "NA") for n in names])
        for k in NAMES:
            m = c == k
            rows.append({"cohort": cohort, "slide": s, "group": "MSI-H" if y == 1 else "non-MSI-H", "cls": k,
                         "attn": a[m].sum(), "share": m.mean()})
    d = pd.DataFrame(rows)
    g = d.groupby(["cohort", "group", "cls"]).agg(attn=("attn", "mean"), share=("share", "mean")).reset_index()
    g["enrichment"] = g.attn / g.share
    return g


def plot_enrichment(e, path):
    keep = e.groupby("cls").share.mean()
    keep = keep[keep >= 0.01].index
    e = e[e.cls.isin(keep)]
    order = (e[(e.cohort == "tcga") & (e.group == "MSI-H")].set_index("cls").enrichment.sort_values().index)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.4), dpi=120, sharey=True, facecolor=SURFACE)
    for ax, cohort, title in zip(axes, ("tcga", "paip"), ("TCGA (out-of-fold, 413 slides)", "PAIP (external, 78 slides)")):
        ax.set_facecolor(SURFACE)
        y = np.arange(len(order))
        for off, grp, col in ((0.2, "MSI-H", C_MSI), (-0.2, "non-MSI-H", C_NON)):
            v = e[(e.cohort == cohort) & (e.group == grp)].set_index("cls").enrichment.reindex(order)
            ax.barh(y + off, np.log2(v.to_numpy()), height=0.36, color=col, label=grp, edgecolor=SURFACE, linewidth=2)
        ax.axvline(0, color=MUTED, linewidth=1.2)
        ax.set_yticks(y, [f"{NAMES[c]} ({c})" for c in order], fontsize=15, color=INK)
        ax.set_title(title, fontsize=18, color=INK, loc="left", pad=12)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        ticks = [-2, -1, 0, 1, 2]
        ax.set_xticks(ticks, ["¼×", "½×", "1×", "2×", "4×"], fontsize=14, color=MUTED)
        ax.set_xlim(-2.6, 2.6)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].legend(frameon=False, fontsize=15, loc="lower right")
    fig.text(0.5, 0.01, "attention share ÷ area share (log scale; right of 1× = attended more than its area)",
             ha="center", fontsize=15, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_heatmaps(model, paip, path):
    pos = paip[paip.target == 1].sort_values("score", ascending=False).head(3)
    neg = paip[paip.target == 0].sort_values("score").head(1)
    pick = pd.concat([pos, neg])
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=SURFACE)
    gs = fig.add_gridspec(2, len(pick), height_ratios=[1.35, 1], hspace=0.12, wspace=0.08)
    scored = (RES / "scores").exists()
    if scored:                                           # top-k MIL: one colour scale shared by all slides
        allv = np.concatenate([np.load(RES / "scores" / "paip" / f"{s}.npy") for s in pick.slide_id])
        g_lo, g_hi = np.quantile(allv, 0.01), np.quantile(allv, 0.995)
    for j, r in enumerate(pick.itertuples()):
        names = bag_names("paip", model, r.slide_id)
        if scored:                                       # each patch's learned MSI-H score
            rel = np.load(RES / "scores" / "paip" / f"{r.slide_id}.npy")
            lo, hi = g_lo, g_hi
        else:                                            # ABMIL: relative attention N * a_i (1 = average patch)
            a = np.load(RES / "attention" / "paip" / f"{r.slide_id}.npy")
            rel = a * len(a)
            lo, hi = 0.0, np.quantile(rel, 0.995)
        xy = np.array([[int(m.group(1)), int(m.group(2))] for m in (XY.search(n) for n in names)])
        ax = fig.add_subplot(gs[0, j])
        ax.set_facecolor(SURFACE)
        o = np.argsort(rel)
        ax.scatter(xy[o, 0], -xy[o, 1], c=np.clip(rel[o], lo, hi), cmap=ATTN, vmin=lo, vmax=hi, s=6, marker="s", linewidths=0)
        ax.set_aspect("equal"); ax.axis("off")
        lab = "MSI-H" if r.target == 1 else "non-MSI-H"
        ax.set_title(f"{lab} · score {r.score:.2f}", fontsize=16, color=C_MSI if r.target == 1 else C_NON, pad=6)
        top = np.argsort(rel)[::-1][:4]
        sub = gs[1, j].subgridspec(2, 2, wspace=0.04, hspace=0.04)
        for t, i in enumerate(top):
            ax2 = fig.add_subplot(sub[t // 2, t % 2])
            ax2.imshow(Image.open(PATCHES["paip"] / r.slide_id / f"{names[i]}.png").convert("RGB"))
            ax2.axis("off")
    what = ("Learned MSI-H score per patch on external PAIP slides (darker = more MSI-H evidence); below: each slide’s four highest-scoring patches"
            if (RES / "scores").exists() else
            "Attention on external PAIP slides (darker = more attention); below: each slide’s four most-attended patches")
    fig.text(0.01, 0.97, what, fontsize=16, color=INK, va="top")
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--method", choices=["abmil", "topk"], default="abmil")
    a = ap.parse_args()
    global RES
    res = RES = HERE / "results" / a.model / a.method
    tcga = pd.read_csv(res / "tcga_oof.csv")
    paip = pd.read_csv(res / "paip_ev.csv")
    e = pd.concat([enrichment("tcga", a.model, tcga), enrichment("paip", a.model, paip)], ignore_index=True)
    e.to_csv(res / "enrichment.csv", index=False)
    plot_enrichment(e, res / "fig_enrichment.png")
    plot_heatmaps(a.model, paip, res / "fig_heatmaps.png")
    t = e.pivot_table(index="cls", columns=["cohort", "group"], values="enrichment").round(2)
    t["share_tcga"] = e[e.cohort == "tcga"].groupby("cls").share.mean().round(3)
    pd.set_option("display.width", 200)
    print(t.sort_values(("tcga", "MSI-H"), ascending=False).to_string())


if __name__ == "__main__":
    main()
