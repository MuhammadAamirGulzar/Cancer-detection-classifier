"""All approaches side by side: AUROC per slide representation (rows) and encoder (columns), TCGA-CV and PAIP-EV.

Fixed rules (Averaging, Caption-14/15, TTC) use the LR head, the best head in every setting, from the final analysis
(crop-level grouping). MIL rows are the per-encoder MIL runs; an encoder whose MIL run is not finished shows a dash.

Writes results/overview.csv and results/fig_overview.png.

  <conda>/envs/exaonepath/python.exe experiments/attention_mil/07_overview.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ENC = [("H-Optimus-1", "H-Optimus-1"), ("UNI2", "UNI2-h"), ("Virchow2", "Virchow2"), ("Conch1_5", "CONCH 1.5"),
       ("ConchV1", "CONCH v1")]
FIXED = [("Averaging", "Averaging"), ("Caption_based_aggregation", "Caption-14"),
         ("Caption_based_aggregation_15_classes", "Caption-15"), ("Tissue_Type_Clustering", "Tissue-type clustering")]
MIL = [("abmil", "Attention MIL"), ("topk", "Top-k MIL"), ("patchsum", "Patch-sum (control)")]
SETTINGS = ["TCGA-CV", "PAIP-EV"]
GROUP = {"Averaging": "Fixed rule + LR", "Caption-14": "Fixed rule + LR", "Caption-15": "Fixed rule + LR",
         "Tissue-type clustering": "Fixed rule + LR", "Attention MIL": "Learned (new)", "Top-k MIL": "Learned (new)",
         "Patch-sum (control)": "Control"}
INK, MUTED, SURFACE = "#1F2A44", "#5B6475", "#FBFBF8"
RAMP = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
                                                  "#0d366b"])
LO, HI = 0.75, 0.93


def table():
    ci = pd.read_csv(REPO / "experiments" / "final_analysis" / "results" / "ci_per_config.csv")
    ci = ci[ci.classifier == "lin"]
    rows = []
    for st in SETTINGS:
        for meth, name in FIXED:
            for m, _ in ENC:
                v = ci[(ci.experiment == st) & (ci.method == meth) & (ci.model == m)].auroc
                rows.append({"setting": st, "approach": name, "model": m, "auroc": float(v.iloc[0])})
        for meth, name in MIL:
            for m, _ in ENC:
                f = HERE / "results" / m / meth / "compare.csv"
                v = np.nan
                if f.exists():
                    c = pd.read_csv(f)
                    v = float(c[c.setting == st].iloc[0].auroc)
                rows.append({"setting": st, "approach": name, "model": m, "auroc": v})
    return pd.DataFrame(rows)


def main():
    df = table()
    df.to_csv(HERE / "results" / "overview.csv", index=False)
    names = [n for _, n in FIXED] + [n for _, n in MIL]
    ypos, y = [], 0.0
    for i, n in enumerate(names):
        if i and GROUP[n] != GROUP[names[i - 1]]:
            y += 0.45
        ypos.append(y)
        y += 1
    plt.rcParams.update({"font.family": "Arial", "font.size": 15})
    fig, axes = plt.subplots(1, 2, figsize=(16.6, 6.6), facecolor=SURFACE)
    cols = [lab for _, lab in ENC] + ["Mean"]
    for ax, st in zip(axes, SETTINGS):
        ax.set_facecolor(SURFACE)
        M = df[df.setting == st].pivot(index="approach", columns="model", values="auroc").reindex(
            index=names, columns=[m for m, _ in ENC])
        done = M.notna().all(1)
        M["Mean"] = M.mean(1).where(done)
        colmax = M.loc[names[:6]].max(0)
        for i, n in enumerate(names):
            for j, c in enumerate(M.columns):
                v = M.loc[n, c]
                x0, y0 = j + (0.25 if c == "Mean" else 0), ypos[i]
                if np.isnan(v):
                    ax.add_patch(plt.Rectangle((x0 + 0.03, y0 + 0.04), 0.94, 0.92, color="#ECEAE3", lw=0))
                    ax.text(x0 + 0.5, y0 + 0.5, "running", ha="center", va="center", color=MUTED, fontsize=12)
                    continue
                t = np.clip((v - LO) / (HI - LO), 0, 1)
                ax.add_patch(plt.Rectangle((x0 + 0.03, y0 + 0.04), 0.94, 0.92, color=RAMP(t), lw=0))
                best = n in names[:6] and round(v, 3) == round(colmax[c], 3)
                ax.text(x0 + 0.5, y0 + 0.5, f"{v:.3f}", ha="center", va="center", fontsize=15,
                        color="white" if t > 0.5 else INK, fontweight="bold" if best else "normal")
                if best:
                    ax.add_patch(plt.Rectangle((x0 + 0.06, y0 + 0.07), 0.88, 0.86, fill=False, ec="#C2582E", lw=2.5))
        ax.set_xlim(0, len(M.columns) + 0.25)
        ax.set_ylim(ypos[-1] + 1.05, -0.05)
        ax.set_xticks([j + (0.25 if c == "Mean" else 0) + 0.5 for j, c in enumerate(M.columns)])
        ax.set_xticklabels(cols, fontsize=14, color=INK)
        ax.xaxis.tick_top()
        ax.set_yticks([p + 0.5 for p in ypos])
        ax.set_yticklabels(names if ax is axes[0] else [], fontsize=15, color=INK)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{st}  ·  {'TCGA 4-fold CV, 413 slides' if st == 'TCGA-CV' else 'train TCGA → test PAIP, 78 slides'}",
                     fontsize=17, color=INK, fontweight="bold", loc="left", pad=36)
    fig.text(0.008, 0.035, "AUROC. Darker = higher (0.75 or below = lightest). Orange box = best in its column. "
             "Fixed rules use the LR head; MIL learns its own pooling.", fontsize=13, color=MUTED)
    fig.subplots_adjust(left=0.135, right=0.995, top=0.86, bottom=0.08, wspace=0.05)
    fig.savefig(HERE / "results" / "fig_overview.png", dpi=150, facecolor=SURFACE)
    print(df.pivot_table(index=["setting", "approach"], columns="model", values="auroc").round(3).to_string())


if __name__ == "__main__":
    main()
