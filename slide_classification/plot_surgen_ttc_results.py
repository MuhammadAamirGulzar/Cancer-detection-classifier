import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_SURGEN = os.path.join(
    "/media/dp-psau/Datum/Aamir/Azfaar",
    "surgen_processing",
    "Surgen_TTC_exp", "Tissue_Type_Clustering",
    "Global",               # Phase 2 summaries live under Global/<task>/
    "1-MSIH",
)

MODEL_NAMES_CHART = ["h-optimus-1"]   # extend as more models are added

CLASSIFIER    = "ann"
RANK_METRIC   = "bacc"

# Must match k_values used in the experiment (max = 8, background excluded)
K_VALUES_SHOW = [3, 5, 8]

TISSUE_ABBREV = {
    "adipose":       "ADI",
    "debris":        "DEB",
    "lymphocyte":    "LYM",
    "mucin":         "MUC",
    "smooth_muscle": "MUS",
    "normal_mucosa": "NORM",
    "stroma":        "STR",
    "tumor":         "TUM",
}


def abbrev_combo(combo_str: str) -> str:
    """Replace tissue names in a '+'-separated combo string with abbreviations."""
    parts = [p.strip() for p in combo_str.split("+") if p.strip()]
    return "\n".join(TISSUE_ABBREV.get(p, p) for p in parts)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# Phase 2 writes one file per k: tissue_topk_summary_surgen_k{k}.xlsx
# Columns: foundation_model, combo, k, classifier, bacc, …
# ─────────────────────────────────────────────────────────────────────────────

def load_k(k_val: int) -> pd.DataFrame | None:
    """
    Load the single top-k combo result file for the given k.
    Returns a DataFrame filtered to the requested classifier,
    or None if the file doesn't exist / has no matching rows.
    """
    fname = f"tissue_topk_summary_surgen_k{k_val}.xlsx"
    path  = os.path.join(BASE_SURGEN, fname)

    if not os.path.exists(path):
        print(f"[WARN] Not found: {path}")
        return None

    df = pd.read_excel(path)
    if df.empty:
        print(f"[WARN] Empty: {path}")
        return None

    df["classifier"] = df["classifier"].astype(str)
    df_clf = df[df["classifier"].str.lower().str.startswith(CLASSIFIER.lower())].copy()

    if df_clf.empty:
        print(f"[WARN] No '{CLASSIFIER}' rows in {path}")
        return None

    print(f"[OK] k={k_val}: {len(df_clf)} '{CLASSIFIER}' rows across "
          f"{df_clf['foundation_model'].nunique()} models")
    return df_clf


# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 — Results across k values
#
# x-axis  : k values  (one group per k)
# bars    : one bar per foundation model inside each k group
# y-axis  : bacc
# No mean bar.
# Each bar is labelled with the abbreviated combo text inside the bar.
# ─────────────────────────────────────────────────────────────────────────────

def plot_across_k(
    k_data: dict,   # {k_val: DataFrame}
    metric: str = RANK_METRIC,
):
    valid_k = sorted([k for k, df in k_data.items() if df is not None])
    if not valid_k:
        print("[WARN] No valid k data to plot.")
        return

    n_models = len(MODEL_NAMES_CHART)
    palette  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    bar_w   = 0.13
    group_w = n_models * bar_w + 0.08
    x       = np.arange(len(valid_k)) * group_w

    fig, ax = plt.subplots(figsize=(max(14, len(valid_k) * 3.5), 7))

    for mi, model_name in enumerate(MODEL_NAMES_CHART):
        color  = palette[mi % len(palette)]
        offset = (mi - n_models / 2 + 0.5) * bar_w

        for ki, k_val in enumerate(valid_k):
            df = k_data[k_val]
            if df is None:
                continue
            sub = df[df["foundation_model"] == model_name]
            if sub.empty:
                continue

            val   = sub[metric].mean()
            combo = sub["combo"].iloc[0]
            xpos  = x[ki] + offset

            ax.bar(xpos, val, width=bar_w - 0.01,
                   color=color, edgecolor="white", linewidth=0.8,
                   label=model_name if ki == 0 else "")

            # Value above bar
            ax.text(xpos, val + 0.005, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=6.5,
                    fontweight="bold", color="#222222")

            # Abbreviated combo inside bar
            abbrev = abbrev_combo(combo)
            if val > 0.08:
                ax.text(xpos, val * 0.45, abbrev,
                        ha="center", va="center",
                        fontsize=6, color="white", fontweight="bold",
                        linespacing=1.3, clip_on=True,
                        multialignment="center")

    # x-tick labels: show k value and which tissues it contains
    xtick_labels = []
    for k_val in valid_k:
        df = k_data[k_val]
        if df is not None and not df.empty:
            combo  = df["combo"].iloc[0]
            abbrev = abbrev_combo(combo)
            xtick_labels.append(f"k={k_val}\n{abbrev}")
        else:
            xtick_labels.append(f"k={k_val}")

    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, fontsize=8, linespacing=1.4)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_title(
        f"SurGen Top-k Fixed Combo Results (9 classes) | "
        f"Classifier: {CLASSIFIER.upper()} | Metric: {metric.upper()}\n"
        f"k values shown: {valid_k}  —  each k uses globally top-k tissue classes by avg bacc",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    handles, lbls = ax.get_legend_handles_labels()
    seen = dict(zip(lbls, handles))
    ax.legend(seen.values(), seen.keys(), fontsize=9,
              loc="upper right", framealpha=0.9)

    fig.tight_layout()
    k_str    = "_".join(str(k) for k in valid_k)
    out_path = os.path.join(os.getcwd(), f"surgen_across_k_{k_str}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[OK] Chart saved → {out_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 — Per foundation model, top-3 k values ranked by bacc
#
# x-axis  : foundation model
# bars    : top-3 k values (ranked by bacc for that model), one bar each
# y-axis  : bacc
# Bar interior shows k=N + abbreviated combo.
# ─────────────────────────────────────────────────────────────────────────────

def plot_by_model(
    k_data: dict,   # {k_val: DataFrame}
    metric: str = RANK_METRIC,
    top_n:  int = 3,
):
    valid_k = sorted([k for k, df in k_data.items() if df is not None])
    if not valid_k:
        print("[WARN] No valid k data to plot.")
        return

    palette  = ["#4C72B0", "#DD8452", "#55A868"]
    models   = MODEL_NAMES_CHART
    n_models = len(models)

    # For each model, collect (k_val, bacc, combo) sorted by bacc desc
    model_ranked: dict = {}
    for model_name in models:
        entries = []
        for k_val in valid_k:
            df = k_data[k_val]
            if df is None:
                continue
            sub = df[df["foundation_model"] == model_name]
            if sub.empty:
                continue
            val   = sub[metric].mean()
            combo = sub["combo"].iloc[0]
            entries.append({"k": k_val, metric: val, "combo": combo})
        entries.sort(key=lambda d: d[metric], reverse=True)
        model_ranked[model_name] = entries[:top_n]

    width   = 0.3
    offsets = np.linspace(-(top_n - 1) / 2, (top_n - 1) / 2, top_n) * width
    x       = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(max(10, n_models * 3), 7))

    for ci in range(top_n):
        color = palette[ci % len(palette)]
        for mi, model_name in enumerate(models):
            entries = model_ranked.get(model_name, [])
            if ci >= len(entries):
                continue
            entry = entries[ci]
            val   = entry[metric]
            k_val = entry["k"]
            combo = entry["combo"]
            xpos  = x[mi] + offsets[ci]

            ax.bar(xpos, val, width=width - 0.03,
                   color=color, edgecolor="white", linewidth=0.8,
                   label=f"Rank {ci + 1}" if mi == 0 else "")

            # Value above bar
            ax.text(xpos, val + 0.005, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold", color="#222222")

            # k label + abbreviated combo inside bar
            label_text = f"k={k_val}\n{abbrev_combo(combo)}"
            if val > 0.08:
                ax.text(xpos, val * 0.45, label_text,
                        ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold",
                        linespacing=1.3, clip_on=True,
                        multialignment="center")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_title(
        f"SurGen Top {top_n} k-values per Foundation Model (9 classes) | "
        f"Classifier: {CLASSIFIER.upper()} | Metric: {metric.upper()}",
        fontsize=13, fontweight="bold", pad=12
    )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    handles, lbls = ax.get_legend_handles_labels()
    seen = dict(zip(lbls, handles))
    ax.legend(seen.values(), seen.keys(), fontsize=10,
              loc="upper right", framealpha=0.9)

    fig.tight_layout()
    out_path = os.path.join(os.getcwd(), "surgen_by_model_topk.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[OK] Chart saved → {out_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — load all requested k values then produce both charts
# ─────────────────────────────────────────────────────────────────────────────

k_data = {}
for k in K_VALUES_SHOW:
    df = load_k(k)
    k_data[k] = df

    if df is not None:
        print(f"\nResults for k={k}:")
        print(
            df[["foundation_model", "combo", "k", RANK_METRIC]]
            .sort_values(RANK_METRIC, ascending=False)
            .to_string(index=False)
        )

# Chart 1: results across all k values, bars = foundation models
plot_across_k(k_data, RANK_METRIC)

# Chart 2: per foundation model, best k values as ranked bars
plot_by_model(k_data, RANK_METRIC, top_n=3)
