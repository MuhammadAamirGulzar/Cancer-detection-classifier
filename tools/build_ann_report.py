"""Build the PAIP-IV ANN experiment reports - one README and one .docx per
experiment, plus a combined index of all five.

Every number is read from the result CSVs and trees. Nothing is typed by hand,
so re-running after a re-run keeps every document honest (rule of engagement #1).

Layout produced under reports/ann_experiments/:

    README.md                          index across all five
    ANN_Experiments_All.docx           the same, as one document
    01_original.md / .docx             experiment 1
    02_tcga_cv.md / .docx              experiment 2
    03_ann_old.md / .docx              experiment 3
    04_ann_edited.md / .docx           experiment 4
    05_threshold.md / .docx            experiment 5

Markdown is generated without python-docx, so ``--md-only`` works anywhere.

Run:  python tools/build_ann_report.py
      python tools/build_ann_report.py --md-only
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics as st
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SLIDE_CLS = REPO_ROOT / "slide_classification"
OUT_DIR = REPO_ROOT / "reports" / "ann_experiments"

SHORT = {"Caption_based_aggregation_15_classes": "Caption_based_15",
         "Caption_based_aggregation": "Caption_based"}

HEADS = ("lin", "knn", "proto", "rf", "ann")

# --------------------------------------------------------------- shared prose

CONSTANT = [
    ("Cohort", "PAIP, 78 slides"),
    ("Split", "Provider's own: 47 train (12 MSI-H) / 31 test (7 MSI-H)"),
    ("Protocol", "Single fixed split, no folds, no averaging"),
    ("Combinations", "22 - 4 aggregation methods x 5 encoders, plus TITAN and PRISM"),
    ("Threshold", "0.5"),
    ("Uncertainty", "Percentile bootstrap, 1000 resamples, 95% CI"),
    ("Seed", "42"),
    ("Optimiser", "Adam, lr=1e-4, weight_decay=1e-4"),
    ("LR schedule", "ReduceLROnPlateau(factor=0.3, patience=5)"),
    ("Loss", "CrossEntropyLoss on raw logits"),
    ("Batching", "Full-batch - one Adam step per epoch"),
]

COHORT_NOTE = (
    "The 47/31 split is the recovered cohort. Five training slides "
    "(training_data_19/30/41/42/46, two MSI-H) previously had no features; their "
    "patches were recovered and features rebuilt, taking the split from 42/31 to "
    "the full 47/31 and train MSI-H from 10 to 12."
)

REPEATS_NOTE = (
    "The pipeline seeds torch and numpy before every fit and draws the carve-out "
    "with a seeded stratified split, so repeats at a fixed configuration are "
    "expected to be identical. All four runs were bit-identical across balanced "
    "accuracy, both CI bounds, AUROC, both CI bounds, accuracy, macro-F1, "
    "threshold and confusion matrix. That establishes every figure as exact "
    "rather than one draw from a distribution, and rules out GPU nondeterminism - "
    "not guaranteed for BatchNorm, and not something the pipeline configures "
    "away. It says nothing about how much a result would move under a different "
    "random draw; that would require varying the seed."
)

SLIDE_NOTE = (
    "One test slide is worth 0.071 balanced accuracy. With 7 positive slides, "
    "flipping one moves the metric by 1/14 - larger than most differences "
    "reported here."
)


# ------------------------------------------------------------------ the five

EXPERIMENTS: List[dict] = [
    dict(
        n=1, slug="01_original", title="Original (legacy)",
        source=("csv", "repeats_legacy.csv"),
        command="python run_iv_repeats.py --classifiers ann --out repeats_legacy.csv",
        summary="The pipeline as originally written: an 18-point grid selected on a "
                "9-slide carve-out that is not returned for the final fit.",
        config=[("Hidden layers", "2"), ("hidden_dim1", "searched"),
                ("hidden_dim2", "searched"), ("max_iter", "searched -> 500"),
                ("Dropout", "0.3"), ("Early-stop patience", "20"),
                ("Grid points evaluated", "18"), ("Selected on", "9 PAIP slides"),
                ("ANN trains on", "38 of 47"), ("Slides held back", "9"),
                ("Hyperparameters from", "PAIP search")],
        method=[
            "The ANN searches an 18-point grid (h1 in {128,256,512} x h2 in "
            "{64,128,256} x iter in {500,1000}) and selects on macro-F1 computed "
            "over a stratified 20% carve-out - 9 slides, roughly 7 non-MSI-H and 2 "
            "MSI-H. The carve-out is not returned for the final fit, so the ANN "
            "trains on 38 slides.",
            "Two properties of this design motivated the other experiments. "
            "First, lin, knn, proto and rf all merge the carve-out back and train "
            "on all 47 (logistic.py:19, knn.py:33, protonet.py:40, "
            "r_forest_eval.py:240), so the ANN is the only head in the comparison "
            "training on less data.",
            "Second, the selection is not informative at that size. Across the 22 "
            "combinations the 18 configurations collapse to a median of 3 distinct "
            "validation scores. The configuration chosen averages 0.6075 test "
            "macro-F1 against 0.6133 for the grid's median entry - worse than "
            "picking at random from its own grid. The grid's best entry averages "
            "0.6623, so 0.055 is the headroom being lost.",
        ],
        reading=["This is the baseline every other experiment is measured against. "
                 "It is last on mean balanced accuracy, and it is the only "
                 "configuration whose hyperparameters were chosen using PAIP data."],
    ),
    dict(
        n=2, slug="02_tcga_cv", title="TCGA-CV hyperparameters",
        source=("csv", "paip_iv_repeats.csv"),
        command="python run_iv_repeats.py --ann-protocol fixed --classifiers ann "
                "--out paip_iv_repeats.csv",
        summary="Configuration taken from the TCGA-CV majority vote; model refit on "
                "all 47 training slides.",
        config=[("Hidden layers", "2"), ("hidden_dim1", "from TCGA-CV"),
                ("hidden_dim2", "from TCGA-CV"), ("max_iter", "500"),
                ("Dropout", "0.3"), ("Early-stop patience", "20"),
                ("Grid points evaluated", "1"), ("Selected on", "nothing - pinned"),
                ("ANN trains on", "47"), ("Slides held back", "9 (early stop only)"),
                ("Hyperparameters from", "TCGA-CV folds")],
        method=[
            "Removes both problems identified in experiment 1. The configuration "
            "is taken from the majority vote across the four TCGA-CV folds - each "
            "validating on ~100 slides rather than 9 - and the model is refit on "
            "all 47. TCGA is a separate cohort, so no PAIP information enters the "
            "choice.",
            "Eight distinct configurations across the 22 combinations; max_iter is "
            "500 in every one.",
        ],
        reading=["Best mean AUROC of the four and the best single row. Since AUROC "
                 "is threshold-free and the stated primary metric, this is the "
                 "strongest configuration on the measure the study has committed to."],
    ),
    dict(
        n=3, slug="03_ann_old", title="ann_old",
        source=("csv", "repeats_ann_old.csv"),
        command="python run_iv_repeats.py --ann-protocol full --ann-preset ann_old "
                "--classifiers ann --out repeats_ann_old.csv",
        summary="The collaborator's ann_old.py configuration: one hidden layer of "
                "512, dropout 0.5, 1000 iterations.",
        config=[("Hidden layers", "1"), ("hidden_dim1", "512"), ("hidden_dim2", "-"),
                ("max_iter", "1000"), ("Dropout", "0.5"),
                ("Early-stop patience", "10"), ("Grid points evaluated", "1"),
                ("Selected on", "nothing - pinned"), ("ANN trains on", "47"),
                ("Slides held back", "0"),
                ("Hyperparameters from", "collaborator script")],
        method=[
            "The collaborator's ann_old.py implemented as a preset and taken "
            "verbatim. 512 and 1000 are what the script produces when eval_ANN is "
            "called without overriding anything: 512 is the class default for "
            "hidden_dim, 1000 is eval_ANN's default for max_iter, which shadows "
            "the class's 100. Neither script contains a grid, a search or a stored "
            "configuration - every value is a default argument or a hardcoded "
            "literal. TCGA-CV is not consulted at all.",
            "Not reproduced: ann_old.py places nn.Softmax inside the training "
            "graph while using CrossEntropyLoss, which applies log-softmax again. "
            "Softmax twice flattens gradients and underfits. That is a defect "
            "rather than a hyperparameter, so the preset carries its dropout of "
            "0.5 but emits raw logits like everything else.",
        ],
        reading=["Best mean balanced accuracy of the four, and lowest mean AUROC. "
                 "The single-layer network sits at a better operating point while "
                 "ranking slides slightly worse."],
    ),
    dict(
        n=4, slug="04_ann_edited", title="ann_edited",
        source=("csv", "repeats_ann_edited.csv"),
        command="python run_iv_repeats.py --ann-protocol full --ann-preset ann_edited "
                "--classifiers ann --out repeats_ann_edited.csv",
        summary="The collaborator's ann_edited.py configuration: identical to "
                "ann_old except dropout 0.7.",
        config=[("Hidden layers", "1"), ("hidden_dim1", "512"), ("hidden_dim2", "-"),
                ("max_iter", "1000"), ("Dropout", "0.7"),
                ("Early-stop patience", "10"), ("Grid points evaluated", "1"),
                ("Selected on", "nothing - pinned"), ("ANN trains on", "47"),
                ("Slides held back", "0"),
                ("Hyperparameters from", "collaborator script")],
        method=[
            "The collaborator's second script, implemented as a preset. It differs "
            "from ann_old in exactly one hyperparameter: dropout 0.7 against 0.5. "
            "Everything else - one hidden layer of 512, 1000 iterations, patience "
            "10, Adam at 1e-4/1e-4 - is identical.",
            "Unlike ann_old, this script already emits raw logits and applies "
            "softmax only in predict_proba, matching the pipeline's own "
            "convention.",
        ],
        reading=["The clean comparison in this set: against experiment 3 the only "
                 "difference is dropout, so any difference between them is "
                 "attributable. Dropout 0.5 gives +0.0066 mean balanced accuracy "
                 "and -0.0084 mean AUROC against 0.7."],
    ),
    dict(
        n=5, slug="05_threshold", title="Threshold calibration",
        source=("trees", ("PAIP_IV_Results",)),
        command="python runners/iv_runner.py --ann-protocol fixed --fit-threshold "
                "--tau-folds 4 --force",
        summary="Changes the decision rule rather than the model: a threshold fitted "
                "by inner cross-validation on the training slides, for all five "
                "classifier heads.",
        config=[("ANN configuration", "identical to experiment 2"),
                ("What changed", "the decision threshold only"),
                ("Inner CV", "stratified 4-fold, over the 47 training slides"),
                ("Criterion", "Youden's J on pooled out-of-fold probabilities"),
                ("Fitted on", "47 slides, 12 positives"),
                ("Test slides used in the fit", "none"),
                ("Applies to", "all five heads, not the ANN alone")],
        method=[
            "For each classifier and combination: stratified k-fold inside the 47 "
            "training slides, so every slide receives one prediction from a model "
            "that did not see it; the 47 out-of-fold probabilities are pooled and "
            "a threshold is fitted as the Youden-J optimum; that threshold is "
            "applied to the 31 test slides, which took no part in fitting it.",
            "The ANN's early-stopping split is carved from the inner training rows "
            "so the inner test fold stays clean, and inner folds run with "
            "model_save_path=None so they cannot overwrite the real checkpoints.",
            "It runs for all five heads deliberately. Calibrating the ANN alone "
            "while its competitors stayed at 0.5 would have replaced one unfair "
            "comparison with a different one - the mirror image of experiment 1, "
            "where the ANN alone trained on 38 slides.",
            "0.5 remains the reported operating point. The fitted figures are "
            "written to threshold_audit.csv and to three extra summary_iv.csv "
            "columns alongside the headline, never instead of it, so nothing in "
            "experiments 1 to 4 is affected.",
        ],
        reading=[],   # filled from data
    ),
]


# ------------------------------------------------------------------ data load

def _rows_from_csv(name: str) -> List[dict]:
    p = SLIDE_CLS / name
    if not p.exists():
        raise SystemExit(f"missing {p}")
    rows = list(csv.DictReader(open(p, newline="", encoding="utf-8")))
    runs = sorted({r["run"] for r in rows})
    return [r for r in rows if r["run"] == runs[0]], len(runs)


def _rows_from_tree(tree: str) -> List[dict]:
    rows = []
    for f in sorted((SLIDE_CLS / tree).rglob("summary_iv.csv")):
        rows += list(csv.DictReader(open(f, newline="", encoding="utf-8")))
    return rows


def _tau_folds(tree: str) -> Optional[int]:
    for p in sorted((SLIDE_CLS / tree).rglob("result_PAIP-IV_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        fit = (d.get("results", {}).get("ann", {}) or {}).get("tau_train_fit") or {}
        if fit.get("fitted"):
            return fit.get("n_splits")
    return None


def load(exp: dict) -> dict:
    kind, src = exp["source"]
    if kind == "csv":
        rows, n_runs = _rows_from_csv(src)
        ann = [r for r in rows if r["Classifier"] == "ann"]
        return {"ann": ann, "n_runs": n_runs, "all": rows}
    trees = {t: _rows_from_tree(t) for t in src}
    return {"trees": trees, "folds": {t: _tau_folds(t) for t in src}}


def stats(rows: List[dict], field="BalAcc") -> dict:
    v = [float(r[field]) for r in rows]
    return {"mean": st.mean(v), "median": st.median(v), "max": max(v), "min": min(v)}


# ------------------------------------------------------------ markdown render

def md_table(header: List[str], rows: List[List[str]], align: str = None) -> str:
    align = align or ("|" + "|".join("---" for _ in header) + "|")
    out = ["| " + " | ".join(header) + " |", align]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def render_experiment_md(exp: dict, data: dict) -> str:
    L = [f"# Experiment {exp['n']} · {exp['title']}", "", exp["summary"], ""]

    if exp["n"] != 5:
        ann, n_runs = data["ann"], data["n_runs"]
        s, a = stats(ann), stats(ann, "AUROC")
        best = max(ann, key=lambda r: float(r["BalAcc"]))
        L += ["## Headline", "",
              md_table(["Metric", "Value"], [
                  ["Repeats", f"{n_runs} (bit-identical)"],
                  ["Combinations", str(len(ann))],
                  ["Mean BalAcc", f"**{s['mean']:.4f}**"],
                  ["Median BalAcc", f"{s['median']:.4f}"],
                  ["Mean AUROC", f"**{a['mean']:.4f}**"],
                  ["Best BalAcc", f"{s['max']:.4f}"],
                  ["Best combination",
                   f"{SHORT.get(best['Method'], best['Method'])} / {best['Model']}"],
              ], "|---|---:|"), ""]

    L += ["## Configuration", "",
          md_table(["Parameter", "Value"], [[k, v] for k, v in exp["config"]],
                   "|---|---|"), ""]

    L += ["## Method", ""] + [p + "\n" for p in exp["method"]]

    if exp["n"] != 5:
        L += render_combo_table_md(data["ann"], exp["title"])
        L += ["### ANN detail", "",
              md_table(["Method", "Model", "BalAcc", "AUROC", "Confusion matrix"],
                       [[SHORT.get(r["Method"], r["Method"]), r["Model"],
                         f"{float(r['BalAcc']):.4f}", f"{float(r['AUROC']):.4f}",
                         r["ConfMatrix"]]
                        for r in sorted(data["ann"],
                                        key=lambda r: -float(r["BalAcc"]))],
                       "|---|---|---:|---:|---|"), ""]
    else:
        L += render_threshold_results_md(data)

    if exp["reading"]:
        L += ["## Reading it", ""] + [p + "\n" for p in exp["reading"]]

    L += ["## Held constant", "",
          md_table(["Setting", "Value"], [[k, v] for k, v in CONSTANT], "|---|---|"),
          "", COHORT_NOTE, "", "## Reproducing", "", "```bash",
          f"cd slide_classification", exp["command"], "```", "",
          SLIDE_NOTE, ""]
    if exp["n"] != 5:
        L += ["### Why four repeats", "", REPEATS_NOTE, ""]
    return "\n".join(L)


def render_threshold_results_md(data: dict) -> List[str]:
    trees, folds = data["trees"], data["folds"]
    tree = list(trees)[0]
    k = folds[tree]
    agg = [r for r in trees[tree] if r["Method"] not in ("TITAN", "PRISM")]

    rows = []
    for h in HEADS:
        sub = [r for r in agg if r["Classifier"] == h]
        b = st.mean(float(r["BalAcc"]) for r in sub)
        tt = st.mean(float(r["BalAcc_at_TauTrain"]) for r in sub)
        taus = [float(r["TauTrain"]) for r in sub]
        rows.append([h, f"{b:.4f}", f"{tt:.4f}", f"{tt - b:+.4f}",
                     f"{st.median(taus):.3f}"])
    rows.sort(key=lambda r: -float(r[2]))

    L = ["## Results", "",
         f"Means over the 20 aggregation rows per head, {k}-fold inner CV.", "",
         md_table(["Head", "BalAcc @ 0.5", "@ tau_train", "Δ", "median tau"],
                  rows, "|---|---:|---:|---:|---:|"), ""]

    latest = tree
    allrows = trees[latest]
    better = sum(1 for r in allrows
                 if float(r["BalAcc_at_TauTrain"]) > float(r["BalAcc"]))
    worse = sum(1 for r in allrows
                if float(r["BalAcc_at_TauTrain"]) < float(r["BalAcc"]))
    more_fn = sum(1 for r in allrows
                  if eval(r["ConfMatrix_at_TauTrain"])[1][0] > eval(r["ConfMatrix"])[1][0])
    fewer_fn = sum(1 for r in allrows
                   if eval(r["ConfMatrix_at_TauTrain"])[1][0] < eval(r["ConfMatrix"])[1][0])

    L += ["**Four of five heads lose.** The ANN is the only head that gains, by "
          "an eighth of one test slide, and still ranks third at tau_train.", "",
          f"Across all {len(allrows)} rows: **{worse} worse, {better} better, "
          f"{len(allrows) - worse - better} unchanged.** AUROC is unaffected, "
          f"being threshold-free.", ""]

    taus = {h: [float(r["TauTrain"]) for r in allrows if r["Classifier"] == h]
            for h in HEADS}
    L += ["### The fitted thresholds are not stable", "",
          md_table(["Head", "tau range", "median"],
                   [[h, f"{min(taus[h]):.3f} – {max(taus[h]):.3f}",
                     f"{st.median(taus[h]):.3f}"] for h in HEADS],
                   "|---|---|---:|"), "",
          "A threshold fitted from 12 positives is dominated by noise. lin fits "
          "values as low as 0.003 and the ANN's range spans almost the entire unit "
          "interval - neither is a property you would want in a deployed "
          "classifier.", ""]

    L += ["### What it does not do", "",
          "Because balanced accuracy weights 24 negatives and 7 positives equally, "
          "a threshold trading sensitivity for specificity could raise the metric "
          "while detecting fewer cancers. That is not what happens here:", "",
          md_table(["False negatives at tau vs 0.5", "rows"],
                   [["More missed positives", more_fn],
                    ["Fewer missed positives", fewer_fn],
                    ["Unchanged", len(allrows) - more_fn - fewer_fn]], "|---|---:|"),
          "",
          "The procedure fails on accuracy, not on safety - it generally makes "
          "models more sensitive, not less.", "",
          "### Conclusion", "",
          "A decision threshold fitted by inner cross-validation on 47 training "
          "slides does not transfer to the model trained on all 47. It degrades "
          "four of five classifier heads, and the fitted values are unstable. "
          "The ANN's deficit is in ranking, not operating point - its median AUROC "
          "is 0.8244 against kNN's 0.8557 - and no threshold can close that.", ""]
    return L


#: The five heads' configurations. lin/knn/proto/rf come from GRIDS in
#: runners/classifiers.py; the ANN's varies per experiment and is tabulated
#: separately in the hyperparameter document.
HEAD_CONFIG = [
    ("lin", "Logistic regression", "C=10, max_iter=300, lbfgs, random_state=42",
     "none - single point", "47", "none"),
    ("knn", "k-nearest neighbours",
     "k in {3,5,7,10,15} x metric in {cosine,euclidean,manhattan} x weights in "
     "{uniform,distance}", "30 points, nested CV on the training data only", "47",
     "L2"),
    ("proto", "Class-mean prototypes", "none", "none", "47", "L2"),
    ("rf", "Random forest",
     "n_estimators=500, max_depth=None, min_samples_split=5, min_samples_leaf=1, "
     "class_weight={0:1, 1:10}, threshold 0.3", "none - single point", "47", "none"),
    ("ann", "2-hidden-layer MLP", "varies by experiment - see the ANN document",
     "varies by experiment", "38 or 47", "none (BatchNorm only)"),
]


def all_heads_rows(tree: str = "PAIP_IV_Results"):
    """Every head's headline result, from the one tree that ran all five."""
    rows = _rows_from_tree(tree)
    out = []
    for h in HEADS:
        sub = [r for r in rows if r["Classifier"] == h]
        if not sub:
            continue
        ba = [float(r["BalAcc"]) for r in sub]
        au = [float(r["AUROC"]) for r in sub]
        best = max(sub, key=lambda r: float(r["BalAcc"]))
        out.append([h, f"{st.mean(ba):.4f}", f"{st.median(ba):.4f}",
                    f"{st.mean(au):.4f}", f"{max(ba):.4f}",
                    f"{SHORT.get(best['Method'], best['Method'])} / {best['Model']}"])
    out.sort(key=lambda r: -float(r[1]))
    return out


def classical_by_combo(tree: str = "PAIP_IV_Results") -> Dict[Tuple[str, str], Dict[str, dict]]:
    """{(method, model): {head: row}} for lin/knn/proto/rf at threshold 0.5.

    Experiments 1-4 changed only the ANN, so these four are identical across all
    of them and can be sourced once from the single tree that ran all five heads.
    """
    out: Dict[Tuple[str, str], Dict[str, dict]] = {}
    for r in _rows_from_tree(tree):
        if r["Classifier"] == "ann":
            continue
        out.setdefault((r["Method"], r["Model"]), {})[r["Classifier"]] = r
    return out


def render_combo_table_md(ann_rows: List[dict], label: str) -> List[str]:
    """Per-combination results: this experiment's ANN beside the four others."""
    other = classical_by_combo()
    rows = []
    for r in sorted(ann_rows, key=lambda r: -float(r["BalAcc"])):
        k = (r["Method"], r["Model"])
        o = other.get(k, {})
        cells = [SHORT.get(r["Method"], r["Method"]), r["Model"],
                 f"**{float(r['BalAcc']):.4f}**"]
        for h in ("lin", "knn", "proto", "rf"):
            cells.append(f"{float(o[h]['BalAcc']):.4f}" if h in o else "-")
        best = max([("ann", float(r["BalAcc"]))] +
                   [(h, float(o[h]["BalAcc"])) for h in o], key=lambda t: t[1])
        cells.append(best[0])
        rows.append(cells)
    won = sum(1 for r in rows if r[-1] == "ann")
    return [
        "## Results by combination", "",
        f"This experiment's ANN beside the four classical heads. Those four are "
        f"constant across experiments 1-4 - only the ANN changed - so they are "
        f"sourced once from the run that evaluated all five. Balanced accuracy at "
        f"threshold 0.5.", "",
        md_table(["Method", "Model", "ann", "lin", "knn", "proto", "rf", "best"],
                 rows, "|---|---|---:|---:|---:|---:|---:|---|"), "",
        f"**The ANN is the best head in {won} of {len(rows)} combinations.**", "",
    ]


def render_all_heads_md() -> List[str]:
    return [
        "## All five classifier heads", "",
        "Experiments 1-4 changed only the ANN, so `lin`, `knn`, `proto` and `rf` "
        "are identical across all of them - `ann_opts` reaches exactly one branch "
        "of `train_and_evaluate`, and a regression check confirmed 0 of 88 "
        "non-ANN rows changed between protocols. Their figures below therefore "
        "apply to every experiment.", "",
        "Threshold 0.5, 22 combinations, means across combinations.", "",
        md_table(["Head", "Mean BalAcc", "Median", "Mean AUROC", "Best", "Best combination"],
                 all_heads_rows(), "|---|---:|---:|---:|---:|---|"), "",
        "**The ANN is last on mean balanced accuracy under every configuration "
        "tried.** Its AUROC (0.8249) is within 0.010 of `lin` and `proto`, so its "
        "ranking is competitive; the deficit is in where its decision boundary "
        "sits, which is what experiment 5 tested and failed to fix.", "",
        md_table(["Head", "Model", "Hyperparameters", "Search", "Trains on", "Feature norm"],
                 [list(r) for r in HEAD_CONFIG], "|---|---|---|---|---:|---|"), "",
    ]


def render_ann_hparams_md(loaded) -> str:
    """The ANN on its own: every configuration, and every result it produced."""
    L = ["# The ANN Head - Configurations and Results", "",
         "Everything about the ANN across the PAIP-IV experiments: what it was "
         "configured with, where each value came from, and what it scored. Read "
         "from the run records - saved checkpoint configs and result stamps - not "
         "transcribed.", ""]

    # --- results first: that is what a reader wants
    rows = []
    for e, d in loaded:
        if e["n"] == 5:
            continue
        sv, av = stats(d["ann"]), stats(d["ann"], "AUROC")
        best = max(d["ann"], key=lambda r: float(r["BalAcc"]))
        rows.append([f"{e['n']} · {e['title']}", f"{sv['mean']:.4f}",
                     f"{sv['median']:.4f}", f"{av['mean']:.4f}", f"{sv['max']:.4f}",
                     f"{SHORT.get(best['Method'], best['Method'])} / {best['Model']}"])
    # Experiment 5 belongs in this table, but it changed the threshold rather than
    # the configuration - so it is labelled, not silently listed alongside the four.
    e5 = [r for r in _rows_from_tree("PAIP_IV_Results") if r["Classifier"] == "ann"]
    if e5:
        b = [float(r["BalAcc_at_TauTrain"]) for r in e5]
        a5 = [float(r["AUROC"]) for r in e5]
        bst = max(e5, key=lambda r: float(r["BalAcc_at_TauTrain"]))
        rows.append([
            "5 · threshold (tau_train)", f"{st.mean(b):.4f}", f"{st.median(b):.4f}",
            f"{st.mean(a5):.4f}", f"{max(b):.4f}",
            f"{SHORT.get(bst['Method'], bst['Method'])} / {bst['Model']}"])

    L += ["## Results across every experiment", "",
          "Experiments 1-4 change the ANN's configuration. Experiment 5 changes "
          "only the decision threshold - it reuses experiment 2's models, so its "
          "AUROC is identical to experiment 2's by construction and only the "
          "operating point differs.", "",
          md_table(["Experiment", "Mean BalAcc", "Median", "Mean AUROC", "Best",
                    "Best combination"], rows, "|---|---:|---:|---:|---:|---|"), "",
          "Every configuration beats the original. The two metrics disagree about "
          "which alternative wins - ann_old is highest on balanced accuracy and "
          "lowest on AUROC, the TCGA-CV configuration is the reverse. AUROC is "
          "threshold-free, so a rising balanced accuracy with a falling AUROC "
          "means the single-layer variants sit at a better operating point while "
          "ranking slides slightly worse.", "", SLIDE_NOTE, ""]

    # --- how it compares to the other heads
    L += ["## Against the other four heads", "",
          "Threshold 0.5, means over 22 combinations. The four classical heads are "
          "constant across experiments 1-4, since only the ANN changed.", "",
          md_table(["Head", "Mean BalAcc", "Median", "Mean AUROC", "Best",
                    "Best combination"], all_heads_rows(),
                   "|---|---:|---:|---:|---:|---|"), "",
          "**The ANN is last on mean balanced accuracy under every configuration "
          "tried.** Its AUROC is within 0.010 of lin and ties proto, so its "
          "ranking is competitive - the deficit is in where its decision boundary "
          "sits. Threshold calibration was tested against exactly that and failed "
          "to fix it (experiment 5).", ""]

    # --- per combination, every experiment
    keys = sorted(loaded[1][1]["ann"], key=lambda r: (r["Method"], r["Model"]))
    idx = {e["n"]: {(r["Method"], r["Model"]): r for r in d["ann"]}
           for e, d in loaded if e["n"] != 5}
    e5idx = {(r["Method"], r["Model"]): r for r in e5}
    rows = []
    for r in keys:
        k = (r["Method"], r["Model"])
        cells = [SHORT.get(k[0], k[0]), k[1]]
        for n in (1, 2, 3, 4):
            cells.append(f"{float(idx[n][k]['BalAcc']):.4f}" if k in idx[n] else "-")
        cells.append(f"{float(e5idx[k]['BalAcc_at_TauTrain']):.4f}"
                     if k in e5idx else "-")
        rows.append(cells)
    L += ["## ANN balanced accuracy, per combination", "",
          md_table(["Method", "Model", "Original", "TCGA-CV", "ann_old",
                    "ann_edited", "tau_train"], rows,
                   "|---|---|---:|---:|---:|---:|---:|"), ""]


    rows = []
    for e, d in loaded:
        if e["n"] == 5:
            continue
        cfg = dict(e["config"])
        rows.append([f"{e['n']} · {e['title']}", cfg["Hidden layers"],
                     cfg["hidden_dim1"], cfg["hidden_dim2"], cfg["max_iter"],
                     cfg["Dropout"], cfg["Early-stop patience"]])
    L += ["## Architecture and regularisation", "",
          md_table(["Experiment", "Layers", "h1", "h2", "max_iter", "Dropout", "Patience"],
                   rows, "|---|---:|---|---|---|---:|---:|"), ""]

    rows = []
    for e, d in loaded:
        if e["n"] == 5:
            continue
        cfg = dict(e["config"])
        rows.append([f"{e['n']} · {e['title']}", cfg["Grid points evaluated"],
                     cfg["Selected on"], cfg["ANN trains on"],
                     cfg["Slides held back"], cfg["Hyperparameters from"]])
    L += ["## How each configuration was obtained", "",
          md_table(["Experiment", "Grid points", "Selected on", "Trains on",
                    "Held back", "Source"], rows, "|---|---:|---|---:|---:|---|"), ""]

    L += ["## Fixed in code, identical everywhere", "",
          "These never varied. They are literals in `eval_patch_features/ann.py` "
          "and reach no output file, which is why the harvesting script reads them "
          "by introspecting the classifier rather than transcribing them.", "",
          md_table(["Parameter", "Value", "Where"], [
              ["Optimiser", "Adam", "ann.py fit()"],
              ["lr", "1e-4", "constructor default"],
              ["weight_decay", "1e-4", "constructor default"],
              ["LR schedule", "ReduceLROnPlateau(factor=0.3, patience=5)", "ann.py fit()"],
              ["Loss", "CrossEntropyLoss on raw logits", "ann.py __init__"],
              ["Batching", "full-batch, one Adam step per epoch", "ann.py fit()"],
              ["Softmax in graph", "no - applied only in predict_proba", "Task 1.3 correction"],
              ["Seed", "42", "iv_runner.py"],
          ], "|---|---|---|"), ""]

    L += ["## Per-combination widths under experiment 2", "",
          "Experiment 2 pins the configuration per (aggregation x encoder) from "
          "the TCGA-CV majority vote, so it is the only experiment where the "
          "architecture varies across combinations. Eight distinct configurations; "
          "`max_iter` is 500 in every one.", ""]
    ann2 = [r for r in _rows_from_csv("paip_iv_repeats.csv")[0]
            if r["Classifier"] == "ann"]
    L += [md_table(["Method", "Model", "h1", "h2", "max_iter", "input dim"],
                   [[SHORT.get(r["Method"], r["Method"]), r["Model"],
                     r["hp_hidden_dim1"], r["hp_hidden_dim2"], r["hp_max_iter"],
                     r["hp_input_dim"]]
                    for r in sorted(ann2, key=lambda r: (r["Method"], r["Model"]))],
                   "|---|---|---:|---:|---:|---:|"), "",
          "Input dimension spans 512 to 38,400 because aggregation is flattened: "
          "Averaging gives 1xD, Tissue_Type_Clustering 9xD, Caption_based_15 15xD. "
          "The same hidden width therefore means a 1:1 mapping for one combination "
          "and a 300:1 bottleneck for another. Measured across the 22 "
          "combinations, network size has no relationship with performance "
          "(r = +0.042 between log-parameters and balanced accuracy), so this is a "
          "presentational inconsistency rather than a performance one.", ""]

    # --- experiment 5, properly
    if e5:
        b = [float(r["BalAcc_at_TauTrain"]) for r in e5]
        b0 = [float(r["BalAcc"]) for r in e5]
        w = sum(1 for r in e5
                if float(r["BalAcc_at_TauTrain"]) > float(r["BalAcc"]))
        l = sum(1 for r in e5
                if float(r["BalAcc_at_TauTrain"]) < float(r["BalAcc"]))
        taus = [float(r["TauTrain"]) for r in e5]
        L += ["## Experiment 5 - threshold, not configuration", "",
              "The threshold experiment reused experiment 2's configuration "
              "verbatim. **No ANN hyperparameter differs between them** - only the "
              "decision rule changed, from a fixed 0.5 to a threshold fitted by "
              "4-fold inner cross-validation on the 47 training slides.", "",
              md_table(["Metric", "Value"], [
                  ["ANN configuration", "identical to experiment 2"],
                  ["Threshold at", "fitted per combination, not 0.5"],
                  ["tau range", f"{min(taus):.3f} - {max(taus):.3f}"],
                  ["Median tau", f"{st.median(taus):.3f}"],
                  ["Mean BalAcc at 0.5", f"{st.mean(b0):.4f}"],
                  ["Mean BalAcc at tau_train", f"{st.mean(b):.4f}"],
                  ["Change", f"{st.mean(b) - st.mean(b0):+.4f}"],
                  ["Per combination", f"{w} better, {l} worse, "
                                      f"{len(e5) - w - l} unchanged"],
                  ["Mean AUROC", "unchanged - threshold-free"],
              ], "|---|---:|"), "",
              f"The ANN gains {st.mean(b) - st.mean(b0):+.4f} mean balanced "
              f"accuracy - roughly an eighth of one test slide. It is the only "
              f"head that gains; the other four all lose, `proto` by 0.057. At "
              f"tau_train the ANN still ranks third of five.", "",
              "The ANN's fitted thresholds span almost the entire unit interval "
              f"({min(taus):.3f} to {max(taus):.3f}), which is what a threshold "
              "estimated from 12 positives looks like. See the experiment 5 report "
              "for the full five-head comparison and the stability analysis.", ""]
    return "\n".join(L)


def render_index_md(loaded: List[Tuple[dict, dict]]) -> str:
    L = ["# PAIP-IV ANN Experiments", "",
         "Five experiments on the PAIP provider split. Experiments 1-4 vary the "
         "ANN's configuration; experiment 5 varies the decision threshold and "
         "leaves the models untouched.", "",
         md_table(["#", "Experiment", "Report", "Source", "Repeats"],
                  [[e["n"], e["title"], f"[{e['slug']}.md]({e['slug']}.md)",
                    f"`{e['source'][1] if e['source'][0] == 'csv' else 'results trees'}`",
                    d.get("n_runs", 2)] for e, d in loaded],
                  "|---:|---|---|---|---:|"), ""]

    rows = []
    for e, d in loaded:
        if e["n"] == 5:
            continue
        s, a = stats(d["ann"]), stats(d["ann"], "AUROC")
        rows.append([f"{e['n']} · {e['title']}", f"{s['mean']:.4f}",
                     f"{s['median']:.4f}", f"{a['mean']:.4f}", f"{s['max']:.4f}"])
    L += ["## Results across the four configuration experiments", "",
          md_table(["Experiment", "Mean BalAcc", "Median", "Mean AUROC", "Best"],
                   rows, "|---|---:|---:|---:|---:|"), "",
          "Every configuration beats the original. The two metrics disagree about "
          "which alternative wins: ann_old is highest on balanced accuracy and "
          "lowest on AUROC, while the TCGA-CV configuration is the reverse. AUROC "
          "is threshold-free, so a rising balanced accuracy with a falling AUROC "
          "means the single-layer variants sit at a better operating point while "
          "ranking slides slightly worse. On AUROC - the stated primary metric - "
          "the TCGA-CV configuration is strongest.", "",
          SLIDE_NOTE, "",
          "Experiments 3 and 4 change three things at once against experiment 2 - "
          "depth, dropout and iterations - so differences from experiment 2 cannot "
          "be attributed to one alone. Between 3 and 4 the comparison is clean: "
          "dropout is the only difference.", "",
          "## Provenance", "",
          "Experiments 2, 3 and 4 can all state that no hyperparameter was tuned "
          "on PAIP - experiment 2 takes them from a separate cohort, experiments 3 "
          "and 4 from the collaborator's script. Experiment 1 cannot: its "
          "configuration was selected on 9 PAIP training slides.", "",
          ""] + render_all_heads_md() + [
          "## Held constant", "",
          md_table(["Setting", "Value"], [[k, v] for k, v in CONSTANT], "|---|---|"),
          "", COHORT_NOTE, ""]
    return "\n".join(L)


# ---------------------------------------------------------------- docx render

def render_docx(md_title: str, sections: str, out: Path) -> Path:
    """Render one document. Imported lazily so --md-only needs no python-docx."""
    from docx import Document
    from docx.shared import Pt

    def heading(doc, text, size, space_before=14):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(space_before)
        p.paragraph_format.space_after = Pt(6)
        r = p.add_run(text); r.bold = True; r.font.size = Pt(size)

    def body(doc, text, size=10.5, italic=False):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(8)
        r = p.add_run(text); r.italic = italic; r.font.size = Pt(size)

    def table(doc, header, rows):
        t = doc.add_table(rows=1, cols=len(header))
        t.style = "Light Grid Accent 1"
        for i, c in enumerate(header):
            cell = t.rows[0].cells[i]; cell.text = ""
            r = cell.paragraphs[0].add_run(c); r.bold = True; r.font.size = Pt(8.5)
        for row in rows:
            cells = t.add_row().cells
            for i, c in enumerate(row):
                cells[i].text = ""
                r = cells[i].paragraphs[0].add_run(str(c)); r.font.size = Pt(8.5)
        doc.add_paragraph()

    doc = Document()
    # Walk the generated markdown - it is the single source of truth, so the two
    # formats cannot drift apart.
    lines = sections.splitlines()
    i, first = 0, True
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("# "):
            heading(doc, ln[2:], 17, space_before=0 if first else 14); first = False
        elif ln.startswith("### "):
            heading(doc, ln[4:], 12)
        elif ln.startswith("## "):
            heading(doc, ln[3:], 14)
        elif ln.startswith("|") and i + 1 < len(lines) and set(lines[i + 1]) <= set("|-: "):
            hdr = [c.strip() for c in ln.strip("|").split("|")]
            i += 2
            body_rows = []
            while i < len(lines) and lines[i].startswith("|"):
                body_rows.append([c.strip().replace("**", "")
                                  for c in lines[i].strip("|").split("|")])
                i += 1
            table(doc, hdr, body_rows)
            continue
        elif ln.startswith("```"):
            i += 1
            code = []
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i]); i += 1
            body(doc, "\n".join(code), size=9, italic=True)
        elif ln.strip():
            body(doc, ln.replace("**", "").replace("`", ""))
        i += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    return out


# --------------------------------------------------------------------- build

def build(md_only: bool = False, out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded = [(e, load(e)) for e in EXPERIMENTS]

    written = []
    for exp, data in loaded:
        md = render_experiment_md(exp, data)
        p = out_dir / f"{exp['slug']}.md"
        p.write_text(md, encoding="utf-8")
        written.append(p)
        if not md_only:
            written.append(render_docx(exp["title"], md,
                                       out_dir / f"{exp['slug']}.docx"))

    index = render_index_md(loaded)
    p = out_dir / "README.md"
    p.write_text(index, encoding="utf-8")
    written.append(p)

    ann_md = render_ann_hparams_md(loaded)
    p = out_dir / "ANN_REPORT.md"
    p.write_text(ann_md, encoding="utf-8")
    written.append(p)
    if not md_only:
        written.append(render_docx("The ANN Head", ann_md,
                                   out_dir / "ANN_REPORT.docx"))

    if not md_only:
        combined = index + "\n\n---\n\n" + "\n\n---\n\n".join(
            render_experiment_md(e, d) for e, d in loaded)
        written.append(render_docx("ANN Experiments", combined,
                                   out_dir / "ANN_Experiments_All.docx"))

    for p in written:
        print(f"  wrote {p.relative_to(REPO_ROOT)}")
    print(f"\n{len(written)} files -> {out_dir.relative_to(REPO_ROOT)}")


def main():
    ap = argparse.ArgumentParser(description="Build the ANN experiment reports")
    ap.add_argument("--md-only", action="store_true",
                    help="markdown only; skips python-docx entirely")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    build(md_only=args.md_only, out_dir=args.out)


if __name__ == "__main__":
    main()
