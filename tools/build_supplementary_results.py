"""Generate the rewritten Results sections for the supplementary report.

The existing ``Supplementary_Report_DRAFT_v2.docx`` is 23 MB with 45 embedded
figures. This script does **not** edit it in place - it emits a companion
document containing the replacement sections, so the original is never at risk
and the owner controls the splice.

What needs replacing, and why
-----------------------------
The draft's results headings use a vocabulary that does not survive review:

    4.1 TCGA (IV) Validation Results     -> is 4-fold CROSS-VALIDATION, not IV
    4.2 PAIP (IV) Validation Results     -> is PAIP-CV, an experiment now RETIRED
    4.3 TCGA -> PAIP (EV) Validation      -> correct, but single-variant and built
                                            on the wrong ANN checkpoint
    4.4 SurGen (IV) Validation Results   -> is cross-validation, and leak-inflated

Three of four say "IV" for cross-validation experiments, and one documents a
withdrawn experiment. The canonical taxonomy (work order section 1) replaces
them with TCGA-CV / PAIP-IV / PAIP-EV / SurGen-CV / SurGen-EV.

Every table here is generated from the result files, so re-running after a
re-run keeps the document consistent with the numbers on disk.

Run:  python tools/build_supplementary_results.py
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

REPO_ROOT = Path(__file__).resolve().parents[1]
SLIDE_CLS = REPO_ROOT / "slide_classification"
RADAR_ROOT = REPO_ROOT / "Analysis_and_Visualization" / "Radar_Plots"
sys.path.insert(0, str(REPO_ROOT / "tools"))

from build_report import (discover, _rows_from, SLIDE_LEVEL_ENCODERS,  # noqa: E402
                          sort_summary, fill_threshold_scheme)
from build_ev_experiment_report import (load_results, promoted_state,  # noqa: E402
                                        config_rows as ev_config_rows, CLF_NAME)

OUT_DOCX = REPO_ROOT / "Supplementary_Results_UPDATED.docx"

CONFIG_HEADER = ["Classifier", "Model configuration", "Trained on",
                 "Threshold / decision rule", "Selection basis"]

SECTIONS = [
    ("4.1", "TCGA-CV — Internal Cross-Validation on TCGA", "TCGA-CV"),
    ("4.2", "PAIP-IV — Internal Validation on the PAIP Provider Split", "PAIP-IV"),
    ("4.3", "PAIP-EV — External Validation, TCGA to PAIP", "PAIP-EV"),
    ("4.4", "SurGen-CV — Case-Grouped Cross-Validation on SurGen", "SurGen-CV"),
    ("4.5", "SurGen-EV — External Validation, TCGA to SurGen", "SurGen-EV"),
]

BLURB = {
    "TCGA-CV": (
        "Four-fold cross-validation on the TCGA colorectal cohort. Each of the 413 "
        "slides that has both a label and extracted features is tested exactly once. "
        "The rotation is test = fold i, validation = fold i+1, train = the remaining "
        "two folds, so each model trains on roughly 50% of the cohort rather than the "
        "75% a reader may assume from \"4-fold\". Reported at the default decision "
        "threshold of 0.5, since train and test distributions are matched.\n\n"
        "This section was previously headed \"TCGA (IV) Validation Results\". The "
        "experiment is cross-validation, not internal validation against a "
        "provider-defined split, and has been renamed accordingly."),
    "PAIP-IV": (
        "The PAIP provider's own train/test split, evaluated as a single fixed split "
        "with no averaging. All 47 declared training slides have extracted features "
        "(12 MSI-H); all 31 test slides have features (7 MSI-H). A stratified 20% of "
        "the 47 training slides is held out for ANN early stopping; the 31 test "
        "slides take no part in any tuning decision. Reported at threshold 0.5.\n\n"
        "Five training slides (training_data_19/30/41/42/46, two of them MSI-H) "
        "previously had no features and the split was reported as 42/31. Their patches "
        "were recovered, features and slide aggregations rebuilt, and the split is now "
        "the full 47/31 the provider defines.\n\n"
        "The ANN head searches its 18-point grid on a 9-slide stratified carve-out of "
        "the 47 training slides (the `legacy` protocol) and is fitted on the "
        "remaining 38; the linear, kNN, prototype and random-forest heads merge that "
        "carve-out back and train on all 47. A TCGA-CV-derived, no-search alternative "
        "(the `fixed` protocol, training the ANN on all 47) was evaluated and is not "
        "the published configuration - see §4.2.1 for the current, exact per-classifier "
        "training split.\n\n"
        "Because the test set is only 31 slides with 7 positives, every metric is "
        "accompanied by a percentile bootstrap 95% confidence interval (1000 resamples "
        "with replacement). The intervals are wide and should be quoted alongside the "
        "point estimates.\n\n"
        "This section replaces the previous \"PAIP (IV) Validation Results\", which "
        "reported a four-fold cross-validation on PAIP. That experiment has been "
        "retired - see the note at the end of this section."),
    "PAIP-EV": (
        "External validation: models trained entirely on TCGA, applied to all 78 PAIP "
        "slides that have features. No PAIP data enters training, and no threshold is "
        "tuned on PAIP - doing so would silently convert external validation into "
        "internal validation.\n\n"
        "Three training variants are reported side by side. TCGA-FULL is the primary "
        "artifact, a single model trained on all 413 TCGA slides. The 4-fold ensemble "
        "averages the four cross-validation fold-models' predicted probabilities into "
        "one prediction per slide before thresholding. The 4-fold average is the legacy "
        "convention, applying each fold-model separately and averaging the four metric "
        "sets; it is retained for continuity with previously circulated numbers.\n\n"
        "TWO THRESHOLD SCHEMES ARE IN USE IN THIS SECTION, and the tables below mix "
        "them, so every row states which one produced it.\n\n"
        "kNN and Random Forest are reported under a corrected scheme. kNN is scored at "
        "k = 35 neighbours with uniform weights and thresholded at a cut-point refitted "
        "by Youden's J on TCGA out-of-fold probabilities AT THAT k; because k changes "
        "the scores themselves, kNN is the only classifier whose AUROC changes. Random "
        "Forest keeps its scores and takes a rate-matched (quantile) threshold: the "
        "fraction of TCGA out-of-fold slides that the fixed TCGA threshold calls "
        "positive is measured, and PAIP is cut at its own corresponding quantile - so "
        "Random Forest's AUROC is unchanged and only its operating point moves.\n\n"
        "Logistic Regression, the ANN and ProtoNet are unchanged. They keep the fixed "
        "TCGA threshold, fitted by Youden's J on pooled TCGA out-of-fold predictions and "
        "applied to PAIP as-is, and their figures are identical to those previously "
        "circulated.\n\n"
        "Both schemes are derived from TCGA alone: k, the refitted cut-point and the "
        "matched rate all come from TCGA out-of-fold predictions, and the quantile rule "
        "reads PAIP SCORES but never PAIP LABELS, which enter only the reported metrics. "
        "External validation is therefore not compromised. The motivation is that the "
        "fixed TCGA threshold left several kNN and Random Forest configurations at a "
        "dead operating point - predicting no positives at all, which yields a balanced "
        "accuracy of 0.50 while still looking like a working threshold. Under the "
        "corrected scheme no configuration is dead or saturated.\n\n"
        "SurGen-EV is unaffected and remains entirely on the fixed TCGA threshold, so "
        "kNN and Random Forest are NOT directly comparable between the two external "
        "cohorts."),
    "SurGen-CV": (
        "Four-fold cross-validation within the SurGen cohort, with folds constructed at "
        "CASE level. Seventy of SurGen's 554 labelled cases contribute two slides each; "
        "assigning slides to folds independently placed roughly two thirds of those "
        "cases on both sides of the train/test boundary. Two sections from one tumour "
        "are far more alike than two different tumours, so this inflated every "
        "previously reported SurGen figure.\n\n"
        "All results in this section use case-grouped folds and are therefore lower "
        "than the corresponding numbers in earlier drafts. The reduction is the "
        "correction taking effect, not a regression.\n\n"
        "This section was previously headed \"SurGen (IV) Validation Results\"; the "
        "experiment is cross-validation and has been renamed."),
    "SurGen-EV": (
        "External validation: TCGA-trained models applied to the SurGen cohort, using "
        "the same three variants and the same fixed TCGA threshold as PAIP-EV.\n\n"
        "Coverage is partial by design. SurGen aggregated features are available on "
        "this workstation only for Conch1_5 and Virchow2; the remaining foundation "
        "models were processed on the group's Linux server. The identical code run "
        "there completes the matrix, and the two result trees merge without conflict."),
}

CAVEATS = [
    ("Cohort sizes differ from the raw label files",
     "A slide must have both a label and extracted features to be modelled. TCGA: 416 "
     "labelled, 413 modelled (TCGA-AD-6895_MSIH, TCGA-AD-6899_nonMSIH and "
     "TCGA-CM-6680_nonMSIH have no features). PAIP: 78 labelled, 73 modelled. SurGen: "
     "624 usable labels, 622 modelled (SR386_40X_HE_T086_01 and SR386_40X_HE_T339_01 "
     "have no features). Earlier drafts quoted the label-file counts."),
    ("Training uses 50% of each cohort in cross-validation",
     "The fold rotation reserves one fold for test and one for validation, leaving two "
     "for training. This is deliberate, but a reader comparing against five-fold work "
     "will otherwise assume more training data than was used."),
    ("\"Validation fold\" means two different things",
     "Logistic regression, KNN, ProtoNet and Random Forest merge the validation fold "
     "back into training before fitting the final model. Only the ANN uses it as a true "
     "early-stopping set."),
    ("ANN results in earlier drafts are affected by three separate defects",
     "Hyperparameters were selected on the test fold rather than the validation fold; "
     "the checkpoint written to disk was the last grid entry rather than the selected "
     "one, so external validation ran a different network from the one whose "
     "cross-validation metrics were reported beside it; and a softmax was applied twice "
     "because the network's final Softmax layer was fed to a cross-entropy loss that "
     "applies log-softmax internally. All three are corrected here."),
    ("Empty tissue rows are indistinguishable from zero-valued ones",
     "In tissue-type clustering an absent tissue class produces an all-zero row. Across "
     "the TCGA slides, lymphocyte is absent in 109 slides, adipose in 69, smooth muscle "
     "in 51 and normal mucosa in 51. The classifier cannot distinguish \"absent\" from "
     "\"present but zero-valued\"."),
]


# --------------------------------------------------------------------------


def load() -> pd.DataFrame:
    rows: List[dict] = []
    for rec in discover(SLIDE_CLS):
        rows.extend(_rows_from(rec))
    if not rows:
        raise SystemExit("no result files found - run the experiments first")
    return fill_threshold_scheme(pd.DataFrame(rows))


def load_comparison() -> Optional[pd.DataFrame]:
    p = SLIDE_CLS / "comparison_old_vs_new.xlsx"
    if not p.exists():
        return None
    try:
        return pd.read_excel(p, sheet_name="Summary")
    except Exception:
        return None


def add_heading(doc, text, size, bold=True, space_before=14):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    r.bold = bold
    r.font.size = Pt(size)
    return p


def add_body(doc, text, italic=False, size=10.5):
    for chunk in text.split("\n\n"):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(8)
        r = p.add_run(chunk.strip())
        r.italic = italic
        r.font.size = Pt(size)
    return doc


def add_table(doc, df: pd.DataFrame, cols: List[str], max_rows: int = None,
              float_fmt: str = "{:.3f}"):
    cols = [c for c in cols if c in df.columns]
    if df.empty or not cols:
        add_body(doc, "(no rows)", italic=True)
        return
    d = df[cols].head(max_rows) if max_rows else df[cols]
    t = doc.add_table(rows=1, cols=len(cols))
    t.style = "Light Grid Accent 1"
    for i, c in enumerate(cols):
        cell = t.rows[0].cells[i]
        cell.text = ""
        r = cell.paragraphs[0].add_run(str(c))
        r.bold = True
        r.font.size = Pt(8.5)
    for _, row in d.iterrows():
        cells = t.add_row().cells
        for i, c in enumerate(cols):
            v = row[c]
            if isinstance(v, float):
                txt = "—" if pd.isna(v) else float_fmt.format(v)
            else:
                txt = "—" if pd.isna(v) else str(v)
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(txt)
            r.font.size = Pt(8.5)
    doc.add_paragraph()


def _cv_reference_record(experiment: str) -> Optional[dict]:
    """One representative result_<exp>_*_default.json - any combo will do; the
    protocol, grid size and train/val split are the same for all of them."""
    tree = "TCGA_Results" if experiment == "TCGA-CV" else "SurGen_Results"
    files = sorted(glob.glob(str(SLIDE_CLS / tree / "**" / f"result_{experiment}_*_default.json"),
                             recursive=True))
    return json.loads(Path(files[0]).read_text(encoding="utf-8")) if files else None


def cv_config_rows(experiment: str) -> List[List[str]]:
    """TCGA-CV / SurGen-CV: identical classifier treatment, read from one fold
    record so the grid size and split counts stay correct if GRIDS ever changes,
    rather than being retyped by hand."""
    rec = _cv_reference_record(experiment)
    if rec is None:
        return []
    ann_fold = next((f for f in rec["folds"] if f["classifier"] == "ann"), None)
    lin_fold = next((f for f in rec["folds"] if f["classifier"] == "lin"), None)
    n_grid = len(ann_fold["selection"]["grid_trace"]) if ann_fold else 18
    n_tr = lin_fold["n_train"] if lin_fold else "?"
    n_val = lin_fold["n_val"] if lin_fold else "?"
    n_merged = (n_tr + n_val) if isinstance(n_tr, int) else "?"
    return [
        ["Logistic Regression", "C=10, max_iter=300",
         f"train+val merged ({n_merged} rows, ~75% of the cohort)", "0.5", "none - single-point grid"],
        ["Neural Network", f"{n_grid}-point grid "
         f"(hidden_dim1∈{{128,256,512}} × hidden_dim2∈{{64,128,256}} × "
         f"max_iter∈{{500,1000}})",
         f"train only ({n_tr} rows, ~50%); val ({n_val} rows) used for "
         f"selection and early stopping, never merged in",
         "0.5", "selected on validation macro-F1 (this fold's own val split)"],
        ["k-Nearest Neighbours", "internal GridSearchCV, 30 points "
         "(k∈{3,5,7,10,15} × metric∈{cosine,euclidean,manhattan} × "
         "weights∈{uniform,distance}), scored on balanced accuracy",
         f"train+val merged ({n_merged} rows)", "0.5",
         "GridSearchCV on the merged training data, not this fold's test set"],
        ["ProtoNet", "no hyperparameters - class means of L2-normalised features",
         f"train+val merged ({n_merged} rows)", "0.5", "none - no hyperparameters"],
        ["Random Forest", "n_estimators=500, max_depth=None, "
         "min_samples_split=5, min_samples_leaf=1, class_weight={0:1, 1:10}",
         f"train+val merged ({n_merged} rows)", "0.5", "none - single-point grid"],
    ]


def paip_iv_config_rows() -> List[List[str]]:
    """PAIP-IV: read straight from the published stamp, not retyped by hand -
    the ANN protocol here has already changed once (fixed -> legacy) without
    every piece of prose describing it being updated at the same time."""
    files = sorted(glob.glob(str(SLIDE_CLS / "PAIP_IV_Results" / "**" /
                                 "result_PAIP-IV_*_default.json"), recursive=True))
    if not files:
        return []
    st = json.loads(Path(files[0]).read_text(encoding="utf-8"))["stamp"]
    protocol = st.get("ann_protocol", "legacy")
    held = st.get("ann_held_back_for_tuning", 9)
    trained_on = st.get("ann_trained_on", "38 of 47 training slides")
    hp_source = st.get("ann_hparam_source", "searched on the PAIP early-stop carve-out")
    fixed = ("all 47 training slides", "0.5", "none - single-point grid")
    return [
        ["Logistic Regression", "C=10, max_iter=300", *fixed],
        ["Neural Network", f"18-point grid (protocol: {protocol})",
         trained_on, "0.5", hp_source],
        ["k-Nearest Neighbours", "internal GridSearchCV, 30 points, scored on "
         "balanced accuracy", *fixed],
        ["ProtoNet", "no hyperparameters - class means of L2-normalised features", *fixed],
        ["Random Forest", "n_estimators=500, max_depth=None, "
         "min_samples_split=5, min_samples_leaf=1, class_weight={0:1, 1:10}", *fixed],
    ]


def ev_config_rows_for(experiment: str) -> List[List[str]]:
    """PAIP-EV / SurGen-EV: reuse the exact data-driven function that builds
    the standalone EV reports, so this table cannot silently disagree with
    those - both read the same promoted/frozen state from the same files."""
    payloads = load_results(experiment)
    return ev_config_rows(payloads) if payloads else []


def _combos_for(experiment: str) -> List[tuple]:
    """Every valid (method, model) pair, aggregation combinations AND the two
    slide-level baselines (TITAN/Conch1_5, PRISM/PRISM) - both are still real
    trained combinations with their own ANN grid search and kNN GridSearchCV,
    just with the encoder itself standing in for a foundation-model choice."""
    from config import paths as P
    return [(m, mo) for m in P.AGGREGATION_METHODS for mo in P.CANONICAL_MODELS
            if P.is_combination_valid(m, mo)]


def _ann_config(models_dir: Path, fold_tag: str) -> Optional[dict]:
    p = models_dir / f"{fold_tag}_ann_config.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _knn_hparams(pkl_path: Path):
    """(n_neighbors, metric, weights) read straight off the fitted sklearn
    estimator. This is the ONLY place these live - eval_knn's GridSearchCV
    winner is never written into any result JSON, so there is no faster path
    that does not risk quietly reporting the nominal single-point grid
    (n_neighbors=3) instead of what was actually selected per combination."""
    if not pkl_path.exists():
        return None
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import joblib
            m = joblib.load(pkl_path)
        return (getattr(m, "n_neighbors", None), getattr(m, "metric", None),
                getattr(m, "weights", None))
    except Exception:
        return None


def hparam_rows(experiment: str) -> List[List[str]]:
    """Per-combination ANN and kNN hyperparameters, actually read off disk -
    the saved *_ann_config.json (no pickle needed) and the fitted kNN
    estimator's own attributes (only place n_neighbors/metric/weights live).

    CV experiments (TCGA-CV, SurGen-CV) select separately per fold, so one
    row per (combination, fold). PAIP-IV is a single split - one "fold0" row.
    PAIP-EV and SurGen-EV both apply the SAME TCGA-FULL artifact (seed 42),
    so this reads TCGA_FULL_Models rather than either EV tree.
    """
    from config import paths as P
    from runners.full_trainer import artifact_dir

    rows = []
    if experiment in ("TCGA-CV", "SurGen-CV"):
        fold_tags = ["fold0", "fold1", "fold2", "fold3"]
        for method, model in _combos_for(experiment):
            models_dir = P.results_root(experiment, method, model, "MSIH") / "models"
            for i, tag in enumerate(fold_tags, start=1):
                ann = _ann_config(models_dir, tag)
                knn = _knn_hparams(models_dir / f"{tag}_knn_model.pkl")
                if ann is None and knn is None:
                    continue
                rows.append([method, model, f"Fold {i}",
                            str(ann["hidden_dim1"]) if ann else "—",
                            str(ann["hidden_dim2"]) if ann else "—",
                            str(ann["max_iter"]) if ann else "—",
                            str(knn[0]) if knn else "—",
                            knn[1] if knn else "—", knn[2] if knn else "—"])
    elif experiment == "PAIP-IV":
        for method, model in _combos_for(experiment):
            models_dir = P.results_root("PAIP-IV", method, model, "MSIH") / "models"
            ann = _ann_config(models_dir, "fold0")
            knn = _knn_hparams(models_dir / "fold0_knn_model.pkl")
            if ann is None and knn is None:
                continue
            rows.append([method, model, "single split",
                        str(ann["hidden_dim1"]) if ann else "—",
                        str(ann["hidden_dim2"]) if ann else "—",
                        str(ann["max_iter"]) if ann else "—",
                        str(knn[0]) if knn else "—",
                        knn[1] if knn else "—", knn[2] if knn else "—"])
    elif experiment in ("PAIP-EV", "SurGen-EV"):
        for method, model in _combos_for(experiment):
            models_dir = artifact_dir(method, model, 42, "MSIH")
            ann = _ann_config(models_dir, "foldfull")
            knn = _knn_hparams(models_dir / "full_knn_model.pkl")
            if ann is None and knn is None:
                continue
            rows.append([method, model, "TCGA-FULL (seed 42)",
                        str(ann["hidden_dim1"]) if ann else "—",
                        str(ann["hidden_dim2"]) if ann else "—",
                        str(ann["max_iter"]) if ann else "—",
                        str(knn[0]) if knn else "—",
                        knn[1] if knn else "—", knn[2] if knn else "—"])
    return rows


HPARAM_HEADER = ["Method", "Model", "Fold", "ANN h1", "ANN h2", "ANN max_iter",
                 "kNN k", "kNN metric", "kNN weights"]


def add_radar_pair(doc, experiment: str) -> bool:
    """Embed the AUROC and BalAcc radar PNGs for one experiment. Returns False
    (and adds nothing) if the figures have not been generated for it."""
    rdir = RADAR_ROOT / experiment
    paths = [rdir / f"{experiment}_AUROC_radar.png", rdir / f"{experiment}_BalAcc_radar.png"]
    if not all(p.exists() for p in paths):
        return False
    for p in paths:
        doc.add_picture(str(p), width=Inches(6.3))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph()
    return True


def coverage_df(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    sub = df[df["Experiment"] == experiment]
    methods = sorted(set(sub["Method"]))
    models = sorted(set(sub["Model"]))
    rows = []
    for me in methods:
        row = {"Aggregation method": me}
        for mo in models:
            row[mo] = "yes" if len(sub[(sub["Method"] == me) & (sub["Model"] == mo)]) else "—"
        rows.append(row)
    return pd.DataFrame(rows)


def build(out: Path = OUT_DOCX) -> Path:
    df = load()
    cmp_df = load_comparison()
    doc = Document()

    add_heading(doc, "Supplementary Report — Updated Results Sections", 17, space_before=0)
    add_body(doc,
             "This document replaces sections 4.1–4.4 of Supplementary_Report_DRAFT_v2 "
             "and adds a new section 4.5. Every table is generated directly from the "
             "result files produced by the corrected pipeline; none is transcribed by "
             "hand.")

    add_heading(doc, "Why the section names changed", 13)
    add_body(doc,
             "The earlier draft labelled three cross-validation experiments as \"(IV)\". "
             "Internal validation and cross-validation are different protocols, and a "
             "reviewer will read the distinction as claimed rather than incidental. The "
             "naming below follows one vocabulary throughout:")
    add_table(doc, pd.DataFrame([
        {"Previous heading": "4.1 TCGA (IV) Validation Results",
         "Now": "4.1 TCGA-CV", "Reason": "4-fold cross-validation, not a provider split"},
        {"Previous heading": "4.2 PAIP (IV) Validation Results",
         "Now": "4.2 PAIP-IV", "Reason": "the old PAIP cross-validation is retired; "
                                          "replaced by the provider's own split"},
        {"Previous heading": "4.3 TCGA → PAIP (EV) Validation Results",
         "Now": "4.3 PAIP-EV", "Reason": "unchanged protocol; now three training variants"},
        {"Previous heading": "4.4 SurGen (IV) Validation Results",
         "Now": "4.4 SurGen-CV", "Reason": "cross-validation, now case-grouped"},
        {"Previous heading": "(did not exist)",
         "Now": "4.5 SurGen-EV", "Reason": "new external validation on SurGen"},
    ]), ["Previous heading", "Now", "Reason"])

    add_body(doc,
             "CV denotes cross-validation: the cohort is split into K folds, each serves "
             "as test once, and the reported metric is the mean across folds. IV denotes "
             "internal validation against a single train/test split defined by the data "
             "provider, with no averaging. EV denotes external validation: trained "
             "entirely on one cohort, tested on the whole of another, with no data from "
             "the test cohort in training.")

    # ---- per section ----------------------------------------------------
    for number, title, exp in SECTIONS:
        if exp not in set(df["Experiment"]):
            continue
        sub = df[df["Experiment"] == exp]
        agg = sub[~sub["Method"].isin(SLIDE_LEVEL_ENCODERS)]
        enc = sub[sub["Method"].isin(SLIDE_LEVEL_ENCODERS)]
        n_test = sub["N_test"].dropna()
        n_test = int(n_test.iloc[0]) if len(n_test) else "—"

        doc.add_page_break()
        add_heading(doc, f"{number}  {title}", 15, space_before=0)
        add_body(doc, f"Test set size: N = {n_test}.")
        add_body(doc, BLURB[exp])

        # ---- classifier configuration: how each of the five heads was ------
        # actually trained and thresholded for THIS experiment. Read from the
        # result files rather than retyped, so a protocol change (PAIP-IV's
        # ANN already changed once) cannot leave this table quietly wrong.
        add_heading(doc, f"{number}.1  Classifier configuration", 12)
        if exp in ("TCGA-CV", "SurGen-CV"):
            cfg_rows = cv_config_rows(exp)
        elif exp == "PAIP-IV":
            cfg_rows = paip_iv_config_rows()
        else:
            cfg_rows = ev_config_rows_for(exp)
        if cfg_rows:
            add_table(doc, pd.DataFrame(cfg_rows, columns=CONFIG_HEADER),
                      CONFIG_HEADER)
        else:
            add_body(doc, "(configuration unavailable - no result file found)",
                     italic=True)

        add_heading(doc, f"{number}.2  Coverage", 12)
        add_table(doc, coverage_df(df, exp), list(coverage_df(df, exp).columns))

        add_heading(doc, f"{number}.3  Results by aggregation method", 12)
        add_body(doc,
                 "Sorted by balanced accuracy with AUROC as the tie-break. AUROC is the "
                 "primary metric throughout: it is threshold-free and therefore immune "
                 "to the calibration differences between cohorts.", italic=True)
        cols = ["Method", "Model", "Classifier", "Variant", "BalAcc", "AUROC",
                "Acc", "MacroF1"]
        if exp == "PAIP-IV":
            cols = ["Method", "Model", "Classifier", "BalAcc", "BalAcc_CI",
                    "AUROC", "AUROC_CI", "Acc", "MacroF1"]
        # PAIP-EV mixes two threshold schemes, so a row's metrics do not say what
        # produced them. Carry the scheme into the table rather than leaving it to
        # the prose, since these tables get lifted out of the document.
        if "Threshold_scheme" in sub.columns and sub["Threshold_scheme"].notna().any():
            cols = cols + ["Threshold_scheme"]
        add_table(doc, sort_summary(agg), cols, max_rows=30)
        if len(agg) > 30:
            add_body(doc, f"Showing the 30 best of {len(agg)} rows; the complete table "
                          f"is in best_of_all_exps_metric.xlsx.", italic=True, size=9)

        add_heading(doc, f"{number}.4  Radar plots", 12)
        add_body(doc, "Primary variant, all aggregation-method × encoder "
                      "combinations. Each spoke is one (classifier, aggregation "
                      "method) pair; colour distinguishes the foundation model.",
                 italic=True)
        if not add_radar_pair(doc, exp):
            add_body(doc, "(radar plots not found - run "
                          "tools/make_radar_plots.py first)", italic=True)

        if not enc.empty:
            add_heading(doc, f"{number}.5  TITAN and PRISM (slide-level encoders)", 12)
            add_body(doc,
                     "TITAN and PRISM produce a slide-level embedding directly and have "
                     "no patch-aggregation step. They are baselines rather than "
                     "aggregation strategies, and are tabulated separately so they are "
                     "not read as comparable rows in the table above.", italic=True)
            add_table(doc, sort_summary(enc), cols, max_rows=20)

        if exp == "PAIP-IV":
            add_heading(doc, f"{number}.6  Note on the retired PAIP cross-validation", 12)
            add_body(doc,
                     "Earlier drafts reported a four-fold cross-validation on PAIP. It "
                     "has been withdrawn for two reasons. First, its folds were built by "
                     "slicing the label CSV in order, so they tracked PAIP's own "
                     "acquisition split rather than a random partition: folds 1 and 2 "
                     "consisted entirely of training_data_* slides and fold 4 entirely of "
                     "validation_data_* slides, and the class balance came out even by "
                     "luck of ordering rather than by design.\n\n"
                     "Second, and decisively, the experiment is dominated by the one "
                     "reported here. Even with fold construction repaired it would train "
                     "on roughly 36 slides and test on 18 with about 4 positives, whereas "
                     "the provider split trains on 42 and tests on 31. It therefore "
                     "trained on less data and tested on smaller sets while answering the "
                     "same question. Where uncertainty on PAIP is needed, the bootstrap "
                     "intervals above are the appropriate tool at this sample size.")

    # ---- comparison ------------------------------------------------------
    if cmp_df is not None and len(cmp_df):
        doc.add_page_break()
        add_heading(doc, "4.6  Comparison against previously reported results", 15,
                    space_before=0)
        add_body(doc,
                 "Mean change per experiment and metric between the previously reported "
                 "results and those above. The three rows behave differently, and the "
                 "differences are informative rather than incidental.")
        add_table(doc, cmp_df, ["Experiment", "Metric", "n", "mean_delta",
                                "median_delta", "n_improved", "n_worse"],
                  float_fmt="{:+.4f}")
        add_body(doc,
                 "TCGA-CV is essentially unchanged: the corrections that apply to it "
                 "largely offset one another, and logistic regression and ProtoNet are "
                 "unchanged to four decimal places, which confirms the corrections "
                 "affected only what they were intended to affect.\n\n"
                 "SurGen-CV falls, because sibling slides from the same case no longer "
                 "appear on both sides of the train/test boundary. This is the cost of "
                 "the correction and should be presented as such.\n\n"
                 "PAIP-EV rises substantially, for two compounding reasons: the decision "
                 "threshold is now fixed on TCGA out-of-fold predictions rather than left "
                 "at an uncalibrated default, and the primary model trains on all 413 "
                 "TCGA slides rather than the roughly 208 available to a single "
                 "cross-validation fold-model.")

    # ---- caveats ---------------------------------------------------------
    doc.add_page_break()
    add_heading(doc, "4.7  Caveats to state in Methods", 15, space_before=0)
    for title, body in CAVEATS:
        add_heading(doc, title, 11.5, space_before=10)
        add_body(doc, body)

    # ---- appendix: per-combination hyperparameters ------------------------
    # The classifier-configuration tables in each 4.N.1 give the SEARCH SPACE
    # for ANN and kNN (both have real multi-point grids), not which point won
    # for each of the ~20 aggregation-method x encoder combinations - a
    # 5-row-per-classifier table cannot say that. This appendix does, read
    # straight off the saved artifacts rather than retyped: ANN from each
    # fold's *_ann_config.json, kNN from the fitted estimator's own
    # n_neighbors/metric/weights attributes - eval_knn's GridSearchCV winner
    # is never written into any result JSON, so this is the only source.
    doc.add_page_break()
    add_heading(doc, "Appendix A  Per-combination hyperparameters", 17, space_before=0)
    add_body(doc,
             "LR, ProtoNet and Random Forest have single-point grids (stated fully "
             "in each section's 4.N.1) so there is nothing per-combination to add. "
             "ANN and kNN are searched separately for every aggregation-method x "
             "foundation-model combination; this appendix states what each search "
             "actually selected, read directly off the saved model files rather "
             "than retyped by hand.")
    for i, exp in enumerate(["TCGA-CV", "PAIP-IV", "PAIP-EV",
                             "SurGen-CV", "SurGen-EV"], start=1):
        rows = hparam_rows(exp)
        add_heading(doc, f"A.{i}  {exp}", 12)
        if exp in ("PAIP-EV", "SurGen-EV"):
            add_body(doc,
                     "Both external-validation experiments apply the same "
                     "TCGA-FULL artifact, so this table is identical to the "
                     "other EV experiment's.", italic=True, size=9)
        if rows:
            add_table(doc, pd.DataFrame(rows, columns=HPARAM_HEADER), HPARAM_HEADER)
        else:
            add_body(doc, "(no saved model files found for this experiment)",
                     italic=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DOCX)
    args = ap.parse_args()
    p = build(args.out)
    size = p.stat().st_size / 1024
    print(f"wrote {p}  ({size:.0f} KB)")
    print("The original Supplementary_Report_DRAFT_v2.docx is untouched.")


if __name__ == "__main__":
    main()
