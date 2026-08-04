"""
slide_classification_surgen_best_k_labels_exp.py
─────────────────────────────────────────────────────────────────────────────
SurGen-dataset adaptation of tissue_combination_search_9cls.py.

Phase 1 — Individual class evaluation (global, across all foundation models)
    Tests each of the 8 active tissue classes (background excluded) in
    isolation (1×N vector) across every foundation model in one combined pass.
    Each class gets a bacc column per foundation model.
    A final 'avg_bacc' column averages across models and produces a global
    class ranking.
    Results saved to: …/Global/<task>/tissue_individual_class_results_9_global.xlsx
    Generates and saves cross-validated confusion matrix heatmap graphs.

Phase 2 — Fixed top-k combo evaluation
    For each k in k_values, selects the globally top-k tissue classes by
    avg_bacc and runs that single fixed combination across all foundation
    models. No exhaustive combination search.
    Results saved to: …/Global/<task>/tissue_topk_summary_surgen_k{k}.xlsx
    Generates and saves the final combination confusion matrix heatmap graphs.

Model saving
    One best model is kept per (foundation model, k value).
    Saved to: …/models/best_{k}labels_TTC/
    All intermediate checkpoints are deleted immediately after each run.
"""

import os
import gc
import shutil
import random
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from itertools import product as iproduct

from eval_patch_features.logistic import eval_linear
from eval_patch_features.ann import eval_ANN
from eval_patch_features.knn import eval_knn
from eval_patch_features.protonet import eval_protonet
from eval_patch_features.r_forest_eval import eval_r_forest
from eval_patch_features.metrics import get_eval_metrics, print_metrics
from utility import calculate_metric_averages, average_confusion_matrices, write_data_in_excel, build_probs_df

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAMES = ["h-optimus-1"]

MODEL_BASE_DIMS = {
    "h-optimus-1": 1536,
    "UNI2":        1536,
    "Conch1_5":     768,
    "Virchow2":    2560,
    "ConchV1":      512,
}

TASK_PREFIXES = {
    "MSIH": "1-MSIH",
    "BRAF": "2-BRAF",
    "KRAS": "3-KRAS",
    "TP53": "4-TP53",
    "CIMP": "5-CIMP",
}

SURGEN_TASK_CONFIG = {
    "MSIH": ("label_desc", 0),
    "BRAF": ("BRAF_mutation", "WT"),
    "KRAS": ("KRAS_mutation", "WT"),
    "TP53": ("TP53_mutation", "WT"),
}

TISSUE_CLASS_NAMES = [
    "adipose",       # index 0
    "background",    # index 1  ← NEVER used
    "debris",        # index 2
    "lymphocyte",    # index 3
    "mucin",         # index 4
    "smooth_muscle", # index 5
    "normal_mucosa", # index 6
    "stroma",        # index 7
    "tumor",         # index 8
]
NUM_TISSUE_CLASSES = 9   # total rows in each .pt tensor (do not change)

# Background is meaningless for classification — exclude it everywhere.
# NUM_TISSUE_CLASSES stays 9 because the .pt files still have 9 rows;
# ACTIVE_CLASS_INDICES is the only list passed as row_indices to the dataset.
BACKGROUND_IDX       = TISSUE_CLASS_NAMES.index("background")   # → 1
ACTIVE_CLASS_INDICES = [i for i in range(NUM_TISSUE_CLASSES) if i != BACKGROUND_IDX]
NUM_ACTIVE_CLASSES   = len(ACTIVE_CLASS_INDICES)                 # → 8

BATCH_SIZE  = 4
task_names  = ["MSIH"]
model_types = ["ann"]
k_values    = [3, 5, 8]
rank_by     = "bacc"

NUM_FOLDS   = 4
RANDOM_SEED = 42

BASE_ROOT       = "/media/dp-psau/Datum/Aamir/Azfaar"
PROJECT_ROOT    = os.path.join(BASE_ROOT, "surgen_processing")
LABELS_CSV_PATH = "/home/mle/Aamir/Azfaar/surgen_processing/surgen_labels.csv"

RESULT_FOLDER      = "Surgen_TTC_exp"
AGGREGATION_METHOD = "Tissue_Type_Clustering"
sheet_name         = "tissue_search"


def build_runtime_folds(slide_ids: List[str], num_folds: int = NUM_FOLDS, seed: int = RANDOM_SEED) -> List[List[str]]:
    ids = list(slide_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)

    folds = [[] for _ in range(num_folds)]
    for i, sid in enumerate(ids):
        folds[i % num_folds].append(sid)
    return folds


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH GENERATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def save_confusion_matrix_graph(cm: np.ndarray, title: str, save_path: str):
    """Plots and outputs a clear confusion matrix heatmap layout."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=True,
                xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"    ✓ Graph saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class TissueCombinationDataset(Dataset):
    """
    Loads flattened (num_classes × feature_dim) .pt tensors and exposes
    only the rows listed in `row_indices`, re-flattened to (k × feature_dim).
    SurGen-specific: labels are integers (0/1), -1 means exclude.
    """

    def __init__(
        self,
        save_dir:     str,
        fold_ids:     List[str],
        row_indices:  List[int],
        num_classes:  int,
        feature_dim:  int,
        label_col:    str,
        neg_value:    str,
        k_folds_path: str,
    ):
        self.save_dir    = save_dir
        self.fold_ids    = set(fold_ids)
        self.row_indices = row_indices
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.label_col   = label_col
        self.neg_value   = neg_value

        folds_df      = pd.read_csv(k_folds_path)
        self.folds_df = folds_df.set_index("WSI_Id")

        self.data: List[Tuple[torch.Tensor, int, str]] = []
        self._load()

    def _load(self):
        expected_len = self.num_classes * self.feature_dim

        for fname in os.listdir(self.save_dir):
            if not fname.endswith(".pt"):
                continue
            wsi_id = os.path.splitext(fname)[0]
            if wsi_id not in self.fold_ids:
                continue
            if wsi_id not in self.folds_df.index:
                continue

            try:
                feats = torch.load(os.path.join(self.save_dir, fname), map_location="cpu")
                flat  = feats.cpu().flatten()

                if flat.numel() != expected_len:
                    print(f"[WARN] {fname}: expected {expected_len} elements, got {flat.numel()}. Skipping.")
                    continue

                matrix   = flat.view(self.num_classes, self.feature_dim)
                selected = matrix[self.row_indices].flatten()

                raw = self.folds_df.at[wsi_id, self.label_col]
                if pd.isna(raw):
                    continue

                raw = int(raw)
                if raw == -1:
                    continue
                if raw not in (0, 1):
                    print(f"[WARN] Unexpected label {raw} for {wsi_id}")
                    continue

                self.data.append((selected, raw, wsi_id))

            except Exception as exc:
                print(f"[ERROR] {fname}: {exc}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feats, label, wsi_id = self.data[idx]
        return feats, label, wsi_id


# ─────────────────────────────────────────────────────────────────────────────
# MODEL SAVING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

class BestModelTracker:
    """
    Tracks the single best model checkpoint for one (foundation model, k) pair.
    Deletes any checkpoint that is not the current best immediately.
    Call .cleanup(keep_dir) at the end to move the winner to its final location.
    """

    def __init__(self, label: str, stage_dir: Optional[str] = None):
        self.label      = label
        self.best_score = -1.0
        self.best_path: Optional[str] = None
        self.stage_dir  = stage_dir

    def update(self, score: float, candidate_path: Optional[str]) -> bool:
        if score <= self.best_score:
            if candidate_path and os.path.exists(candidate_path):
                try:
                    os.remove(candidate_path)
                except OSError:
                    pass
            return False

        if self.best_path and os.path.exists(self.best_path):
            try:
                os.remove(self.best_path)
            except OSError:
                pass

        self.best_score = score

        if candidate_path and os.path.exists(candidate_path) and self.stage_dir:
            os.makedirs(self.stage_dir, exist_ok=True)
            ext       = os.path.splitext(candidate_path)[1]
            safe_path = os.path.join(self.stage_dir, f"volatile_best_{self.label}{ext}")
            shutil.copy2(candidate_path, safe_path)
            self.best_path = safe_path
        else:
            self.best_path = candidate_path

        return True

    def cleanup(self, keep_dir: str):
        if self.best_path and os.path.exists(self.best_path):
            os.makedirs(keep_dir, exist_ok=True)
            dest = os.path.join(
                keep_dir,
                f"best_{self.label}_{rank_by}{self.best_score:.4f}.pt",
            )
            shutil.copy2(self.best_path, dest)
            try:
                os.remove(self.best_path)
            except OSError:
                pass
            print(f"  ✓ Best model saved → {dest}")
        else:
            print(f"  [WARN] No best model found for {self.label} — nothing saved.")


def _temp_model_dir(output_save_path: str, label: str) -> str:
    """Scratch directory for a single combo's intermediate checkpoints."""
    d = os.path.join(output_save_path, "_tmp_models", label.replace("+", "_"))
    os.makedirs(d, exist_ok=True)
    return d


def _cleanup_temp_dir(output_save_path: str):
    tmp = os.path.join(output_save_path, "_tmp_models")
    if os.path.isdir(tmp):
        shutil.rmtree(tmp, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# K-FOLD RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_combo_kfold(
    task_name:       str,
    data_path:       str,
    k_folds_path:    str,
    folds:           List[List[str]],
    row_indices:     List[int],
    num_classes:     int,
    feature_dim:     int,
    model_type:      str,
    model_save_path: str,
    label_col:       str,
    neg_value:       str,
    tracker:         BestModelTracker,
) -> List[Dict[str, Any]]:
    """
    Runs k-fold CV for one (combo, model_type) pair.
    Updates `tracker` after every fold.
    """
    vector_dim       = len(row_indices) * feature_dim
    # [CORRECTION 2026-08-04, work order Task 1.1] For the ANN this resolved to
    # "ann_macro_f1", computed by eval_ANN on the TEST set, so the best of the 4
    # grid configurations was selected on test performance. Other classifiers have
    # single-point grids, so no selection occurs and their key is unchanged.
    metric_key       = f"val_{model_type}_macro_f1" if model_type == "ann" else f"{model_type}_macro_f1"
    results_per_fold = []
    num_folds        = len(folds)

    for i in range(num_folds):
        test_ids  = folds[i]
        val_ids   = folds[(i + 1) % num_folds]
        train_ids = [
            wsi_id
            for j, fold in enumerate(folds)
            if j != i and j != (i + 1) % num_folds
            for wsi_id in fold
        ]

        def make_loader(ids, shuffle):
            ds = TissueCombinationDataset(
                save_dir     = data_path,
                fold_ids     = ids,
                row_indices  = row_indices,
                num_classes  = num_classes,
                feature_dim  = feature_dim,
                label_col    = label_col,
                neg_value    = neg_value,
                k_folds_path = k_folds_path,
            )
            if len(ds) == 0:
                raise ValueError(f"Empty dataset for fold {i+1} — check paths.")
            return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

        train_loader = make_loader(train_ids, shuffle=True)
        val_loader   = make_loader(val_ids,   shuffle=False)
        test_loader  = make_loader(test_ids,  shuffle=False)

        def collect(loader):
            feats, labels, ids = [], [], []
            for f, l, w in loader:
                feats.append(f)
                labels.append(l)
                ids.extend(w if isinstance(w, (list, tuple)) else [w])
            return torch.cat(feats), torch.cat(labels), ids

        tr_f, tr_l, _            = collect(train_loader)
        va_f, va_l, _            = collect(val_loader)
        te_f, te_l, test_ids_out = collect(test_loader)

        best_metrics, best_dump, best_params = None, None, None
        last_saved_path = None

        if model_type == "lin":
            for C, max_iter in iproduct([10], [300]):
                m, d = eval_linear(
                    fold=i, train_feats=tr_f, train_labels=tr_l,
                    valid_feats=va_f, valid_labels=va_l,
                    test_feats=te_f,  test_labels=te_l,
                    max_iter=max_iter, save_path=model_save_path,
                    C=C, verbose=False,
                )
                last_saved_path = os.path.join(model_save_path, f"lin_fold{i}.pkl")
                if best_metrics is None or m[metric_key] > best_metrics[metric_key]:
                    best_metrics, best_dump, best_params = m, d, (C, max_iter)

        elif model_type == "ann":
            for h1, h2, max_iter in iproduct([128, 256], [64, 128], [500]):
                m, d = eval_ANN(
                    fold=i, train_feats=tr_f, train_labels=tr_l,
                    valid_feats=va_f, valid_labels=va_l,
                    test_feats=te_f,  test_labels=te_l,
                    input_dim=vector_dim, hidden_dim=h1, hidden_dim2=h2,
                    max_iter=max_iter, model_save_path=model_save_path,
                    verbose=False,
                )
                last_saved_path = os.path.join(model_save_path, f"ann_fold{i}.pt")
                if best_metrics is None or m[metric_key] > best_metrics[metric_key]:
                    best_metrics, best_dump, best_params = m, d, (h1, h2, max_iter)

        elif model_type == "knn":
            for k_n in [3]:
                m, d = eval_knn(
                    fold=i, train_feats=tr_f, train_labels=tr_l,
                    val_feats=va_f,   val_labels=va_l,
                    test_feats=te_f,  test_labels=te_l,
                    n_neighbors=k_n, normalize_feats=True,
                    model_save_path=model_save_path, verbose=False,
                )
                last_saved_path = os.path.join(model_save_path, f"knn_fold{i}.pkl")
                if best_metrics is None or m[metric_key] > best_metrics[metric_key]:
                    best_metrics, best_dump, best_params = m, d, k_n

        elif model_type == "proto":
            best_metrics, best_dump = eval_protonet(
                fold=i, train_feats=tr_f, train_labels=tr_l,
                val_feats=va_f, val_labels=va_l,
                test_feats=te_f, test_labels=te_l,
                normalize_feats=True, model_save_path=model_save_path,
            )
            last_saved_path = os.path.join(model_save_path, f"proto_fold{i}.pt")
            best_params = None

        elif model_type == "rf":
            for n_est, max_d, min_split, min_leaf, cw, thresh in iproduct(
                [500], [None], [5], [1], [{0: 1, 1: 10}], [0.3]
            ):
                m, d = eval_r_forest(
                    fold=i, train_feats=tr_f, train_labels=tr_l,
                    valid_feats=va_f, valid_labels=va_l,
                    test_feats=te_f,  test_labels=te_l,
                    n_estimators=n_est, max_depth=max_d,
                    min_samples_split=min_split, min_samples_leaf=min_leaf,
                    class_weight=cw, prediction_threshold=thresh,
                    save_path=model_save_path, verbose=False,
                )
                last_saved_path = os.path.join(model_save_path, f"rf_fold{i}.pkl")
                if best_metrics is None or m[metric_key] > best_metrics[metric_key]:
                    best_metrics, best_dump, best_params = m, d, (
                        n_est, max_d, min_split, min_leaf, cw, thresh)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        fold_score = best_metrics.get(f"{model_type}_{rank_by}", 0.0)
        tracker.update(fold_score, last_saved_path)

        if best_params:
            print(f"  Fold {i+1} | {model_type} best params: {best_params} | {rank_by}={fold_score:.4f}")

        results_per_fold.append({
            **best_metrics,
            **best_dump,
            "wsi_ids": test_ids_out,
            "fold":    i + 1,
        })

    return results_per_fold


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — individual class evaluation (global, across all foundation models)
# ─────────────────────────────────────────────────────────────────────────────

def run_phase1_individual_classes(
    task_name:          str,
    all_model_configs:  List[Dict],   # list of dicts: model_name, data_path, feature_dim, output_save_path
    k_folds_path:       str,
    folds:              List[List[str]],
    label_col:          str,
    neg_value:          str,
    global_output_path: str,          # where to write the combined Phase 1 xlsx
) -> pd.DataFrame:
    """
    Runs Phase 1 (individual class evaluation) across ALL foundation models in
    one pass. Only the 8 active classes are evaluated (background is skipped).
    Each class gets a bacc column per foundation model. A final 'avg_bacc'
    column averages across models and is used to rank classes globally.

    Returns a DataFrame sorted by avg_bacc descending — used by Phase 2 to pick
    the top-k fixed combo.
    """
    print(f"\n{'─'*60}")
    print(f"  PHASE 1 — Individual class evaluation "
          f"({NUM_ACTIVE_CLASSES} active classes, background excluded, all models)")
    print(f"  Task: {task_name}")
    print(f"{'─'*60}")

    metric_indices = {"acc": 0, "bacc": 1, "macro_f1": 2, "weighted_f1": 3, "auroc": 4}

    # Initialise result rows — one per ACTIVE class only
    rows = [
        {"class_idx": i, "class_name": TISSUE_CLASS_NAMES[i]}
        for i in ACTIVE_CLASS_INDICES
    ]
    # Map class_idx → position in rows list for fast lookup
    idx_to_row = {i: pos for pos, i in enumerate(ACTIVE_CLASS_INDICES)}

    for cfg in all_model_configs:
        model_name       = cfg["model_name"]
        data_path        = cfg["data_path"]
        feature_dim      = cfg["feature_dim"]
        output_save_path = cfg["output_save_path"]

        print(f"\n  ── Model: {model_name} ──")

        for class_idx in ACTIVE_CLASS_INDICES:
            class_name = TISSUE_CLASS_NAMES[class_idx]
            print(f"    Class {class_idx}: {class_name}")

            model_save_path = _temp_model_dir(output_save_path, f"p1_{class_name}")
            tracker = BestModelTracker(f"{model_name}_phase1_{class_name}")

            bacc_vals = []
            for model_type in model_types:
                try:
                    fold_results = _run_one_combo_kfold(
                        task_name       = task_name,
                        data_path       = data_path,
                        k_folds_path    = k_folds_path,
                        folds           = folds,
                        row_indices     = [class_idx],
                        num_classes     = NUM_TISSUE_CLASSES,
                        feature_dim     = feature_dim,
                        model_type      = model_type,
                        model_save_path = model_save_path,
                        label_col       = label_col,
                        neg_value       = neg_value,
                        tracker         = tracker,
                    )

                    avg = calculate_metric_averages(
                        [{k: v for k, v in r.items()
                          if k in [f"{model_type}_{m}" for m in metric_indices]}
                         for r in fold_results],
                        metric_indices,
                        model_prefix=model_type,
                    )
                    b = avg.get(f"{model_type}_bacc", float("nan"))
                    bacc_vals.append(b)

                    # Store per-metric results keyed by foundation model + classifier
                    for m in metric_indices:
                        col = f"{model_name}_{model_type}_{m}"
                        rows[idx_to_row[class_idx]][col] = avg.get(f"{model_type}_{m}", float("nan"))

                    # Confusion matrix graph per model × class
                    avg_cm     = average_confusion_matrices(fold_results, model_prefix=model_type)
                    graph_path = os.path.join(
                        output_save_path,
                        f"confusion_matrix_individual_{class_name}_{model_name}_{model_type}.png"
                    )
                    save_confusion_matrix_graph(
                        avg_cm,
                        f"CM: {class_name} | {model_name} ({model_type.upper()})",
                        graph_path,
                    )

                except Exception as exc:
                    print(f"      [ERROR] {model_type}: {exc}")
                    for m in metric_indices:
                        rows[idx_to_row[class_idx]][f"{model_name}_{model_type}_{m}"] = float("nan")

            # Best bacc for this foundation model (across classifier variants)
            valid = [v for v in bacc_vals if not np.isnan(v)]
            rows[idx_to_row[class_idx]][f"{model_name}_bacc"] = (
                max(valid) if valid else float("nan")
            )

        _cleanup_temp_dir(output_save_path)

    # Compute average bacc across foundation models
    bacc_cols = [f"{cfg['model_name']}_bacc" for cfg in all_model_configs]
    df = pd.DataFrame(rows)
    df["avg_bacc"] = df[bacc_cols].mean(axis=1, skipna=True)
    df = df.sort_values("avg_bacc", ascending=False).reset_index(drop=True)
    df["global_rank"] = df.index + 1

    print(f"\n  Global class ranking by avg_bacc:")
    for _, row in df.iterrows():
        print(f"    #{int(row['global_rank']):2d}  {row['class_name']:<22s}  avg_bacc={row['avg_bacc']:.4f}")

    os.makedirs(global_output_path, exist_ok=True)
    out_path = os.path.join(global_output_path, "tissue_individual_class_results_9_global.xlsx")
    write_data_in_excel(out_path, df, sheet_name="individual_classes_9")
    print(f"\n  ✓ Phase 1 global results → {out_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — fixed top-k combo evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_phase2_topk_combo(
    task_name:          str,
    all_model_configs:  List[Dict],
    k_folds_path:       str,
    folds:              List[List[str]],
    label_col:          str,
    neg_value:          str,
    class_df:           pd.DataFrame,   # output of Phase 1, sorted by avg_bacc desc
    global_output_path: str,
):
    """
    For each k value, selects the globally top-k tissue classes (by avg_bacc)
    and trains/evaluates that single fixed combination across all foundation
    models. Writes one summary xlsx per k containing all model results.
    """
    metric_indices = {"acc": 0, "bacc": 1, "macro_f1": 2, "weighted_f1": 3, "auroc": 4}

    ranked_df = class_df.sort_values("avg_bacc", ascending=False).reset_index(drop=True)

    for k in k_values:
        if k > NUM_ACTIVE_CLASSES:
            print(f"  [WARN] k={k} exceeds number of active classes ({NUM_ACTIVE_CLASSES}), skipping.")
            continue

        top_k_rows  = ranked_df.head(k)
        row_indices = top_k_rows["class_idx"].tolist()
        combo_label = "+".join(TISSUE_CLASS_NAMES[i] for i in row_indices)
        combo_baccs = top_k_rows["avg_bacc"].tolist()

        print(f"\n{'─'*60}")
        print(f"  PHASE 2 — k={k} | Fixed top-{k} combo")
        print(f"  Combo : {combo_label}")
        print(f"  Avg bacc of selected classes: "
              + ", ".join(f"{b:.4f}" for b in combo_baccs))
        print(f"{'─'*60}")

        summary_rows: List[Dict]            = []
        best_combo_probs: Optional[pd.DataFrame] = None

        for cfg in all_model_configs:
            model_name       = cfg["model_name"]
            data_path        = cfg["data_path"]
            feature_dim      = cfg["feature_dim"]
            output_save_path = cfg["output_save_path"]
            models_root_dir  = cfg["models_root_dir"]

            k_best_dir = os.path.join(models_root_dir, f"best_{k}labels_TTC")
            os.makedirs(k_best_dir, exist_ok=True)

            print(f"\n  Model: {model_name}")

            stage_dir       = os.path.join(output_save_path, "_tracked_bests")
            k_tracker       = BestModelTracker(f"{model_name}_surgen_k{k}", stage_dir=stage_dir)
            model_save_path = _temp_model_dir(output_save_path, combo_label)
            combo_probs_df: Optional[pd.DataFrame] = None

            for model_type in model_types:
                try:
                    fold_results = _run_one_combo_kfold(
                        task_name       = task_name,
                        data_path       = data_path,
                        k_folds_path    = k_folds_path,
                        folds           = folds,
                        row_indices     = row_indices,
                        num_classes     = NUM_TISSUE_CLASSES,
                        feature_dim     = feature_dim,
                        model_type      = model_type,
                        model_save_path = model_save_path,
                        label_col       = label_col,
                        neg_value       = neg_value,
                        tracker         = k_tracker,
                    )

                    model_df = build_probs_df(fold_results, model_name=model_type)
                    combo_probs_df = (
                        model_df if combo_probs_df is None
                        else pd.merge(combo_probs_df, model_df,
                                      on=["Fold", "WSI_ID", "Target"], how="outer")
                    )

                    avg = calculate_metric_averages(
                        [{key: v for key, v in r.items()
                          if key in [f"{model_type}_{m}" for m in metric_indices]}
                         for r in fold_results],
                        metric_indices,
                        model_prefix=model_type,
                    )

                    summary_rows.append({
                        "foundation_model": model_name,
                        "combo":            combo_label,
                        "row_indices":      str(row_indices),
                        "k":                k,
                        "classifier":       model_type,
                        **{m: avg.get(f"{model_type}_{m}", float("nan"))
                           for m in metric_indices},
                    })

                    print(f"    {model_type}: bacc={avg.get(f'{model_type}_bacc', float('nan')):.4f}  "
                          f"macro_f1={avg.get(f'{model_type}_macro_f1', float('nan')):.4f}")

                    # Confusion matrix graph per model × k
                    avg_cm     = average_confusion_matrices(fold_results, model_prefix=model_type)
                    graph_path = os.path.join(
                        output_save_path,
                        f"confusion_matrix_combo_k{k}_{model_name}_{model_type}.png"
                    )
                    save_confusion_matrix_graph(
                        avg_cm,
                        f"CM: Top-{k} Combo | {model_name} ({model_type.upper()})",
                        graph_path,
                    )

                except Exception as exc:
                    print(f"    [ERROR] {model_type}: {exc}")

            if combo_probs_df is not None and best_combo_probs is None:
                best_combo_probs = combo_probs_df.copy()

            if os.path.isdir(model_save_path):
                shutil.rmtree(model_save_path, ignore_errors=True)

            if os.path.isdir(stage_dir):
                shutil.rmtree(stage_dir, ignore_errors=True)

            k_tracker.cleanup(keep_dir=k_best_dir)
            gc.collect()

        # Write summary for this k — rows are all foundation models × classifiers
        summary_df   = pd.DataFrame(summary_rows).sort_values(rank_by, ascending=False)
        summary_path = os.path.join(global_output_path, f"tissue_topk_summary_surgen_k{k}.xlsx")
        write_data_in_excel(summary_path, summary_df, sheet_name=f"k{k}")
        print(f"\n  ✓ Summary k={k} → {summary_path}")
        print(f"  Combo: {combo_label}")

        if best_combo_probs is not None:
            probs_path = os.path.join(global_output_path, f"tissue_topk_probs_surgen_k{k}.xlsx")
            write_data_in_excel(probs_path, best_combo_probs, sheet_name=f"k{k}_probs")
            print(f"  Probs → {probs_path}")

    _cleanup_temp_dir(global_output_path)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

labels_df       = pd.read_csv(LABELS_CSV_PATH)
valid_slides_df = labels_df[labels_df["label_desc"] != -1]

print("MSIH:",    (valid_slides_df["label_desc"] == 1).sum())
print("nonMSIH:", (valid_slides_df["label_desc"] == 0).sum())

valid_ids = set(labels_df.loc[labels_df["label_desc"] != -1, "WSI_Id"])

for task_name in task_names:
    folder_task_name   = TASK_PREFIXES.get(task_name, task_name)
    label_col, neg_value = SURGEN_TASK_CONFIG[task_name]

    # Build config list for all foundation models
    all_model_configs = []
    for MODEL_NAME in MODEL_NAMES:
        feature_dim = MODEL_BASE_DIMS[MODEL_NAME]

        DATA_PATH = os.path.join(
            BASE_ROOT, "surgen_processed", MODEL_NAME,
            "features", "slide_aggregation", AGGREGATION_METHOD, MODEL_NAME
        )

        if not os.path.isdir(DATA_PATH):
            print(f"\n[WARN] No feature folder for {MODEL_NAME} at {DATA_PATH} — skipping.")
            continue

        RESULT_ROOT      = os.path.join(PROJECT_ROOT, RESULT_FOLDER, AGGREGATION_METHOD, MODEL_NAME)
        OUTPUT_SAVE_PATH = os.path.join(RESULT_ROOT, folder_task_name, "Output")
        MODELS_ROOT_DIR  = os.path.join(RESULT_ROOT, folder_task_name, "models")
        os.makedirs(OUTPUT_SAVE_PATH, exist_ok=True)
        os.makedirs(MODELS_ROOT_DIR,  exist_ok=True)

        all_model_configs.append({
            "model_name":       MODEL_NAME,
            "data_path":        DATA_PATH,
            "feature_dim":      feature_dim,
            "output_save_path": OUTPUT_SAVE_PATH,
            "models_root_dir":  MODELS_ROOT_DIR,
        })

    if not all_model_configs:
        print(f"\n[WARN] No valid model configs for task {task_name} — skipping.")
        continue

    # Build runtime folds from the first available model's data directory
    # (all models share the same slide IDs)
    first_data_path = all_model_configs[0]["data_path"]
    all_slide_ids   = [
        os.path.splitext(f)[0] for f in os.listdir(first_data_path)
        if f.endswith(".pt") and os.path.splitext(f)[0] in valid_ids
    ]
    folds = build_runtime_folds(all_slide_ids, num_folds=NUM_FOLDS, seed=RANDOM_SEED)

    # Global output path — one level above any single model's folder
    GLOBAL_OUTPUT = os.path.join(
        PROJECT_ROOT, RESULT_FOLDER,
        AGGREGATION_METHOD, "Global", folder_task_name
    )
    os.makedirs(GLOBAL_OUTPUT, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Task: {task_name} | {NUM_ACTIVE_CLASSES}-class TTC (background excluded) | "
          f"Models: {[c['model_name'] for c in all_model_configs]}")
    print(f"  {len(all_slide_ids)} slides | {NUM_FOLDS} folds")
    print(f"{'='*70}")

    # Phase 1 — evaluate all 8 active classes across all models; get global ranking
    class_df = run_phase1_individual_classes(
        task_name          = task_name,
        all_model_configs  = all_model_configs,
        k_folds_path       = LABELS_CSV_PATH,
        folds              = folds,
        label_col          = label_col,
        neg_value          = neg_value,
        global_output_path = GLOBAL_OUTPUT,
    )

    # Phase 2 — test single top-k combo per k value, across all models
    run_phase2_topk_combo(
        task_name          = task_name,
        all_model_configs  = all_model_configs,
        k_folds_path       = LABELS_CSV_PATH,
        folds              = folds,
        label_col          = label_col,
        neg_value          = neg_value,
        class_df           = class_df,
        global_output_path = GLOBAL_OUTPUT,
    )

    print(f"\n  ✓ Done: {task_name} ({NUM_ACTIVE_CLASSES} active classes, background excluded, top-k fixed combos)")
