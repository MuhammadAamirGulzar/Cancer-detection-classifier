"""
tissue_combination_experiment.py
─────────────────────────────────
Drop-in extension for your existing classification pipeline.

What it does
────────────
For a Tissue_Type_Clustering feature vector of shape (NUM_CLASSES × N)
stored as a flattened .pt tensor, this module tests every combination of
`k` tissue rows to find which subset gives the best classification.

Usage
─────
    from tissue_combination_experiment import run_tissue_combination_search

    run_tissue_combination_search(
        task_name        = "MSIH",
        data_path        = DATA_PATH,           # folder of .pt files
        k_folds_path     = K_FOLDS_PATH,        # CSV with fold / WSI_Id / label cols
        folds            = folds,               # List[List[str]] already built in your script
        output_save_path = OUTPUT_SAVE_PATH,
        num_tissue_classes = 9,                 # total rows in the clustered vector
        feature_dim        = 768,               # N  (per-row dimension, e.g. 768 or 1536)
        k_values           = [3],               # which combination sizes to test
        model_types        = ["lin"],           # subset of ['lin','ann','knn','proto','rf']
        rank_by            = "auroc",           # metric used to rank combinations
    )

The function writes two Excel files to output_save_path:
  • tissue_combo_summary_k{k}.xlsx  – one row per combination, sorted by best metric
  • tissue_combo_probs_k{k}.xlsx    – per-WSI probabilities for the best combination
"""

import os
import itertools
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Dataset wrapper
# ──────────────────────────────────────────────────────────────────────────────

class TissueCombinationDataset(Dataset):
    """
    Wraps the raw .pt feature files for a Tissue_Type_Clustering run.

    Each .pt file must contain a 1-D tensor of length (num_tissue_classes * feature_dim).
    The tensor is interpreted as a (num_tissue_classes × feature_dim) matrix; only the
    rows listed in `row_indices` are kept, then re-flattened.

    Parameters
    ----------
    save_dir    : directory containing .pt feature files
    fold_ids    : WSI IDs that belong to this split
    row_indices : which tissue-class rows to keep  (e.g. [0, 3, 7])
    num_classes : total number of tissue classes   (default 9)
    feature_dim : per-class feature dimension      (e.g. 768 or 1536)
    task_name   : used to look up the correct label column
    k_folds_path: path to the fold CSV
    """

    # Label-column spec per task  ─ extend as needed
    LABEL_COLS = {
        "MSIH": ("label_desc",      lambda v: 0 if v == "nonMSIH" else 1),
        "BRAF": ("BRAF_mutation",   lambda v: 0 if v == "WT" else 1),
        "KRAS": ("KRAS_mutation",   lambda v: 0 if v == "WT" else 1),
        "TP53": ("TP53_mutation",   lambda v: 0 if v == "WT" else 1),
        "CIMP": ("CIMP",            lambda v: 0 if v == "neg" else 1),
    }

    def __init__(
        self,
        save_dir: str,
        fold_ids: List[str],
        row_indices: List[int],
        num_classes: int,
        feature_dim: int,
        task_name: str,
        k_folds_path: str,
    ):
        self.save_dir    = save_dir
        self.fold_ids    = set(fold_ids)
        self.row_indices = row_indices
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.task_name   = task_name

        if task_name not in self.LABEL_COLS:
            raise ValueError(f"Unknown task '{task_name}'. Add it to LABEL_COLS.")

        folds_df = pd.read_csv(k_folds_path)
        self.folds_df = folds_df.set_index("WSI_Id")

        self.data: List[Tuple[torch.Tensor, int, str]] = []
        self._load()

    # ------------------------------------------------------------------
    def _load(self):
        col_name, label_fn = self.LABEL_COLS[self.task_name]
        expected_len       = self.num_classes * self.feature_dim

        for fname in os.listdir(self.save_dir):
            if not fname.endswith(".pt"):
                continue
            wsi_id = os.path.splitext(fname)[0]
            if wsi_id not in self.fold_ids:
                continue
            if wsi_id not in self.folds_df.index:
                continue

            try:
                feats = torch.load(os.path.join(self.save_dir, fname),
                                   map_location="cpu")
                if feats.is_cuda:
                    feats = feats.cpu()

                # ── reshape to (num_classes, feature_dim) ──
                flat = feats.flatten()
                if flat.numel() != expected_len:
                    print(
                        f"[WARN] {fname}: expected {expected_len} elements, "
                        f"got {flat.numel()}. Skipping."
                    )
                    continue
                matrix = flat.view(self.num_classes, self.feature_dim)

                # ── select rows ──
                selected = matrix[self.row_indices]          # (k, feature_dim)
                final    = selected.flatten()                # (k * feature_dim,)

                # ── label ──
                raw_label = self.folds_df.at[wsi_id, col_name]
                label     = label_fn(raw_label)

                self.data.append((final, label, wsi_id))

            except Exception as exc:
                print(f"[ERROR] {fname}: {exc}")

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feats, label, wsi_id = self.data[idx]
        return feats, label, wsi_id


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Main search function
# ──────────────────────────────────────────────────────────────────────────────

def run_tissue_combination_search(
    task_name:          str,
    data_path:          str,
    k_folds_path:       str,
    folds:              List[List[str]],
    output_save_path:   str,
    num_tissue_classes: int  = 9,
    feature_dim:        int  = 768,
    k_values:           List[int] = None,
    model_types:        List[str] = None,
    batch_size:         int  = 4,
    rank_by:            str  = "auroc",
    tissue_class_names: List[str] = None,
):
    """
    Exhaustively test every combination of k tissue rows.

    Parameters
    ----------
    tissue_class_names : optional human-readable names for the 9 classes,
                         e.g. ["background","stroma","tumour","necrosis",…]
                         If None, classes are labelled 0–8.
    rank_by            : one of "auroc", "macro_f1", "bacc", "acc"
    """
    # ── imports from your existing pipeline ─────────────────────────────────
    # These must be importable in the same environment.
    from eval_patch_features.logistic import eval_linear
    from eval_patch_features.ann import eval_ANN
    from eval_patch_features.knn import eval_knn
    from eval_patch_features.protonet import eval_protonet
    from eval_patch_features.r_forest_eval import eval_r_forest
    from eval_patch_features.metrics import get_eval_metrics, print_metrics
    from utility import calculate_metric_averages, average_confusion_matrices, write_data_in_excel, build_probs_df
    # If your helpers live inline (not a separate file), move them to a module
    # or paste them above this function.

    if k_values   is None: k_values   = [3]
    if model_types is None: model_types = ["lin"]

    if tissue_class_names is None:
        tissue_class_names = [str(i) for i in range(num_tissue_classes)]

    os.makedirs(output_save_path, exist_ok=True)

    metric_indices = {
        "acc":         0,
        "bacc":        1,
        "macro_f1":    2,
        "weighted_f1": 3,
        "auroc":       4,
    }

    for k in k_values:
        combos = list(itertools.combinations(range(num_tissue_classes), k))
        print(f"\n{'='*60}")
        print(f"Task={task_name}  k={k}  → {len(combos)} combinations")
        print(f"{'='*60}")

        summary_rows: List[Dict[str, Any]] = []
        best_combo_probs_df = None
        best_combo_score    = -1.0
        best_combo_label    = ""

        for combo_idx, row_indices in enumerate(combos):
            combo_label = "+".join(tissue_class_names[i] for i in row_indices)
            vector_dim  = k * feature_dim

            print(f"\n[{combo_idx+1}/{len(combos)}]  rows={list(row_indices)}  ({combo_label})")

            # ── k-fold cross-validation for this combination ─────────────
            probs_all_df     = None
            eval_rows        = []
            num_folds        = len(folds)
            per_fold_results = []

            for model in model_types:
                k_fold_results = _run_kfold(
                    task_name       = task_name,
                    data_path       = data_path,
                    k_folds_path    = k_folds_path,
                    folds           = folds,
                    row_indices     = list(row_indices),
                    num_classes     = num_tissue_classes,
                    feature_dim     = feature_dim,
                    vector_dim      = vector_dim,
                    model_type      = model,
                    batch_size      = batch_size,
                    eval_linear     = eval_linear,
                    eval_ANN        = eval_ANN,
                    eval_knn        = eval_knn,
                    eval_protonet   = eval_protonet,
                    eval_r_forest   = eval_r_forest,
                    print_metrics   = print_metrics,
                    output_save_path= output_save_path,
                    combo_label     = combo_label,
                )

                model_df = build_probs_df(k_fold_results, model_name=model)
                probs_all_df = (
                    model_df if probs_all_df is None
                    else pd.merge(probs_all_df, model_df,
                                  on=["Fold", "WSI_ID", "Target"], how="outer")
                )

                avg = calculate_metric_averages(
                    [{k2: v for k2, v in r.items()
                      if k2 in [f"{model}_{m}" for m in metric_indices]}
                     for r in k_fold_results],
                    metric_indices,
                    model_prefix=model,
                )

                # Collect one summary row per model per combo
                row = {
                    "combo":   combo_label,
                    "rows":    str(list(row_indices)),
                    "k":       k,
                    "model":   model,
                }
                for m in metric_indices:
                    row[m] = avg.get(f"{model}_{m}", float("nan"))
                summary_rows.append(row)

                rank_score = avg.get(f"{model}_{rank_by}", -1.0)
                if rank_score > best_combo_score:
                    best_combo_score  = rank_score
                    best_combo_probs_df = probs_all_df.copy()
                    best_combo_label  = combo_label

        # ── Write summary Excel ──────────────────────────────────────────
        summary_df = pd.DataFrame(summary_rows)
        summary_df.sort_values(by=rank_by, ascending=False, inplace=True)
        summary_excel = os.path.join(
            output_save_path, f"tissue_combo_summary_k{k}.xlsx"
        )
        write_data_in_excel(summary_excel, summary_df, sheet_name="summary")
        print(f"\n✓ Summary saved → {summary_excel}")
        print(f"  Best combo ({rank_by}={best_combo_score:.4f}): {best_combo_label}")

        # ── Write best-combo probs Excel ─────────────────────────────────
        if best_combo_probs_df is not None:
            probs_excel = os.path.join(
                output_save_path, f"tissue_combo_probs_k{k}.xlsx"
            )
            write_data_in_excel(probs_excel, best_combo_probs_df, sheet_name="best_combo")
            print(f"  Best-combo probs → {probs_excel}")


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Internal k-fold runner (mirrors run_k_fold_cross_validation in your script)
# ──────────────────────────────────────────────────────────────────────────────

def _run_kfold(
    task_name, data_path, k_folds_path, folds,
    row_indices, num_classes, feature_dim, vector_dim,
    model_type, batch_size,
    eval_linear, eval_ANN, eval_knn, eval_protonet, eval_r_forest,
    print_metrics, output_save_path, combo_label,
):
    from itertools import product as iproduct

    num_folds        = len(folds)
    results_per_fold = []

    # Temporary model-save dir (one per combo to avoid cross-contamination)
    model_save_path = os.path.join(
        output_save_path, "combo_models", combo_label.replace("+", "_")
    )
    os.makedirs(model_save_path, exist_ok=True)

    metric_key = {
        "lin":   "lin_macro_f1",
        "ann":   "ann_macro_f1",
        "knn":   "knn_macro_f1",
        "proto": "proto_macro_f1",
        "rf":    "rf_macro_f1",
    }[model_type]

    for i in range(num_folds):
        test_ids = folds[i]
        val_ids  = folds[(i + 1) % num_folds]
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
                task_name    = task_name,
                k_folds_path = k_folds_path,
            )
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

        train_loader = make_loader(train_ids, shuffle=True)
        val_loader   = make_loader(val_ids,   shuffle=False)
        test_loader  = make_loader(test_ids,  shuffle=False)

        # ── gather tensors ──────────────────────────────────────────────
        def collect(loader):
            feats, labels, ids = [], [], []
            for f, l, w in loader:
                feats.append(f); labels.append(l)
                ids.extend(w if isinstance(w, (list, tuple)) else [w])
            return torch.cat(feats), torch.cat(labels), ids

        tr_f, tr_l, _          = collect(train_loader)
        va_f, va_l, _          = collect(val_loader)
        te_f, te_l, test_ids_out = collect(test_loader)

        # ── model-specific eval ─────────────────────────────────────────
        best_metrics, best_dump, best_params = None, None, None

        if model_type == "lin":
            for C, max_iter in iproduct([10], [300]):
                m, d = eval_linear(
                    fold=i, train_feats=tr_f, train_labels=tr_l,
                    valid_feats=va_f, valid_labels=va_l,
                    test_feats=te_f,  test_labels=te_l,
                    max_iter=max_iter, save_path=model_save_path,
                    C=C, verbose=False,
                )
                if best_metrics is None or m[metric_key] > best_metrics[metric_key]:
                    best_metrics, best_dump, best_params = m, d, (C, max_iter)

        elif model_type == "ann":
            for h1, h2, max_iter in iproduct([256], [64], [500]):
                m, d = eval_ANN(
                    fold=i, train_feats=tr_f, train_labels=tr_l,
                    valid_feats=va_f, valid_labels=va_l,
                    test_feats=te_f,  test_labels=te_l,
                    input_dim=vector_dim, hidden_dim=h1, hidden_dim2=h2,
                    max_iter=max_iter, model_save_path=model_save_path,
                    verbose=False,
                )
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
                if best_metrics is None or m[metric_key] > best_metrics[metric_key]:
                    best_metrics, best_dump, best_params = m, d, k_n

        elif model_type == "proto":
            best_metrics, best_dump = eval_protonet(
                fold=i, train_feats=tr_f, train_labels=tr_l,
                val_feats=va_f, val_labels=va_l,
                test_feats=te_f, test_labels=te_l,
                normalize_feats=True, model_save_path=model_save_path,
            )
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
                if best_metrics is None or m[metric_key] > best_metrics[metric_key]:
                    best_metrics, best_dump, best_params = m, d, (
                        n_est, max_d, min_split, min_leaf, cw, thresh)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        print_metrics(best_metrics)
        results_per_fold.append({
            **best_metrics,
            **best_dump,
            "wsi_ids": test_ids_out,
            "fold":    i + 1,
        })

    return results_per_fold


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Example: how to call this from your existing script
# ──────────────────────────────────────────────────────────────────────────────
#
# Add this block at the bottom of your main script (inside the task_name loop),
# right after your existing model_types loop:
#
# ── PASTE START ──────────────────────────────────────────────────────────────
#
# if AGGREGATION_METHOD == "Tissue_Type_Clustering":
#     from tissue_combination_experiment import run_tissue_combination_search
#
#     # Optional: name your 9 tissue classes for readable output
#     TISSUE_CLASS_NAMES = [
#         "background",   # 0
#         "stroma",        # 1
#         "tumour",        # 2
#         "necrosis",      # 3
#         "lymphocyte",    # 4
#         "mucin",         # 5
#         "adipose",       # 6
#         "normal_mucosa", # 7
#         "debris",        # 8
#     ]
#
#     run_tissue_combination_search(
#         task_name          = task_name,
#         data_path          = DATA_PATH,
#         k_folds_path       = K_FOLDS_PATH,
#         folds              = folds,
#         output_save_path   = OUTPUT_SAVE_PATH,
#         num_tissue_classes = 9,
#         feature_dim        = MODEL_BASE_DIMS[MODEL_NAME],  # 768 or 1536
#         k_values           = [3, 4, 5],    # test triplets, quads, and quintuples
#         model_types        = ["lin", "rf"], # fastest models first
#         rank_by            = "auroc",
#         tissue_class_names = TISSUE_CLASS_NAMES,
#     )
#
# ── PASTE END ────────────────────────────────────────────────────────────────
