import torch
import torchvision
import os
from os.path import join as j_
from PIL import Image
import numpy as np
import time
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Lambda
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
# loading all packages here to start
# from dataloader import WSIDataset
from eval_patch_features.logistic import eval_linear
from eval_patch_features.ann import eval_ANN
from eval_patch_features.knn import eval_knn
from eval_patch_features.protonet import eval_protonet
from eval_patch_features.r_forest_eval import eval_r_forest
from eval_patch_features.metrics import get_eval_metrics, print_metrics
from utility import calculate_metric_averages, average_confusion_matrices, write_data_in_excel, build_probs_df
import warnings
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import torch
from typing import List
import pandas as pd
import random
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# --- DYNAMIC CONFIGURATIONS ---
# WINDOWS RUN — conch1-5 and virchow2 were saved to two DIFFERENT drives
# (D: and F:), so there's no single PROJECT_ROOT that both live under like
# on the Linux machine. MODEL_NAME is set to the real saved folder name for
# each, and DATA_PATH/RESULT_ROOT are built per-model from MODEL_ROOTS below
# instead of one shared PROJECT_ROOT + MODEL_NAME join.
AGGREGATION_METHODS = ["Averaging", "Tissue_Type_Clustering", "Caption_based_aggregation"]  # ["Averaging", "Caption_based_aggregation", "Tissue_Type_Clustering", "TITAN", "PRISM"]
MODEL_NAMES = ["conch1-5", "virchow2"]  # Conch1_5, Virchow2 — real saved folder names

# Root directory each model's "features" folder lives directly under.
# (conch1-5 -> D: drive, virchow2 -> D: drive — separate roots, separate drives.)
MODEL_ROOTS = {
    "conch1-5": r"D:\Aamir Gulzar\KSA_project2\surgen_data\surgen_processed\conch1-5",
    "virchow2": r"D:\Aamir Gulzar\KSA_project2\surgen_data\surgen_processed\virchow2",
}

# Where results get written. Both models now write into this single shared
# results folder (instead of each writing under its own model_root/drive).
RESULT_ROOT_BASE = r"D:\Aamir Gulzar\KSA_project2\Cancer-detection-classifier\slide_classification\SurGen_Results"

# Define Base Dimensions for each model (keys must match MODEL_NAMES exactly)
MODEL_BASE_DIMS = {
    "conch1-5": 768,
    "virchow2": 2560,
    "PRISM": 1280,  # inactive — no PRISM entry in MODEL_NAMES, no known path on this machine
}

# Define Multipliers for each aggregation method
AGG_MULTIPLIERS = {
    "Caption_based_aggregation": 14,
    "Tissue_Type_Clustering": 9,
    "Averaging": 1
}

TASK_PREFIXES = {
    "MSIH": "1-MSIH",
    "BRAF": "2-BRAF",
    "KRAS": "3-KRAS",
    "TP53": "4-TP53",
    "CIMP": "5-CIMP"
}

HIDDEN_DIM = 128
BATCH_SIZE = 4
task_names = ["MSIH"]  # "MSIH", "BRAF", "KRAS", "TP53", "CIMP"

# ---- RUNTIME FOLD CONFIG (replaces reading folds from a pre-built CSV) ----
NUM_FOLDS = 4
RANDOM_SEED = 42

# Path to your SurGen label source: must have a slide-id column and a label
# column for the task(s) you run (e.g. 'WSI_Id' and 'label_desc' for MSIH).
# label_desc is numeric: 0 = nonMSIH, 1 = MSIH, -1 = unknown/unlabeled.
# This is ONLY used to attach labels to each slide id discovered on disk —
# fold membership itself is no longer read from this file, it's generated
# at runtime from whatever slide ids exist in DATA_PATH.
LABELS_CSV_PATH = r"D:\Aamir Gulzar\KSA_project2\Cancer-detection-classifier\slide_classification\surgen_labels.csv"

# No shared PROJECT_ROOT on this machine — conch1-5 and virchow2 live on
# separate drives (see MODEL_ROOTS above). DATA_PATH / RESULT_ROOT are built
# per-model further down instead.

sheet_name = "baseline"


def build_runtime_folds(slide_ids: List[str], labels_csv_path: str, num_folds: int = NUM_FOLDS,
                         seed: int = RANDOM_SEED) -> List[List[str]]:
    """
    Splits slide ids into num_folds folds at runtime, replacing the old
    behavior of reading fold membership from a pre-built CSV.

    Stratified by label_desc so each fold gets an (as close to) equal share
    of label-0 (nonMSIH) and label-1 (MSIH) slides. Slides with label_desc
    == -1 (unlabeled) are excluded entirely — they never enter any fold.
    Slides with no row at all in labels_csv_path are also excluded, since
    they have no usable label.
    """
    labels_df = pd.read_csv(labels_csv_path)
    label_lookup = dict(zip(labels_df['WSI_Id'], labels_df['label_desc']))

    rng = random.Random(seed)

    folds = [[] for _ in range(num_folds)]

    # Group available slide ids by label, dropping -1/unknown and missing labels
    by_label = {}
    skipped_unlabeled = 0
    skipped_missing = 0
    for sid in slide_ids:
        if sid not in label_lookup:
            skipped_missing += 1
            continue
        lbl = label_lookup[sid]
        if lbl == -1:
            skipped_unlabeled += 1
            continue
        by_label.setdefault(lbl, []).append(sid)

    print(f"  [folds] Excluded {skipped_unlabeled} unlabeled (-1) slides, "
          f"{skipped_missing} slides missing from labels CSV.")

    # Distribute each label group round-robin across folds after shuffling,
    # so every fold ends up with an equal (±1) share of each class.
    for lbl, ids in sorted(by_label.items()):
        ids = list(ids)
        rng.shuffle(ids)
        counts = [0] * num_folds
        for i, sid in enumerate(ids):
            fold_idx = i % num_folds
            folds[fold_idx].append(sid)
            counts[fold_idx] += 1
        print(f"  [folds] Label {lbl}: {len(ids)} slides -> per-fold counts {counts}")

    return folds


# ===========================================================================
# OUTER LOOP: Iterate over all (aggregation_method, model_name) combinations
# ===========================================================================
for AGGREGATION_METHOD in AGGREGATION_METHODS:
    for MODEL_NAME in MODEL_NAMES:

        # TITAN is only used with Conch1_5, so we skip incompatible combinations
        if AGGREGATION_METHOD == "TITAN" and MODEL_NAME != "Conch1_5":
            continue
        if AGGREGATION_METHOD == "PRISM" and MODEL_NAME != "PRISM":
            continue


        print(f"\n{'='*70}")
        print(f"  Aggregation: {AGGREGATION_METHOD} | Model: {MODEL_NAME}")
        print(f"{'='*70}\n")

        # Calculate VECTOR_DIM automatically
        # TITAN is a special case (768 regardless), so we handle it with a fallback
        if AGGREGATION_METHOD == "TITAN":
            VECTOR_DIM = 768
        elif AGGREGATION_METHOD == "PRISM":
            VECTOR_DIM = 1280
        else:
            base = MODEL_BASE_DIMS.get(MODEL_NAME, 768)
            mult = AGG_MULTIPLIERS.get(AGGREGATION_METHOD, 1)
            VECTOR_DIM = base * mult

        if AGGREGATION_METHOD == "PRISM":
            # Not currently reachable — no "PRISM" entry in MODEL_ROOTS on this
            # machine. Add one to MODEL_ROOTS above before enabling this.
            raise ValueError(
                "PRISM has no known root path on this machine. "
                "Add MODEL_ROOTS['PRISM'] before using AGGREGATION_METHOD='PRISM'."
            )

        else:
            model_root = MODEL_ROOTS[MODEL_NAME]

            DATA_PATH = os.path.join(
                model_root,
                "features",
                "slide_aggregation",
                AGGREGATION_METHOD,
                MODEL_NAME
            )

            RESULT_ROOT = os.path.join(
                RESULT_ROOT_BASE,
                AGGREGATION_METHOD,
                MODEL_NAME
            )

        for task_name in task_names:
            folder_task_name = TASK_PREFIXES.get(task_name, task_name)

            # Build task-specific paths
            MODEL_SAVE_PATH = os.path.join(RESULT_ROOT, folder_task_name, "models")
            OUTPUT_SAVE_PATH = os.path.join(RESULT_ROOT, folder_task_name, "Output")

            # Create directories
            os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
            os.makedirs(OUTPUT_SAVE_PATH, exist_ok=True)

            # Final File Paths
            EVAL_METRICS_EXCEL = os.path.join(OUTPUT_SAVE_PATH, "cross_valid_avg_eval_metrics.xlsx")
            PROBS_ALL_EXCEL = os.path.join(OUTPUT_SAVE_PATH, "cross_valid_avg_probs_all.xlsx")

            print(f"Processing Task: {task_name} using {MODEL_NAME} via {AGGREGATION_METHOD}")

            class WSIDataset(Dataset):
                def __init__(self, save_dir: str, fold_ids: List[str]):
                    self.data = []
                    self.save_dir = save_dir
                    self.fold_ids = fold_ids
                    self._load_data()

                def _load_data(self):
                    files = os.listdir(self.save_dir)

                    # Labels are read from a single labels CSV regardless of task,
                    # since fold membership is no longer sourced from a per-task
                    # fold CSV. The CSV must contain 'WSI_Id' plus whichever label
                    # column the active task_name needs.
                    folds_df = pd.read_csv(LABELS_CSV_PATH)

                    for wsi_file in files:
                        wsi_path = os.path.join(self.save_dir, wsi_file)

                        # Extract WSI ID without extension
                        wsi_id = os.path.splitext(wsi_file)[0]

                        # Skip files not in fold_ids
                        if wsi_id not in self.fold_ids:
                            continue

                        # Only process .pt files
                        if not wsi_path.endswith('.pt'):
                            continue

                        try:
                            # Load WSI features
                            wsi_features = torch.load(wsi_path)

                            if wsi_features.is_cuda:
                                wsi_features = wsi_features.cpu()

                            # Average or flatten features
                            if wsi_features.dim() > 1:
                                final_features = wsi_features.flatten()
                            else:
                                final_features = wsi_features

                            # Determine label based on task_name
                            if task_name == "MSIH":
                                # MSIH classification — label_desc is numeric: 0=nonMSIH, 1=MSIH, -1=unknown
                                if wsi_id in folds_df['WSI_Id'].values:
                                    fold_label = folds_df.loc[folds_df['WSI_Id'] == wsi_id, 'label_desc'].values[0]
                                    if fold_label == -1:
                                        # Unlabeled slide — should already be excluded from fold_ids,
                                        # but skip defensively if one slips through.
                                        continue
                                    label = int(fold_label)

                            elif task_name == "BRAF":
                                # BRAF classification
                                if wsi_id in folds_df['WSI_Id'].values:
                                    fold_label = folds_df.loc[folds_df['WSI_Id'] == wsi_id, 'BRAF_mutation'].values[0]
                                    label = 0 if fold_label == 'WT' else 1

                            elif task_name == "KRAS":
                                # KRAS classification
                                if wsi_id in folds_df['WSI_Id'].values:
                                    fold_label = folds_df.loc[folds_df['WSI_Id'] == wsi_id, 'KRAS_mutation'].values[0]
                                    label = 0 if fold_label == 'WT' else 1

                            elif task_name == "TP53":
                                # TP53 classification
                                if wsi_id in folds_df['WSI_Id'].values:
                                    fold_label = folds_df.loc[folds_df['WSI_Id'] == wsi_id, 'TP53_mutation'].values[0]
                                    label = 0 if fold_label == 'WT' else 1

                            else:
                                raise ValueError(f"Unknown task name: {task_name}")

                            # Store in dataset
                            self.data.append((final_features, label, wsi_id))

                        except Exception as e:
                            print(f"[ERROR] Loading failed for {wsi_path}: {e}")

                def __len__(self):
                    return len(self.data)

                def __getitem__(self, idx):
                    features, label, wsi_id = self.data[idx]
                    return features, label, wsi_id


            # Pipeline 2
            from itertools import product

            def train_and_evaluate(fold, train_loader, val_loader, test_loader, model_type='linear'):
                def get_feats_labels(loader):
                    feats, labels, ids = [], [], []
                    for features, lbls, wsi_ids in loader:
                        feats.append(features)
                        labels.append(lbls)
                        if isinstance(wsi_ids, (list, tuple)):
                            ids.extend(wsi_ids)
                        else:
                            ids.append(wsi_ids)
                    return torch.cat(feats), torch.cat(labels), ids

                train_feats, train_labels, _ = get_feats_labels(train_loader)
                val_feats, val_labels, _ = get_feats_labels(val_loader)
                test_feats, test_labels, all_test_ids = get_feats_labels(test_loader)

                train_feats = train_feats.float()
                val_feats = val_feats.float()
                test_feats = test_feats.float()

                best_metrics = None
                best_dump = None
                best_params = None

                if model_type == 'lin':
                    param_grid = {
                        'C': [10],
                        'max_iter': [300]
                    }
                    for C, max_iter in product(param_grid['C'], param_grid['max_iter']):
                        eval_metrics, eval_dump = eval_linear(
                            fold=fold,
                            train_feats=train_feats,
                            train_labels=train_labels,
                            valid_feats=val_feats,
                            valid_labels=val_labels,
                            test_feats=test_feats,
                            test_labels=test_labels,
                            max_iter=max_iter,
                            save_path=MODEL_SAVE_PATH,
                            C=C,
                            verbose=False,
                        )
                        if not best_metrics or eval_metrics['lin_macro_f1'] > best_metrics['lin_macro_f1']:
                            best_metrics = eval_metrics
                            best_dump = eval_dump
                            best_params = (C, max_iter)

                elif model_type == 'ann':
                    param_grid = {
                        'hidden_dim1': [64, 128, 256, 512],
                        'hidden_dim2': [32, 64, 128, 256],
                        'max_iter': [300, 500]
                    }
                    for h1, h2, max_iter in product(param_grid['hidden_dim1'], param_grid['hidden_dim2'], param_grid['max_iter']):
                        eval_metrics, eval_dump = eval_ANN(
                            fold=fold,
                            train_feats=train_feats,
                            train_labels=train_labels,
                            valid_feats=val_feats,
                            valid_labels=val_labels,
                            test_feats=test_feats,
                            test_labels=test_labels,
                            input_dim=VECTOR_DIM,
                            hidden_dim=h1,
                            hidden_dim2=h2,
                            max_iter=max_iter,
                            model_save_path=MODEL_SAVE_PATH,
                            verbose=False,
                        )
                        if not best_metrics or eval_metrics['ann_macro_f1'] > best_metrics['ann_macro_f1']:
                            best_metrics = eval_metrics
                            best_dump = eval_dump
                            best_params = (h1, h2, max_iter)

                elif model_type == 'knn':
                    for k in [3,5,7]:
                        eval_metrics, eval_dump = eval_knn(
                            fold=fold,
                            train_feats=train_feats,
                            train_labels=train_labels,
                            val_feats=val_feats,
                            val_labels=val_labels,
                            test_feats=test_feats,
                            test_labels=test_labels,
                            n_neighbors=k,
                            normalize_feats=True,
                            model_save_path=MODEL_SAVE_PATH,
                            verbose=False
                        )
                        if not best_metrics or eval_metrics['knn_macro_f1'] > best_metrics['knn_macro_f1']:
                            best_metrics = eval_metrics
                            best_dump = eval_dump
                            best_params = k

                elif model_type == 'proto':
                    eval_metrics, eval_dump = eval_protonet(
                        fold=fold,
                        train_feats=train_feats,
                        train_labels=train_labels,
                        val_feats=val_feats,
                        val_labels=val_labels,
                        test_feats=test_feats,
                        test_labels=test_labels,
                        normalize_feats=True,
                        model_save_path=MODEL_SAVE_PATH,
                    )
                    best_metrics = eval_metrics
                    best_dump = eval_dump
                    best_params = None

                elif model_type == 'rf':
                    param_grid = {
                        'n_estimators': [500],
                        'max_depth': [None],
                        'min_samples_split': [5],
                        'min_samples_leaf': [1],
                        'class_weight': [{0: 1, 1: 10}],
                        'prediction_threshold': [0.3]
                    }
                    for n_est, max_d, min_split, min_leaf, cw, thresh in product(
                        param_grid['n_estimators'],
                        param_grid['max_depth'],
                        param_grid['min_samples_split'],
                        param_grid['min_samples_leaf'],
                        param_grid['class_weight'],
                        param_grid['prediction_threshold']
                    ):
                        eval_metrics, eval_dump = eval_r_forest(
                            fold=fold,
                            train_feats=train_feats,
                            train_labels=train_labels,
                            valid_feats=val_feats,
                            valid_labels=val_labels,
                            test_feats=test_feats,
                            test_labels=test_labels,
                            n_estimators=n_est,
                            max_depth=max_d,
                            min_samples_split=min_split,
                            min_samples_leaf=min_leaf,
                            class_weight=cw,
                            prediction_threshold=thresh,
                            save_path=MODEL_SAVE_PATH,
                            verbose=False,
                        )
                        if not best_metrics or eval_metrics['rf_macro_f1'] > best_metrics['rf_macro_f1']:
                            best_metrics = eval_metrics
                            best_dump = eval_dump
                            best_params = (n_est, max_d, min_split, min_leaf, cw, thresh)

                else:
                    raise ValueError(f"Unsupported model type: {model_type}")

                if best_params:
                    print(f"Fold {fold} | Best Params for {model_type}: {best_params}")
                return best_metrics, best_dump, all_test_ids

            from typing import List
            def run_k_fold_cross_validation(save_dir: str, folds: List[List[str]], model_type: str = 'lin'):
                results_per_fold = []
                num_folds = len(folds)

                for i in range(num_folds):
                    # Define test and validation folds
                    test_ids = folds[i]
                    val_ids = folds[(i + 1) % num_folds]

                    # Use remaining folds as training
                    train_ids = []
                    for j in range(num_folds):
                        if j != i and j != (i + 1) % num_folds:
                            train_ids.extend(folds[j])
                    print(f"Running Fold {i + 1} with model {model_type}...")
                    print(f"  Train IDs count: {len(train_ids)}, Val IDs count: {len(val_ids)}, Test IDs count: {len(test_ids)}")
                    # Create datasets and loaders
                    train_dataset = WSIDataset(save_dir, train_ids)
                    val_dataset = WSIDataset(save_dir, val_ids)
                    test_dataset = WSIDataset(save_dir, test_ids)
                    print(f"  Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
                    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
                        print(f"  ERROR: Empty dataset detected!")
                        print(f"  Sample train IDs: {train_ids[:3] if train_ids else 'None'}")
                        print(f"  Files in directory: {os.listdir(save_dir)[:5]}")
                        raise ValueError("Empty dataset - check WSI ID matching with files in directory")
                    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
                    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
                    eval_metrics, eval_dump, all_test_ids = train_and_evaluate(i, train_loader, val_loader, test_loader, model_type=model_type)
                    print_metrics(eval_metrics)
                    result = {
                        **eval_metrics,
                        **eval_dump,
                        "wsi_ids": all_test_ids,
                        "fold": i + 1
                    }
                    results_per_fold.append(result)
                return results_per_fold

            # ---- Build folds at runtime from slide ids found in DATA_PATH ----
            # (replaces: folds_df = pd.read_csv(K_FOLDS_PATH); group by 'fold')
            # Stratified by label_desc, with -1/unlabeled and unknown slides excluded.
            all_slide_ids = [
                os.path.splitext(f)[0]
                for f in os.listdir(DATA_PATH)
                if f.endswith('.pt')
            ]
            num_folds = NUM_FOLDS
            folds = build_runtime_folds(all_slide_ids, labels_csv_path=LABELS_CSV_PATH,
                                         num_folds=num_folds, seed=RANDOM_SEED)

            # Run k-fold cross-validation with different models
            model_types = ['lin', 'ann', 'knn', 'proto', 'rf']
            metric_indices = {
                'acc': 0,
                'bacc': 1,
                'macro_f1': 2,
                'weighted_f1': 3,
                'auroc': 4
            }

            eval_metrics__for_excel = []
            probs_all_for_excel = None
            for model in model_types:
                predictions_list = []
                print(f"\n\n ********* Training with model: {model}********* \n\n")
                k_folds_results = run_k_fold_cross_validation(DATA_PATH, folds, model_type=model)
                model_df = build_probs_df(k_folds_results, model_name=model)
                # === Merge predictions across models ===
                if probs_all_for_excel is None:
                    probs_all_for_excel = model_df
                else:
                    probs_all_for_excel = pd.merge(probs_all_for_excel, model_df, on=["Fold", "WSI_ID", "Target"], how="outer")

                # === Average metrics (only pass metric parts of result dicts)
                average_results = calculate_metric_averages(
                    [{k: v for k, v in result.items() if k in [f"{model}_{m}" for m in metric_indices.keys()]}
                    for result in k_folds_results],
                    metric_indices,
                    model_prefix=model
                )
                # === Confusion matrices
                confusion_matrices = [np.array(result[f"{model}_conf_matrix"]) for result in k_folds_results if f"{model}_conf_matrix" in result]

                avg_conf_matrix = average_confusion_matrices(confusion_matrices)
                print("\n\n Average results for all folds:")
                for metric, value in average_results.items():
                    print(f"{metric}: {value:.4f}")
                # Append per metric rows for each fold + average
                for metric in metric_indices.keys():
                    row = [f"{model}_{metric}"]
                    for result in k_folds_results:
                        row.append(result.get(f"{model}_{metric}", 'N/A'))
                    row.append(average_results.get(f"{model}_{metric}", 'N/A'))
                    eval_metrics__for_excel.append(row)

                # Append confusion matrix as string (per fold)
                row = [f"{model}_conf_matrix"]
                for result in k_folds_results:
                    row.append(str(result.get(f"{model}_conf_matrix", "N/A")))
                row.append(str(avg_conf_matrix))
                eval_metrics__for_excel.append(row)

            eval_metrics_df = pd.DataFrame(eval_metrics__for_excel,
                                    columns=["Metric", "Fold1", "Fold2", "Fold3", "Fold4", "AvgFolds"])
            write_data_in_excel(EVAL_METRICS_EXCEL, eval_metrics_df, sheet_name=sheet_name)
            write_data_in_excel(PROBS_ALL_EXCEL, probs_all_for_excel, sheet_name=sheet_name)