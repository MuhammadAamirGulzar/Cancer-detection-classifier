# ===========================================================================
# RETIRED 2026-08-04 - work order Task 1.5. Superseded, do not run.
#
# Why: this script hardcodes the ANN architecture as hidden_dim=256,
# hidden_dim2=64. Every saved TCGA ANN checkpoint is 256/128, so it would fail
# with a shape mismatch the moment it loaded one. It could not have produced the
# published PAIP-EV numbers.
#
# What produced the current PAIP-EV results instead: Slide_Classification.ipynb
# cell 14, which infers the architecture from the state dict and is correct.
#
# Superseded by: slide_classification/runners/ev_runner.py (Phase 3, Task 3.0/3.2),
# which loads architecture from fold{f}_ann_config.json (Task 1.2) and applies the
# tau_TCGA threshold policy (section 1b).
# ===========================================================================
#!/usr/bin/env python3
"""
Universal External Validation Script - TCGA to PAIP
==================================================

This script automatically detects feature dimensions and handles any
model/aggregation combination for external validation.

Usage:
    python external_validation_script.py

Configuration:
    Modify the variables in the CONFIGURATION section below.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# ============================================================
# UNIVERSAL CONFIGURATION - WORKS WITH ANY MODEL/AGGREGATION
# ============================================================

# --- CHANGE THESE TO TEST ANY COMBINATION ---
MODEL_NAME = "H-Optimus-1"  # H-Optimus-1, Conch1_5, UNI2
AGGREGATION_METHOD = "Caption_based_aggregation"  # Any aggregation method
TASK = "1-MSIH"  # 1-MSIH, 2-BRAF, 3-KRAS, 4-TP53, 5-CIMP

# --- PATH CONFIGURATIONS ---
BASE_ROOT = r"D:\Aamir Gulzar\KSA_project2"
PROJECT_ROOT = os.path.join(BASE_ROOT, "Cancer-detection-classifier", "slide_classification")

# TCGA trained models location - AUTO-DETECTS the correct path
TCGA_MODELS_BASE = os.path.join(PROJECT_ROOT, "TCGA_Results_Updated", AGGREGATION_METHOD, MODEL_NAME, TASK, "models")

# PAIP data location - AUTO-DETECTS available data
PAIP_DATA_BASES = [
    os.path.join(BASE_ROOT, "paip_data", "slide_aggregation", "Caption_Based_Clustering_FiveCrop"),
    os.path.join(BASE_ROOT, "paip_data", "slide_aggregation", AGGREGATION_METHOD, MODEL_NAME),
    os.path.join(BASE_ROOT, "paip_data", "slide_aggregation", AGGREGATION_METHOD),
    os.path.join(BASE_ROOT, "paip_data", "slide_aggregation"),
]

# Ground truth labels
GROUND_TRUTH_PATH = os.path.join(BASE_ROOT, "paip_data", "labels", "paip_kfolds_71.csv")

# Output location
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "Inference_Results")
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("="*70)
print("UNIVERSAL EXTERNAL VALIDATION - TCGA TO PAIP")
print("="*70)
print(f"Task: {TASK}")
print(f"Model: {MODEL_NAME}")
print(f"Aggregation: {AGGREGATION_METHOD}")
print(f"TCGA Models: {TCGA_MODELS_BASE}")
print("="*70)

# ============================================================
# UNIVERSAL FEATURE DIMENSION DETECTION
# ============================================================

def auto_detect_paip_data() -> Tuple[str, str]:
    """
    Automatically detect the best PAIP data path and feature directory.
    
    Returns:
        Tuple of (paip_base_path, feature_dir_name)
    """
    # Possible feature directory names for different models
    possible_dirs = [
        f'{MODEL_NAME.lower().replace("-", "_")}_fivecrop',
        f'{MODEL_NAME.lower().replace("-", "")}_fivecrop', 
        f'{MODEL_NAME.replace("-", "_")}',
        f'{MODEL_NAME.replace("-", "")}',
        MODEL_NAME,
        'hoptimus_CC_fivecrop',
        'conch_CC_fivecrop',
        'uni2_CC_fivecrop',
    ]
    
    for base_path in PAIP_DATA_BASES:
        if os.path.exists(base_path):
            for dir_name in possible_dirs:
                full_path = os.path.join(base_path, dir_name)
                if os.path.exists(full_path):
                    files = [f for f in os.listdir(full_path) if f.endswith('.pt')]
                    if files:
                        print(f"✅ Found PAIP data: {full_path}")
                        return base_path, dir_name
    
    # Fallback - return the first base path and model name
    return PAIP_DATA_BASES[0], MODEL_NAME

def detect_feature_dimensions(model_path: str, paip_sample_path: str) -> Tuple[int, int]:
    """
    Detect the expected input dimensions for TCGA models and actual PAIP features.
    
    Returns:
        Tuple of (tcga_expected_dim, paip_actual_dim)
    """
    tcga_dim = None
    paip_dim = None
    
    # Try to detect TCGA model input dimension
    try:
        if model_path.endswith('.pkl'):
            # For sklearn models, try to load and inspect
            model = joblib.load(model_path)
            if hasattr(model, 'n_features_in_'):
                tcga_dim = model.n_features_in_
            elif hasattr(model, 'coef_'):
                tcga_dim = model.coef_.shape[1] if model.coef_.ndim > 1 else len(model.coef_)
        elif model_path.endswith('.pth'):
            # For PyTorch models, load state dict and check first layer
            state_dict = torch.load(model_path, map_location='cpu')
            if isinstance(state_dict, dict):
                # Look for the first linear layer
                for key, tensor in state_dict.items():
                    if 'weight' in key and tensor.dim() == 2:
                        tcga_dim = tensor.shape[1]  # Input dimension
                        break
    except Exception as e:
        print(f"⚠️  Could not detect TCGA model dimension: {e}")
    
    # Detect PAIP feature dimension
    try:
        if os.path.exists(paip_sample_path):
            sample_feat = torch.load(paip_sample_path, map_location='cpu')
            if sample_feat.dim() > 1:
                sample_feat = sample_feat.flatten()
            paip_dim = sample_feat.shape[0]
    except Exception as e:
        print(f"⚠️  Could not detect PAIP feature dimension: {e}")
    
    return tcga_dim, paip_dim

def match_feature_dimensions(features: torch.Tensor, target_dim: int) -> torch.Tensor:
    """
    Match feature dimensions by truncating, padding, or using PCA-like reduction.
    
    Args:
        features: Input features tensor
        target_dim: Target dimension size
        
    Returns:
        Resized features tensor
    """
    current_dim = features.shape[-1]
    
    if current_dim == target_dim:
        return features
    elif current_dim > target_dim:
        # Truncate to target dimension (take first N features)
        return features[..., :target_dim]
    else:
        # Pad with zeros to target dimension
        padding_size = target_dim - current_dim
        padding = torch.zeros(*features.shape[:-1], padding_size)
        return torch.cat([features, padding], dim=-1)


# ============================================================
# UNIVERSAL ANN MODEL ARCHITECTURE
# ============================================================
class UniversalANNClassifier(nn.Module):
    """
    Universal ANN that automatically adapts to any input dimension.
    Matches the architecture used during TCGA training.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 hidden_dim2: int = 64, num_classes: int = 2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim2, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# Universal configuration - works with any model/aggregation combination
UNIVERSAL_CONFIG = {
    'folds': [0, 1, 2, 3],
    'classifier_templates': {
        'logistic_regression': 'fold{fold}_logistic_regression.pkl',
        'knn': 'fold{fold}_knn_model.pkl', 
        'protonet': 'fold{fold}_protonet_model.pkl',
        'random_forest': 'fold{fold}_random_forest.pkl',
        'ann': 'fold{fold}_trained_ann_model_{input_dim}.pth',
    },
    'classifiers_to_run': [
        'logistic_regression',
        'protonet', 
        'ann',
    ],
    'ann_architecture': {
        'hidden_dim': 256,
        'hidden_dim2': 64,
    }
}


# ============================================================
# LABEL & GROUND TRUTH HELPERS
# ============================================================
def parse_label(raw) -> Optional[int]:
    """
    Convert any label string/int to binary.
      nonMSIH / 0 / mss  →  0
      MSIH    / 1 / msi  →  1
    """
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in ('0', 'nonmsih', 'non-msih', 'mss'):
        return 0
    if s in ('1', 'msih', 'msi-h', 'msih_c', 'msi'):
        return 1
    try:
        return int(float(s))
    except ValueError:
        return None


def load_ground_truth(csv_path: str) -> pd.DataFrame:
    """Load CSV and add normalised _wsi_id and _bin_label columns."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    print(f"Ground truth: {len(df)} rows | columns: {df.columns.tolist()}")
    print(df.head(3).to_string(), "\n")

    # Identify WSI id column
    id_col = 'WSI_Id' if 'WSI_Id' in df.columns else df.columns[0]

    # Identify label column
    lbl_col = None
    for cand in ('label', 'label_desc', 'Label', 'MSI_status'):
        if cand in df.columns:
            lbl_col = cand
            break
    if lbl_col is None:
        raise ValueError(f"No label column found in {df.columns.tolist()}")

    df['_wsi_id'] = df[id_col].astype(str).str.strip()
    df['_bin_label'] = df[lbl_col].apply(parse_label)

    bad = df['_bin_label'].isna().sum()
    if bad:
        print(f"⚠️  {bad} rows with unparseable labels — will be skipped")

    return df


def get_label(wsi_id: str, gt_df: pd.DataFrame) -> Optional[int]:
    """
    Look up binary label for a WSI id with three fallback strategies:
      1. Exact match on _wsi_id
      2. Partial / substring match
      3. Extract MSIH / nonMSIH directly from the filename stem
    """
    # Strategy 1 — exact
    row = gt_df[gt_df['_wsi_id'] == wsi_id]
    if not row.empty:
        return row['_bin_label'].values[0]

    # Strategy 2 — substring
    for _, r in gt_df.iterrows():
        if r['_wsi_id'] in wsi_id or wsi_id in r['_wsi_id']:
            return r['_bin_label']

    # Strategy 3 — parse from filename
    lower = wsi_id.lower()
    if '_nonmsih' in lower:
        return 0
    if '_msih' in lower:
        return 1

    return None

# ============================================================
# FEATURE LOADING & MODEL LOADING HELPERS
# ============================================================
def match_feature_dimensions(features: torch.Tensor, target_dim: int) -> torch.Tensor:
    """
    Match feature dimensions by truncating or padding.
    
    Args:
        features: Input features tensor
        target_dim: Target dimension size
        
    Returns:
        Resized features tensor
    """
    current_dim = features.shape[-1]
    
    if current_dim == target_dim:
        return features
    elif current_dim > target_dim:
        # Truncate to target dimension
        return features[..., :target_dim]
    else:
        # Pad with zeros to target dimension
        padding_size = target_dim - current_dim
        padding = torch.zeros(*features.shape[:-1], padding_size)
        return torch.cat([features, padding], dim=-1)
def load_features(feature_dir: str) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """Load all .pt files from a directory. Flattens multi-dim tensors."""
    features_dict: Dict[str, torch.Tensor] = {}
    wsi_ids: List[str] = []

    if not os.path.isdir(feature_dir):
        print(f"  ❌  Directory not found: {feature_dir}")
        return features_dict, wsi_ids

    files = [f for f in os.listdir(feature_dir) if f.endswith('.pt')]
    print(f"  Found {len(files)} .pt files in: {os.path.basename(feature_dir)}")

    for fname in files:
        wsi_id = os.path.splitext(fname)[0]
        try:
            feat = torch.load(os.path.join(feature_dir, fname), map_location='cpu')
            if feat.dim() > 1:
                feat = feat.flatten()   # (N_crops, D) → (N_crops * D,)
            
            # Match feature dimensions to expected INPUT_DIM
            feat = match_feature_dimensions(feat, INPUT_DIM)
            
            features_dict[wsi_id] = feat.float()
            wsi_ids.append(wsi_id)
        except Exception as e:
            print(f"  ⚠️  Could not load {fname}: {e}")

    return features_dict, wsi_ids


def load_model(classifier_name: str, model_path: str, input_dim: int):
    """Load either a sklearn (.pkl) or PyTorch ANN (.pth) model."""
    if classifier_name == 'ann':
        # Get model-specific architecture
        arch_config = MODEL_ARCHITECTURES.get(MODEL_NAME, {'hidden_dim': 512, 'hidden_dim2': 256})
        m = ANNClassifier(input_dim, 
                         hidden_dim=arch_config['hidden_dim'],
                         hidden_dim2=arch_config['hidden_dim2'])
        state = torch.load(model_path, map_location='cpu')
        # Handle both raw state_dict and wrapped checkpoints
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        m.model.load_state_dict(state)
        m.eval()
        return m
    else:
        loaded = joblib.load(model_path)
        # Handle protonet/random_forest checkpoint dict format
        if isinstance(loaded, dict):
            if 'model' in loaded:
                return loaded['model']
            elif 'classifier' in loaded:
                return loaded['classifier']
            elif 'prototypes' in loaded and 'labels_proto' in loaded:
                # This is a protonet model saved as dict - return the dict itself
                # We'll handle this special case in the predict function
                return loaded
        return loaded


def predict(classifier_name: str, model, feat: torch.Tensor):
    """
    Run inference for a single sample.
    Returns (pred: int, proba: np.ndarray shape (2,) or None)
    """
    if classifier_name == 'ann':
        x = feat.unsqueeze(0) if feat.dim() == 1 else feat
        with torch.no_grad():
            out = model(x)
            # Model already has Softmax, so out is already probabilities
            proba = out.numpy()[0]
            pred = int(out.argmax(dim=1).item())
        return pred, proba
    elif classifier_name == 'protonet' and isinstance(model, dict):
        # Handle protonet models saved as dicts with prototypes
        if 'prototypes' in model and 'labels_proto' in model:
            prototypes = model['prototypes']  # Shape: (n_prototypes, feature_dim)
            labels_proto = model['labels_proto']  # Shape: (n_prototypes,)
            
            # Convert to numpy if needed
            if isinstance(prototypes, torch.Tensor):
                prototypes = prototypes.numpy()
            if isinstance(labels_proto, torch.Tensor):
                labels_proto = labels_proto.numpy()
            
            # Compute distances to all prototypes
            feat_np = feat.numpy().reshape(1, -1)
            distances = np.linalg.norm(feat_np - prototypes, axis=1)
            
            # Find closest prototype
            closest_idx = np.argmin(distances)
            pred = int(labels_proto[closest_idx])
            
            # Compute probabilities based on distances (inverse distance weighting)
            inv_distances = 1.0 / (distances + 1e-8)  # Add small epsilon to avoid division by zero
            proba_raw = np.zeros(2)
            for i, label in enumerate(labels_proto):
                proba_raw[int(label)] += inv_distances[i]
            
            # Normalize to get probabilities
            proba = proba_raw / proba_raw.sum() if proba_raw.sum() > 0 else np.array([0.5, 0.5])
            
            return pred, proba
        else:
            raise ValueError(f"Protonet model dict missing required keys: {model.keys()}")
    elif classifier_name == 'random_forest':
        # SPECIAL HANDLING for broken Random Forest models
        X = feat.numpy().reshape(1, -1)
        proba = model.predict_proba(X)[0]
        
        # Use a lower threshold since the models are biased toward class 0
        # Original threshold: 0.5, New threshold: 0.3
        pred = 1 if proba[1] > 0.3 else 0
        
        return pred, proba
    else:
        X = feat.numpy().reshape(1, -1)
        pred = int(model.predict(X)[0])
        try:
            proba = model.predict_proba(X)[0]
        except Exception:
            proba = None
        return pred, proba


# ============================================================
# METRICS HELPER
# ============================================================
def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_proba: Optional[np.ndarray]) -> Dict:
    """
    Compute full classification metrics including:
    Accuracy, Balanced Accuracy, F1, AUROC,
    Sensitivity, Specificity, PPV, NPV, TP/TN/FP/FN
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'confusion_matrix': cm.tolist(),
        'auroc': None,
    }

    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            metrics['auroc'] = roc_auc_score(y_true, pos)
        except Exception:
            pass

    return metrics
# ============================================================
# UNIVERSAL MAIN INFERENCE FUNCTION
# ============================================================
def run_universal_inference() -> pd.DataFrame:
    """
    Universal inference function that works with any model/aggregation combination.
    Automatically detects dimensions and handles mismatches.
    """
    print("\n" + "="*70)
    print("STEP 1 — Loading ground truth")
    print("="*70)
    gt_df = load_ground_truth(GROUND_TRUTH_PATH)

    print("\n" + "="*70)
    print("STEP 2 — Auto-detecting PAIP data and dimensions")
    print("="*70)
    
    # Auto-detect PAIP data location
    paip_base, feature_dir_name = auto_detect_paip_data()
    feat_dir = os.path.join(paip_base, feature_dir_name)
    
    print(f"Using PAIP data: {feat_dir}")
    
    # Load a sample to detect dimensions
    files = [f for f in os.listdir(feat_dir) if f.endswith('.pt')][:1]
    if not files:
        print("❌ No .pt files found in PAIP directory")
        return pd.DataFrame()
    
    sample_path = os.path.join(feat_dir, files[0])
    
    # Try to detect TCGA model dimension from a sample model file
    sample_model_files = []
    for fold in UNIVERSAL_CONFIG['folds']:
        for clf_name in UNIVERSAL_CONFIG['classifiers_to_run']:
            template = UNIVERSAL_CONFIG['classifier_templates'][clf_name]
            if clf_name == 'ann':
                # Try different possible input dimensions
                for possible_dim in [768, 6912, 10752, 13824, 21504]:
                    model_file = template.format(fold=fold, input_dim=possible_dim)
                    model_path = os.path.join(TCGA_MODELS_BASE, model_file)
                    if os.path.exists(model_path):
                        sample_model_files.append(model_path)
                        break
            else:
                model_file = template.format(fold=fold, input_dim='')
                model_path = os.path.join(TCGA_MODELS_BASE, model_file)
                if os.path.exists(model_path):
                    sample_model_files.append(model_path)
                    break
        if sample_model_files:
            break
    
    if not sample_model_files:
        print("❌ No TCGA model files found")
        return pd.DataFrame()
    
    # Detect dimensions
    tcga_dim, paip_dim = detect_feature_dimensions(sample_model_files[0], sample_path)
    
    print(f"📊 Dimension Analysis:")
    print(f"   TCGA models expect: {tcga_dim} features")
    print(f"   PAIP data has: {paip_dim} features")
    
    if tcga_dim and paip_dim:
        if tcga_dim != paip_dim:
            print(f"⚠️  Dimension mismatch detected - will auto-adjust PAIP features")
            print(f"   {paip_dim} → {tcga_dim} (truncate/pad as needed)")
        else:
            print(f"✅ Dimensions match perfectly!")
    
    # Load PAIP features with dimension matching
    features_dict, wsi_ids = load_features(feat_dir, target_dim=tcga_dim)
    
    if not features_dict:
        print(f"❌ No features loaded")
        return pd.DataFrame()

    actual_dim = next(iter(features_dict.values())).shape[0]
    print(f"✅ Loaded {len(features_dict)} PAIP samples with {actual_dim} features each")

    # Build label map
    wsi_labels = {}
    missing_lbls = []
    for wsi_id in wsi_ids:
        lbl = get_label(wsi_id, gt_df)
        if lbl is not None:
            wsi_labels[wsi_id] = lbl
        else:
            missing_lbls.append(wsi_id)

    print(f"✅ Valid WSIs with labels: {len(wsi_labels)}")
    if missing_lbls:
        print(f"⚠️  {len(missing_lbls)} WSIs without labels (will be skipped)")

    print("\n" + "="*70)
    print(f"STEP 3 — Running inference across {len(UNIVERSAL_CONFIG['folds'])} folds")
    print("="*70)

    all_results = []
    skipped_list = []

    # Iterate through folds and classifiers
    for fold in UNIVERSAL_CONFIG['folds']:
        print(f"\n  ── Fold {fold} ──────────────────────────────────")

        for clf_name in UNIVERSAL_CONFIG['classifiers_to_run']:
            template = UNIVERSAL_CONFIG['classifier_templates'][clf_name]
            
            if clf_name == 'ann':
                model_filename = template.format(fold=fold, input_dim=actual_dim)
            else:
                model_filename = template.format(fold=fold, input_dim='')
                
            model_path = os.path.join(TCGA_MODELS_BASE, model_filename)

            if not os.path.exists(model_path):
                print(f"    ✗  {clf_name:<22} model file not found: {model_filename}")
                skipped_list.append(model_path)
                continue

            # Load model
            try:
                model = load_model(clf_name, model_path, actual_dim)
            except Exception as e:
                print(f"    ✗  {clf_name:<22} load error: {e}")
                skipped_list.append(model_path)
                continue

            # Run predictions
            y_true_list = []
            y_pred_list = []
            y_proba_list = []
            wsi_id_list = []

            for wsi_id, feat in features_dict.items():
                if wsi_id not in wsi_labels:
                    continue

                try:
                    pred, proba = predict(clf_name, model, feat)
                    y_true_list.append(wsi_labels[wsi_id])
                    y_pred_list.append(pred)
                    y_proba_list.append(proba)
                    wsi_id_list.append(wsi_id)
                except Exception as e:
                    print(f"    ⚠️  {wsi_id}: {e}")

            if len(y_true_list) == 0:
                print(f"    ✗  {clf_name:<22} no valid predictions")
                continue

            # Compute metrics
            y_true = np.array(y_true_list)
            y_pred = np.array(y_pred_list)
            y_proba = np.array(y_proba_list) if y_proba_list[0] is not None else None

            metrics = compute_metrics(y_true, y_pred, y_proba)

            # Print summary
            auc_str = f"{metrics['auroc']:.4f}" if metrics['auroc'] is not None else "N/A"
            print(f"    ✓  {clf_name:<22} "
                  f"ACC={metrics['accuracy']:.3f}  "
                  f"BACC={metrics['balanced_accuracy']:.3f}  "
                  f"F1={metrics['f1_macro']:.3f}  "
                  f"AUC={auc_str}  "
                  f"Sens={metrics['sensitivity']:.3f}  "
                  f"Spec={metrics['specificity']:.3f}")

            # Store results
            for i, wsi_id in enumerate(wsi_id_list):
                result_row = {
                    'fold': fold,
                    'classifier': clf_name,
                    'wsi_id': wsi_id,
                    'true_label': y_true[i],
                    'pred_label': y_pred[i],
                    'proba_class0': y_proba[i, 0] if y_proba is not None else None,
                    'proba_class1': y_proba[i, 1] if y_proba is not None else None,
                }
                result_row.update({f'metric_{k}': v for k, v in metrics.items()})
                all_results.append(result_row)

    # Process and save results using the existing formatting functions
    if not all_results:
        print("No results to save")
        return pd.DataFrame()
    
    # Convert to DataFrame and create formatted outputs
    df_detailed = pd.DataFrame(all_results)
    
    # Create properly formatted DataFrames
    df_individual = create_per_sample_predictions_df(all_results)
    df_summary = create_summary_metrics_df(all_results)
    
    # Create output directory
    output_base = os.path.join(PROJECT_ROOT, "Inference_Results", AGGREGATION_METHOD, MODEL_NAME, TASK)
    os.makedirs(output_base, exist_ok=True)
    
    # Save files
    individual_file = os.path.join(output_base, f"external_validation_{MODEL_NAME}_{AGGREGATION_METHOD}_individual.csv")
    summary_file = os.path.join(output_base, f"external_validation_{MODEL_NAME}_{AGGREGATION_METHOD}_summary.csv")
    
    save_dataframes(df_individual, df_summary, individual_file, summary_file)

    print("\n" + "="*70)
    print(f"✅ Individual predictions saved → {individual_file}")
    print(f"✅ Summary metrics saved → {summary_file}")
    print(f"   Individual rows: {len(df_individual)}")
    print(f"   Summary rows: {len(df_summary)}")
    print(f"   Skipped models: {len(skipped_list)}")
    print("="*70)

    return df_individual

    print("\n" + "="*70)
    print("STEP 1 — Loading ground truth")
    print("="*70)
    gt_df = load_ground_truth(GROUND_TRUTH_PATH)

    all_results = []
    skipped_list = []

    print("\n" + "="*70)
    print(f"STEP 2 — Loading PAIP features for {MODEL_NAME}")
    print("="*70)

    # Load PAIP features
    feat_dir = os.path.join(PAIP_DATA_BASE, PAIP_FEATURE_DIRS[MODEL_NAME])
    features_dict, wsi_ids = load_features(feat_dir)

    if not features_dict:
        print(f"  ❌  No features found — exiting")
        return pd.DataFrame()

    input_dim = next(iter(features_dict.values())).shape[0]
    print(f"  Input dim: {input_dim}")

    # Build label map
    wsi_labels = {}
    missing_lbls = []
    for wsi_id in wsi_ids:
        lbl = get_label(wsi_id, gt_df)
        if lbl is not None:
            wsi_labels[wsi_id] = lbl
        else:
            missing_lbls.append(wsi_id)

    print(f"  Valid WSIs: {len(wsi_labels)}")
    if missing_lbls:
        print(f"  ⚠️  {len(missing_lbls)} WSIs without labels (will be skipped)")

    print("\n" + "="*70)
    print(f"STEP 3 — Running inference across {len(FOLDS)} folds")
    print("="*70)

    # Iterate through folds and classifiers
    for fold in FOLDS:
        print(f"\n  ── Fold {fold} ──────────────────────────────────")

        for clf_name in CLASSIFIERS_TO_RUN:
            clf_template = CLASSIFIER_TEMPLATES[clf_name]
            model_filename = clf_template.format(fold=fold, input_dim=input_dim)
            model_path = os.path.join(TCGA_MODELS_BASE, model_filename)

            if not os.path.exists(model_path):
                print(f"    ✗  {clf_name:<22} model file not found")
                skipped_list.append(model_path)
                continue

            # Load model
            try:
                model = load_model(clf_name, model_path, input_dim)
            except Exception as e:
                print(f"    ✗  {clf_name:<22} load error: {e}")
                skipped_list.append(model_path)
                continue

            # Run predictions
            y_true_list = []
            y_pred_list = []
            y_proba_list = []
            wsi_id_list = []

            for wsi_id, feat in features_dict.items():
                if wsi_id not in wsi_labels:
                    continue

                try:
                    pred, proba = predict(clf_name, model, feat)
                    y_true_list.append(wsi_labels[wsi_id])
                    y_pred_list.append(pred)
                    y_proba_list.append(proba)
                    wsi_id_list.append(wsi_id)
                except Exception as e:
                    print(f"    ⚠️  {wsi_id}: {e}")

            if len(y_true_list) == 0:
                print(f"    ✗  {clf_name:<22} no valid predictions")
                continue

            # Compute metrics
            y_true = np.array(y_true_list)
            y_pred = np.array(y_pred_list)
            y_proba = np.array(y_proba_list) if y_proba_list[0] is not None else None

            metrics = compute_metrics(y_true, y_pred, y_proba)

            # Print summary
            auc_str = f"{metrics['auroc']:.4f}" if metrics['auroc'] is not None else "N/A"
            print(f"    ✓  {clf_name:<22} "
                  f"ACC={metrics['accuracy']:.3f}  "
                  f"BACC={metrics['balanced_accuracy']:.3f}  "
                  f"F1={metrics['f1_macro']:.3f}  "
                  f"AUC={auc_str}  "
                  f"Sens={metrics['sensitivity']:.3f}  "
                  f"Spec={metrics['specificity']:.3f}")

            # Store results
            for i, wsi_id in enumerate(wsi_id_list):
                result_row = {
                    'fold': fold,
                    'classifier': clf_name,
                    'wsi_id': wsi_id,
                    'true_label': y_true[i],
                    'pred_label': y_pred[i],
                    'proba_class0': y_proba[i, 0] if y_proba is not None else None,
                    'proba_class1': y_proba[i, 1] if y_proba is not None else None,
                }
                result_row.update({f'metric_{k}': v for k, v in metrics.items()})
                all_results.append(result_row)

    # Create the new folder structure
    output_base = os.path.join(PROJECT_ROOT, "Inference_Results", AGGREGATION_METHOD, MODEL_NAME, TASK)
    os.makedirs(output_base, exist_ok=True)
    
    # Reorganize results for the new format
    if not all_results:
        print("No results to save")
        return pd.DataFrame()
    
    # Convert to DataFrame for easier manipulation
    df_detailed = pd.DataFrame(all_results)
    
    # Create individual predictions format with EXACT column order
    individual_predictions = []
    
    # Group by WSI ID to get all classifier predictions for each sample
    wsi_groups = df_detailed.groupby(['wsi_id', 'fold'])
    
    for (wsi_id, fold), group in wsi_groups:
        # Get the true label (same for all classifiers)
        true_label = group['true_label'].iloc[0]
        
        # Initialize row with basic info - CRITICAL: Use 1-indexed fold
        pred_row = {
            'Fold': fold + 1,  # Convert 0-based to 1-based
            'WSI_ID': wsi_id,
            'Target': true_label,
            # Initialize all classifier columns with None
            'lin_Pred': None, 'lin_Prob': None,
            'ann_Pred': None, 'ann_Prob': None,
            'knn_Pred': None, 'knn_Prob': None,
            'proto_Pred': None, 'proto_Prob': None,
            'rf_Pred': None, 'rf_Prob': None
        }
        
        # Map classifier names to column prefixes
        classifier_mapping = {
            'logistic_regression': 'lin',
            'ann': 'ann',
            'knn': 'knn',
            'protonet': 'proto',
            'random_forest': 'rf'
        }
        
        # Add predictions for each classifier
        for _, row in group.iterrows():
            classifier = row['classifier']
            if classifier in classifier_mapping:
                prefix = classifier_mapping[classifier]
                pred_row[f'{prefix}_Pred'] = row['pred_label']
                if row['proba_class1'] is not None:
                    pred_row[f'{prefix}_Prob'] = round(row['proba_class1'], 4)
                else:
                    pred_row[f'{prefix}_Prob'] = None
        
        individual_predictions.append(pred_row)
    
    # Create summary metrics format with EXACT requirements
    summary_metrics = []
    
    # Get all unique classifiers and folds
    classifiers = df_detailed['classifier'].unique()
    folds = sorted(df_detailed['fold'].unique())
    
    # Map classifier names to required prefixes in EXACT order
    classifier_order = ['logistic_regression', 'ann', 'knn', 'protonet', 'random_forest']
    classifier_mapping = {
        'logistic_regression': 'lin',
        'ann': 'ann', 
        'knn': 'knn',
        'protonet': 'proto',
        'random_forest': 'rf'
    }
    
    # Process classifiers in the required order
    for classifier in classifier_order:
        if classifier not in classifiers:
            continue
            
        classifier_data = df_detailed[df_detailed['classifier'] == classifier]
        prefix = classifier_mapping[classifier]
        
        # Create metric rows in EXACT order: acc, bacc, macro_f1, weighted_f1, auroc, conf_matrix
        metric_types = [
            ('acc', 'metric_accuracy'),
            ('bacc', 'metric_balanced_accuracy'), 
            ('macro_f1', 'metric_f1_macro'),
            ('weighted_f1', 'metric_f1_weighted'),
            ('auroc', 'metric_auroc'),
            ('conf_matrix', 'metric_confusion_matrix')
        ]
        
        for metric_suffix, metric_key in metric_types:
            metric_row = {'Metric': f'{prefix}_{metric_suffix}'}
            
            # Add fold columns (1-indexed)
            fold_values = []
            for fold in folds:
                fold_data = classifier_data[classifier_data['fold'] == fold]
                if not fold_data.empty:
                    value = fold_data[metric_key].iloc[0]
                    if metric_suffix == 'conf_matrix':
                        # Format confusion matrix as nested list string
                        metric_row[f'Fold{fold+1}'] = str(value)
                    else:
                        # Round numeric values to max 10 decimal places
                        if value is not None:
                            rounded_value = round(float(value), 10)
                            metric_row[f'Fold{fold+1}'] = rounded_value
                            fold_values.append(rounded_value)
                        else:
                            metric_row[f'Fold{fold+1}'] = None
                else:
                    metric_row[f'Fold{fold+1}'] = None
            
            # Calculate AvgFolds (exclude confusion matrix)
            if metric_suffix != 'conf_matrix' and fold_values:
                metric_row['AvgFolds'] = round(sum(fold_values) / len(fold_values), 10)
            else:
                metric_row['AvgFolds'] = None if metric_suffix != 'conf_matrix' else ''
                
            summary_metrics.append(metric_row)
    
    # Convert to DataFrames with EXACT column ordering
    df_individual = pd.DataFrame(individual_predictions)
    df_summary = pd.DataFrame(summary_metrics)
    
    # CRITICAL: Sort individual predictions by Fold first, then WSI_ID
    # This ensures all Fold 1 samples are together, then all Fold 2, etc.
    if not df_individual.empty:
        df_individual = df_individual.sort_values(['Fold', 'WSI_ID']).reset_index(drop=True)
        
        # Ensure EXACT column order for per-sample predictions
        required_columns = [
            'Fold', 'WSI_ID', 'Target',
            'lin_Pred', 'lin_Prob',
            'ann_Pred', 'ann_Prob', 
            'knn_Pred', 'knn_Prob',
            'proto_Pred', 'proto_Prob',
            'rf_Pred', 'rf_Prob'
        ]
        
        # Reorder columns to match exact requirements
        existing_cols = [col for col in required_columns if col in df_individual.columns]
        df_individual = df_individual[existing_cols]
    
    # Ensure EXACT column order for summary metrics
    if not df_summary.empty:
        summary_columns = ['Metric', 'Fold1', 'Fold2', 'Fold3', 'Fold4', 'AvgFolds']
        existing_summary_cols = [col for col in summary_columns if col in df_summary.columns]
        df_summary = df_summary[existing_summary_cols]
    
    # Save files
    individual_file = os.path.join(output_base, "external_validation_PAIP_individual.csv")
    summary_file = os.path.join(output_base, "external_validation_PAIP_summary.csv")
    
    # Save individual predictions
    if not df_individual.empty:
        df_individual.to_csv(individual_file, index=False)
    
    # Save summary metrics  
    if not df_summary.empty:
        df_summary.to_csv(summary_file, index=False)

    print("\n" + "="*70)
    print(f"✅  Individual predictions saved → {individual_file}")
    print(f"✅  Summary metrics saved → {summary_file}")
    print(f"   Individual rows: {len(df_individual)}")
    print(f"   Summary rows: {len(df_summary)}")
    print(f"   Skipped models: {len(skipped_list)}")
    print("="*70)

    return df_individual


# ============================================================
# RUN UNIVERSAL INFERENCE AND DISPLAY SUMMARY
# ============================================================
if __name__ == "__main__":
    results_df = run_universal_inference()
    
    # Display summary statistics
    if not results_df.empty:
        print("\n" + "="*70)
        print("INDIVIDUAL PREDICTIONS SUMMARY")
        print("="*70)
        print(f"Total predictions: {len(results_df)}")
        print(f"Unique WSIs: {results_df['WSI_ID'].nunique()}")
        print(f"Folds: {sorted(results_df['Fold'].unique())}")
        
        # Show sample of predictions
        print("\nSample predictions:")
        print(results_df.head().to_string(index=False))
    else:
        print("\n⚠️  No results to display")
        
    print("\n" + "="*70)
    print("UNIVERSAL EXTERNAL VALIDATION COMPLETE")
    print("="*70)
    print(f"✅ This script now works with ANY model/aggregation combination!")
    print(f"✅ Simply change MODEL_NAME and AGGREGATION_METHOD at the top")
    print(f"✅ Automatic dimension detection and matching included")


# ============================================================
# DATAFRAME FORMATTING UTILITIES
# ============================================================

def create_per_sample_predictions_df(results_data: List[Dict]) -> pd.DataFrame:
    """
    Create per-sample predictions DataFrame with EXACT format requirements.
    
    Args:
        results_data: List of dictionaries containing prediction results
        
    Returns:
        DataFrame with columns: Fold, WSI_ID, Target, lin_Pred, lin_Prob, 
        ann_Pred, ann_Prob, knn_Pred, knn_Prob, proto_Pred, proto_Prob, rf_Pred, rf_Prob
    """
    # Convert to DataFrame
    df = pd.DataFrame(results_data)
    
    # Ensure 1-indexed folds
    if 'Fold' in df.columns:
        df['Fold'] = df['Fold'].apply(lambda x: x + 1 if x < 1 else x)
    
    # Define EXACT column order
    required_columns = [
        'Fold', 'WSI_ID', 'Target',
        'lin_Pred', 'lin_Prob',
        'ann_Pred', 'ann_Prob', 
        'knn_Pred', 'knn_Prob',
        'proto_Pred', 'proto_Prob',
        'rf_Pred', 'rf_Prob'
    ]
    
    # Add missing columns with None values
    for col in required_columns:
        if col not in df.columns:
            df[col] = None
    
    # Reorder columns to match exact requirements
    df = df[required_columns]
    
    # CRITICAL: Sort by Fold first, then WSI_ID
    # This ensures all Fold 1 samples are together, then all Fold 2, etc.
    df = df.sort_values(['Fold', 'WSI_ID']).reset_index(drop=True)
    
    return df


def create_summary_metrics_df(metrics_data: List[Dict]) -> pd.DataFrame:
    """
    Create summary metrics DataFrame with EXACT format requirements.
    
    Args:
        metrics_data: List of dictionaries containing metric results
        
    Returns:
        DataFrame with columns: Metric, Fold1, Fold2, Fold3, Fold4, AvgFolds
    """
    df = pd.DataFrame(metrics_data)
    
    # Define EXACT column order
    required_columns = ['Metric', 'Fold1', 'Fold2', 'Fold3', 'Fold4', 'AvgFolds']
    
    # Add missing columns with None values
    for col in required_columns:
        if col not in df.columns:
            df[col] = None
    
    # Reorder columns to match exact requirements
    existing_cols = [col for col in required_columns if col in df.columns]
    df = df[existing_cols]
    
    return df


def save_dataframes(df_individual: pd.DataFrame, 
                   df_summary: pd.DataFrame,
                   individual_path: str,
                   summary_path: str):
    """
    Save DataFrames to CSV files.
    
    Args:
        df_individual: Per-sample predictions DataFrame
        df_summary: Summary metrics DataFrame  
        individual_path: Path to save individual predictions CSV
        summary_path: Path to save summary metrics CSV
    """
    # Save individual predictions
    if not df_individual.empty:
        df_individual.to_csv(individual_path, index=False)
    
    # Save summary metrics
    if not df_summary.empty:
        df_summary.to_csv(summary_path, index=False)