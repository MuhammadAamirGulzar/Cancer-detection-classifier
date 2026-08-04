#!/usr/bin/env python3
"""
Simple Random Forest Validation Test
====================================

This script tests the pretrained Random Forest models to diagnose issues
and validate their performance on PAIP data.
"""

import os
import torch
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from typing import Dict, List, Tuple, Optional

# Configuration
MODEL_DIR = r"TCGA_Results_Updated\Caption_based_aggregation\Conch1_5\1-MSIH\models"
PAIP_FEATURE_DIR = r"D:\Aamir Gulzar\KSA_project2\paip_data\slide_aggregation\Caption_Based_Clustering_FiveCrop\conch_CC_fivecrop"
GROUND_TRUTH_PATH = r"D:\Aamir Gulzar\KSA_project2\paip_data\labels\paip_kfolds_71.csv"

def parse_label(raw) -> Optional[int]:
    """Convert label to binary (0=nonMSIH, 1=MSIH)"""
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

def load_ground_truth() -> pd.DataFrame:
    """Load and process ground truth labels"""
    df = pd.read_csv(GROUND_TRUTH_PATH)
    df.columns = [c.strip() for c in df.columns]
    
    # Get WSI ID and label columns
    id_col = 'WSI_Id' if 'WSI_Id' in df.columns else df.columns[0]
    lbl_col = 'label' if 'label' in df.columns else df.columns[1]
    
    df['_wsi_id'] = df[id_col].astype(str).str.strip()
    df['_bin_label'] = df[lbl_col].apply(parse_label)
    
    return df

def get_label(wsi_id: str, gt_df: pd.DataFrame) -> Optional[int]:
    """Get label for WSI ID with fallback strategies"""
    # Exact match
    row = gt_df[gt_df['_wsi_id'] == wsi_id]
    if not row.empty:
        return row['_bin_label'].values[0]
    
    # Substring match
    for _, r in gt_df.iterrows():
        if r['_wsi_id'] in wsi_id or wsi_id in r['_wsi_id']:
            return r['_bin_label']
    
    # Parse from filename
    lower = wsi_id.lower()
    if '_nonmsih' in lower:
        return 0
    if '_msih' in lower:
        return 1
    
    return None

def load_paip_features() -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """Load PAIP feature files"""
    features_dict = {}
    wsi_ids = []
    
    if not os.path.isdir(PAIP_FEATURE_DIR):
        print(f"❌ Directory not found: {PAIP_FEATURE_DIR}")
        return features_dict, wsi_ids
    
    files = [f for f in os.listdir(PAIP_FEATURE_DIR) if f.endswith('.pt')]
    print(f"Found {len(files)} .pt files")
    
    for fname in files:
        wsi_id = os.path.splitext(fname)[0]
        try:
            feat = torch.load(os.path.join(PAIP_FEATURE_DIR, fname), map_location='cpu')
            if feat.dim() > 1:
                feat = feat.flatten()
            features_dict[wsi_id] = feat.float()
            wsi_ids.append(wsi_id)
        except Exception as e:
            print(f"⚠️ Could not load {fname}: {e}")
    
    return features_dict, wsi_ids

def test_random_forest_model(fold: int, features_dict: Dict[str, torch.Tensor], 
                           wsi_labels: Dict[str, int], use_custom_threshold: bool = False) -> Dict:
    """Test a single Random Forest model"""
    
    model_path = os.path.join(MODEL_DIR, f"fold{fold}_random_forest.pkl")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return {}
    
    # Load model
    try:
        model = joblib.load(model_path)
        print(f"✅ Loaded Random Forest model for fold {fold}")
        print(f"   Model type: {type(model)}")
        print(f"   N_estimators: {model.n_estimators}")
        print(f"   Classes: {model.classes_}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return {}
    
    # Prepare data
    X_list = []
    y_true_list = []
    wsi_id_list = []
    
    for wsi_id, feat in features_dict.items():
        if wsi_id in wsi_labels:
            X_list.append(feat.numpy())
            y_true_list.append(wsi_labels[wsi_id])
            wsi_id_list.append(wsi_id)
    
    if len(X_list) == 0:
        print("❌ No valid samples found")
        return {}
    
    X = np.array(X_list)
    y_true = np.array(y_true_list)
    
    print(f"   Testing on {len(X)} samples")
    print(f"   True label distribution: {np.bincount(y_true)}")
    
    # Make predictions
    try:
        y_proba = model.predict_proba(X)
        
        if use_custom_threshold:
            # Use custom threshold to handle bias
            threshold = 0.3
            y_pred = (y_proba[:, 1] > threshold).astype(int)
            print(f"   Using custom threshold: {threshold}")
        else:
            # Use default prediction
            y_pred = model.predict(X)
            print(f"   Using default prediction")
        
        print(f"   Predicted label distribution: {np.bincount(y_pred)}")
        
    except Exception as e:
        print(f"❌ Error making predictions: {e}")
        return {}
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Confusion Matrix:")
    print(f"   {cm}")
    
    # Detailed analysis
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    print(f"   Sensitivity (Recall): {sensitivity:.4f}")
    print(f"   Specificity: {specificity:.4f}")
    
    # Check probability distribution
    print(f"   Probability stats for class 1:")
    print(f"   Min: {y_proba[:, 1].min():.4f}, Max: {y_proba[:, 1].max():.4f}")
    print(f"   Mean: {y_proba[:, 1].mean():.4f}, Std: {y_proba[:, 1].std():.4f}")
    
    # Show some individual predictions
    print(f"   Sample predictions:")
    for i in range(min(5, len(wsi_id_list))):
        wsi_id = wsi_id_list[i]
        true_lbl = y_true[i]
        pred_lbl = y_pred[i]
        prob_1 = y_proba[i, 1]
        print(f"   {wsi_id[:25]:<25} | True: {true_lbl} | Pred: {pred_lbl} | Prob_1: {prob_1:.4f}")
    
    return {
        'fold': fold,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm.tolist(),
        'n_samples': len(X),
        'prob_stats': {
            'min': float(y_proba[:, 1].min()),
            'max': float(y_proba[:, 1].max()),
            'mean': float(y_proba[:, 1].mean()),
            'std': float(y_proba[:, 1].std())
        }
    }

def main():
    """Main function to test Random Forest models"""
    
    print("="*60)
    print("RANDOM FOREST VALIDATION TEST")
    print("="*60)
    
    # Load ground truth
    print("\n1. Loading ground truth...")
    gt_df = load_ground_truth()
    print(f"   Loaded {len(gt_df)} ground truth labels")
    
    # Load PAIP features
    print("\n2. Loading PAIP features...")
    features_dict, wsi_ids = load_paip_features()
    print(f"   Loaded {len(features_dict)} feature files")
    
    # Build label mapping
    print("\n3. Building label mapping...")
    wsi_labels = {}
    missing_labels = []
    
    for wsi_id in wsi_ids:
        label = get_label(wsi_id, gt_df)
        if label is not None:
            wsi_labels[wsi_id] = label
        else:
            missing_labels.append(wsi_id)
    
    print(f"   Valid labels: {len(wsi_labels)}")
    print(f"   Missing labels: {len(missing_labels)}")
    
    if len(wsi_labels) == 0:
        print("❌ No valid labels found - exiting")
        return
    
    # Test each fold
    print("\n4. Testing Random Forest models...")
    results = []
    
    for fold in [0, 1, 2, 3]:
        print(f"\n--- Testing Fold {fold} ---")
        
        # Test with default threshold
        print("Default threshold:")
        result_default = test_random_forest_model(fold, features_dict, wsi_labels, use_custom_threshold=False)
        if result_default:
            result_default['threshold_type'] = 'default'
            results.append(result_default)
        
        print("\nCustom threshold (0.3):")
        result_custom = test_random_forest_model(fold, features_dict, wsi_labels, use_custom_threshold=True)
        if result_custom:
            result_custom['threshold_type'] = 'custom_0.3'
            results.append(result_custom)
        
        print("-" * 40)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if results:
        df_results = pd.DataFrame(results)
        
        print("\nResults by threshold type:")
        for threshold_type in ['default', 'custom_0.3']:
            subset = df_results[df_results['threshold_type'] == threshold_type]
            if not subset.empty:
                print(f"\n{threshold_type.upper()} THRESHOLD:")
                print(f"  Average Accuracy: {subset['accuracy'].mean():.4f}")
                print(f"  Average Sensitivity: {subset['sensitivity'].mean():.4f}")
                print(f"  Average Specificity: {subset['specificity'].mean():.4f}")
                
                # Check if it ever predicts class 1
                total_tp = subset.apply(lambda row: row['confusion_matrix'][1][1], axis=1).sum()
                total_fn = subset.apply(lambda row: row['confusion_matrix'][1][0], axis=1).sum()
                total_positives = total_tp + total_fn
                
                if total_tp == 0:
                    print(f"  🚨 NEVER predicts class 1 (MSIH)!")
                else:
                    print(f"  ✅ Predicts both classes (TP: {total_tp}/{total_positives})")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()