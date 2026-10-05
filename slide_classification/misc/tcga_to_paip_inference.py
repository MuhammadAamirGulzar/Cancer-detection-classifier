"""
TCGA Models Inference on PAIP Data
This script loads trained models from TCGA dataset (fold3 only) and runs inference on PAIP data
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix
)
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================
TCGA_MODELS_BASE = r"D:\Aamir Gulzar\KSA_project2\Cancer-detection-classifier\slide_classification\TCGA_Result\5-Caption_based_aggregation"
PAIP_DATA_BASE = r"D:\Aamir Gulzar\KSA_project2\paip_data\slide_aggregation\Caption_Based_Clustering_FiveCrop"
GROUND_TRUTH_PATH = r"D:\Aamir Gulzar\KSA_project2\paip_data\labels\paip_78_labels.csv"
OUTPUT_PATH = r"D:\Aamir Gulzar\KSA_project2\Cancer-detection-classifier\slide_classification\Inference_Results"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Model configurations
FEATURE_MODELS = ['Conch1_5', 'H-Optimus-1', 'UNI2']
PAIP_FEATURE_DIRS = {
    'Conch1_5': 'conch_CC_fivecrop',
    'H-Optimus-1': 'hoptimus_CC_fivecrop',
    'UNI2': 'uni2_CC_fivecrop'
}
CLASSIFIER_TYPES = {
    'logistic_regression': 'fold3_logistic_regression.pkl',
    'ann': 'fold3_trained_ann_model_10752.pth',
    'knn': 'fold3_knn_model.pkl',
    'protonet': 'fold3_protonet_model.pkl'
}
TASK = 'MSIH'
FOLD_TO_USE = 3  # Only using fold3


# ============================================
# HELPER FUNCTIONS
# ============================================

def load_paip_features(feature_dir: str) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """Load all PAIP feature files from a directory"""
    features_dict = {}
    wsi_ids = []
    
    print(f"Loading features from: {feature_dir}")
    files = [f for f in os.listdir(feature_dir) if f.endswith('.pt')]
    print(f"Found {len(files)} feature files")
    
    for file in files:
        wsi_id = os.path.splitext(file)[0]
        file_path = os.path.join(feature_dir, file)
        
        try:
            features = torch.load(file_path)
            if features.is_cuda:
                features = features.cpu()
            
            # Flatten if needed
            if features.dim() > 1:
                features = features.flatten()
            
            features_dict[wsi_id] = features
            wsi_ids.append(wsi_id)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return features_dict, wsi_ids


def load_ground_truth(csv_path: str) -> pd.DataFrame:
    """Load ground truth labels from CSV"""
    df = pd.read_csv(csv_path)
    print(f"Loaded ground truth with {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    return df


def get_label_from_wsi_id(wsi_id: str, ground_truth_df: pd.DataFrame) -> int:
    """Extract label from ground truth dataframe"""
    if 'WSI_Id' in ground_truth_df.columns:
        match = ground_truth_df[ground_truth_df['WSI_Id'] == wsi_id]
        if not match.empty:
            if 'label' in ground_truth_df.columns:
                label_val = match['label'].values[0]
            elif 'label_desc' in ground_truth_df.columns:
                label_val = match['label_desc'].values[0]
            else:
                raise ValueError("No label column found")
            
            # Convert to binary
            if isinstance(label_val, str):
                return 0 if label_val.lower() == 'nonmsih' else 1
            else:
                return int(label_val)
    
    return None


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict:
    """Calculate classification metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            if y_proba.ndim > 1:
                y_proba = y_proba[:, 1]
            metrics['auroc'] = roc_auc_score(y_true, y_proba)
        except:
            metrics['auroc'] = None
    
    return metrics


# ============================================
# MODEL LOADING FUNCTIONS
# ============================================
class ANNClassifier(nn.Module):
    """ANN Classifier architecture"""
    def __init__(self, input_dim, hidden_dim=256, hidden_dim2=128, num_classes=2):
        super(ANNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_dim2, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


def load_sklearn_model(model_path: str):
    """Load sklearn-based models"""
    return joblib.load(model_path)


def load_ann_model(model_path: str, input_dim: int, hidden_dim: int = 256, hidden_dim2: int = 128):
    """Load PyTorch ANN model"""
    model = ANNClassifier(input_dim, hidden_dim, hidden_dim2)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


def predict_sklearn(model, features: torch.Tensor):
    """Make predictions with sklearn models"""
    X = features.numpy() if isinstance(features, torch.Tensor) else features
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    predictions = model.predict(X)
    try:
        probabilities = model.predict_proba(X)
    except:
        probabilities = None
    
    return predictions, probabilities


def predict_ann(model, features: torch.Tensor):
    """Make predictions with ANN model"""
    if features.ndim == 1:
        features = features.unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(features)
        probabilities = F.softmax(outputs, dim=1).numpy()
        predictions = outputs.argmax(dim=1).numpy()
    
    return predictions, probabilities



# ============================================
# MAIN INFERENCE FUNCTION
# ============================================
def run_inference():
    """Main function to run inference on PAIP data using TCGA trained models (fold3 only)"""
    
    # Load ground truth
    print("="*80)
    print("Loading Ground Truth Labels")
    print("="*80)
    ground_truth_df = load_ground_truth(GROUND_TRUTH_PATH)
    
    # Results storage
    all_results = []
    
    # Iterate through each feature model (Conch, H-Optimus, UNI)
    for feature_model in FEATURE_MODELS:
        print(f"\n{'='*80}")
        print(f"Processing Feature Model: {feature_model}")
        print(f"{'='*80}")
        
        # Load PAIP features
        paip_feature_dir = os.path.join(PAIP_DATA_BASE, PAIP_FEATURE_DIRS[feature_model])
        features_dict, wsi_ids = load_paip_features(paip_feature_dir)
        
        if not features_dict:
            print(f"No features found for {feature_model}, skipping...")
            continue
        
        # Get input dimension
        sample_feature = next(iter(features_dict.values()))
        input_dim = sample_feature.shape[0]
        print(f"Feature dimension: {input_dim}")
        
        # Iterate through each classifier type
        for classifier_name, model_filename in CLASSIFIER_TYPES.items():
            print(f"\n{'-'*60}")
            print(f"Classifier: {classifier_name}")
            print(f"{'-'*60}")
            
            # Model path for fold3
            model_path = os.path.join(
                TCGA_MODELS_BASE,
                feature_model,
                TASK,
                'models',
                model_filename
            )
            
            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}")
                continue
            
            print(f"Loading model: {model_filename}")
            
            # Load model
            try:
                if classifier_name == 'ann':
                    model = load_ann_model(model_path, input_dim)
                else:
                    model = load_sklearn_model(model_path)
            except Exception as e:
                print(f"Error loading model: {e}")
                continue
            
            # Make predictions on all PAIP samples
            predictions = []
            probabilities = []
            true_labels = []
            valid_wsi_ids = []
            
            for wsi_id in wsi_ids:
                # Get features
                features = features_dict[wsi_id]
                
                # Get true label
                true_label = get_label_from_wsi_id(wsi_id, ground_truth_df)
                if true_label is None:
                    continue
                
                # Make prediction
                try:
                    if classifier_name == 'ann':
                        pred, proba = predict_ann(model, features)
                    else:
                        pred, proba = predict_sklearn(model, features)
                    
                    predictions.append(pred[0] if isinstance(pred, np.ndarray) else pred)
                    probabilities.append(proba[0] if proba is not None else None)
                    true_labels.append(true_label)
                    valid_wsi_ids.append(wsi_id)
                except Exception as e:
                    print(f"Error predicting for {wsi_id}: {e}")
                    continue
            
            print(f"Processed {len(predictions)} samples")
            
            # Calculate metrics
            if predictions:
                probabilities_array = np.array(probabilities) if probabilities[0] is not None else None
                
                metrics = calculate_metrics(
                    np.array(true_labels),
                    np.array(predictions),
                    probabilities_array
                )
                
                print(f"\nResults for {feature_model} - {classifier_name}:")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
                print(f"  F1 Macro: {metrics['f1_macro']:.4f}")
                print(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")
                if metrics.get('auroc'):
                    print(f"  AUROC: {metrics['auroc']:.4f}")
                print(f"  Confusion Matrix: {metrics['confusion_matrix']}")
                
                # Store results
                result_entry = {
                    'feature_model': feature_model,
                    'classifier': classifier_name,
                    'fold': FOLD_TO_USE,
                    'num_samples': len(true_labels),
                    **metrics
                }
                all_results.append(result_entry)
    
    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    output_file = os.path.join(OUTPUT_PATH, f"tcga_to_paip_inference_{TASK}_fold{FOLD_TO_USE}.csv")
    results_df.to_csv(output_file, index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")
    
    return results_df


# ============================================
# RUN INFERENCE
# ============================================
if __name__ == "__main__":
    results = run_inference()
    print("\n" + "="*80)
    print("INFERENCE COMPLETE!")
    print("="*80)
    print(results.to_string())
