import logging
from typing import Any, List, Tuple, Dict

import numpy as np
import pandas as pd
import sklearn.neighbors
import torch
from torch.nn.functional import normalize
from torch.utils.data import Sampler
from tqdm import tqdm
from .metrics import get_eval_metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import joblib, os

def eval_knn(
    fold: int,
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    val_feats: torch.Tensor,
    val_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    n_neighbors: int = 3,
    normalize_feats: bool = True,
    prefix: str = "knn_",
    model_save_path: str=None,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    # Combine train and validation data
    if val_feats is not None:
        train_feats = torch.cat((train_feats, val_feats), dim=0)
        train_labels = torch.cat((train_labels, val_labels), dim=0)
    # Normalize features
    if normalize_feats:
        train_feats = normalize(train_feats, dim=-1, p=2)
        test_feats = normalize(test_feats, dim=-1, p=2)

    # Convert tensors to numpy for sklearn
    train_feats_np = train_feats.numpy()
    test_feats_np = test_feats.numpy()
    train_labels_np = train_labels.numpy()
    test_labels_np = test_labels.numpy()

    # Train KNN classifier
    param_grid = {'n_neighbors': [3, 5, 7, 10, 15], 'metric': ['cosine', 'euclidean','minkowski'], 'weights': ['uniform', 'distance']}
    knn = GridSearchCV(KNeighborsClassifier(), param_grid, n_jobs=-1, verbose=0, scoring='balanced_accuracy')
    knn.fit(train_feats_np, train_labels_np)
    if verbose:
        print(f"Best Params: {knn.best_params_}")
        print(f"Best AUROC score: {knn.best_score_}")
    knn = knn.best_estimator_
    if model_save_path is not None:
        # Save model
        model_path = os.path.join(model_save_path, f"fold{fold}_knn_model.pkl")
        joblib.dump(knn, model_path)
    # Predict and evaluate
    predicted_labels = knn.predict(test_feats_np)
    probs_all = knn.predict_proba(test_feats_np)
    metrics = get_eval_metrics(test_labels_np, predicted_labels,probs_all, prefix=prefix)
    dump = {
        "preds_all": predicted_labels,
        "probs_all": probs_all,
        "targets_all": test_labels_np
    }
    return metrics, dump

def test_saved_knn_model(test_feats: torch.Tensor, test_labels: torch.Tensor, model_path="knn_model.pkl"):
    # Load trained KNN model
    knn = joblib.load(model_path)

    # Convert test features to numpy
    test_feats_np = test_feats.numpy()
    test_labels_np = test_labels.numpy()

    # Get predictions
    predicted_labels = knn.predict(test_feats_np)
    probs_all = knn.predict_proba(test_feats_np)    # Probabilities for each class

    # Compute evaluation metrics
    eval_metrics = get_eval_metrics(test_labels_np, predicted_labels,probs_all,True, prefix="knn_")
    dump = {"preds_all": predicted_labels, "probs_all": probs_all, "targets_all": test_labels_np}
    return eval_metrics, dump
