# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import confusion_matrix
# import numpy as np
# from typing import Dict, List, Tuple, Any
# from .metrics import get_eval_metrics
# import time
# import torch
# import joblib, os

# def eval_r_forest(
#     fold: int,
#     train_feats: torch.Tensor,
#     train_labels: torch.Tensor,
#     valid_feats: torch.Tensor,
#     valid_labels: torch.Tensor,
#     test_feats: torch.Tensor,
#     test_labels: torch.Tensor,
#     n_estimators: int = 500,
#     max_depth: int = 6 ,#None,
#     min_samples_split: int = 20, #5,
#     min_samples_leaf: int = 10, #1,
#     class_weight: 'balanced',#Any = {0: 1, 1: 10},
#     prediction_threshold: float = 0.3,
#     combine_trainval: bool = False,#True,
#     prefix: str = "rf_",
#     save_path: str = None,
#     verbose: bool = True,
# ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
#     """
#     Evaluate a Random Forest classifier using Scikit-learn.

#     Args:
#         fold: Current fold number.
#         train_feats: Training feature vectors.
#         train_labels: Training labels.
#         valid_feats: Validation feature vectors.
#         valid_labels: Validation labels.
#         test_feats: Test feature vectors.
#         test_labels: Test labels.
#         n_estimators: Number of trees in the forest.
#         max_depth: Maximum depth of the tree.
#         min_samples_split: Minimum samples required to split an internal node.
#         min_samples_leaf: Minimum samples required to be at a leaf node.
#         class_weight: Class weights for handling imbalance (dict or 'balanced').
#         prediction_threshold: Threshold for predicting minority class (default 0.3).
#         combine_trainval: Whether to combine training and validation data for the final model.
#         prefix: Prefix for metric names.
#         save_path: Path to save the trained model.
#         verbose: Whether to print debug information.

#     Returns:
#         results: Dictionary containing evaluation metrics (accuracy, F1, ROC-AUC).
#         dump: Dictionary containing predictions and probabilities.
#     """
#     if verbose:
#         print(f"Train Shape: {train_feats.shape}, Test Shape: {test_feats.shape}")
#         if valid_feats is not None:
#             print(f"Validation Shape: {valid_feats.shape}")
#         print(f"Using class_weight: {class_weight}, threshold: {prediction_threshold}")
    
#     start = time.time()
    
#     # Train Random Forest classifier
#     classifier = train_r_forest(
#         train_feats,
#         train_labels,
#         valid_feats,
#         valid_labels,
#         n_estimators=n_estimators,
#         max_depth=max_depth,
#         min_samples_split=min_samples_split,
#         min_samples_leaf=min_samples_leaf,
#         class_weight=class_weight,
#         combine_trainval=combine_trainval,
#         verbose=verbose,
#     )
    
#     if save_path is not None:
#         model_path = os.path.join(save_path, f"fold{fold}_random_forest.pkl")
#         joblib.dump(classifier, model_path)
    
#     # Test Random Forest classifier with custom threshold
#     results, dump = test_r_forest(
#         classifier, 
#         test_feats, 
#         test_labels, 
#         prediction_threshold=prediction_threshold,
#         prefix=prefix
#     )
    
#     if verbose:
#         print(f"Random Forest Evaluation Time: {time.time() - start:.3f} s")

#     return results, dump


# def train_r_forest(
#     train_feats,
#     train_labels,
#     valid_feats,
#     valid_labels,
#     n_estimators=500,
#     max_depth=None,
#     min_samples_split=5,
#     min_samples_leaf=1,
#     class_weight={0: 1, 1: 10},
#     combine_trainval=True,
#     verbose=True,
# ) -> RandomForestClassifier:
#     """
#     Train a Random Forest classifier using Scikit-learn.

#     Args:
#         train_feats: Feature vectors for training.
#         train_labels: Labels for training.
#         valid_feats: Validation feature vectors.
#         valid_labels: Validation labels.
#         n_estimators: Number of trees in the forest.
#         max_depth: Maximum depth of the tree.
#         min_samples_split: Minimum samples required to split an internal node.
#         min_samples_leaf: Minimum samples required to be at a leaf node.
#         class_weight: Class weights for handling imbalance.
#         combine_trainval: Whether to combine training and validation data.
#         verbose: Whether to print debug information.

#     Returns:
#         A trained Scikit-learn Random Forest model.
#     """
#     # Combine train and validation sets if required
#     if combine_trainval and (valid_feats is not None):
#         train_feats = torch.cat([train_feats, valid_feats], dim=0)
#         train_labels = torch.cat([train_labels, valid_labels], dim=0)
#         if verbose:
#             print(f"Combined Train and Validation Shape: {train_feats.shape}")
    
#     classifier = RandomForestClassifier(
#         n_estimators=n_estimators,
#         max_depth=max_depth,
#         min_samples_split=min_samples_split,
#         min_samples_leaf=min_samples_leaf,
#         max_features="sqrt",
#         class_weight=class_weight,
#         random_state=42,
#         n_jobs=-1,
#         verbose=1 if verbose else 0
#     )
    
#     classifier.fit(train_feats, train_labels)
    
#     if verbose:
#         print(f"Training complete. Number of trees: {classifier.n_estimators}")
    
#     return classifier


# def test_r_forest(
#     classifier: RandomForestClassifier,
#     test_feats: torch.Tensor,
#     test_labels: torch.Tensor,
#     num_classes: int = None,
#     prediction_threshold: float = 0.3,
#     prefix: str = "rf_",
# ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
#     """
#     Evaluate a trained Random Forest classifier on the test set.

#     Args:
#         classifier: A trained Scikit-learn Random Forest model.
#         test_feats: Feature vectors for testing.
#         test_labels: Labels for testing.
#         num_classes: Number of classes (optional).
#         prediction_threshold: Threshold for predicting minority class (class 1).
#         prefix: Prefix for metric names.

#     Returns:
#         results: Dictionary containing evaluation metrics (accuracy, F1, ROC-AUC).
#         dump: Dictionary containing predictions and probabilities.
#     """
#     # Evaluate
#     NUM_C = len(set(test_labels.cpu().numpy())) if num_classes is None else num_classes
    
#     # Get probabilities
#     probs_all = classifier.predict_proba(test_feats)
    
#     # Apply custom threshold for binary classification
#     if NUM_C == 2:
#         # Use custom threshold for minority class (class 1)
#         preds_all = (probs_all[:, 1] >= prediction_threshold).astype(int)
#         roc_kwargs = {}
#     else:
#         # For multiclass, use default prediction
#         preds_all = classifier.predict(test_feats)
#         roc_kwargs = {"multi_class": "ovo", "average": "macro"}

#     targets_all = test_labels.detach().cpu().numpy()
    
#     eval_metrics = get_eval_metrics(targets_all, preds_all, probs_all, True, prefix, roc_kwargs)
#     dump = {"preds_all": preds_all, "probs_all": probs_all, "targets_all": targets_all}

#     return eval_metrics, dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import numpy as np
from typing import Dict, List, Tuple, Any
from .metrics import get_eval_metrics
import time
import torch
import joblib, os


def find_best_threshold(y_true: np.ndarray, y_proba_pos: np.ndarray) -> float:
    """Find threshold that maximises balanced accuracy on validation set."""
    thresholds  = np.arange(0.05, 0.96, 0.01)
    best_thresh = 0.5
    best_score  = -1.0
    for t in thresholds:
        preds = (y_proba_pos >= t).astype(int)
        score = balanced_accuracy_score(y_true, preds)
        if score > best_score:
            best_score  = score
            best_thresh = t
    return float(best_thresh)


def eval_r_forest(
    fold: int,
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    valid_feats: torch.Tensor,
    valid_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    n_estimators: int = 1500,
    max_depth: int = None,
    min_samples_split: int = 10,
    min_samples_leaf: int = 4,
    max_features: str = "sqrt",
    class_weight: Any = "balanced",
    prediction_threshold: float = None,   # None = auto-tune on val set
    combine_trainval: bool = True,
    prefix: str = "rf_",
    save_path: str = None,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    if verbose:
        print(f"--- FOLD {fold} EVALUATION ---")
        print(f"Train Shape: {train_feats.shape}, Test Shape: {test_feats.shape}")
        print(f"Using class_weight: {class_weight}")

    start = time.time()

    classifier, best_thresh = train_r_forest(
        train_feats,
        train_labels,
        valid_feats,
        valid_labels,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        combine_trainval=combine_trainval,
        verbose=verbose,
    )

    # Use auto-tuned threshold unless one is explicitly provided
    threshold = prediction_threshold if prediction_threshold is not None else best_thresh

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        model_path = os.path.join(save_path, f"fold{fold}_random_forest.pkl")
        joblib.dump(classifier, model_path)

    results, dump = test_r_forest(
        classifier,
        test_feats,
        test_labels,
        prediction_threshold=threshold,
        prefix=prefix
    )

    if verbose:
        print(f"Random Forest Evaluation Time: {time.time() - start:.3f} s")

    return results, dump


def train_r_forest(
    train_feats,
    train_labels,
    valid_feats,
    valid_labels,
    n_estimators=1500,
    max_depth=None,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features="sqrt",
    class_weight="balanced",
    combine_trainval=True,
    verbose=True,
) -> Tuple[RandomForestClassifier, float]:

    X_train = train_feats.cpu().numpy()
    y_train = train_labels.cpu().numpy()
    X_val   = valid_feats.cpu().numpy()
    y_val   = valid_labels.cpu().numpy()

    # Step 1 — train on train only to find best threshold on val set
    clf_thresh = RandomForestClassifier(
        n_estimators      = n_estimators,
        max_depth         = max_depth,
        min_samples_split = min_samples_split,
        min_samples_leaf  = min_samples_leaf,
        max_features      = max_features,
        class_weight      = class_weight,
        bootstrap         = True,
        oob_score         = False,
        random_state      = 42,
        n_jobs            = -1,
    )
    clf_thresh.fit(X_train, y_train)

    val_proba   = clf_thresh.predict_proba(X_val)[:, 1]
    best_thresh = find_best_threshold(y_val, val_proba)

    # Step 2 — retrain on train+val for final model
    if combine_trainval:
        X_full = np.concatenate([X_train, X_val], axis=0)
        y_full = np.concatenate([y_train, y_val], axis=0)
    else:
        X_full, y_full = X_train, y_train

    classifier = RandomForestClassifier(
        n_estimators      = n_estimators,
        max_depth         = max_depth,
        min_samples_split = min_samples_split,
        min_samples_leaf  = min_samples_leaf,
        max_features      = max_features,
        class_weight      = class_weight,
        bootstrap         = True,
        oob_score         = True,
        random_state      = 42,
        n_jobs            = -1,
    )
    classifier.fit(X_full, y_full)

    if verbose and hasattr(classifier, 'oob_score_'):
        print(f"Out-of-Bag Score: {classifier.oob_score_:.4f}")

    return classifier, best_thresh


def test_r_forest(
    classifier: RandomForestClassifier,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int = None,
    prediction_threshold: float = 0.5,
    prefix: str = "rf_",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    X_test      = test_feats.cpu().numpy()
    targets_all = test_labels.cpu().numpy()

    probs_all = classifier.predict_proba(X_test)

    if probs_all.shape[1] == 2:
        preds_all  = (probs_all[:, 1] >= prediction_threshold).astype(int)
        roc_kwargs = {}
    else:
        preds_all  = classifier.predict(X_test)
        roc_kwargs = {"multi_class": "ovo", "average": "macro"}

    eval_metrics = get_eval_metrics(targets_all, preds_all, probs_all, True, prefix, roc_kwargs)
    dump = {"preds_all": preds_all, "probs_all": probs_all, "targets_all": targets_all}

    return eval_metrics, dump