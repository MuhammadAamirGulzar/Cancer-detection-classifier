from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
from typing import Dict, List, Tuple, Any
from .metrics import get_eval_metrics
import time
import torch
import joblib, os

def eval_linear(
    fold: int,
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    valid_feats: torch.Tensor,
    valid_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    max_iter: int = 1000,
    combine_trainval: bool = True,
    C: float = 1.0,
    prefix: str = "lin_",
    save_path: str=None,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate a linear logistic regression probe using Scikit-learn.

    Args:
        train_feats: Training feature vectors.
        train_labels: Training labels.
        valid_feats: Validation feature vectors.
        valid_labels: Validation labels.
        test_feats: Test feature vectors.
        test_labels: Test labels.
        max_iter: Maximum number of iterations for the logistic regression solver.
        combine_trainval: Whether to combine training and validation data for the final model.
        C: Regularization strength (inverse of regularization constant).
        verbose: Whether to print debug information.

    Returns:
        results: Dictionary containing evaluation metrics (accuracy, F1, ROC-AUC).
        dump: Dictionary containing predictions and probabilities.
    """
    if verbose:
        print(f"Train Shape: {train_feats.shape}, Test Shape: {test_feats.shape}")
        if valid_feats is not None:
            print(f"Validation Shape: {valid_feats.shape}")
    start = time.time()
    # train linear probe
    classifier = train_linear_probe(
        train_feats,
        train_labels,
        valid_feats,
        valid_labels,
        max_iter=max_iter,
        combine_trainval=combine_trainval,
        C=C,
        verbose=verbose,
    )
    if save_path is not None:
        model_path = os.path.join(save_path, f"fold{fold}_logistic_regression.pkl")
        joblib.dump(classifier, model_path)
    # test linear probe
    results, dump = test_linear_probe(classifier, test_feats, test_labels, prefix=prefix)    
    if verbose:
        print(f"Linear Probe Evaluation Time: {time.time() - start:.3f} s")

    return results, dump

def test_saved_logistic_model(test_feats: torch.Tensor, test_labels: torch.Tensor, model_path="logistic_regression.pkl"):
    # Load trained logistic regression model
    classifier = joblib.load(model_path)
    #Get predictions
    probs_all = classifier.predict_proba(test_feats)  # Probabilities for class 1
    preds_all = classifier.predict(test_feats)  # Predicted class labels

    # Convert labels to numpy
    targets_all = test_labels.cpu().numpy()
    # Compute evaluation metrics
    eval_metrics = get_eval_metrics(targets_all, preds_all, probs_all, True, prefix="lin_")
    dump = {"preds_all": preds_all, "probs_all": probs_all, "targets_all": targets_all}
    return eval_metrics, dump



def train_linear_probe(
    train_feats,
    train_labels,
    valid_feats,
    valid_labels,
    max_iter=1000,
    combine_trainval=True,
    C: float = 1.0,
    verbose=True,
) -> LogisticRegression:
    """
    Train a logistic regression classifier using Scikit-learn.

    Args:
        feats: Feature vectors for training.
        labels: Labels for training.
        C: Regularization strength (inverse of regularization constant).
        max_iter: Maximum number of iterations for the logistic regression solver.
        verbose: Whether to print debug information.

    Returns:
        A trained Scikit-learn logistic regression model.
    """
    # Combine train and validation sets if required
    # train final classifier
    if combine_trainval and (valid_feats is not None):
        train_feats = torch.cat([train_feats, valid_feats], dim=0)
        train_labels = torch.cat([train_labels, valid_labels], dim=0)
        if verbose:
            print(f"Combined Train and Validation Shape: {train_feats.shape}")
    classifier = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs", verbose=verbose, random_state=42)
    classifier.fit(train_feats, train_labels)
    if verbose:
        print(f"Training complete. Coefficients shape: {classifier.coef_.shape}")
    return classifier


def test_linear_probe(
    classifier: LogisticRegression,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int = None,
    prefix: str = "lin_",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate a trained logistic regression classifier on the test set.

    Args:
        classifier: A trained Scikit-learn logistic regression model.
        test_feats: Feature vectors for testing.
        test_labels: Labels for testing.
        verbose: Whether to print debug information.

    Returns:
        results: Dictionary containing evaluation metrics (accuracy, F1, ROC-AUC).
        dump: Dictionary containing predictions and probabilities.
    """
    # evaluate
    NUM_C = len(set(test_labels.cpu().numpy())) if num_classes is None else num_classes
    # predict and get probabilities 
    if NUM_C == 2:
        probs_all = classifier.predict_proba(test_feats)
        roc_kwargs = {}
    else:
        probs_all = classifier.predict_proba(test_feats)
        roc_kwargs = {"multi_class": "ovo", "average": "macro"}

    preds_all = classifier.predict(test_feats)
    targets_all = test_labels.detach().cpu().numpy()
    eval_metrics = get_eval_metrics(targets_all, preds_all, probs_all, True, prefix, roc_kwargs)
    dump = {"preds_all": preds_all, "probs_all": probs_all, "targets_all": targets_all}

    return eval_metrics, dump

