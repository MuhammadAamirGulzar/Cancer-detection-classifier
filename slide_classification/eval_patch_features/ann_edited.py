import torch
import torch.nn.functional as F
import numpy as np
import random, os
import time
from collections import defaultdict
from typing import Tuple, Dict, Any, List
from warnings import simplefilter
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from .metrics import get_eval_metrics
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import torch
import torch.nn as nn
import torch.optim as optim

# ANN Binary Classifier (Now Outputs Two Class Probabilities)

class ANNBinaryClassifier:
    def __init__(
        self,
        input_dim=512,
        hidden_dim=512,
        max_iter=100,
        lr=1e-4,
        weight_decay=1e-4,
        patience=10,
        verbose=True
    ):
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define the model (Two output logits, no Softmax)
        # CrossEntropyLoss expects raw logits.
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.7),
            nn.Linear(hidden_dim, 2)  # Outputs raw logits for [MSS, MSI]
        ).to(self.device)

        self.loss_func = nn.CrossEntropyLoss()

        # Optimizer & Scheduler
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_iter = max_iter
        self.patience = patience

    def compute_loss(self, logits, labels):
        # labels shape: (batch,) containing class indices
        return self.loss_func(logits, labels)

    def predict_proba(self, feats):
        """
        Returns probabilities [P(MSS), P(MSI)] for each sample.
        """
        self.model.eval()
        feats = feats.to(self.device)
        with torch.no_grad():
            logits = self.model(feats)
            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=1)
        return probs

    def fit(self, train_feats, train_labels, val_feats=None, val_labels=None, combine_trainval=False):
        train_feats, train_labels = train_feats.to(self.device), train_labels.to(self.device)
        if val_feats is not None:
            val_feats, val_labels = val_feats.to(self.device), val_labels.to(self.device)
        
        if combine_trainval and val_feats is not None:
            train_feats = torch.cat([train_feats, val_feats], dim=0)
            train_labels = torch.cat([train_labels, val_labels], dim=0)
            # Setting val_feats = None ensures only a single training set
            val_feats, val_labels = None, None

        # Adam Optimizer with ReduceLROnPlateau
        opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.3, patience=5, verbose=self.verbose
        )

        best_val_loss = float("inf")
        epochs_no_improve = 0

        train_loss_history, val_loss_history = [], []

        for epoch in range(self.max_iter):
            self.model.train()

            # Forward pass
            logits = self.model(train_feats)
            loss = self.compute_loss(logits, train_labels)

            # Backprop
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss = loss.item()
            train_loss_history.append(train_loss)

            # Validation phase (only if we have a separate validation set)
            val_loss = None
            if val_feats is not None:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(val_feats)
                    val_loss_tensor = self.compute_loss(val_logits, val_labels)
                    val_loss = val_loss_tensor.item()
                    val_loss_history.append(val_loss)

                # Scheduler step & early stopping
                scheduler.step(val_loss_tensor)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch}")
                        break
            else:
                # If no validation set is used, we just pass training loss to the scheduler
                scheduler.step(loss)

            # Logging
            if self.verbose and (epoch % 10 == 0):
                if val_loss is not None:
                    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")

        return train_loss_history, val_loss_history



# Training and Evaluation Functions
def eval_ANN(
    fold: int,
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    valid_feats: torch.Tensor,
    valid_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    input_dim: int = 512,
    hidden_dim: int = 512,
    max_iter: int = 1000,
    combine_trainval: bool = False,
    model_save_path: str="",
    verbose: bool = False,
) -> tuple:
    if verbose:
        print(f"Train Shape: {train_feats.shape}, Validation Shape: {valid_feats.shape}, Test Shape: {test_feats.shape}")

    classifier = ANNBinaryClassifier(input_dim=input_dim, hidden_dim=hidden_dim, max_iter=max_iter, verbose=verbose)
    train_loss, val_loss = classifier.fit(train_feats, train_labels, valid_feats, valid_labels, combine_trainval)

    #   Save model
    model_path = os.path.join(model_save_path, f"fold{fold}_trained_ann_model_{input_dim}.pth")
    torch.save(classifier.model.state_dict(), model_path)

    #   Testing phase
    probs_all = classifier.predict_proba(test_feats).cpu().numpy()

    preds_all = np.argmax(probs_all, axis=1)  #   Predict class labels
    targets_all = test_labels.cpu().numpy()

    #   Compute evaluation metrics (restored get_eval_metrics)
    eval_metrics = get_eval_metrics(targets_all, preds_all, probs_all)
    if verbose:
        plot_training_logs({"train_loss": train_loss, "valid_loss": val_loss})
        plot_roc_auc(targets_all, probs_all[:, 1])
    dump = {"preds_all": preds_all, "probs_all": probs_all, "targets_all": targets_all}
    return eval_metrics, dump


#   Function to Load and Test Saved ANN Model
def test_saved_ann_model(input_dim: int, hidden_dim: int, test_feats: torch.Tensor, test_labels: torch.Tensor, model_path="best_ann_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #   Define model structure (Must match trained model)
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_dim),
        nn.Dropout(0.5),
        nn.Linear(hidden_dim, 2),  #   Two outputs for [MSS, MSI]
        nn.Softmax(dim=1)  #   Ensure outputs sum to 1
    ).to(device)

    #   Load trained model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode

    #   Convert test features to tensor
    test_feats = test_feats.to(device)

    #   Get predictions
    with torch.no_grad():
        probabilities = model(test_feats).cpu().numpy()

    #   Convert probabilities to class labels (binary classification)
    predictions = np.argmax(probabilities, axis=1)
    targets_all = test_labels.cpu().numpy()

    #   Compute evaluation metrics
    eval_metrics = get_eval_metrics(targets_all, predictions, probabilities,True, prefix="ann_")

    return eval_metrics


def plot_training_logs(training_logs):
    plt.figure(figsize=(10, 6))
    plt.plot(training_logs["train_loss"], label="Train Loss", marker="o")
    if "valid_loss" in training_logs and training_logs["valid_loss"]:
        plt.plot(training_logs["valid_loss"], label="Validation Loss", marker="x")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_roc_auc(targets, probs):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(targets, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
