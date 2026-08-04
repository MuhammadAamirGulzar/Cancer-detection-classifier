import torch
import torch.nn.functional as F
import numpy as np
import random, os
import json
import time
from datetime import datetime, timezone
from collections import defaultdict
from typing import Tuple, Dict, Any, List
from warnings import simplefilter
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from .metrics import get_eval_metrics
from sklearn.metrics import roc_curve, auc
import torch.nn as nn
import torch.optim as optim

# ANN Binary Classifier (Now Outputs Two Class Probabilities)
class ANNBinaryClassifier:
    def __init__(self, input_dim=512, hidden_dim1=512, hidden_dim2=128, max_iter=100, verbose=True):
        self.max_iter = max_iter
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        # [CORRECTION 2026-08-04, work order Task 1.3] The final nn.Softmax was
        # removed from the training graph. nn.CrossEntropyLoss applies log-softmax
        # internally and expects raw logits, so the old network applied softmax
        # twice - flattening gradients and leaving the network underfitted.
        # The network now emits logits; softmax is applied explicitly in
        # predict_proba() only.
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim1),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim1),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim1, self.hidden_dim2),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim2),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim2, 2),
        ).to(self.device)

        self.loss_func = nn.CrossEntropyLoss()

    def compute_loss(self, preds, labels):
        return self.loss_func(preds, labels)

    def predict_logits(self, feats):
        """Raw pre-softmax scores. Used by the loss and by threshold sweeps."""
        feats = feats.to(self.device)
        self.model.eval()
        with torch.no_grad():
            return self.model(feats)

    def predict_proba(self, feats):
        """Class probabilities. Softmax lives here, and nowhere else."""
        return torch.softmax(self.predict_logits(feats), dim=1)

    def fit(self, train_feats, train_labels, val_feats=None, val_labels=None, combine_trainval=False):
        train_feats, train_labels = train_feats.to(self.device), train_labels.to(self.device)
        if val_feats is not None:
            val_feats, val_labels = val_feats.to(self.device), val_labels.to(self.device)
        if combine_trainval and val_feats is not None:
            train_feats = torch.cat([train_feats, val_feats], dim=0)
            train_labels = torch.cat([train_labels, val_labels], dim=0)

        # opt = optim.Adam(self.model.parameters(), lr=1e-4)
        # scheduler = optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)  #   Using original StepLR
        opt = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.3, patience=5, verbose=True)

        train_loss_history, val_loss_history = [], []
        best_val_loss = float("inf")
        patience, epochs_no_improve = 20, 0

        for epoch in range(self.max_iter):
            self.model.train()
            preds = self.model(train_feats)
            loss = self.compute_loss(preds, train_labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss_history.append(loss.item())

            # Validation phase
            val_loss = None
            if val_feats is not None and not combine_trainval:
                self.model.eval()
                with torch.no_grad():
                    val_preds = self.model(val_feats)
                    val_loss = self.compute_loss(val_preds, val_labels)

                val_loss_history.append(val_loss.item())

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch}")
                        break

            #   Keep StepLR scheduler
            if val_loss is not None:
                scheduler.step(val_loss)  # Call with val_loss if available
            else:
                scheduler.step(loss)  # Call without val_loss when validation data is missing

            if self.verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss: {loss:.3f}, Val Loss: {val_loss:.3f}" if val_loss else f"Epoch {epoch}: Loss: {loss:.3f}")

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
    hidden_dim2: int = 128,
    max_iter: int = 1000,
    prefix: str = "ann_",
    combine_trainval: bool = False,
    model_save_path: str = None,
    verbose: bool = False,
) -> tuple:

    if verbose:
        print(f"Train Shape: {train_feats.shape}, Test Shape: {test_feats.shape}")

    classifier = ANNBinaryClassifier(
        input_dim=input_dim,
        hidden_dim1=hidden_dim,       # Pass old `hidden_dim` as `hidden_dim1`
        hidden_dim2=hidden_dim2,
        max_iter=max_iter,
        verbose=verbose
    )

    train_loss, val_loss = classifier.fit(train_feats, train_labels, valid_feats, valid_labels, combine_trainval)

    # [CORRECTION 2026-08-04, work order Task 1.2] The torch.save that used to sit
    # here ran *inside* the caller's hyperparameter loop and wrote every grid
    # entry to the same path (input_dim is constant across the grid), so the file
    # on disk was the LAST configuration tried, not the selected one. The model is
    # now returned; the caller saves once, after selection, via
    # save_ann_checkpoint(). model_save_path is accepted but ignored, so old
    # callers keep working without silently writing a wrong checkpoint.
    if model_save_path is not None and verbose:
        print("[INFO] eval_ANN no longer saves inside the grid; "
              "use save_ann_checkpoint() on the selected configuration.")

    #   Testing phase
    probs_all = classifier.predict_proba(test_feats).cpu().numpy()

    preds_all = np.argmax(probs_all, axis=1)  #   Predict class labels
    targets_all = test_labels.cpu().numpy()

    #   Compute evaluation metrics (restored get_eval_metrics)
    eval_metrics = get_eval_metrics(targets_all, preds_all, probs_all, prefix=prefix)

    # [CORRECTION 2026-08-04, work order Task 1.1] Hyperparameters used to be
    # selected on eval_metrics['ann_macro_f1'], which is computed on the TEST
    # set - the best of four configurations was chosen using test performance and
    # then reported as a test result (optimistic bias). Validation metrics are now
    # returned under a 'val_' prefix so callers can select on held-out data.
    if valid_feats is not None and valid_labels is not None and not combine_trainval:
        val_probs = classifier.predict_proba(valid_feats).cpu().numpy()
        val_preds = np.argmax(val_probs, axis=1)
        val_targets = valid_labels.cpu().numpy()
        val_metrics = get_eval_metrics(val_targets, val_preds, val_probs, prefix=f"val_{prefix}")
        eval_metrics.update(val_metrics)

    if verbose:
        plot_training_logs({"train_loss": train_loss, "valid_loss": val_loss})
        plot_roc_auc(targets_all, probs_all[:, 1])
    dump = {
        "preds_all": preds_all,
        "probs_all": probs_all,
        "targets_all": targets_all,
        "classifier": classifier,
        "hparams": {"input_dim": input_dim, "hidden_dim1": hidden_dim,
                    "hidden_dim2": hidden_dim2, "max_iter": max_iter},
    }
    return eval_metrics, dump


def save_ann_checkpoint(
    model_save_path: str,
    fold: int,
    classifier: "ANNBinaryClassifier",
    selection_metric: str = "",
    selection_value: float = None,
) -> str:
    """Persist the ANN configuration that was actually selected (Task 1.2).

    Writes ``fold{fold}_ann_{input_dim}_{h1}_{h2}.pth`` plus a sibling
    ``fold{fold}_ann_config.json``. The old filename
    (``fold{fold}_trained_ann_model_{input_dim}.pth``) could not distinguish the
    four grid entries from one another, so every one of them overwrote the last.
    """
    os.makedirs(model_save_path, exist_ok=True)
    h1, h2, d = classifier.hidden_dim1, classifier.hidden_dim2, classifier.input_dim
    ckpt_name = f"fold{fold}_ann_{d}_{h1}_{h2}.pth"
    ckpt_path = os.path.join(model_save_path, ckpt_name)
    torch.save(classifier.model.state_dict(), ckpt_path)

    config = {
        "input_dim": d,
        "hidden_dim1": h1,
        "hidden_dim2": h2,
        "max_iter": classifier.max_iter,
        "checkpoint": ckpt_name,
        "selection_metric": selection_metric,
        "selection_value": selection_value,
        "softmax_in_graph": False,  # Task 1.3: model emits logits
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(model_save_path, f"fold{fold}_ann_config.json"), "w") as fh:
        json.dump(config, fh, indent=2)
    return ckpt_path


def load_ann_checkpoint(model_save_path: str, fold: int, input_dim: int = None):
    """Rebuild a saved ANN from its config JSON, inferring dims if absent.

    Reads ``fold{fold}_ann_config.json`` when present. Falls back to inferring
    the architecture from the state dict's weight shapes, which is what
    ``Slide_Classification.ipynb`` cell 14 did - kept as a safety net for
    checkpoints written before Task 1.2 landed.
    """
    cfg_path = os.path.join(model_save_path, f"fold{fold}_ann_config.json")
    state_path = None
    legacy = False

    if os.path.exists(cfg_path):
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        state_path = os.path.join(model_save_path, cfg["checkpoint"])
        h1, h2, d = cfg["hidden_dim1"], cfg["hidden_dim2"], cfg["input_dim"]
        legacy = bool(cfg.get("softmax_in_graph", False))
    else:
        # Legacy checkpoint: infer from state-dict shapes.
        candidates = [f for f in os.listdir(model_save_path)
                      if f.startswith(f"fold{fold}_") and f.endswith(".pth")]
        if not candidates:
            raise FileNotFoundError(
                f"No ANN checkpoint for fold {fold} in {model_save_path}")
        state_path = os.path.join(model_save_path, sorted(candidates)[0])
        sd = torch.load(state_path, map_location="cpu")
        d = sd["0.weight"].shape[1]
        h1 = sd["0.weight"].shape[0]
        h2 = sd["4.weight"].shape[0]
        legacy = True
        print(f"[WARN] {os.path.basename(state_path)} has no config.json; "
              f"inferred input_dim={d} h1={h1} h2={h2}. This checkpoint predates "
              f"Task 1.2/1.3 - it still has Softmax in the graph and its "
              f"architecture may not be the one that was selected.")

    if input_dim is not None and input_dim != d:
        raise ValueError(
            f"Checkpoint input_dim={d} does not match expected {input_dim} "
            f"({state_path}). Refusing to load a mismatched architecture.")

    clf = ANNBinaryClassifier(input_dim=d, hidden_dim1=h1, hidden_dim2=h2, verbose=False)
    state = torch.load(state_path, map_location=clf.device)
    clf.model.load_state_dict(state)
    clf.model.eval()
    return clf, {"input_dim": d, "hidden_dim1": h1, "hidden_dim2": h2,
                 "checkpoint": os.path.basename(state_path), "legacy": legacy}


#   Function to Load and Test Saved ANN Model
def test_saved_ann_model(model_save_path: str, fold: int, test_feats: torch.Tensor,
                         test_labels: torch.Tensor, input_dim: int = None):
    """Evaluate a saved ANN checkpoint on a test set.

    [CORRECTION 2026-08-04, work order Tasks 1.2/1.3] The previous version
    hardcoded a *single*-hidden-layer architecture with a trailing Softmax,
    which matched no checkpoint this pipeline has ever written (they are all
    two-hidden-layer). It would have failed with a shape mismatch. It now
    rebuilds the architecture from the checkpoint's config.json, and softmax is
    applied by predict_proba rather than assumed to be baked into the graph.
    """
    classifier, meta = load_ann_checkpoint(model_save_path, fold, input_dim=input_dim)
    probabilities = classifier.predict_proba(test_feats).cpu().numpy()
    predictions = np.argmax(probabilities, axis=1)
    targets_all = test_labels.cpu().numpy()

    eval_metrics = get_eval_metrics(targets_all, predictions, probabilities, True, prefix="ann_")
    dump = {"preds_all": predictions, "probs_all": probabilities,
            "targets_all": targets_all, "checkpoint_meta": meta}
    return eval_metrics, dump


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
