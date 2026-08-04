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
import joblib,os
import torch
import numpy as np
import joblib
import os
from torch.nn.functional import normalize
from sklearn.metrics import roc_auc_score, confusion_matrix

def eval_protonet(
    fold: int,
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    val_feats: torch.Tensor,
    val_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    normalize_feats: bool = True,
    prefix: str = "proto_",
    model_save_path: str=None,
    verbose: bool = False,
) -> tuple:
    
    if verbose:
        print(f"Train Features Shape: {train_feats.shape}, Train Label Shape: {train_labels.shape}")
        if val_feats is not None:
            print(f"Validation Features Shape: {val_feats.shape}, Validation Label Shape: {val_labels.shape}")
        print(f"Test Features Shape: {test_feats.shape}, Test Label Shape: {test_labels.shape}")

    #    Combine train and validation data (since ProtoNet does not use val separately)
    if val_feats is not None:
        train_feats = torch.cat((train_feats, val_feats), dim=0)
        train_labels = torch.cat((train_labels, val_labels), dim=0)

    #    Normalize features
    if normalize_feats:
        train_feats = normalize(train_feats, dim=-1, p=2)
        test_feats = normalize(test_feats, dim=-1, p=2)

    #    Compute class prototypes
    class_ids = sorted(np.unique(train_labels.numpy()))
    prototypes = torch.stack(
        [train_feats[train_labels == class_id].mean(dim=0) for class_id in class_ids]
    )
    labels_proto = torch.tensor(class_ids)
    if model_save_path is not None:
        #    Save the trained ProtoNet model (prototypes & labels)
        model_path = os.path.join(model_save_path, f"fold{fold}_protonet_model.pkl")
        joblib.dump({"prototypes": prototypes.cpu(), "labels_proto": labels_proto.cpu()}, model_path)

    #    Compute pairwise distances
    pairwise_distances = (test_feats[:, None] - prototypes[None, :]).norm(dim=-1, p=2)

    #    Predict labels based on the closest prototype
    predicted_labels = labels_proto[pairwise_distances.argmin(dim=1)]

    #    Compute class probabilities using softmax on distances
    probs_all = torch.nn.functional.softmax(-pairwise_distances, dim=1).cpu().numpy()  # Convert distances into probabilities

    #    Compute evaluation metrics
    metrics = get_eval_metrics(test_labels.numpy(), predicted_labels.numpy(), probs_all, prefix=prefix)

    #    Store results
    dump = {
        "preds_all": predicted_labels.numpy(),
        "targets_all": test_labels.numpy(),
        "pairwise_distances": pairwise_distances.cpu().numpy(),
        "prototypes": prototypes.cpu().numpy(),
        "probs_all": probs_all
    }
    return metrics, dump


#    Function to Load & Test ProtoNet Model
def test_saved_protonet_model(test_feats: torch.Tensor, test_labels: torch.Tensor, model_path="protonet_model.pkl"):
    #    Load trained class prototypes & labels
    saved_model = joblib.load(model_path)
    prototypes = torch.tensor(saved_model["prototypes"])
    labels_proto = torch.tensor(saved_model["labels_proto"])

    #    Compute pairwise distances
    pairwise_distances = (test_feats[:, None] - prototypes[None, :]).norm(dim=-1, p=2)

    #    Predict labels based on the closest prototype
    predicted_labels = labels_proto[pairwise_distances.argmin(dim=1)].cpu().numpy()

    #    Compute class probabilities using softmax on distances
    probs_all = torch.nn.functional.softmax(-pairwise_distances, dim=1).cpu().numpy()
    targets_all = test_labels.cpu().numpy()

    #    Compute evaluation metrics
    eval_metrics = get_eval_metrics(targets_all, predicted_labels, probs_all, True,prefix="proto_")
    dump = {
        "preds_all": predicted_labels,
        "targets_all": targets_all,
        "pairwise_distances": pairwise_distances.cpu().numpy(),
        "prototypes": prototypes.cpu().numpy(),
        "probs_all": probs_all
    }
    return eval_metrics, dump
