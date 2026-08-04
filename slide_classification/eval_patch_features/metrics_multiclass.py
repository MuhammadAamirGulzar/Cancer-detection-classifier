from typing import Optional, Dict, Any, Union, List
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    accuracy_score,
    classification_report,
    f1_score,
    confusion_matrix,
)


def get_eval_metrics(
    targets_all: Union[List[int], np.ndarray],
    preds_all: Union[List[int], np.ndarray],
    probs_all: Optional[Union[List[float], np.ndarray]] = None,
    get_report: bool = True,
    prefix: str = "",
    roc_kwargs: Dict[str, Any] = {},
) -> Dict[str, Any]:
    """
    Calculate evaluation metrics for binary or multi-class classification.
    """
    bacc = balanced_accuracy_score(targets_all, preds_all)
    acc = accuracy_score(targets_all, preds_all)
    macro_f1 = f1_score(targets_all, preds_all, average="macro")
    cls_rep = classification_report(targets_all, preds_all, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(targets_all, preds_all)

    eval_metrics = {
        f"{prefix}acc": acc,
        f"{prefix}bacc": bacc,
        f"{prefix}macro_f1": macro_f1,
        f"{prefix}weighted_f1": cls_rep["weighted avg"]["f1-score"],
    }

    if probs_all is not None:
        try:
            n_classes = probs_all.shape[1]
            if n_classes > 2:
                # Multi-class ROC AUC (One-vs-Rest)
                roc_auc = roc_auc_score(
                    targets_all,
                    probs_all,
                    multi_class="ovr",
                    average="macro",
                    **roc_kwargs
                )
            else:
                # Binary ROC AUC
                roc_auc = roc_auc_score(targets_all, probs_all[:, 1], **roc_kwargs)
            eval_metrics[f"{prefix}auroc"] = roc_auc
        except ValueError:
            eval_metrics[f"{prefix}auroc"] = np.nan

    eval_metrics[f"{prefix}conf_matrix"] = conf_matrix
    return eval_metrics

def print_metrics(eval_metrics):
    for key, value in eval_metrics.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


avg_fold_metrics = {}
classifier_fold_count = {}  # Track fold count per classifier

def save_results_to_csv(
    save_path: str,
    classifier: str,
    fold: Union[int, str],
    eval_metrics: Dict[str, Any],
):
# first make path of the csv file using the save path, linear_classifier, fold
    results_file_path = os.path.join(save_path, f"results.csv")
    # check if already not exists then create a new file and write the header
    if not os.path.exists(results_file_path):
        with open(results_file_path, "w") as f:
            f.write("classifier,fold,Accuracy,Balanced Accuracy,Macro-F1,Weighted F1-Score,AUROC")
    # now open the file in append mode and write the results
    with open(results_file_path, "a") as f:
        f.write(
            f"\n{classifier},{fold},{eval_metrics['acc']:.4f},{eval_metrics['bacc']:.4f},{eval_metrics['macro_f1']:.4f},{eval_metrics['weighted_f1']:.4f},{eval_metrics['auroc']:.4f}"
        )
        if classifier not in avg_fold_metrics:
            avg_fold_metrics[classifier] = {k: 0 for k in eval_metrics.keys()}
            classifier_fold_count[classifier] = 0
        for k, v in eval_metrics.items():
            avg_fold_metrics[classifier][k] += v
        classifier_fold_count[classifier] += 1

        if classifier_fold_count[classifier] == 4:
            avg_results = {k: v / 4 for k, v in avg_fold_metrics[classifier].items()}   
            f.write(f"\n{classifier},Average,{avg_results['acc']:.4f},{avg_results['bacc']:.4f},{avg_results['macro_f1']:.4f},{avg_results['weighted_f1']:.4f},{avg_results['auroc']:.4f}")

        # reset the fold count and classifier dict
        avg_fold_metrics.pop(classifier)
        classifier_fold_count.pop(classifier)
