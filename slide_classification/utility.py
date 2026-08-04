# [ADDED]
import os
import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import numpy as np
import pandas as pd


def write_data_in_excel(excel_path: str, df: pd.DataFrame, sheet_name: str, index: bool=False):
    """
    Writes 'df' to 'sheet_name' in 'excel_path'. 
    - If the file doesn't exist, creates a new file with one sheet.
    - If the file exists:
        - Reads all sheets into a dict of DataFrames.
        - Replaces or creates 'sheet_name' with 'df'.
        - Writes all sheets back, preserving all other sheets.
    """
    if not os.path.exists(excel_path):
        # Create the file from scratch
        df.to_excel(excel_path, sheet_name=sheet_name, index=index)
        return
    
    # Load existing file
    with pd.ExcelFile(excel_path, engine='openpyxl') as xls:
        sheets_dict = {s: pd.read_excel(xls, sheet_name=s) for s in xls.sheet_names}

    # Replace or add the target sheet
    sheets_dict[sheet_name] = df
    
    # Now write the entire workbook in write mode
    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='w') as writer:
        for s_name, s_df in sheets_dict.items():
            s_df.to_excel(writer, sheet_name=s_name, index=index)


def build_probs_df(results_per_fold, model_name):
    """
    Builds a DataFrame with predictions and probabilities for a given model
    from the cross-validation results.
    
    Args:
        results_per_fold (List[Dict]): List of results from each fold
        model_name (str): Model identifier to prefix prediction columns

    Returns:
        pd.DataFrame: Concatenated DataFrame across folds
    """
    predictions_list = []

    for result in results_per_fold:
        test_ids = result.get("wsi_ids", [])
        preds = result.get("preds_all", [])
        probs = result.get("probs_all", [])
        targets = result.get("targets_all", [])
        fold_number = result.get("fold", -1)

        if not (len(test_ids) == len(preds) == len(probs) == len(targets)):
            print(f"[WARNING] Skipping Fold {fold_number} due to length mismatch.")
            continue

        df_preds = pd.DataFrame({
            "Fold": [fold_number] * len(test_ids),
            "WSI_ID": test_ids,
            "Target": targets,
            f"{model_name}_Pred": preds,
            f"{model_name}_Prob": [round(p[int(pred)], 2) for p, pred in zip(probs, preds)]
        })

        predictions_list.append(df_preds)

    if predictions_list:
        return pd.concat(predictions_list, ignore_index=True)
    else:
        return pd.DataFrame()  # return empty DF if all folds failed


def calculate_metric_averages(all_fold_results, metric_indices,model_prefix):
    """
    Calculate the average of specified metrics over multiple folds.

    Args:
        all_fold_results (list of dicts): Results for each fold.
        metric_indices (dict): Mapping of metric names to their indices.

    Returns:
        dict: Averages of the specified metrics across folds.
    """
    # Initialize averages dictionary
    averages = {f'{model_prefix}_{metric}': 0 for metric in metric_indices.keys()}
    counts = {f'{model_prefix}_{metric}': 0 for metric in metric_indices.keys()}  # Keep track of valid metrics
    num_folds = len(all_fold_results)

    for result in all_fold_results:
        # Iterate through metrics by their index
        for metric, index in metric_indices.items():
            prefixed_metric = f"{model_prefix}_{metric}"
            try:
                metric_name = list(result.keys())[index]  # Extract the metric name by index
                if metric_name in result and isinstance(result[metric_name], (int, float)):  # Check if metric exists and is numeric
                    averages[prefixed_metric] += result[metric_name]
                    counts[prefixed_metric] += 1
            except IndexError:
                # Metric not present in this result due to model differences
                continue
            except Exception as e:
                print(f"Error processing metric '{metric}': {e}")
    # Compute average only for metrics with valid values
    for metric in averages:
        if counts[metric] > 0:
            averages[metric] /= counts[metric]
    return averages

def average_confusion_matrices(conf_matrices):
    """
    Takes a list of confusion matrices (as np arrays) and returns the element-wise average.
    """
    conf_matrices = [np.array(cm) for cm in conf_matrices]
    return np.mean(conf_matrices, axis=0).round().astype(int).tolist() 