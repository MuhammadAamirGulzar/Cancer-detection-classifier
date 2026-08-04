import os
import torch
from torch.utils.data import Dataset
from typing import List
import pandas as pd

class WSIDataset(Dataset):
    def __init__(self, save_dir: str, fold_ids: List[str]):
        self.data = []
        self.save_dir = save_dir
        self.fold_ids = fold_ids
        self._load_data()

    def _load_data(self):
        files = os.listdir(self.save_dir)
        # Change this depending on your classification task
        task_name = "MSIH"  # Example task name, change as needed

        if task_name == "MSIH":
            K_FOLDS_PATH = r"D:\Aamir Gulzar\KSA_project2\Cancer-detection-classifier\Slide_Classification\kfolds_IDARS_fixed.csv"
        elif task_name == "BRAF":
            K_FOLDS_PATH = r"D:\Aamir Gulzar\KSA_project2\Cancer-detection-classifier\Slide_Classification\kfold_BRAF.csv"
        elif task_name == "KRAS":
            K_FOLDS_PATH = r"D:\Aamir Gulzar\KSA_project2\Cancer-detection-classifier\Slide_Classification\kfold_KRAS.csv"
        elif task_name == "TP53":
            K_FOLDS_PATH = r"D:\Aamir Gulzar\KSA_project2\Cancer-detection-classifier\Slide_Classification\kfold_TP53.csv"
        elif task_name == "CIMP":
            K_FOLDS_PATH = r"D:\Aamir Gulzar\KSA_project2\Cancer-detection-classifier\Slide_Classification\kfold_CIMP.csv"
        else:
            raise ValueError(f"Unknown task name: {task_name}")
        
        folds_df = pd.read_csv(K_FOLDS_PATH)

        for wsi_file in files:
            wsi_path = os.path.join(self.save_dir, wsi_file)

            # Extract WSI ID without extension
            wsi_id = os.path.splitext(wsi_file)[0]

            # Skip files not in fold_ids
            if wsi_id not in self.fold_ids:
                continue

            # Only process .pt files
            if not wsi_path.endswith('.pt'):
                continue

            try:
                # Load WSI features
                wsi_features = torch.load(wsi_path)

                if wsi_features.is_cuda:
                    wsi_features = wsi_features.cpu()

                # Average or flatten features
                if wsi_features.dim() > 1:
                    final_features = wsi_features.flatten()
                else:
                    final_features = wsi_features

                # Determine label based on task_name
                if task_name == "MSIH":
                    # MSIH classification
                    label = 0 if '_nonMSI' in wsi_file else 1
                
                elif task_name == "BRAF":
                    # BRAF classification
                    if wsi_id in folds_df['WSI_Id'].values:
                        fold_label = folds_df.loc[folds_df['WSI_Id'] == wsi_id, 'BRAF_mutation'].values[0]
                        label = 0 if fold_label == 'WT' else 1
                
                elif task_name == "KRAS":
                    # KRAS classification
                    # Assuming KRAS mutation is in folds_df and you want the same approach as BRAF
                    if wsi_id in folds_df['WSI_Id'].values:
                        fold_label = folds_df.loc[folds_df['WSI_Id'] == wsi_id, 'KRAS_mutation'].values[0]
                        label = 0 if fold_label == 'WT' else 1
                
                elif task_name == "TP53":
                    # TP53 classification
                    if wsi_id in folds_df['WSI_Id'].values:
                        fold_label = folds_df.loc[folds_df['WSI_Id'] == wsi_id, 'TP53_mutation'].values[0]
                        label = 0 if fold_label == 'WT' else 1

                elif task_name == "CIMP":
                    # CIMP classification
                    if wsi_id in folds_df['WSI_Id'].values:
                        fold_label = folds_df.loc[folds_df['WSI_Id'] == wsi_id, 'HypermethylationCategory'].values[0]
                        if fold_label == 'Non-CIMP':
                            label = 0
                        elif fold_label == 'CIMP-H':
                            label = 1
                        elif fold_label == 'CRC CIMP-L':
                            label = 2

                else:
                    raise ValueError(f"Unknown task name: {task_name}")

                # Store in dataset
                self.data.append((final_features, label, wsi_id))

            except Exception as e:
                print(f"[ERROR] Loading failed for {wsi_path}: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, label, wsi_id = self.data[idx]
        return features, label, wsi_id