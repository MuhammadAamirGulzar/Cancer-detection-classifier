import os
import torch
from torch.utils.data import Dataset
from typing import List

class WSIDataset(Dataset):
    def __init__(self, save_dir: str, fold_ids: List[str]):
        self.data = []
        self.save_dir = save_dir
        self.fold_ids = fold_ids
        self._load_data()

    def _load_data(self):
        for wsi_file in os.listdir(self.save_dir):
            wsi_path = os.path.join(self.save_dir, wsi_file)
            # Extract WSI ID without extension
            wsi_id = os.path.splitext(wsi_file)[0] # Get the file name without extension
            # use below wsi_id method to get the first 12 characters in tcga cross validation
            # wsi_id = wsi_file[:12]  #Extract first 12 characters
            if wsi_id not in self.fold_ids:
                continue  # Skip if the WSI is not in the current fold

            if wsi_path.endswith('.pt'):
                try:
                    # Load WSI features
                    wsi_features = torch.load(wsi_path)
                    if wsi_features.is_cuda:
                        wsi_features = wsi_features.cpu()
                    # Average all feature vectors in the .pt file
                    if wsi_features.dim() > 1:
                        # final_features = torch.mean(wsi_features, dim=0)
                        final_features = wsi_features.flatten()
                    else:
                        final_features = wsi_features  # In case it is already a single vector
                    
                    # Determine label based on WSI file name
                    label = 0 if '_nonMSI' in wsi_file else 1

                    # Append the averaged or flattened features, label, and WSI ID
                    self.data.append((final_features, label, wsi_id))
                except Exception as e:
                    print(f"Error loading {wsi_path}: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, label, wsi_id = self.data[idx]

        return features, label, wsi_id