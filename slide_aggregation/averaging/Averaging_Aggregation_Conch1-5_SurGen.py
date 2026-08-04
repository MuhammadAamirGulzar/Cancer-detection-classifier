# CONCH 1.5  (saved via the combined pipeline script, on the Windows machine)
import os
import torch

def average_patch_features(source_root, dest_root):
    os.makedirs(dest_root, exist_ok=True)

    for subfolder_name in os.listdir(source_root):
        subfolder_path = os.path.join(source_root, subfolder_name)
        if not os.path.isdir(subfolder_path) or subfolder_name == "slide_aggregation":
            continue

        pt_files = [f for f in os.listdir(subfolder_path) if f.endswith('.pt')]
        if not pt_files:
            print(f"⚠ No .pt files in {subfolder_name}, skipping...")
            continue

        all_features = []

        for pt_file in pt_files:
            file_path = os.path.join(subfolder_path, pt_file)
            try:
                data = torch.load(file_path)  # shape: 5 × 768
                if data.ndim != 2 or data.shape[0] != 5:
                    raise ValueError(f"Unexpected tensor shape {data.shape} in {file_path}")

                # Average over the 5 crops → 1 × 768
                mean_feature = data.mean(dim=0, keepdim=True)
                all_features.append(mean_feature)

            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
                continue

        if not all_features:
            print(f"⚠ No valid .pt files processed in {subfolder_name}")
            continue

        # Stack all patch-level vectors and average across patches → 1 × 768
        final_feature = torch.cat(all_features, dim=0).mean(dim=0, keepdim=True)

        dest_file_path = os.path.join(dest_root, f"{subfolder_name}.pt")
        torch.save(final_feature, dest_file_path)
        print(f"✅ Saved averaged feature: {dest_file_path}")

source_path = r"D:\Aamir Gulzar\KSA_project2\surgen_data\surgen_processed\conch1-5\features"
dest_path = r"D:\Aamir Gulzar\KSA_project2\surgen_data\surgen_processed\conch1-5\features\slide_aggregation\Averaging\conch1-5"
average_patch_features(source_path, dest_path)
