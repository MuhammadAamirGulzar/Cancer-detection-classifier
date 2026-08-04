# Virchow2  (saved via the combined pipeline script, on the Windows machine)
import os
import torch
from tqdm import tqdm

FEATURE_DIM = 2560
# If True, slides whose output .pt already exists and passes validation
# are skipped entirely — no re-reading patches, no re-averaging.
SKIP_EXISTING = True
# If True, the skip-check does a full torch.load() + shape check on every
# existing output file (safest, but slow when many files already exist —
# each one is a disk read before any real work starts, with no progress
# shown). If False (default), it just checks the file exists and is above
# a trivial size threshold — fast, and catches zero-byte/truncated files
# from a hard crash, but won't catch a file that's non-empty but corrupt.
STRICT_VALIDATION = False
MIN_VALID_BYTES = 1024  # a real (1, 2560) float32 tensor is ~10KB+; anything under this is suspect


def output_already_valid(save_path):
    """
    Returns True if save_path should be treated as already-done.
    """
    if not os.path.exists(save_path):
        return False

    if not STRICT_VALIDATION:
        try:
            return os.path.getsize(save_path) >= MIN_VALID_BYTES
        except OSError:
            return False

    try:
        existing = torch.load(save_path, map_location="cpu")
        return (
            isinstance(existing, torch.Tensor)
            and tuple(existing.shape) == (1, FEATURE_DIM)
        )
    except Exception:
        return False


def average_patch_features(source_root, dest_root):
    os.makedirs(dest_root, exist_ok=True)

    n_skipped = 0
    n_saved = 0

    subfolder_names = sorted(
        d for d in os.listdir(source_root)
        if os.path.isdir(os.path.join(source_root, d)) and d != "slide_aggregation"
    )

    pbar = tqdm(subfolder_names, desc="Slides", unit="slide")
    for subfolder_name in pbar:
        pbar.set_postfix_str(subfolder_name[:30])
        subfolder_path = os.path.join(source_root, subfolder_name)

        dest_file_path = os.path.join(dest_root, f"{subfolder_name}.pt")

        # Resume check: only this slide's own output matters. If it
        # already exists and passes validation, skip re-reading and
        # re-averaging it entirely.
        if SKIP_EXISTING and output_already_valid(dest_file_path):
            n_skipped += 1
            continue

        pt_files = [f for f in os.listdir(subfolder_path) if f.endswith('.pt')]
        if not pt_files:
            tqdm.write(f"⚠ No .pt files in {subfolder_name}, skipping...")
            continue

        all_features = []

        for pt_file in pt_files:
            file_path = os.path.join(subfolder_path, pt_file)
            try:
                data = torch.load(file_path)  # shape: 5 × 2560
                if data.ndim != 2 or data.shape[0] != 5:
                    raise ValueError(f"Unexpected tensor shape {data.shape} in {file_path}")

                # Average over the 5 crops → 1 × 2560
                mean_feature = data.mean(dim=0, keepdim=True)
                all_features.append(mean_feature)

            except Exception as e:
                tqdm.write(f"❌ Error loading {file_path}: {e}")
                continue

        if not all_features:
            tqdm.write(f"⚠ No valid .pt files processed in {subfolder_name}")
            continue

        # Stack all patch-level vectors and average across patches → 1 × 2560
        final_feature = torch.cat(all_features, dim=0).mean(dim=0, keepdim=True)

        # Atomic write: save to a temp file then rename, so a crash/kill
        # mid-write never leaves a truncated .pt that a later resume run
        # would treat as valid.
        tmp_path = dest_file_path + ".tmp"
        torch.save(final_feature, tmp_path)
        os.replace(tmp_path, dest_file_path)
        n_saved += 1
        tqdm.write(f"✅ Saved averaged feature: {dest_file_path}")

    print(f"\nDone. Saved {n_saved} slide(s). Skipped {n_skipped} slide(s) with existing valid output.")


source_path = r"F:\surgen_processed\virchow2\features"
dest_path = r"D:\Aamir Gulzar\KSA_project2\surgen_data\surgen_processed\virchow2\features\slide_aggregation\Averaging\virchow2"
average_patch_features(source_path, dest_path)