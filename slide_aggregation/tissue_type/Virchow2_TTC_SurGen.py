import os
import torch
import numpy as np
from tqdm import tqdm

# ========================= CONFIGURATION =========================
FEATURES_ROOT = r"F:\surgen_processed\virchow2\features"
OUTPUT_DIR    = r"D:\Aamir Gulzar\KSA_project2\surgen_data\surgen_processed\virchow2\features\slide_aggregation\Tissue_Type_Clustering\virchow2"
MODEL_PATH    = r"D:\Aamir Gulzar\KSA_project2\Cancer-detection-classifier\TissueClassifier_CRC100K\models\ann_crc100k_Virchow2.pth"

NUM_CLASSES = 9
FEATURE_DIM = 2560
BATCH_SIZE  = 8

# If True, slides whose output .pt already exists and passes validation
# are skipped entirely — no re-reading patches, no re-running the classifier.
SKIP_EXISTING = True
# ====================================================================


def output_already_valid(save_path):
    """
    Returns True if save_path exists and contains a well-formed
    (NUM_CLASSES, FEATURE_DIM) tensor. Guards against resuming onto a
    truncated/corrupt file left behind by a previous crashed run.
    """
    if not os.path.exists(save_path):
        return False
    try:
        existing = torch.load(save_path, map_location="cpu")
        return (
            isinstance(existing, torch.Tensor)
            and tuple(existing.shape) == (NUM_CLASSES, FEATURE_DIM)
        )
    except Exception:
        return False


def build_model(device):
    """
    Loads the trained ANN tissue classifier.
    torch.load here returns the full nn.Sequential module (as in the
    original notebook, model.model = torch.load(model_path)).
    If you instead saved a state_dict, swap this for the appropriate
    architecture + load_state_dict call.
    """
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    return model


def process_slide(slide_path, model, device):
    """
    Runs classifier over every crop in every patch file of a slide,
    then averages feature vectors per predicted tissue class.
    Returns a (NUM_CLASSES, FEATURE_DIM) tensor, or None if no valid patches.
    """
    pt_files = sorted(f for f in os.listdir(slide_path) if f.endswith(".pt"))
    if not pt_files:
        return None

    all_feats = []     # list of 1D tensors (2560,)
    batch_feats = []   # buffer for batched inference

    def flush_batch(buffer_feats, sink_feats, sink_preds):
        if not buffer_feats:
            return
        tensor_batch = torch.stack(buffer_feats).to(device, dtype=torch.float32)  # (B, 2560)
        with torch.no_grad():
            outputs = model(tensor_batch)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
        sink_feats.extend(buffer_feats)
        sink_preds.extend(preds)
        buffer_feats.clear()

    all_preds = []

    for pt_file in pt_files:
        pt_path = os.path.join(slide_path, pt_file)
        try:
            features = torch.load(pt_path, map_location="cpu")
            if not isinstance(features, torch.Tensor):
                raise ValueError(f"Expected tensor, got {type(features)}")

            # Normalize shape: accept (5, 2560) or a lone (2560,) vector
            if features.ndim == 1:
                features = features.unsqueeze(0)
            if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
                print(f"⚠️ Unexpected shape in {pt_file}: {tuple(features.shape)}, skipping")
                continue

            for i in range(features.shape[0]):
                batch_feats.append(features[i])
                all_feats.append(features[i])

        except Exception as e:
            print(f"❌ Error reading {pt_path}: {e}")
            continue

        if len(batch_feats) >= BATCH_SIZE:
            flush_batch(batch_feats, [], all_preds)

    # final partial batch
    flush_batch(batch_feats, [], all_preds)

    if not all_feats:
        return None

    feats = torch.stack(all_feats).numpy().astype(np.float32)   # (N, 2560)
    preds = np.array(all_preds, dtype=np.int64)                 # (N,)

    group_avgs = np.zeros((NUM_CLASSES, FEATURE_DIM), dtype=np.float32)
    for cls in range(NUM_CLASSES):
        mask = preds == cls
        if mask.any():
            group_avgs[cls] = feats[mask].mean(axis=0)
        # else leave as zeros for classes absent in this slide

    return torch.from_numpy(group_avgs)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(device)

    slide_names = sorted(
        d for d in os.listdir(FEATURES_ROOT)
        if os.path.isdir(os.path.join(FEATURES_ROOT, d)) and d != "slide_aggregation"
    )

    n_skipped = 0
    n_saved = 0

    for slide_name in tqdm(slide_names, desc="Slides"):
        save_path = os.path.join(OUTPUT_DIR, f"{slide_name}.pt")

        # Resume check: only this slide's own output matters. If it
        # already exists and passes shape validation, skip re-reading
        # and re-classifying it entirely.
        if SKIP_EXISTING and output_already_valid(save_path):
            n_skipped += 1
            continue

        slide_path = os.path.join(FEATURES_ROOT, slide_name)

        result = process_slide(slide_path, model, device)
        if result is None:
            print(f"⚠️ No valid patches for {slide_name}, skipping")
            continue

        # Atomic write: save to a temp file then rename, so a crash/kill
        # mid-write never leaves a truncated .pt that a later resume run
        # would treat as valid.
        tmp_path = save_path + ".tmp"
        torch.save(result, tmp_path)
        os.replace(tmp_path, save_path)
        n_saved += 1
        print(f"✅ Saved: {save_path}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nDone. Saved {n_saved} slide(s). Skipped {n_skipped} slide(s) with existing valid output.")


if __name__ == "__main__":
    main()