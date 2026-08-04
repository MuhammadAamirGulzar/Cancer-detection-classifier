import os
import torch
import numpy as np
from tqdm import tqdm

# ========================= CONFIGURATION =========================
FEATURES_ROOT = "/media/dp-psau/Datum/Aamir/Azfaar/surgen_processed/conch-v1/features"
OUTPUT_DIR    = "/media/dp-psau/Datum/Aamir/Azfaar/surgen_processed/conch-v1/features/slide_aggregation/Tissue_Type_Clustering/conch-v1"
# ASSUMPTION: following the same naming convention as ann_crc100k_Hoptimus.pth —
# confirm/replace with the actual checkpoint path for the CONCH v1 tissue classifier.
MODEL_PATH    = "/home/mle/Aamir/Azfaar/surgen_processing/ann_crc100k_ConchV1.pth"

NUM_CLASSES = 9
FEATURE_DIM = 512
BATCH_SIZE  = 8
# ====================================================================


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

    all_feats = []     # list of 1D tensors (512,)
    batch_feats = []   # buffer for batched inference

    def flush_batch(buffer_feats, sink_feats, sink_preds):
        if not buffer_feats:
            return
        tensor_batch = torch.stack(buffer_feats).to(device,dtype=torch.float32)  # (B, 512)
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

            # Normalize shape: accept (5, 512) or a lone (512,) vector
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

    feats = torch.stack(all_feats).numpy().astype(np.float32)   # (N, 512)
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

    for slide_name in tqdm(slide_names, desc="Slides"):
        slide_path = os.path.join(FEATURES_ROOT, slide_name)

        result = process_slide(slide_path, model, device)
        if result is None:
            print(f"⚠️ No valid patches for {slide_name}, skipping")
            continue

        save_path = os.path.join(OUTPUT_DIR, f"{slide_name}.pt")
        torch.save(result, save_path)
        print(f"✅ Saved: {save_path}")

        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
