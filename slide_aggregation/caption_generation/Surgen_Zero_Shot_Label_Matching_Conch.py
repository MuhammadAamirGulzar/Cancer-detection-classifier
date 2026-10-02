"""
SurGen Zero-Shot Label Matching (CONCH)
=========================================
Matches each patch in a non-white patch metadata CSV to one caption/class
using a zero-shot classifier built from CONCH V1 text embeddings.

Patches are NOT read from saved PNGs — they are read directly from the
source WSI CZI files on the fly, using the same patch_size/target_mag
coordinate scheme used during feature extraction (patch_x, patch_y from
the nonwhite CSV map to native CZI pixel coordinates).

Pipeline order:
  1. Configuration       — edit all paths/settings here
  2. Load CONCH model    — load_conch_model()
  3. Load prompts        — load_class_prompts() (classnames + templates from JSON)
  4. Zero-shot weights   — build_zeroshot_weights() / load existing .pt, controlled by REBUILD_WEIGHTS
  5. CZI helper          — open_czi_at_target_mag() (matches the main extraction pipeline)
  6. Run matching        — run_zero_shot_matching() reads patches from CZIs, scores them,
                            writes results to CSV with resume + periodic checkpointing

Usage:
    python Surgen_Zero_Shot_Label_Matching_Conch.py
"""

# ══════════════════════════════════════════════════════════════════════════════
# ░░  IMPORTS  ░░
# ══════════════════════════════════════════════════════════════════════════════

import os
import json
import csv
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
from conch.downstream.zeroshot_path import zero_shot_classifier
from pylibCZIrw import czi as pyczi


# ══════════════════════════════════════════════════════════════════════════════
# ░░  CONFIGURATION — edit everything here  ░░
# ══════════════════════════════════════════════════════════════════════════════

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ── CONCH model / prompts ────────────────────────────────────────────────────
CHECKPOINT_PATH  = r"/media/dp-psau/Datum/Aamir/Azfaar/surgen_processed/checkpoints/CONCH/pytorch_model.bin"
PROMPT_FILE      = r"/home/mle/Aamir/Azfaar/surgen_processing/conch/config_15_classes.json"
FORCE_IMAGE_SIZE = 224

# ── Zero-shot classifier weights ─────────────────────────────────────────────
ZEROSHOT_WEIGHTS_PATH = r"/home/mle/Aamir/Azfaar/surgen_processing/conch/Conch_zeroshot_weights_15_classes.pt"
# Set True to recompute weights from class_prompts (e.g. if captions/templates changed).
# Set False to just load the existing .pt file.
# If the weights path above doesn't exist yet, this will build+save fresh
# weights automatically on first run regardless of this flag. You can leave it
# False afterwards to reuse the cached 15-class weights.
REBUILD_WEIGHTS = False

# ── Input: patch metadata + source CZIs ──────────────────────────────────────
CZI_ROOT     = r"/media/dp-psau/dp-psau-wsi1/SurGen/S-BIAD1285/Files/"   # folder containing the .czi WSIs
# Updated to the nonwhite CSV produced by the UNI2-h pipeline
# (same slide_name/patch_number/patch_x/patch_y schema — compatible as-is).
NONWHITE_CSV = r"/media/dp-psau/Datum/Aamir/Azfaar/surgen_processed/h-optimus-1/patch_metadata_nonwhite.csv"

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_CSV = r"Results/classification_results_surgen_15_classes.csv"

# ── Patch coordinate settings (must match the extraction pipeline) ──────────
PATCH_SIZE = 512   # pixels at target magnification
TARGET_MAG = 20    # desired magnification (x)

# ── Checkpointing ─────────────────────────────────────────────────────────────
CHECKPOINT_INTERVAL = 10000   # save progress every N patches

# ══════════════════════════════════════════════════════════════════════════════
# ░░  SHARED UTILITIES  ░░
# ══════════════════════════════════════════════════════════════════════════════

def log(msg: str):
    print(msg, flush=True)
    sys.stdout.flush()


# ══════════════════════════════════════════════════════════════════════════════
# ░░  STEP 1 — Load CONCH model  ░░
# ══════════════════════════════════════════════════════════════════════════════

def load_conch_model():
    log(f"Loading CONCH V1 on {DEVICE} ...")
    model, preprocess = create_model_from_pretrained(
        model_cfg='conch_ViT-B-16',
        checkpoint_path=CHECKPOINT_PATH,
        device=DEVICE,
        force_image_size=FORCE_IMAGE_SIZE,
    )
    model.eval()
    log("CONCH V1 ready.")
    return model, preprocess


# ══════════════════════════════════════════════════════════════════════════════
# ░░  STEP 2 — Load class prompts  ░░
# ══════════════════════════════════════════════════════════════════════════════

def load_class_prompts(prompt_file: str):
    with open(prompt_file) as f:
        prompt_data = json.load(f)['0']

    classnames = prompt_data['classnames']
    templates  = prompt_data['templates']

    # Create idx_to_class based on JSON order (ensures consistent ordering)
    idx_to_class = {i: cls_name for i, cls_name in enumerate(classnames.keys())}
    n_classes    = len(classnames)

    # List of lists (each class → its synonyms)
    classnames_text = [classnames[idx_to_class[idx]] for idx in range(n_classes)]

    log(f"Loaded {n_classes} class(es) from {prompt_file}")
    for class_idx, classname in enumerate(classnames_text):
        log(f'  {class_idx}: {classname}')

    return classnames_text, templates, idx_to_class


# ══════════════════════════════════════════════════════════════════════════════
# ░░  STEP 3 — Zero-shot classifier weights  ░░
# ══════════════════════════════════════════════════════════════════════════════

def build_zeroshot_weights(model, classnames_text, templates):
    log("Building zero-shot classifier weights (averaging synonyms × templates) ...")
    zeroshot_weights = zero_shot_classifier(model, classnames_text, templates, device=DEVICE)
    log(f"  Zero-shot weights shape: {tuple(zeroshot_weights.shape)}")
    return zeroshot_weights


def save_zeroshot_weights(weights, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(weights, path)
    log(f"  Saved zeroshot weights to {path}")


def load_zeroshot_weights(path: str):
    weights = torch.load(path, map_location=DEVICE)
    log(f"  Loaded weights shape: {tuple(weights.shape)}")
    return weights


def get_zeroshot_weights(model, classnames_text, templates):
    if REBUILD_WEIGHTS or not os.path.exists(ZEROSHOT_WEIGHTS_PATH):
        weights = build_zeroshot_weights(model, classnames_text, templates)
        save_zeroshot_weights(weights, ZEROSHOT_WEIGHTS_PATH)
        return weights
    else:
        log(f"Loading existing zero-shot weights from {ZEROSHOT_WEIGHTS_PATH}")
        weights = load_zeroshot_weights(ZEROSHOT_WEIGHTS_PATH)
        # Safety check: catches exactly the stale-weights-from-old-class-count
        # scenario described above, instead of failing deep inside a matmul.
        if weights.shape[1] != len(classnames_text):
            raise ValueError(
                f"Loaded zero-shot weights have {weights.shape[1]} classes but "
                f"the prompt file defines {len(classnames_text)}. The cached "
                f".pt file at {ZEROSHOT_WEIGHTS_PATH} is stale for this prompt "
                f"file — delete it or set REBUILD_WEIGHTS=True."
            )
        return weights


# ══════════════════════════════════════════════════════════════════════════════
# ░░  STEP 4 — CZI helper (matches the main extraction pipeline)  ░░
# ══════════════════════════════════════════════════════════════════════════════

def open_czi_at_target_mag(czi_path: str, target_mag: int = 20, patch_size: int = 512):
    with pyczi.open_czi(czi_path) as czidoc:
        try:
            native_mag = int(
                czidoc.metadata['ImageDocument']['Metadata']
                ['Information']['Instrument']['Objectives']
                ['Objective']['NominalMagnification']
            )
        except Exception:
            native_mag = 40

        bbox    = czidoc.total_bounding_box
        X_start = bbox['X'][0]
        Y_start = bbox['Y'][0]
        W       = bbox['X'][1] - bbox['X'][0]
        H       = bbox['Y'][1] - bbox['Y'][0]

        downsample = max(1, native_mag // target_mag)
        level_mag  = native_mag // downsample

    return {
        'czi_path'  : czi_path,
        'W'         : W, 'H': H,
        'X_start'   : X_start, 'Y_start': Y_start,
        'native_mag': native_mag,
        'level_mag' : level_mag,
        'downsample': downsample,
    }


def find_all_czi_files(root: str) -> dict:
    """Index all .czi files under root, keyed by filename stem."""
    czi_index = {}
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith('.czi'):
                czi_index[Path(f).stem] = os.path.join(dirpath, f)
    return czi_index


# ══════════════════════════════════════════════════════════════════════════════
# ░░  STEP 5 — Zero-shot matching over patches read directly from CZIs  ░░
# ══════════════════════════════════════════════════════════════════════════════

def run_zero_shot_matching(model, preprocess, loaded_weights, idx_to_class):
    # ── Load patch metadata (slide_name, patch_number, patch_x, patch_y) ──────
    nonwhite_df = pd.read_csv(NONWHITE_CSV)
    log(f"Loaded {len(nonwhite_df):,} patches from {NONWHITE_CSV}")

    # Build a unique patch_id the same way the extraction pipeline does, for resume-matching
    nonwhite_df['patch_id'] = (
        nonwhite_df['slide_name'] + '_' + nonwhite_df['patch_number'].astype(str)
    )

    # ── Resume: skip already-processed patches ─────────────────────────────────
    if os.path.exists(OUTPUT_CSV):
        processed_patch_ids = set(pd.read_csv(OUTPUT_CSV)['patch_id'].tolist())
        log(f"Resuming — {len(processed_patch_ids)} patches already done")
    else:
        processed_patch_ids = set()
        os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)

    patches_to_process = nonwhite_df[~nonwhite_df['patch_id'].isin(processed_patch_ids)]
    log(f"Patches to process after resume filter: {len(patches_to_process):,}")

    if patches_to_process.empty:
        log("Nothing to process — all patches already matched.")
        return

    # ── Index available CZI files on disk ───────────────────────────────────────
    czi_index = find_all_czi_files(CZI_ROOT)
    log(f"Found {len(czi_index)} CZI file(s) under {CZI_ROOT}")

    # ── Process patch-by-patch, grouped by slide so each CZI is opened once ────
    csv_data    = []
    n_processed = 0

    slide_groups = patches_to_process.groupby('slide_name')

    with torch.inference_mode():
        for slide_stem, group in tqdm(slide_groups, desc="Slides", unit="slide"):

            if slide_stem not in czi_index:
                log(f"  [SKIP] CZI not found for slide '{slide_stem}'")
                continue

            czi_path = czi_index[slide_stem]
            try:
                slide_info = open_czi_at_target_mag(czi_path, TARGET_MAG, PATCH_SIZE)
            except Exception as e:
                log(f"  [ERROR] Cannot open {slide_stem}: {e}")
                continue

            downsample  = slide_info['downsample']
            x_start     = slide_info['X_start']
            y_start     = slide_info['Y_start']
            zoom        = 1.0 / downsample
            size_native = PATCH_SIZE * downsample

            rows = group.to_dict('records')

            with pyczi.open_czi(czi_path) as czidoc:
                for row in tqdm(rows, desc=f"  {slide_stem}", unit="patch", leave=False):
                    patch_id = row['patch_id']
                    x_native = x_start + (int(row['patch_x']) * downsample)
                    y_native = y_start + (int(row['patch_y']) * downsample)

                    try:
                        raw = czidoc.read(
                            roi=(x_native, y_native, size_native, size_native),
                            zoom=zoom, plane={'C': 0, 'Z': 0, 'T': 0},
                        )
                        image = Image.fromarray(raw[:, :, :3].astype('uint8')).convert('RGB')

                        image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
                        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16):
                            image_features = model.encode_image(image_tensor)
                            sim_scores     = (image_features @ loaded_weights).squeeze(0)
                        top_score, top_idx = torch.max(sim_scores, dim=0)
                        top_class = idx_to_class[top_idx.item()]

                        csv_data.append({
                            'patch_id'    : patch_id,
                            'slide_name'  : slide_stem,
                            'patch_number': int(row['patch_number']),
                            'label'       : top_class,
                            'score'       : float(top_score.item())
                        })

                        # Free per-patch tensors promptly since we're in a long loop
                        del image_tensor, image_features, sim_scores

                    except Exception as e:
                        log(f"Error processing {patch_id}: {e}")
                        continue

                    n_processed += 1
                    if n_processed % CHECKPOINT_INTERVAL == 0:
                        df = pd.DataFrame(csv_data)
                        write_header = not os.path.exists(OUTPUT_CSV)
                        df.to_csv(OUTPUT_CSV, mode='a', header=write_header, index=False)
                        csv_data = []
                        log(f"Saved checkpoint at {n_processed} patches")

    if csv_data:
        df = pd.DataFrame(csv_data)
        write_header = not os.path.exists(OUTPUT_CSV)
        df.to_csv(OUTPUT_CSV, mode='a', header=write_header, index=False)

    log(f"\nResults saved to '{OUTPUT_CSV}'")
    log(f"Total new patches processed: {n_processed}")


# ══════════════════════════════════════════════════════════════════════════════
# ░░  STEP 6 — Entry count validation  ░░
# ══════════════════════════════════════════════════════════════════════════════

def count_entries(csv_file: str) -> int:
    """Count the number of rows (excluding header) in a CSV file."""
    with open(csv_file, newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        row_count = sum(1 for row in reader) - 1   # subtract header row
    return row_count


def validate_output_entry_count():
    if not os.path.exists(OUTPUT_CSV):
        log(f"[VALIDATE] Output CSV not found yet: {OUTPUT_CSV}")
        return

    n_entries = count_entries(OUTPUT_CSV)
    log(f"[VALIDATE] {OUTPUT_CSV} → {n_entries:,} entries")

    if os.path.exists(NONWHITE_CSV):
        n_expected = count_entries(NONWHITE_CSV)
        log(f"[VALIDATE] {NONWHITE_CSV} → {n_expected:,} entries expected")
        if n_entries != n_expected:
            log(f"[VALIDATE] [WARN] Mismatch: {n_expected - n_entries:,} patch(es) missing from output")
        else:
            log("[VALIDATE] Entry counts match.")


# ══════════════════════════════════════════════════════════════════════════════
# ░░  MAIN  ░░
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log("╔══════════════════════════════════════════════════════════════════════╗")
    log("║       SurGen Zero-Shot Label Matching (CONCH)                       ║")
    log("╚══════════════════════════════════════════════════════════════════════╝")
    log(f"  Device           : {DEVICE}")
    log(f"  Checkpoint       : {CHECKPOINT_PATH}")
    log(f"  Prompt file      : {PROMPT_FILE}")
    log(f"  Rebuild weights  : {REBUILD_WEIGHTS}")
    log(f"  CZI root         : {CZI_ROOT}")
    log(f"  Nonwhite CSV     : {NONWHITE_CSV}")
    log(f"  Output CSV       : {OUTPUT_CSV}")
    log(f"  Patch size       : {PATCH_SIZE}  |  Target mag: {TARGET_MAG}x")

    model, preprocess = load_conch_model()
    classnames_text, templates, idx_to_class = load_class_prompts(PROMPT_FILE)
    loaded_weights = get_zeroshot_weights(model, classnames_text, templates)

    run_zero_shot_matching(model, preprocess, loaded_weights, idx_to_class)

    validate_output_entry_count()

    log("\nDone.")


if __name__ == "__main__":
    main()