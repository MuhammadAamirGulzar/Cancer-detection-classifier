"""
WSI CZI Processing Pipeline
============================
Pipeline order:
  1. Configuration  — edit all paths/settings here
  2. Step A-1       — per-slide RGB stats CSV (crash-safe: resumes from last completed slide)
  3. Step A-2       — merge CSVs → pixel-rule pre-filter → SVM classification
                      (skipped automatically if merged CSVs already exist on disk)
  4. Step QC        — visual patch inspection (VISUALIZE_PATCHES flag controls this)
  5. Step B         — patch extraction helper (used inline during Step C)
  6. Step C         — CONCH V1 feature extraction → .pt files per patch

Performance notes (see inline comments at each site for details):
  - Step A-1 and Step C each use a single dedicated CZI-reader thread that
    stays one unit of work ahead of the main thread via a small bounded
    queue. This overlaps slow CZI I/O with CPU/GPU compute instead of
    running read -> compute -> read -> compute fully serially, and it keeps
    all CZI reads on ONE thread (pylibCZIrw is not safe for concurrent reads
    against the same czidoc from multiple threads).
  - Step C additionally writes finished .pt files from a small async writer
    pool so disk I/O doesn't block the next GPU batch, and preprocesses
    (FiveCrop) each batch's images across a CPU thread pool instead of a
    single-threaded Python loop.
  - Step C uses autocast (fp16) for the GPU forward pass.
"""

# ══════════════════════════════════════════════════════════════════════════════
# Cell 1 — Configuration (edit everything here)
# ══════════════════════════════════════════════════════════════════════════════
import os
import threading
import queue

# ── Input paths ───────────────────────────────────────────────────────────────
CZI_ROOT          = "/media/dp-psau/dp-psau-wsi/SurGen/S-BIAD1285/Files"
SR386_LABELS_CSV  = "/media/dp-psau/dp-psau-wsi/SurGen/S-BIAD1285/Files/SR386_labels.csv"
SR1482_LABELS_CSV = "/media/dp-psau/dp-psau-wsi/SurGen/S-BIAD1285/Files/SR1482_labels.csv"
SVM_MODEL_PATH    = "/home/mle/Aamir/Azfaar/surgen_processing/svm_model.pkl"

# ── CONCH V1 checkpoint path ──────────────────────────────────────────────────
# Set to None to load remotely from HuggingFace (requires HF_TOKEN)
CONCH_CHECKPOINT  = "/media/dp-psau/Datum/Aamir/Azfaar/surgen_processed/checkpoints/CONCH/pytorch_model.bin"

# ── Excluded slides ───────────────────────────────────────────────────────────
# List of slide filenames or stems to skip from processing entirely.
# If empty, no slides are skipped.
EXCLUDED_SLIDES = []

# ── Output path ───────────────────────────────────────────────────────────────
OUTPUT_ROOT = "/media/dp-psau/Datum/Aamir/Azfaar/surgen_processed/conch-v1"

# ── Patch settings ────────────────────────────────────────────────────────────
PATCH_SIZE = 512   # pixels at target magnification
TARGET_MAG = 20    # desired magnification (x)

# ── Feature extraction ────────────────────────────────────────────────────────
# The Hugging Face token is read from the environment (or a local, git-ignored
# .env file). Never hard-code it: this file is tracked in a public repository.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    # No python-dotenv in this env: read .env by hand (script folder, then up to
    # two parents), without overriding anything already in os.environ.
    _here = os.path.dirname(os.path.abspath(__file__))
    for _root in (_here, os.path.dirname(_here), os.path.dirname(os.path.dirname(_here))):
        _envf = os.path.join(_root, ".env")
        if os.path.isfile(_envf):
            with open(_envf, encoding="utf-8") as _fh:
                for _ln in _fh:
                    _ln = _ln.strip()
                    if _ln and not _ln.startswith("#") and "=" in _ln:
                        _k, _v = _ln.split("=", 1)
                        os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
            break
HF_TOKEN = os.environ.get("HF_TOKEN", "")
DEVICE   = "cuda"   # "cuda" or "cpu"

# ── Pixel filter thresholds (Stage 1 of Step A-2) ────────────────────────────
BLACK_PIXEL_RATIO = 0.20   # >20% black pixels  → edge/fold artifact
WHITE_PIXEL_RATIO = 0.90   # >90% white pixels  → background
WHITE_MEAN_THRESH = 210    # mean R/G/B all above → likely glass background
WHITE_STD_THRESH  = 12     # std  R/G/B all below → no tissue texture

# ── Flags ─────────────────────────────────────────────────────────────────────
# Set to 1 to save QC PNG patches (original behaviour).
# Set to 0 to skip Step QC entirely — no PNGs written, much faster overall.
VISUALIZE_PATCHES = 0

# How many patch-rows to read per CZI call in Step A-1 (row-strip batching).
# Each strip = ROW_BATCH * PATCH_SIZE native pixels tall.
# Lower to 4 or 8 if you see memory warnings.
ROW_BATCH = 16

# Number of patches per GPU forward pass in Step C.
# Increased from 8 -> 16: with autocast (fp16) enabled below, a larger batch
# keeps the GPU busier per kernel launch. Lower this back down if you see
# CUDA OOM errors on your GPU.
FEATURE_BATCH_SIZE = 16

# ── Derived output paths (do not edit) ───────────────────────────────────────
PATCH_METADATA_DIR    = "/media/dp-psau/Datum/Aamir/Azfaar/surgen_processed/h-optimus-1/patch_metadata"
MERGED_METADATA_CSV   = "/media/dp-psau/Datum/Aamir/Azfaar/surgen_processed/h-optimus-1/patch_metadata_merged.csv"
NONWHITE_METADATA_CSV = "/media/dp-psau/Datum/Aamir/Azfaar/surgen_processed/h-optimus-1/patch_metadata_nonwhite.csv"
FEATURES_ROOT         = os.path.join(OUTPUT_ROOT, "features")
QC_FILTERED_DIR       = os.path.join(OUTPUT_ROOT, "qc_patches", "filtered_by_rule")

for _d in [PATCH_METADATA_DIR, FEATURES_ROOT, QC_FILTERED_DIR]:
    os.makedirs(_d, exist_ok=True)

print("Configuration loaded.")
print(f"  CZI root         : {CZI_ROOT}")
print(f"  Output root      : {OUTPUT_ROOT}")
print(f"  Target mag       : {TARGET_MAG}x  |  Patch size: {PATCH_SIZE}")
print(f"  SVM model        : {SVM_MODEL_PATH}")
print(f"  CONCH checkpoint : {CONCH_CHECKPOINT or 'remote (HuggingFace)'}")
print(f"  Visualize patches: {'ON' if VISUALIZE_PATCHES else 'OFF'}")
print(f"  Row batch        : {ROW_BATCH} rows per CZI read")
print(f"  Feature batch    : {FEATURE_BATCH_SIZE} patches/GPU call")
print(f"  Black filter     : >{BLACK_PIXEL_RATIO:.0%} black pixels")
print(f"  White filter     : >{WHITE_PIXEL_RATIO:.0%} white pixels  |  mean>{WHITE_MEAN_THRESH} & std<{WHITE_STD_THRESH}")
print(f"  Excluded slides  : {len(EXCLUDED_SLIDES)} specified")


# ══════════════════════════════════════════════════════════════════════════════
# Step A-1 — Helpers: label lookup & CZI reader
# ══════════════════════════════════════════════════════════════════════════════
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ── Label DataFrames loaded once at first call ────────────────────────────────
_sr386_labels  = None
_sr1482_labels = None


def _load_label_dfs():
    global _sr386_labels, _sr1482_labels
    if _sr386_labels is None:
        _sr386_labels = pd.read_csv(SR386_LABELS_CSV)
        _sr386_labels['case_id'] = _sr386_labels['case_id'].astype(str).str.zfill(3)
        print(f"  Loaded SR386 labels: {len(_sr386_labels)} rows")
    if _sr1482_labels is None:
        _sr1482_labels = pd.read_csv(SR1482_LABELS_CSV)
        _sr1482_labels['case_id'] = _sr1482_labels['case_id'].astype(str).str.zfill(3)
        print(f"  Loaded SR1482 labels: {len(_sr1482_labels)} rows")


def get_slide_label(czi_filename: str) -> str | None:
    _load_label_dfs()
    stem = Path(czi_filename).stem

    match = re.search(r'_T(\d{3})_', stem)
    if not match:
        print(f"  [SKIP] {stem}: could not extract case ID from filename")
        return None
    case_id = match.group(1)

    if stem.upper().startswith("SR386"):
        row = _sr386_labels[_sr386_labels['case_id'] == case_id]
        if row.empty:
            print(f"  [SKIP] {stem}: case_id '{case_id}' not found in SR386 labels CSV")
            return None
        val = row.iloc[0]['mmr_loss_binary']
        if val == 1:   return 'msih'
        elif val == 0: return 'nonmsih'
        print(f"  [SKIP] {stem}: unknown mmr_loss_binary value '{val}'")
        return None

    elif stem.upper().startswith("SR1482"):
        row = _sr1482_labels[_sr1482_labels['case_id'] == case_id]
        if row.empty:
            print(f"  [SKIP] {stem}: case_id '{case_id}' not found in SR1482 labels CSV")
            return None
        val = str(row.iloc[0]['MSI']).strip()
        if val.lower() == 'msi high':          return 'msih'
        elif val.lower() == 'no msi':          return 'nonmsih'
        elif val.lower() in ('not performed', 'insufficient', 'failed'):
            print(f"  [SKIP] {stem}: MSI test not usable ('{val}')"); return None
        elif val.lower() == 'msi low':
            print(f"  [SKIP] {stem}: MSI Low — excluded per protocol"); return None
        print(f"  [SKIP] {stem}: unknown MSI value '{val}'")
        return None

    print(f"  [SKIP] {stem}: filename does not match SR386 or SR1482 prefix")
    return None


def should_exclude_slide(czi_path: str, excluded_list: list[str]) -> bool:
    if not excluded_list:
        return False
    filename = os.path.basename(czi_path)
    stem = Path(filename).stem.lower()
    excluded_stems = {Path(item).stem.lower() for item in excluded_list}
    return stem in excluded_stems


def find_all_czi_files(root: str, excluded_list: list[str] = None) -> list[str]:
    """Recursively find all .czi files under root, optionally filtering out excluded ones."""
    results = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith('.czi'):
                full_path = os.path.join(dirpath, f)
                if excluded_list and should_exclude_slide(full_path, excluded_list):
                    continue
                results.append(full_path)
    return sorted(results)


print("Helpers loaded.")


# ══════════════════════════════════════════════════════════════════════════════
# Step A-1 — CZI reader with magnification handling
# ══════════════════════════════════════════════════════════════════════════════
from pylibCZIrw import czi as pyczi


def open_czi_at_target_mag(czi_path: str, target_mag: int = 20, patch_size: int = 512):
    with pyczi.open_czi(czi_path) as czidoc:
        try:
            native_mag = int(czidoc.metadata['ImageDocument']['Metadata']
                             ['Information']['Instrument']['Objectives']
                             ['Objective']['NominalMagnification'])
        except Exception:
            native_mag = 40

        bbox    = czidoc.total_bounding_box
        X_start = bbox['X'][0]
        Y_start = bbox['Y'][0]
        W       = bbox['X'][1] - bbox['X'][0]
        H       = bbox['Y'][1] - bbox['Y'][0]

        print(f"    bbox raw: X={bbox['X']}, Y={bbox['Y']}")

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


# ══════════════════════════════════════════════════════════════════════════════
# Step A-1 — Generate per-slide RGB stats CSV (row-strip batching, double-buffered I/O)
# ══════════════════════════════════════════════════════════════════════════════
import time
from tqdm import tqdm
import concurrent.futures


def _compute_patch_stats(patches: np.ndarray, patch_size: int) -> tuple:
    avg = patches.mean(axis=(1, 2))
    std = patches.std(axis=(1, 2))
    p8  = patches.astype(np.uint8)
    black_mask = (p8[:,:,:,0] < 15)  & (p8[:,:,:,1] < 15)  & (p8[:,:,:,2] < 15)
    white_mask = (p8[:,:,:,0] > 210) & (p8[:,:,:,1] > 210) & (p8[:,:,:,2] > 210)
    total_px          = patch_size * patch_size
    black_pixel_ratio = black_mask.sum(axis=(1, 2)) / total_px
    white_pixel_ratio = white_mask.sum(axis=(1, 2)) / total_px
    return avg, std, black_pixel_ratio, white_pixel_ratio


def _append_patch_rows(all_rows, patch_num_offset,
                        avg, std, black_pixel_ratio, white_pixel_ratio,
                        abs_row_indices, col_indices, patch_size):
    N       = len(abs_row_indices)
    patch_x = col_indices     * patch_size
    patch_y = abs_row_indices * patch_size
    for i in range(N):
        all_rows.append({
            'patch_number'     : patch_num_offset + i,
            'patch_x'          : int(patch_x[i]),
            'patch_y'          : int(patch_y[i]),
            'avg_R'            : float(avg[i, 0]),
            'avg_G'            : float(avg[i, 1]),
            'avg_B'            : float(avg[i, 2]),
            'std_R'            : float(std[i, 0]),
            'std_G'            : float(std[i, 1]),
            'std_B'            : float(std[i, 2]),
            'black_pixel_ratio': float(black_pixel_ratio[i]),
            'white_pixel_ratio': float(white_pixel_ratio[i]),
        })


def log(msg):
    """Force-flushed print so every line appears immediately."""
    print(msg, flush=True)
    sys.stdout.flush()


def _read_one_strip(czidoc, strip_idx, total_strips, strip_start_row, strip_end_row,
                     n_rows_in_strip, n_patch_cols, patch_size,
                     x_start, y_start, downsample, zoom):
    """
    I/O ONLY for a single strip (full-strip read, or per-row fallback on
    failure). Called exclusively from the dedicated reader thread — never
    from the main thread — because pylibCZIrw is only safe with a single
    thread touching a given `czidoc`. All numpy processing happens back on
    the main thread after this returns, so the two overlap across strips.
    """
    y_native_strip = y_start + (strip_start_row * patch_size * downsample)
    height_native  = n_rows_in_strip * patch_size * downsample
    width_native   = n_patch_cols * patch_size * downsample
    x_native_strip = x_start

    log(f"  [DEBUG] ── Strip {strip_idx+1}/{total_strips} "
        f"(rows {strip_start_row}–{strip_end_row-1}) starting ──")
    log(f"  [DEBUG] Strip {strip_idx+1}: attempting full strip read ...")
    t0 = time.time()
    try:
        strip = czidoc.read(
            roi=(x_native_strip, y_native_strip, width_native, height_native),
            zoom=zoom,
            plane={'C': 0, 'Z': 0, 'T': 0},
        )[:, :, :3]
        log(f"  [DEBUG] Strip {strip_idx+1}: full read OK in {time.time()-t0:.1f}s")
        return {
            'mode': 'full', 'strip': strip, 'strip_idx': strip_idx,
            'strip_start_row': strip_start_row, 'n_rows_in_strip': n_rows_in_strip,
        }

    except Exception as strip_err:
        log(f"  [WARN] Strip {strip_idx+1} rows {strip_start_row}–{strip_end_row-1} "
            f"failed after {time.time()-t0:.1f}s ({strip_err}), retrying row-by-row ...")

        rows_out = []
        for single_row in range(strip_start_row, strip_end_row):
            log(f"    [DEBUG] Row {single_row}: starting read attempt ...")
            t1 = time.time()
            y_native_row  = y_start + (single_row * patch_size * downsample)
            height_single = patch_size * downsample
            try:
                log(f"    [DEBUG] Row {single_row}: calling czidoc.read() ...")
                row_strip = czidoc.read(
                    roi=(x_native_strip, y_native_row, width_native, height_single),
                    zoom=zoom,
                    plane={'C': 0, 'Z': 0, 'T': 0},
                )[:, :, :3]
                log(f"    [DEBUG] Row {single_row}: read returned in {time.time()-t1:.1f}s")
                rows_out.append((single_row, row_strip))
            except Exception as row_err:
                log(f"    [DEBUG] Row {single_row}: exception after {time.time()-t1:.1f}s: {row_err}")
                log(f"      [SKIP] Row {single_row} failed, skipping entire row ...")
                rows_out.append((single_row, None))

        log(f"  [DEBUG] Strip {strip_idx+1}: row-by-row fallback done")
        return {'mode': 'rowfallback', 'strip_idx': strip_idx, 'rows': rows_out}


def _strip_reader_thread(czidoc, n_patch_rows, n_patch_cols, patch_size, row_batch,
                          x_start, y_start, downsample, zoom, out_queue):
    """Runs on its own thread: sequentially reads every strip for the slide
    and pushes results into `out_queue` (maxsize=1 -> one-strip lookahead)."""
    strip_starts  = list(range(0, n_patch_rows, row_batch))
    total_strips  = len(strip_starts)
    for strip_idx, strip_start_row in enumerate(strip_starts):
        strip_end_row   = min(strip_start_row + row_batch, n_patch_rows)
        n_rows_in_strip = strip_end_row - strip_start_row
        result = _read_one_strip(
            czidoc, strip_idx, total_strips, strip_start_row, strip_end_row,
            n_rows_in_strip, n_patch_cols, patch_size,
            x_start, y_start, downsample, zoom,
        )
        out_queue.put(result)
    out_queue.put(None)   # sentinel: no more strips


def compute_patch_rgb_stats_rowbatch(slide_info: dict, patch_size: int,
                                      row_batch: int = 16,
                                      row_timeout: int = 30) -> pd.DataFrame:
    H, W       = slide_info['H'], slide_info['W']
    downsample = slide_info['downsample']
    czi_path   = slide_info['czi_path']
    x_start    = slide_info['X_start']
    y_start    = slide_info['Y_start']
    zoom       = 1.0 / downsample

    H_target     = H // downsample
    W_target     = W // downsample
    n_patch_rows = H_target // patch_size
    n_patch_cols = W_target // patch_size

    log(f"  [DEBUG] slide dims: H_target={H_target}, W_target={W_target}")
    log(f"  [DEBUG] grid: {n_patch_rows} patch rows x {n_patch_cols} patch cols")

    if n_patch_rows == 0 or n_patch_cols == 0:
        log("  [DEBUG] grid is empty, returning early")
        return pd.DataFrame()

    all_rows      = []
    patch_num     = 0
    skipped_rows  = 0
    total_strips  = len(range(0, n_patch_rows, row_batch))

    log(f"  [DEBUG] opening CZI file ...")
    with pyczi.open_czi(czi_path) as czidoc:
        log(f"  [DEBUG] CZI opened OK")
        log(f"  [DEBUG] total strips to process: {total_strips}")

        # Double-buffered I/O: one dedicated reader thread stays a single
        # strip ahead (queue maxsize=1) while this (main) thread runs the
        # numpy stats computation for the previous strip. This overlaps
        # CZI I/O latency with CPU compute instead of doing them serially.
        # Only the reader thread ever calls czidoc.read().
        strip_queue = queue.Queue(maxsize=1)
        reader_thread = threading.Thread(
            target=_strip_reader_thread,
            args=(czidoc, n_patch_rows, n_patch_cols, patch_size, row_batch,
                  x_start, y_start, downsample, zoom, strip_queue),
            daemon=True,
        )
        reader_thread.start()

        while True:
            result = strip_queue.get()
            if result is None:
                break

            strip_idx = result['strip_idx']

            if result['mode'] == 'full':
                strip_start_row = result['strip_start_row']
                n_rows_in_strip = result['n_rows_in_strip']
                N_strip         = n_rows_in_strip * n_patch_cols
                strip           = result['strip']

                strip_H = n_rows_in_strip * patch_size
                strip_W = n_patch_cols    * patch_size
                strip   = strip[:strip_H, :strip_W, :]

                strip_f32 = strip.astype(np.float32)
                grid      = strip_f32.reshape(n_rows_in_strip, patch_size,
                                               n_patch_cols,    patch_size, 3)
                grid      = grid.transpose(0, 2, 1, 3, 4)
                patches   = grid.reshape(N_strip, patch_size, patch_size, 3)

                avg, std, bpr, wpr = _compute_patch_stats(patches, patch_size)

                row_idx, col_idx = np.unravel_index(np.arange(N_strip),
                                                     (n_rows_in_strip, n_patch_cols))
                _append_patch_rows(all_rows, patch_num,
                                   avg, std, bpr, wpr,
                                   row_idx + strip_start_row, col_idx, patch_size)
                patch_num += N_strip
                del strip, strip_f32, grid, patches
                log(f"  [DEBUG] Strip {strip_idx+1}: done, patch_num now={patch_num}")

            else:  # 'rowfallback'
                for single_row, row_strip in result['rows']:
                    if row_strip is None:
                        patch_num    += n_patch_cols
                        skipped_rows += 1
                        log(f"    [DEBUG] Row {single_row}: skipped, patch_num={patch_num}, "
                            f"moving to next row")
                        continue

                    row_strip = row_strip[:patch_size, :n_patch_cols * patch_size, :]
                    row_f32   = row_strip.astype(np.float32)
                    grid      = row_f32.reshape(1, patch_size,
                                                 n_patch_cols, patch_size, 3)
                    grid      = grid.transpose(0, 2, 1, 3, 4)
                    patches   = grid.reshape(n_patch_cols, patch_size, patch_size, 3)

                    avg, std, bpr, wpr = _compute_patch_stats(patches, patch_size)

                    col_idx = np.arange(n_patch_cols)
                    _append_patch_rows(all_rows, patch_num,
                                       avg, std, bpr, wpr,
                                       np.full(n_patch_cols, single_row), col_idx,
                                       patch_size)
                    patch_num += n_patch_cols
                    del row_strip, row_f32, grid, patches
                    log(f"    [DEBUG] Row {single_row}: done OK, patch_num={patch_num}")

            log(f"  [DEBUG] Strip {strip_idx+1}/{total_strips} complete. "
                f"Rows collected so far: {len(all_rows)}, skipped_rows={skipped_rows}")

        reader_thread.join()

    log(f"  [DEBUG] All strips done. Total rows in df: {len(all_rows)}, "
        f"skipped_rows={skipped_rows}")

    if skipped_rows:
        log(f"    [INFO] {skipped_rows} corrupt row(s) skipped "
            f"({skipped_rows * n_patch_cols} patches excluded) for this slide")

    return pd.DataFrame(all_rows)


def run_step_A1(czi_root, output_metadata_dir, patch_size, target_mag, row_batch, excluded_list=None):
    czi_files = find_all_czi_files(czi_root, excluded_list)
    log(f"Found {len(czi_files)} CZI file(s) under {czi_root}")

    for czi_path in tqdm(czi_files, desc="Step A1: RGB stats", unit="slide"):
        filename = os.path.basename(czi_path)
        stem     = Path(filename).stem

        label = get_slide_label(filename)
        if label is None:
            continue

        final_csv = os.path.join(output_metadata_dir, f"{stem}.csv")
        tmp_csv   = os.path.join(output_metadata_dir, f"{stem}.tmp")

        if os.path.exists(final_csv):
            log(f"  [SKIP] {stem}: CSV already exists")
            continue

        if os.path.exists(tmp_csv):
            log(f"  [RESUME] {stem}: removing stale .tmp from previous run, restarting")
            os.remove(tmp_csv)

        log(f"  Processing {stem}  (label={label}) ...")
        try:
            slide_info = open_czi_at_target_mag(czi_path, target_mag, patch_size)
        except Exception as e:
            log(f"  [ERROR] Could not open {stem}: {e}")
            continue

        log(f"    native={slide_info['native_mag']}x  "
            f"target size=({slide_info['W']//slide_info['downsample']} x "
            f"{slide_info['H']//slide_info['downsample']})  "
            f"downsample={slide_info['downsample']}")

        try:
            df = compute_patch_rgb_stats_rowbatch(slide_info, patch_size, row_batch)
        except Exception as e:
            log(f"  [ERROR] Stats failed for {stem}: {e}")
            if os.path.exists(tmp_csv):
                os.remove(tmp_csv)
            continue

        if df.empty:
            log(f"  [WARN] {stem}: no patches extracted (slide too small?)")
            continue

        df.insert(0, 'slide_name', stem)
        df.insert(1, 'label',      label)

        df.to_csv(tmp_csv, index=False)
        os.rename(tmp_csv, final_csv)
        log(f"    → {len(df):,} patches | saved to {final_csv}")


# ══════════════════════════════════════════════════════════════════════════════
# Step A-2 — SVM whiteness classifier
# ══════════════════════════════════════════════════════════════════════════════
import joblib
import glob


def apply_svm_whiteness(svm_input_df: pd.DataFrame, svm_model_path: str) -> pd.DataFrame:
    """
    Run the trained SVM on patches that passed the pixel-rule pre-filter.
    Returns the input DataFrame with a 'white_label' column added.

    The SVM was trained on features: ['avg_G', 'avg_B', 'std_R'].
    white_label = 1  → white/background
    white_label = 0  → tissue
    """
    svm_bundle = joblib.load(svm_model_path)

    if isinstance(svm_bundle, dict):
        svm_model = svm_bundle['model']
        scaler    = svm_bundle.get('scaler', None)
    elif isinstance(svm_bundle, (list, tuple)) and len(svm_bundle) == 2:
        svm_model, scaler = svm_bundle
    else:
        svm_model = svm_bundle
        scaler    = None

    required_features = ['avg_G', 'avg_B', 'std_R']
    for col in required_features:
        if col not in svm_input_df.columns:
            raise ValueError(f"Missing required feature column for SVM: '{col}'")

    X = svm_input_df[required_features].values
    if scaler is not None:
        X = scaler.transform(X)

    y_pred    = svm_model.predict(X)
    result_df = svm_input_df.copy()
    result_df['white_label'] = y_pred

    unique_vals = np.unique(y_pred)
    if len(unique_vals) != 2:
        print(f"  WARNING: SVM predicted only one class: {unique_vals}")

    white_class = max(unique_vals)
    n_white     = (y_pred == white_class).sum()
    n_tissue    = (y_pred != white_class).sum()
    total       = len(y_pred)
    print(f"  SVM results  — white: {n_white:,} ({100*n_white/total:.1f}%)  "
          f"tissue: {n_tissue:,} ({100*n_tissue/total:.1f}%)")

    return result_df


# ══════════════════════════════════════════════════════════════════════════════
# Step A-2 — Run: merge → pixel pre-filter → SVM → save CSVs
# ══════════════════════════════════════════════════════════════════════════════

def tag_bad_patches(df: pd.DataFrame):
    """
    Stage 1: apply three pixel-based rules to flag obvious white/background patches.
    Patches flagged here are labelled white immediately and skipped by the SVM.

    Returns (df_with_labels, bad_mask).
    """
    if 'black_pixel_ratio' not in df.columns or 'white_pixel_ratio' not in df.columns:
        raise ValueError(
            "[ERROR] 'black_pixel_ratio' / 'white_pixel_ratio' columns missing.\n"
            "  → Re-run Step A-1 to regenerate per-slide CSVs."
        )

    is_black       = df['black_pixel_ratio'] > BLACK_PIXEL_RATIO
    is_white_ratio = df['white_pixel_ratio'] > WHITE_PIXEL_RATIO
    is_white_mean  = (
        (df['avg_R'] > WHITE_MEAN_THRESH) & (df['avg_G'] > WHITE_MEAN_THRESH) &
        (df['avg_B'] > WHITE_MEAN_THRESH) & (df['std_R'] < WHITE_STD_THRESH)  &
        (df['std_G'] < WHITE_STD_THRESH)  & (df['std_B'] < WHITE_STD_THRESH)
    )
    bad = is_black | is_white_ratio | is_white_mean

    df = df.copy()
    df['white_label']   = float('nan')
    df['reject_reason'] = ''
    df.loc[bad, 'white_label']             = 1
    df.loc[is_black,       'reject_reason'] = 'black_pixel_ratio'
    df.loc[is_white_ratio, 'reject_reason'] = 'white_pixel_ratio'
    df.loc[is_white_mean,  'reject_reason'] = 'white_mean_std'

    print(f"  Stage 1 — pixel-rule pre-filter:")
    print(f"    black_pixel_ratio  (>{BLACK_PIXEL_RATIO:.0%}) : {is_black.sum():,}")
    print(f"    white_pixel_ratio  (>{WHITE_PIXEL_RATIO:.0%}) : {is_white_ratio.sum():,}")
    print(f"    white_mean_std  fallback           : {is_white_mean.sum():,}")
    print(f"    Total force-labelled white         : {bad.sum():,}")
    print(f"    Passed to SVM (Stage 2)            : {(~bad).sum():,}")
    return df, bad


def save_df_atomic(df: pd.DataFrame, path: str, **kwargs):
    """Save DataFrame to a CSV file atomically via a temporary file."""
    tmp_path = path + ".tmp"
    df.to_csv(tmp_path, **kwargs)
    os.replace(tmp_path, path)


def run_step_A2(patch_metadata_dir, merged_csv_path, nonwhite_csv_path, svm_model_path, excluded_list=None):
    # ── Early exit check: both CSVs exist and match current expected metadata files ──
    if os.path.exists(merged_csv_path) and os.path.exists(nonwhite_csv_path):
        try:
            csv_files = sorted(glob.glob(os.path.join(patch_metadata_dir, '*.csv')))
            expected_stems = {
                Path(f).stem for f in csv_files
                if not should_exclude_slide(f, excluded_list)
            }

            merged_df_cols = pd.read_csv(merged_csv_path, usecols=['slide_name'])
            merged_stems = set(merged_df_cols['slide_name'].unique())

            if expected_stems == merged_stems and expected_stems:
                print(f"Step A-2 — merged CSVs already exist on disk and match the metadata files. Skipping.")
                print(f"  Merged CSV   : {merged_csv_path}")
                print(f"  Nonwhite CSV : {nonwhite_csv_path}")
                nonwhite_df = pd.read_csv(nonwhite_csv_path)
                print(f"  Loaded {len(nonwhite_df):,} non-white (tissue) patches.")
                return nonwhite_df
            else:
                print("Step A-2 — metadata files do not match existing merged CSV or merged CSV is empty. Re-running merge + filter + SVM.")
        except Exception as e:
            print(f"Step A-2 — error checking existing merged CSV ({e}). Re-running merge + filter + SVM.")

    # ── 1. Merge all per-slide CSVs (excluding excluded slides) ──────────────────────────
    csv_files = sorted(glob.glob(os.path.join(patch_metadata_dir, '*.csv')))
    csv_files = [f for f in csv_files if not should_exclude_slide(f, excluded_list)]

    if not csv_files:
        raise FileNotFoundError(
            f"[ERROR] No per-slide CSVs found (or all were excluded) in: {patch_metadata_dir}\n"
            f"  → Run Step A-1 first."
        )

    print(f"Merging {len(csv_files)} per-slide CSV(s) ...")
    dfs = []
    for csv_path in csv_files:
        try:
            dfs.append(pd.read_csv(csv_path))
        except Exception as e:
            print(f"  [WARN] Could not read {csv_path}: {e} — skipping")

    if not dfs:
        raise RuntimeError("[ERROR] All per-slide CSVs failed to load.")

    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"  → {len(merged_df):,} total patches from {len(dfs)} slide(s)")

    # ── 2. Stage 1: pixel-rule pre-filter ───────────────────────────────────
    merged_df, bad_mask = tag_bad_patches(merged_df)

    filtered_df = merged_df[bad_mask][[
        'slide_name', 'patch_number', 'patch_x', 'patch_y', 'reject_reason',
        'black_pixel_ratio', 'white_pixel_ratio',
        'avg_R', 'avg_G', 'avg_B', 'std_R', 'std_G', 'std_B'
    ]].copy()
    filtered_csv = os.path.join(QC_FILTERED_DIR, "rule_filtered_patches.csv")
    save_df_atomic(filtered_df, filtered_csv, index=False)
    print(f"  Pixel-rule rejected patch list → {filtered_csv}")

    # ── 3. Stage 2: SVM on patches that passed the pixel rules ──────────────
    svm_input_df = merged_df[merged_df['white_label'].isna()].drop(
        columns=['white_label', 'reject_reason']
    )
    print(f"\n  Stage 2 — SVM on {len(svm_input_df):,} patches ...")
    svm_result_df = apply_svm_whiteness(svm_input_df, svm_model_path)

    # ── 4. Merge SVM labels back into main DataFrame ─────────────────────────
    merged_df.loc[merged_df['white_label'].isna(), 'white_label'] = \
        svm_result_df['white_label'].values
    merged_df['white_label'] = merged_df['white_label'].astype(int)

    # ── 5. Save merged CSV (all patches + white_label) ───────────────────────
    save_df = merged_df.drop(columns=['reject_reason'])
    save_df_atomic(save_df, merged_csv_path, index=False)

    # ── 6. Save nonwhite CSV (tissue patches only) ───────────────────────────
    nonwhite_class = save_df['white_label'].min()
    nonwhite_df    = save_df[save_df['white_label'] == nonwhite_class].reset_index(drop=True)
    save_df_atomic(nonwhite_df, nonwhite_csv_path, index=False)

    total    = len(save_df)
    n_white  = (save_df['white_label'] != nonwhite_class).sum()
    n_tissue = len(nonwhite_df)

    print(f"\nStep A-2 complete.")
    print(f"  Total patches      : {total:,}")
    print(f"  White (excluded)   : {n_white:,}  ({100*n_white/total:.1f}%)")
    print(f"  Tissue (kept)      : {n_tissue:,}  ({100*n_tissue/total:.1f}%)")
    print(f"  Merged CSV         → {merged_csv_path}")
    print(f"  Non-white CSV      → {nonwhite_csv_path}")
    return nonwhite_df


# ══════════════════════════════════════════════════════════════════════════════
# Step QC — Visual patch inspection
# ══════════════════════════════════════════════════════════════════════════════
from PIL import Image


def run_step_QC(czi_root, output_root, merged_csv, nonwhite_csv, patch_size, target_mag, excluded_list=None):
    MAX_PATCHES_PER_CLASS = 2000
    MAX_FILTERED_PREVIEW  = 2000

    if VISUALIZE_PATCHES == 0:
        print("[Step QC] VISUALIZE_PATCHES=0 — skipping all PNG patch saving.")
        return

    print("[Step QC] VISUALIZE_PATCHES=1 — saving QC patches ...")

    for path, name in [(merged_csv, 'Merged'), (nonwhite_csv, 'Non-white')]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"[ERROR] {name} CSV not found: {path}\n  → Run Step A-2 first.")

    merged_df   = pd.read_csv(merged_csv)
    nonwhite_df = pd.read_csv(nonwhite_csv)

    if 'white_label' not in merged_df.columns:
        raise ValueError("[ERROR] 'white_label' column missing.\n  → Run Step A-2 first.")

    nonwhite_class = merged_df['white_label'].min()
    white_df_all   = merged_df[merged_df['white_label'] != nonwhite_class]
    tissue_df_all  = merged_df[merged_df['white_label'] == nonwhite_class]

    print(f"Total patches  : {len(merged_df):,}")
    print(f"White (excluded): {len(white_df_all):,}")
    print(f"Tissue (kept)  : {len(tissue_df_all):,}")
    print(f"Sampling up to {MAX_PATCHES_PER_CLASS} per class ...\n")

    white_sample  = white_df_all.sample(min(MAX_PATCHES_PER_CLASS, len(white_df_all)),
                                         random_state=42).reset_index(drop=True)
    tissue_sample = tissue_df_all.sample(min(MAX_PATCHES_PER_CLASS, len(tissue_df_all)),
                                          random_state=42).reset_index(drop=True)

    qc_white_dir    = os.path.join(output_root, "qc_patches", "white")
    qc_nonwhite_dir = os.path.join(output_root, "qc_patches", "nonwhite")
    os.makedirs(qc_white_dir,    exist_ok=True)
    os.makedirs(qc_nonwhite_dir, exist_ok=True)

    czi_index = {Path(p).stem: p for p in find_all_czi_files(czi_root, excluded_list)}

    def save_patches_to_dir(sample_df: pd.DataFrame, out_dir: str, label_tag: str):
        """Open each CZI once and save all sampled patches from it as PNGs."""
        for slide_stem, group in tqdm(sample_df.groupby('slide_name'),
                                       desc=f"Saving {label_tag}", unit="slide"):
            if slide_stem not in czi_index:
                print(f"  [WARN] CZI not found for '{slide_stem}' — skipping")
                continue
            czi_path = czi_index[slide_stem]
            try:
                slide_info = open_czi_at_target_mag(czi_path, target_mag, patch_size)
            except Exception as e:
                print(f"  [ERROR] Cannot open {slide_stem}: {e}")
                continue

            downsample  = slide_info['downsample']
            x_start     = slide_info['X_start']
            y_start     = slide_info['Y_start']
            zoom        = 1.0 / downsample
            size_native = patch_size * downsample

            with pyczi.open_czi(czi_path) as czidoc:
                for _, row in group.iterrows():
                    x_native = x_start + (int(row['patch_x']) * downsample)
                    y_native = y_start + (int(row['patch_y']) * downsample)
                    try:
                        raw = czidoc.read(
                            roi=(x_native, y_native, size_native, size_native),
                            zoom=zoom, plane={'C': 0, 'Z': 0, 'T': 0},
                        )
                        img   = Image.fromarray(raw[:, :, :3].astype('uint8')).convert('RGB')
                        fname = f"{slide_stem}_p{int(row['patch_number'])}.png"
                        img.save(os.path.join(out_dir, fname))
                    except Exception as e:
                        tqdm.write(f"    [ERROR] patch {row['patch_number']} of {slide_stem}: {e}")

    save_patches_to_dir(white_sample,  qc_white_dir,    "white")
    save_patches_to_dir(tissue_sample, qc_nonwhite_dir, "non-white (tissue)")

    print(f"\nWhite QC     → {qc_white_dir}  ({len(white_sample)} patches)")
    print(f"Nonwhite QC  → {qc_nonwhite_dir}  ({len(tissue_sample)} patches)")

    qc_filtered_dir = os.path.join(output_root, "qc_patches", "filtered_by_rule")
    filtered_csv    = os.path.join(qc_filtered_dir, "rule_filtered_patches.csv")
    if not os.path.exists(filtered_csv):
        print("\n[SKIP] rule_filtered_patches.csv not found — run Step A-2 first.")
    else:
        filtered_df = pd.read_csv(filtered_csv)
        print(f"\nSaving filtered-by-rule previews ({len(filtered_df):,} total filtered patches) ...")
        for reason, group in filtered_df.groupby('reject_reason'):
            reason_dir = os.path.join(qc_filtered_dir, reason)
            os.makedirs(reason_dir, exist_ok=True)
            sample = group.sample(min(MAX_FILTERED_PREVIEW, len(group)), random_state=42)
            save_patches_to_dir(sample, reason_dir, f"filtered/{reason}")
            print(f"  {reason}: saved {len(sample)} sample patches → {reason_dir}")
        print(f"\nFiltered QC  → {qc_filtered_dir}")
        print(f"  Subfolders : {[d for d in os.listdir(qc_filtered_dir) if os.path.isdir(os.path.join(qc_filtered_dir, d))]}")


# ══════════════════════════════════════════════════════════════════════════════
# Step B — Patch extraction helper (used inline during Step C)
# ══════════════════════════════════════════════════════════════════════════════

def extract_patch(slide_array: np.ndarray, x: int, y: int,
                  patch_size: int) -> Image.Image:
    """
    Extract a single patch from a slide numpy array at position (x, y).
    Returns a PIL Image in RGB mode of size (patch_size, patch_size).
    """
    patch = slide_array[y : y + patch_size, x : x + patch_size]
    return Image.fromarray(patch.astype(np.uint8)).convert("RGB")


print("Step B helper loaded.  Patches will be extracted in-memory during Step C.")


# ══════════════════════════════════════════════════════════════════════════════
# Step C — Feature extraction with CONCH V1 (batched GPU inference, pipelined I/O)
# ══════════════════════════════════════════════════════════════════════════════
import torch
from torchvision import transforms
from huggingface_hub import login
from conch.open_clip_custom import create_model_from_pretrained
from concurrent.futures import ThreadPoolExecutor


class ConchV1Extractor:
    """
    Loads CONCH V1 and extracts FiveCrop features from patch images.

    Feature dimension : 512  (CONCH V1 ViT-B/16 image encoder output)
    Output shape      : (5, 512) per patch  /  (B, 5, 512) for batch

    extract()       — single patch → (5, 512)   [kept for compatibility]
    extract_batch() — list of patches → (B, 5, 512)  [much faster]
    """

    def __init__(self, device: str = 'cuda', hf_token: str = None,
                 checkpoint: str = None, prep_workers: int = None):
        if hf_token:
            login(token=hf_token)

        if device == 'cuda' and not torch.cuda.is_available():
            print("  [WARN] CUDA requested but not available — falling back to CPU")
            device = 'cpu'

        self.device = torch.device(device)
        print(f"  Loading CONCH V1 on {self.device} ...")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")

        if checkpoint:
            # Load from local checkpoint
            print(f"  Checkpoint: {checkpoint}")
            model, preprocess = create_model_from_pretrained(
                'conch_ViT-B-16', checkpoint
            )
        else:
            # Load remotely from HuggingFace
            print("  Checkpoint: remote (hf_hub:MahmoodLab/conch)")
            model, preprocess = create_model_from_pretrained(
                'conch_ViT-B-16', 'hf_hub:MahmoodLab/conch',
                hf_auth_token=hf_token
            )

        self.model = model.to(self.device)
        self.model.eval()

        # preprocess is the standard CONCH eval transform
        self._eval_transform = preprocess

        # FiveCrop: take five 256×256 crops from a 512×512 patch, transform each
        self._fivecrop = transforms.Compose([
            transforms.FiveCrop(256),
            transforms.Lambda(
                lambda crops: torch.stack([self._eval_transform(c) for c in crops])
            )
        ])

        # Persistent CPU thread pool for FiveCrop preprocessing. Each call to
        # extract_batch() farms the per-image crop/normalize work out across
        # these threads instead of a single-threaded Python loop, so CPU
        # preprocessing for a batch doesn't serialize on one core while the
        # GPU (and other cores) sit idle.
        self._prep_workers = prep_workers or min(8, (os.cpu_count() or 4))
        self._prep_pool = ThreadPoolExecutor(max_workers=self._prep_workers)

        print(f"  CONCH V1 ready.  Feature dim: 512  |  CPU preprocess workers: {self._prep_workers}")

    def extract(self, image: Image.Image) -> torch.Tensor:
        """Single-patch extract — returns (5, 512)."""
        crops = self._fivecrop(image.convert("RGB")).to(self.device)
        with torch.inference_mode():
            features = self.model.encode_image(
                crops, proj_contrast=False, normalize=False
            )   # (5, 512)
        return features.cpu()

    def extract_batch(self, images: list) -> torch.Tensor:
        """
        Batch extract — runs all patches in one GPU forward pass.
        Returns (B, 5, 512).  Keeps GPU near 100% utilisation.

        Each patch's 5 crops are stacked into a single (B*5, 3, 224, 224) tensor
        so the model sees one large batch instead of B tiny ones.
        """
        B = len(images)

        # Parallelize the CPU-side FiveCrop + normalize step across threads
        # instead of a serial list comprehension.
        crop_tensors = list(self._prep_pool.map(
            lambda img: self._fivecrop(img.convert("RGB")), images
        ))
        batch = torch.cat(crop_tensors, dim=0).to(self.device)   # (B*5, 3, 224, 224)

        autocast_enabled = (self.device.type == 'cuda')
        with torch.inference_mode():
            with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                 enabled=autocast_enabled):
                features = self.model.encode_image(
                    batch, proj_contrast=False, normalize=False
                )   # (B*5, 512)

        return features.cpu().reshape(B, 5, 512)   # (B, 5, 512)


def _read_patch_worker(czidoc, x_native, y_native, size_native, zoom):
    """Read one patch from an open czidoc — called from the single dedicated
    reader thread only (see _slide_patch_reader)."""
    raw = czidoc.read(
        roi=(x_native, y_native, size_native, size_native),
        zoom=zoom, plane={'C': 0, 'Z': 0, 'T': 0},
    )
    return Image.fromarray(raw[:, :, :3].astype('uint8')).convert('RGB')


def _slide_patch_reader(czidoc, rows, x_start, y_start, downsample, size_native, zoom,
                         slide_stem, out_queue):
    """
    Runs on ONE dedicated thread for the whole slide. pylibCZIrw is not safe
    for concurrent reads against the same czidoc from multiple threads, so
    all patch reads for this slide happen serially here, while the main
    thread overlaps GPU inference and async disk writes for previously-read
    batches. Pushes (patch_id, image, error) tuples; error is None on
    success. Pushes a final `None` sentinel when done.
    """
    for row in rows:
        x_native = x_start + (int(row['patch_x']) * downsample)
        y_native = y_start + (int(row['patch_y']) * downsample)
        patch_id = f"{slide_stem}_{int(row['patch_number'])}"
        try:
            img = _read_patch_worker(czidoc, x_native, y_native, size_native, zoom)
            out_queue.put((patch_id, img, None))
        except Exception as e:
            out_queue.put((patch_id, None, e))
    out_queue.put(None)


def _save_feature_atomic(feat_dir: str, patch_id: str, feat: "torch.Tensor") -> str:
    """
    Atomic save used by the async writer pool.

    `feat` is a (5, 512) slice taken by iterating over a (B, 5, 512) batch
    tensor, i.e. it is a VIEW into the larger batch tensor's storage, not an
    independent tensor. `.clone()` is essential here: without it, torch.save
    would serialize the entire underlying storage of the batch tensor (not
    just this patch's 5x512 slice), bloating every .pt file to the size of
    the whole batch. This is the same view-bloat bug already fixed elsewhere
    in this pipeline. `.clone()` forces a fresh, exactly-sized, contiguous
    tensor before saving.
    """
    final_path = os.path.join(feat_dir, patch_id + '.pt')
    tmp_path   = final_path + '.tmp'
    torch.save(feat.to(torch.float16).clone(), tmp_path)
    os.replace(tmp_path, final_path)   # atomic on Linux
    return patch_id


def _validate_and_clean_pt_files(feat_dir: str) -> set:
    """
    Scan all .pt files in feat_dir. Delete any that are corrupt (can't be loaded).
    Returns the set of valid patch IDs (stems of surviving .pt files).
    """
    valid_ids = set()
    removed   = 0
    for fname in os.listdir(feat_dir):
        if not fname.endswith('.pt'):
            continue
        fpath = os.path.join(feat_dir, fname)
        try:
            t = torch.load(fpath, map_location='cpu', weights_only=True)
            # Sanity-check shape: expect (5, 512) stored as float16
            if not isinstance(t, torch.Tensor) or t.ndim != 2 or t.shape != (5, 512):
                raise ValueError(f"Unexpected shape {t.shape}")
            valid_ids.add(Path(fname).stem)
        except Exception as e:
            tqdm.write(f"    [CORRUPT] Removing {fname}: {e}")
            try:
                os.remove(fpath)
            except OSError:
                pass
            removed += 1
    if removed:
        tqdm.write(f"  [CLEAN] Removed {removed} corrupt .pt file(s) from {feat_dir}")
    return valid_ids


def run_step_C(czi_root, nonwhite_csv, features_root,
               extractor, patch_size, target_mag, batch_size: int = 8, excluded_list=None):
    """
    Pipelined + prefetched feature extraction — safe to interrupt and resume.

    Crash-safety guarantees
    ───────────────────────
    0. Fast skip     : if the on-disk .pt count already matches the expected
                       patch count for a slide, the slide is skipped
                       immediately — no CZI open, no per-file validation.
    1. Atomic saves  : every .pt is written to a .tmp file first, then
                       os.rename()'d into place.  A half-written file is
                       always a .tmp — it can never masquerade as a valid .pt.
    2. Corrupt scan  : only runs when the patch count doesn't match (a
                       partial/interrupted slide) — existing .pt files are
                       loaded and shape-checked, and any that fail are
                       deleted so they will be re-extracted on this run.
    3. Per-patch IDs : resume granularity is individual patches, not batches
                       or slides.  Only the patches whose .pt is missing or
                       corrupt are reprocessed.
    4. Stale .tmp    : any leftover .tmp files from a prior crash are deleted
                       at slide startup before the corrupt scan runs.

    Performance
    ───────────
    A single dedicated reader thread streams patches for the slide into a
    bounded queue (respecting the pylibCZIrw single-thread-per-czidoc
    constraint). The main thread assembles GPU batches from that queue,
    runs inference (CPU preprocessing inside extract_batch is itself
    thread-parallel, and the GPU forward pass uses autocast), and hands
    finished features off to a small async writer pool so disk I/O never
    blocks the next GPU batch. All three stages — I/O, GPU compute, and
    disk writes — overlap across batches instead of running serially.
    """
    if not os.path.exists(nonwhite_csv):
        raise FileNotFoundError(
            f"[ERROR] Non-white CSV not found: {nonwhite_csv}\n"
            f"  → Run Step A-1 and Step A-2 first."
        )

    nonwhite_df  = pd.read_csv(nonwhite_csv)
    slide_groups = nonwhite_df.groupby('slide_name')
    print(f"Found {len(slide_groups)} slide(s) with non-white patches.")

    czi_index = {Path(p).stem: p for p in find_all_czi_files(czi_root, excluded_list)}

    for slide_stem, group in tqdm(slide_groups, desc="Step C: Feature extraction", unit="slide"):
        feat_dir = os.path.join(features_root, slide_stem)
        os.makedirs(feat_dir, exist_ok=True)

        # ── Fast path: patch count already matches → skip without touching CZI
        #    or validating individual .pt files ───────────────────────────────
        n_expected = len(group)
        n_existing = sum(1 for f in os.listdir(feat_dir) if f.endswith('.pt'))
        if n_existing == n_expected:
            tqdm.write(f"  [SKIP] All {n_expected} patch(es) already processed for {slide_stem}")
            continue

        # ── 1. Remove any stale .tmp files from a previous interrupted run ──
        stale_tmps = [f for f in os.listdir(feat_dir) if f.endswith('.tmp')]
        for stale in stale_tmps:
            os.remove(os.path.join(feat_dir, stale))
        if stale_tmps:
            tqdm.write(f"  [CLEAN] Removed {len(stale_tmps)} stale .tmp file(s) for {slide_stem}")

        # ── 2. Count mismatch (partial/corrupt run) → validate & delete bad .pt files ──
        existing = _validate_and_clean_pt_files(feat_dir)

        if slide_stem not in czi_index:
            print(f"  [WARN] CZI not found for slide '{slide_stem}' — skipping")
            continue

        czi_path = czi_index[slide_stem]
        print(f"  Loading {slide_stem} ...")
        try:
            slide_info = open_czi_at_target_mag(czi_path, target_mag, patch_size)
        except Exception as e:
            print(f"  [ERROR] Cannot open {slide_stem}: {e}")
            continue

        downsample  = slide_info['downsample']
        x_start     = slide_info['X_start']
        y_start     = slide_info['Y_start']
        zoom        = 1.0 / downsample
        size_native = patch_size * downsample

        patches_to_process = group.reset_index(drop=True)
        patches_to_process['_id'] = (
            patches_to_process['slide_name'] + '_'
            + patches_to_process['patch_number'].astype(str)
        )
        patches_to_process = patches_to_process[
            ~patches_to_process['_id'].isin(existing)
        ]

        if patches_to_process.empty:
            tqdm.write(f"  [SKIP] All patches already processed for {slide_stem}")
            continue

        tqdm.write(f"  Resuming {slide_stem}: "
                   f"{len(patches_to_process)} patch(es) remaining "
                   f"({len(existing)} already done)")

        rows    = patches_to_process.to_dict('records')
        n_total = len(rows)
        n_done  = 0

        # Bounded queue: reader thread stays a few batches ahead of GPU/writer
        # so CZI I/O overlaps with GPU inference + disk writes instead of the
        # three stages running fully serially, as in the previous version.
        read_queue = queue.Queue(maxsize=batch_size * 4)

        with pyczi.open_czi(czi_path) as czidoc, \
             ThreadPoolExecutor(max_workers=2) as writer_pool:

            reader_thread = threading.Thread(
                target=_slide_patch_reader,
                args=(czidoc, rows, x_start, y_start, downsample, size_native, zoom,
                      slide_stem, read_queue),
                daemon=True,
            )
            reader_thread.start()

            write_futures = []
            batch_ids, batch_imgs = [], []
            pbar = tqdm(total=n_total, desc=f"  Patches: {slide_stem}",
                        unit="patch", leave=False)

            def flush_batch():
                nonlocal batch_ids, batch_imgs
                if not batch_imgs:
                    return
                try:
                    batch_features = extractor.extract_batch(batch_imgs)   # (B, 5, 512)
                except Exception as e:
                    tqdm.write(f"    [ERROR] batch inference at "
                               f"{len(batch_imgs)} patches: {e}")
                    batch_ids, batch_imgs = [], []
                    return

                for pid, feat in zip(batch_ids, batch_features):
                    write_futures.append(
                        writer_pool.submit(_save_feature_atomic, feat_dir, pid, feat)
                    )
                pbar.update(len(batch_ids))
                batch_ids, batch_imgs = [], []

            while True:
                item = read_queue.get()
                if item is None:
                    break
                patch_id, img, err = item
                if err is not None:
                    tqdm.write(f"    [ERROR] patch {patch_id}: {err}")
                    continue
                batch_ids.append(patch_id)
                batch_imgs.append(img)
                if len(batch_imgs) >= batch_size:
                    flush_batch()

            flush_batch()   # final partial batch
            reader_thread.join()
            pbar.close()

            # Writer pool submissions are async relative to GPU batches above;
            # drain them here to count successes and surface any save errors.
            for fut in write_futures:
                try:
                    fut.result()
                    n_done += 1
                except Exception as e:
                    tqdm.write(f"    [ERROR] saving a patch for {slide_stem}: {e}")

        tqdm.write(
            f"  ✓ {slide_stem}: saved {n_done}/{n_total} feature files → {feat_dir}"
        )

# ══════════════════════════════════════════════════════════════════════════════
# Main — run all steps in order
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Step A-1: per-slide RGB stats
    run_step_A1(CZI_ROOT, PATCH_METADATA_DIR, PATCH_SIZE, TARGET_MAG, ROW_BATCH, EXCLUDED_SLIDES)

    # Step A-2: merge + filter + SVM
    # (automatically skipped if patch_metadata_merged.csv and
    #  patch_metadata_nonwhite.csv already exist on disk and match processed metadata files)
    run_step_A2(
        patch_metadata_dir = PATCH_METADATA_DIR,
        merged_csv_path    = MERGED_METADATA_CSV,
        nonwhite_csv_path  = NONWHITE_METADATA_CSV,
        svm_model_path     = SVM_MODEL_PATH,
        excluded_list      = EXCLUDED_SLIDES,
    )

    # Step QC: optional visual inspection
    run_step_QC(CZI_ROOT, OUTPUT_ROOT, MERGED_METADATA_CSV, NONWHITE_METADATA_CSV,
                PATCH_SIZE, TARGET_MAG, EXCLUDED_SLIDES)

    # Step C: feature extraction (loads CONCH V1 model first)
    extractor = ConchV1Extractor(
        device     = DEVICE,
        hf_token   = HF_TOKEN,
        checkpoint = CONCH_CHECKPOINT,
    )
    run_step_C(
        czi_root      = CZI_ROOT,
        nonwhite_csv  = NONWHITE_METADATA_CSV,
        features_root = FEATURES_ROOT,
        extractor     = extractor,
        patch_size    = PATCH_SIZE,
        target_mag    = TARGET_MAG,
        batch_size    = FEATURE_BATCH_SIZE,
        excluded_list = EXCLUDED_SLIDES,
    )

    print("\nPipeline complete.")
    print(f"  Feature files are in : {FEATURES_ROOT}")
    print(f"  Non-white patch CSV  : {NONWHITE_METADATA_CSV}")
    print(f"  Merged metadata CSV  : {MERGED_METADATA_CSV}")