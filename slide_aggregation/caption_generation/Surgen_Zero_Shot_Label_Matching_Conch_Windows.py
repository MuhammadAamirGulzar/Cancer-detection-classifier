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
  1. Configuration       — edit all paths/settings here (Windows paths, matching
                            the layout used in combined_pipeline_final_error_checks_v6.py)
  2. Load CONCH model    — load_conch_model()
  3. Load prompts        — load_class_prompts() (classnames + templates from JSON)
  4. Zero-shot weights   — build_zeroshot_weights() / load existing .pt, controlled by REBUILD_WEIGHTS
  5. CZI download        — download_czi_if_missing() fetches a slide's .czi from the EBI
                            BioStudies mirror via aria2c on demand (same source/mechanism
                            used in combined_pipeline_final_error_checks_v6.py's download_batch()),
                            with a disk-space guard and optional delete-after-processing.
  6. CZI helper          — open_czi_at_target_mag() (matches the main extraction pipeline)
  7. Run matching        — run_zero_shot_matching() downloads (if needed) + reads patches
                            from CZIs, scores them, writes results to a gzip CSV with
                            resume + periodic checkpointing, then deletes the CZI.

Usage:
    python Surgen_Zero_Shot_Label_Matching_Conch.py
"""

# ══════════════════════════════════════════════════════════════════════════════
# ░░  IMPORTS  ░░
# ══════════════════════════════════════════════════════════════════════════════

import os
import gzip
import json
import csv
import shutil
import subprocess
import sys
import time
import threading
import traceback
import faulthandler
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
# Same checkpoint location used for CONCH v1 in combined_pipeline_final_error_checks_v6.py.
CHECKPOINT_PATH  = r"D:\Aamir Gulzar\KSA_project2\Cancer-detection-classifier\slide_aggregation\caption_generation\checkpoints\CONCH\pytorch_model.bin"
# NOTE: no equivalent of these two paths exists in combined_pipeline_final_error_checks_v6.py
# (that script only does feature extraction, not zero-shot label matching), so they're
# placed alongside the checkpoint/results folders below. Adjust if you keep them elsewhere.
PROMPT_FILE      = r"D:\Aamir Gulzar\KSA_project2\Cancer-detection-classifier\slide_aggregation\caption_generation\config_files\config_15_classes.json"
FORCE_IMAGE_SIZE = 224

# ── Zero-shot classifier weights ─────────────────────────────────────────────
ZEROSHOT_WEIGHTS_PATH = r"D:\Aamir Gulzar\KSA_project2\Cancer-detection-classifier\slide_aggregation\caption_generation\Conch_zeroshot_weights_15_classes.pt"
# Set True to recompute weights from class_prompts (e.g. if captions/templates changed).
# Set False to just load the existing .pt file.
# If the weights path above doesn't exist yet, this will build+save fresh
# weights automatically on first run regardless of this flag. You can leave it
# False afterwards to reuse the cached 15-class weights.
REBUILD_WEIGHTS = False

# ── Input: patch metadata + source CZIs ──────────────────────────────────────
# CZI_DIR is both where downloaded WSIs land and where they're read from —
# same role/value as CZI_DIR in combined_pipeline_final_error_checks_v6.py.
CZI_DIR      = r"F:\CZI_Files"
# Same NONWHITE_METADATA_CSV location as combined_pipeline_final_error_checks_v6.py
# (OUTPUT_BASE\patch_metadata_nonwhite.csv). Same slide_name/patch_number/
# patch_x/patch_y schema — compatible as-is.
NONWHITE_CSV = r"D:\Aamir Gulzar\KSA_project2\surgen_data\patch_metadata_nonwhite.csv"

# ── Output ────────────────────────────────────────────────────────────────────
# .csv.gz — pandas infers gzip compression from the extension automatically
# for both to_csv(mode='a') and read_csv().
OUTPUT_CSV = r"D:\Aamir Gulzar\KSA_project2\Cancer-detection-classifier\slide_aggregation\caption_generation\Results\classification_results_surgen_15_classes.csv.gz"

# ── Patch coordinate settings (must match the extraction pipeline) ──────────
PATCH_SIZE = 512   # pixels at target magnification
TARGET_MAG = 20    # desired magnification (x)

# ── Checkpointing ─────────────────────────────────────────────────────────────
CHECKPOINT_INTERVAL = 10000   # save progress every N patches

# ── CZI download settings (mirrors combined_pipeline_final_error_checks_v6.py) ─
# EBI BioStudies mirror the WSIs are hosted on.
_EBI_BASE_URL = "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/285/S-BIAD1285/Files"
# aria2c executable — same path used in combined_pipeline_final_error_checks_v6.py.
ARIA2C_EXE = r"C:\Users\datainsight\AppData\Local\Microsoft\WinGet\Packages\aria2.aria2_Microsoft.Winget.Source_8wekyb3d8bbwe\aria2-1.37.0-win-64bit-build1\aria2c.exe"
# Drive to monitor for free space before each download (same as combined pipeline).
DISK_TO_CHECK = "D:\\"
MIN_FREE_GB   = 20      # halt if DISK_TO_CHECK has less than this free
# Delete each slide's CZI once every patch for that slide has been scored, to
# keep disk usage bounded over the ~11.87M-patch run (same idea as
# delete_batch_czis() in combined_pipeline_final_error_checks_v6.py).
DELETE_CZI_AFTER_PROCESSING = True

# ── Hang watchdog ─────────────────────────────────────────────────────────────
# If no patch finishes processing within this many seconds, the run is treated
# as frozen (this happens when pylibCZIrw's native .read() call stalls on a
# particular ROI/slide and never returns or raises). The watchdog dumps stack
# traces for every thread (so you can see exactly where it's stuck) and force-
# exits so the script never just sits there silently. Re-running picks up
# where it left off via the resume/checkpoint logic.
WATCHDOG_TIMEOUT_SEC = 180

# ══════════════════════════════════════════════════════════════════════════════
# ░░  SHARED UTILITIES  ░░
# ══════════════════════════════════════════════════════════════════════════════

def log(msg: str):
    print(msg, flush=True)
    sys.stdout.flush()


class HangWatchdog:
    """
    Background thread that detects a frozen main loop and forces an exit
    instead of letting the script sit there indefinitely.

    The main loop calls .heartbeat("description of what it's about to do")
    right before anything that could block (e.g. a CZI read). If no new
    heartbeat arrives within `timeout` seconds, this thread assumes the
    process is stuck inside that call, prints:
      - how long it's been stuck and what it was doing
      - a full stack trace of every thread (faulthandler), which pinpoints
        the exact line it's frozen on
    and then calls os._exit(1). A hard exit is necessary here because the
    hang is almost always inside a native (C/C++) call that never returns
    control to Python, so the main thread can't be asked nicely to stop.
    """

    def __init__(self, timeout: int = WATCHDOG_TIMEOUT_SEC, poll_interval: int = 5):
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._last_beat = time.monotonic()
        self._where = "startup"
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def heartbeat(self, where: str):
        with self._lock:
            self._last_beat = time.monotonic()
            self._where = where

    def _watch(self):
        while not self._stop_event.wait(self.poll_interval):
            with self._lock:
                elapsed = time.monotonic() - self._last_beat
                where = self._where
            if elapsed > self.timeout:
                log("\n" + "=" * 78)
                log(f"[WATCHDOG] No progress for {elapsed:.0f}s (limit {self.timeout}s).")
                log(f"[WATCHDOG] Last known activity: {where}")
                log("[WATCHDOG] Full thread stack traces (to pinpoint the exact hang):")
                faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                sys.stderr.flush()
                log(
                    "[WATCHDOG] Treating this as frozen — most likely a stalled "
                    "native pylibCZIrw .read() call on this ROI/slide (or a "
                    "stuck disk/network read). Forcing the process to exit now "
                    "instead of hanging forever. Re-run the script: the resume "
                    "logic will skip everything already checkpointed and, if "
                    "the same patch hangs again, you'll see the exact slide/"
                    "coordinates above so you can flag or skip that patch."
                )
                log("=" * 78)
                sys.stdout.flush()
                os._exit(1)


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
# ░░  STEP 5 — CZI download (on demand, via aria2c — same source/mechanism as  ░░
# ░░           download_batch() in combined_pipeline_final_error_checks_v6.py) ░░
# ══════════════════════════════════════════════════════════════════════════════

def czi_url_for_stem(stem: str) -> str | None:
    """
    Build the EBI BioStudies download URL for a slide stem, using the same
    prefix → subfolder inference as build_wsi_list() in
    combined_pipeline_final_error_checks_v6.py (SR1482 → SR1482_WSIs,
    SR386 → SR386_WSIs). Returns None for an unrecognised prefix.
    """
    stem_upper = stem.upper()
    if stem_upper.startswith("SR1482"):
        subfolder = "SR1482_WSIs"
    elif stem_upper.startswith("SR386"):
        subfolder = "SR386_WSIs"
    else:
        return None
    return f"{_EBI_BASE_URL}/{subfolder}/{stem}.czi"


def get_free_gb(path: str) -> float:
    """Return free space in GB for the drive that contains `path`."""
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def check_disk_space_or_halt(threshold_gb: float = MIN_FREE_GB):
    free_gb = get_free_gb(DISK_TO_CHECK)
    if free_gb < threshold_gb:
        log(f"[HALT] Only {free_gb:.1f} GB free on {DISK_TO_CHECK} "
            f"(threshold: {threshold_gb} GB). Stopping to protect the drive.")
        sys.exit(1)


_ARIA2C_RESOLVED = None


def _resolve_aria2c() -> str:
    global _ARIA2C_RESOLVED
    if _ARIA2C_RESOLVED:
        return _ARIA2C_RESOLVED
    resolved = shutil.which(ARIA2C_EXE) or (ARIA2C_EXE if os.path.isfile(ARIA2C_EXE) else None)
    if resolved is None:
        log(f"\n[FATAL] aria2c not found!")
        log(f"  Tried  : '{ARIA2C_EXE}'")
        log(f"  Fix    : set ARIA2C_EXE to the full path of aria2c.exe")
        sys.exit(1)
    _ARIA2C_RESOLVED = resolved
    return resolved


def download_czi_if_missing(stem: str, watchdog: "HangWatchdog | None" = None) -> str | None:
    """
    Ensure CZI_DIR\\<stem>.czi exists locally, downloading it via aria2c from
    the EBI mirror if necessary (skips download if already present — same
    behaviour as download_batch() in combined_pipeline_final_error_checks_v6.py).
    Returns the local path, or None if the slide couldn't be resolved/downloaded.
    """
    out_path = os.path.join(CZI_DIR, stem + ".czi")
    if os.path.exists(out_path):
        return out_path

    url = czi_url_for_stem(stem)
    if url is None:
        log(f"  [SKIP DOWNLOAD] Unrecognised slide prefix for '{stem}' — cannot build URL")
        return None

    check_disk_space_or_halt()
    os.makedirs(CZI_DIR, exist_ok=True)
    aria2c_resolved = _resolve_aria2c()

    log(f"  [DOWNLOAD] {stem}.czi <- {url}")
    cmd = [
        aria2c_resolved,
        "-x", "16", "-s", "16", "-c",
        "-d", CZI_DIR,
        "-o", stem + ".czi",
        url,
    ]

    t0 = time.time()
    proc = subprocess.Popen(cmd)
    # Poll instead of a blocking subprocess.run() so we can keep heartbeating
    # the watchdog during long downloads (large CZIs can easily exceed
    # WATCHDOG_TIMEOUT_SEC) without falsely triggering a hang-exit.
    while proc.poll() is None:
        if watchdog is not None:
            watchdog.heartbeat(f"downloading '{stem}.czi' via aria2c")
        time.sleep(5)

    elapsed = time.time() - t0
    if proc.returncode == 0 and os.path.exists(out_path):
        size_gb = os.path.getsize(out_path) / (1024 ** 3)
        log(f"  [DOWNLOAD OK] {stem}.czi — {size_gb:.2f} GB in {elapsed:.1f}s")
        return out_path
    else:
        log(f"  [ERROR] aria2c exited with code {proc.returncode} downloading {stem}.czi "
            f"after {elapsed:.1f}s")
        return None


def delete_czi(stem: str):
    """Delete CZI_DIR\\<stem>.czi (and any stale aria2c control file) to free disk space."""
    path = os.path.join(CZI_DIR, stem + ".czi")
    if os.path.exists(path):
        try:
            os.remove(path)
            log(f"  [CLEANUP] Deleted {stem}.czi")
        except Exception as e:
            log(f"  [WARN] Could not delete {stem}.czi: {e}")
    ctrl = path + ".aria2"
    if os.path.exists(ctrl):
        try:
            os.remove(ctrl)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# ░░  STEP 6 — Zero-shot matching over patches read directly from CZIs  ░░
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

    # ── Index CZI files already on disk (others are downloaded on demand below) ─
    czi_index = find_all_czi_files(CZI_DIR)
    log(f"Found {len(czi_index)} CZI file(s) already present under {CZI_DIR}")

    # ── Process patch-by-patch, grouped by slide so each CZI is opened once ────
    csv_data    = []
    n_processed = 0

    slide_groups = patches_to_process.groupby('slide_name')

    watchdog = HangWatchdog(timeout=WATCHDOG_TIMEOUT_SEC)
    watchdog.start()
    log(f"[WATCHDOG] Active — will force-exit with diagnostics if no patch "
        f"completes within {WATCHDOG_TIMEOUT_SEC}s.")

    try:
        with torch.inference_mode():
            for slide_stem, group in tqdm(slide_groups, desc="Slides", unit="slide"):

                watchdog.heartbeat(f"resolving CZI for slide '{slide_stem}'")

                czi_path = czi_index.get(slide_stem)
                if czi_path is None:
                    czi_path = download_czi_if_missing(slide_stem, watchdog=watchdog)
                    if czi_path is None:
                        log(f"  [SKIP] Could not obtain CZI for slide '{slide_stem}'")
                        continue
                    czi_index[slide_stem] = czi_path

                watchdog.heartbeat(f"opening slide '{slide_stem}' ({czi_path})")
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

                        watchdog.heartbeat(
                            f"reading patch {patch_id} from '{slide_stem}' "
                            f"(roi=({x_native},{y_native},{size_native},{size_native}))"
                        )

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

                        watchdog.heartbeat(f"finished patch {patch_id} from '{slide_stem}'")

                        n_processed += 1
                        if n_processed % CHECKPOINT_INTERVAL == 0:
                            df = pd.DataFrame(csv_data)
                            write_header = not os.path.exists(OUTPUT_CSV)
                            df.to_csv(OUTPUT_CSV, mode='a', header=write_header, index=False)
                            csv_data = []
                            log(f"Saved checkpoint at {n_processed} patches")

                if DELETE_CZI_AFTER_PROCESSING:
                    # Flush pending results first so a crash right after deletion
                    # can never lose already-scored patches for this slide.
                    if csv_data:
                        df = pd.DataFrame(csv_data)
                        write_header = not os.path.exists(OUTPUT_CSV)
                        df.to_csv(OUTPUT_CSV, mode='a', header=write_header, index=False)
                        csv_data = []
                    delete_czi(slide_stem)
                    czi_index.pop(slide_stem, None)

    except Exception:
        # Any real (non-hang) exception: flush whatever we have, print the
        # full traceback so the failure is visible, then exit non-zero
        # instead of leaving the process in a silent/ambiguous state.
        log("\n[FATAL] Unhandled exception — saving partial progress before exit:")
        traceback.print_exc()
        if csv_data:
            df = pd.DataFrame(csv_data)
            write_header = not os.path.exists(OUTPUT_CSV)
            df.to_csv(OUTPUT_CSV, mode='a', header=write_header, index=False)
            log(f"[FATAL] Saved {len(csv_data)} pending result(s) to {OUTPUT_CSV} before exiting.")
        watchdog.stop()
        sys.exit(1)

    watchdog.stop()

    if csv_data:
        df = pd.DataFrame(csv_data)
        write_header = not os.path.exists(OUTPUT_CSV)
        df.to_csv(OUTPUT_CSV, mode='a', header=write_header, index=False)

    log(f"\nResults saved to '{OUTPUT_CSV}'")
    log(f"Total new patches processed: {n_processed}")


# ══════════════════════════════════════════════════════════════════════════════
# ░░  STEP 7 — Entry count validation  ░░
# ══════════════════════════════════════════════════════════════════════════════

def count_entries(csv_file: str) -> int:
    """Count the number of rows (excluding header) in a CSV file (gzip-aware)."""
    opener = gzip.open if str(csv_file).lower().endswith('.gz') else open
    with opener(csv_file, 'rt', newline='', encoding='utf-8') as file:
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
    log(f"  CZI dir          : {CZI_DIR}")
    log(f"  Delete after use : {DELETE_CZI_AFTER_PROCESSING}")
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