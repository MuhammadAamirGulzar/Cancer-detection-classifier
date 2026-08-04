"""
SurGen Batch Processing Orchestrator
======================================
Downloads WSIs in batches, runs the full pipeline (Step A-1 → A-2 → Step C)
for each selected model, deletes the CZI files, then rsyncs features to a
remote machine over SSH via Tailscale.

The WSI catalogue is built dynamically from surgen_labels.csv — any slide with
label_desc == -1 is excluded automatically. No hardcoded filename lists needed.

Usage:
    python surgen_batch_pipeline.py
"""
import os
from dotenv import load_dotenv
load_dotenv()
# DO NOT RUN THIS SCRIPT DIRECTLY, USE run_pipeline_supervisor.py instead to catch errors

# ══════════════════════════════════════════════════════════════════════════════
# ░░  IMPORTS  ░░
# ══════════════════════════════════════════════════════════════════════════════

import os
import re
import sys
import gc
import glob
import shutil
import subprocess
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
# ░░  CONFIGURATION — edit everything here  ░░
# ══════════════════════════════════════════════════════════════════════════════

# ── Model selection ───────────────────────────────────────────────────────────
# Remove a model name from this list to skip it entirely during feature
# extraction. The shared pre-processing (Step A-1 / A-2) always runs once.
ACTIVE_MODELS = [  # "uni2-h", "h-optimus-1", "conch1-5", "conch-v1", "virchow2"
    "conch1-5",
    "virchow2"
]

# ── Download / storage settings ───────────────────────────────────────────────
BATCH_SIZE    = 10      # WSIs per batch
MIN_FREE_GB   = 50     # halt if D: drive has less than this free after a batch
DISK_TO_CHECK = "D:\\"  # drive to monitor for free space

# ── Paths ─────────────────────────────────────────────────────────────────────
CZI_DIR     = r"D:\Aamir Gulzar\KSA_project2\surgen_data\CZI_Files"
OUTPUT_BASE = r"D:\Aamir Gulzar\KSA_project2\surgen_data"
FEATURES_BASE = r"D:\Aamir Gulzar\KSA_project2\surgen_data\surgen_processed"

# ── Labels CSV ────────────────────────────────────────────────────────────────
# Columns: WSI_Id (slide stem), label_desc (0=nonmsih, 1=msih, -1=exclude)
SURGEN_LABELS_CSV = r"D:\Aamir Gulzar\KSA_project2\surgen_data\surgen_labels.csv"

# ── Model weights / checkpoints ───────────────────────────────────────────────
SVM_MODEL_PATH      = r"D:\Aamir Gulzar\KSA_project2\Cancer-detection-classifier\Complete_Pipeline\svm_model.pkl"
CONCH_V1_CHECKPOINT = r"D:\Aamir Gulzar\KSA_project2\Cancer-detection-classifier\slide_aggregation\caption_generation\checkpoints\CONCH\pytorch_model.bin"

# ── HuggingFace token ─────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ── aria2c executable ─────────────────────────────────────────────────────────
ARIA2C_EXE = r"C:\Users\datainsight\AppData\Local\Microsoft\WinGet\Packages\aria2.aria2_Microsoft.Winget.Source_8wekyb3d8bbwe\aria2-1.37.0-win-64bit-build1\aria2c.exe"

# ── Remote sync settings (rsync over SSH via Tailscale) ──────────────────────
REMOTE_SYNC_ENABLED = False
REMOTE_USER         = "mle"                                        # SSH username on remote machine
REMOTE_HOST         = "100.116.197.88"                             # Tailscale IP of remote machine
REMOTE_BASE         = "/media/dp-psau/Datum/Aamir/Azfaar/surgen_processed"  # destination root on remote
SSH_KEY             = r"C:\Users\datainsight\.ssh\id_ed25519"      # path to your private key
REMOTE_SSH_PORT     = 22                                           # SSH port
RSYNC_EXE           = "wsl"   # sync runs via WSL rsync; this value is kept for reference only

# ── Patch / extraction settings ───────────────────────────────────────────────
PATCH_SIZE         = 512
TARGET_MAG         = 20
ROW_BATCH          = 16
FEATURE_BATCH_SIZE = 16
DEVICE             = "cuda"
VISUALIZE_PATCHES  = 0    # 0 = skip QC PNG generation (saves space & time)
VIRCHOW2_CROP_SIZE = 256

# ── Pixel filter thresholds ───────────────────────────────────────────────────
BLACK_PIXEL_RATIO = 0.20
WHITE_PIXEL_RATIO = 0.90
WHITE_MEAN_THRESH = 210
WHITE_STD_THRESH  = 12

# ── Output paths per model ────────────────────────────────────────────────────
_HOPT_ROOT            = os.path.join(FEATURES_BASE, "h-optimus-1")
PATCH_METADATA_DIR    = os.path.join(OUTPUT_BASE, "patch_metadata")
MERGED_METADATA_CSV   = os.path.join(OUTPUT_BASE, "patch_metadata_merged.csv")
NONWHITE_METADATA_CSV = os.path.join(OUTPUT_BASE, "patch_metadata_nonwhite.csv")

MODEL_OUTPUT_ROOTS = {
    "h-optimus-1": _HOPT_ROOT,
    "conch1-5"   : os.path.join(FEATURES_BASE, "conch1-5"),
    "conch-v1"   : os.path.join(FEATURES_BASE, "conch-v1"),
    "uni2-h"     : os.path.join(FEATURES_BASE, "uni2-h"),
    "virchow2"   : os.path.join(FEATURES_BASE, "virchow2"),
}

# ── EBI base URL ─────────────────────────────────────────────────────────────
_BASE_URL = "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/285/S-BIAD1285/Files"


# ══════════════════════════════════════════════════════════════════════════════
# ░░  WSI CATALOGUE  ░░
# ══════════════════════════════════════════════════════════════════════════════

def build_wsi_list(labels_csv: str) -> list:
    """
    Read surgen_labels.csv and return (filename, url) pairs for every slide
    whose label_desc != -1. The EBI subfolder is inferred from the slide
    name prefix (SR1482 → SR1482_WSIs, SR386 → SR386_WSIs).
    """
    df = pd.read_csv(labels_csv)

    if "WSI_Id" not in df.columns or "label_desc" not in df.columns:
        raise ValueError(
            f"Expected columns 'WSI_Id' and 'label_desc' in {labels_csv}. "
            f"Found: {df.columns.tolist()}"
        )

    valid_stems = df.loc[df["label_desc"] != -1, "WSI_Id"].tolist()

    wsis    = []
    skipped = 0
    for stem in valid_stems:
        stem_upper = stem.upper()
        if stem_upper.startswith("SR1482"):
            subfolder = "SR1482_WSIs"
        elif stem_upper.startswith("SR386"):
            subfolder = "SR386_WSIs"
        else:
            print(f"[WARN] Unrecognised prefix for '{stem}' — skipping.", flush=True)
            skipped += 1
            continue
        filename = stem + ".czi"
        url      = f"{_BASE_URL}/{subfolder}/{filename}"
        wsis.append((filename, url))

    print(
        f"[CATALOGUE] {len(wsis)} valid WSI(s) loaded from {labels_csv} "
        f"({len(df) - len(valid_stems)} excluded with label=-1"
        + (f", {skipped} skipped due to unknown prefix" if skipped else "")
        + ")",
        flush=True,
    )
    return wsis


# ══════════════════════════════════════════════════════════════════════════════
# ░░  SHARED UTILITIES  ░░
# ══════════════════════════════════════════════════════════════════════════════

def log(msg: str):
    print(msg, flush=True)
    sys.stdout.flush()


# ══════════════════════════════════════════════════════════════════════════════
# ░░  DISK SPACE CHECK  ░░
# ══════════════════════════════════════════════════════════════════════════════

def get_free_gb(path: str) -> float:
    """Return free space in GB for the drive that contains `path`."""
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def check_disk_space_or_halt(threshold_gb: float = MIN_FREE_GB):
    free_gb = get_free_gb(DISK_TO_CHECK)
    log(f"\n[DISK CHECK] Free space on {DISK_TO_CHECK}: {free_gb:.1f} GB  "
        f"(threshold: {threshold_gb} GB)")
    if free_gb < threshold_gb:
        log(f"[HALT] Less than {threshold_gb} GB free on {DISK_TO_CHECK}. "
            f"Stopping to protect the drive.")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# ░░  DOWNLOAD  ░░
# ══════════════════════════════════════════════════════════════════════════════

def download_batch(batch: list):
    """Download a list of (filename, url) pairs using aria2c (skips existing)."""
    import shutil as _shutil

    os.makedirs(CZI_DIR, exist_ok=True)
    log(f"\n{'═'*70}")
    log(f"[DOWNLOAD] Downloading {len(batch)} WSI(s) → {CZI_DIR}")
    log(f"{'═'*70}")

    # ── Verify aria2c is reachable BEFORE attempting any download ─────────────
    aria2c_resolved = _shutil.which(ARIA2C_EXE) or (
        ARIA2C_EXE if os.path.isfile(ARIA2C_EXE) else None
    )
    if aria2c_resolved is None:
        log(f"\n[FATAL] aria2c not found!")
        log(f"  Tried  : '{ARIA2C_EXE}'")
        log(f"  PATH   : {os.environ.get('PATH', '<not set>')}")
        log(f"  Fix    : set ARIA2C_EXE to the full path of aria2c.exe")
        sys.exit(1)
    else:
        log(f"[DOWNLOAD] aria2c found at: {aria2c_resolved}")

    # ── Version check (non-fatal) ─────────────────────────────────────────────
    try:
        ver = subprocess.run(
            [aria2c_resolved, "--version"],
            capture_output=True, text=True, timeout=10,
        )
        first_line = ver.stdout.splitlines()[0] if ver.stdout else "(no output)"
        log(f"[DOWNLOAD] aria2c version: {first_line}")
    except Exception as ve:
        log(f"[WARN] Could not get aria2c version: {ve}")

    # ── Download each file ────────────────────────────────────────────────────
    for i, (filename, url) in enumerate(batch, 1):
        out_path = os.path.join(CZI_DIR, filename)

        log(f"\n  [{i}/{len(batch)}] {filename}")
        log(f"    URL     : {url}")
        log(f"    Dest    : {out_path}")

        if os.path.exists(out_path):
            size_gb = os.path.getsize(out_path) / (1024 ** 3)
            log(f"    [SKIP] Already exists ({size_gb:.2f} GB)")
            continue

        cmd = [
            aria2c_resolved,
            "-x", "16", "-s", "16", "-c",
            "-d", CZI_DIR,
            "-o", filename,
            url,
        ]
        log(f"    CMD     : {' '.join(cmd)}")

        t0 = time.time()
        try:
            result  = subprocess.run(cmd, capture_output=False)
            elapsed = time.time() - t0

            if result.returncode == 0:
                final_size = (
                    os.path.getsize(out_path) / (1024 ** 3)
                    if os.path.exists(out_path) else 0
                )
                log(f"    [OK] Downloaded in {elapsed:.1f}s — {final_size:.2f} GB")
            else:
                log(f"    [ERROR] aria2c exited with code {result.returncode} "
                    f"after {elapsed:.1f}s for {filename}")

        except FileNotFoundError:
            log(f"    [FATAL] aria2c binary disappeared mid-run: {aria2c_resolved}")
            sys.exit(1)
        except Exception as e:
            log(f"    [ERROR] Unexpected error downloading {filename}: "
                f"{type(e).__name__}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# ░░  DELETE CZIs  ░░
# ══════════════════════════════════════════════════════════════════════════════

def delete_batch_czis(batch: list):
    """Delete the CZI files for the given batch after processing."""
    log(f"\n[CLEANUP] Deleting {len(batch)} CZI file(s) ...")
    for filename, _ in batch:
        path = os.path.join(CZI_DIR, filename)
        if os.path.exists(path):
            try:
                os.remove(path)
                log(f"  Deleted: {filename}")
            except Exception as e:
                log(f"  [WARN] Could not delete {filename}: {e}")
        else:
            log(f"  [SKIP] Not found (already gone?): {filename}")

    # Remove any stale aria2c control files
    for filename, _ in batch:
        ctrl = os.path.join(CZI_DIR, filename + ".aria2")
        if os.path.exists(ctrl):
            try:
                os.remove(ctrl)
            except Exception:
                pass


def _slide_features_complete(slide_stem: str) -> bool:
    """
    Return True if every active model already has at least one .pt feature file
    for this slide, meaning the CZI was fully processed in a previous batch.
    """
    for model_name in ACTIVE_MODELS:
        feat_dir = os.path.join(MODEL_OUTPUT_ROOTS[model_name], "features", slide_stem)
        if not os.path.isdir(feat_dir) or not any(f.endswith(".pt") for f in os.listdir(feat_dir)):
            return False
    return True


def filter_batch_for_download(batch: list) -> list:
    """
    Remove slides whose features are already fully extracted so we don't
    re-download CZIs that were deleted after a completed previous batch.
    """
    to_download = []
    for filename, url in batch:
        stem = Path(filename).stem
        if _slide_features_complete(stem):
            log(f"  [SKIP DOWNLOAD] {filename} — features already extracted for all models")
        else:
            to_download.append((filename, url))
    return to_download


# ══════════════════════════════════════════════════════════════════════════════
# ░░  REMOTE SYNC  ░░
# ══════════════════════════════════════════════════════════════════════════════

def _win_to_wsl(path: str) -> str:
    """Convert a Windows path (e.g. D:\\Foo\\Bar) to its WSL mount path
    (e.g. /mnt/d/Foo/Bar)."""
    p = path.replace("\\", "/")
    drive = p[0].lower()
    return f"/mnt/{drive}{p[2:]}"


def _prepare_wsl_ssh_key() -> str | None:
    key_src_wsl = _win_to_wsl(SSH_KEY)
    dest = "/tmp/sync_test_id_ed25519"
    cmd = ["wsl", "bash", "-c", f"cp '{key_src_wsl}' {dest} && chmod 600 {dest}"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    if result.returncode != 0:
        log(f"  [SYNC ERROR] Could not stage SSH key: {result.stderr.strip()}")
        log(f"  stdout: {result.stdout.strip()}")
        log(f"  returncode: {result.returncode}")
        return None
    # Verify the file actually exists after staging
    verify = subprocess.run(
        ["wsl", "bash", "-c", f"ls {dest}"],
        capture_output=True, text=True, timeout=10,
    )
    if verify.returncode != 0:
        log(f"  [SYNC ERROR] Key staging appeared to succeed but file not found: {dest}")
        return None
    return dest


def sync_to_remote(label: str = ""):
    """
    Rsync OUTPUT_BASE to REMOTE_BASE over SSH via Tailscale.
    - Excludes CZI, .tmp, and .aria2 files.
    - Called after each batch so features accumulate on the remote incrementally.
    - Non-fatal: logs errors but does not stop the pipeline.
    """
    if not REMOTE_SYNC_ENABLED:
        log(f"\n[SYNC] Remote sync disabled — skipping.")
        return

    log(f"\n{'═'*70}")
    log(f"[SYNC] Pushing features to {REMOTE_USER}@{REMOTE_HOST}:{REMOTE_BASE}  {label}")
    log(f"{'═'*70}")

    # ── Verify WSL rsync is available ─────────────────────────────────────────
    try:
        ver = subprocess.run(
            ["wsl", "rsync", "--version"],
            capture_output=True, text=True, timeout=10,
        )
    except FileNotFoundError:
        log(f"  [SYNC ERROR] 'wsl' not found on PATH — is WSL installed?")
        log(f"  Features are safe locally at {OUTPUT_BASE} — sync skipped.")
        return

    if ver.returncode != 0:
        log(f"  [SYNC ERROR] WSL rsync not found.")
        log(f"  Fix: inside WSL run: sudo apt-get update && sudo apt-get install rsync")
        log(f"  Features are safe locally at {OUTPUT_BASE} — sync skipped.")
        return

    first_line = ver.stdout.splitlines()[0] if ver.stdout else "(no output)"
    log(f"  WSL rsync : {first_line}")

    # ── Verify SSH key exists on the Windows side ─────────────────────────────
    if not os.path.isfile(SSH_KEY):
        log(f"  [SYNC ERROR] SSH key not found at: {SSH_KEY}")
        log(f"  Features are safe locally — sync skipped.")
        return

    # ── Stage SSH key inside WSL native fs (DrvFs /mnt/... always shows 777) ─
    ssh_key_wsl = _prepare_wsl_ssh_key()
    if ssh_key_wsl is None:
        log(f"  Features are safe locally at {OUTPUT_BASE} — sync skipped.")
        return

    # ── Create destination directory on remote ────────────────────────────────
    mkdir_cmd = [
        "wsl", "ssh",
        "-i", ssh_key_wsl,
        "-p", str(REMOTE_SSH_PORT),
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=15",
        "-o", "BatchMode=yes",
        f"{REMOTE_USER}@{REMOTE_HOST}",
        f"mkdir -p {REMOTE_BASE}",
    ]
    try:
        mkdir_result = subprocess.run(mkdir_cmd, capture_output=True, text=True, timeout=20)
        if mkdir_result.returncode != 0:
            log(f"  [WARN] Remote mkdir returned code {mkdir_result.returncode}: "
                f"{mkdir_result.stderr.strip()}")
    except subprocess.TimeoutExpired:
        log(f"  [WARN] Remote mkdir timed out — continuing, rsync will fail if "
            f"the remote dir does not exist.")

    # ── Build rsync command using the WSL-staged key ──────────────────────────
    local_dir_wsl = _win_to_wsl(FEATURES_BASE)
    ssh_opts = (
        f"ssh -i {ssh_key_wsl} -p {REMOTE_SSH_PORT} "
        f"-o StrictHostKeyChecking=no -o ConnectTimeout=15 -o BatchMode=yes"
    )

    cmd = [
        "wsl", "rsync",
        "-avz",              # archive mode, verbose, compress in transit
        "--progress",        # show per-file progress
        "--exclude=*.czi",   # never sync raw WSI files
        "--exclude=*.tmp",   # skip in-progress atomic writes
        "--exclude=*.aria2", # skip aria2c control files
        f"--rsh={ssh_opts}",
        local_dir_wsl + "/",                              # trailing slash = contents only
        f"{REMOTE_USER}@{REMOTE_HOST}:{REMOTE_BASE}/",
    ]

    log(f"  CMD: {' '.join(cmd)}")
    t0 = time.time()
    try:
        result  = subprocess.run(cmd, capture_output=False, timeout=3600)  # 1hr max
        elapsed = time.time() - t0
        if result.returncode == 0:
            log(f"  [SYNC OK] Completed in {elapsed:.1f}s")
        else:
            log(f"  [SYNC ERROR] rsync exited with code {result.returncode} "
                f"after {elapsed:.1f}s")
            log(f"  Features are safe locally at {OUTPUT_BASE}")
    except FileNotFoundError:
        log(f"  [SYNC FATAL] 'wsl' binary disappeared at runtime.")
        log(f"  Features are safe locally at {OUTPUT_BASE}")
    except Exception as e:
        log(f"  [SYNC ERROR] {type(e).__name__}: {e}")
        log(f"  Features are safe locally at {OUTPUT_BASE}")


# ══════════════════════════════════════════════════════════════════════════════
# ░░  LABEL LOOKUP  ░░
# ══════════════════════════════════════════════════════════════════════════════

_surgen_labels_df = None


def _load_label_df():
    """Load and cache the unified labels CSV, indexed by WSI_Id."""
    global _surgen_labels_df
    if _surgen_labels_df is None:
        df = pd.read_csv(SURGEN_LABELS_CSV)
        _surgen_labels_df = df.set_index("WSI_Id")
        log(f"  Loaded labels: {len(_surgen_labels_df)} rows from {SURGEN_LABELS_CSV}")


def get_slide_label(czi_filename: str):
    """
    Return 'msih', 'nonmsih', or None (skip) for the given CZI filename.

      label_desc == 1  → 'msih'
      label_desc == 0  → 'nonmsih'
      label_desc == -1 → None  (excluded)
      not in CSV       → None  (unknown)
    """
    _load_label_df()
    stem = Path(czi_filename).stem

    if stem not in _surgen_labels_df.index:
        log(f"  [SKIP] {stem}: not found in labels CSV")
        return None

    val = int(_surgen_labels_df.at[stem, "label_desc"])

    if val == 1:
        return "msih"
    elif val == 0:
        return "nonmsih"
    else:
        log(f"  [SKIP] {stem}: label_desc={val} — excluded")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# ░░  CZI HELPERS  ░░
# ══════════════════════════════════════════════════════════════════════════════

def find_all_czi_files(root: str, excluded_list: list = None) -> list:
    excluded_stems = set()
    if excluded_list:
        excluded_stems = {Path(e).stem.lower() for e in excluded_list}
    results = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(".czi"):
                if Path(f).stem.lower() not in excluded_stems:
                    results.append(os.path.join(dirpath, f))
    return sorted(results)


def open_czi_at_target_mag(czi_path: str, target_mag: int = 20, patch_size: int = 512):
    from pylibCZIrw import czi as pyczi

    with pyczi.open_czi(czi_path) as czidoc:
        try:
            native_mag = int(
                czidoc.metadata["ImageDocument"]["Metadata"]
                ["Information"]["Instrument"]["Objectives"]
                ["Objective"]["NominalMagnification"]
            )
        except Exception:
            native_mag = 40

        bbox    = czidoc.total_bounding_box
        X_start = bbox["X"][0]
        Y_start = bbox["Y"][0]
        W       = bbox["X"][1] - bbox["X"][0]
        H       = bbox["Y"][1] - bbox["Y"][0]


        downsample = max(1, native_mag // target_mag)
        level_mag  = native_mag // downsample

    return {
        "czi_path"  : czi_path,
        "W"         : W,  "H": H,
        "X_start"   : X_start, "Y_start": Y_start,
        "native_mag": native_mag,
        "level_mag" : level_mag,
        "downsample": downsample,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ░░  STEP A-1 — RGB stats per slide  ░░
# ══════════════════════════════════════════════════════════════════════════════

def _compute_patch_stats(patches, patch_size):
    avg = patches.mean(axis=(1, 2))
    std = patches.std(axis=(1, 2))
    p8  = patches.astype(np.uint8)
    black_mask        = (p8[:, :, :, 0] < 15) & (p8[:, :, :, 1] < 15) & (p8[:, :, :, 2] < 15)
    white_mask        = (p8[:, :, :, 0] > 210) & (p8[:, :, :, 1] > 210) & (p8[:, :, :, 2] > 210)
    total_px          = patch_size * patch_size
    black_pixel_ratio = black_mask.sum(axis=(1, 2)) / total_px
    white_pixel_ratio = white_mask.sum(axis=(1, 2)) / total_px
    return avg, std, black_pixel_ratio, white_pixel_ratio


def _append_patch_rows(all_rows, patch_num_offset,
                        avg, std, bpr, wpr,
                        abs_row_indices, col_indices, patch_size):
    N       = len(abs_row_indices)
    patch_x = col_indices     * patch_size
    patch_y = abs_row_indices * patch_size
    for i in range(N):
        all_rows.append({
            "patch_number"     : patch_num_offset + i,
            "patch_x"          : int(patch_x[i]),
            "patch_y"          : int(patch_y[i]),
            "avg_R"            : float(avg[i, 0]),
            "avg_G"            : float(avg[i, 1]),
            "avg_B"            : float(avg[i, 2]),
            "std_R"            : float(std[i, 0]),
            "std_G"            : float(std[i, 1]),
            "std_B"            : float(std[i, 2]),
            "black_pixel_ratio": float(bpr[i]),
            "white_pixel_ratio": float(wpr[i]),
        })


def compute_patch_rgb_stats_rowbatch(slide_info: dict, patch_size: int,
                                      row_batch: int = 16) -> pd.DataFrame:
    from pylibCZIrw import czi as pyczi

    H, W       = slide_info["H"], slide_info["W"]
    downsample = slide_info["downsample"]
    czi_path   = slide_info["czi_path"]
    x_start    = slide_info["X_start"]
    y_start    = slide_info["Y_start"]
    zoom       = 1.0 / downsample

    H_target     = H // downsample
    W_target     = W // downsample
    n_patch_rows = H_target // patch_size
    n_patch_cols = W_target // patch_size

    if n_patch_rows == 0 or n_patch_cols == 0:
        log(f"  [WARN] grid is empty for {slide_info['czi_path']}, returning early")
        return pd.DataFrame()

    all_rows     = []
    patch_num    = 0
    skipped_rows = 0

    with pyczi.open_czi(czi_path) as czidoc:
        for strip_idx, strip_start_row in enumerate(range(0, n_patch_rows, row_batch)):
            strip_end_row   = min(strip_start_row + row_batch, n_patch_rows)
            n_rows_in_strip = strip_end_row - strip_start_row
            N_strip         = n_rows_in_strip * n_patch_cols

            y_native_strip = y_start + (strip_start_row * patch_size * downsample)
            height_native  = n_rows_in_strip * patch_size * downsample
            width_native   = n_patch_cols * patch_size * downsample

            try:
                strip = czidoc.read(
                    roi=(x_start, y_native_strip, width_native, height_native),
                    zoom=zoom, plane={"C": 0, "Z": 0, "T": 0},
                )[:, :, :3]
                strip   = strip[:n_rows_in_strip * patch_size, :n_patch_cols * patch_size, :]
                grid    = strip.astype(np.float32).reshape(
                    n_rows_in_strip, patch_size, n_patch_cols, patch_size, 3
                ).transpose(0, 2, 1, 3, 4)
                patches = grid.reshape(N_strip, patch_size, patch_size, 3)

                avg, std, bpr, wpr = _compute_patch_stats(patches, patch_size)
                row_idx, col_idx   = np.unravel_index(
                    np.arange(N_strip), (n_rows_in_strip, n_patch_cols)
                )
                _append_patch_rows(all_rows, patch_num, avg, std, bpr, wpr,
                                   row_idx + strip_start_row, col_idx, patch_size)
                patch_num += N_strip
                del strip, grid, patches

            except Exception as strip_err:
                log(f"  [WARN] Strip {strip_idx+1} failed ({strip_err}), retrying row-by-row …")
                for single_row in range(strip_start_row, strip_end_row):
                    y_native_row = y_start + (single_row * patch_size * downsample)
                    try:
                        row_strip = czidoc.read(
                            roi=(x_start, y_native_row, width_native, patch_size * downsample),
                            zoom=zoom, plane={"C": 0, "Z": 0, "T": 0},
                        )[:, :, :3]
                        row_strip = row_strip[:patch_size, :n_patch_cols * patch_size, :]
                        grid      = row_strip.astype(np.float32).reshape(
                            1, patch_size, n_patch_cols, patch_size, 3
                        ).transpose(0, 2, 1, 3, 4)
                        patches   = grid.reshape(n_patch_cols, patch_size, patch_size, 3)
                        avg, std, bpr, wpr = _compute_patch_stats(patches, patch_size)
                        _append_patch_rows(all_rows, patch_num, avg, std, bpr, wpr,
                                           np.full(n_patch_cols, single_row),
                                           np.arange(n_patch_cols), patch_size)
                        patch_num += n_patch_cols
                        del row_strip, grid, patches
                    except Exception as row_err:
                        log(f"    [SKIP] Row {single_row} failed: {row_err}")
                        patch_num    += n_patch_cols
                        skipped_rows += 1

    if skipped_rows:
        log(f"  [INFO] {skipped_rows} corrupt row(s) skipped for this slide")
    return pd.DataFrame(all_rows)


def run_step_A1(czi_root, output_metadata_dir, patch_size, target_mag, row_batch):
    """Generate per-slide RGB stats CSVs. Resume-safe: skips completed slides."""
    czi_files = find_all_czi_files(czi_root)
    log(f"Found {len(czi_files)} CZI file(s) under {czi_root}")

    for czi_path in tqdm(czi_files, desc="Step A-1: RGB stats", unit="slide"):
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
            log(f"  [RESUME] {stem}: removing stale .tmp, restarting")
            os.remove(tmp_csv)

        log(f"  Processing {stem}  (label={label}) ...")
        try:
            slide_info = open_czi_at_target_mag(czi_path, target_mag, patch_size)
        except Exception as e:
            log(f"  [ERROR] Could not open {stem}: {e}")
            continue

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

        df.insert(0, "slide_name", stem)
        df.insert(1, "label",      label)
        df.to_csv(tmp_csv, index=False)
        os.rename(tmp_csv, final_csv)
        log(f"    → {len(df):,} patches | saved to {final_csv}")


# ══════════════════════════════════════════════════════════════════════════════
# ░░  STEP A-2 — SVM whiteness classification  ░░
# ══════════════════════════════════════════════════════════════════════════════

def tag_bad_patches(df: pd.DataFrame, qc_filtered_dir: str):
    is_black       = df["black_pixel_ratio"] > BLACK_PIXEL_RATIO
    is_white_ratio = df["white_pixel_ratio"] > WHITE_PIXEL_RATIO
    is_white_mean  = (
        (df["avg_R"] > WHITE_MEAN_THRESH) & (df["avg_G"] > WHITE_MEAN_THRESH) &
        (df["avg_B"] > WHITE_MEAN_THRESH) & (df["std_R"] < WHITE_STD_THRESH)  &
        (df["std_G"] < WHITE_STD_THRESH)  & (df["std_B"] < WHITE_STD_THRESH)
    )
    bad = is_black | is_white_ratio | is_white_mean
    df  = df.copy()
    df["white_label"]   = float("nan")
    df["reject_reason"] = ""
    df.loc[bad,            "white_label"]   = 1
    df.loc[is_black,       "reject_reason"] = "black_pixel_ratio"
    df.loc[is_white_ratio, "reject_reason"] = "white_pixel_ratio"
    df.loc[is_white_mean,  "reject_reason"] = "white_mean_std"

    log(f"  Stage 1 — pixel-rule pre-filter:")
    log(f"    black_pixel_ratio : {is_black.sum():,}")
    log(f"    white_pixel_ratio : {is_white_ratio.sum():,}")
    log(f"    white_mean_std    : {is_white_mean.sum():,}")
    log(f"    Total force-white : {bad.sum():,}")
    log(f"    Passed to SVM     : {(~bad).sum():,}")

    os.makedirs(qc_filtered_dir, exist_ok=True)
    filtered_df = df[bad][[
        "slide_name", "patch_number", "patch_x", "patch_y", "reject_reason",
        "black_pixel_ratio", "white_pixel_ratio",
        "avg_R", "avg_G", "avg_B", "std_R", "std_G", "std_B",
    ]].copy()
    filtered_csv = os.path.join(qc_filtered_dir, "rule_filtered_patches.csv")
    filtered_df.to_csv(filtered_csv, index=False)

    return df, bad


def apply_svm_whiteness(svm_input_df: pd.DataFrame) -> pd.DataFrame:
    import joblib

    svm_bundle = joblib.load(SVM_MODEL_PATH)
    if isinstance(svm_bundle, dict):
        svm_model = svm_bundle["model"]
        scaler    = svm_bundle.get("scaler", None)
    elif isinstance(svm_bundle, (list, tuple)) and len(svm_bundle) == 2:
        svm_model, scaler = svm_bundle
    else:
        svm_model = svm_bundle
        scaler    = None

    X = svm_input_df[["avg_G", "avg_B", "std_R"]].values
    if scaler is not None:
        X = scaler.transform(X)

    y_pred    = svm_model.predict(X)
    result_df = svm_input_df.copy()
    result_df["white_label"] = y_pred

    unique_vals = np.unique(y_pred)
    white_class = max(unique_vals)
    n_white  = (y_pred == white_class).sum()
    n_tissue = (y_pred != white_class).sum()
    log(f"  SVM — white: {n_white:,}  tissue: {n_tissue:,}")
    return result_df


def run_step_A2(patch_metadata_dir, merged_csv_path, nonwhite_csv_path, qc_filtered_dir):
    """
    Merge per-slide CSVs, apply pixel filter + SVM, write merged and nonwhite CSVs.
    Always re-runs to incorporate new slides from the current batch.
    """
    csv_files = sorted(glob.glob(os.path.join(patch_metadata_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(
            f"[ERROR] No per-slide CSVs found in: {patch_metadata_dir}\n"
            f"  → Run Step A-1 first."
        )

    log(f"Merging {len(csv_files)} per-slide CSV(s) ...")
    dfs = []
    for csv_path in csv_files:
        try:
            dfs.append(pd.read_csv(csv_path))
        except Exception as e:
            log(f"  [WARN] Could not read {csv_path}: {e} — skipping")

    if not dfs:
        raise RuntimeError("[ERROR] All per-slide CSVs failed to load.")

    merged_df = pd.concat(dfs, ignore_index=True)
    log(f"  → {len(merged_df):,} total patches from {len(dfs)} slide(s)")

    merged_df, bad_mask = tag_bad_patches(merged_df, qc_filtered_dir)

    svm_input_df = merged_df[merged_df["white_label"].isna()].drop(
        columns=["white_label", "reject_reason"]
    )
    log(f"\n  Stage 2 — SVM on {len(svm_input_df):,} patches ...")
    svm_result_df = apply_svm_whiteness(svm_input_df)

    merged_df.loc[merged_df["white_label"].isna(), "white_label"] = \
        svm_result_df["white_label"].values
    merged_df["white_label"] = merged_df["white_label"].astype(int)

    save_df = merged_df.drop(columns=["reject_reason"])
    save_df.to_csv(merged_csv_path, index=False)

    nonwhite_class = save_df["white_label"].min()
    nonwhite_df    = save_df[save_df["white_label"] == nonwhite_class].reset_index(drop=True)
    nonwhite_df.to_csv(nonwhite_csv_path, index=False)

    total    = len(save_df)
    n_tissue = len(nonwhite_df)
    n_white  = total - n_tissue
    log(f"\nStep A-2 complete.")
    log(f"  Total: {total:,}  |  White: {n_white:,}  |  Tissue: {n_tissue:,}")
    log(f"  Merged CSV   → {merged_csv_path}")
    log(f"  Nonwhite CSV → {nonwhite_csv_path}")
    return nonwhite_df


# ══════════════════════════════════════════════════════════════════════════════
# ░░  STEP C — feature extraction  ░░
# ══════════════════════════════════════════════════════════════════════════════

def _read_batch_sequential(czidoc, batch_rows, slide_stem,
                            x_start, y_start, downsample, size_native, zoom):
    """
    Read every patch in a batch, one after another, on whichever thread calls
    this function. pylibCZIrw's czidoc is NOT thread-safe for concurrent reads,
    so this must never be called from more than one thread at a time against
    the same czidoc — but it's fine to call it from a single dedicated
    background thread while the main thread does GPU work, since that keeps
    reads sequential (only ever one in flight) while still overlapping I/O
    with compute.
    """
    from PIL import Image

    images, valid_ids = [], []
    for row in batch_rows:
        x_native = x_start + (int(row["patch_x"]) * downsample)
        y_native = y_start + (int(row["patch_y"]) * downsample)
        patch_id = f"{slide_stem}_{int(row['patch_number'])}"
        try:
            raw = czidoc.read(
                roi=(x_native, y_native, size_native, size_native),
                zoom=zoom, plane={"C": 0, "Z": 0, "T": 0},
            )
            images.append(Image.fromarray(raw[:, :, :3].astype("uint8")).convert("RGB"))
            valid_ids.append(patch_id)
        except Exception as e:
            tqdm.write(f"    [ERROR] patch {patch_id}: {e}")
    return images, valid_ids


def run_step_C(czi_root, nonwhite_csv, features_root,
               extractor, patch_size, target_mag, batch_size: int = 8):
    """
    Batched feature extraction — resume-safe via patch-level skip.
    Atomic saves (tmp → replace) to survive crashes.

    Speed-up: double-buffered read/compute overlap. czidoc is NOT thread-safe,
    so reads still happen strictly one-at-a-time — but they run on a single
    dedicated background thread, and the NEXT batch's reads are kicked off
    right before the GPU starts working on the CURRENT batch. That means I/O
    and GPU compute genuinely overlap without ever letting two threads touch
    czidoc at once.
    """
    import torch
    from pylibCZIrw import czi as pyczi
    from concurrent.futures import ThreadPoolExecutor

    if not os.path.exists(nonwhite_csv):
        raise FileNotFoundError(f"[ERROR] Non-white CSV not found: {nonwhite_csv}")

    nonwhite_df  = pd.read_csv(nonwhite_csv)
    slide_groups = nonwhite_df.groupby("slide_name")
    log(f"Found {len(slide_groups)} slide(s) with non-white patches (across all batches).")

    # Only index CZI files that are actually present right now (batch CZIs)
    czi_index = {Path(p).stem: p for p in find_all_czi_files(czi_root)}

    # Filter to only slides present on disk — keeps progress bar scoped to current batch
    batch_slide_groups = [(stem, grp) for stem, grp in slide_groups if stem in czi_index]
    log(f"  Processing {len(batch_slide_groups)} slide(s) present in current batch.")

    for slide_stem, group in tqdm(batch_slide_groups, desc="Step C: Feature extraction", unit="slide"):
        feat_dir = os.path.join(features_root, slide_stem)
        os.makedirs(feat_dir, exist_ok=True)

        # Resume: skip already-extracted patches
        existing = {Path(f).stem for f in os.listdir(feat_dir) if f.endswith(".pt")}

        czi_path = czi_index[slide_stem]
        log(f"  Loading {slide_stem} ...")
        try:
            slide_info = open_czi_at_target_mag(czi_path, target_mag, patch_size)
        except Exception as e:
            log(f"  [ERROR] Cannot open {slide_stem}: {e}")
            continue

        downsample  = slide_info["downsample"]
        x_start     = slide_info["X_start"]
        y_start     = slide_info["Y_start"]
        zoom        = 1.0 / downsample
        size_native = patch_size * downsample

        patches_to_process = group.reset_index(drop=True)
        patches_to_process["_id"] = (
            patches_to_process["slide_name"] + "_" +
            patches_to_process["patch_number"].astype(str)
        )
        patches_to_process = patches_to_process[
            ~patches_to_process["_id"].isin(existing)
        ]

        if patches_to_process.empty:
            tqdm.write(f"  [SKIP] All patches already processed for {slide_stem}")
            continue

        rows      = patches_to_process.to_dict("records")
        n_total   = len(rows)
        n_done    = 0
        n_batches = (n_total + batch_size - 1) // batch_size

        log(f"  [INFO] {n_total} patches to process | downsample={downsample} size_native={size_native}px")

        # pylibCZIrw czidoc is NOT thread-safe for concurrent reads — the
        # single-worker pool below guarantees reads still happen one at a
        # time, just on a background thread so they overlap with GPU compute.
        with pyczi.open_czi(czi_path) as czidoc, \
             ThreadPoolExecutor(max_workers=1) as reader_pool:

            # ── Prime the pipeline: kick off the read for batch 0 up front ───
            next_batch_rows = rows[0:batch_size]
            next_future = reader_pool.submit(
                _read_batch_sequential, czidoc, next_batch_rows, slide_stem,
                x_start, y_start, downsample, size_native, zoom
            )

            for b_idx in tqdm(range(n_batches),
                               desc=f"  Patches: {slide_stem}",
                               unit="batch", leave=False):

                # This batch's images were already being read in the
                # background (primed above, or submitted last iteration).
                images, valid_ids = next_future.result()

                # ── Submit the NEXT batch's read now, before GPU work starts,
                #    so the background thread fetches it while the GPU is busy.
                #    Still only one thread ever calls czidoc.read() at a time.
                if b_idx + 1 < n_batches:
                    next_batch_rows = rows[(b_idx + 1) * batch_size : (b_idx + 2) * batch_size]
                    next_future = reader_pool.submit(
                        _read_batch_sequential, czidoc, next_batch_rows, slide_stem,
                        x_start, y_start, downsample, size_native, zoom
                    )

                if not images:
                    continue

                try:
                    batch_features = extractor.extract_batch(images)
                except Exception as e:
                    tqdm.write(f"    [ERROR] batch inference at idx {b_idx}: {e}")
                    continue

                for patch_id, feat in zip(valid_ids, batch_features):
                    save_path = os.path.join(feat_dir, patch_id + ".pt")
                    tmp_path  = save_path + ".tmp"
                    torch.save(feat.to(torch.float16).clone(), tmp_path)
                    os.replace(tmp_path, save_path)
                    n_done += 1

                # Explicit cleanup each batch to prevent RAM accumulation
                del images, batch_features, valid_ids
                gc.collect()

        gc.collect()

        tqdm.write(
            f"  ✓ {slide_stem}: saved {n_done}/{n_total} feature files → {feat_dir}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# ░░  MODEL LOADER  ░░
# ══════════════════════════════════════════════════════════════════════════════

def load_extractor(model_name: str):
    """Instantiate and return the feature extractor for the given model."""
    import torch
    from huggingface_hub import login
    login(token=HF_TOKEN)

    # ── H-Optimus-1 ───────────────────────────────────────────────────────────
    if model_name == "h-optimus-1":
        import timm
        from torchvision import transforms

        class HOptimusExtractor:
            def __init__(self, device, hf_token):
                self.device = torch.device(device)
                log(f"  Loading H-Optimus-1 on {self.device} ...")
                self.model = timm.create_model(
                    "hf-hub:bioptimus/H-optimus-1",
                    pretrained=True, init_values=1e-5, dynamic_img_size=False,
                ).to(self.device)
                self.model.eval()
                self._fivecrop = transforms.Compose([
                    transforms.FiveCrop(224),
                    transforms.Lambda(lambda crops: torch.stack([
                        transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=(0.707223, 0.578729, 0.703617),
                                std=(0.211883, 0.230117, 0.177517),
                            ),
                        ])(c) for c in crops
                    ]))
                ])
                log("  H-Optimus-1 ready.  Feature dim: 1536")

            def extract_batch(self, images):
                batch = torch.cat(
                    [self._fivecrop(img.convert("RGB")) for img in images], dim=0
                ).to(self.device)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    with torch.inference_mode():
                        features = self.model(batch)
                return features.cpu().reshape(len(images), 5, 1536)

        return HOptimusExtractor(device=DEVICE, hf_token=HF_TOKEN)

    # ── CONCH 1.5 ─────────────────────────────────────────────────────────────
    elif model_name == "conch1-5":
        from transformers import AutoModel
        from torchvision import transforms

        class Conch15Extractor:
            def __init__(self, device, hf_token):
                self.device = torch.device(device)
                log(f"  Loading CONCH 1.5 (via TITAN) on {self.device} ...")
                titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
                self.conch, self.eval_transform = titan.return_conch()
                self.conch.to(self.device).eval()
                self._fivecrop = transforms.Compose([
                    transforms.FiveCrop(256),
                    transforms.Lambda(
                        lambda crops: torch.stack([self.eval_transform(c) for c in crops])
                    )
                ])
                with torch.inference_mode():
                    _d = torch.zeros(1, 3, 224, 224, device=self.device)
                    self.embed_dim = self.conch(_d).shape[-1]
                log(f"  CONCH 1.5 ready.  Embedding dim: {self.embed_dim}")

            def extract_batch(self, images):
                batch = torch.cat(
                    [self._fivecrop(img.convert("RGB")) for img in images], dim=0
                ).to(self.device)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    with torch.inference_mode():
                        features = self.conch(batch)
                return features.cpu().reshape(len(images), 5, self.embed_dim)

        return Conch15Extractor(device=DEVICE, hf_token=HF_TOKEN)

    # ── CONCH V1 ──────────────────────────────────────────────────────────────
    elif model_name == "conch-v1":
        from conch.open_clip_custom import create_model_from_pretrained
        from torchvision import transforms

        class ConchV1Extractor:
            def __init__(self, device, hf_token, checkpoint=None):
                self.device = torch.device(device)
                log(f"  Loading CONCH V1 on {self.device} ...")
                if checkpoint and os.path.exists(checkpoint):
                    model, preprocess = create_model_from_pretrained(
                        "conch_ViT-B-16", checkpoint
                    )
                else:
                    model, preprocess = create_model_from_pretrained(
                        "conch_ViT-B-16", "hf_hub:MahmoodLab/conch",
                        hf_auth_token=hf_token,
                    )
                self.model = model.to(self.device).eval()
                self._fivecrop = transforms.Compose([
                    transforms.FiveCrop(256),
                    transforms.Lambda(
                        lambda crops: torch.stack([preprocess(c) for c in crops])
                    )
                ])
                log("  CONCH V1 ready.  Feature dim: 512")

            def extract_batch(self, images):
                batch = torch.cat(
                    [self._fivecrop(img.convert("RGB")) for img in images], dim=0
                ).to(self.device)
                with torch.inference_mode():
                    features = self.model.encode_image(
                        batch, proj_contrast=False, normalize=False
                    )
                return features.cpu().reshape(len(images), 5, 512).to(torch.float16)

        return ConchV1Extractor(device=DEVICE, hf_token=HF_TOKEN,
                                checkpoint=CONCH_V1_CHECKPOINT)

    # ── UNI2-h ────────────────────────────────────────────────────────────────
    elif model_name == "uni2-h":
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        from torchvision import transforms

        class UNI2hExtractor:
            def __init__(self, device, hf_token):
                self.device = torch.device(device)
                log(f"  Loading UNI2-h on {self.device} ...")
                timm_kwargs = dict(
                    img_size=224, patch_size=14, depth=24, num_heads=24,
                    init_values=1e-5, embed_dim=1536,
                    mlp_ratio=2.66667 * 2, num_classes=0,
                    no_embed_class=True,
                    mlp_layer=timm.layers.SwiGLUPacked,
                    act_layer=torch.nn.SiLU,
                    reg_tokens=8, dynamic_img_size=True,
                )
                self.model = timm.create_model(
                    "hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs
                ).to(self.device).eval()

                config    = resolve_data_config(self.model.pretrained_cfg, model=self.model)
                transform = create_transform(**config)
                self._fivecrop = transforms.Compose([
                    transforms.FiveCrop(224),
                    transforms.Lambda(
                        lambda crops: torch.stack([transform(c) for c in crops])
                    )
                ])
                log("  UNI2-h ready.  Feature dim: 1536")

            def extract_batch(self, images):
                batch = torch.cat(
                    [self._fivecrop(img.convert("RGB")) for img in images], dim=0
                ).to(self.device)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    with torch.inference_mode():
                        features = self.model(batch)
                return features.cpu().reshape(len(images), 5, 1536)

        return UNI2hExtractor(device=DEVICE, hf_token=HF_TOKEN)

    # ── Virchow2 ──────────────────────────────────────────────────────────────
    elif model_name == "virchow2":
        import timm
        from timm.data import resolve_data_config
        from timm.layers import SwiGLUPacked
        from torchvision import transforms

        class Virchow2Extractor:
            def __init__(self, device, hf_token, crop_size=256):
                self.device = torch.device(device)
                log(f"  Loading Virchow2 on {self.device} ...")
                self.model = timm.create_model(
                    "hf-hub:paige-ai/Virchow2", pretrained=True,
                    mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU,
                ).to(self.device).eval()

                data_cfg = resolve_data_config(self.model.pretrained_cfg, model=self.model)
                per_crop = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=data_cfg["mean"], std=data_cfg["std"]),
                ])
                self._crop_transform = transforms.Compose([
                    transforms.FiveCrop(crop_size),
                    transforms.Lambda(
                        lambda crops: torch.stack([per_crop(c) for c in crops])
                    )
                ])
                log(f"  Virchow2 ready.  Embedding dim: 2560  (crop_size={crop_size})")

            @torch.inference_mode()
            def _forward(self, crops_tensor):
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    output = self.model(crops_tensor)
                class_token  = output[:, 0].float()
                patch_tokens = output[:, 5:].float()
                return torch.cat(
                    [class_token, patch_tokens.mean(dim=1)], dim=-1
                ).cpu()

            def extract_batch(self, images):
                batch = torch.cat(
                    [self._crop_transform(img.convert("RGB")) for img in images], dim=0
                ).to(self.device)
                feats = self._forward(batch)
                return feats.reshape(len(images), 5, 2560).to(torch.float16)

        return Virchow2Extractor(device=DEVICE, hf_token=HF_TOKEN,
                                 crop_size=VIRCHOW2_CROP_SIZE)

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ══════════════════════════════════════════════════════════════════════════════
# ░░  MAIN ORCHESTRATOR  ░░
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import torch

    # Patch size (512px) is fixed across the whole run, so cuDNN can safely
    # benchmark and cache the fastest conv algorithms for that shape. TF32
    # lets any remaining fp32 matmuls (e.g. inside TITAN/conch1-5 internals
    # not covered by autocast) run faster on Ampere+ GPUs.
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    log("╔══════════════════════════════════════════════════════════════════════╗")
    log("║       SurGen Batch Processing Orchestrator                          ║")
    log("╚══════════════════════════════════════════════════════════════════════╝")

    # ── Build the WSI catalogue dynamically from the labels CSV ───────────────
    all_wsis = build_wsi_list(SURGEN_LABELS_CSV)

    log(f"  Total valid WSIs  : {len(all_wsis)}")
    log(f"  Batch size        : {BATCH_SIZE}")
    log(f"  Active models     : {ACTIVE_MODELS}")
    log(f"  Min free GB       : {MIN_FREE_GB} GB on {DISK_TO_CHECK}")
    log(f"  CZI dir           : {CZI_DIR}")
    log(f"  Output base       : {OUTPUT_BASE}")
    log(f"  Labels CSV        : {SURGEN_LABELS_CSV}")
    log(f"  VISUALIZE_PATCHES : {'ON' if VISUALIZE_PATCHES else 'OFF'}")
    log(f"  Remote sync       : {'ENABLED → ' + REMOTE_USER + '@' + REMOTE_HOST if REMOTE_SYNC_ENABLED else 'DISABLED'}")

    # Create all output directories up front
    os.makedirs(PATCH_METADATA_DIR, exist_ok=True)
    for model_name in ACTIVE_MODELS:
        root = MODEL_OUTPUT_ROOTS[model_name]
        os.makedirs(os.path.join(root, "features"),                       exist_ok=True)
        os.makedirs(os.path.join(root, "qc_patches", "filtered_by_rule"), exist_ok=True)

    # Split the catalogue into batches
    batches = [all_wsis[i:i + BATCH_SIZE] for i in range(0, len(all_wsis), BATCH_SIZE)]
    log(f"\n  Batches to process: {len(batches)}")

    for batch_idx, batch in enumerate(batches):
        log(f"\n{'█'*72}")
        log(f"█  BATCH {batch_idx + 1}/{len(batches)}  —  {len(batch)} WSI(s)")
        log(f"{'█'*72}")

        # ── 1. Download ───────────────────────────────────────────────────────
        batch_to_download = filter_batch_for_download(batch)
        if batch_to_download:
            download_batch(batch_to_download)
        else:
            log(f"  [SKIP DOWNLOAD] All slides in this batch already processed — skipping download.")

        # ── 2 & 3. Step A-1 + A-2 ────────────────────────────────────────────
        # Skipped entirely if the nonwhite CSV already exists — it contains
        # all slides and does not need to be rebuilt per batch.
        if os.path.exists(NONWHITE_METADATA_CSV):
            log(f"\n[STEP A-1 & A-2 SKIPPED] Nonwhite CSV already present:")
            log(f"  {NONWHITE_METADATA_CSV}")
            log(f"  Delete that file to force re-extraction and re-filtering.")
        else:
            log(f"\n[STEP A-1] Generating per-slide patch metadata ...")
            run_step_A1(CZI_DIR, PATCH_METADATA_DIR, PATCH_SIZE, TARGET_MAG, ROW_BATCH)

            log(f"\n[STEP A-2] Merging metadata and running SVM filter ...")
            qc_filtered_dir = os.path.join(_HOPT_ROOT, "qc_patches", "filtered_by_rule")
            os.makedirs(qc_filtered_dir, exist_ok=True)
            run_step_A2(
                patch_metadata_dir = PATCH_METADATA_DIR,
                merged_csv_path    = MERGED_METADATA_CSV,
                nonwhite_csv_path  = NONWHITE_METADATA_CSV,
                qc_filtered_dir    = qc_filtered_dir,
            )

        # ── 4. Feature extraction per active model ────────────────────────────
        for model_name in ACTIVE_MODELS:
            log(f"\n{'─'*70}")
            log(f"[MODEL: {model_name.upper()}]  Loading and extracting features ...")
            log(f"{'─'*70}")

            features_root = os.path.join(MODEL_OUTPUT_ROOTS[model_name], "features")

            try:
                extractor = load_extractor(model_name)
                run_step_C(
                    czi_root      = CZI_DIR,
                    nonwhite_csv  = NONWHITE_METADATA_CSV,
                    features_root = features_root,
                    extractor     = extractor,
                    patch_size    = PATCH_SIZE,
                    target_mag    = TARGET_MAG,
                    batch_size    = FEATURE_BATCH_SIZE,
                )
            except Exception as e:
                log(f"[ERROR] {model_name} failed: {e}")
                import traceback
                traceback.print_exc()
            finally:
                try:
                    del extractor
                except NameError:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    log(f"  GPU memory freed after {model_name}.")

        # ── 5. Delete CZIs ────────────────────────────────────────────────────
        delete_batch_czis(batch)

        # ── 6. Sync to remote ─────────────────────────────────────────────────
        sync_to_remote(label=f"(batch {batch_idx + 1}/{len(batches)})")

        # ── 7. Disk space guard ───────────────────────────────────────────────
        check_disk_space_or_halt(MIN_FREE_GB)

        log(f"\n[BATCH {batch_idx + 1}/{len(batches)} COMPLETE]")

    log("\n╔══════════════════════════════════════════════════════════════════════╗")
    log("║  ALL BATCHES COMPLETE                                               ║")
    log("╚══════════════════════════════════════════════════════════════════════╝")
    log(f"  Feature outputs by model:")
    for model_name in ACTIVE_MODELS:
        log(f"    {model_name:15s} → {MODEL_OUTPUT_ROOTS[model_name]}")
    log(f"  Shared patch metadata  → {PATCH_METADATA_DIR}")
    log(f"  Nonwhite CSV           → {NONWHITE_METADATA_CSV}")
    log(f"  Remote destination     → {REMOTE_USER}@{REMOTE_HOST}:{REMOTE_BASE}")


if __name__ == "__main__":
    main()