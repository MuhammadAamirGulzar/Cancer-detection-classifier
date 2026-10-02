"""
SurGen WSI Processing Pipeline — PRISM slide embeddings
========================================================
Downloads SurGen WSIs in batches of BATCH_SIZE, encodes every non-white patch
with Virchow V1, aggregates each slide's tile embeddings with PRISM's Perceiver
into a single (1, 1280) slide embedding, then DELETES that batch's CZI files
before moving on. Disk usage therefore stays bounded at roughly
"BATCH_SIZE CZIs + one slide's tile cache" no matter how many slides are run.

    download 10 CZIs  →  encode + aggregate them  →  delete those 10 CZIs
    →  download the next 10  →  ...

Usage:
    conda activate prism_env
    python surgen_processing_prism.py

    # or, auto-restarting after a crash (recommended for the full run):
    python run_pipeline_supervisor.py surgen_processing_prism.py

──────────────────────────────────────────────────────────────────────────────
WHERE THIS SITS IN THE FAMILY
──────────────────────────────────────────────────────────────────────────────
This is the PRISM sibling of combined_pipeline_final_error_checks_v6.py (the
batch download/process/delete orchestrator) and of surgen_titan.py (the other
slide-level aggregator). It reuses the former's batching, download, resume and
crash-diagnostics machinery verbatim, and differs in exactly one place — what
Step C produces:

  * combined_pipeline_final_error_checks_v6.py saves one small .pt PER PATCH
    (five 224 crops per 512@20x patch, kept as a (5, D) tensor) for every
    patch-level model. Across SurGen that is ~11.9M files per model.
  * PRISM is a slide-level encoder. Its Perceiver consumes the whole slide's
    tile sequence at once and emits ONE (1, 1280) vector. So Step C here saves
    one .pt PER SLIDE — 622 files, ~3 MB in total.

That difference is the reason this script exists rather than another entry in
combined_pipeline_final_error_checks_v6.py's ACTIVE_MODELS list.

TWO OUTPUTS
-----------
  1. PRISM slide embeddings — (1, 1280) per slide, in SLIDE_EMBED_DIR. This is
     what the slide classifier reads.
  2. Virchow V1 tile embeddings — (N, 2560) per slide, in TILE_EMBED_DIR,
     controlled by SAVE_TILE_EMBEDDINGS (on by default). One crop-averaged
     vector per non-white patch, paired with its patch_number. ~58 GB across
     SurGen.

Tile embeddings are streamed through a per-slide shard cache (TILE_CACHE_DIR)
as they are produced, so an interrupted slide resumes from its last shard
instead of re-encoding hours of Virchow V1. The cache is deleted once the slide
is finished; the permanent copy in TILE_EMBED_DIR is written before PRISM runs,
so a failure in aggregation never costs the expensive Stage 1 work.

──────────────────────────────────────────────────────────────────────────────
PIPELINE ORDER
──────────────────────────────────────────────────────────────────────────────
  1. Configuration        — edit all paths/settings below
  2. Crash diagnostics    — faulthandler + breadcrumb checkpoints (same as the
                            combined pipeline; see crash_diagnostics/)
  3. WSI catalogue        — built from surgen_labels.csv (label_desc == -1 is
                            excluded automatically)
  4. Non-white index      — patch_metadata_nonwhite.csv, cached to a compact
                            .npz so restarts do not re-parse 2.1 GB of CSV
  5. Per batch of 10      — download (aria2c) → Step C → delete CZIs → disk guard
  6. Step C (per slide)   — stream patches out of the CZI, Virchow V1 → one
                            2560-d tile vector each, shard-cached to disk, then
                            PRISM's Perceiver once → (1, 1280) → atomic save

Steps A-1 (per-slide RGB stats) and A-2 (pixel rule + SVM whiteness filter) are
NOT re-run here. They are model-independent and were already run for the whole
dataset — their output, patch_metadata_nonwhite.csv, is consumed as-is. If that
file is missing this script stops and tells you to run the combined pipeline
first, rather than silently producing embeddings over a different patch set
than every other model in the project.

──────────────────────────────────────────────────────────────────────────────
TWO CONVENTIONS WORTH KNOWING BEFORE YOU CHANGE ANYTHING
──────────────────────────────────────────────────────────────────────────────
1. VIRCHOW_V1_PATCH_TOKEN_START (default 5, Paige's spec is 1)

   Virchow V1 emits 257 tokens: 1 CLS + 256 patch tokens, and NO register
   tokens (verified on this machine: num_prefix_tokens=1, num_reg_tokens=0).
   Virchow **2** is the one with 4 register tokens, which is why every Virchow2
   path in this repo slices output[:, 5:].

   The project's existing PRISM features — PAIP (78 slides) and TCGA (417) —
   were extracted with output[:, 5:] applied to Virchow **V1**, i.e. averaging
   252 of the 256 patch tokens instead of all 256. The default here matches
   that so SurGen embeddings live in the same space as the cohorts they will be
   compared against. Set it to 1 for Paige's exact recipe, but if you do, PAIP
   and TCGA must be re-extracted too or cross-cohort validation is comparing
   two different feature spaces.

2. USE_FIVECROP (default True)

   The house convention across every model in this repo is FiveCrop: five
   256x256 crops per 512@20x patch, each resized to 224, encoded, and averaged.
   PRISM natively expects one embedding per tile; averaging the five crops into
   a single 2560-d vector is what the PAIP/TCGA PRISM notebooks did, so it is
   kept. It also costs 5x the GPU time — see the runtime note below.

──────────────────────────────────────────────────────────────────────────────
RUNTIME — READ THIS BEFORE STARTING THE FULL RUN
──────────────────────────────────────────────────────────────────────────────
SurGen has 11,870,098 non-white patches across 622 slides (two labelled slides
have no tissue at all). With USE_FIVECROP that is ~59.4M Virchow V1 (ViT-H/14,
632M params) forward passes.

Measured on this machine's RTX 4090, with patches read from a real CZI:
    44 patch/s end-to-end  ->  ~75 GPU-hours  ->  ~3.1 days of pure compute
The GPU is the bottleneck, not I/O: a pure forward pass tops out at 53 patch/s
while CPU-side FiveCrop preprocessing sustains 238 patch/s, so the overlapped
reader keeps the GPU ~83% fed. Budget 4-5 days wall-clock once the 622 CZI
downloads are included.

Two levers, both with consequences:
  * USE_FIVECROP = False is 5x faster but produces embeddings that are NOT
    comparable with the existing PAIP/TCGA PRISM features.
  * This torch build reports "not compiled with flash attention". A build with
    flash attention would cut ViT-H time appreciably; nothing in this script
    can work around it.

The run is fully resume-safe at two levels — finished slides are skipped, and
an interrupted slide resumes from its last completed tile shard — so stopping
and restarting costs at most SHARD_PATCHES patches of redundant work.
"""

import os
from dotenv import load_dotenv
load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# ░░  IMPORTS  ░░
# ══════════════════════════════════════════════════════════════════════════════

import gc
import json
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
import faulthandler
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
# ░░  CONFIGURATION — edit everything here  ░░
# ══════════════════════════════════════════════════════════════════════════════

# ── Download / storage settings ───────────────────────────────────────────────
BATCH_SIZE    = 10      # WSIs downloaded, processed and deleted per cycle
MIN_FREE_GB   = 20      # halt if DISK_TO_CHECK has less than this free
DISK_TO_CHECK = "F:\\"  # drive the CZIs land on — that's the one that fills up

# ── Paths ─────────────────────────────────────────────────────────────────────
# Downloads land here and are deleted from here after each batch.
#
# Deliberately NOT F:\CZI_Files. That directory is shared by
# combined_pipeline_final_error_checks_v6.py and
# Surgen_Zero_Shot_Label_Matching_Conch.py, and both of those delete a slide's
# CZI as soon as *they* are finished with it. Pointing PRISM at the same folder
# means one job can delete a multi-GB file the other is midway through reading —
# which surfaces as a random, unreproducible read failure, not an obvious error.
# A separate folder costs a second download of any slide two jobs both need, and
# buys the ability to run them at the same time without coordination.
# Point this at F:\CZI_Files only if nothing else is running.
CZI_DIR     = r"F:\CZI_Files_PRISM"
OUTPUT_BASE = r"D:\Aamir Gulzar\KSA_project2\surgen_data"
OUTPUT_ROOT = os.path.join(OUTPUT_BASE, "surgen_processed", "prism")

# Where the (1, 1280) slide embeddings go. This exact path is what
# slide_classification/config/paths.py::feature_dir("surgen", "PRISM", "PRISM")
# resolves to, so the classifier picks them up with no further configuration:
#   surgen_processed / prism / features / slide_aggregation / PRISM / prism
SLIDE_EMBED_DIR = os.path.join(
    OUTPUT_ROOT, "features", "slide_aggregation", "PRISM", "prism"
)

# Transient per-slide Virchow V1 tile embeddings (shard cache). Deleted as soon
# as the slide's PRISM embedding is written. Peak size is one slide's worth:
# the largest SurGen slide has 42,174 patches -> ~216 MB in fp16. Keep this on
# the roomy drive, not on D:.
TILE_CACHE_DIR = r"F:\prism_tile_cache"

# Input patch set — produced by the combined pipeline's Steps A-1 + A-2. Shared
# with every other model so all of them see the same patches.
NONWHITE_METADATA_CSV = os.path.join(OUTPUT_BASE, "patch_metadata_nonwhite.csv")

# Compact cached form of the four columns of the above that this script needs.
# Rebuilt automatically whenever the CSV is newer. Turns a ~4-minute startup
# parse of a 2.1 GB CSV into a ~2-second load on every restart, which matters a
# lot when the supervisor is relaunching after crashes.
NONWHITE_INDEX_NPZ = os.path.join(OUTPUT_BASE, "patch_metadata_nonwhite_index.npz")

# Labels — the single source of truth for SurGen, tracked in the repo so both
# machines run the byte-identical file. Only the `slide_filename` and `include`
# columns are used here (the catalogue); per-slide MSI labels are attached
# downstream by the classifier, not by this script.
SURGEN_LABELS_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "surgen_slide_labels.csv"
)

# ── Model settings ────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
DEVICE   = "cuda"

# Pin the exact PRISM snapshot. trust_remote_code=True means PRISM's Python
# source is downloaded from the Hub and executed — an unpinned revision could
# change the Perceiver under us mid-run. This is the revision already cached on
# this machine and used for the PAIP/TCGA PRISM features.
PRISM_REVISION = "b5ef311e78a0811c71ecf618f6f7140e9ebb5fc6"

# See "TWO CONVENTIONS" in the module docstring before changing either of these.
VIRCHOW_V1_PATCH_TOKEN_START = 5      # 1 = Paige's spec; 5 = this project's existing PRISM features
USE_FIVECROP                 = True   # False is 5x faster and 5x incomparable
CROP_SIZE                    = 256    # FiveCrop size taken out of the 512px patch

# ── Patch / extraction settings (must match the rest of the pipeline) ─────────
PATCH_SIZE         = 512   # pixels at target magnification
TARGET_MAG         = 20    # desired magnification (x)
# 32 patches = 160 crops per forward. Benchmarked on this machine's RTX 4090:
# 8->42.8, 16->43.3, 32->44.1, 48->44.4, 64->44.4 patch/s, so throughput is
# flat past 32 while VRAM keeps climbing (7.6 GB at 32, 9.5 GB at 64). 32 takes
# the plateau and leaves headroom for PRISM's cross-attention over a 42k-tile
# slide and for anything else sharing the GPU.
FEATURE_BATCH_SIZE = 32
SHARD_PATCHES      = 2048  # tile embeddings per shard file (~10 MB in fp16)

# ── Virchow V1 tile embeddings (Stage 1 output) ──────────────────────────────
# Keep the per-patch Virchow V1 tile embeddings as a permanent output, not just
# as the transient staging area PRISM consumes.
#
# COST: one 2560-d fp16 vector per patch x 11,870,098 patches = ~58 GB for the
# full SurGen dataset, ~97 MB for an average slide. Check free space on
# TILE_EMBED_DIR's drive before a full run.
#
# These are the crop-AVERAGED vectors — the exact tile sequence PRISM's
# Perceiver consumes, one vector per 512@20x patch. They are NOT the same
# artifact as the repo's other patch-level features (virchow2, conch1-5, ...),
# which keep the crop dimension as (5, D). Averaging happens before anything is
# written, so the five crops cannot be recovered from these files.
SAVE_TILE_EMBEDDINGS = True
TILE_EMBED_DIR       = r"F:\surgen_processed\virchow-v1\tile_embeddings"

# Keep all five FiveCrop vectors per patch — (N, 5, 2560) instead of (N, 2560),
# matching the (5, D) layout the repo's other patch-level models use.
#
# DOES NOT CURRENTLY FIT ON THIS MACHINE. Measured 2026-08-21:
#     crop-averaged (N, 2560)   ->  56.6 GB
#     per-crop      (N, 5, 2560) -> 283.0 GB
#     F: free                    -> 274.2 GB, and it also holds each 10-CZI
#                                   batch (~30-60 GB) plus the shard cache
# Enabling this needs ~340 GB free on TILE_EMBED_DIR's drive. The startup
# projection below refuses to start rather than filling the disk mid-run.
#
# Note the two are not additive: the averaged form is just .mean(axis=1) of the
# per-crop form, so keeping crops supersedes it rather than adding to it.
TILE_EMBED_KEEP_CROPS = False

# "per_slide"  — one <slide>.npz holding embeddings (N, 2560) fp16 plus the
#                matching patch_number (N,) int32. 622 files. Loads as one
#                array, which is what re-running PRISM or any slide-level
#                aggregation wants.
# "per_patch"  — one <slide>/<slide>_<patch_number>.pt per patch, matching the
#                directory convention of the other models. 11.9M files: far
#                slower to write and to enumerate on NTFS, and it holds a
#                (2560,) vector where the other models hold (5, D) — so it is
#                a convention match, not a drop-in replacement for them.
TILE_EMBED_FORMAT    = "per_slide"

# ── Other optional outputs ───────────────────────────────────────────────────
# PRISM's second return value: image_latents, (1, 512, 1280) — the Perceiver's
# latent bank, which its BioGPT decoder consumes for captioning/zero-shot.
# 1.3 MB/slide fp16, 0.76 GB for all of SurGen.
#
# OFF by deliberate choice, not oversight: this run saves the PRISM slide
# embeddings and the Virchow V1 tile embeddings, nothing else. Turning it on
# later means re-encoding every slide, so if PRISM captioning is on the roadmap
# it is far cheaper to enable it now than to rerun 75 GPU-hours for it.
SAVE_IMAGE_LATENTS = False
# Keep the raw shard staging directory too. Almost never wanted: with
# SAVE_TILE_EMBEDDINGS the same vectors are already saved in a tidier form,
# and this doubles their footprint.
KEEP_TILE_CACHE    = False

# ── aria2c executable ─────────────────────────────────────────────────────────
ARIA2C_EXE = r"C:\Users\datainsight\AppData\Local\Microsoft\WinGet\Packages\aria2.aria2_Microsoft.Winget.Source_8wekyb3d8bbwe\aria2-1.37.0-win-64bit-build1\aria2c.exe"

# ── EBI BioStudies mirror the WSIs are hosted on ─────────────────────────────
_BASE_URL = "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/285/S-BIAD1285/Files"

# ── Console output ────────────────────────────────────────────────────────────
# True  = one live progress bar that rewrites itself in place, carrying slide,
#         batch, rate, ETA and current phase. Per-slide chatter is folded into
#         the bar; only finished slides and real problems scroll past above it.
# False = the old behaviour, a new line for every phase of every slide.
#
# The bar is written to stderr using carriage returns, so
# run_pipeline_supervisor.py streams
# the child's output byte-for-byte instead of line-by-line — otherwise the
# frames pile up in the pipe and arrive all at once, which is what makes it
# look like every update is on its own line.
PROGRESS_SINGLE_LINE = True

# ── Hang watchdog ─────────────────────────────────────────────────────────────
# pylibCZIrw's native .read() has been observed to stall indefinitely on
# particular ROIs. If nothing makes progress for this long, dump every thread's
# stack and force-exit so the supervisor can restart us — the shard cache means
# a restart loses at most SHARD_PATCHES patches of work.
WATCHDOG_TIMEOUT_SEC = 600


# ══════════════════════════════════════════════════════════════════════════════
# ░░  DIAGNOSTIC INSTRUMENTATION  ░░
# ══════════════════════════════════════════════════════════════════════════════
# Same scheme as combined_pipeline_final_error_checks_v6.py: after a crash read
# faulthandler_<pid>.log (was it a native segfault, and where), then
# last_state.json (exact last checkpoint), then breadcrumbs.log (full timeline).

CRASH_DIAG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "crash_diagnostics")
os.makedirs(CRASH_DIAG_DIR, exist_ok=True)

_PID                   = os.getpid()
_FAULTHANDLER_LOG_PATH = os.path.join(CRASH_DIAG_DIR, f"faulthandler_prism_{_PID}.log")
_LAST_STATE_PATH       = os.path.join(CRASH_DIAG_DIR, "last_state_prism.json")
_BREADCRUMBS_PATH      = os.path.join(CRASH_DIAG_DIR, "breadcrumbs_prism.log")

_fh_log = open(_FAULTHANDLER_LOG_PATH, "a", buffering=1)
faulthandler.enable(file=_fh_log, all_threads=True)


def _sweep_empty_faulthandler_logs():
    """Delete faulthandler logs left empty by runs that did not crash natively.

    One file is created per process, and the supervisor may restart this script
    dozens of times over a multi-day run. Without this, the one log that
    actually recorded a segfault ends up buried among hundreds of empty ones —
    which defeats the reason the file exists.
    """
    for name in os.listdir(CRASH_DIAG_DIR):
        if not (name.startswith("faulthandler_prism_") and name.endswith(".log")):
            continue
        path = os.path.join(CRASH_DIAG_DIR, name)
        if path == _FAULTHANDLER_LOG_PATH:
            continue
        try:
            if os.path.getsize(path) == 0:
                os.remove(path)
        except OSError:
            pass   # still held open by a concurrent run — leave it alone


_sweep_empty_faulthandler_logs()


def _raw_log(msg: str):
    print(msg, flush=True)
    sys.stdout.flush()


def log(msg: str):
    """Print a line, cooperating with the live progress bar if there is one.

    While a bar is drawing, a bare print() lands in the middle of a frame
    and corrupts it. tqdm.write() clears the bar, emits the line, then
    redraws underneath. Putting that here rather than at each call site
    means banners, download output and cleanup messages all behave, with
    no chance of missing one.
    """
    bar = getattr(_PROGRESS, "bar", None) if "_PROGRESS" in globals() else None
    if bar is not None:
        bar.write(msg, file=sys.stderr)
    else:
        _raw_log(msg)


def _now_iso():
    return datetime.now().isoformat(timespec="seconds")


def write_checkpoint(stage: str, **details):
    """Overwrite last_state.json and append to breadcrumbs.log, both fsync'd so
    they survive a hard process kill."""
    payload = {"ts": _now_iso(), "pid": _PID, "stage": stage, **details}
    try:
        tmp = _LAST_STATE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, _LAST_STATE_PATH)
    except Exception:
        pass
    try:
        with open(_BREADCRUMBS_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
            fh.flush()
    except Exception:
        pass


def log_exception(context: str, exc: BaseException):
    write_checkpoint("exception", context=context,
                     error_type=type(exc).__name__, error=str(exc))
    try:
        with open(_BREADCRUMBS_PATH, "a", encoding="utf-8") as fh:
            fh.write(f"--- traceback for {context} ---\n")
            traceback.print_exc(file=fh)
            fh.flush()
    except Exception:
        pass


def _get_gpu_mem_snapshot():
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return {
            "alloc_mb"   : round(torch.cuda.memory_allocated() / 2**20, 1),
            "reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 1),
        }
    except Exception:
        return None


class HangWatchdog:
    """Force-exits the process if the main loop stops making progress.

    The main loop calls .heartbeat("what I am about to do") right before
    anything that could block. If no heartbeat arrives for `timeout` seconds we
    assume we are wedged inside a native call that will never return control to
    Python (so no amount of asking nicely would help), dump every thread's
    stack, and os._exit(1). The supervisor restarts us and the shard cache
    resumes the interrupted slide.
    """

    def __init__(self, timeout: int = WATCHDOG_TIMEOUT_SEC, poll_interval: int = 5):
        self.timeout       = timeout
        self.poll_interval = poll_interval
        self._last_beat    = time.monotonic()
        self._where        = "startup"
        self._lock         = threading.Lock()
        self._stop_event   = threading.Event()
        self._thread       = threading.Thread(target=self._watch, daemon=True)

    def start(self):
        """Idempotent and re-startable. A Thread object can only be started
        once, so a stopped watchdog gets a fresh one — otherwise calling main()
        twice in a process (tests, notebooks) dies on "threads can only be
        started once" before doing any work."""
        if self._thread.is_alive():
            return
        self._stop_event.clear()
        if self._thread.ident is not None:      # this Thread object already ran
            self._thread = threading.Thread(target=self._watch, daemon=True)
        self._last_beat = time.monotonic()
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def heartbeat(self, where: str):
        with self._lock:
            self._last_beat = time.monotonic()
            self._where     = where

    def _watch(self):
        while not self._stop_event.wait(self.poll_interval):
            with self._lock:
                elapsed = time.monotonic() - self._last_beat
                where   = self._where
            if elapsed > self.timeout:
                log("\n" + "=" * 78)
                log(f"[WATCHDOG] No progress for {elapsed:.0f}s (limit {self.timeout}s).")
                log(f"[WATCHDOG] Last known activity: {where}")
                log("[WATCHDOG] Full thread stack traces:")
                faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                sys.stderr.flush()
                write_checkpoint("watchdog_forced_exit", where=where,
                                 stalled_sec=round(elapsed, 1))
                log("[WATCHDOG] Forcing exit. Re-run (or let the supervisor "
                    "relaunch): the slide resumes from its last tile shard.")
                log("=" * 78)
                sys.stdout.flush()
                os._exit(1)


_WATCHDOG = HangWatchdog()


# ══════════════════════════════════════════════════════════════════════════════
# ░░  PROGRESS BAR  ░░
# ══════════════════════════════════════════════════════════════════════════════

class RunProgress:
    """One bar for the whole run, measured in patches rather than slides.

    Slides vary from 481 to 42,174 patches, so a slide-counting bar would jump
    unevenly and give a useless ETA. Counting patches makes the rate directly
    comparable to the benchmark number (~44 patch/s) and the ETA meaningful.

    Everything that used to be its own print — which slide, which batch, what
    phase, how fast — becomes part of this one line. Genuine events (a slide
    finished, an error) go through .write(), which tqdm clears the bar for and
    redraws underneath, so the bar always stays on the bottom line.
    """

    def __init__(self, total_patches: int, enabled: bool = True):
        self.enabled = enabled and total_patches > 0
        self.bar = None
        self._slide = ""
        self._phase = ""
        self._batch = ""
        self._slide_no = ""
        if self.enabled:
            self.bar = tqdm(
                total=total_patches, unit="patch", unit_scale=True,
                dynamic_ncols=True, mininterval=0.5, smoothing=0.05,
                file=sys.stderr,
                # Two decimals on the percentage, not zero. Over 11.8M
                # patches a whole-number percentage sits on "0%" for the
                # first 90 minutes and ticks once every ~50 minutes after
                # that, which reads as a stalled run. At 2 decimals it
                # moves every ~30s, so the bar visibly reflects progress.
                bar_format="{desc} {percentage:5.2f}%|{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            )

    # -- context -----------------------------------------------------------
    def set_batch(self, batch_idx: int, n_batches: int):
        self._batch = f"b{batch_idx}/{n_batches}"
        self._redraw()

    def set_slide(self, stem: str, done: int, total: int):
        self._slide, self._slide_no = stem, f"{done}/{total}"
        self._redraw()

    def set_phase(self, phase: str):
        self._phase = phase
        self._redraw()

    def _redraw(self):
        if not self.bar:
            return
        desc = f"[{self._batch}] {self._slide} ({self._slide_no})"
        if self._phase:
            desc += f" {self._phase}"
        self.bar.set_description_str(desc[:70], refresh=False)

    # -- movement ----------------------------------------------------------
    def advance(self, n: int):
        if self.bar:
            self.bar.update(n)

    def write(self, msg: str):
        """Print a line above the bar without corrupting it."""
        if self.bar:
            self.bar.write(msg, file=sys.stderr)
        else:
            _raw_log(msg)

    def close(self):
        if self.bar:
            self.bar.close()
            self.bar = None


#: Set up in main() once the amount of outstanding work is known. Until then
#: it is a disabled instance, so every call site works before/without a bar.
_PROGRESS = RunProgress(0, enabled=False)


def plog(msg: str):
    """Kept as an explicit alias; log() is bar-aware on its own now."""
    log(msg)


# ══════════════════════════════════════════════════════════════════════════════
# ░░  WSI CATALOGUE  ░░
# ══════════════════════════════════════════════════════════════════════════════

def build_wsi_list(labels_csv: str) -> list:
    """(filename, url) for every slide with include == TRUE in
    surgen_slide_labels.csv. The EBI subfolder is inferred from the slide name
    prefix (SR1482 -> SR1482_WSIs, SR386 -> SR386_WSIs) — identical to the
    combined pipeline's catalogue."""
    df = pd.read_csv(labels_csv)

    required = {"slide_filename", "include"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Expected columns {sorted(required)} in {labels_csv}. "
            f"Found: {df.columns.tolist()}"
        )

    included = df["include"].astype(str).str.strip().str.upper() == "TRUE"
    valid_stems = [Path(str(f)).stem for f in df.loc[included, "slide_filename"]]

    wsis, skipped = [], 0
    for stem in valid_stems:
        subfolder = _ebi_subfolder(stem)
        if subfolder is None:
            log(f"[WARN] Unrecognised prefix for '{stem}' — skipping.")
            skipped += 1
            continue
        wsis.append((stem + ".czi", f"{_BASE_URL}/{subfolder}/{stem}.czi"))

    log(f"[CATALOGUE] {len(wsis)} valid WSI(s) from {labels_csv} "
        f"({len(df) - int(included.sum())} excluded (include != TRUE)"
        + (f", {skipped} skipped due to unknown prefix" if skipped else "") + ")")
    return wsis


def _ebi_subfolder(stem: str):
    stem_upper = stem.upper()
    if stem_upper.startswith("SR1482"):
        return "SR1482_WSIs"
    if stem_upper.startswith("SR386"):
        return "SR386_WSIs"
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ░░  NON-WHITE PATCH INDEX  ░░
# ══════════════════════════════════════════════════════════════════════════════

def load_nonwhite_index() -> dict:
    """Return {slide_stem: (patch_number, patch_x, patch_y) int32 arrays, sorted
    by patch_number}.

    patch_metadata_nonwhite.csv is 2.1 GB and has 15 columns; this script needs
    4 of them. Parsing it costs minutes and several GB of RAM, which would be
    paid again on every supervisor restart, so the compact form is cached to a
    .npz and reused until the CSV changes.
    """
    if not os.path.exists(NONWHITE_METADATA_CSV):
        raise FileNotFoundError(
            f"Non-white patch metadata not found:\n  {NONWHITE_METADATA_CSV}\n"
            "This file is produced by Steps A-1/A-2 of "
            "combined_pipeline_final_error_checks_v6.py and is shared by every "
            "model. Run that pipeline's preprocessing first — do not let PRISM "
            "run over a different patch set than the other models."
        )

    csv_mtime = os.path.getmtime(NONWHITE_METADATA_CSV)

    if (os.path.exists(NONWHITE_INDEX_NPZ)
            and os.path.getmtime(NONWHITE_INDEX_NPZ) >= csv_mtime):
        log(f"[INDEX] Loading cached patch index from {NONWHITE_INDEX_NPZ}")
        with np.load(NONWHITE_INDEX_NPZ, allow_pickle=False) as z:
            slides  = z["slides"]
            offsets = z["offsets"]
            numbers = z["patch_number"]
            xs      = z["patch_x"]
            ys      = z["patch_y"]
    else:
        log(f"[INDEX] Building patch index from {NONWHITE_METADATA_CSV} "
            f"(one-off; ~2.1 GB to parse) ...")
        t0 = time.time()
        parts = []
        reader = pd.read_csv(
            NONWHITE_METADATA_CSV,
            usecols=["slide_name", "patch_number", "patch_x", "patch_y"],
            dtype={"slide_name": "string", "patch_number": "int32",
                   "patch_x": "int32", "patch_y": "int32"},
            chunksize=2_000_000,
        )
        for i, chunk in enumerate(reader, 1):
            parts.append(chunk)
            log(f"  [INDEX] read chunk {i} ({sum(len(p) for p in parts):,} rows so far)")
        df = pd.concat(parts, ignore_index=True)
        del parts

        # Group-contiguous layout: sort by (slide, patch_number) once, then a
        # slide is a contiguous slice — no per-slide DataFrame objects to hold.
        df = df.sort_values(["slide_name", "patch_number"], kind="mergesort")
        slide_col = df["slide_name"].to_numpy(dtype=object).astype(str)
        slides, starts = np.unique(slide_col, return_index=True)
        order   = np.argsort(starts)
        slides  = slides[order]
        starts  = starts[order]
        offsets = np.append(starts, len(df)).astype(np.int64)
        numbers = df["patch_number"].to_numpy(dtype=np.int32)
        xs      = df["patch_x"].to_numpy(dtype=np.int32)
        ys      = df["patch_y"].to_numpy(dtype=np.int32)
        del df

        np.savez(NONWHITE_INDEX_NPZ, slides=slides.astype(str), offsets=offsets,
                 patch_number=numbers, patch_x=xs, patch_y=ys)
        log(f"[INDEX] Built and cached in {time.time() - t0:.1f}s -> {NONWHITE_INDEX_NPZ}")

    index = {}
    for i, stem in enumerate(slides):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        index[str(stem)] = (numbers[lo:hi], xs[lo:hi], ys[lo:hi])

    total = sum(len(v[0]) for v in index.values())
    log(f"[INDEX] {len(index)} slide(s) with tissue, {total:,} non-white patch(es)")
    return index


# ══════════════════════════════════════════════════════════════════════════════
# ░░  DISK SPACE  ░░
# ══════════════════════════════════════════════════════════════════════════════

def get_free_gb(path: str) -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def project_storage_or_halt(nonwhite_index: dict, slides_remaining: int):
    """Project total output size for the enabled options and refuse to start if
    it cannot fit.

    A run that fills the disk on day two of four has wasted two days and leaves
    a half-written output tree. The projection is cheap and exact — the patch
    count is known up front from the index — so there is no reason to discover
    this the hard way.
    """
    patches_remaining = sum(
        len(nonwhite_index[s][0]) for s in nonwhite_index
        if not os.path.exists(slide_embedding_path(s))
    )

    lines, need = [], {}

    def _add(label, drive, nbytes):
        need[drive] = need.get(drive, 0) + nbytes
        lines.append(f"    {label:<44} {nbytes / 2**30:>8.2f} GB  -> {drive}")

    _add("PRISM slide embeddings (1, 1280)",
         os.path.splitdrive(SLIDE_EMBED_DIR)[0] or "C:",
         slides_remaining * 1280 * 4)

    if SAVE_IMAGE_LATENTS:
        _add("PRISM image_latents (1, 512, 1280)",
             os.path.splitdrive(SLIDE_EMBED_DIR)[0] or "C:",
             slides_remaining * 512 * 1280 * 2)

    if SAVE_TILE_EMBEDDINGS:
        per_patch_bytes = 5 * 2560 * 2 if TILE_EMBED_KEEP_CROPS else 2560 * 2
        _add(f"Virchow V1 tiles "
             f"({'N, 5, 2560' if TILE_EMBED_KEEP_CROPS else 'N, 2560'})",
             os.path.splitdrive(TILE_EMBED_DIR)[0] or "C:",
             patches_remaining * per_patch_bytes)

    log("\n[STORAGE] Projected output for the work still outstanding "
        f"({slides_remaining:,} slide(s), {patches_remaining:,} patch(es)):")
    for line in lines:
        log(line)

    halt = False
    for drive, nbytes in sorted(need.items()):
        root = drive + "\\"
        try:
            free = get_free_gb(root)
        except Exception:
            log(f"    [WARN] Could not read free space for {root}")
            continue
        want = nbytes / 2**30
        # CZI batches land on CZI_DIR's drive and are transient, but they must
        # coexist with whatever is being written there, so reserve for them.
        reserve = MIN_FREE_GB
        if drive.upper() == (os.path.splitdrive(CZI_DIR)[0] or "").upper():
            reserve += 60      # a 10-CZI batch at ~6 GB/slide
        verdict = "OK" if free >= want + reserve else "NOT ENOUGH"
        log(f"    {root} needs {want:.1f} GB + {reserve:.0f} GB headroom, "
            f"{free:.1f} GB free  -> {verdict}")
        if verdict != "OK":
            halt = True

    if halt:
        log("\n[HALT] The enabled outputs do not fit. Either free space, point "
            "TILE_EMBED_DIR at a bigger drive, or turn off the option that "
            "dominates the projection (TILE_EMBED_KEEP_CROPS costs 5x "
            "SAVE_TILE_EMBEDDINGS).")
        write_checkpoint("storage_projection_halt", need_gb=
                         {d: round(b / 2**30, 1) for d, b in need.items()})
        sys.exit(1)


def check_disk_space_or_halt(threshold_gb: float = MIN_FREE_GB):
    free_gb = get_free_gb(DISK_TO_CHECK)
    log(f"\n[DISK CHECK] Free on {DISK_TO_CHECK}: {free_gb:.1f} GB "
        f"(threshold: {threshold_gb} GB)")
    if free_gb < threshold_gb:
        log(f"[HALT] Less than {threshold_gb} GB free on {DISK_TO_CHECK}. "
            f"Stopping to protect the drive.")
        write_checkpoint("disk_halt", free_gb=round(free_gb, 1))
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# ░░  DOWNLOAD / DELETE  ░░
# ══════════════════════════════════════════════════════════════════════════════

_ARIA2C_RESOLVED = None


def _resolve_aria2c() -> str:
    global _ARIA2C_RESOLVED
    if _ARIA2C_RESOLVED:
        return _ARIA2C_RESOLVED
    resolved = shutil.which(ARIA2C_EXE) or (ARIA2C_EXE if os.path.isfile(ARIA2C_EXE) else None)
    if resolved is None:
        log("\n[FATAL] aria2c not found!")
        log(f"  Tried : '{ARIA2C_EXE}'")
        log("  Fix   : set ARIA2C_EXE to the full path of aria2c.exe")
        sys.exit(1)
    _ARIA2C_RESOLVED = resolved
    return resolved


def download_batch(batch: list):
    """Download (filename, url) pairs with aria2c, skipping any already present.

    aria2c is polled rather than waited on so the watchdog keeps getting
    heartbeats — a multi-GB CZI can easily take longer than WATCHDOG_TIMEOUT_SEC
    and must not be mistaken for a hang.
    """
    os.makedirs(CZI_DIR, exist_ok=True)
    log(f"\n{'═' * 70}")
    log(f"[DOWNLOAD] {len(batch)} WSI(s) → {CZI_DIR}")
    log(f"{'═' * 70}")

    aria2c = _resolve_aria2c()

    # aria2c prints its own progress directly to the console; leaving our
    # bar on screen underneath it produces two things redrawing the same
    # lines. Hide it until the batch is downloaded.
    if _PROGRESS.bar is not None:
        _PROGRESS.bar.clear()

    incomplete = set()

    for i, (filename, url) in enumerate(batch, 1):
        out_path  = os.path.join(CZI_DIR, filename)
        ctrl_path = out_path + ".aria2"
        log(f"\n  [{i}/{len(batch)}] {filename}")

        # aria2c preallocates the file at its FULL final size on the first byte
        # written, so size — even an exact match against Content-Length — says
        # nothing about completeness. The control file is the only reliable
        # signal: aria2c removes it if and only if the download finished. A
        # leftover one means the bytes are still full of holes, and reading a
        # patch that lands in a hole fails silently and drops it from the
        # slide's embedding. Resume (-c picks up where it stopped) rather than
        # reuse the partial.
        if os.path.exists(out_path) and not os.path.exists(ctrl_path):
            log(f"    [SKIP] Already present ({os.path.getsize(out_path) / 2**30:.2f} GB)")
            continue
        if os.path.exists(ctrl_path):
            log(f"    [RESUME] Partial download found (.aria2 control file present)"
                f" — resuming rather than reusing the incomplete file.")

        check_disk_space_or_halt()
        write_checkpoint("download_start", filename=filename, url=url)

        cmd = [aria2c, "-x", "16", "-s", "16", "-c", "-d", CZI_DIR, "-o", filename, url]
        t0 = time.time()
        try:
            proc = subprocess.Popen(cmd)
        except Exception as e:
            log_exception(f"download_spawn {filename}", e)
            log(f"    [ERROR] Could not launch aria2c: {type(e).__name__}: {e}")
            incomplete.add(Path(filename).stem)
            continue

        while proc.poll() is None:
            _WATCHDOG.heartbeat(f"downloading {filename} via aria2c")
            time.sleep(5)

        elapsed = time.time() - t0
        # Both conditions matter: a zero exit code with the control file still
        # on disk has been observed, and that file is still full of holes.
        finished = (proc.returncode == 0
                    and os.path.exists(out_path)
                    and not os.path.exists(ctrl_path))
        if finished:
            log(f"    [OK] {os.path.getsize(out_path) / 2**30:.2f} GB in {elapsed:.1f}s")
            write_checkpoint("download_ok", filename=filename,
                             elapsed_sec=round(elapsed, 1))
        else:
            incomplete.add(Path(filename).stem)
            log(f"    [ERROR] aria2c exited with code {proc.returncode} "
                f"after {elapsed:.1f}s for {filename}"
                + ("  (control file still present — file is INCOMPLETE)"
                   if os.path.exists(ctrl_path) else ""))
            log(f"    [SKIP SLIDE] {filename} will not be encoded this pass — "
                f"an incomplete CZI silently drops patches from the embedding.")
            write_checkpoint("download_failed", filename=filename,
                             returncode=proc.returncode,
                             ctrl_present=os.path.exists(ctrl_path))

    # Downloads are done; bring the bar back for the encoding phase.
    if _PROGRESS.bar is not None:
        _PROGRESS.bar.refresh()

    return incomplete


def delete_batch_czis(batch: list):
    """Delete this batch's CZIs — the whole point of batching. Runs even for
    slides that failed to process, because a failed slide's CZI is re-downloaded
    on the next pass anyway and keeping it around defeats the disk budget."""
    log(f"\n[CLEANUP] Deleting {len(batch)} CZI file(s) ...")
    freed = 0.0
    for filename, _ in batch:
        path = os.path.join(CZI_DIR, filename)
        if os.path.exists(path):
            try:
                size_gb = os.path.getsize(path) / 2**30
                os.remove(path)
                freed += size_gb
                log(f"  Deleted: {filename} ({size_gb:.2f} GB)")
            except Exception as e:
                log(f"  [WARN] Could not delete {filename}: {e}")
        else:
            log(f"  [SKIP] Not found (already gone?): {filename}")

        ctrl = path + ".aria2"
        if os.path.exists(ctrl):
            try:
                os.remove(ctrl)
            except Exception:
                pass

    log(f"[CLEANUP] Freed {freed:.2f} GB")
    write_checkpoint("batch_czis_deleted", n=len(batch), freed_gb=round(freed, 2))


# ══════════════════════════════════════════════════════════════════════════════
# ░░  COMPLETION CHECKS (resume)  ░░
# ══════════════════════════════════════════════════════════════════════════════

def slide_embedding_path(slide_stem: str) -> str:
    return os.path.join(SLIDE_EMBED_DIR, slide_stem + ".pt")


def _slide_complete(slide_stem: str, nonwhite_index: dict) -> bool:
    """A slide is done if its embedding exists, or if it has no tissue at all.

    The second case matters: patch_metadata_nonwhite.csv covers the whole
    dataset, so a slide absent from it has been *proven* to have zero usable
    patches. Without this it would be re-downloaded and re-examined on every
    single run, forever, to produce nothing.
    """
    if slide_stem not in nonwhite_index:
        return True
    if not os.path.exists(slide_embedding_path(slide_stem)):
        return False
    # A slide finished before SAVE_TILE_EMBEDDINGS was switched on has its
    # PRISM embedding but no tile embeddings. Without this it would count as
    # complete and never produce them, leaving a silently partial tile set.
    if SAVE_TILE_EMBEDDINGS and not tile_embeddings_exist(slide_stem):
        return False
    return True


def _batch_fully_complete(batch: list, nonwhite_index: dict, verbose: bool = False) -> bool:
    all_complete = True
    for filename, _ in batch:
        stem = Path(filename).stem
        done = _slide_complete(stem, nonwhite_index)
        if verbose:
            log(f"    [RESUME SCAN]   {filename} — {'complete' if done else 'INCOMPLETE'}")
        if not done:
            all_complete = False
            if not verbose:
                return False
    return all_complete


def filter_batch_for_download(batch: list, nonwhite_index: dict) -> list:
    to_download = []
    for filename, url in batch:
        stem = Path(filename).stem
        if stem not in nonwhite_index:
            log(f"  [SKIP DOWNLOAD] {filename} — confirmed zero tissue patches")
        elif _slide_complete(stem, nonwhite_index):
            log(f"  [SKIP DOWNLOAD] {filename} — already fully processed")
        else:
            to_download.append((filename, url))
    return to_download


# ══════════════════════════════════════════════════════════════════════════════
# ░░  CZI HELPERS  ░░
# ══════════════════════════════════════════════════════════════════════════════

def find_all_czi_files(root: str) -> dict:
    """{stem: path} for every .czi currently under root."""
    index = {}
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(".czi"):
                index[Path(f).stem] = os.path.join(dirpath, f)
    return index


def open_czi_at_target_mag(czi_path: str, target_mag: int = 20, patch_size: int = 512):
    """Geometry needed to map nonwhite-CSV patch coordinates onto native CZI
    pixels. Identical to the combined pipeline's version — the coordinate scheme
    must match exactly or the patches read here are not the patches the metadata
    describes."""
    from pylibCZIrw import czi as pyczi

    with pyczi.open_czi(czi_path) as czidoc:
        try:
            native_mag = int(
                czidoc.metadata["ImageDocument"]["Metadata"]["Information"]
                ["Instrument"]["Objectives"]["Objective"]["NominalMagnification"]
            )
        except Exception:
            native_mag = 40

        bbox    = czidoc.total_bounding_box
        X_start = bbox["X"][0]
        Y_start = bbox["Y"][0]
        W       = bbox["X"][1] - bbox["X"][0]
        H       = bbox["Y"][1] - bbox["Y"][0]

        downsample = max(1, native_mag // target_mag)

    return {
        "czi_path"  : czi_path,
        "W"         : W, "H": H,
        "X_start"   : X_start, "Y_start": Y_start,
        "native_mag": native_mag,
        "level_mag" : native_mag // downsample,
        "downsample": downsample,
    }


# ──────────────────────────────────────────────────────────────────────────────
# PERSISTENT CZI READER THREAD
# ──────────────────────────────────────────────────────────────────────────────
# pylibCZIrw's decoder keeps native, thread-affine state. Creating and tearing
# down a worker thread per slide (as an earlier version of the combined pipeline
# did) caused intermittent access violations — always on the first read of a
# freshly spawned thread. One reader thread for the whole process fixes it.
# Reads stay strictly serialized (max_workers=1); only the thread identity is
# now stable.
_CZI_READER_POOL = None
_CZI_READER_POOL_LOCK = threading.Lock()


def _get_czi_reader_pool():
    global _CZI_READER_POOL
    with _CZI_READER_POOL_LOCK:
        if _CZI_READER_POOL is None:
            from concurrent.futures import ThreadPoolExecutor
            _CZI_READER_POOL = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="czi-reader"
            )
    return _CZI_READER_POOL


def _shutdown_czi_reader_pool():
    global _CZI_READER_POOL
    with _CZI_READER_POOL_LOCK:
        if _CZI_READER_POOL is not None:
            try:
                _CZI_READER_POOL.shutdown(wait=True, cancel_futures=False)
            except Exception:
                pass
            _CZI_READER_POOL = None


import atexit
atexit.register(_shutdown_czi_reader_pool)


def _read_patch_chunk(czidoc, numbers, xs, ys, slide_stem,
                      x_start, y_start, downsample, size_native, zoom):
    """Read one chunk of patches sequentially on whichever thread calls this.

    Must only ever be called from the single reader thread — czidoc is not
    thread-safe. Returns (images, ok_count); a patch that fails to read is
    dropped, which is why the caller tracks rows-consumed separately from
    embeddings-produced.
    """
    from PIL import Image

    images, kept = [], []
    for n, px, py in zip(numbers, xs, ys):
        x_native = x_start + (int(px) * downsample)
        y_native = y_start + (int(py) * downsample)
        try:
            raw = czidoc.read(
                roi=(x_native, y_native, size_native, size_native),
                zoom=zoom, plane={"C": 0, "Z": 0, "T": 0},
            )
            images.append(Image.fromarray(raw[:, :, :3].astype("uint8")).convert("RGB"))
            kept.append(int(n))
        except Exception as e:
            log_exception(f"patch_read slide={slide_stem} patch={int(n)}", e)
            tqdm.write(f"    [ERROR] patch {slide_stem}_{int(n)}: {e}")
    return images, np.asarray(kept, dtype=np.int32)


# ══════════════════════════════════════════════════════════════════════════════
# ░░  TILE SHARD CACHE  ░░
# ══════════════════════════════════════════════════════════════════════════════

class TileShardCache:
    """Crash-resumable staging area for one slide's Virchow V1 tile embeddings.

    A single SurGen slide holds up to 42k patches — several GPU-hours with
    FiveCrop. Losing that to a native CZI crash and restarting the slide from
    zero could mean a slide that never finishes at all under a crash loop. So
    tile embeddings are appended to disk in shards of SHARD_PATCHES as they are
    produced, and a restart picks up from the last complete shard.

    Shards are named ``{start_row:09d}_{end_row:09d}.npz`` where the rows are
    indices into this slide's patch list. A shard may hold FEWER vectors than
    its row span if some patches failed to read — that is exactly why the span
    is in the name rather than inferred from the array length, so the resume
    cursor tracks rows consumed, never rows produced.

    Because of that, each shard stores its patch numbers (``n``) next to the
    vectors (``v``). Row position alone would NOT identify a patch once any
    read has failed, and the saved tile embeddings would then be silently
    mislabelled — so the mapping is carried explicitly rather than inferred.

    Peak disk is one slide: 42,174 x 2560 fp16 = 216 MB. The directory is
    removed as soon as the slide's PRISM embedding is written.
    """

    def __init__(self, slide_stem: str, n_rows: int):
        self.slide_stem = slide_stem
        self.n_rows     = n_rows
        self.dir        = os.path.join(TILE_CACHE_DIR, slide_stem)
        os.makedirs(self.dir, exist_ok=True)

    def _shards(self):
        """Contiguous shards from row 0, in order. Stops at the first gap or
        overlap — anything after that is unusable and gets discarded."""
        found = []
        for name in sorted(os.listdir(self.dir)):
            if not name.endswith(".npz"):
                continue
            try:
                start_s, end_s = os.path.splitext(name)[0].split("_")
                found.append((int(start_s), int(end_s), os.path.join(self.dir, name)))
            except ValueError:
                continue
        found.sort()

        good, cursor = [], 0
        for start, end, path in found:
            if start != cursor:
                break
            good.append((start, end, path))
            cursor = end
        return good, cursor

    def resume_row(self) -> int:
        """First row index not yet encoded. Also drops any shard past a gap and
        any .tmp left behind by a crash mid-write."""
        good, cursor = self._shards()
        good_paths = {p for _, _, p in good}
        for name in os.listdir(self.dir):
            path = os.path.join(self.dir, name)
            if name.endswith(".tmp") or (name.endswith(".npz") and path not in good_paths):
                try:
                    os.remove(path)
                    log(f"    [SHARD] Discarded unusable shard file {name}")
                except Exception:
                    pass
        return cursor

    def write(self, start_row: int, end_row: int, vectors, numbers):
        """Append one shard. Written to .tmp then renamed, so a crash mid-write
        can never leave a torn shard that resume would trust."""
        arr = vectors.numpy() if hasattr(vectors, "numpy") else np.asarray(vectors)
        num = np.asarray(numbers, dtype=np.int32)
        if len(arr) != len(num):
            raise RuntimeError(
                f"{self.slide_stem}: shard {start_row}-{end_row} has {len(arr)} "
                f"vector(s) but {len(num)} patch number(s) — refusing to write a "
                f"shard whose patch mapping is already wrong."
            )
        final = os.path.join(self.dir, f"{start_row:09d}_{end_row:09d}.npz")
        tmp   = final + ".tmp"
        with open(tmp, "wb") as fh:
            np.savez(fh, v=arr.astype(np.float16, copy=False), n=num)
        os.replace(tmp, final)

    def load_all(self):
        """Every cached tile vector plus its patch number, in patch order.

        Returns ``(embeddings (N, 2560) fp16, patch_numbers (N,) int32)``.
        """
        good, cursor = self._shards()
        if cursor != self.n_rows:
            raise RuntimeError(
                f"{self.slide_stem}: tile cache covers {cursor} of {self.n_rows} "
                f"rows — refusing to aggregate a partial slide."
            )
        if not good:
            empty = ((0, 5, 2560) if TILE_EMBED_KEEP_CROPS else (0, 2560))
            return (np.zeros(empty, dtype=np.float16),
                    np.zeros((0,), dtype=np.int32))

        vecs, nums = [], []
        for _, _, path in good:
            with np.load(path) as z:
                vecs.append(z["v"])
                nums.append(z["n"])
        return np.concatenate(vecs, axis=0), np.concatenate(nums, axis=0)

    def cleanup(self):
        if KEEP_TILE_CACHE:
            return
        try:
            shutil.rmtree(self.dir, ignore_errors=True)
        except Exception as e:
            log(f"    [WARN] Could not remove tile cache {self.dir}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# ░░  MODEL — Virchow V1 tile encoder + PRISM Perceiver  ░░
# ══════════════════════════════════════════════════════════════════════════════

class PrismExtractor:
    """Virchow V1 encodes patches; PRISM's Perceiver turns a slide's worth of
    tile vectors into one slide embedding.

    Virchow V1 — not Virchow2 — is deliberate: PRISM's Perceiver was pretrained
    on V1 embeddings. V2 produces a different distribution and would silently
    degrade PRISM without erroring anywhere.
    """

    def __init__(self, device: str = DEVICE):
        import torch
        import timm
        from timm.data import resolve_data_config
        from timm.layers import SwiGLUPacked
        from torchvision import transforms
        from transformers import AutoModel
        from huggingface_hub import login

        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if self.device.type != device:
            log(f"  [WARN] CUDA unavailable — falling back to {self.device}. "
                f"This run would take weeks on CPU.")

        if not HF_TOKEN:
            raise ValueError(
                "HF_TOKEN is not set. Both paige-ai/Virchow and paige-ai/Prism "
                "are gated repos — put HF_TOKEN in the .env file at the repo root."
            )
        login(token=HF_TOKEN)

        write_checkpoint("load_virchow_start")
        log(f"  Loading Virchow V1 on {self.device} ...")
        self.tile_encoder = timm.create_model(
            "hf-hub:paige-ai/Virchow", pretrained=True,
            mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU, num_classes=0,
        ).to(self.device).eval()

        cfg = resolve_data_config(self.tile_encoder.pretrained_cfg, model=self.tile_encoder)
        per_crop = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg["mean"], std=cfg["std"]),
        ])
        if USE_FIVECROP:
            self.transform = transforms.Compose([
                transforms.FiveCrop(CROP_SIZE),
                transforms.Lambda(lambda crops: torch.stack([per_crop(c) for c in crops])),
            ])
            self.crops_per_patch = 5
        else:
            self.transform = transforms.Lambda(lambda img: per_crop(img).unsqueeze(0))
            self.crops_per_patch = 1

        # Verify the token layout rather than trusting it. Virchow V1 is
        # 1 CLS + 256 patch tokens and no registers; Virchow2 is the one with
        # 4 registers. Getting this wrong is silent — the shapes still work.
        with torch.inference_mode():
            probe = self.tile_encoder.forward_features(
                torch.zeros(1, 3, 224, 224, device=self.device)
            )
        n_tokens = probe.shape[1]
        n_prefix = getattr(self.tile_encoder, "num_prefix_tokens", 1)
        del probe
        if n_tokens != 257:
            raise RuntimeError(
                f"Expected Virchow V1 to emit 257 tokens (1 CLS + 256 patch), "
                f"got {n_tokens}. That is not Virchow V1 — check the hub id."
            )
        dropped = VIRCHOW_V1_PATCH_TOKEN_START - n_prefix
        log(f"  Virchow V1 ready. {n_tokens} tokens, {n_prefix} prefix token(s), "
            f"embedding dim 2560, {self.crops_per_patch} crop(s)/patch.")
        log(f"  Patch tokens averaged from index {VIRCHOW_V1_PATCH_TOKEN_START} "
            f"({dropped} patch token(s) dropped — "
            f"{'Paige spec' if dropped == 0 else 'matches this project existing PAIP/TCGA PRISM features'}).")

        write_checkpoint("load_prism_start")
        log(f"  Loading PRISM (rev {PRISM_REVISION[:8]}) on {self.device} ...")
        self.prism = AutoModel.from_pretrained(
            "paige-ai/Prism", trust_remote_code=True, revision=PRISM_REVISION,
        ).to(self.device).eval()
        log("  PRISM ready. Slide embedding dim 1280.")
        write_checkpoint("models_ready", gpu_mem=_get_gpu_mem_snapshot())

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    def encode_tiles(self, images):
        """[PIL.Image, ...] -> fp16 tile embeddings on CPU.

        Shape depends on TILE_EMBED_KEEP_CROPS:
          False -> (B, 2560)      crops averaged, one vector per patch
          True  -> (B, 5, 2560)   every crop kept; PRISM still gets the mean,
                                  taken later in process_slide
        """
        torch = self.torch
        batch = torch.cat([self.transform(img) for img in images], dim=0)
        if self.device.type == "cuda":
            batch = batch.pin_memory().to(self.device, non_blocking=True)
        else:
            batch = batch.to(self.device)

        with torch.inference_mode():
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                out = self.tile_encoder.forward_features(batch)
            cls_token  = out[:, 0].float()
            patch_mean = out[:, VIRCHOW_V1_PATCH_TOKEN_START:].float().mean(dim=1)
            combined   = torch.cat([cls_token, patch_mean], dim=-1)   # (B*crops, 2560)

        combined = combined.reshape(len(images), self.crops_per_patch, 2560)
        if not TILE_EMBED_KEEP_CROPS:
            # Average the crops back down to one vector per patch — PRISM's
            # tile sequence has one entry per tile, not one per crop.
            combined = combined.mean(dim=1)
        return combined.to(torch.float16).cpu()

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    def aggregate(self, tiles_np):
        """(N, 2560) tile embeddings -> ((1, 1280) embedding, (1, 512, 1280) latents).

        Falls back to CPU on CUDA OOM rather than losing the slide: the largest
        SurGen slide is 42k tiles and the Perceiver's cross-attention over that
        is the one place in this pipeline where VRAM could realistically run out.
        """
        torch = self.torch
        tiles = torch.from_numpy(np.ascontiguousarray(tiles_np)).float().unsqueeze(0)

        try:
            with torch.inference_mode():
                reprs = self.prism.slide_representations(tiles.to(self.device))
            emb     = reprs["image_embedding"].float().cpu()
            latents = reprs.get("image_latents")
            latents = latents.float().cpu() if latents is not None else None
            return emb, latents
        except torch.cuda.OutOfMemoryError as e:
            log(f"    [OOM] PRISM aggregation over {tiles.shape[1]:,} tiles did not "
                f"fit in VRAM ({e}); retrying on CPU.")
            write_checkpoint("prism_aggregate_oom", n_tiles=int(tiles.shape[1]))
            torch.cuda.empty_cache()
            self.prism.to("cpu")
            try:
                with torch.inference_mode():
                    reprs = self.prism.slide_representations(tiles)
                emb     = reprs["image_embedding"].float().cpu()
                latents = reprs.get("image_latents")
                latents = latents.float().cpu() if latents is not None else None
                return emb, latents
            finally:
                self.prism.to(self.device)


# ══════════════════════════════════════════════════════════════════════════════
# ░░  ATOMIC SAVE  ░░
# ══════════════════════════════════════════════════════════════════════════════

def atomic_save(tensor, path: str):
    """torch.save to .tmp then rename. Resume logic treats "the file exists" as
    "the slide is done", so a half-written file would poison the run."""
    import torch
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(tensor, tmp)
    os.replace(tmp, path)


# ══════════════════════════════════════════════════════════════════════════════
# ░░  VIRCHOW V1 TILE EMBEDDINGS (Stage 1 output)  ░░
# ══════════════════════════════════════════════════════════════════════════════

def tile_embeddings_path(slide_stem: str) -> str:
    """Where this slide's tile embeddings live. For "per_patch" this is the
    slide's directory; for "per_slide" it is the single .npz file. Either way,
    its existence is what marks Stage 1 as saved for this slide."""
    if TILE_EMBED_FORMAT == "per_patch":
        return os.path.join(TILE_EMBED_DIR, slide_stem)
    return os.path.join(TILE_EMBED_DIR, slide_stem + ".npz")


def save_tile_embeddings(slide_stem: str, tiles, patch_numbers):
    """Persist one slide's Virchow V1 tile embeddings.

    ``tiles`` is (N, 2560) fp16 and ``patch_numbers`` is the matching (N,)
    int32 — N is the number of patches that actually READ successfully, which
    can be fewer than the slide's patch count, so the two must stay paired.
    """
    if len(tiles) != len(patch_numbers):
        raise RuntimeError(
            f"{slide_stem}: {len(tiles)} tile vector(s) but "
            f"{len(patch_numbers)} patch number(s) — refusing to save an "
            f"embedding set whose patch mapping is wrong."
        )

    out = tile_embeddings_path(slide_stem)

    if TILE_EMBED_FORMAT == "per_slide":
        os.makedirs(TILE_EMBED_DIR, exist_ok=True)
        tmp = out + ".tmp"
        with open(tmp, "wb") as fh:
            np.savez(
                fh,
                embeddings=np.asarray(tiles, dtype=np.float16),
                patch_number=np.asarray(patch_numbers, dtype=np.int32),
            )
        os.replace(tmp, out)
        size_mb = os.path.getsize(out) / 2**20
        if not PROGRESS_SINGLE_LINE:
            log(f"  [TILES] {slide_stem}: {tuple(np.shape(tiles))} → {out} ({size_mb:.0f} MB)")

    elif TILE_EMBED_FORMAT == "per_patch":
        import torch
        os.makedirs(out, exist_ok=True)
        vectors = torch.from_numpy(np.asarray(tiles, dtype=np.float16))
        for vec, num in zip(vectors, patch_numbers):
            # Same <slide>_<patch_number>.pt naming the other models use, so
            # these sit alongside them predictably.
            atomic_save(vec.clone(), os.path.join(out, f"{slide_stem}_{int(num)}.pt"))
        if not PROGRESS_SINGLE_LINE:
            log(f"  [TILES] {slide_stem}: {len(tiles):,} per-patch .pt file(s) → {out}")

    else:
        raise ValueError(
            f"TILE_EMBED_FORMAT must be 'per_slide' or 'per_patch', "
            f"got {TILE_EMBED_FORMAT!r}"
        )


def tile_embeddings_exist(slide_stem: str) -> bool:
    """True if Stage 1 output for this slide is already saved.

    For "per_patch" an empty directory does not count — it is what a crash
    midway through writing 20k files leaves behind.
    """
    path = tile_embeddings_path(slide_stem)
    if TILE_EMBED_FORMAT == "per_patch":
        return os.path.isdir(path) and any(
            f.endswith(".pt") for f in os.listdir(path)
        )
    return os.path.exists(path)


# ══════════════════════════════════════════════════════════════════════════════
# ░░  STEP C — per-slide tile encoding + PRISM aggregation  ░░
# ══════════════════════════════════════════════════════════════════════════════

def process_slide(slide_stem: str, czi_path: str, patches, extractor) -> bool:
    """Encode every non-white patch of one slide and write its PRISM embedding.

    Returns True if the (1, 1280) embedding was written.

    Reads are double-buffered against GPU work: the next chunk's patches are
    fetched on the reader thread while the current chunk goes through Virchow.
    Only ever one thread touches czidoc, so the overlap is free.
    """
    from pylibCZIrw import czi as pyczi

    numbers, xs, ys = patches
    n_rows = len(numbers)

    slide_t0 = time.time()
    write_checkpoint("slide_start", slide=slide_stem, n_patches=n_rows,
                     gpu_mem=_get_gpu_mem_snapshot())

    _WATCHDOG.heartbeat(f"opening slide '{slide_stem}'")
    try:
        slide_info = open_czi_at_target_mag(czi_path, TARGET_MAG, PATCH_SIZE)
    except Exception as e:
        log_exception(f"open_czi slide={slide_stem}", e)
        log(f"  [ERROR] Cannot open {slide_stem}: {e}")
        return False

    downsample  = slide_info["downsample"]
    x_start     = slide_info["X_start"]
    y_start     = slide_info["Y_start"]
    zoom        = 1.0 / downsample
    size_native = PATCH_SIZE * downsample

    cache     = TileShardCache(slide_stem, n_rows)
    start_row = cache.resume_row()

    if start_row >= n_rows:
        if not PROGRESS_SINGLE_LINE:
            log(f"  [RESUME] {slide_stem}: all {n_rows:,} tiles already encoded, "
                f"aggregating only")
    else:
        if start_row > 0:
            # Worth a line even in single-line mode: it explains why the bar is
            # about to jump forward.
            plog(f"  [RESUME] {slide_stem}: {start_row:,}/{n_rows:,} tiles already "
                 f"encoded, continuing from there")
            _PROGRESS.advance(start_row)
        if not PROGRESS_SINGLE_LINE:
            log(f"  [INFO] {slide_stem}: {n_rows - start_row:,} patch(es) to encode | "
                f"native_mag={slide_info['native_mag']} downsample={downsample} "
                f"size_native={size_native}px")

        ok = _encode_tiles_for_slide(
            slide_stem, czi_path, numbers, xs, ys, start_row, n_rows, cache,
            extractor, pyczi, x_start, y_start, downsample, size_native, zoom,
        )
        if not ok:
            return False

    # ── Aggregate ─────────────────────────────────────────────────────────────
    _WATCHDOG.heartbeat(f"aggregating slide '{slide_stem}' with PRISM")
    try:
        tiles, tile_numbers = cache.load_all()
    except RuntimeError as e:
        log(f"  [ERROR] {e}")
        write_checkpoint("slide_partial_cache", slide=slide_stem)
        return False

    if len(tiles) == 0:
        log(f"  [SKIP] {slide_stem}: every patch failed to read — no embedding written")
        write_checkpoint("slide_no_tiles", slide=slide_stem)
        cache.cleanup()
        return False

    # Save the Stage 1 output before Stage 2 runs. If PRISM aggregation fails
    # or OOMs, the expensive part — hours of Virchow V1 — is already on disk.
    if SAVE_TILE_EMBEDDINGS:
        _WATCHDOG.heartbeat(f"saving tile embeddings for '{slide_stem}'")
        _PROGRESS.set_phase("saving tiles")
        try:
            save_tile_embeddings(slide_stem, tiles, tile_numbers)
        except Exception as e:
            log_exception(f"save_tile_embeddings slide={slide_stem}", e)
            log(f"  [ERROR] Could not save tile embeddings for {slide_stem}: {e}")
            return False

    # PRISM's tile sequence is one vector per tile. When crops are retained the
    # saved array is (N, 5, 2560), so collapse the crop axis for the Perceiver —
    # the same mean encode_tiles would otherwise have applied, just deferred so
    # the crops could be written out first.
    prism_tiles = tiles.astype(np.float32).mean(axis=1) if tiles.ndim == 3 else tiles

    _PROGRESS.set_phase("prism")
    if not PROGRESS_SINGLE_LINE:
        log(f"  [PRISM] {slide_stem}: {len(prism_tiles):,} tile(s) → Perceiver")
    write_checkpoint("prism_aggregate_start", slide=slide_stem, n_tiles=len(prism_tiles))
    try:
        embedding, latents = extractor.aggregate(prism_tiles)
    except Exception as e:
        log_exception(f"prism_aggregate slide={slide_stem}", e)
        log(f"  [ERROR] PRISM aggregation failed for {slide_stem}: {e}")
        return False

    import torch
    if not torch.isfinite(embedding).all():
        log(f"  [ERROR] {slide_stem}: PRISM produced non-finite values — not saving")
        write_checkpoint("slide_nonfinite", slide=slide_stem)
        return False

    atomic_save(embedding, slide_embedding_path(slide_stem))

    if SAVE_IMAGE_LATENTS and latents is not None:
        latents_dir = os.path.join(OUTPUT_ROOT, "features", "slide_aggregation",
                                   "PRISM", "prism_latents")
        atomic_save(latents.to(torch.float16),
                    os.path.join(latents_dir, slide_stem + ".pt"))

    cache.cleanup()

    elapsed = time.time() - slide_t0
    if PROGRESS_SINGLE_LINE:
        # One compact line per finished slide, scrolling above the live bar.
        plog(f"  ✓ {slide_stem:<24} {len(tiles):>6,} tiles  {elapsed / 60:5.1f} min"
             f"  ({len(tiles) / max(elapsed, 1e-9):5.1f} patch/s)")
    else:
        log(f"  ✓ {slide_stem}: {tuple(embedding.shape)} → {slide_embedding_path(slide_stem)} "
            f"({elapsed / 60:.1f} min, {len(tiles):,} tiles)")
    write_checkpoint("slide_done", slide=slide_stem, n_tiles=len(tiles),
                     elapsed_sec=round(elapsed, 1), gpu_mem=_get_gpu_mem_snapshot())
    return True


def _encode_tiles_for_slide(slide_stem, czi_path, numbers, xs, ys, start_row, n_rows,
                            cache, extractor, pyczi, x_start, y_start,
                            downsample, size_native, zoom) -> bool:
    """Stage 1 for one slide: stream patches → Virchow V1 → tile shards.

    Split out of process_slide purely so the CZI open/close and the shard
    bookkeeping sit in one place with a single exit path.
    """
    import torch

    reader_pool = _get_czi_reader_pool()

    def _open_on_worker(path):
        ctx = pyczi.open_czi(path)
        return ctx, ctx.__enter__()

    _WATCHDOG.heartbeat(f"opening CZI handle for '{slide_stem}'")
    try:
        czi_ctx, czidoc = reader_pool.submit(_open_on_worker, czi_path).result()
    except Exception as e:
        log_exception(f"czi_handle slide={slide_stem}", e)
        log(f"  [ERROR] Could not open CZI handle for {slide_stem}: {e}")
        return False

    row_chunks = [(s, min(s + FEATURE_BATCH_SIZE, n_rows))
                  for s in range(start_row, n_rows, FEATURE_BATCH_SIZE)]

    def _submit(idx):
        lo, hi = row_chunks[idx]
        return reader_pool.submit(
            _read_patch_chunk, czidoc, numbers[lo:hi], xs[lo:hi], ys[lo:hi],
            slide_stem, x_start, y_start, downsample, size_native, zoom,
        )

    buffer          = []          # tile vectors not yet flushed to a shard
    buffer_nums     = []          # their patch numbers, kept strictly in step
    buffer_start    = start_row   # first row the buffer covers
    n_encoded       = 0
    phase           = {"wait_read": 0.0, "inference": 0.0, "shard": 0.0}
    chunk_times     = []
    LOG_EVERY       = 100

    try:
        next_future = _submit(0) if row_chunks else None

        for i in range(len(row_chunks)):
            lo, hi = row_chunks[i]
            chunk_t0 = time.time()

            _WATCHDOG.heartbeat(
                f"reading patches {lo}-{hi} of '{slide_stem}' "
                f"(patch_number {int(numbers[lo])}..{int(numbers[hi - 1])})"
            )
            t = time.time()
            _PROGRESS.set_phase("read")
            images, kept_numbers = next_future.result()
            phase["wait_read"] += time.time() - t

            # Queue the next read before the GPU starts, so I/O overlaps compute.
            if i + 1 < len(row_chunks):
                next_future = _submit(i + 1)

            if images:
                _WATCHDOG.heartbeat(f"encoding patches {lo}-{hi} of '{slide_stem}'")
                _PROGRESS.set_phase("encode")
                t = time.time()
                try:
                    vecs = extractor.encode_tiles(images)
                    # Append vectors and their patch numbers together, and only
                    # on success — a failed encode must not leave numbers in the
                    # buffer with no vectors beside them.
                    buffer.append(vecs)
                    buffer_nums.append(kept_numbers)
                    n_encoded += len(images)
                except Exception as e:
                    log_exception(f"encode slide={slide_stem} rows={lo}-{hi}", e)
                    tqdm.write(f"    [ERROR] encode failed for rows {lo}-{hi}: {e}")
                phase["inference"] += time.time() - t
                del images

            # Flush a shard once enough rows are covered. Rows, not vectors:
            # unreadable patches still advance the resume cursor, otherwise a
            # permanently bad patch would make the slide restart forever.
            if hi - buffer_start >= SHARD_PATCHES or i == len(row_chunks) - 1:
                t = time.time()
                empty_shape = ((0, 5, 2560) if TILE_EMBED_KEEP_CROPS else (0, 2560))
                vectors = (torch.cat(buffer, dim=0) if buffer
                           else torch.zeros(empty_shape, dtype=torch.float16))
                nums = (np.concatenate(buffer_nums) if buffer_nums
                        else np.zeros(0, dtype=np.int32))
                cache.write(buffer_start, hi, vectors, nums)
                phase["shard"] += time.time() - t
                write_checkpoint("shard_written", slide=slide_stem,
                                 start_row=buffer_start, end_row=hi,
                                 n_vectors=int(vectors.shape[0]))
                buffer, buffer_nums, buffer_start = [], [], hi
                del vectors, nums

            chunk_times.append(time.time() - chunk_t0)

            # The global bar counts patches, so advance it by this chunk's rows
            # (rows, not vectors — a patch that failed to read is still work
            # done and must not stall the bar).
            _PROGRESS.advance(hi - lo)

            # What used to be a [TIMING] line every 100 chunks is now just the
            # bar's postfix, refreshed in place.
            if (i + 1) % LOG_EVERY == 0 or i == len(row_chunks) - 1:
                window = chunk_times[-LOG_EVERY:]
                busy   = phase["wait_read"] + phase["inference"] + phase["shard"]
                if _PROGRESS.bar is not None and busy > 0:
                    _PROGRESS.bar.set_postfix_str(
                        f"gpu {100 * phase['inference'] / busy:.0f}% "
                        f"io {100 * phase['wait_read'] / busy:.0f}% "
                        f"| {statistics.median(window):.2f}s/chunk",
                        refresh=True,
                    )
        return True

    except Exception as e:
        log_exception(f"encode_slide slide={slide_stem}", e)
        log(f"  [ERROR] Tile encoding failed for {slide_stem}: {e}")
        traceback.print_exc()
        return False

    finally:
        # Close on the same worker thread that opened it — cross-thread close is
        # the same native-state hazard as cross-thread read. The thread itself
        # (the persistent pool) is not torn down.
        try:
            reader_pool.submit(lambda: czi_ctx.__exit__(None, None, None)).result()
        except Exception as e:
            log(f"  [WARN] Error closing CZI for {slide_stem}: {e}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not PROGRESS_SINGLE_LINE:
            log(f"  [STAGE 1] {slide_stem}: {n_encoded:,} tile(s) encoded this pass")


def run_step_C(nonwhite_index: dict, extractor, batch_stems: list,
               slides_done: int = 0, slides_total: int = 0):
    """Process every slide of the current batch whose CZI is on disk."""
    czi_index = find_all_czi_files(CZI_DIR)

    todo = [s for s in batch_stems
            if s in czi_index and s in nonwhite_index
            and not _slide_complete(s, nonwhite_index)]

    absent = [s for s in batch_stems
              if s in nonwhite_index
              and not _slide_complete(s, nonwhite_index)
              and s not in czi_index]
    for s in absent:
        plog(f"  [SKIP] {s}: CZI not on disk (download failed?) — will retry next run")

    if not PROGRESS_SINGLE_LINE:
        log(f"  Processing {len(todo)} slide(s) present in this batch.")

    n_ok = 0
    for i, slide_stem in enumerate(todo):
        _PROGRESS.set_slide(slide_stem, slides_done + i + 1, slides_total)
        try:
            if process_slide(slide_stem, czi_index[slide_stem],
                             nonwhite_index[slide_stem], extractor):
                n_ok += 1
        except Exception as e:
            log_exception(f"process_slide slide={slide_stem}", e)
            log(f"  [ERROR] {slide_stem} failed: {e}")
            traceback.print_exc()
    return n_ok


# ══════════════════════════════════════════════════════════════════════════════
# ░░  MAIN ORCHESTRATOR  ░░
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import torch

    # Patch size is fixed for the whole run, so cuDNN can safely benchmark and
    # cache the fastest algorithms for that shape.
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    log("╔══════════════════════════════════════════════════════════════════════╗")
    log("║       SurGen — PRISM slide embeddings (Virchow V1 tiles)            ║")
    log("╚══════════════════════════════════════════════════════════════════════╝")
    log(f"  Batch size        : {BATCH_SIZE} WSI(s) per download/process/delete cycle")
    log(f"  CZI dir           : {CZI_DIR}")
    log(f"  Slide embeddings  : {SLIDE_EMBED_DIR}")
    if SAVE_TILE_EMBEDDINGS:
        shape = "(N, 5, 2560) all crops" if TILE_EMBED_KEEP_CROPS else "(N, 2560) crops averaged"
        log(f"  Tile embeddings   : {TILE_EMBED_DIR}")
        log(f"                      {TILE_EMBED_FORMAT}, {shape}")
    else:
        log("  Tile embeddings   : NOT SAVED (SAVE_TILE_EMBEDDINGS = False)")
    log(f"  Tile cache        : {TILE_CACHE_DIR}  (kept: {KEEP_TILE_CACHE})")
    log(f"  Non-white CSV     : {NONWHITE_METADATA_CSV}")
    log(f"  Labels CSV        : {SURGEN_LABELS_CSV}")
    log(f"  Patch size        : {PATCH_SIZE} @ {TARGET_MAG}x")
    log(f"  FiveCrop          : {USE_FIVECROP} (crop {CROP_SIZE} → 224)")
    log(f"  Patch token start : {VIRCHOW_V1_PATCH_TOKEN_START}")
    log(f"  Save latents      : {SAVE_IMAGE_LATENTS}")
    log(f"  Disk guard        : {MIN_FREE_GB} GB on {DISK_TO_CHECK}")
    log(f"  Crash diagnostics : {CRASH_DIAG_DIR}")
    if torch.cuda.is_available():
        log(f"  CUDA device       : {torch.cuda.get_device_name(0)} (torch {torch.__version__})")
    else:
        log("  CUDA device       : NONE — this run is not viable on CPU")

    os.makedirs(SLIDE_EMBED_DIR, exist_ok=True)
    os.makedirs(TILE_CACHE_DIR, exist_ok=True)
    if SAVE_TILE_EMBEDDINGS:
        if TILE_EMBED_FORMAT not in ("per_slide", "per_patch"):
            sys.exit(f"[FATAL] TILE_EMBED_FORMAT must be 'per_slide' or "
                     f"'per_patch', got {TILE_EMBED_FORMAT!r}")
        os.makedirs(TILE_EMBED_DIR, exist_ok=True)

    nonwhite_index = load_nonwhite_index()
    all_wsis       = build_wsi_list(SURGEN_LABELS_CSV)

    no_tissue = [f for f, _ in all_wsis if Path(f).stem not in nonwhite_index]
    if no_tissue:
        log(f"  {len(no_tissue)} labelled slide(s) have zero non-white patches and "
            f"are skipped entirely: {', '.join(Path(f).stem for f in no_tissue)}")

    batches = [all_wsis[i:i + BATCH_SIZE] for i in range(0, len(all_wsis), BATCH_SIZE)]
    log(f"\n  Batches to process: {len(batches)}")

    _WATCHDOG.start()
    log(f"  [WATCHDOG] Active — force-exits if nothing progresses for "
        f"{WATCHDOG_TIMEOUT_SEC}s.")

    # ── Resume scan ───────────────────────────────────────────────────────────
    # Pure directory listing: no downloads, no model load. Jumps straight to the
    # first unfinished batch instead of walking every finished one through the
    # download/model-load machinery on each restart.
    log("\n[RESUME SCAN] Checking which batches are already complete ...")
    t0 = time.time()
    start_batch_idx = len(batches)
    for i, b in enumerate(batches):
        _WATCHDOG.heartbeat(f"resume scan batch {i + 1}/{len(batches)}")
        if not _batch_fully_complete(b, nonwhite_index):
            start_batch_idx = i
            break
    done_now = sum(1 for f, _ in all_wsis
                   if os.path.exists(slide_embedding_path(Path(f).stem)))
    log(f"  [RESUME SCAN] Finished in {time.time() - t0:.1f}s. "
        f"{done_now}/{len(nonwhite_index)} slide embedding(s) already on disk.")

    if start_batch_idx == len(batches):
        log(f"  [RESUME SCAN] All {len(batches)} batch(es) complete — nothing to do.")
    else:
        log(f"  [RESUME SCAN] Starting at batch {start_batch_idx + 1}/{len(batches)}.")
        project_storage_or_halt(nonwhite_index, len(nonwhite_index) - done_now)
    write_checkpoint("resume_scan_complete", start_batch_idx=start_batch_idx,
                     total_batches=len(batches), slides_done=done_now)

    # ── Model is loaded lazily: a fully-complete run should never pay for it ──
    # One progress bar for the whole run, sized in patches still outstanding
    # rather than slides: slides range from 481 to 42,174 patches, so a
    # slide-counting bar would lurch and its ETA would be meaningless.
    # Counting patches also makes the rate directly comparable to the
    # ~44 patch/s benchmark.
    global _PROGRESS
    patches_remaining = sum(
        len(nonwhite_index[s][0]) for s in nonwhite_index
        if not _slide_complete(s, nonwhite_index)
    )
    _bar_wanted = PROGRESS_SINGLE_LINE and start_batch_idx < len(batches)
    if _bar_wanted:
        log(f"\n  Progress: {patches_remaining:,} patch(es) outstanding across "
            f"{len(nonwhite_index) - done_now} slide(s).")
        log("  One live line below; finished slides scroll above it.\n")
    _PROGRESS = RunProgress(patches_remaining, enabled=_bar_wanted)

    extractor = None
    slides_done_running = done_now

    for batch_idx, batch in enumerate(batches[start_batch_idx:], start=start_batch_idx):
        log(f"\n{'█' * 72}")
        log(f"█  BATCH {batch_idx + 1}/{len(batches)}  —  {len(batch)} WSI(s)")
        log(f"{'█' * 72}")
        write_checkpoint("batch_start", batch_idx=batch_idx, n_batches=len(batches))
        _PROGRESS.set_batch(batch_idx + 1, len(batches))

        if _batch_fully_complete(batch, nonwhite_index):
            log(f"  [SKIP BATCH] Every slide already has a PRISM embedding.")
            write_checkpoint("batch_skip_complete", batch_idx=batch_idx)
            continue

        # ── 1. Download ───────────────────────────────────────────────────────
        to_download = filter_batch_for_download(batch, nonwhite_index)
        if to_download:
            incomplete = download_batch(to_download)
        else:
            incomplete = set()
            log("  [SKIP DOWNLOAD] Nothing left to fetch for this batch.")

        # ── 2. Encode + aggregate ─────────────────────────────────────────────
        if extractor is None:
            log(f"\n{'─' * 70}")
            log("[MODEL] Loading Virchow V1 + PRISM ...")
            log(f"{'─' * 70}")
            extractor = PrismExtractor(device=DEVICE)

        # A slide whose CZI did not download completely is left for a later
        # pass. Encoding it would succeed, quietly drop every patch that lands
        # in a missing byte range, and write an embedding that then marks the
        # slide done forever.
        batch_stems = [Path(f).stem for f, _ in batch
                       if Path(f).stem not in incomplete]
        if incomplete:
            log(f"  [DEFERRED] {len(incomplete)} slide(s) skipped this pass "
                f"(incomplete download): {', '.join(sorted(incomplete))}")
        try:
            n_ok = run_step_C(nonwhite_index, extractor, batch_stems,
                              slides_done=slides_done_running,
                              slides_total=len(nonwhite_index))
            slides_done_running += n_ok
            plog(f"  [BATCH {batch_idx + 1}/{len(batches)}] {n_ok}/{len(batch_stems)} "
                 f"slide embedding(s) written this pass.")
        except Exception as e:
            log_exception(f"step_c batch_idx={batch_idx}", e)
            plog(f"[ERROR] Step C failed for batch {batch_idx + 1}: {e}")
            traceback.print_exc()

        # ── 3. Delete this batch's CZIs ───────────────────────────────────────
        delete_batch_czis(batch)

        # ── 4. Disk guard ─────────────────────────────────────────────────────
        check_disk_space_or_halt(MIN_FREE_GB)

        log(f"\n[BATCH {batch_idx + 1}/{len(batches)} COMPLETE]")
        write_checkpoint("batch_complete", batch_idx=batch_idx)

    _WATCHDOG.stop()
    _PROGRESS.close()

    n_done = len([f for f in os.listdir(SLIDE_EMBED_DIR) if f.endswith(".pt")])
    log("\n╔══════════════════════════════════════════════════════════════════════╗")
    log("║  ALL BATCHES COMPLETE                                               ║")
    log("╚══════════════════════════════════════════════════════════════════════╝")
    log(f"  Slide embeddings written : {n_done} / {len(nonwhite_index)} slide(s) with tissue")
    log(f"  Location                 : {SLIDE_EMBED_DIR}")
    log("  Verify with              : python verify_prism_features.py")
    write_checkpoint("all_batches_complete", n_slide_embeddings=n_done)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as e:
        # Native segfaults never reach here — faulthandler catches those.
        log_exception("main_top_level", e)
        write_checkpoint("main_crashed_python_exception", error=str(e))
        traceback.print_exc()
        sys.exit(1)
