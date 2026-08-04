"""
Caption-based slide aggregation for SurGen.

Unlike the TCGA/PAIP notebook versions, the SurGen classification CSV already
gives `slide_name` directly, and the per-patch feature file for row `patch_id`
is saved at:  <features_dir>/<slide_name>/<patch_id>.pt
(confirmed from _save_feature_atomic / patch_id construction in both the
combined pipeline and the standalone CONCH v1 script) — so no filename
splitting is needed to locate features.

NOTE on ACTIVE_MODELS / paths: SurGen features live on three different
machines (h-optimus-1 + uni2-h + conch-v1 on the Linux box; conch1-5 +
virchow2 on the Windows box, on two different drives). You can't run all
five in one process anyway — just set ACTIVE_MODELS below to whichever
models' feature folders are actually reachable from wherever you run this,
and CSV_FILE to the matching path for that machine.

NOTE on ALL_GROUPS: defaulted to the 14-class list (no TIL/BACK) since
that's what was in your original single-model caption scripts. Swap in the
15-class list (with TIL/BACK, used in your later TCGA/PAIP notebook cells)
if that's the classifier version behind classification_results_surgen.csv.
"""
import pandas as pd
import torch
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from tqdm import tqdm

# ─── CONFIG ───────────────────────────────────────────────────────────────────
ACTIVE_MODELS = [
    "Virchow2"
]

# Pick the CSV path matching the machine you're running on.
CSV_FILE_LINUX   = "/home/mle/Aamir/Azfaar/surgen_processing/classification_results_surgen.csv"
CSV_FILE_WINDOWS = r"D:\Aamir Gulzar\KSA_project2\Cancer-detection-classifier\slide_aggregation\caption_generation\Results\classification_results_surgen.csv"
CSV_FILE = CSV_FILE_WINDOWS if os.name == "nt" else CSV_FILE_LINUX

MODEL_CONFIG = {
    "ConchV1": {
        "features_dir"  : "/media/dp-psau/Datum/Aamir/Azfaar/surgen_processed/conch-v1/features",
        "output_dir"    : "/media/dp-psau/Datum/Aamir/Azfaar/surgen_processed/conch-v1/features/slide_aggregation/Caption_based_aggregation/conch-v1",
        "feature_dim"   : 512,
        "expected_shape": torch.Size([5, 512]),
    },
    "Conch1_5": {
        "features_dir"  : r"D:\Aamir Gulzar\KSA_project2\surgen_data\surgen_processed\conch1-5\features",
        "output_dir"    : r"D:\Aamir Gulzar\KSA_project2\surgen_data\surgen_processed\conch1-5\features\slide_aggregation\Caption_based_aggregation\conch1-5",
        "feature_dim"   : 768,
        "expected_shape": torch.Size([5, 768]),
    },
    "H-Optimus-1": {
        # ASSUMPTION: matches the Linux path pattern from your Averaging/TTC h-optimus-1 scripts.
        "features_dir"  : "/media/dp-psau/Datum/Aamir/Azfaar/surgen_processed/h-optimus-1/features",
        "output_dir"    : "/media/dp-psau/Datum/Aamir/Azfaar/surgen_processed/h-optimus-1/features/slide_aggregation/Caption_based_aggregation/h-optimus-1",
        "feature_dim"   : 1536,
        "expected_shape": torch.Size([5, 1536]),
    },
    "UNI2": {
        # ASSUMPTION: inferred by analogy to h-optimus-1's Linux path — confirm/edit.
        "features_dir"  : "/media/dp-psau/Datum/Aamir/Azfaar/surgen_processed/uni2-h/features",
        "output_dir"    : "/media/dp-psau/Datum/Aamir/Azfaar/surgen_processed/uni2-h/features/slide_aggregation/Caption_based_aggregation/uni2-h",
        "feature_dim"   : 1536,
        "expected_shape": torch.Size([5, 1536]),
    },
    "Virchow2": {
        "features_dir"  : r"F:\surgen_processed\virchow2\features",
        "output_dir"    : r"D:\Aamir Gulzar\KSA_project2\surgen_data\surgen_processed\virchow2\features\slide_aggregation\Caption_based_aggregation\virchow2",
        "feature_dim"   : 2560,
        "expected_shape": torch.Size([5, 2560]),
    },
}

# 14-class default (see NOTE above) — swap for the 15-class list if needed:
# ALL_GROUPS = ['ADI', 'DEB', 'TIL', 'PLC', 'LYA', 'LYM', 'MUS', 'MUC',
#               'NORM', 'ADE', 'STR', 'CAR', 'PDC', 'SIG', 'BACK']
ALL_GROUPS = ['ADI', 'DEB', 'LYM', 'PLC', 'LYA', 'MUC', 'MUS', 'NORM',
              'ADE', 'STR', 'MES', 'CAR', 'PDC', 'SIG']

MAX_WORKERS = 12

# You've verified the missing-file count is small and stable across runs,
# and load_one() below already catches + logs missing/bad files inline —
# so the ~15-20min exists-check pre-pass is now mostly redundant cost.
# Set True to skip straight to loading (missing files just get logged as [ERR]).
SKIP_EXISTS_CHECK = True

# How many files to sample for the startup load-time benchmark.
BENCHMARK_SAMPLE_SIZE = 50
# ──────────────────────────────────────────────────────────────────────────────


def run_for_model(model_name: str, cfg: dict, df_base: pd.DataFrame):
    print(f"\n{'═'*60}")
    print(f"  MODEL: {model_name}")
    print(f"{'═'*60}")

    features_dir   = cfg["features_dir"]
    output_dir     = cfg["output_dir"]
    feature_dim    = cfg["feature_dim"]
    expected_shape = cfg["expected_shape"]

    os.makedirs(output_dir, exist_ok=True)

    # ── Fast-fail sanity check on the drive/dir before touching millions of rows ──
    print(f"  Checking features_dir is reachable: {features_dir}", flush=True)
    t0 = time.time()
    if not os.path.isdir(features_dir):
        print(f"  [ERROR] features_dir does not exist or is unreachable: {features_dir}", flush=True)
        print(f"  (If this is a mapped/network/external drive, confirm it's connected and the letter/path is correct.)", flush=True)
        return
    print(f"  ✓ features_dir reachable ({time.time()-t0:.2f}s)", flush=True)

    # Quick single-file latency probe so you know roughly what 11M+ calls will cost
    sample_slide = df_base['slide_name'].iloc[0]
    sample_dir = os.path.join(features_dir, sample_slide)
    t0 = time.time()
    _ = os.path.isdir(sample_dir)
    probe_latency = time.time() - t0
    print(f"  Probe: checking one slide subdir took {probe_latency*1000:.1f} ms "
          f"(rough est. for {len(df_base):,} serial checks: {probe_latency*len(df_base)/60:.1f} min if run serially)", flush=True)

    # ── REAL multi-file torch.load benchmark (ground truth for what 11M+ loads will cost) ──
    n_bench = min(BENCHMARK_SAMPLE_SIZE, len(df_base))
    bench_rows = df_base.sample(n=n_bench, random_state=42) if len(df_base) > n_bench else df_base
    print(f"  Benchmarking {n_bench} torch.load calls (random sample)...", flush=True)

    bench_times = []
    missing_in_bench = 0
    for _, row in bench_rows.iterrows():
        fp = os.path.join(features_dir, row['slide_name'], f"{row['patch_id']}.pt")
        if not os.path.isfile(fp):
            missing_in_bench += 1
            continue
        t0 = time.time()
        _ = torch.load(fp, map_location="cpu", weights_only=True)
        bench_times.append(time.time() - t0)

    if bench_times:
        avg_t = sum(bench_times) / len(bench_times)
        min_t, max_t = min(bench_times), max(bench_times)
        print(f"  ✓ {len(bench_times)} files loaded (missing: {missing_in_bench}) — "
              f"avg {avg_t*1000:.1f} ms | min {min_t*1000:.1f} ms | max {max_t*1000:.1f} ms", flush=True)
        est_serial_hours = avg_t * len(df_base) / 3600
        print(f"  → At this average rate, {len(df_base):,} SERIAL loads would take "
              f"~{est_serial_hours:.1f}h (threading helps a lot on SSD/NVMe, "
              f"much less on external/USB HDD — actual threaded rate may differ significantly).", flush=True)
        if max_t > 5 * avg_t:
            print(f"  [WARN] max ({max_t*1000:.0f} ms) is much higher than avg — "
                  f"suggests inconsistent latency (drive spin-up, contention, or caching effects), "
                  f"not a uniformly slow drive.", flush=True)
    else:
        print(f"  [WARN] All {n_bench} sampled files were missing — check the path pattern.", flush=True)

    # ── Build feature paths: <features_dir>/<slide_name>/<patch_id>.pt ────────
    df = df_base.copy()
    df['feature_path'] = (features_dir + os.sep
                          + df['slide_name'] + os.sep
                          + df['patch_id'] + '.pt')

    # ── Filter missing (parallelized + progress bar, since this is I/O bound) ──
    if SKIP_EXISTS_CHECK:
        print(f"  Skipping exists-check pre-pass (SKIP_EXISTS_CHECK=True) — "
              f"missing files will be caught and logged individually during loading instead.", flush=True)
    else:
        print(f"  Checking existence of {len(df):,} feature files with {MAX_WORKERS} threads...", flush=True)
        t0 = time.time()
        paths = df['feature_path'].tolist()
        exists_flags = [False] * len(paths)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_idx = {executor.submit(os.path.exists, p): i for i, p in enumerate(paths)}
            for future in tqdm(as_completed(future_to_idx), total=len(paths), desc=f"  {model_name} exists-check"):
                idx = future_to_idx[future]
                try:
                    exists_flags[idx] = future.result()
                except Exception:
                    exists_flags[idx] = False
        df['exists'] = exists_flags
        print(f"  Found : {df['exists'].sum():,}  |  Missing: {(~df['exists']).sum():,}  "
              f"(exists-check took {time.time()-t0:.1f}s)", flush=True)
        df = df[df['exists']].reset_index(drop=True)

    # ── Parallel load ──────────────────────────────────────────────────────────
    SLOW_READ_THRESHOLD_S = 2.0  # log any single file read slower than this

    def load_one(args):
        feature_path, slide_name, label = args
        t0 = time.time()
        try:
            tensor = torch.load(feature_path, map_location="cpu", weights_only=True)
            dt = time.time() - t0
            if dt > SLOW_READ_THRESHOLD_S:
                print(f"  [SLOW {dt:.1f}s] {feature_path}", flush=True)
            if tensor.shape != expected_shape:
                print(f"  [BAD SHAPE {tuple(tensor.shape)}] {feature_path}", flush=True)
                return None
            return (slide_name, label, tensor.mean(dim=0))
        except Exception as e:
            print(f"  [ERR] {feature_path}: {e}", flush=True)
            return None

    args_list = list(zip(df['feature_path'], df['slide_name'], df['label']))
    slide_features = defaultdict(lambda: defaultdict(list))

    print(f"  Loading {len(args_list):,} files with {MAX_WORKERS} threads...")

    # Heartbeat thread: prints real completed count + rate every 15s,
    # independent of tqdm (tqdm's own rate estimate can sit at "?it/s"
    # indefinitely if nothing has completed yet, which looks identical
    # to a real hang — this makes it unambiguous).
    completed_count = 0
    completed_lock = threading.Lock()
    stop_heartbeat = threading.Event()

    def heartbeat(total):
        start = time.time()
        while not stop_heartbeat.wait(15):
            with completed_lock:
                n = completed_count
            elapsed = time.time() - start
            if n > 0:
                rate = n / elapsed
                eta_hours = (total - n) / rate / 3600
                print(f"  [heartbeat] {n:,}/{total:,} done | {elapsed:.0f}s elapsed | "
                      f"{rate:.2f} files/s | ETA ~{eta_hours:.1f}h", flush=True)
            else:
                print(f"  [heartbeat] 0/{total:,} done after {elapsed:.0f}s — "
                      f"nothing has completed yet, this is the bottleneck to watch", flush=True)

    hb_thread = threading.Thread(target=heartbeat, args=(len(args_list),), daemon=True)
    hb_thread.start()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(load_one, a): a for a in args_list}
        for future in tqdm(as_completed(futures), total=len(args_list), desc=f"  {model_name}"):
            result = future.result()
            with completed_lock:
                completed_count += 1
            if result is not None:
                slide_name, label, avg = result
                slide_features[slide_name][label].append(avg)

    stop_heartbeat.set()
    hb_thread.join(timeout=1)

    print(f"  Loaded features for {len(slide_features):,} slides")

    # ── Aggregate + save ───────────────────────────────────────────────────────
    n_groups = len(ALL_GROUPS)
    for slide_name, group_data in tqdm(slide_features.items(), desc=f"  Aggregating {model_name}"):
        final_features = []
        for group in ALL_GROUPS:
            if group_data[group]:
                avg_features = torch.stack(group_data[group]).mean(dim=0, keepdim=True)
            else:
                avg_features = torch.zeros(1, feature_dim)
            final_features.append(avg_features)

        final_tensor = torch.cat(final_features, dim=0)   # (n_groups, feature_dim)
        torch.save(final_tensor, os.path.join(output_dir, f"{slide_name}.pt"))

    print(f"  ✓ Saved {len(slide_features):,} slide tensors ({n_groups} × {feature_dim}) → {output_dir}")


def main():
    print("📂 Loading CSV...", flush=True)
    df = pd.read_csv(CSV_FILE)
    print(f"  Total rows     : {len(df):,}", flush=True)
    print(f"  Unique slides  : {df['slide_name'].nunique():,}", flush=True)
    print(f"  Labels found   : {sorted(df['label'].unique())}", flush=True)
    print(f"\n  Active models  : {ACTIVE_MODELS}", flush=True)

    for model_name in ACTIVE_MODELS:
        if model_name not in MODEL_CONFIG:
            print(f"\n[WARN] '{model_name}' not in MODEL_CONFIG — skipping.")
            continue
        run_for_model(model_name, MODEL_CONFIG[model_name], df)

    print(f"\n{'═'*60}")
    print("✅ ALL DONE")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()