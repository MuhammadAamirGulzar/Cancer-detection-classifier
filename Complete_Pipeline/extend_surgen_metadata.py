"""
Incrementally extend the SurGen tissue-patch metadata to cover every slide
with include == TRUE in surgen_slide_labels.csv, WITHOUT recomputing the
slides that are already in patch_metadata_nonwhite.csv.

Why this script exists
----------------------
The 622-slide `patch_metadata_nonwhite.csv` is the shared input every SurGen
feature extractor reads. surgen_slide_labels.csv now marks 991 slides as
usable (369 more than before). The per-slide Step A-1 CSVs that would let the
combined pipeline resume incrementally are gone, so a naive re-run of Step A-2
would overwrite the 622-slide CSV. This script instead:

  1. BACKS UP  patch_metadata_nonwhite.csv (+ the .npz index).
  2. Step A-1 for the NEW slides only: download CZI -> per-slide RGB-stats CSV
     -> delete CZI. Batched so disk stays bounded. Resume-safe (skips a slide
     that already has a per-slide CSV).
  3. Step A-2 for the NEW slides only: the identical pixel-rule + SVM whiteness
     filter (same thresholds, same svm_model.pkl), on just the new per-slide
     CSVs, then APPENDS the resulting tissue rows to patch_metadata_nonwhite.csv
     with the same header/'2_merge_extracted_slides_LOCAL.py' safety checks
     (header match, no duplicate slides, row-count verify, atomic swap).
  4. Rebuilds patch_metadata_nonwhite_index.npz from the full CSV.
  5. Verifies: every one of the original 622 slides keeps its exact non-white
     patch count (checked against patch_counts_per_slide.csv), the CSV now
     covers the expected number of slides, and prints the new-slide counts.

NO preprocessing parameter is touched. PATCH_SIZE / TARGET_MAG / ROW_BATCH and
the four pixel thresholds and svm_model.pkl are read straight from the pipeline
module.

Run (prism_env has dotenv + pylibCZIrw):
    cd Complete_Pipeline
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 \
      C:/Users/datainsight/anaconda3/envs/prism_env/python.exe -u extend_surgen_metadata.py

Resumable: re-running continues where it stopped. Pass --a1-only to stop after
Step A-1, --a2-only to skip straight to the merge (all per-slide CSVs must
already exist).
"""
import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import combined_pipeline_final_error_checks_v6 as m

# ── scratch / config ────────────────────────────────────────────────────────
CZI_DIR        = r"F:\CZI_Files_EXTEND"
BATCH_SIZE     = 6                 # CZIs downloaded, A-1'd, then deleted per cycle
MIN_FREE_GB_F  = 40
PROGRESS_JSON  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "extend_surgen_metadata_progress.json")
PATCH_COUNTS_CSV = os.path.join(m.OUTPUT_BASE, "patch_counts_per_slide.csv")

NONWHITE_CSV = m.NONWHITE_METADATA_CSV
INDEX_NPZ    = os.path.join(m.OUTPUT_BASE, "patch_metadata_nonwhite_index.npz")
META_DIR     = m.PATCH_METADATA_DIR
QC_DIR       = os.path.join(m.OUTPUT_BASE, "qc_patches_extend_369")

EXPECT_HEADER = ["slide_name", "label", "patch_number", "patch_x", "patch_y",
                 "avg_R", "avg_G", "avg_B", "std_R", "std_G", "std_B",
                 "black_pixel_ratio", "white_pixel_ratio", "white_label"]


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{now()}] {msg}", flush=True)


def load_progress():
    if os.path.exists(PROGRESS_JSON):
        with open(PROGRESS_JSON) as f:
            return json.load(f)
    return {"a1_done": [], "a2_merged": False, "started": now()}


def save_progress(p):
    p["last_updated"] = now()
    tmp = PROGRESS_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(p, f, indent=2)
    os.replace(tmp, PROGRESS_JSON)


# ═══════════════════════════════════════════════════════════════════════════
# slide selection
# ═══════════════════════════════════════════════════════════════════════════

def stems_in_nonwhite_csv():
    if not os.path.exists(NONWHITE_CSV):
        return set()
    s = set()
    with open(NONWHITE_CSV, newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            s.add(row[0])
    return s


def select_slides():
    """(filename, url) for every included slide that is NOT already covered by
    patch_metadata_nonwhite.csv and has no per-slide A-1 CSV yet.

    Slides already in the nonwhite CSV are the 622 we must not recompute.
    """
    catalogue = m.build_wsi_list(m.SURGEN_LABELS_CSV)          # 991 (filename, url)
    already = stems_in_nonwhite_csv()                          # 622 done
    have_csv = {Path(p).stem for p in glob.glob(os.path.join(META_DIR, "*.csv"))}
    todo = [(f, u) for (f, u) in catalogue
            if Path(f).stem not in already and Path(f).stem not in have_csv]
    return catalogue, todo


# ═══════════════════════════════════════════════════════════════════════════
# Step A-1
# ═══════════════════════════════════════════════════════════════════════════

def download_one(filename, url):
    os.makedirs(CZI_DIR, exist_ok=True)
    out = os.path.join(CZI_DIR, filename)
    ctrl = out + ".aria2"
    if os.path.exists(out) and not os.path.exists(ctrl):
        return True, os.path.getsize(out) / 2**30, 0.0
    cmd = [m.ARIA2C_EXE, "-x", "16", "-s", "16", "-c",
           "--file-allocation=none", "-d", CZI_DIR, "-o", filename, url]
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    dt = time.time() - t0
    ok = rc == 0 and os.path.exists(out) and not os.path.exists(ctrl)
    size = os.path.getsize(out) / 2**30 if os.path.exists(out) else 0.0
    return ok, size, dt


def a1_one_slide(stem, czi_path):
    """run_step_A1's body for one slide, production params, -> per-slide CSV."""
    label = m.get_slide_label(stem + ".czi")
    if label is None:
        log(f"  [SKIP] {stem}: get_slide_label -> None")
        return None
    slide_info = m.open_czi_at_target_mag(czi_path, m.TARGET_MAG, m.PATCH_SIZE)
    t0 = time.time()
    df = m.compute_patch_rgb_stats_rowbatch(slide_info, m.PATCH_SIZE, m.ROW_BATCH)
    dt = time.time() - t0
    if df.empty:
        log(f"  [WARN] {stem}: 0 patches extracted (slide too small?)")
        return {"stem": stem, "n_patches": 0, "a1_s": round(dt, 1),
                "native_mag": slide_info["native_mag"]}
    df.insert(0, "slide_name", stem)
    df.insert(1, "label", label)
    final = os.path.join(META_DIR, f"{stem}.csv")
    tmp = final + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, final)
    return {"stem": stem, "n_patches": len(df), "a1_s": round(dt, 1),
            "native_mag": slide_info["native_mag"],
            "patch_per_s": round(len(df) / max(dt, 1e-9), 1), "label": label}


def run_a1(todo, progress):
    os.makedirs(META_DIR, exist_ok=True)
    os.makedirs(CZI_DIR, exist_ok=True)
    done = set(progress["a1_done"])
    todo = [(f, u) for (f, u) in todo if Path(f).stem not in done
            and not os.path.exists(os.path.join(META_DIR, Path(f).stem + ".csv"))]
    log(f"Step A-1: {len(todo)} slide(s) still to process "
        f"(batch size {BATCH_SIZE}).")
    if not todo:
        return

    batches = [todo[i:i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
    t_start = time.time()
    n_ok = 0
    for bi, batch in enumerate(batches, 1):
        free = shutil.disk_usage(CZI_DIR).free / 2**30
        log(f"── batch {bi}/{len(batches)}  (F: {free:.0f} GB free) ──")
        if free < MIN_FREE_GB_F:
            log(f"[HALT] F: below {MIN_FREE_GB_F} GB free — stopping to protect the drive.")
            sys.exit(2)

        present = []
        for filename, url in batch:
            stem = Path(filename).stem
            log(f"  download {stem} ...")
            ok, size, dt = download_one(filename, url)
            if ok:
                log(f"    ok  {size:.2f} GB  {dt:.0f}s")
                present.append((stem, os.path.join(CZI_DIR, filename)))
            else:
                log(f"    [ERROR] download failed for {stem} — will retry a later run")

        for stem, czi_path in present:
            log(f"  A-1 {stem} ...")
            try:
                res = a1_one_slide(stem, czi_path)
            except Exception as e:
                log(f"    [ERROR] A-1 failed for {stem}: {type(e).__name__}: {e}")
                res = None
            if res is not None:
                progress["a1_done"].append(stem)
                save_progress(progress)
                n_ok += 1
                if "patch_per_s" in res:
                    log(f"    -> {res['n_patches']:,} patches  {res['a1_s']}s "
                        f"({res['patch_per_s']} patch/s)  mag={res['native_mag']}")
            try:
                os.remove(czi_path)
            except OSError:
                pass

        el = time.time() - t_start
        log(f"  batch {bi} done. {n_ok} slide(s) this run, "
            f"{el/60:.1f} min elapsed, ~{el/max(n_ok,1):.0f}s/slide.")

    log(f"Step A-1 finished: {n_ok} slide(s) processed this run.")


# ═══════════════════════════════════════════════════════════════════════════
# Step A-2 (incremental)
# ═══════════════════════════════════════════════════════════════════════════

def run_a2_merge(catalogue, progress):
    if progress.get("a2_merged"):
        log("Step A-2: already merged (progress flag set) — skipping.")
        return

    existing = stems_in_nonwhite_csv()
    log(f"Step A-2: nonwhite CSV currently has {len(existing)} slide(s).")

    all_csvs = sorted(glob.glob(os.path.join(META_DIR, "*.csv")))
    new_csvs = [c for c in all_csvs if Path(c).stem not in existing]
    log(f"  per-slide CSVs on disk: {len(all_csvs)}   not yet in nonwhite CSV: {len(new_csvs)}")

    catalogue_stems = {Path(f).stem for f, _ in catalogue}
    missing = catalogue_stems - existing - {Path(c).stem for c in new_csvs}
    if missing:
        log(f"  [WARN] {len(missing)} included slide(s) still have no per-slide CSV "
            f"(A-1 incomplete): {sorted(missing)[:8]}{' ...' if len(missing) > 8 else ''}")
        log("  Run Step A-1 to completion before the merge. Aborting A-2.")
        sys.exit(3)

    if not new_csvs:
        log("  Nothing new to merge.")
        progress["a2_merged"] = True
        save_progress(progress)
        return

    # ── merge new per-slide CSVs, run the exact pixel-rule + SVM filter ──────
    dfs = [pd.read_csv(c) for c in new_csvs]
    merged = pd.concat(dfs, ignore_index=True)
    log(f"  {len(merged):,} patches across {merged['slide_name'].nunique()} new slide(s)")

    os.makedirs(QC_DIR, exist_ok=True)
    tagged, bad = m.tag_bad_patches(merged, QC_DIR)          # pixel rules
    svm_in = tagged[tagged["white_label"].isna()].drop(columns=["white_label", "reject_reason"])
    svm_res = m.apply_svm_whiteness(svm_in)                  # SVM
    tagged.loc[tagged["white_label"].isna(), "white_label"] = svm_res["white_label"].values
    tagged["white_label"] = tagged["white_label"].astype(int)

    save_df = tagged.drop(columns=["reject_reason"])[EXPECT_HEADER]
    nonwhite_class = int(save_df["white_label"].min())
    new_nonwhite = save_df[save_df["white_label"] == nonwhite_class].reset_index(drop=True)
    log(f"  new tissue rows: {len(new_nonwhite):,}  (white rows dropped: "
        f"{len(save_df) - len(new_nonwhite):,})")

    # ── safety-checked atomic append (mirrors 2_merge_extracted_slides_LOCAL) ─
    with open(NONWHITE_CSV, newline="") as f:
        header = next(csv.reader(f))
    if header != EXPECT_HEADER:
        raise ValueError(f"nonwhite CSV header != expected\n  {header}\n  {EXPECT_HEADER}")

    new_stems = set(new_nonwhite["slide_name"].unique())
    dupes = new_stems & existing
    if dupes:
        raise ValueError(f"[ABORT] {len(dupes)} slide(s) already in nonwhite CSV: {sorted(dupes)[:8]}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{NONWHITE_CSV}.backup_{ts}"
    shutil.copy2(NONWHITE_CSV, backup)
    log(f"  backup -> {backup}")
    if os.path.exists(INDEX_NPZ):
        shutil.copy2(INDEX_NPZ, f"{INDEX_NPZ}.backup_{ts}")
        log(f"  backup -> {INDEX_NPZ}.backup_{ts}")

    with open(NONWHITE_CSV, newline="") as f:
        rows_before = sum(1 for _ in csv.reader(f)) - 1

    with open(NONWHITE_CSV, "rb") as f:
        f.seek(-1, os.SEEK_END)
        needs_newline = f.read(1) not in (b"\n", b"\r")

    tmp = NONWHITE_CSV + ".merge_tmp"
    with open(NONWHITE_CSV, "rb") as fin, open(tmp, "wb") as fout:
        shutil.copyfileobj(fin, fout, length=16 * 1024 * 1024)
    with open(tmp, "a", newline="") as fout:
        if needs_newline:
            fout.write("\n")
        w = csv.writer(fout)
        for rec in new_nonwhite.itertuples(index=False):
            w.writerow(list(rec))

    with open(tmp, newline="") as f:
        rows_after = sum(1 for _ in csv.reader(f)) - 1
    expect = rows_before + len(new_nonwhite)
    if rows_after != expect:
        os.remove(tmp)
        raise RuntimeError(f"[ABORT] row-count mismatch: expected {expect:,} got {rows_after:,} "
                           f"(temp left nowhere; original untouched)")
    os.replace(tmp, NONWHITE_CSV)
    log(f"  appended. rows {rows_before:,} -> {rows_after:,}  (+{len(new_nonwhite):,}); "
        f"slides {len(existing)} -> {len(existing) + len(new_stems)}")

    # per-slide non-white counts for the new slides -> extend patch_counts_per_slide.csv
    counts = new_nonwhite.groupby("slide_name").size()
    if os.path.exists(PATCH_COUNTS_CSV):
        pc = pd.read_csv(PATCH_COUNTS_CSV, index_col=0)["n_patches"].to_dict()
    else:
        pc = {}
    for s, n in counts.items():
        pc[s] = int(n)
    for s in new_stems:                       # zero-tissue new slides
        pc.setdefault(s, 0)
    pd.Series(pc, name="n_patches").sort_index().to_csv(PATCH_COUNTS_CSV)
    log(f"  patch_counts_per_slide.csv -> {len(pc)} slides")

    progress["a2_merged"] = True
    progress["a2_backup"] = backup
    save_progress(progress)


# ═══════════════════════════════════════════════════════════════════════════
# index rebuild + verification
# ═══════════════════════════════════════════════════════════════════════════

def rebuild_index():
    log("Rebuilding patch_metadata_nonwhite_index.npz from the full CSV ...")
    t0 = time.time()
    parts = []
    for chunk in pd.read_csv(NONWHITE_CSV,
                             usecols=["slide_name", "patch_number", "patch_x", "patch_y"],
                             dtype={"slide_name": "string", "patch_number": "int32",
                                    "patch_x": "int32", "patch_y": "int32"},
                             chunksize=2_000_000):
        parts.append(chunk)
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["slide_name", "patch_number"], kind="mergesort")
    slide_col = df["slide_name"].to_numpy(dtype=object).astype(str)
    slides, starts = np.unique(slide_col, return_index=True)
    order = np.argsort(starts)
    slides, starts = slides[order], starts[order]
    offsets = np.append(starts, len(df)).astype(np.int64)
    np.savez(INDEX_NPZ,
             slides=slides.astype(str), offsets=offsets,
             patch_number=df["patch_number"].to_numpy(np.int32),
             patch_x=df["patch_x"].to_numpy(np.int32),
             patch_y=df["patch_y"].to_numpy(np.int32))
    log(f"  index: {len(slides)} slides, {len(df):,} patches, {time.time()-t0:.0f}s")


def verify(catalogue):
    log("=" * 74)
    log("VERIFICATION")
    log("=" * 74)
    ok = True

    # non-white counts per slide from the (new) CSV
    cur = {}
    with open(NONWHITE_CSV, newline="") as f:
        r = csv.reader(f); next(r)
        for row in r:
            cur[row[0]] = cur.get(row[0], 0) + 1

    # 1. the original 622 must be byte-for-byte unchanged in count
    ref = pd.read_csv(PATCH_COUNTS_CSV, index_col=0)["n_patches"].to_dict()
    # reference file now includes the new slides; the "original 622" are those
    # present in the pre-existing index backup if any, else fall back to the
    # 622 known set = slides with a feature dir.
    feat_dir = os.path.join(m.OUTPUT_BASE, "surgen_processed", "conch1-5", "features")
    orig = {d for d in os.listdir(feat_dir)} if os.path.isdir(feat_dir) else set()
    orig &= set(cur)
    changed = [s for s in orig if cur.get(s) != ref.get(s)]
    if changed:
        ok = False
        log(f"  [FAIL] {len(changed)} original slide(s) changed non-white count: "
            f"{changed[:10]}")
    else:
        log(f"  [OK] all {len(orig)} original (conch1-5-feature) slides keep their exact "
            f"non-white patch count")

    # 2. slide coverage
    included = {Path(f).stem for f, _ in catalogue}
    covered = set(cur)
    per_slide = {Path(p).stem for p in glob.glob(os.path.join(META_DIR, '*.csv'))}
    zero_tissue = (included & per_slide) - covered
    missing = included - covered - zero_tissue
    log(f"  included slides            : {len(included)}")
    log(f"  in nonwhite CSV            : {len(covered & included)}")
    log(f"  screened, zero tissue      : {len(zero_tissue)}  {sorted(zero_tissue)}")
    log(f"  included but NOT screened  : {len(missing)}  "
        f"{sorted(missing)[:8]}{' ...' if len(missing) > 8 else ''}")
    if missing:
        ok = False
        log("  [FAIL] some included slides never got a per-slide CSV")

    # 3. index matches CSV
    with np.load(INDEX_NPZ, allow_pickle=False) as z:
        idx_slides = set(map(str, z["slides"]))
        idx_total = int(z["offsets"][-1])
    csv_total = sum(cur.values())
    if idx_slides != covered or idx_total != csv_total:
        ok = False
        log(f"  [FAIL] index/CSV mismatch: idx {len(idx_slides)}/{idx_total:,} "
            f"vs csv {len(covered)}/{csv_total:,}")
    else:
        log(f"  [OK] index matches CSV: {len(idx_slides)} slides, {idx_total:,} patches")

    # 4. label mix of the newly added slides
    labmap = {}
    with open(NONWHITE_CSV, newline="") as f:
        r = csv.reader(f); next(r)
        for row in r:
            labmap.setdefault(row[0], row[1])
    new_slides = covered - orig
    from collections import Counter
    lc = Counter(labmap[s] for s in new_slides)
    log(f"  newly added slides         : {len(new_slides)}  ({dict(lc)})")
    new_patches = sum(cur[s] for s in new_slides)
    log(f"  newly added tissue patches : {new_patches:,}")
    log(f"  total tissue patches       : {csv_total:,} across {len(covered)} slides")

    log("=" * 74)
    log("RESULT: " + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE — see [FAIL] lines"))
    log("=" * 74)
    return ok


# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a1-only", action="store_true")
    ap.add_argument("--a2-only", action="store_true")
    args = ap.parse_args()

    log("SurGen metadata incremental extension")
    log(f"  PATCH_SIZE={m.PATCH_SIZE} TARGET_MAG={m.TARGET_MAG} ROW_BATCH={m.ROW_BATCH}")
    log(f"  pixel: BLACK={m.BLACK_PIXEL_RATIO} WHITE_RATIO={m.WHITE_PIXEL_RATIO} "
        f"WHITE_MEAN={m.WHITE_MEAN_THRESH} WHITE_STD={m.WHITE_STD_THRESH}")
    log(f"  SVM  : {m.SVM_MODEL_PATH}")
    log(f"  nonwhite CSV : {NONWHITE_CSV}")
    log(f"  per-slide dir: {META_DIR}")
    log(f"  CZI scratch  : {CZI_DIR}")

    progress = load_progress()
    catalogue, todo = select_slides()
    log(f"  catalogue {len(catalogue)} slides; {len(todo)} without a per-slide CSV")

    if not args.a2_only:
        run_a1(todo, progress)

    if args.a1_only:
        log("--a1-only: stopping before the merge.")
        return 0

    run_a2_merge(catalogue, progress)
    rebuild_index()
    ok = verify(catalogue)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
