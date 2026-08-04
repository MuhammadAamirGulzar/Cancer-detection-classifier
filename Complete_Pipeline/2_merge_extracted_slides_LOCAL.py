"""
Run this on your LOCAL machine, AFTER transferring the small file produced
by 1_extract_excluded_slides_REMOTE.py.

Appends the extracted rows to your existing local CSV — but only after a
series of checks, and never by mutating your original file in place:

  1. Header of the extracted file must exactly match the header of your
     local CSV (column order included).
  2. None of the excluded slides may already exist in your local CSV
     (guards against double-appending if you run this twice).
  3. Every row in the extracted file must belong to one of EXCLUDED_SLIDES
     (guards against accidentally merging in rows for the wrong slides).
  4. Your local CSV is backed up (copy, untouched) before anything else
     happens.
  5. The merge is written to a NEW temp file; row counts are verified;
     only then is the temp file swapped in to replace the original
     (atomic os.replace, so you're never left with a half-written CSV).

Like the remote script, rows are copied through as exact original strings
(csv module, not pandas) — no reformatting, no precision loss.
"""

import csv
import os
import shutil
from pathlib import Path
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════════
LOCAL_CSV_PATH     = r"D:\Aamir Gulzar\KSA_project2\surgen_data\patch_metadata_nonwhite.csv"   # <-- EDIT: your existing local CSV
EXTRACTED_CSV_PATH = r"D:\Aamir Gulzar\KSA_project2\Cancer-detection-classifier\Complete_Pipeline\extracted_excluded_slides.csv"             # <-- EDIT: the file you transferred over

EXCLUDED_SLIDES = [
    "SR386_40X_HE_T574_01.czi",
    "SR1482_40X_HE_T372_01.czi",
    "SR1482_40X_HE_T372_02.czi",
    "SR1482_40X_HE_T377_01.czi",
    "SR1482_40X_HE_T377_02.czi",
    "SR1482_40X_HE_T390_01.czi",
    "SR1482_40X_HE_T399_01.czi",
    "SR1482_40X_HE_T399_02.czi",
    "SR1482_40X_HE_T408_01.czi",
    "SR1482_40X_HE_T412_01.czi",
    "SR1482_40X_HE_T419_01.czi",
    "SR1482_40X_HE_T425_01.czi",
    "SR1482_40X_HE_T426_01.czi",
    "SR1482_40X_HE_T426_02.czi",
    "SR1482_40X_HE_T427_01.czi",
    "SR1482_40X_HE_T427_02.czi",
    "SR1482_40X_HE_T433_01.czi",
    "SR1482_40X_HE_T434_01.czi",
]
EXCLUDED_STEMS = {Path(s).stem for s in EXCLUDED_SLIDES}
# ══════════════════════════════════════════════════════════════════════════

for p, label in [(LOCAL_CSV_PATH, "LOCAL_CSV_PATH"), (EXTRACTED_CSV_PATH, "EXTRACTED_CSV_PATH")]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"{label} does not exist: {p}")

# ── 1. Header check ─────────────────────────────────────────────────────
with open(LOCAL_CSV_PATH, "r", newline="") as f:
    local_header = next(csv.reader(f))
with open(EXTRACTED_CSV_PATH, "r", newline="") as f:
    extracted_header = next(csv.reader(f))

if local_header != extracted_header:
    raise ValueError(
        "Header mismatch — refusing to merge.\n"
        f"  Local     : {local_header}\n"
        f"  Extracted : {extracted_header}"
    )
slide_col_idx = local_header.index("slide_name")
print(f"[OK] Headers match ({len(local_header)} columns).")

# ── 2. Make sure none of the excluded slides already exist locally ───────
already_present = set()
local_row_count = 0
local_slide_counts = {}
with open(LOCAL_CSV_PATH, "r", newline="") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        local_row_count += 1
        s = row[slide_col_idx]
        local_slide_counts[s] = local_slide_counts.get(s, 0) + 1
        if s in EXCLUDED_STEMS:
            already_present.add(s)

if already_present:
    raise ValueError(
        f"[ABORT] {len(already_present)} excluded slide(s) already exist in your local CSV — "
        f"merging would create duplicates:\n  {sorted(already_present)}\n"
        "Remove them from EXCLUDED_SLIDES (or clean the local CSV) before re-running."
    )
print(f"[OK] None of the {len(EXCLUDED_STEMS)} excluded slides are already present locally.")
print(f"     Local CSV currently has {local_row_count:,} rows across {len(local_slide_counts):,} slides.")

# ── 3. Validate every row in the extracted file belongs to an excluded slide ─
extracted_row_count = 0
extracted_slide_counts = {}
unexpected_slides = set()
with open(EXTRACTED_CSV_PATH, "r", newline="") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        extracted_row_count += 1
        s = row[slide_col_idx]
        extracted_slide_counts[s] = extracted_slide_counts.get(s, 0) + 1
        if s not in EXCLUDED_STEMS:
            unexpected_slides.add(s)

if unexpected_slides:
    raise ValueError(
        f"[ABORT] The extracted file contains slide(s) NOT in EXCLUDED_SLIDES — "
        f"refusing to merge unexpected data:\n  {sorted(unexpected_slides)}"
    )

missing_from_extract = EXCLUDED_STEMS - set(extracted_slide_counts.keys())
print(f"[OK] All {extracted_row_count:,} extracted rows belong to expected slides "
      f"({len(extracted_slide_counts)}/{len(EXCLUDED_STEMS)} of EXCLUDED_SLIDES found in the extract).")
if missing_from_extract:
    print(f"  [NOTE] These excluded slides had no rows in the extracted file "
          f"(nothing to add for them): {sorted(missing_from_extract)}")

# ── 4. Backup local CSV before touching anything ──────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = f"{LOCAL_CSV_PATH}.backup_{timestamp}"
shutil.copy2(LOCAL_CSV_PATH, backup_path)
print(f"[OK] Backup written: {backup_path}")

# ── 5. Write merged result to a temp file, verify, then atomically swap in ─
tmp_path = f"{LOCAL_CSV_PATH}.merge_tmp"
with open(tmp_path, "w", newline="") as f_out:
    writer = csv.writer(f_out)
    writer.writerow(local_header)

    with open(LOCAL_CSV_PATH, "r", newline="") as f_local:
        reader = csv.reader(f_local)
        next(reader)
        for row in reader:
            writer.writerow(row)

    with open(EXTRACTED_CSV_PATH, "r", newline="") as f_ext:
        reader = csv.reader(f_ext)
        next(reader)
        for row in reader:
            writer.writerow(row)

# ── Verify the temp file's row count before swapping in ───────────────────
with open(tmp_path, "r", newline="") as f:
    tmp_row_count = sum(1 for _ in csv.reader(f)) - 1  # minus header

expected_row_count = local_row_count + extracted_row_count
if tmp_row_count != expected_row_count:
    os.remove(tmp_path)
    raise RuntimeError(
        f"[ABORT] Row count mismatch after merge — did NOT touch your original file.\n"
        f"  Expected: {expected_row_count:,} ({local_row_count:,} local + {extracted_row_count:,} extracted)\n"
        f"  Got     : {tmp_row_count:,}\n"
        f"  Temp file left at {tmp_path} for inspection."
    )

os.replace(tmp_path, LOCAL_CSV_PATH)

print(f"\n[SUCCESS] Merge complete.")
print(f"  Rows before : {local_row_count:,}")
print(f"  Rows added  : {extracted_row_count:,}")
print(f"  Rows after  : {tmp_row_count:,}")
print(f"  Slides added: {len(extracted_slide_counts)}")
print(f"  Backup kept : {backup_path}")
