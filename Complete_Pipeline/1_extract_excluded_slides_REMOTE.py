"""
Run this on the REMOTE machine.

Extracts ONLY the rows belonging to EXCLUDED_SLIDES from the remote's full
metadata CSV (patch_metadata_merged.csv or patch_metadata_nonwhite.csv —
whichever one you're filling the gap for) into a small standalone CSV.
That small file is what you transfer back to your local machine, instead
of the whole multi-GB CSV.

Reads and writes row-by-row (csv module, not pandas) so:
  - it never loads the full remote CSV into memory
  - every field is copied through as the exact original string — no
    float reparsing/reformatting, so no precision is lost or altered
"""

import csv
import os
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════
REMOTE_CSV_PATH  = "/path/to/remote/patch_metadata_nonwhite.csv"   # <-- EDIT: the remote CSV to pull from
OUTPUT_EXTRACT_CSV = "./extracted_excluded_slides.csv"             # <-- this is the small file you'll transfer

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
# CSV's slide_name column has no extension (Path(czi_filename).stem is what
# the pipeline writes there) — strip .czi the same way here so they match.
EXCLUDED_STEMS = {Path(s).stem for s in EXCLUDED_SLIDES}
# ══════════════════════════════════════════════════════════════════════════

if not os.path.exists(REMOTE_CSV_PATH):
    raise FileNotFoundError(f"REMOTE_CSV_PATH does not exist: {REMOTE_CSV_PATH}")

per_slide_counts = {stem: 0 for stem in EXCLUDED_STEMS}
total_rows_written = 0

with open(REMOTE_CSV_PATH, "r", newline="") as f_in, \
     open(OUTPUT_EXTRACT_CSV, "w", newline="") as f_out:

    reader = csv.reader(f_in)
    writer = csv.writer(f_out)

    header = next(reader)
    if "slide_name" not in header:
        raise ValueError(f"'slide_name' column not found in header: {header}")
    slide_col_idx = header.index("slide_name")
    writer.writerow(header)

    for row in reader:
        slide_val = row[slide_col_idx]
        if slide_val in per_slide_counts:
            writer.writerow(row)
            per_slide_counts[slide_val] += 1
            total_rows_written += 1

# ── Report ──────────────────────────────────────────────────────────────
print(f"Source CSV : {REMOTE_CSV_PATH}")
print(f"Output CSV : {OUTPUT_EXTRACT_CSV}")
print(f"\nRows extracted per slide:")
missing_slides = []
for stem, count in per_slide_counts.items():
    status = "" if count > 0 else "  <-- NOT FOUND in source CSV"
    print(f"  {stem:35s}: {count:6,d}{status}")
    if count == 0:
        missing_slides.append(stem)

print(f"\nTotal rows written : {total_rows_written:,}")
print(f"Slides matched      : {len(EXCLUDED_STEMS) - len(missing_slides)}/{len(EXCLUDED_STEMS)}")

if missing_slides:
    print(f"\n[WARNING] {len(missing_slides)} excluded slide(s) had ZERO matching rows "
          f"in the source CSV — they may not have been processed on the remote either:")
    for s in missing_slides:
        print(f"    {s}")
    print("Double-check these before transferring — the merge step will treat this as expected,")
    print("but you won't actually be closing the gap for these slides.")
else:
    print("\nAll excluded slides were found in the source CSV. Safe to transfer "
          f"{OUTPUT_EXTRACT_CSV} to your local machine.")
