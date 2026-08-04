"""
Build CRC-100K Relabel Review Excel for 4 short labels: ADE, MES, SIG, PLC.
"""

import os
import io
import math
import pandas as pd
from PIL import Image
import openpyxl
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.drawing.image import Image as XLImage

# ── Paths ────────────────────────────────────────────────────────────────────
CSV_PATH = r"D:\Aamir Gulzar\KSA_project2\Cancer-detection-classifier\Analysis_and_Visualization\Results\CRC100K_ZeroShot_Statistics\crc100k_raw_predictions.csv"
CRC_DIR   = r"D:\Aamir Gulzar\dataset\CRC100K\NCT-CRC-HE-100K-NONORM"
CRC_VAL_DIR = r"D:\Aamir Gulzar\dataset\CRC100K\CRC-VAL-HE-7K"
OUTPUT_PATH = r"D:\Aamir Gulzar\KSA_project2\Cancer-detection-classifier\Analysis_and_Visualization\Results\CRC100K_ZeroShot_Statistics\crc100k_relabel_review.xlsx"

TARGET_LABELS = ["ADE", "MES", "SIG", "PLC"]
BUDGET = 30
THUMB_SIZE = (120, 120)

# ── Load CSV ─────────────────────────────────────────────────────────────────
print("Loading CSV …")
df = pd.read_csv(CSV_PATH)
print(f"  Total rows: {len(df)}")

# ── Allocation ───────────────────────────────────────────────────────────────
def allocate(df_pred: pd.DataFrame, budget: int) -> pd.DataFrame:
    """
    Proportional allocation with floor-of-1 and 'take all if n < share'.
    Returns sorted rows for the sheet.
    """
    counts = (
        df_pred[df_pred["gt_label"] != "BACK"]
        .groupby("gt_label")
        .size()
        .rename("n_gt")
        .reset_index()
        .sort_values("n_gt", ascending=False)
    )
    total = counts["n_gt"].sum()
    if total == 0:
        return pd.DataFrame(), pd.DataFrame()

    # raw proportional share
    counts["share_raw"] = counts["n_gt"] / total * budget
    counts["share"] = counts["share_raw"].apply(lambda x: max(1, round(x)))
    # cap at actual available
    counts["taken"] = counts.apply(lambda r: min(r["n_gt"], r["share"]), axis=1)
    return counts


def sample_rows(df_pred: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    """Top-`taken` by score descending for each gt_label."""
    frames = []
    for _, row in counts.iterrows():
        subset = df_pred[df_pred["gt_label"] == row["gt_label"]].nlargest(int(row["taken"]), "score")
        frames.append(subset)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


print("\nAllocation breakdown:")
print("=" * 70)

summary_rows = []   # for Summary sheet
selected_all = []   # for Review sheet

for label in TARGET_LABELS:
    df_pred = df[df["pred_label"] == label].copy()
    df_no_back = df_pred[df_pred["gt_label"] != "BACK"]
    counts = allocate(df_no_back, BUDGET)

    if isinstance(counts, tuple):          # empty case guard
        print(f"\n{label}: no candidates after BACK exclusion")
        continue

    print(f"\n  {label}  (total non-BACK candidates = {counts['n_gt'].sum()})")
    print(f"  {'GT Source':<12} {'n_gt':>6} {'share':>6} {'taken':>6}")
    print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*6}")
    for _, r in counts.iterrows():
        print(f"  {r['gt_label']:<12} {r['n_gt']:>6} {r['share']:>6} {r['taken']:>6}")
        summary_rows.append({
            "Target Label": label,
            "GT Source": r["gt_label"],
            "Total Candidates (n_gt)": int(r["n_gt"]),
            "Allocated Share": int(r["share"]),
            "Actually Taken": int(r["taken"]),
        })
    tot_taken = int(counts["taken"].sum())
    print(f"  {'TOTAL':<12} {counts['n_gt'].sum():>6} {counts['share'].sum():>6} {tot_taken:>6}")
    summary_rows.append({
        "Target Label": f"{label} TOTAL",
        "GT Source": "",
        "Total Candidates (n_gt)": int(counts["n_gt"].sum()),
        "Allocated Share": int(counts["share"].sum()),
        "Actually Taken": tot_taken,
    })

    sampled = sample_rows(df_no_back, counts)
    sampled["_sort_label"] = label
    selected_all.append(sampled)

# Grand total row
review_df = pd.concat(selected_all, ignore_index=True)
summary_rows.append({
    "Target Label": "GRAND TOTAL",
    "GT Source": "",
    "Total Candidates (n_gt)": "",
    "Allocated Share": "",
    "Actually Taken": len(review_df),
})

print(f"\n{'='*70}")
print(f"Total selected for Review sheet: {len(review_df)}")

# Sort Review sheet: by target label group, then by gt_label within group
review_df["_sort_label"] = pd.Categorical(review_df["_sort_label"], categories=TARGET_LABELS, ordered=True)
review_df = review_df.sort_values(["_sort_label", "gt_label", "score"], ascending=[True, True, False]).reset_index(drop=True)

# ── Locate images ─────────────────────────────────────────────────────────────
def find_image(patch_name: str, gt_label: str) -> str | None:
    for base in (CRC_DIR, CRC_VAL_DIR):
        p = os.path.join(base, gt_label, patch_name)
        if os.path.isfile(p):
            return p
    return None

print("\nLocating images …")
not_found = []
paths = []
for _, row in review_df.iterrows():
    p = find_image(row["patch_name"], row["gt_label"])
    paths.append(p)
    if p is None:
        not_found.append(row["patch_name"])

review_df["_img_path"] = paths
print(f"  Found: {len(review_df) - len(not_found)}  |  Not found: {len(not_found)}")
if not_found:
    for n in not_found[:10]:
        print(f"    MISSING: {n}")

# ── Build Excel ───────────────────────────────────────────────────────────────
print("\nBuilding Excel …")
wb = openpyxl.Workbook()

# ── Helper styles ─────────────────────────────────────────────────────────────
HDR_FILL   = PatternFill("solid", fgColor="1F4E79")
HDR_FONT   = Font(bold=True, color="FFFFFF", size=11)
SUBHDR_FILL = PatternFill("solid", fgColor="2E75B6")
SUBHDR_FONT = Font(bold=True, color="FFFFFF", size=10)
TOTAL_FILL  = PatternFill("solid", fgColor="D6E4F0")
TOTAL_FONT  = Font(bold=True, size=10)
GRAND_FILL  = PatternFill("solid", fgColor="9DC3E6")
GRAND_FONT  = Font(bold=True, size=11)
CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT   = Alignment(horizontal="left",   vertical="center", wrap_text=True)

thin = Side(style="thin")
BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)

def apply_hdr(cell, fill, font, alignment=CENTER):
    cell.fill = fill; cell.font = font; cell.alignment = alignment; cell.border = BORDER

def apply_cell(cell, alignment=CENTER):
    cell.alignment = alignment; cell.border = BORDER

# ── Summary sheet ─────────────────────────────────────────────────────────────
ws_sum = wb.active
ws_sum.title = "Summary"

headers = ["Target Label", "GT Source", "Total Candidates (n_gt)", "Allocated Share", "Actually Taken"]
for col, h in enumerate(headers, 1):
    c = ws_sum.cell(row=1, column=col, value=h)
    apply_hdr(c, HDR_FILL, HDR_FONT)

row_idx = 2
for sr in summary_rows:
    is_total_row  = str(sr["Target Label"]).endswith("TOTAL")
    is_grand_row  = sr["Target Label"] == "GRAND TOTAL"
    for col, key in enumerate(headers, 1):
        c = ws_sum.cell(row=row_idx, column=col, value=sr[key])
        if is_grand_row:
            c.fill = GRAND_FILL; c.font = GRAND_FONT
        elif is_total_row:
            c.fill = TOTAL_FILL; c.font = TOTAL_FONT
        c.alignment = CENTER; c.border = BORDER
    row_idx += 1

ws_sum.column_dimensions["A"].width = 18
ws_sum.column_dimensions["B"].width = 14
ws_sum.column_dimensions["C"].width = 24
ws_sum.column_dimensions["D"].width = 18
ws_sum.column_dimensions["E"].width = 16
ws_sum.freeze_panes = "A2"

# ── Review sheet ──────────────────────────────────────────────────────────────
ws_rev = wb.create_sheet("Review")

rev_headers = [
    "Thumbnail",
    "Filename",
    "Original CRC Label (GT)",
    "Model Predicted Label",
    "Model Confidence",
    "Pathologist Final Label",
    "Notes",
]
col_widths = [18, 28, 24, 22, 18, 24, 30]
ROW_HEIGHT = 95  # points (~125 px)
THUMB_COL   = 1
FNAME_COL   = 2
GT_COL      = 3
PRED_COL    = 4
CONF_COL    = 5
PATH_LABEL_COL = 6
NOTES_COL   = 7

for col, (h, w) in enumerate(zip(rev_headers, col_widths), 1):
    c = ws_rev.cell(row=1, column=col, value=h)
    apply_hdr(c, HDR_FILL, HDR_FONT)
    ws_rev.column_dimensions[get_column_letter(col)].width = w

ws_rev.row_dimensions[1].height = 22
ws_rev.freeze_panes = "B2"

print("  Embedding thumbnails …")
for i, (_, row) in enumerate(review_df.iterrows(), start=2):
    ws_rev.row_dimensions[i].height = ROW_HEIGHT

    # Thumbnail
    img_path = row["_img_path"]
    if img_path:
        try:
            pil = Image.open(img_path).convert("RGB")
            pil.thumbnail(THUMB_SIZE, Image.LANCZOS)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            buf.seek(0)
            xl_img = XLImage(buf)
            xl_img.width  = THUMB_SIZE[0]
            xl_img.height = THUMB_SIZE[1]
            # anchor to column A of the current row
            cell_anchor = f"A{i}"
            ws_rev.add_image(xl_img, cell_anchor)
        except Exception as e:
            ws_rev.cell(row=i, column=THUMB_COL, value=f"[error: {e}]")
    else:
        ws_rev.cell(row=i, column=THUMB_COL, value="[not found]")
        ws_rev.cell(row=i, column=THUMB_COL).alignment = CENTER

    # Text cells
    ws_rev.cell(row=i, column=FNAME_COL,      value=row["patch_name"]).alignment    = LEFT
    ws_rev.cell(row=i, column=GT_COL,         value=row["gt_label"]).alignment      = CENTER
    ws_rev.cell(row=i, column=PRED_COL,       value=row["pred_label"]).alignment    = CENTER
    ws_rev.cell(row=i, column=CONF_COL,       value=round(float(row["score"]), 4)).alignment = CENTER
    ws_rev.cell(row=i, column=PATH_LABEL_COL, value="").alignment                  = CENTER
    ws_rev.cell(row=i, column=NOTES_COL,      value="").alignment                  = LEFT

    for col in range(1, 8):
        c = ws_rev.cell(row=i, column=col)
        c.border = BORDER
        if not c.alignment:
            c.alignment = CENTER

    # Alternate row shading for readability (skip thumbnail col to not clash with image)
    if i % 2 == 0:
        shade = PatternFill("solid", fgColor="EBF3FB")
        for col in range(2, 8):
            ws_rev.cell(row=i, column=col).fill = shade

    if i % 20 == 0:
        print(f"    … row {i - 1} / {len(review_df)}")

print("  Saving workbook …")
wb.save(OUTPUT_PATH)
print(f"\nDone. Saved to:\n  {OUTPUT_PATH}")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n-- Final counts --------------------------------------------------")
for label in TARGET_LABELS:
    sub = review_df[review_df["_sort_label"] == label]
    print(f"\n  {label} — {len(sub)} total selected")
    for gt, grp in sub.groupby("gt_label"):
        print(f"    {gt}: {len(grp)}")
print(f"\n  Images not found on disk: {len(not_found)}")
if not_found:
    print("  Missing patches:")
    for n in not_found:
        print(f"    {n}")
