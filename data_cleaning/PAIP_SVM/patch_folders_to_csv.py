import os
import csv

# ── CONFIG ──────────────────────────────────────────────────────────────────
LABEL0_DIR = "test_label0"
LABEL1_DIR = "test_label1"
OUTPUT_CSV = "patch_labels.csv"
# ────────────────────────────────────────────────────────────────────────────

rows = []

for label, folder in [(0, LABEL0_DIR), (1, LABEL1_DIR)]:
    if not os.path.isdir(folder):
        print(f"[WARN] Folder not found: {folder}, skipping.")
        continue
    for fname in os.listdir(folder):
        # Strip extension if present (e.g. .png, .jpg, .tif)
        patch_name = os.path.splitext(fname)[0]
        rows.append({"patch_name": patch_name, "label": label})

rows.sort(key=lambda r: r["patch_name"])

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["patch_name", "label"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Done. {len(rows)} patches written to '{OUTPUT_CSV}'.")