import os
import re
import pandas as pd
from pathlib import Path

# ==========================================================
# PATHS
# ==========================================================

CZI_ROOT          = r"/media/dp-psau/dp-psau-wsi/SurGen/S-BIAD1285/Files"
SR386_LABELS_CSV  = r"/media/dp-psau/dp-psau-wsi/SurGen/S-BIAD1285/Files/SR386_labels.csv"
SR1482_LABELS_CSV = r"/media/dp-psau/dp-psau-wsi/SurGen/S-BIAD1285/Files/SR1482_labels.csv"

OUTPUT_CSV = "surgen_labels.csv"

# ==========================================================
# LOAD LABEL FILES
# ==========================================================

sr386 = pd.read_csv(SR386_LABELS_CSV)
sr386['case_id'] = sr386['case_id'].astype(str).str.zfill(3)

sr1482 = pd.read_csv(SR1482_LABELS_CSV)
sr1482['case_id'] = sr1482['case_id'].astype(str).str.zfill(3)

# ==========================================================
# LABEL LOOKUP
# ==========================================================

def get_numeric_label(czi_filename):
    stem = Path(czi_filename).stem

    match = re.search(r'_T(\d{3})_', stem)
    if not match:
        return -1

    case_id = match.group(1)

    if stem.upper().startswith("SR386"):
        row = sr386[sr386['case_id'] == case_id]

        if row.empty:
            return -1

        val = row.iloc[0]['mmr_loss_binary']

        if val == 1:
            return 1
        elif val == 0:
            return 0
        else:
            return -1

    elif stem.upper().startswith("SR1482"):
        row = sr1482[sr1482['case_id'] == case_id]

        if row.empty:
            return -1

        val = str(row.iloc[0]['MSI']).strip().lower()

        if val == "msi high":
            return 1

        elif val == "no msi":
            return 0

        else:
            # MSI Low, Not Performed, Failed, Insufficient, etc.
            return -1

    return -1

# ==========================================================
# FIND ALL CZI FILES
# ==========================================================

all_rows = []

for dirpath, _, filenames in os.walk(DATASET_ROOT):
    for fname in filenames:

        if not fname.lower().endswith(".czi"):
            continue

        label = get_numeric_label(fname)

        all_rows.append({
            "WSI_Id": fname,      # exact slide name including .czi
            "label_desc": label
        })

# ==========================================================
# SAVE
# ==========================================================

df = pd.DataFrame(all_rows)

df = df.sort_values("WSI_Id").reset_index(drop=True)

df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved {len(df)} rows to {OUTPUT_CSV}")

print(df['label_desc'].value_counts(dropna=False))