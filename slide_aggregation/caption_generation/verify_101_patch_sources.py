"""
verify_101_patch_sources.py
----------------------------
For each of the 101 pathologist patches in fewshot_101_patches.xlsx,
determine which dataset the image file actually comes from:
  - NCT-CRC-HE-100K-NONORM  (CRC-100K train set)
  - CRC-VAL-HE-7K           (CRC-100K validation set)
  - TCGA patch_data          (TCGA slides)
  - NOT FOUND                (missing from all locations)
"""

import os
import pandas as pd
from collections import Counter

# ── Paths ─────────────────────────────────────────────────────────────────────
EXCEL_FILE  = 'fewshot_101_patches.xlsx'
CRC_DIR     = 'D:/Aamir Gulzar/dataset/CRC100K/NCT-CRC-HE-100K-NONORM'
CRC_VAL_DIR = 'D:/Aamir Gulzar/dataset/CRC100K/CRC-VAL-HE-7K'
TCGA_DIR    = 'D:/Aamir Gulzar/KSA_project2/dataset/patch_data'

# ── Load Excel ────────────────────────────────────────────────────────────────
df = pd.read_excel(EXCEL_FILE)
df['Filename']      = df['Filename'].astype(str).str.strip()
df['Primary Label'] = df['Primary Label'].astype(str).str.strip()

print(f'Loaded {len(df)} patches from {EXCEL_FILE}')
print()

# ── Check each patch ──────────────────────────────────────────────────────────
results = []

for _, row in df.iterrows():
    fname  = row['Filename']
    label  = row['Primary Label']
    source = row.get('Source Sheet', '')

    # CRC-100K filenames follow CLASS-XXXXXXXX.tif pattern
    crc_class = fname.split('-')[0]

    found_in = 'NOT FOUND'
    found_path = ''

    # 1. NCT-CRC-HE-100K-NONORM
    p = os.path.join(CRC_DIR, crc_class, fname)
    if os.path.exists(p):
        found_in   = 'NCT-CRC-HE-100K-NONORM'
        found_path = p

    # 2. CRC-VAL-HE-7K
    elif os.path.exists(os.path.join(CRC_VAL_DIR, crc_class, fname)):
        found_in   = 'CRC-VAL-HE-7K'
        found_path = os.path.join(CRC_VAL_DIR, crc_class, fname)

    # 3. TCGA patch_data — slide directories contain the filename
    else:
        # TCGA files: TCGA-XX-XXXX_..._patchNNNNN.png  (no CLASS- prefix)
        # Try a direct search: slide = first part before _x
        stem  = os.path.splitext(fname)[0]
        slide = stem.split('_x')[0] if '_x' in stem else stem
        tcga_path = os.path.join(TCGA_DIR, slide, fname)
        if os.path.exists(tcga_path):
            found_in   = 'TCGA patch_data'
            found_path = tcga_path

    results.append({
        '#':            row['#'],
        'Filename':     fname,
        'Primary Label': label,
        'Source Sheet': source,
        'Found In':     found_in,
        'Full Path':    found_path,
    })

df_out = pd.DataFrame(results)

# ── Summary ───────────────────────────────────────────────────────────────────
print('=' * 60)
print('SOURCE BREAKDOWN')
print('=' * 60)
source_counts = Counter(df_out['Found In'])
for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
    pct = 100 * cnt / len(df_out)
    print(f'  {src:<30s}  {cnt:3d}  ({pct:.1f}%)')

print()
print('=' * 60)
print('PER-CLASS BREAKDOWN')
print('=' * 60)
pivot = df_out.groupby(['Primary Label', 'Found In']).size().unstack(fill_value=0)
print(pivot.to_string())

print()
print('=' * 60)
print('NOT FOUND PATCHES')
print('=' * 60)
not_found = df_out[df_out['Found In'] == 'NOT FOUND']
if not_found.empty:
    print('  All 101 patches found.')
else:
    print(f'  {len(not_found)} patches not found:')
    for _, r in not_found.iterrows():
        print(f'    #{r["#"]:3d}  {r["Filename"]}  [{r["Primary Label"]}]')

# ── Save detailed CSV ─────────────────────────────────────────────────────────
out_csv = 'Results/patch_source_verification_101.csv'
os.makedirs('Results', exist_ok=True)
df_out.to_csv(out_csv, index=False)
print()
print(f'Detailed results saved -> {out_csv}')
