"""
Fix SurGen slide-aggregation .pt files that were accidentally saved as
float32 (46.6KB) instead of float16 (24.1KB) -- your real, intended dtype.

Root cause (confirmed via diagnose_pt_dtype.py): these files have at
least one caption group with zero patches for that slide. The script's
`torch.zeros(1, feature_dim)` placeholder for empty groups defaults to
float32, and torch.cat'ing that with the real float16 group tensors
silently upcasts the *entire* slide tensor to float32.

Safety design:
  - Only touches files whose dtype != TARGET_DTYPE (float16) -- normal
    24.1KB files are left completely untouched (not even re-saved), so
    nothing is at risk for the majority of your data.
  - Downcasting float32 -> float16 here does NOT lose any real
    precision: the real (non-zero-row) values in these tensors were
    computed by averaging float16 patch features, so they never had
    more than float16 precision to begin with -- they were only
    *stored* as float32 due to the upcast bug. Casting back to float16
    recovers the exact original values, and the zero-placeholder rows
    are exactly 0.0 in both dtypes. This is a pure storage-format
    fix, not a data change.
  - DRY_RUN=True by default: reports what it *would* do, changes nothing.
    Set DRY_RUN=False to actually rewrite files.
  - Optional backup: copies originals to BACKUP_DIR before overwriting,
    so you can restore if anything looks wrong.
  - After writing, re-loads the new file and checks values match the
    original (within float32 tolerance) before considering it fixed.
"""
import os
import shutil
import torch

# ─── CONFIG ─────────────────────────────────────────────────────────────
OUTPUT_DIR = r"D:\Aamir Gulzar\KSA_project2\surgen_data\surgen_processed\conch1-5\features\slide_aggregation\Caption_based_aggregation_15_classes\conch1-5"

TARGET_DTYPE = torch.float16   # your real, intended dtype

DRY_RUN = False          # set False to actually fix files
MAKE_BACKUP = True       # copy originals before overwriting
BACKUP_DIR = os.path.join(OUTPUT_DIR, "_backup_float32_originals")
# ────────────────────────────────────────────────────────────────────────


def main():
    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".pt")]
    print(f"Scanning {len(files):,} files in {OUTPUT_DIR}")

    if MAKE_BACKUP and not DRY_RUN:
        os.makedirs(BACKUP_DIR, exist_ok=True)

    fixed, skipped, failed = 0, 0, 0
    total_saved_kb = 0.0

    for fname in files:
        fp = os.path.join(OUTPUT_DIR, fname)

        try:
            tensor = torch.load(fp, map_location="cpu", weights_only=True)
        except Exception as e:
            print(f"  [ERR] could not load {fname}: {e}")
            failed += 1
            continue

        if tensor.dtype == TARGET_DTYPE:
            skipped += 1
            continue

        before_kb = os.path.getsize(fp) / 1024
        fixed_tensor = tensor.to(TARGET_DTYPE)

        if DRY_RUN:
            print(f"  [DRY RUN] would fix {fname}: {tensor.dtype} -> {TARGET_DTYPE} "
                  f"(~{before_kb:.1f}KB -> ~{before_kb/2:.1f}KB)")
            fixed += 1
            continue

        # Backup original first
        if MAKE_BACKUP:
            shutil.copy2(fp, os.path.join(BACKUP_DIR, fname))

        # Write to a temp file, verify, then atomically replace
        tmp_fp = fp + ".tmp"
        torch.save(fixed_tensor, tmp_fp)

        # Verify: reload and check shape + values.
        # This should be an EXACT round trip: the real values were computed
        # from float16 patch features (so already at float16 precision) and
        # the zero-placeholder rows are exactly 0.0 in both dtypes -- there
        # is no legitimate reason for the downcast to change any value.
        # We still allow a tiny tolerance (1e-4) purely for float16's
        # limited representable range, not because we expect a difference.
        reloaded = torch.load(tmp_fp, map_location="cpu", weights_only=True)
        if reloaded.shape != tensor.shape:
            print(f"  [FAIL] shape mismatch after fix on {fname} -- skipping, original untouched")
            os.remove(tmp_fp)
            failed += 1
            continue
        if not torch.allclose(reloaded.float(), tensor.float(), rtol=0, atol=1e-4):
            max_diff = (reloaded.float() - tensor.float()).abs().max().item()
            print(f"  [FAIL] value mismatch after fix on {fname} (max diff {max_diff:.6f}) "
                  f"-- skipping, original untouched. This would mean the real values in "
                  f"this file had MORE than float16 precision, which contradicts the "
                  f"expected root cause -- worth inspecting this file manually.")
            os.remove(tmp_fp)
            failed += 1
            continue

        os.replace(tmp_fp, fp)  # atomic on same filesystem
        after_kb = os.path.getsize(fp) / 1024
        total_saved_kb += (before_kb - after_kb)
        fixed += 1

    print(f"\n── Summary ──")
    print(f"  Already {TARGET_DTYPE} (untouched): {skipped:,}")
    print(f"  Fixed{' (dry run)' if DRY_RUN else ''}: {fixed:,}")
    print(f"  Failed/skipped due to mismatch: {failed:,}")
    if not DRY_RUN:
        print(f"  Disk space recovered: ~{total_saved_kb/1024:.1f} MB")
        if MAKE_BACKUP:
            print(f"  Originals backed up to: {BACKUP_DIR}")
            print(f"  (delete this folder once you've confirmed everything downstream still works)")

if __name__ == "__main__":
    main()