"""
Validate .pt Feature Files — SurGen Pipeline
==============================================
Checks one .pt file (or every .pt file under a folder, recursively) for the
failure modes this pipeline has actually hit — both structural and,
now, value-level:

  STRUCTURAL
  1. File loads and contains a torch.Tensor.
  2. Shape is (5, D) — five crops per patch, D = model's embedding dim.
  3. dtype is float16.
  4. No "view-bloat": the tensor's underlying storage is exactly sized to
     the tensor itself. If a tensor was saved without .clone() after being
     sliced out of a larger (B, 5, D) batch tensor, its storage still holds
     the WHOLE batch's data even though only one patch is "visible" — this
     is exactly the bug that caused ~200KB files instead of the expected
     ~20-40KB. This check catches it directly, regardless of exact file size.
  5. On-disk file size is reported (min/max/mean/median) so outliers are
     easy to spot even if #4 somehow doesn't catch them.

  VALUE-LEVEL
  6. No NaN / Inf values.
  7. Not silently all-zero (usually means extraction failed on that patch).
  8. Per-crop L2 norm isn't near-zero (a "dead"/degenerate embedding).
  9. No absurdly large-magnitude values (numerical blow-up before the
     float16 cast, or garbage data read as an image).
 10. Values aren't nearly constant across the whole tensor (a collapsed,
     uninformative embedding — e.g. the model output didn't actually
     depend on the input).
 11. The 5 crops within one file aren't (near-)identical to each other —
     a common symptom of a broken FiveCrop transform or patch-reading bug
     (all 5 "crops" end up being the same image).
 12. (Directory mode only) Cross-file duplicate detection: hashes each
     tensor's raw bytes and flags groups of DIFFERENT patches that produced
     BYTE-IDENTICAL embeddings — a strong signal that something upstream is
     repeatedly reading/encoding the same (often blank/broken) patch.

Usage: edit TARGET_PATH and MODEL_NAME below, then run:
    python validate_pt_features.py
"""

import sys
import hashlib
import statistics
import torch
from pathlib import Path
from collections import defaultdict

# ══════════════════════════════════════════════════════════════════════════
# Configuration — edit these two lines
# ══════════════════════════════════════════════════════════════════════════

# A single .pt file OR a directory to scan recursively for all .pt files
# (e.g. a slide's feature folder, or a whole model's features/ root).
TARGET_PATH = r"D:\Aamir Gulzar\KSA_project2\surgen_data\surgen_processed\conch1-5\features\SR1482_40X_HE_T080_01\SR1482_40X_HE_T080_01_20850.pt"

# Which model produced these features — used to check the expected embedding
# dimension. Set to None to skip the dimension check (e.g. CONCH 1.5, whose
# dim can vary by checkpoint).
MODEL_NAME = "conch1-5"   # "conch-v1" | "uni2-h" | "h-optimus-1" | "virchow2" | None | "conch1-5"

EXPECTED_DIMS = {
    "conch-v1"   : 512,
    "conch1-5"   : 768,
    "uni2-h"     : 1536,
    "h-optimus-1": 1536,
    "virchow2"   : 2560,
}

N_CROPS = 5   # FiveCrop — every patch produces 5 crop embeddings

# If TARGET_PATH is a directory, cap how many files get fully validated.
# 0 = check every .pt file found (fine for thousands, slow for millions —
# set a sample size to spot-check large feature sets quickly instead).
MAX_FILES_TO_CHECK = 0

# ── Value-level check thresholds ─────────────────────────────────────────────
# Per-crop L2 norm below this is treated as a "dead"/degenerate embedding.
MIN_L2_NORM = 1e-4

# Any single value with |x| above this is flagged as suspiciously large —
# real embeddings from these models don't normally produce values this big.
MAX_ABS_VALUE = 1e4

# If the std of ALL values in the tensor is below this, the tensor is
# essentially constant — almost certainly a broken/collapsed extraction.
MIN_STD_ACROSS_TENSOR = 1e-6

# Flag if any two of the 5 crops in a file are (near-)identical.
CHECK_DUPLICATE_CROPS = True
DUPLICATE_CROP_ATOL   = 1e-3

# Directory mode only: flag DIFFERENT patches whose saved tensors are
# byte-identical. Adds a small hashing cost per file.
CHECK_CROSS_FILE_DUPLICATES = True


# ══════════════════════════════════════════════════════════════════════════
# Validation
# ══════════════════════════════════════════════════════════════════════════

def find_pt_files(target_path: str) -> list:
    p = Path(target_path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(p.rglob("*.pt"))
    raise FileNotFoundError(f"TARGET_PATH does not exist: {target_path}")


def _storage_nbytes(t: torch.Tensor) -> int:
    """Works across PyTorch versions (untyped_storage is 2.0+)."""
    try:
        return t.untyped_storage().nbytes()
    except AttributeError:
        return t.storage().size() * t.element_size()


def validate_one(pt_path: Path, expected_dim) -> dict:
    """Run all checks on a single .pt file. Returns a result dict."""
    result = {
        "path": str(pt_path), "ok": True, "errors": [], "warnings": [],
        "shape": None, "dtype": None,
        "file_size_bytes": None, "tensor_bytes": None, "storage_bytes": None,
        "l2_norms": None, "max_abs_value": None, "overall_std": None,
        "content_hash": None,
    }

    try:
        result["file_size_bytes"] = pt_path.stat().st_size
    except OSError as e:
        result["ok"] = False
        result["errors"].append(f"Could not stat file: {e}")
        return result

    try:
        t = torch.load(pt_path, map_location="cpu", weights_only=True)
    except Exception as e:
        result["ok"] = False
        result["errors"].append(f"Failed to load: {e}")
        return result

    if not isinstance(t, torch.Tensor):
        result["ok"] = False
        result["errors"].append(f"Not a tensor — got {type(t)}")
        return result

    result["shape"] = tuple(t.shape)
    result["dtype"] = str(t.dtype)

    # ── Shape ─────────────────────────────────────────────────────────────
    if t.ndim != 2:
        result["ok"] = False
        result["errors"].append(f"Expected 2D tensor, got {t.ndim}D {tuple(t.shape)}")
    else:
        if t.shape[0] != N_CROPS:
            result["ok"] = False
            result["errors"].append(f"Expected {N_CROPS} crops, got {t.shape[0]}")
        if expected_dim is not None and t.shape[1] != expected_dim:
            result["ok"] = False
            result["errors"].append(
                f"Expected embedding dim {expected_dim}, got {t.shape[1]}"
            )

    # ── Dtype ─────────────────────────────────────────────────────────────
    if t.dtype != torch.float16:
        result["warnings"].append(f"Expected float16, got {t.dtype}")

    # ── NaN / Inf ─────────────────────────────────────────────────────────
    if torch.isnan(t).any():
        result["ok"] = False
        result["errors"].append("Contains NaN values")
    if torch.isinf(t).any():
        result["ok"] = False
        result["errors"].append("Contains Inf values")

    # ── All-zero sanity check ────────────────────────────────────────────
    if t.numel() > 0 and torch.count_nonzero(t) == 0:
        result["warnings"].append("Tensor is entirely zeros — extraction may have failed")

    # ── View-bloat check: storage must be exactly sized to the tensor ──────
    tensor_bytes  = t.numel() * t.element_size()
    storage_bytes = _storage_nbytes(t)
    result["tensor_bytes"]  = tensor_bytes
    result["storage_bytes"] = storage_bytes

    if storage_bytes > tensor_bytes:
        result["ok"] = False
        result["errors"].append(
            f"VIEW-BLOAT DETECTED: storage is {storage_bytes:,} bytes but the "
            f"tensor only needs {tensor_bytes:,} bytes — saved without .clone(), "
            f"file is carrying hidden extra data from a larger parent tensor."
        )

    # ── Value-level checks (skip if NaN/Inf already broke things) ──────────
    has_nan_or_inf = torch.isnan(t).any() or torch.isinf(t).any()
    if t.numel() > 0 and not has_nan_or_inf:
        t_f = t.float()

        # Per-crop L2 norm — near-zero means a dead/degenerate embedding
        if t_f.ndim == 2:
            norms = t_f.norm(dim=1)
            result["l2_norms"] = [round(x, 4) for x in norms.tolist()]
            if (norms < MIN_L2_NORM).any():
                result["ok"] = False
                dead_idx = (norms < MIN_L2_NORM).nonzero().flatten().tolist()
                result["errors"].append(
                    f"Degenerate embedding: crop(s) {dead_idx} have near-zero "
                    f"L2 norm (< {MIN_L2_NORM}) — likely a blank/broken patch."
                )

        # Absurdly large values — numerical blow-up or garbage input
        max_abs = t_f.abs().max().item()
        result["max_abs_value"] = round(max_abs, 4)
        if max_abs > MAX_ABS_VALUE:
            result["ok"] = False
            result["errors"].append(
                f"Value magnitude too large: max |x| = {max_abs:.2f} "
                f"(threshold {MAX_ABS_VALUE}) — possible numerical instability."
            )

        # Overall constant/collapsed tensor
        overall_std = t_f.std().item()
        result["overall_std"] = round(overall_std, 6)
        if overall_std < MIN_STD_ACROSS_TENSOR:
            result["ok"] = False
            result["errors"].append(
                f"Tensor values are nearly constant (std={overall_std:.2e}) — "
                f"embedding appears collapsed/uninformative."
            )

        # Duplicate crops within the same file
        if CHECK_DUPLICATE_CROPS and t_f.ndim == 2 and t_f.shape[0] == N_CROPS:
            dup_pairs = []
            for i in range(N_CROPS):
                for j in range(i + 1, N_CROPS):
                    if torch.allclose(t_f[i], t_f[j], atol=DUPLICATE_CROP_ATOL):
                        dup_pairs.append((i, j))
            if dup_pairs:
                result["warnings"].append(
                    f"Crops appear (near-)identical: pairs {dup_pairs} — "
                    f"check the FiveCrop transform / patch reading logic."
                )

    # ── Content hash for cross-file duplicate detection ─────────────────────
    if CHECK_CROSS_FILE_DUPLICATES:
        try:
            result["content_hash"] = hashlib.md5(
                t.contiguous().numpy().tobytes()
            ).hexdigest()
        except Exception:
            pass  # non-fatal — duplicate detection just skips this file

    return result


def main():
    pt_files = find_pt_files(TARGET_PATH)
    if not pt_files:
        print(f"[ERROR] No .pt files found under: {TARGET_PATH}")
        sys.exit(1)

    if MAX_FILES_TO_CHECK and len(pt_files) > MAX_FILES_TO_CHECK:
        print(f"[INFO] Found {len(pt_files)} .pt files — checking first "
              f"{MAX_FILES_TO_CHECK} (set MAX_FILES_TO_CHECK = 0 to check all).")
        pt_files = pt_files[:MAX_FILES_TO_CHECK]
    else:
        print(f"[INFO] Found {len(pt_files)} .pt file(s) to check.")

    expected_dim = EXPECTED_DIMS.get(MODEL_NAME) if MODEL_NAME else None
    if MODEL_NAME and expected_dim is None:
        print(f"[WARN] Unknown MODEL_NAME '{MODEL_NAME}' — dimension check skipped.")

    n_ok = n_bad = 0
    file_sizes   = []
    all_results  = []
    bad_results  = []
    hash_to_paths = defaultdict(list)

    for pt_path in pt_files:
        res = validate_one(pt_path, expected_dim)
        all_results.append(res)
        if res["ok"]:
            n_ok += 1
        else:
            n_bad += 1
            bad_results.append(res)
        if res["file_size_bytes"] is not None:
            file_sizes.append(res["file_size_bytes"])
        if res["content_hash"] is not None:
            hash_to_paths[res["content_hash"]].append(res["path"])

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Checked : {len(pt_files)}")
    print(f"  OK      : {n_ok}")
    print(f"  FAILED  : {n_bad}")

    if file_sizes:
        sizes_kb = [s / 1024 for s in file_sizes]
        print(f"\n  File size (KB) — min: {min(sizes_kb):.1f}  "
              f"max: {max(sizes_kb):.1f}  "
              f"mean: {statistics.mean(sizes_kb):.1f}  "
              f"median: {statistics.median(sizes_kb):.1f}")
        if len(sizes_kb) > 1 and statistics.stdev(sizes_kb) > statistics.mean(sizes_kb) * 0.5:
            print(f"  [WARN] File sizes vary a lot (stdev {statistics.stdev(sizes_kb):.1f} KB) "
                  f"— worth checking the outliers below.")

    if bad_results:
        print(f"\n{'='*70}")
        print(f"FAILED FILES (showing up to 20)")
        print(f"{'='*70}")
        for res in bad_results[:20]:
            print(f"\n  {res['path']}")
            print(f"    shape: {res['shape']}  dtype: {res['dtype']}  "
                  f"size: {(res['file_size_bytes'] or 0)/1024:.1f} KB")
            if res["l2_norms"] is not None:
                print(f"    L2 norms per crop : {res['l2_norms']}")
            if res["max_abs_value"] is not None:
                print(f"    max |value|       : {res['max_abs_value']}")
            if res["overall_std"] is not None:
                print(f"    overall std       : {res['overall_std']}")
            for err in res["errors"]:
                print(f"    [ERROR] {err}")
            for warn in res["warnings"]:
                print(f"    [WARN] {warn}")
        if len(bad_results) > 20:
            print(f"\n  ... and {len(bad_results) - 20} more failures not shown.")

    # ── Single-file mode: always show full value details, pass or fail ──────
    # (the summary alone doesn't tell you anything about the actual numbers,
    # and that's usually exactly what you want to see when checking one file)
    if len(pt_files) == 1 and all_results[0]["ok"]:
        res = all_results[0]
        print(f"\n{'='*70}")
        print(f"FILE DETAILS")
        print(f"{'='*70}")
        print(f"  {res['path']}")
        print(f"    shape: {res['shape']}  dtype: {res['dtype']}  "
              f"size: {(res['file_size_bytes'] or 0)/1024:.1f} KB")
        print(f"    tensor bytes      : {res['tensor_bytes']:,}")
        print(f"    storage bytes     : {res['storage_bytes']:,}"
              + ("  (matches tensor — no view-bloat)" if res['storage_bytes'] == res['tensor_bytes'] else ""))
        if res["l2_norms"] is not None:
            print(f"    L2 norms per crop : {res['l2_norms']}")
        if res["max_abs_value"] is not None:
            print(f"    max |value|       : {res['max_abs_value']}")
        if res["overall_std"] is not None:
            print(f"    overall std       : {res['overall_std']}")
        if res["warnings"]:
            for warn in res["warnings"]:
                print(f"    [WARN] {warn}")

    # ── Cross-file duplicate report ─────────────────────────────────────────
    dup_groups = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}
    if CHECK_CROSS_FILE_DUPLICATES and dup_groups:
        n_dup_files = sum(len(paths) for paths in dup_groups.values())
        print(f"\n{'='*70}")
        print(f"CROSS-FILE DUPLICATE EMBEDDINGS")
        print(f"{'='*70}")
        print(f"  {len(dup_groups)} group(s) of identical embeddings across "
              f"{n_dup_files} file(s) total — different patches producing the "
              f"exact same feature vector usually means something upstream is "
              f"stuck reading/encoding the same patch repeatedly.")
        for i, (h, paths) in enumerate(list(dup_groups.items())[:10]):
            print(f"\n  Group {i+1} (hash {h[:8]}…, {len(paths)} files):")
            for p in paths[:5]:
                print(f"    {p}")
            if len(paths) > 5:
                print(f"    ... and {len(paths) - 5} more")
        if len(dup_groups) > 10:
            print(f"\n  ... and {len(dup_groups) - 10} more duplicate group(s) not shown.")

    print(f"\n{'='*70}")
    overall_ok = (n_bad == 0) and not dup_groups
    if overall_ok:
        print(f"✓ All {n_ok} file(s) passed validation.")
    else:
        print(f"✗ Validation failed — {n_bad} file(s) with errors"
              + (f", {len(dup_groups)} duplicate group(s)" if dup_groups else "")
              + ".")
    print(f"{'='*70}")

    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()