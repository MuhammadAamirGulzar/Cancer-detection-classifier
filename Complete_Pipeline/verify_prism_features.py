"""
Verify the SurGen PRISM slide embeddings
=========================================
Checks that what surgen_processing_prism.py wrote is actually loadable by the
slide classifier, and reports exactly which slides are still outstanding.

This is the PRISM counterpart of verify_titan_patch.py. It answers three
questions, in the order they matter:

  1. Is every embedding structurally valid?  — (1, 1280) float32, finite, not
     an all-zero vector. A malformed file is worse than a missing one: the
     pipeline's resume logic treats "the file exists" as "this slide is done",
     so a bad file would be silently carried into the classifier.
  2. Is coverage complete?  — one embedding per labelled slide that has tissue.
     Slides with zero non-white patches are expected to be absent and are
     reported separately, not as failures.
  3. Do they land where the classifier looks?  — the path is compared against
     slide_classification/config/paths.py so a layout drift is caught here
     rather than as an empty cohort weeks later.

Usage:
    python verify_prism_features.py
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch

# Import the pipeline's own configuration so this can never drift from it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from surgen_processing_prism import (           # noqa: E402
    SLIDE_EMBED_DIR,
    SURGEN_LABELS_CSV,
    NONWHITE_INDEX_NPZ,
    NONWHITE_METADATA_CSV,
    TILE_CACHE_DIR,
    TILE_EMBED_DIR,
    TILE_EMBED_FORMAT,
    SAVE_TILE_EMBEDDINGS,
    OUTPUT_ROOT,
)

EXPECTED_SHAPE = (1, 1280)


def _patch_counts() -> dict:
    """{slide_stem: number of non-white patches}, from the cached index."""
    if not os.path.exists(NONWHITE_INDEX_NPZ):
        return {}
    import numpy as np
    with np.load(NONWHITE_INDEX_NPZ, allow_pickle=False) as z:
        slides, offsets = z["slides"], z["offsets"]
        return {str(s): int(offsets[i + 1] - offsets[i])
                for i, s in enumerate(slides)}


def check_tile_embeddings(expected: set, counts: dict) -> list:
    """Verify the Virchow V1 tile embeddings: present, correctly shaped, and
    with one patch_number per vector.

    The patch-count comparison is the one that matters. A tile file is written
    atomically, so it is never half-written — but it CAN legitimately hold
    fewer vectors than the slide has patches, because unreadable patches are
    skipped. A big shortfall means a slide lost a lot of patches to read
    errors and is worth investigating; a small one is normal.
    """
    import numpy as np

    print(f"\n[4/4] Virchow V1 tile embeddings ({TILE_EMBED_FORMAT})")
    print(f"      {TILE_EMBED_DIR}")
    if not os.path.isdir(TILE_EMBED_DIR):
        print("      [PENDING] Directory does not exist yet.")
        return sorted(expected)

    problems, missing, short = [], [], []
    total_bytes = 0

    for stem in sorted(expected):
        if TILE_EMBED_FORMAT == "per_patch":
            d = os.path.join(TILE_EMBED_DIR, stem)
            if not os.path.isdir(d):
                missing.append(stem)
                continue
            n = sum(1 for f in os.listdir(d) if f.endswith(".pt"))
            total_bytes += sum(os.path.getsize(os.path.join(d, f))
                               for f in os.listdir(d) if f.endswith(".pt"))
        else:
            path = os.path.join(TILE_EMBED_DIR, stem + ".npz")
            if not os.path.exists(path):
                missing.append(stem)
                continue
            total_bytes += os.path.getsize(path)
            try:
                with np.load(path) as z:
                    emb, pn = z["embeddings"], z["patch_number"]
            except Exception as e:
                problems.append(f"{stem}: unreadable ({e})")
                continue
            if emb.ndim != 2 or emb.shape[1] != 2560:
                problems.append(f"{stem}: shape {emb.shape}, expected (N, 2560)")
                continue
            if len(emb) != len(pn):
                problems.append(
                    f"{stem}: {len(emb)} vectors but {len(pn)} patch numbers")
                continue
            if len(set(pn.tolist())) != len(pn):
                problems.append(f"{stem}: duplicate patch numbers")
                continue
            n = len(emb)

        want = counts.get(stem)
        if want and n < want * 0.99:
            short.append((stem, n, want))

    done = len(expected) - len(missing)
    print(f"      {done}/{len(expected)} slide(s) have tile embeddings "
          f"({total_bytes / 2**30:.1f} GB on disk)")
    for p in problems[:20]:
        print(f"      [BAD] {p}")
    if short:
        print(f"      [WARN] {len(short)} slide(s) lost >1% of patches to read errors:")
        for stem, n, want in short[:10]:
            print(f"        {stem}: {n:,} of {want:,} patches "
                  f"({100 * (1 - n / want):.1f}% lost)")
    if missing:
        print(f"      [PENDING] {len(missing)} slide(s) have no tile embeddings yet")

    return missing + [p.split(":")[0] for p in problems]


def _tissue_slides() -> set:
    """Slide stems known to have at least one non-white patch."""
    if os.path.exists(NONWHITE_INDEX_NPZ):
        import numpy as np
        with np.load(NONWHITE_INDEX_NPZ, allow_pickle=False) as z:
            return {str(s) for s in z["slides"]}
    if os.path.exists(NONWHITE_METADATA_CSV):
        print("[INFO] No cached index; scanning the CSV (slow) ...")
        seen = set()
        for chunk in pd.read_csv(NONWHITE_METADATA_CSV, usecols=["slide_name"],
                                 chunksize=2_000_000):
            seen.update(chunk["slide_name"].astype(str).unique())
        return seen
    print("[WARN] Neither the cached index nor the non-white CSV is present — "
          "coverage cannot be checked, only file validity.")
    return set()


def check_file(path: Path) -> dict:
    r = {"file": path.name, "ok": False, "error": None, "shape": None,
         "dtype": None, "mean": None, "std": None}
    try:
        t = torch.load(path, map_location="cpu")
    except Exception as e:
        r["error"] = f"unreadable: {e}"
        return r

    if not isinstance(t, torch.Tensor):
        r["error"] = f"holds {type(t).__name__}, expected a Tensor"
        return r

    r["shape"] = tuple(t.shape)
    r["dtype"] = str(t.dtype)

    if r["shape"] != EXPECTED_SHAPE:
        r["error"] = f"wrong shape {r['shape']}, expected {EXPECTED_SHAPE}"
        return r

    f = t.float().view(-1)
    if torch.isnan(f).any():
        r["error"] = "contains NaN"
        return r
    if torch.isinf(f).any():
        r["error"] = "contains Inf"
        return r
    if bool((f == 0).all()):
        r["error"] = "all-zero vector"
        return r

    r["mean"] = round(float(f.mean()), 6)
    r["std"] = round(float(f.std()), 6)
    r["ok"] = True
    return r


def main():
    print("=" * 74)
    print("SurGen PRISM slide embeddings - verification")
    print("=" * 74)
    print(f"  Directory : {SLIDE_EMBED_DIR}")

    if not os.path.isdir(SLIDE_EMBED_DIR):
        print("\n[FAIL] That directory does not exist - nothing has been written yet.")
        return 1

    files = sorted(Path(SLIDE_EMBED_DIR).glob("*.pt"))
    print(f"  Files     : {len(files)}")
    if not files:
        print("\n[FAIL] No .pt files found.")
        return 1

    # -- 1. Structural validity ----------------------------------------------
    results = [check_file(p) for p in files]
    bad = [r for r in results if not r["ok"]]
    good = [r for r in results if r["ok"]]

    print(f"\n[1/4] Structure - {len(good)} valid, {len(bad)} invalid")
    if good:
        means = [r["mean"] for r in good]
        stds = [r["std"] for r in good]
        print(f"      shape {good[0]['shape']} {good[0]['dtype']}, "
              f"mean in [{min(means):+.4f}, {max(means):+.4f}], "
              f"std in [{min(stds):.4f}, {max(stds):.4f}]")
    for r in bad[:20]:
        print(f"      [BAD] {r['file']}: {r['error']}")
    if len(bad) > 20:
        print(f"      ... and {len(bad) - 20} more")

    # -- 2. Coverage ----------------------------------------------------------
    on_disk = {p.stem for p in files}
    labels = pd.read_csv(SURGEN_LABELS_CSV)
    labelled = set(labels.loc[labels["label_desc"] != -1, "WSI_Id"].astype(str))
    tissue = _tissue_slides()

    expected = (labelled & tissue) if tissue else labelled
    missing = sorted(expected - on_disk)
    no_tissue = sorted(labelled - tissue) if tissue else []
    unlabelled = sorted(on_disk - labelled)

    print(f"\n[2/4] Coverage - {len(on_disk & expected)}/{len(expected)} expected slide(s)")
    if no_tissue:
        print(f"      {len(no_tissue)} labelled slide(s) have no tissue and are "
              f"correctly absent: {', '.join(no_tissue)}")
    if unlabelled:
        print(f"      [WARN] {len(unlabelled)} embedding(s) have no label row: "
              f"{', '.join(unlabelled[:10])}")
    if missing:
        print(f"      [PENDING] {len(missing)} slide(s) still to process")
        for stem in missing[:20]:
            print(f"        {stem}")
        if len(missing) > 20:
            print(f"        ... and {len(missing) - 20} more")

    # -- 3. Classifier path agreement -----------------------------------------
    print("\n[3/4] Classifier path")
    try:
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root))
        from slide_classification.config import paths as P

        want = P.feature_dir("surgen", "PRISM", "PRISM").resolve()
        have = Path(SLIDE_EMBED_DIR).resolve()
        if want == have:
            print("      OK - the classifier reads exactly this directory.")
        else:
            print(f"      [MISMATCH] classifier expects:\n        {want}")
            print(f"      pipeline writes:\n        {have}")
            print("      Fix SLIDE_EMBED_DIR in surgen_processing_prism.py or the "
                  "surgen branch of feature_dir() in config/paths.py.")
        dim = P.expected_dim("PRISM", "PRISM")
        print(f"      expected_dim('PRISM','PRISM') = {dim} "
              f"({'matches' if dim == EXPECTED_SHAPE[1] else 'MISMATCH'})")
    except Exception as e:
        print(f"      [SKIP] Could not import the classifier config: {e}")

    # -- 4. Virchow V1 tile embeddings ----------------------------------------
    tile_problems = []
    if SAVE_TILE_EMBEDDINGS:
        tile_problems = check_tile_embeddings(expected, _patch_counts())
    else:
        print("\n[4/4] Virchow V1 tile embeddings — not saved "
              "(SAVE_TILE_EMBEDDINGS = False)")

    # -- Leftover tile cache ---------------------------------------------------
    if os.path.isdir(TILE_CACHE_DIR):
        leftovers = [d for d in os.listdir(TILE_CACHE_DIR)
                     if os.path.isdir(os.path.join(TILE_CACHE_DIR, d))]
        if leftovers:
            size_gb = sum(
                os.path.getsize(os.path.join(dp, f))
                for d in leftovers
                for dp, _, fs in os.walk(os.path.join(TILE_CACHE_DIR, d))
                for f in fs
            ) / 2**30
            print(f"\n[INFO] {len(leftovers)} slide(s) have a leftover tile cache "
                  f"({size_gb:.2f} GB in {TILE_CACHE_DIR}).")
            print("       That is normal for slides still in progress; a finished "
                  "slide's cache is deleted automatically.")

    # -- Summary ---------------------------------------------------------------
    report = {
        "slide_embed_dir": SLIDE_EMBED_DIR,
        "files": len(files),
        "valid": len(good),
        "invalid": len(bad),
        "expected": len(expected),
        "missing": missing,
        "no_tissue": no_tissue,
        "tile_embeddings": {
            "enabled": SAVE_TILE_EMBEDDINGS,
            "dir": TILE_EMBED_DIR,
            "format": TILE_EMBED_FORMAT,
            "outstanding": tile_problems,
        },
    }
    report_path = os.path.join(OUTPUT_ROOT, "prism_verification.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("\n" + "=" * 74)
    ok = not bad and not missing and not tile_problems
    if ok:
        print("RESULT: PASS - every expected slide has a valid PRISM embedding"
              + (" and tile embeddings." if SAVE_TILE_EMBEDDINGS else "."))
    else:
        print(f"RESULT: INCOMPLETE - {len(bad)} invalid, {len(missing)} missing"
              + (f", {len(tile_problems)} without tile embeddings."
                 if SAVE_TILE_EMBEDDINGS else "."))
    print(f"Report written to {report_path}")
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
