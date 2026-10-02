"""Where SurGen's 1020 slides become the 622 we actually model.

Answers one question with evidence rather than recollection: given that the
SurGen release contains far more WSIs than we train on, what removes each one,
and is any of it accidental?

The funnel is rebuilt from primary sources every run - nothing here trusts a
number written in a report:

  surgen_labels.csv                one row per CZI found in the release
  patch_metadata_nonwhite_index.npz   the tissue index built by Steps A-1/A-2
  <features>/PRISM/prism/*.pt      slide embeddings produced so far

Each stage prints what it removed and *why*, and every slide lost between the
labelled set and the modelled set is named individually - a silent drop there
would be a bug, not a design choice, so it must never be summarised away.

The audit also tests candidate denominators (1020, 991, 624, 622, ...) against
the data, because quoted cohort sizes in this field usually differ by which
question is being asked, not by which files exist.

Nothing is written and nothing is mutated - measurement only.

Run:  python tools/surgen_cohort_audit.py [--data-root PATH] [--csv OUT.csv]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ── Defaults: the machine this project runs on ────────────────────────────────
DEFAULT_DATA_ROOT = Path(r"D:\Aamir Gulzar\KSA_project2\surgen_data")

LABELS_CSV   = "surgen_labels.csv"
NONWHITE_NPZ = "patch_metadata_nonwhite_index.npz"
PATCH_COUNTS = "patch_counts_per_slide.csv"
EMBED_SUBDIR = Path("surgen_processed") / "prism" / "features" / "slide_aggregation" / "PRISM" / "prism"

# Slide stems look like SR1482_40X_HE_T284_01: cohort, magnification, stain,
# T<case>, then the section number. Two sections of one tumour share a case.
CASE_RE = re.compile(r"^(SR\d+)_.*_T(\d{3})_(\d+)$")

LABEL_MEANING = {
    1: "MSI-H / MMR-deficient  (positive)",
    0: "MSS / MMR-proficient   (negative)",
    -1: "unusable MSI status    (excluded)",
}

# Labelled slides already confirmed to carry no tissue at all, so they can hold
# no features and cannot be modelled. Both were downloaded and run through Step
# A, which reported "confirmed zero tissue patches, nothing to process" - they
# are a property of the slides, not a processing failure.
#
# They stay listed here rather than being silently filtered so that any slide
# lost at this stage which is NOT on this list stands out as something to
# investigate.
KNOWN_ZERO_TISSUE = {
    "SR386_40X_HE_T086_01",
    "SR386_40X_HE_T339_01",
}


def hr(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def parse_stem(stem: str):
    """(cohort, case_key, section) for a slide stem, or (None, None, None)."""
    m = CASE_RE.match(stem)
    if not m:
        return None, None, None
    cohort, case, section = m.group(1), m.group(2), m.group(3)
    return cohort, f"{cohort}_{case}", section


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                    help="directory holding surgen_labels.csv and the tissue index")
    ap.add_argument("--csv", type=Path, default=None,
                    help="optional path to write the per-slide audit table")
    args = ap.parse_args()

    root = args.data_root
    labels_path = root / LABELS_CSV
    npz_path    = root / NONWHITE_NPZ
    counts_path = root / PATCH_COUNTS
    embed_dir   = root / EMBED_SUBDIR

    if not labels_path.is_file():
        print(f"[FATAL] No labels CSV at {labels_path}", file=sys.stderr)
        return 1

    # ── Stage 0: every CZI in the release ────────────────────────────────────
    df = pd.read_csv(labels_path)
    df["WSI_Id"] = df["WSI_Id"].astype(str)
    df["label_desc"] = df["label_desc"].astype(int)

    parsed = df["WSI_Id"].map(parse_stem)
    df["cohort"]   = [p[0] for p in parsed]
    df["case_key"] = [p[1] for p in parsed]
    df["section"]  = [p[2] for p in parsed]

    unparsed = df[df["cohort"].isna()]

    hr("STAGE 0 - every WSI in the SurGen release")
    print(f"  source: {labels_path}")
    print(f"  rows (one per CZI found): {len(df)}")
    if len(unparsed):
        print(f"  [WARN] {len(unparsed)} stem(s) did not match the naming pattern:")
        for s in unparsed["WSI_Id"].head(10):
            print(f"           {s}")
    print()
    print(f"  {'cohort':<10} {'slides':>7} {'cases':>7}")
    for coh, g in df.groupby("cohort", dropna=True):
        print(f"  {coh:<10} {len(g):>7} {g['case_key'].nunique():>7}")
    print(f"  {'TOTAL':<10} {len(df):>7} {df['case_key'].nunique():>7}")

    # ── Stage 1: the label is the gate ───────────────────────────────────────
    hr("STAGE 1 - does this slide have a usable MSI/MMR status?")
    dist = Counter(df["label_desc"])
    for val in (1, 0, -1):
        n = dist.get(val, 0)
        print(f"  label_desc = {val:>2}  {n:>5} slides   {LABEL_MEANING[val]}")

    labelled = df[df["label_desc"] != -1].copy()
    dropped_label = df[df["label_desc"] == -1]
    print(f"\n  labelled (label_desc != -1): {len(labelled)} slides, "
          f"{labelled['case_key'].nunique()} cases")
    print(f"  removed here               : {len(dropped_label)} slides")
    print("\n  by cohort:")
    print(f"  {'cohort':<10} {'total':>7} {'labelled':>9} {'dropped':>8}  {'% kept':>7}")
    for coh in sorted(df["cohort"].dropna().unique()):
        tot = int((df["cohort"] == coh).sum())
        lab = int((labelled["cohort"] == coh).sum())
        print(f"  {coh:<10} {tot:>7} {lab:>9} {tot - lab:>8}  {100 * lab / tot:>6.1f}%")

    print("\n  Why slides are dropped here: MSI/MMR status is the prediction")
    print("  target. SR386 carries mmr_loss_binary; SR1482 carries an MSI call")
    print("  where only 'MSI High' and 'No MSI' are usable - 'MSI Low',")
    print("  'Not Performed', 'Failed' and 'Insufficient' cannot supply a label.")
    print("  This is an exclusion of unlabelled data, not of imaging data.")

    # ── Stage 2: does the slide have tissue / extracted features? ────────────
    hr("STAGE 2 - did the slide survive tissue detection (Steps A-1/A-2)?")
    if not npz_path.is_file():
        print(f"  [FATAL] tissue index missing: {npz_path}", file=sys.stderr)
        return 1

    z = np.load(npz_path, allow_pickle=True)
    nonwhite = set(str(s) for s in z["slides"])
    offsets = z["offsets"]
    per_slide_patches = {str(s): int(offsets[i + 1] - offsets[i])
                         for i, s in enumerate(z["slides"])}

    print(f"  source: {npz_path}")
    print(f"  slides carrying >=1 non-white patch: {len(nonwhite)}")
    print(f"  total non-white patches            : {int(offsets[-1]):,}")

    labelled_stems = set(labelled["WSI_Id"])
    modelled = labelled_stems & nonwhite
    lost     = labelled_stems - nonwhite
    extra    = nonwhite - labelled_stems

    print(f"\n  labelled slides                    : {len(labelled_stems)}")
    print(f"  labelled AND in the tissue index    : {len(modelled)}   <-- modelled")
    print(f"  labelled but NOT in the index       : {len(lost)}")
    print(f"  in the index but NOT labelled       : {len(extra)}")

    # Every slide lost at this stage is named. A silent drop here is a bug.
    if lost:
        unexplained = sorted(lost - KNOWN_ZERO_TISSUE)
        print("\n  The slides lost between 'labelled' and 'modelled':")
        for s in sorted(lost):
            row = labelled[labelled["WSI_Id"] == s].iloc[0]
            tag = "confirmed zero tissue" if s in KNOWN_ZERO_TISSUE else "*** UNEXPLAINED ***"
            print(f"    {s}   label={row['label_desc']}   cohort={row['cohort']}   {tag}")
        if unexplained:
            print(f"\n  [ACTION] {len(unexplained)} slide(s) above are not on the known")
            print("  zero-tissue list. A labelled slide with no tissue index entry is")
            print("  either genuinely blank or a Step A failure - check the pipeline")
            print("  logs for 'zero tissue patches' before accepting the loss.")
        else:
            print("\n  All accounted for: each was run through Step A and reported")
            print("  'confirmed zero tissue patches, nothing to process'. No features")
            print("  can exist for a slide with no tissue, so this is a property of")
            print("  the slides rather than a pipeline failure.")
    if extra:
        print("\n  Indexed but unlabelled (tissue found, no MSI status) - "
              f"{len(extra)} slide(s); these cost extraction time but cannot train.")

    if counts_path.is_file():
        cnt = pd.read_csv(counts_path, index_col=0)
        print(f"\n  cross-check {counts_path.name}: {len(cnt)} rows "
              f"({'matches' if len(cnt) == len(nonwhite) else 'DOES NOT MATCH'} the index)")

    # ── Stage 3: extraction progress ─────────────────────────────────────────
    hr("STAGE 3 - how many modelled slides have PRISM embeddings today?")
    if embed_dir.is_dir():
        have = {p.stem for p in embed_dir.glob("*.pt")}
        done = modelled & have
        todo = modelled - have
        print(f"  embedding dir: {embed_dir}")
        print(f"  embeddings present     : {len(have)}")
        print(f"  of the {len(modelled)} modelled : {len(done)} done, {len(todo)} remaining"
              f"   ({100 * len(done) / max(1, len(modelled)):.1f}%)")
        if have - modelled:
            print(f"  [WARN] {len(have - modelled)} embedding(s) outside the modelled set")
    else:
        print(f"  (no embedding directory at {embed_dir} - skipping)")

    # ── Stage 4: which denominator is which ──────────────────────────────────
    hr("STAGE 4 - candidate cohort sizes, matched against the data")
    n_slides   = len(df)
    n_cases    = df["case_key"].nunique()
    n_labelled = len(labelled)
    n_modelled = len(modelled)
    n_lab_case = labelled["case_key"].nunique()
    mod_df     = labelled[labelled["WSI_Id"].isin(modelled)]
    n_mod_case = mod_df["case_key"].nunique()
    n_sec1     = int((df["section"] == "01").sum())

    candidates = {
        "all WSIs in the release":                 n_slides,
        "all cases (patients)":                    n_cases,
        "first section per slide (_01 only)":      n_sec1,
        "slides with a usable MSI/MMR label":      n_labelled,
        "cases with a usable label":               n_lab_case,
        "slides modelled (labelled + features)":   n_modelled,
        "cases modelled":                          n_mod_case,
        "slides with tissue, any label":           len(nonwhite),
    }
    width = max(len(k) for k in candidates)
    for name, val in candidates.items():
        print(f"  {name:<{width}}  {val:>6}")

    print("\n  Sanity: a quoted figure only means something with its question.")
    for probe in (991, 1020, 843, 624, 622):
        hits = [k for k, v in candidates.items() if v == probe]
        verdict = "; ".join(hits) if hits else "NO quantity in this dataset equals it"
        print(f"    {probe:>5}  ->  {verdict}")

    # ── The funnel, one line per stage ───────────────────────────────────────
    hr("THE FUNNEL")
    print(f"  {n_slides:>5}  WSIs in the SurGen release ({n_cases} cases)")
    print(f"  {-(n_slides - n_labelled):>5}  no usable MSI/MMR status - cannot supply a label")
    print(f"  {n_labelled:>5}  labelled slides ({n_lab_case} cases)")
    print(f"  {-(n_labelled - n_modelled):>5}  labelled but absent from the tissue index")
    print(f"  {n_modelled:>5}  MODELLED ({n_mod_case} cases)")

    if args.csv:
        out = df.copy()
        out["labelled"]   = out["label_desc"] != -1
        out["has_tissue"] = out["WSI_Id"].isin(nonwhite)
        out["n_patches"]  = out["WSI_Id"].map(per_slide_patches).fillna(0).astype(int)
        out["modelled"]   = out["labelled"] & out["has_tissue"]
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.csv, index=False)
        print(f"\n  per-slide table written: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
