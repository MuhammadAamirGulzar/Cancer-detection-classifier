"""
Verify the SurGen label lookup against surgen_slide_labels.csv.

Runs get_slide_label() over every row of the CSV and asserts the resolved
counts. Exits non-zero (and prints why) if they do not match the expected
991 labelled / 100 msih / 891 nonmsih.

    python verify_surgen_labels.py
"""
import sys
from collections import Counter

import pandas as pd

from combined_pipeline_final_error_checks_v6 import (
    SURGEN_LABELS_CSV,
    get_slide_label,
)

EXPECT_TOTAL   = 991
EXPECT_MSIH    = 100
EXPECT_NONMSIH = 891


def main() -> int:
    df = pd.read_csv(SURGEN_LABELS_CSV)
    print(f"Label CSV: {SURGEN_LABELS_CSV}")
    print(f"  rows: {len(df)}")

    counts = Counter()
    resolved_stems = []
    for fname in df["slide_filename"].astype(str):
        lab = get_slide_label(fname)
        counts[lab] += 1
        if lab is not None:
            resolved_stems.append(fname)

    n_labelled = counts["msih"] + counts["nonmsih"]
    print()
    print("get_slide_label() over every row:")
    print(f"  msih     : {counts['msih']}")
    print(f"  nonmsih  : {counts['nonmsih']}")
    print(f"  None     : {counts[None]}")
    print(f"  labelled : {n_labelled}")
    print(f"  unique labelled stems: {len(set(resolved_stems))}")

    ok = (
        n_labelled     == EXPECT_TOTAL
        and counts["msih"]    == EXPECT_MSIH
        and counts["nonmsih"] == EXPECT_NONMSIH
        and len(set(resolved_stems)) == EXPECT_TOTAL
    )

    print()
    if ok:
        print(f"OK — {EXPECT_TOTAL} slides resolve to a label "
              f"({EXPECT_MSIH} msih / {EXPECT_NONMSIH} nonmsih).")
        return 0

    print("MISMATCH — expected "
          f"{EXPECT_TOTAL} labelled ({EXPECT_MSIH} msih / {EXPECT_NONMSIH} nonmsih), "
          f"got {n_labelled} ({counts['msih']} msih / {counts['nonmsih']} nonmsih).")
    print("STOP: the label set does not match. Not proceeding to A-1 / A-2 / C.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
