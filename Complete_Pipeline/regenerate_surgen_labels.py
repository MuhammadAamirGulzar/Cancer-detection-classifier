"""
Regenerate surgen_labels.csv (WSI_Id, label_desc) from surgen_slide_labels.csv.

surgen_slide_labels.csv is the single source of truth for SurGen. The older
two-column surgen_labels.csv (WSI_Id / label_desc) is still consumed by the
slide-classification data layer and by verify_prism_features.py, so it must be
kept in lock-step with the master file.

Mapping, per row of surgen_slide_labels.csv (order preserved):

    include == TRUE , label == 1  ->  label_desc =  1
    include == TRUE , label == 0  ->  label_desc =  0
    include != TRUE               ->  label_desc = -1

No MMR/MSI rule is re-derived here; the resolved `label` in the master file is
authoritative.

    python regenerate_surgen_labels.py            # rewrite both copies
    python regenerate_surgen_labels.py --check     # verify only, exit 1 on drift
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
MASTER_CSV = _HERE / "surgen_slide_labels.csv"

# Every copy of the two-column file that must track the master.
TARGETS = [
    Path(r"D:\Aamir Gulzar\KSA_project2\surgen_data\surgen_labels.csv"),
    _HERE.parent / "slide_classification" / "surgen_labels.csv",
]

EXPECT = {1: 100, 0: 891, -1: 29}


def build() -> pd.DataFrame:
    df = pd.read_csv(MASTER_CSV)
    stems = df["slide_filename"].astype(str).map(lambda s: Path(s).stem)
    included = df["include"].astype(str).str.strip().str.upper() == "TRUE"
    label_desc = df["label"].where(included, other=-1).fillna(-1).astype(int)

    out = pd.DataFrame({"WSI_Id": stems.values, "label_desc": label_desc.values})
    if out["WSI_Id"].duplicated().any():
        dups = sorted(out.loc[out["WSI_Id"].duplicated(keep=False), "WSI_Id"].unique())
        raise SystemExit(f"Duplicate slide stems in {MASTER_CSV}: {dups}")

    counts = Counter(out["label_desc"])
    if dict(counts) != EXPECT:
        raise SystemExit(
            f"Resolved label_desc counts {dict(sorted(counts.items()))} "
            f"do not match expected {EXPECT}. Master file changed — update EXPECT "
            f"deliberately if this is intended."
        )
    return out


def main(argv: list[str]) -> int:
    check_only = "--check" in argv
    out = build()
    print(f"master : {MASTER_CSV}  ({len(out)} rows)")
    print(f"counts : {dict(sorted(Counter(out['label_desc']).items()))}")

    drift = False
    for target in TARGETS:
        if target.exists():
            cur = pd.read_csv(target)
            same = (
                list(cur.columns) == ["WSI_Id", "label_desc"]
                and cur.reset_index(drop=True).astype({"label_desc": int}).equals(
                    out.reset_index(drop=True)
                )
            )
        else:
            same = False
        status = "up-to-date" if same else "STALE"
        print(f"  {status:11s} {target}")
        if not same:
            drift = True
            if not check_only:
                target.parent.mkdir(parents=True, exist_ok=True)
                out.to_csv(target, index=False)
                print(f"              -> rewritten ({len(out)} rows)")

    if check_only and drift:
        print("\nDRIFT: at least one copy is stale. Run without --check to rewrite.")
        return 1
    if not check_only:
        print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
