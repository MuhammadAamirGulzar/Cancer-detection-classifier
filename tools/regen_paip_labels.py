"""Regenerate the PAIP label tables cleanly (work order Task 2.3).

Two problems being fixed.

1. ``slide_classification/paip_kfolds_71.csv`` is misleadingly named: it holds
   **78 rows** and **no fold column** at all. It is a label table, not a fold
   table, and its name has caused it to be read as one. Renamed to
   ``paip_78_labels.csv``.

2. ``paip_data/labels/paip_78slides_labels.csv`` is **malformed**. Its ``fold``
   column mixes fold indices with label strings:
   ``{'nonMSIH': 24, '1': 10, '0': 10, '2': 9, '3': 9, '4': 9, 'MSIH': 7}``.
   The 31 ``validation_data_*`` rows carry only three fields, so their label
   landed in ``fold`` and ``label`` is NaN. Regenerated from the two authoritative
   provider files with an explicit ``split`` column (``train``/``test``) instead
   of an overloaded ``fold`` - PAIP has no folds (PAIP-CV is retired, Task 2.2);
   it has a provider split.

Sources of truth:
  paip_data/labels/paip_47slides.csv        - the provider's 47 training ids
  paip_data/labels/paip_31slides_labels.csv - the provider's 31 test ids + labels
  slide_classification/paip_78_labels.csv   - the 78-row label table

Run:  python tools/regen_paip_labels.py [--write]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "slide_classification"))

from config import paths as P  # noqa: E402

OUT = P.ROOTS["paip_labels"] / "paip_78slides_labels.csv"


def label_from_id(wsi_id: str) -> str:
    """PAIP ids encode the label in their suffix: ``..._MSIH`` / ``..._nonMSIH``."""
    return "nonMSIH" if wsi_id.endswith("_nonMSIH") else "MSIH"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write the file (default: dry run)")
    args = ap.parse_args()

    labels_78 = pd.read_csv(P.labels_path("paip"))
    lut = dict(zip(labels_78["WSI_Id"].astype(str),
                   labels_78["label" if "label" in labels_78.columns else "label_desc"].astype(str)))

    train_ids = pd.read_csv(P.paip_split_path("train"))["Fold1"].astype(str).tolist()
    test_df = pd.read_csv(P.paip_split_path("test"))
    test_ids = test_df["WSI_Id"].astype(str).tolist()

    print(f"provider train ids: {len(train_ids)}   provider test ids: {len(test_ids)}")

    rows = []
    for wsi in train_ids:
        rows.append({"WSI_Id": wsi, "split": "train", "label": lut.get(wsi, label_from_id(wsi))})
    test_lut = dict(zip(test_df["WSI_Id"].astype(str), test_df["label"].astype(str)))
    for wsi in test_ids:
        rows.append({"WSI_Id": wsi, "split": "test", "label": test_lut[wsi]})

    out = pd.DataFrame(rows)
    out["label_id"] = (out["label"] != "nonMSIH").astype(int)
    out = out[["WSI_Id", "split", "label", "label_id"]]

    # ---- consistency checks -------------------------------------------------
    print("\nchecks:")
    ok = True

    n_ok = len(out) == 78
    print(f"  78 rows                          : {len(out)}  {'OK' if n_ok else 'FAIL'}")
    ok &= n_ok

    counts = out["split"].value_counts().to_dict()
    s_ok = counts.get("train") == 47 and counts.get("test") == 31
    print(f"  split is 47 train / 31 test      : {counts}  {'OK' if s_ok else 'FAIL'}")
    ok &= s_ok

    lc = out["label"].value_counts().to_dict()
    l_ok = lc.get("MSIH") == 19 and lc.get("nonMSIH") == 59
    print(f"  19 MSI-H / 59 non-MSI-H          : {lc}  {'OK' if l_ok else 'FAIL'}")
    ok &= l_ok

    # The id suffix and the label table must agree - if they ever disagree, one
    # of the two is wrong and the discrepancy must be resolved by hand.
    mismatched = [w for w in out["WSI_Id"] if w in lut and lut[w] != label_from_id(w)]
    m_ok = not mismatched
    print(f"  id suffix agrees with label table: "
          f"{'OK' if m_ok else f'FAIL {mismatched[:5]}'}")
    ok &= m_ok

    dup = out["WSI_Id"].duplicated().sum()
    d_ok = dup == 0
    print(f"  no duplicate ids                 : {dup}  {'OK' if d_ok else 'FAIL'}")
    ok &= d_ok

    missing = set(lut) - set(out["WSI_Id"])
    e_ok = not missing
    print(f"  covers every id in label table   : "
          f"{'OK' if e_ok else f'FAIL missing {sorted(missing)}'}")
    ok &= e_ok

    print(f"\nper-split class balance:")
    print(out.groupby(["split", "label"]).size().to_string())

    if not ok:
        print("\nFAILED consistency checks - refusing to write.")
        sys.exit(1)

    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUT, index=False)
        print(f"\nwrote {OUT}")
    else:
        print(f"\nDRY RUN - pass --write to overwrite {OUT}")
        print(out.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
