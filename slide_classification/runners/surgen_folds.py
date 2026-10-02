"""SurGen fold construction at **case** level (work order Task 2.1).

The bug being fixed
-------------------
``build_runtime_folds`` in ``windows_slide_classification_surgen.py`` (and the
linux/older variants) built folds by shuffling **slide IDs** and dealing them
round-robin. But 70 of the 554 labelled SurGen cases contribute **two slides
each**, so sibling slides landed in different folds roughly 3 times in 4 - the
same patient in train and test. Two sections from one tumour are far more alike
than two different tumours, so this inflated every SurGen number.

TCGA is immune: it has exactly one slide per patient (416 rows, 416 unique
``Case_ID``), so the same code was safe there and was carried over unchanged.

Verified against the real IDs
-----------------------------
  624 usable slides (label 0/1; the 396 label -1 slides are excluded)
  554 unique cases        <- matches the work order's expected count exactly
  484 cases with 1 slide, 70 cases with 2 slides
  0 cases with conflicting labels across their slides, so the case label is
    unambiguous
  case-level labels: 505 negative / 49 positive
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from config import paths as P

#: Slide ids look like ``SR1482_40X_HE_T004_01``; the trailing ``_NN`` is the
#: section number within a case. Anchored, and only strips a *numeric* suffix.
_CASE_RE = re.compile(r"^(.*)_\d+$")

DEFAULT_SEED = 42
DEFAULT_NUM_FOLDS = 4


def case_id_of(slide_id: str) -> str:
    """``SR1482_40X_HE_T004_01`` -> ``SR1482_40X_HE_T004``.

    Returns the id unchanged if it carries no numeric suffix, so an unexpected
    naming scheme degrades to one-case-per-slide rather than silently collapsing
    unrelated slides together.
    """
    m = _CASE_RE.match(slide_id)
    return m.group(1) if m else slide_id


def load_surgen_labels(labels_csv: Optional[str] = None) -> pd.DataFrame:
    """Usable SurGen slides (label 0/1) with their case id attached."""
    path = labels_csv or P.labels_path("surgen")
    df = pd.read_csv(path)
    df["label_desc"] = pd.to_numeric(df["label_desc"], errors="coerce")
    usable = df[df["label_desc"].isin([0, 1])].copy()
    usable["label"] = usable["label_desc"].astype(int)
    usable["case_id"] = usable["WSI_Id"].map(case_id_of)
    return usable[["WSI_Id", "case_id", "label"]].reset_index(drop=True)


def case_label_map(labels: pd.DataFrame) -> Dict[str, int]:
    """``{case_id: label}``, raising if any case has conflicting slide labels."""
    by_case: Dict[str, set] = defaultdict(set)
    for case, lab in zip(labels["case_id"], labels["label"]):
        by_case[case].add(int(lab))
    conflicts = {c: sorted(v) for c, v in by_case.items() if len(v) > 1}
    if conflicts:
        raise ValueError(
            f"{len(conflicts)} SurGen case(s) have conflicting labels across their "
            f"slides, so a case-level label is not well defined: "
            f"{dict(list(conflicts.items())[:5])}"
        )
    return {c: next(iter(v)) for c, v in by_case.items()}


def build_case_level_folds(
    labels_csv: Optional[str] = None,
    num_folds: int = DEFAULT_NUM_FOLDS,
    seed: int = DEFAULT_SEED,
    restrict_to: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> Dict[str, int]:
    """``{slide_id: fold}`` with **whole cases** kept together.

    Cases are grouped by their (unambiguous) case-level label, shuffled with
    ``seed``, and dealt round-robin across folds - the same scheme the old code
    used, lifted from slide level to case level. Folds are numbered 1..K to match
    the TCGA fold CSVs.

    ``restrict_to`` limits the result to slides that actually have features, so
    fold sizes reported by a runner match what it will really train on.
    """
    labels = load_surgen_labels(labels_csv)
    if restrict_to is not None:
        keep = set(restrict_to)
        labels = labels[labels["WSI_Id"].isin(keep)].reset_index(drop=True)

    case_labels = case_label_map(labels)
    slides_by_case: Dict[str, List[str]] = defaultdict(list)
    for wsi, case in zip(labels["WSI_Id"], labels["case_id"]):
        slides_by_case[case].append(wsi)

    by_label: Dict[int, List[str]] = defaultdict(list)
    for case, lab in case_labels.items():
        by_label[lab].append(case)

    rng = random.Random(seed)
    fold_of_case: Dict[str, int] = {}
    for lab in sorted(by_label):
        cases = sorted(by_label[lab])       # sort first so the shuffle is reproducible
        rng.shuffle(cases)
        for i, case in enumerate(cases):
            fold_of_case[case] = (i % num_folds) + 1

    fold_map = {
        wsi: fold_of_case[case]
        for case, wsis in slides_by_case.items()
        for wsi in wsis
    }

    # Assert no case spans folds. A leakage failure must halt loudly (1c.3 #5).
    spans = {
        case: sorted({fold_map[w] for w in wsis})
        for case, wsis in slides_by_case.items()
        if len({fold_map[w] for w in wsis}) > 1
    }
    if spans:
        raise AssertionError(
            f"[LEAKAGE] {len(spans)} case(s) appear in more than one fold: "
            f"{dict(list(spans.items())[:5])}"
        )

    if verbose:
        _print_fold_table(fold_map, slides_by_case, case_labels, labels, num_folds)

    return fold_map


def _print_fold_table(fold_map, slides_by_case, case_labels, labels, num_folds):
    """Per-fold slide count, case count and MSI-H count (Task 2.1 step 4)."""
    slide_label = dict(zip(labels["WSI_Id"], labels["label"]))
    fold_of_case = {c: fold_map[w[0]] for c, w in slides_by_case.items()}

    print(f"  SurGen case-level folds (seed={DEFAULT_SEED}, K={num_folds})")
    print(f"  {'fold':<6}{'slides':>8}{'cases':>8}{'MSI-H slides':>14}{'MSI-H cases':>13}")
    tot_s = tot_c = tot_ms = tot_mc = 0
    for f in range(1, num_folds + 1):
        slides = [w for w, ff in fold_map.items() if ff == f]
        cases = [c for c, ff in fold_of_case.items() if ff == f]
        ms = sum(slide_label[w] for w in slides)
        mc = sum(case_labels[c] for c in cases)
        tot_s += len(slides); tot_c += len(cases); tot_ms += ms; tot_mc += mc
        print(f"  {f:<6}{len(slides):>8}{len(cases):>8}{ms:>14}{mc:>13}")
    print(f"  {'TOTAL':<6}{tot_s:>8}{tot_c:>8}{tot_ms:>14}{tot_mc:>13}")
    multi = sum(1 for w in slides_by_case.values() if len(w) > 1)
    print(f"  {multi} case(s) contribute >1 slide; all kept within a single fold.")


# --------------------------------------------------------------------------
# Legacy (buggy) fold construction - retained ONLY to quantify the correction
# --------------------------------------------------------------------------

def build_slide_level_folds_legacy(
    labels_csv: Optional[str] = None,
    num_folds: int = DEFAULT_NUM_FOLDS,
    seed: int = DEFAULT_SEED,
    restrict_to: Optional[Sequence[str]] = None,
) -> Dict[str, int]:
    """Reproduces the pre-correction ``build_runtime_folds`` exactly.

    Shuffles **slide** ids per label with ``random.Random(seed)`` and deals them
    round-robin. Used only by ``tools/surgen_fold_audit.py`` to measure how much
    leakage the old scheme produced and how far the numbers move once it is
    fixed. **Never use this to produce a reported result.**
    """
    labels = load_surgen_labels(labels_csv)
    if restrict_to is not None:
        labels = labels[labels["WSI_Id"].isin(set(restrict_to))].reset_index(drop=True)

    by_label: Dict[int, List[str]] = defaultdict(list)
    for wsi, lab in zip(labels["WSI_Id"], labels["label"]):
        by_label[int(lab)].append(wsi)

    rng = random.Random(seed)
    fold_map: Dict[str, int] = {}
    for lab in sorted(by_label):
        ids = list(by_label[lab])
        rng.shuffle(ids)
        for i, sid in enumerate(ids):
            fold_map[sid] = (i % num_folds) + 1
    return fold_map


def quantify_leakage(fold_map: Dict[str, int]) -> dict:
    """How many multi-slide cases are split across folds by ``fold_map``."""
    by_case: Dict[str, List[str]] = defaultdict(list)
    for wsi in fold_map:
        by_case[case_id_of(wsi)].append(wsi)

    multi = {c: w for c, w in by_case.items() if len(w) > 1}
    split = {c: sorted({fold_map[w] for w in ws}) for c, ws in multi.items()
             if len({fold_map[w] for w in ws}) > 1}
    leaked_slides = sum(len(multi[c]) for c in split)
    return {
        "n_cases": len(by_case),
        "n_multi_slide_cases": len(multi),
        "n_split_across_folds": len(split),
        "pct_split": (100.0 * len(split) / len(multi)) if multi else 0.0,
        "leaked_slides": leaked_slides,
        "examples": dict(list(split.items())[:5]),
    }


if __name__ == "__main__":  # pragma: no cover
    print("=" * 74)
    print("SurGen fold construction - case level (Task 2.1)")
    print("=" * 74)
    new = build_case_level_folds()
    print()
    old = build_slide_level_folds_legacy()
    print("Leakage under the OLD slide-level scheme:")
    for k, v in quantify_leakage(old).items():
        print(f"  {k}: {v}")
    print("\nLeakage under the NEW case-level scheme:")
    for k, v in quantify_leakage(new).items():
        print(f"  {k}: {v}")
