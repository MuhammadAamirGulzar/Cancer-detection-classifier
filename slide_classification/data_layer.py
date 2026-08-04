"""Single data layer for every slide-classification experiment.

Replaces the ~6 duplicated ``WSIDataset`` implementations that were spread
across the notebooks and SurGen scripts. Work order Task 1.0.

What changed and why
--------------------

**Load once, index many** (1c.1 item 1). The old code rebuilt three
``WSIDataset`` objects per fold, each re-running ``torch.load`` over the whole
feature directory - so every ``.pt`` file was read from disk roughly 20 times
per ``(method, model)`` combination (5 classifiers x 4 folds). Here the cohort
is assembled **once** into a single ``(N, D)`` float32 tensor and every split is
an index slice. The matrices are small: TCGA Virchow2 caption-15 is
413 x 38400 x 4B = 63 MB.

**O(1) label lookup** (1c.1 item 2). The old loader ran
``if wsi_id in folds_df['WSI_Id'].values`` followed by a ``.loc`` mask - two
full linear scans of the DataFrame per file. Labels are now resolved through a
dict built once.

**No DataLoader** (1c.1 item 3). Batching at ``batch_size=4`` and then
``torch.cat``-ing the loader straight back into one tensor accomplished nothing
but overhead. Splits are handed to the classifiers as plain tensors.

**Deterministic row order.** Rows are sorted by WSI id. The old loader used
``os.listdir`` order for test/val and a *reseeded-per-run* ``shuffle=True``
DataLoader for train, which made Random Forest (bootstrap sampling depends on
row order) non-reproducible run to run. Sorted order fixes that.

**Loud about missing data** (Task 1.4). A labelled slide with no feature file
used to vanish silently - that is why TCGA reported N=413 against 416 labels and
PAIP 73 against 78. Both difference sets are now computed, warned about, and
written to ``missing_slides.csv``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from config import paths as P

# --------------------------------------------------------------------------
# Label resolution
# --------------------------------------------------------------------------

#: Per-task label column and the value that maps to class 0.
_BINARY_LABEL_SPEC: Dict[str, Tuple[str, str]] = {
    "MSIH": ("label_desc", "nonMSIH"),
    "BRAF": ("BRAF_mutation", "WT"),
    "KRAS": ("KRAS_mutation", "WT"),
    "TP53": ("TP53_mutation", "WT"),
}


def _label_map_tcga(labels_df: pd.DataFrame, task: str) -> Dict[str, int]:
    """``{wsi_id: 0|1}`` for a TCGA task.

    Task 1.4: the old code left ``label`` **undefined** when a slide was absent
    from the fold CSV, raising a ``NameError`` that a bare ``except`` swallowed.
    Here an unresolvable label is simply absent from the map, and the caller
    reports it.
    """
    if task == "CIMP":
        # HypermethylationCategory has four levels, so CIMP needs dichotomising.
        # The data decides it rather than convention: cross-tabulating the four
        # levels against MSI status across the 416 TCGA slides gives
        #
        #     CIMP-H       n= 54   68.5% MSI-H     <- a distinct biological group
        #     CRC CIMP-L   n=178    5.6% MSI-H
        #     Non-CIMP     n=182    6.6% MSI-H
        #     GEA CIMP-L   n=  2  100%   MSI-H
        #
        # CIMP-L is statistically indistinguishable from Non-CIMP (5.6% vs 6.6%),
        # so folding it into the positive class would bury 178 Non-CIMP-like
        # slides in with 54 genuinely distinct ones. CIMP-H vs rest is also the
        # standard framing in the colorectal literature - CIMP-H is the
        # recognised entity, arising via MLH1 promoter hypermethylation, which is
        # exactly why it is MSI-H enriched. Its 13% prevalence closely matches
        # MSI-H's 15%, so the pipeline's class weights and threshold policy carry
        # over without retuning.
        #
        # Override by passing positive_label if a different definition is wanted.
        column = "HypermethylationCategory"
        if column not in labels_df.columns:
            raise KeyError(
                f"CIMP label column {column!r} not found. "
                f"Available: {list(labels_df.columns)}")
        ids = labels_df["WSI_Id"].astype(str)
        values = labels_df[column].astype(str).str.strip()
        return {wsi: (1 if val == "CIMP-H" else 0) for wsi, val in zip(ids, values)}
    if task not in _BINARY_LABEL_SPEC:
        raise ValueError(f"Unknown TCGA task {task!r}; expected one of {sorted(_BINARY_LABEL_SPEC)}")

    column, negative_value = _BINARY_LABEL_SPEC[task]
    if column not in labels_df.columns:
        raise KeyError(
            f"Label column {column!r} for task {task!r} not found. "
            f"Available columns: {list(labels_df.columns)}"
        )
    ids = labels_df["WSI_Id"].astype(str)
    values = labels_df[column].astype(str)
    return {wsi: (0 if val == negative_value else 1) for wsi, val in zip(ids, values)}


def _label_map_paip(labels_df: pd.DataFrame) -> Dict[str, int]:
    """``{wsi_id: 0|1}`` for PAIP (MSI only)."""
    column = "label" if "label" in labels_df.columns else "label_desc"
    ids = labels_df["WSI_Id"].astype(str)
    values = labels_df[column].astype(str)
    return {wsi: (0 if val == "nonMSIH" else 1) for wsi, val in zip(ids, values)}


def _label_map_surgen(labels_df: pd.DataFrame) -> Dict[str, int]:
    """``{wsi_id: 0|1}`` for SurGen; label ``-1`` means unknown and is dropped."""
    ids = labels_df["WSI_Id"].astype(str)
    values = pd.to_numeric(labels_df["label_desc"], errors="coerce")
    return {
        wsi: int(val)
        for wsi, val in zip(ids, values)
        if pd.notna(val) and int(val) in (0, 1)
    }


def load_label_map(cohort: str, task: str = "MSIH") -> Dict[str, int]:
    """``{wsi_id: label}`` for a cohort, built once (1c.1 item 2)."""
    cohort = cohort.lower()
    csv_path = P.labels_path(cohort, task)
    if not csv_path.exists():
        raise FileNotFoundError(f"Label file for cohort={cohort} task={task} not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if cohort == "tcga":
        return _label_map_tcga(df, task)
    if cohort == "paip":
        return _label_map_paip(df)
    if cohort == "surgen":
        return _label_map_surgen(df)
    raise ValueError(f"Unknown cohort {cohort!r}")


def load_fold_map(cohort: str, task: str = "MSIH") -> Optional[Dict[str, int]]:
    """``{wsi_id: fold}`` where the cohort ships provider-independent folds.

    Only TCGA does. SurGen folds are rebuilt at case level (Task 2.1) and PAIP
    uses the provider's own split (Task 3.1).
    """
    folds_csv = P.folds_path(cohort, task)
    if folds_csv is None or not folds_csv.exists():
        return None
    df = pd.read_csv(folds_csv)
    if "fold" not in df.columns:
        return None
    return {str(w): int(f) for w, f in zip(df["WSI_Id"], df["fold"])}


# --------------------------------------------------------------------------
# Cohort container
# --------------------------------------------------------------------------


@dataclass
class Cohort:
    """One fully-assembled ``(cohort, method, model, task)`` feature matrix."""

    cohort: str
    method: str
    model: str
    task: str
    feats: torch.Tensor           # (N, D) float32, rows sorted by WSI id
    labels: torch.Tensor          # (N,) int64
    ids: List[str]                # length N, sorted
    source_dir: Path
    missing_ids: List[str] = field(default_factory=list)      # labelled, no features
    unlabelled_ids: List[str] = field(default_factory=list)   # features, no label
    from_cache: bool = False

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def dim(self) -> int:
        return int(self.feats.shape[1])

    @property
    def index(self) -> Dict[str, int]:
        """``{wsi_id: row}`` for index-based split construction."""
        if not hasattr(self, "_index_cache"):
            object.__setattr__(self, "_index_cache", {w: i for i, w in enumerate(self.ids)})
        return self._index_cache  # type: ignore[attr-defined]

    def rows_for(self, wsi_ids: Iterable[str]) -> np.ndarray:
        """Row indices for the given ids, silently dropping ids not present."""
        idx = self.index
        return np.array([idx[w] for w in wsi_ids if w in idx], dtype=np.int64)

    def subset(self, rows: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """``(feats, labels, ids)`` for a set of row indices - a view, not a copy."""
        rows_t = torch.as_tensor(rows, dtype=torch.long)
        return self.feats[rows_t], self.labels[rows_t], [self.ids[i] for i in rows]

    def class_counts(self) -> Dict[int, int]:
        vals, counts = np.unique(self.labels.numpy(), return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, counts)}

    def write_missing_report(self, output_dir: os.PathLike | str) -> Optional[Path]:
        """Write ``missing_slides.csv`` if anything is missing (Task 1.4)."""
        if not self.missing_ids and not self.unlabelled_ids:
            return None
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {"cohort": self.cohort, "method": self.method, "model": self.model,
             "task": self.task, "wsi_id": w, "reason": "labelled_but_no_features"}
            for w in self.missing_ids
        ] + [
            {"cohort": self.cohort, "method": self.method, "model": self.model,
             "task": self.task, "wsi_id": w, "reason": "features_but_no_label"}
            for w in self.unlabelled_ids
        ]
        path = out_dir / "missing_slides.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"Cohort({self.cohort}/{self.method}/{self.model}/{self.task}: "
            f"N={self.n}, D={self.dim}, classes={self.class_counts()}, "
            f"missing={len(self.missing_ids)}, cached={self.from_cache})"
        )


# --------------------------------------------------------------------------
# Cache (1c.1 item 5)
# --------------------------------------------------------------------------


def _source_signature(feature_dir: Path) -> str:
    """Fingerprint of a feature directory: file count + newest mtime + names.

    Invalidates the cache when files are added, removed, or rewritten.
    """
    entries = sorted(f for f in os.listdir(feature_dir) if f.endswith(".pt"))
    newest = 0.0
    for f in entries:
        try:
            newest = max(newest, os.path.getmtime(feature_dir / f))
        except OSError:
            pass
    digest = hashlib.sha1("\n".join(entries).encode("utf-8")).hexdigest()[:16]
    return f"{len(entries)}:{newest:.0f}:{digest}"


def _cache_paths(cohort: str, method: str, model: str, task: str) -> Tuple[Path, Path]:
    stem = f"{cohort}_{method}_{model}_{task}"
    d = P.cache_dir()
    return d / f"{stem}.npz", d / f"{stem}.meta.json"


# --------------------------------------------------------------------------
# The loader
# --------------------------------------------------------------------------


def load_cohort(
    cohort: str,
    method: str,
    model: str,
    task: str = "MSIH",
    use_cache: bool = True,
    verbose: bool = True,
) -> Cohort:
    """Assemble one cohort's feature matrix, labels and ids.

    Returns every slide that has **both** a label and a feature file, sorted by
    WSI id. Slides labelled but lacking features, and slides with features but
    no label, are reported on ``Cohort.missing_ids`` / ``Cohort.unlabelled_ids``
    rather than silently dropped.

    Raises on a shape mismatch (fail-fast on correctness, 1c.3 item 5); raises
    ``FileNotFoundError`` if the feature directory is absent, which callers are
    expected to catch and turn into a skip (fail-soft on availability).
    """
    cohort = cohort.lower()
    fdir = P.feature_dir(cohort, method, model)
    if not fdir.is_dir():
        raise FileNotFoundError(
            f"Feature directory missing for {cohort}/{method}/{model}: {fdir}"
        )

    label_map = load_label_map(cohort, task)
    expected_d = P.expected_dim(method, model)

    npz_path, meta_path = _cache_paths(cohort, method, model, task)
    signature = _source_signature(fdir)

    if use_cache and npz_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("signature") == signature and meta.get("expected_dim") == expected_d:
                blob = np.load(npz_path, allow_pickle=False)
                coh = Cohort(
                    cohort=cohort, method=method, model=model, task=task,
                    feats=torch.from_numpy(blob["feats"]),
                    labels=torch.from_numpy(blob["labels"]).long(),
                    ids=[str(x) for x in blob["ids"]],
                    source_dir=fdir,
                    missing_ids=list(meta.get("missing_ids", [])),
                    unlabelled_ids=list(meta.get("unlabelled_ids", [])),
                    from_cache=True,
                )
                if verbose:
                    print(f"[cache] {coh}")
                _warn_missing(coh, verbose)
                return coh
        except Exception as exc:  # corrupt cache is recoverable - rebuild it
            print(f"[WARN] Cache unreadable ({exc}); rebuilding {npz_path.name}")

    # ---- assemble from disk -------------------------------------------------
    on_disk = sorted(
        os.path.splitext(f)[0] for f in os.listdir(fdir) if f.endswith(".pt")
    )
    keep = [w for w in on_disk if w in label_map]
    unlabelled = [w for w in on_disk if w not in label_map]
    missing = sorted(set(label_map) - set(on_disk))

    if not keep:
        raise ValueError(
            f"No slide in {fdir} matched any id in {P.labels_path(cohort, task)}. "
            f"Example feature id: {on_disk[0] if on_disk else '(none)'}; "
            f"example label id: {next(iter(label_map), '(none)')}"
        )

    feats = np.empty((len(keep), expected_d), dtype=np.float32)
    labels = np.empty(len(keep), dtype=np.int64)

    for row, wsi_id in enumerate(keep):
        tensor = torch.load(fdir / f"{wsi_id}.pt", map_location="cpu")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"{fdir / (wsi_id + '.pt')} holds {type(tensor).__name__}, expected a Tensor"
            )
        flat = tensor.detach().flatten().to(torch.float32)
        if flat.numel() != expected_d:
            # Fail-fast: a shape mismatch is a correctness failure, not an
            # availability problem (work order 1c.3 item 5).
            raise ValueError(
                f"Shape mismatch for {wsi_id} in {cohort}/{method}/{model}: "
                f"got {tuple(tensor.shape)} -> {flat.numel()} features, "
                f"expected {expected_d}."
            )
        feats[row] = flat.numpy()
        labels[row] = label_map[wsi_id]

    coh = Cohort(
        cohort=cohort, method=method, model=model, task=task,
        feats=torch.from_numpy(feats),
        labels=torch.from_numpy(labels),
        ids=keep,
        source_dir=fdir,
        missing_ids=missing,
        unlabelled_ids=unlabelled,
        from_cache=False,
    )

    if use_cache:
        try:
            np.savez(npz_path, feats=feats, labels=labels, ids=np.array(keep, dtype=object).astype(str))
            meta_path.write_text(json.dumps({
                "signature": signature,
                "expected_dim": expected_d,
                "n": len(keep),
                "missing_ids": missing,
                "unlabelled_ids": unlabelled,
                "source_dir": str(fdir),
                **P.run_stamp(),
            }, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"[WARN] Could not write cache {npz_path.name}: {exc}")

    if verbose:
        print(f"[load ] {coh}")
    _warn_missing(coh, verbose)
    return coh


def _warn_missing(coh: Cohort, verbose: bool) -> None:
    """Task 1.4: never let a labelled slide disappear without a word."""
    if not verbose:
        return
    if coh.missing_ids:
        print(
            f"[WARN] {len(coh.missing_ids)} labelled slide(s) have no features in "
            f"{coh.cohort}/{coh.method}/{coh.model} -> reported N={coh.n}, "
            f"not {coh.n + len(coh.missing_ids)}:"
        )
        for w in coh.missing_ids:
            print(f"[WARN]   missing features: {w}")
    if coh.unlabelled_ids:
        print(
            f"[INFO] {len(coh.unlabelled_ids)} slide(s) have features but no label "
            f"(harmless, excluded): {', '.join(coh.unlabelled_ids)}"
        )


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------


def build_splits(
    ids: Sequence[str], fold_map: Dict[str, int]
) -> Dict[int, np.ndarray]:
    """``{fold: row indices}`` for the rows of a loaded cohort.

    ``ids`` is ``Cohort.ids``; ``fold_map`` is ``{wsi_id: fold}``. Rows whose id
    is absent from ``fold_map`` land in no fold.
    """
    index = {w: i for i, w in enumerate(ids)}
    by_fold: Dict[int, List[int]] = {}
    for wsi_id, fold in fold_map.items():
        row = index.get(wsi_id)
        if row is not None:
            by_fold.setdefault(int(fold), []).append(row)
    return {f: np.array(sorted(rows), dtype=np.int64) for f, rows in sorted(by_fold.items())}


def cv_rotation(
    splits: Dict[int, np.ndarray], test_fold: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The project's 2-train / 1-val / 1-test fold rotation.

    ``test = fold i``, ``val = fold i+1`` (wrapping), ``train = the rest``.
    Confirmed deliberate by the owner - see "Known issues carried forward" #1 in
    the work order: training uses 50% of the cohort, not the 75% a reader might
    assume from "4-fold CV".
    """
    fold_ids = sorted(splits)
    if test_fold not in splits:
        raise KeyError(f"Fold {test_fold} not in splits {fold_ids}")
    pos = fold_ids.index(test_fold)
    val_fold = fold_ids[(pos + 1) % len(fold_ids)]
    train = np.concatenate(
        [splits[f] for f in fold_ids if f not in (test_fold, val_fold)]
    )
    return np.sort(train), splits[val_fold], splits[test_fold]


def stratified_holdout(
    labels: torch.Tensor, frac: float = 0.15, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified ``(train_rows, holdout_rows)`` split of a label vector.

    Used for the ANN's early-stopping set when training on a full cohort with no
    natural validation fold (Task 3.0a step 2).
    """
    y = labels.numpy()
    rng = np.random.default_rng(seed)
    train_rows, hold_rows = [], []
    for cls in np.unique(y):
        rows = np.flatnonzero(y == cls)
        rng.shuffle(rows)
        n_hold = max(1, int(round(len(rows) * frac)))
        hold_rows.append(rows[:n_hold])
        train_rows.append(rows[n_hold:])
    return np.sort(np.concatenate(train_rows)), np.sort(np.concatenate(hold_rows))


def assert_no_leakage(
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    group_of=None,
    label: str = "split",
) -> None:
    """Halt loudly if any slide - or group - appears in both train and test.

    ``group_of`` maps a slide id to its grouping key (patient/case). Pass it for
    SurGen, where 70 cases contribute two slides each (Task 2.1). A leakage
    failure is a correctness failure and must stop the run (1c.3 item 5).
    """
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise AssertionError(
            f"[LEAKAGE] {len(overlap)} slide(s) in both train and test of {label}: "
            f"{sorted(overlap)[:10]}"
        )
    if group_of is not None:
        train_groups = {group_of(w) for w in train_ids}
        test_groups = {group_of(w) for w in test_ids}
        shared = train_groups & test_groups
        if shared:
            raise AssertionError(
                f"[LEAKAGE] {len(shared)} case(s) in both train and test of {label}: "
                f"{sorted(shared)[:10]}"
            )
