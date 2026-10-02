"""Fold a remote result drop into its canonical place in the results tree.

Why this exists
---------------
Cross-validation is split across two machines. The Windows workstation produces
some (method, model) combinations; the Linux server produces the rest. The
server's output arrives as a whole subtree dumped *inside* the results tree it
belongs to::

    SurGen_Results/
        Averaging/Conch1_5/1-MSIH/...          <- produced here, already correct
        Averaging/Virchow2/1-MSIH/...          <- produced here, already correct
        SurGen_Results_linux/                  <- the drop, nested one level deep
            Averaging/ConchV1/1-MSIH/...
            Averaging/H-Optimus-1/1-MSIH/...
            ...

The drop's internal layout is identical to the destination's - it is in the
wrong place, not the wrong shape. So merging it is a pure relocation, and this
script does exactly that and nothing else: it never renames, rewrites, or
recomputes a single result.

What it will not do
-------------------
Silently destroy a result. A file whose destination is already occupied by
DIFFERENT content is left alone and reported as a conflict; only ``--overwrite``
replaces it. Identical content (same SHA-256) means the drop was already merged,
so the redundant source copy is discarded.

Because a conflict keeps its source file, and empty directories are the only
ones removed, an aborted merge leaves the source tree holding precisely the
files that still need a decision.

Paths are never hardcoded here - every destination is derived from
``config.paths``, and every method/model name is validated against the canonical
vocabulary in that same module. A drop containing a name the pipeline does not
recognise is reported, never guessed at.

Run:  python merge_results.py                     # dry run: print the plan
      python merge_results.py --apply             # do it, then remove the drop
      python merge_results.py --apply --overwrite # let the drop win conflicts
      python merge_results.py --source DIR --experiment TCGA-CV
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config import paths as P  # noqa: E402

DEFAULT_EXPERIMENT = "SurGen-CV"
DEFAULT_SOURCE = HERE / "SurGen_Results" / "SurGen_Results_linux"

#: On-disk task folder ("1-MSIH") -> canonical task name ("MSIH"). The forward
#: mapping is the single source of truth; this is only its inverse.
TASK_DIR_TO_NAME: Dict[str, str] = {v: k for k, v in P.TASK_PREFIXES.items()}

#: Filename prefixes that mark a combination as a COMPLETE run rather than a
#: metrics-only handover. Written by runners.results_io.ResultSet.write.
COMPLETENESS_MARKERS = ("result_", "summary_", "oof_predictions_")

NEW = "new"
IDENTICAL = "identical"
CONFLICT = "conflict"


@dataclass
class Item:
    """One file to relocate, with the combination it belongs to."""

    src: Path
    dst: Path
    method: str
    model: str
    task_dir: str
    status: str = NEW

    @property
    def combo(self) -> Tuple[str, str, str]:
        return (self.method, self.model, self.task_dir)


# ----------------------------------------------------------------- discovery


def experiment_root(experiment: str) -> Path:
    """Top-level results directory for an experiment.

    Derived from ``results_root`` by stripping the method/model/task tail it
    appends, so the experiment -> folder mapping stays defined in exactly one
    place (``config/paths.py``) even though we need it one level up.
    """
    probe = P.results_root(experiment, "Averaging", "Conch1_5", "MSIH")
    return probe.parents[2]


def destination_dir(dest_root: Path, method: str, model: str, task_dir: str) -> Path:
    """Where a combination's files belong under ``dest_root``.

    Mirrors the shape of ``config.paths.results_root``, including its PRISM
    special case (a slide-level encoder: one flat directory, no model subdir).
    """
    if method == "PRISM":
        return dest_root / "PRISM" / task_dir
    return dest_root / method / model / task_dir


def discover(source_root: Path, dest_root: Path) -> Tuple[List[Item], List[Path]]:
    """Map every file in the drop onto its destination.

    Returns ``(items, unmapped)``. A path is unmapped - never guessed at - if it
    is too shallow to carry a combination, or names a method, model, task, or
    pairing the pipeline does not recognise.
    """
    items: List[Item] = []
    unmapped: List[Path] = []

    for src in sorted(p for p in source_root.rglob("*") if p.is_file()):
        rel = src.relative_to(source_root)
        # <Method>/<Model>/<TaskDir>/<tail...>, so at least 4 components.
        if len(rel.parts) < 4:
            unmapped.append(src)
            continue

        method, model, task_dir = rel.parts[0], rel.parts[1], rel.parts[2]
        tail = Path(*rel.parts[3:])

        if (method not in P.AGGREGATION_METHODS
                or model not in P.CANONICAL_MODELS
                or task_dir not in TASK_DIR_TO_NAME
                or not P.is_combination_valid(method, model)):
            unmapped.append(src)
            continue

        dst = destination_dir(dest_root, method, model, task_dir) / tail
        items.append(Item(src=src, dst=dst, method=method, model=model,
                          task_dir=task_dir))

    return items, unmapped


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def classify(items: List[Item]) -> None:
    """Decide new / identical / conflict for each item, before touching disk."""
    for item in items:
        if not item.dst.exists():
            item.status = NEW
        elif sha256(item.src) == sha256(item.dst):
            item.status = IDENTICAL
        else:
            item.status = CONFLICT


# -------------------------------------------------------------------- report


def report(items: List[Item], unmapped: List[Path], source_root: Path,
           dest_root: Path, overwrite: bool) -> None:
    """Print the full source -> destination plan, grouped by combination."""
    by_combo: Dict[Tuple[str, str, str], List[Item]] = defaultdict(list)
    for item in items:
        by_combo[item.combo].append(item)

    print(f"\n  source : {source_root}")
    print(f"  dest   : {dest_root}")
    print(f"\n  {len(items)} files in {len(by_combo)} combinations")
    print("-" * 78)

    last_method = None
    for (method, model, task_dir), group in sorted(by_combo.items()):
        if method != last_method:
            marker = "" if (dest_root / method).is_dir() else "   [NEW METHOD FOLDER]"
            print(f"\n{method}{marker}")
            last_method = method

        counts: Dict[str, int] = defaultdict(int)
        for item in group:
            counts[item.status] += 1
        note = ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
        print(f"  {model}/{task_dir}  ({len(group)} files: {note})")

        for item in sorted(group, key=lambda i: i.dst):
            rel_dst = item.dst.relative_to(dest_root)
            if item.status == CONFLICT:
                verb = "OVERWRITE" if overwrite else "CONFLICT "
            elif item.status == IDENTICAL:
                verb = "already  "
            else:
                verb = "move     "
            print(f"      {verb} -> {rel_dst}")

    print("\n" + "-" * 78)

    if unmapped:
        print(f"\n  {len(unmapped)} UNMAPPED path(s) - not touched, no destination could")
        print("  be derived. Check them against AGGREGATION_METHODS / CANONICAL_MODELS")
        print("  / TASK_PREFIXES in config/paths.py:")
        for path in unmapped:
            print(f"      {path.relative_to(source_root)}")


def completeness_note(items: List[Item], dest_root: Path) -> List[str]:
    """Combinations that arrived metrics-only, with no trained-model artifacts.

    Not an error - a handover of workbooks alone is a normal thing to receive -
    but worth stating, because those combinations end up thinner on disk than
    the ones produced locally.
    """
    thin = []
    for method, model, task_dir in sorted({item.combo for item in items}):
        out = destination_dir(dest_root, method, model, task_dir)
        if not out.is_dir():
            continue
        has_models = (out / "models").is_dir()
        has_records = any(p.name.startswith(COMPLETENESS_MARKERS)
                          for p in out.rglob("*") if p.is_file())
        if has_models and has_records:
            continue
        missing = []
        if not has_models:
            missing.append("models/")
        if not has_records:
            missing.append("result/summary/oof")
        thin.append(f"{method}/{model}/{task_dir}  (no {', '.join(missing)})")
    return thin


# --------------------------------------------------------------------- apply


def apply_merge(items: List[Item], overwrite: bool) -> Dict[str, int]:
    """Move every movable item. Conflicts are left in the source untouched."""
    tally = {"moved": 0, "skipped": 0, "conflicts": 0, "overwritten": 0}

    for item in items:
        if item.status == CONFLICT and not overwrite:
            tally["conflicts"] += 1
            print(f"  CONFLICT, left in source: {item.src.name}")
            print(f"            destination differs: {item.dst}")
            continue

        if item.status == IDENTICAL:
            # Already merged; the source copy is redundant, so discard it.
            item.src.unlink()
            tally["skipped"] += 1
            continue

        item.dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(item.src), str(item.dst))
        if item.status == CONFLICT:
            tally["overwritten"] += 1
        else:
            tally["moved"] += 1

    return tally


def prune_empty(root: Path) -> bool:
    """Remove empty directories bottom-up. Returns True if ``root`` itself went.

    Only ever removes a directory that is already empty, so anything the merge
    deliberately left behind (a conflict) keeps its enclosing directories and
    stays visible.
    """
    for path in sorted((p for p in root.rglob("*") if p.is_dir()),
                       key=lambda p: len(p.parts), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass  # not empty - something was left behind on purpose
    try:
        root.rmdir()
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Move a remote result drop into its canonical place.",
        epilog="Dry run by default. Nothing is written without --apply.",
    )
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                    help=f"drop to merge (default: {DEFAULT_SOURCE.name}/)")
    ap.add_argument("--experiment", default=DEFAULT_EXPERIMENT,
                    help=f"canonical experiment name (default: {DEFAULT_EXPERIMENT})")
    ap.add_argument("--dest", type=Path, default=None,
                    help="results tree to merge into (default: from --experiment)")
    ap.add_argument("--apply", action="store_true",
                    help="actually move the files (without this, nothing is written)")
    ap.add_argument("--overwrite", action="store_true",
                    help="on a content conflict, let the drop replace the destination")
    args = ap.parse_args()

    source_root = args.source.resolve()
    dest_root = (args.dest.resolve() if args.dest is not None
                 else experiment_root(args.experiment))

    print("=" * 78)
    print(f"  merge_results - {args.experiment}"
          f"{'' if args.apply else '   (DRY RUN)'}")
    print("=" * 78)

    if not source_root.is_dir():
        print(f"\n  Nothing to merge: {source_root} does not exist.")
        print("  (If you already merged this drop, that is the expected result.)")
        print("=" * 78)
        return 0

    items, unmapped = discover(source_root, dest_root)

    if not items and not unmapped:
        print(f"\n  Nothing to merge: {source_root} holds no files.")
        if args.apply and prune_empty(source_root):
            print(f"  Removed the empty {source_root.name}/ tree.")
        print("=" * 78)
        return 0

    classify(items)
    report(items, unmapped, source_root, dest_root, args.overwrite)

    n_conflict = sum(1 for i in items if i.status == CONFLICT)

    if not args.apply:
        print("\n  DRY RUN - nothing was written.")
        print(f"  Re-run with --apply to perform the merge"
              f"{' (--overwrite to resolve the conflicts)' if n_conflict else ''}.")
        print("=" * 78)
        return 1 if (n_conflict and not args.overwrite) or unmapped else 0

    print("\n  Applying...")
    tally = apply_merge(items, args.overwrite)
    removed = prune_empty(source_root)

    print("\n" + "=" * 78)
    print(f"  moved            : {tally['moved']}")
    if tally["overwritten"]:
        print(f"  overwritten      : {tally['overwritten']}")
    print(f"  already merged   : {tally['skipped']}  (identical, source discarded)")
    print(f"  conflicts        : {tally['conflicts']}  (left in source, untouched)")
    print(f"  unmapped         : {len(unmapped)}")

    if removed:
        print(f"\n  {source_root.name}/ was empty afterwards and has been removed.")
    else:
        print(f"\n  {source_root.name}/ still exists - it holds the files listed above")
        print("  that were not merged. Resolve them, then re-run.")

    thin = completeness_note(items, dest_root)
    if thin:
        print(f"\n  Note: {len(thin)} merged combination(s) are metrics-only -")
        print("  workbooks without the trained-model artifacts a local run writes:")
        for line in thin:
            print(f"      {line}")

    print("=" * 78)
    return 1 if (tally["conflicts"] or unmapped) else 0


if __name__ == "__main__":
    sys.exit(main())
