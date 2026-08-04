"""Result writing: merge-friendly, provenance-stamped, written once per run.

Work order 1c.1 item 4 - ``write_data_in_excel`` is quadratic: every call opens
the workbook, reads *all* sheets into DataFrames and rewrites *all* of them.
Results are accumulated in memory here and the workbook is written once, at the
end of a run. The old function stays in ``utility.py`` for backwards
compatibility but is no longer called in a loop.

Work order 1c.2 item 4 - one result file per ``(experiment, method, model,
variant)``, each stamped with ``machine``, ``timestamp``, ``git_commit`` and
``n_test``, so merging results produced on two machines is a file copy plus
``tools/merge_results.py``.
"""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from config import paths as P

METRIC_NAMES = ("acc", "bacc", "macro_f1", "weighted_f1", "auroc")

#: Experiment trees already archived in this process - archive at most once.
_ARCHIVED: Set[str] = set()


def archive_experiment_tree(experiment: str, when: str = None) -> Optional[Path]:
    """Rename an existing results tree to ``<name>_ARCHIVED_<date>`` before rewriting.

    Work order rule of engagement #2: *never delete an existing results folder,
    and never edit a results file by hand*. Every result produced after the
    Phase 1 corrections supersedes the equivalent result produced before them,
    but the old numbers may already have been circulated and must stay auditable.

    Called once per experiment per process, before the first write.
    """
    if experiment in _ARCHIVED:
        return None
    _ARCHIVED.add(experiment)

    folder = P._EXPERIMENT_FOLDERS.get(experiment, experiment)
    tree = P.SLIDE_CLS_ROOT / folder
    if not tree.is_dir() or not any(tree.iterdir()):
        return None

    stamp = when or date.today().strftime("%Y%m%d")
    dest = tree.with_name(f"{tree.name}_ARCHIVED_{stamp}")
    n = 1
    while dest.exists():
        n += 1
        dest = tree.with_name(f"{tree.name}_ARCHIVED_{stamp}_{n}")

    n_before = sum(1 for _ in tree.rglob("*") if _.is_file())

    try:
        tree.rename(dest)
        print(f"[ARCHIVE] {tree.name} -> {dest.name} "
              f"({n_before} files, pre-correction results preserved)")
        return dest
    except OSError as exc:
        # Windows refuses to rename a directory while anything holds a handle on
        # it (a file watcher, an indexer, or a virus scanner is enough). Falling
        # back to a verified copy still satisfies the rule that matters - the old
        # numbers remain auditable - after which new results overwrite the live
        # tree file by file.
        print(f"[ARCHIVE] rename failed ({type(exc).__name__}: {exc}); "
              f"falling back to copy")

    shutil.copytree(tree, dest, dirs_exist_ok=False)
    n_after = sum(1 for _ in dest.rglob("*") if _.is_file())
    if n_after != n_before:
        raise RuntimeError(
            f"Archive copy of {tree.name} is incomplete: {n_after} files copied "
            f"vs {n_before} in the source. Refusing to continue - pre-correction "
            f"results must be preserved intact before they are overwritten."
        )
    print(f"[ARCHIVE] {tree.name} -> {dest.name} (copied, {n_after} files verified; "
          f"live tree will be overwritten in place)")
    return dest


def _jsonable(obj):
    """Make numpy/torch scalars and arrays JSON-serialisable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


class ResultAccumulator:
    """Collects one ``(experiment, method, model, variant)`` run in memory."""

    def __init__(self, experiment: str, cohort: str, method: str, model: str,
                 task: str = "MSIH", variant: str = "default", seed: int = 42):
        self.experiment = experiment
        self.cohort = cohort
        self.method = method
        self.model = model
        self.task = task
        self.variant = variant
        self.seed = seed
        self.folds: List[Dict[str, Any]] = []
        self.notes: Dict[str, Any] = {}

    # -- collection --------------------------------------------------------

    def add_fold(self, classifier: str, fold: int, metrics: Dict[str, Any],
                 dump: Dict[str, Any], wsi_ids: List[str],
                 selection: Optional[dict] = None,
                 n_train: int = None, n_val: int = None) -> None:
        pfx = f"{classifier}_"
        row = {
            "classifier": classifier,
            "fold": fold,
            "n_test": len(wsi_ids),
            "n_train": n_train,
            "n_val": n_val,
            "wsi_ids": list(wsi_ids),
            "targets": _jsonable(dump["targets_all"]),
            "preds": _jsonable(dump["preds_all"]),
            "probs_pos": _jsonable(np.asarray(dump["probs_all"])[:, 1]),
            "selection": _jsonable(selection) if selection else None,
        }
        for m in METRIC_NAMES:
            row[m] = float(metrics[f"{pfx}{m}"]) if f"{pfx}{m}" in metrics else None
        cm = metrics.get(f"{pfx}conf_matrix")
        row["conf_matrix"] = _jsonable(cm) if cm is not None else None
        for k, v in metrics.items():
            if k.startswith("val_"):
                row[k] = float(v) if isinstance(v, (int, float, np.floating)) else _jsonable(v)
        self.folds.append(row)

    # -- derived tables ----------------------------------------------------

    def classifiers(self) -> List[str]:
        seen, out = set(), []
        for f in self.folds:
            if f["classifier"] not in seen:
                seen.add(f["classifier"])
                out.append(f["classifier"])
        return out

    def summary_frame(self) -> pd.DataFrame:
        """Per-classifier mean across folds, plus the averaged confusion matrix."""
        rows = []
        for clf in self.classifiers():
            fs = [f for f in self.folds if f["classifier"] == clf]
            row = {"Method": self.method, "Model": self.model, "Classifier": clf,
                   "N_folds": len(fs), "N_test": int(sum(f["n_test"] for f in fs))}
            for m in METRIC_NAMES:
                vals = [f[m] for f in fs if f[m] is not None]
                row[m] = float(np.mean(vals)) if vals else None
                row[f"{m}_sd"] = float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0
            cms = [np.array(f["conf_matrix"]) for f in fs if f["conf_matrix"] is not None]
            if cms:
                row["conf_matrix"] = str(np.mean(cms, axis=0).round().astype(int).tolist())
            rows.append(row)
        return pd.DataFrame(rows)

    def eval_metrics_frame(self) -> pd.DataFrame:
        """The legacy wide layout: one row per metric, one column per fold.

        Kept byte-compatible with what ``Analysis_and_Visualization/Analysis.ipynb``
        already parses out of ``cross_valid_avg_eval_metrics.xlsx``.
        """
        fold_ids = sorted({f["fold"] for f in self.folds})
        cols = ["Metric"] + [f"Fold{i}" for i in fold_ids] + ["AvgFolds"]
        rows = []
        for clf in self.classifiers():
            by_fold = {f["fold"]: f for f in self.folds if f["classifier"] == clf}
            for m in METRIC_NAMES:
                vals = [by_fold[i][m] if i in by_fold else None for i in fold_ids]
                present = [v for v in vals if v is not None]
                rows.append([f"{clf}_{m}"] + vals +
                            [float(np.mean(present)) if present else None])
            cms = [np.array(by_fold[i]["conf_matrix"]) for i in fold_ids
                   if i in by_fold and by_fold[i]["conf_matrix"] is not None]
            avg_cm = str(np.mean(cms, axis=0).round().astype(int).tolist()) if cms else "N/A"
            rows.append([f"{clf}_conf_matrix"] +
                        [str(by_fold[i]["conf_matrix"]) if i in by_fold else "N/A"
                         for i in fold_ids] + [avg_cm])
        return pd.DataFrame(rows, columns=cols)

    def probs_frame(self) -> pd.DataFrame:
        """One row per (fold, slide), one prediction/probability column per classifier."""
        merged = None
        for clf in self.classifiers():
            parts = []
            for f in (x for x in self.folds if x["classifier"] == clf):
                parts.append(pd.DataFrame({
                    "Fold": f["fold"],
                    "WSI_ID": f["wsi_ids"],
                    "Target": f["targets"],
                    f"{clf}_Pred": f["preds"],
                    f"{clf}_Prob": np.round(f["probs_pos"], 6),
                }))
            if not parts:
                continue
            df = pd.concat(parts, ignore_index=True)
            merged = df if merged is None else merged.merge(
                df, on=["Fold", "WSI_ID", "Target"], how="outer")
        return merged if merged is not None else pd.DataFrame()

    def oof_frame(self) -> pd.DataFrame:
        """Out-of-fold predictions: every slide predicted exactly once per classifier.

        This is the input to the threshold policy of work order section 1b -
        tau_TCGA is fitted on pooled out-of-fold TCGA probabilities, which are
        genuinely out-of-sample and involve no external data.
        """
        parts = []
        for f in self.folds:
            parts.append(pd.DataFrame({
                "classifier": f["classifier"],
                "fold": f["fold"],
                "WSI_ID": f["wsi_ids"],
                "target": f["targets"],
                "prob_pos": f["probs_pos"],
                "pred_at_0.5": f["preds"],
            }))
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    # -- writing -----------------------------------------------------------

    def stamp(self) -> Dict[str, Any]:
        n_test = int(sum(f["n_test"] for f in self.folds if f["classifier"] == (
            self.classifiers()[0] if self.classifiers() else None)))
        return P.run_stamp(
            experiment=self.experiment, cohort=self.cohort, method=self.method,
            model=self.model, task=self.task, variant=self.variant, seed=self.seed,
            n_test=n_test, classifiers=self.classifiers(), **self.notes,
        )

    def write(self, output_dir: Path, write_excel: bool = True) -> Dict[str, Path]:
        """Write every artifact for this run. The workbook is written once."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written: Dict[str, Path] = {}

        # Merge-friendly, fully self-describing record.
        payload = {"stamp": self.stamp(), "folds": self.folds,
                   "summary": _jsonable(self.summary_frame().to_dict(orient="records"))}
        p = output_dir / f"result_{self.experiment}_{self.method}_{self.model}_{self.variant}.json"
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written["json"] = p

        summary = self.summary_frame()
        p = output_dir / f"summary_{self.variant}.csv"
        summary.to_csv(p, index=False)
        written["summary"] = p

        oof = self.oof_frame()
        if not oof.empty:
            p = output_dir / f"oof_predictions_{self.variant}.csv"
            oof.to_csv(p, index=False)
            written["oof"] = p

        if write_excel:
            p = output_dir / "cross_valid_avg_eval_metrics.xlsx"
            with pd.ExcelWriter(p, engine="openpyxl", mode="w") as xw:
                self.eval_metrics_frame().to_excel(xw, sheet_name="baseline", index=False)
                summary.to_excel(xw, sheet_name="Summary", index=False)
            written["metrics_xlsx"] = p

            probs = self.probs_frame()
            if not probs.empty:
                p = output_dir / "cross_valid_avg_probs_all.xlsx"
                with pd.ExcelWriter(p, engine="openpyxl", mode="w") as xw:
                    probs.to_excel(xw, sheet_name="baseline", index=False)
                written["probs_xlsx"] = p

        return written
