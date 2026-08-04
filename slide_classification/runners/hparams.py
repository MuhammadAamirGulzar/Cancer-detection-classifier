"""Derive the TCGA-FULL hyperparameters from TCGA-CV alone (Task 3.0a step 1).

Rule, from the work order: *"For each (aggregation method x foundation model x
classifier), take the configuration selected most often across the 4 folds; break
ties by best mean validation score. No external data may influence this step."*

Only the ANN has a real grid, so only the ANN has anything to vote on. The other
four classifiers have single-point grids (verified in Task 1.1), so their
"selection" is just the single configuration they were always given - recorded
here explicitly so ``tcga_full_hparams.json`` is a complete, auditable statement
of what the final models were trained with.

Every value written by this module is traceable to a TCGA-CV *validation* score.
Nothing here reads PAIP or SurGen - :func:`assert_no_external_data` checks that.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import paths as P
from runners.classifiers import GRIDS, MODEL_TYPES

HPARAMS_PATH = P.SLIDE_CLS_ROOT / "tcga_full_hparams.json"


def _cv_result_path(method: str, model: str, task: str = "MSIH") -> Path:
    out = P.results_root("TCGA-CV", method, model, task) / "Output"
    return out / f"result_TCGA-CV_{method}_{model}_default.json"


def _single_point(kind: str) -> Dict[str, Any]:
    """The one configuration a single-point grid can produce."""
    return {k: v[0] for k, v in GRIDS[kind].items()}


def derive_for_combination(
    method: str, model: str, task: str = "MSIH"
) -> Dict[str, Dict[str, Any]]:
    """``{classifier: hparams}`` for one (method, model), from its TCGA-CV run."""
    path = _cv_result_path(method, model, task)
    if not path.exists():
        raise FileNotFoundError(
            f"No TCGA-CV result at {path}. TCGA-FULL hyperparameters are derived "
            f"from TCGA-CV, so TCGA-CV must run first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    folds = payload["folds"]

    out: Dict[str, Dict[str, Any]] = {}
    for kind in MODEL_TYPES:
        rows = [f for f in folds if f["classifier"] == kind]
        if not rows:
            continue

        if kind != "ann":
            out[kind] = {
                **_single_point(kind),
                "_source": "single-point grid; no selection occurs",
                "_n_folds": len(rows),
            }
            continue

        # --- ANN: majority vote across folds, tie-break on mean validation score
        votes: List[Tuple] = []
        val_by_config: Dict[Tuple, List[float]] = {}
        for f in rows:
            sel = f.get("selection") or {}
            chosen = sel.get("selected")
            if not chosen:
                raise ValueError(
                    f"Fold {f['fold']} of {method}/{model} has no ANN selection "
                    f"record. Re-run TCGA-CV with the corrected runner."
                )
            key = (chosen["hidden_dim1"], chosen["hidden_dim2"], chosen["max_iter"])
            votes.append(key)
            for entry in sel.get("grid_trace", []):
                k = (entry["hidden_dim1"], entry["hidden_dim2"], entry["max_iter"])
                val_by_config.setdefault(k, []).append(entry["val_macro_f1"])

        counts = Counter(votes)
        top = max(counts.values())
        tied = sorted(k for k, c in counts.items() if c == top)

        if len(tied) == 1:
            winner, how = tied[0], f"majority vote ({top}/{len(votes)} folds)"
        else:
            # Tie-break by best MEAN VALIDATION score across folds.
            def mean_val(k):
                v = val_by_config.get(k, [])
                return sum(v) / len(v) if v else float("-inf")
            winner = max(tied, key=mean_val)
            how = (f"{len(tied)}-way tie at {top}/{len(votes)} folds, broken by "
                   f"mean validation macro-F1 = {mean_val(winner):.6f}")

        mean_scores = {
            f"{k[0]}_{k[1]}_{k[2]}": round(sum(v) / len(v), 6)
            for k, v in sorted(val_by_config.items())
        }
        out["ann"] = {
            "hidden_dim1": winner[0],
            "hidden_dim2": winner[1],
            "max_iter": winner[2],
            "_source": how,
            "_per_fold_selected": [f"{a}_{b}_{c}" for a, b, c in votes],
            "_mean_val_macro_f1_by_config": mean_scores,
            "_n_folds": len(rows),
        }

    return out


def assert_no_external_data(payload: dict) -> None:
    """Guard the work order's acceptance: no PAIP or SurGen input in this step."""
    blob = json.dumps(payload).lower()
    for forbidden in ("paip", "surgen"):
        if forbidden in blob:
            raise AssertionError(
                f"tcga_full_hparams.json mentions {forbidden!r}. TCGA-FULL "
                f"hyperparameters must be derived from TCGA-CV alone."
            )


def build_all(
    methods: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    task: str = "MSIH",
    path: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """Derive and persist hyperparameters for every available combination."""
    methods = list(methods or P.AGGREGATION_METHODS)
    models = list(models or P.CANONICAL_MODELS)

    store: Dict[str, Any] = {
        "_meta": P.run_stamp(
            policy=("configuration selected most often across the 4 TCGA-CV folds; "
                    "ties broken by best mean VALIDATION macro-F1"),
            source_experiment="TCGA-CV",
            external_data_used="none",
        )
    }
    missing = []
    for method in methods:
        for model in models:
            if not P.is_combination_valid(method, model):
                continue
            try:
                store[f"{method}|{model}|{task}"] = derive_for_combination(method, model, task)
            except FileNotFoundError:
                missing.append(f"{method}/{model}")

    assert_no_external_data(store)

    out = Path(path) if path else HPARAMS_PATH
    out.write_text(json.dumps(store, indent=2), encoding="utf-8")

    if verbose:
        n = len([k for k in store if not k.startswith("_")])
        print(f"tcga_full_hparams.json: {n} combination(s) -> {out}")
        if missing:
            print(f"  no TCGA-CV result yet for: {', '.join(missing)}")
        for key in sorted(k for k in store if not k.startswith("_")):
            ann = store[key].get("ann", {})
            if ann:
                print(f"  {key:<52} ann h1={ann['hidden_dim1']} h2={ann['hidden_dim2']} "
                      f"({ann['_source']})")
    return store


def load_hparams(method: str, model: str, classifier: str, task: str = "MSIH",
                 path: Optional[Path] = None) -> Dict[str, Any]:
    """Look up the fixed hyperparameters for one combination."""
    p = Path(path) if path else HPARAMS_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run runners.hparams first - TCGA-FULL "
            f"hyperparameters are derived from TCGA-CV (Task 3.0a step 1)."
        )
    store = json.loads(p.read_text(encoding="utf-8"))
    key = f"{method}|{model}|{task}"
    try:
        return {k: v for k, v in store[key][classifier].items() if not k.startswith("_")}
    except KeyError as exc:
        raise KeyError(f"No hyperparameters for {key} / {classifier}") from exc


if __name__ == "__main__":  # pragma: no cover
    build_all()
