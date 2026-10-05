"""Single source of truth for every filesystem root used by the pipeline.

Work order section 1c.2: aggregated features are split across two machines
(a local Windows workstation and a remote Linux server). The *same* code must
run unmodified on both, selected by the ``MACHINE`` environment variable:

    MACHINE=local   (default)  ->  D:\\Aamir Gulzar\\KSA_project2
    MACHINE=server             ->  /media/dp-psau/Datum/Aamir/Azfaar

No script or notebook outside this module may hardcode a data root. If you
find yourself typing ``D:\\Aamir Gulzar`` or ``/media/dp-psau`` anywhere else,
add it here instead.

Nothing in this module reads the feature tree at import time except
:func:`scan_available_features`, which is explicit and writes its findings to
``available_features.json``. Availability is always *scanned*, never assumed.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# --------------------------------------------------------------------------
# Machine selection
# --------------------------------------------------------------------------

MACHINE = os.environ.get("MACHINE", "local").strip().lower()

#: Repository root (``Cancer-detection-classifier/``), derived from this file's
#: location so it is correct on every machine without configuration.
REPO_ROOT = Path(__file__).resolve().parents[2]
SLIDE_CLS_ROOT = REPO_ROOT / "slide_classification"

#: Per-machine data roots. ``base`` is the parent that holds the cohort trees.
#:
#: The ``server`` values are taken from the paths hardcoded in the pre-existing
#: SurGen scripts (``slide_classification_surgen.py``,
#: ``slide_classification_surgen_best_k_labels_exp.py``). They have not been
#: verified from this workstation - the first ``MACHINE=server`` run should
#: check ``available_features.json`` before trusting them.
_MACHINE_ROOTS: Dict[str, Dict[str, str]] = {
    "local": {
        "base": r"D:\Aamir Gulzar\KSA_project2",
        "tcga_agg": r"dataset\slide_aggregation",
        "paip_agg": r"paip_data\slide_aggregation",
        "paip_labels": r"paip_data\labels",
        "surgen_processed": r"surgen_data\surgen_processed",
    },
    "server": {
        "base": "/media/dp-psau/Datum/Aamir/Azfaar",
        "tcga_agg": "dataset/slide_aggregation",
        "paip_agg": "paip_data/slide_aggregation",
        "paip_labels": "paip_data/labels",
        "surgen_processed": "surgen_processed",
    },
}

if MACHINE not in _MACHINE_ROOTS:
    raise ValueError(
        f"MACHINE={MACHINE!r} is not configured. "
        f"Known machines: {sorted(_MACHINE_ROOTS)}. "
        f"Add a new entry to _MACHINE_ROOTS in {__file__}."
    )

_R = _MACHINE_ROOTS[MACHINE]
BASE_ROOT = Path(_R["base"])

ROOTS: Dict[str, Path] = {
    "base": BASE_ROOT,
    "repo": REPO_ROOT,
    "slide_classification": SLIDE_CLS_ROOT,
    "tcga_agg": BASE_ROOT / _R["tcga_agg"],
    "paip_agg": BASE_ROOT / _R["paip_agg"],
    "paip_labels": BASE_ROOT / _R["paip_labels"],
    "surgen_processed": BASE_ROOT / _R["surgen_processed"],
}

# --------------------------------------------------------------------------
# Canonical vocabulary (work order section 1)
# --------------------------------------------------------------------------

COHORTS = ("tcga", "paip", "surgen")

CANONICAL_MODELS = ("H-Optimus-1", "Conch1_5", "UNI2", "Virchow2", "ConchV1", "PRISM")

AGGREGATION_METHODS = (
    "Averaging",
    "Caption_based_aggregation",
    "Caption_based_aggregation_15_classes",
    "Tissue_Type_Clustering",
    "TITAN",
    "PRISM",
)

#: Embedding width of each foundation model (one row of the aggregated tensor).
MODEL_BASE_DIMS: Dict[str, int] = {
    "H-Optimus-1": 1536,
    "UNI2": 1536,
    "Conch1_5": 768,
    "Virchow2": 2560,
    "ConchV1": 512,
    "PRISM": 1280,
}

#: Number of rows each aggregation method emits per slide. The classifier input
#: is the flattened ``(rows x dim)`` matrix, so ``expected_dim`` is their product.
AGG_MULTIPLIERS: Dict[str, int] = {
    "Averaging": 1,
    "Caption_based_aggregation": 14,
    "Caption_based_aggregation_15_classes": 15,
    "Tissue_Type_Clustering": 9,
    "TITAN": 1,
    "PRISM": 1,
}

TASK_PREFIXES: Dict[str, str] = {
    "MSIH": "1-MSIH",
    "BRAF": "2-BRAF",
    "KRAS": "3-KRAS",
    "TP53": "4-TP53",
    "CIMP": "5-CIMP",
}

#: Canonical model name -> on-disk directory name, per cohort.
#:
#: Work order Task 3.2: the SurGen feature tree uses lowercase-hyphenated names
#: (``conch1-5``, ``virchow2``) while every results tree uses the canonical
#: ``Conch1_5``/``Virchow2``. This mapping is explicit on purpose - relying on
#: case-insensitive path resolution works on Windows and silently fails on Linux.
_MODEL_DIR_NAMES: Dict[str, Dict[str, str]] = {
    "tcga": {m: m for m in CANONICAL_MODELS},
    "paip": {m: m for m in CANONICAL_MODELS},
    "surgen": {
        "Conch1_5": "conch1-5",
        "Virchow2": "virchow2",
        "H-Optimus-1": "h-optimus-1",
        "UNI2": "uni2-h",
        "ConchV1": "conch-v1",
        "PRISM": "prism",
    },
}


def model_dir_name(cohort: str, model: str) -> str:
    """On-disk directory name for a canonical model name in a given cohort."""
    cohort = cohort.lower()
    try:
        return _MODEL_DIR_NAMES[cohort][model]
    except KeyError as exc:
        raise KeyError(
            f"No on-disk directory name registered for cohort={cohort!r} "
            f"model={model!r}. Add it to _MODEL_DIR_NAMES in {__file__}."
        ) from exc


# --------------------------------------------------------------------------
# Combination validity
# --------------------------------------------------------------------------


def is_combination_valid(method: str, model: str) -> bool:
    """Whether a ``(method, model)`` pair is meaningful at all.

    TITAN is a slide-level encoder built on Conch1_5 only; PRISM is its own
    encoder and is model-agnostic. These guards already existed inline in every
    driver loop - they are centralised here so all runners agree.
    """
    if method == "TITAN":
        return model == "Conch1_5"
    if method == "PRISM":
        return model == "PRISM"
    # PRISM embeddings only exist via the PRISM aggregation method.
    return model != "PRISM"


def expected_dim(method: str, model: str) -> int:
    """Flattened feature width for a ``(method, model)`` combination."""
    if method == "TITAN":
        return 768
    if method == "PRISM":
        return MODEL_BASE_DIMS["PRISM"]
    base = MODEL_BASE_DIMS.get(model, 768)
    return base * AGG_MULTIPLIERS.get(method, 1)


# --------------------------------------------------------------------------
# Feature / label / results locations
# --------------------------------------------------------------------------


def feature_dir(cohort: str, method: str, model: str) -> Path:
    """Directory holding the per-slide ``.pt`` aggregated features.

    Returns the path whether or not it exists - callers decide how to handle a
    missing tree (work order 1c.2 item 3: skip, never fail).
    """
    cohort = cohort.lower()
    if cohort == "tcga":
        root = ROOTS["tcga_agg"]
        # PRISM is a slide-level encoder: one flat directory, no model subdir.
        return root / "PRISM" if method == "PRISM" else root / method / model
    if cohort == "paip":
        root = ROOTS["paip_agg"]
        return root / "PRISM" if method == "PRISM" else root / method / model
    if cohort == "surgen":
        disk_model = model_dir_name("surgen", model)
        return (
            ROOTS["surgen_processed"]
            / disk_model
            / "features"
            / "slide_aggregation"
            / method
            / disk_model
        )
    raise ValueError(f"Unknown cohort {cohort!r}; expected one of {COHORTS}")


def labels_path(cohort: str, task: str = "MSIH") -> Path:
    """CSV carrying the slide labels (and, for TCGA, the fold assignment)."""
    cohort = cohort.lower()
    if cohort == "tcga":
        if task == "MSIH":
            return SLIDE_CLS_ROOT / "kfolds_IDARS_fixed.csv"
        return SLIDE_CLS_ROOT / f"kfold_{task}.csv"
    if cohort == "paip":
        # Task 2.3 (done 2026-08-04): renamed from 'paip_kfolds_71.csv', which
        # held 78 rows and no fold column at all - it is a label table, and the
        # name had it being read as a fold table. PAIP has no folds: PAIP-CV is
        # retired (Task 2.2) and PAIP uses the provider's own 47/31 split.
        return SLIDE_CLS_ROOT / "paip_78_labels.csv"
    if cohort == "surgen":
        return SLIDE_CLS_ROOT / "surgen_labels.csv"
    raise ValueError(f"Unknown cohort {cohort!r}; expected one of {COHORTS}")


def folds_path(cohort: str, task: str = "MSIH") -> Optional[Path]:
    """CSV carrying fold assignments, or ``None`` if folds are built at runtime.

    Only TCGA ships provider-independent fold assignments. SurGen folds are
    rebuilt at case level (Task 2.1); PAIP uses the provider's own split
    (Task 3.1) and has no folds at all (Task 2.2 retired PAIP-CV).
    """
    cohort = cohort.lower()
    if cohort == "tcga":
        return labels_path("tcga", task)
    return None


def paip_split_path(split: str) -> Path:
    """PAIP official provider split: ``train`` (47 ids) or ``test`` (31 ids)."""
    if split == "train":
        return ROOTS["paip_labels"] / "paip_47slides.csv"
    if split == "test":
        return ROOTS["paip_labels"] / "paip_31slides_labels.csv"
    raise ValueError(f"PAIP split must be 'train' or 'test', got {split!r}")


def results_root(experiment: str, method: str, model: str, task: str = "MSIH") -> Path:
    """Output tree for one experiment/method/model/task combination.

    ``experiment`` is a canonical name from work order section 1
    (``TCGA-CV``, ``PAIP-IV``, ``PAIP-EV``, ``SurGen-CV``, ``SurGen-EV``).
    """
    folder = _EXPERIMENT_FOLDERS.get(experiment, experiment)
    task_folder = TASK_PREFIXES.get(task, task)
    if method == "PRISM":
        return SLIDE_CLS_ROOT / folder / "PRISM" / task_folder
    return SLIDE_CLS_ROOT / folder / method / model / task_folder


#: Canonical experiment name -> results directory. Existing directories keep
#: their historical names so previously published numbers stay findable.
_EXPERIMENT_FOLDERS: Dict[str, str] = {
    "TCGA-CV": "TCGA_Results",
    "PAIP-IV": "PAIP_IV_Results",
    "PAIP-EV": "TCGA_PAIP_EV_Results",
    "SurGen-CV": "SurGen_Results",
    "SurGen-EV": "SurGen_EV_Results",
    # Not an experiment - the trained artifact both EV experiments consume
    # (work order Task 3.0a).
    "TCGA-FULL": "TCGA_FULL_Models",
}


def cache_dir() -> Path:
    """Where assembled ``(N, D)`` feature matrices are memoised (1c.1 item 5)."""
    d = SLIDE_CLS_ROOT / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------
# Availability scan (1c.2 item 2) and run stamping (1c.2 item 4)
# --------------------------------------------------------------------------


def scan_available_features(
    cohorts: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    write_to: Optional[Path] = None,
) -> Dict[str, dict]:
    """Scan the feature tree and record every combination physically present.

    Never assume - always scan. Writes ``available_features.json`` listing each
    ``(cohort, method, model)`` with its ``.pt`` file count.
    """
    cohorts = list(cohorts or COHORTS)
    methods = list(methods or AGGREGATION_METHODS)
    models = list(models or CANONICAL_MODELS)

    found: Dict[str, dict] = {}
    for cohort in cohorts:
        for method in methods:
            for model in models:
                if not is_combination_valid(method, model):
                    continue
                d = feature_dir(cohort, method, model)
                if not d.is_dir():
                    continue
                n = sum(1 for f in os.listdir(d) if f.endswith(".pt"))
                if n == 0:
                    continue
                found[f"{cohort}|{method}|{model}"] = {
                    "cohort": cohort,
                    "method": method,
                    "model": model,
                    "path": str(d),
                    "n_files": n,
                    "expected_dim": expected_dim(method, model),
                }

    payload = {
        "machine": MACHINE,
        "scanned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_root": str(BASE_ROOT),
        "n_combinations": len(found),
        "combinations": found,
    }
    out = write_to or (SLIDE_CLS_ROOT / "available_features.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def git_commit(short: bool = True) -> str:
    """Current git commit, for stamping result files. ``'unknown'`` on failure."""
    try:
        args = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"]
        if not short:
            args = ["git", "rev-parse", "HEAD"]
        out = subprocess.run(
            args, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=15
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


#: Code whose state decides what a results run computes. Result files, documents
#: and figures are left out on purpose: a run rewrites those by design.
_CODE_PATHS = ("slide_classification/runners", "slide_classification/eval_patch_features",
               "slide_classification/config", "slide_classification/data_layer.py")


def code_is_dirty() -> Optional[bool]:
    """True when the result-producing code has uncommitted changes.

    ``git_commit`` in a stamp identifies the code only when this is False. The
    results of 2026-08-10..09-03 all carry a commit that did not yet contain the
    code that produced them; recording this flag makes that visible in the file
    instead of leaving it to be reconstructed from logs. ``None`` on failure.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", *_CODE_PATHS],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=15)
        return bool(out.stdout.strip()) if out.returncode == 0 else None
    except Exception:
        return None


def library_versions() -> Dict[str, str]:
    """Interpreter and the libraries a result can depend on."""
    import platform
    from importlib import metadata
    out = {"python": platform.python_version()}
    for label, dist in (("sklearn", "scikit-learn"), ("torch", "torch"),
                        ("numpy", "numpy"), ("pandas", "pandas")):
        try:
            out[label] = metadata.version(dist)
        except Exception:
            out[label] = "unknown"
    return out


def run_stamp(**extra) -> Dict[str, object]:
    """Provenance block attached to every result file (1c.2 item 4)."""
    stamp = {
        "machine": MACHINE,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "git_code_dirty": code_is_dirty(),
        "versions": library_versions(),
    }
    stamp.update(extra)
    return stamp


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    print(f"MACHINE   = {MACHINE}")
    print(f"BASE_ROOT = {BASE_ROOT}")
    print(f"REPO_ROOT = {REPO_ROOT}")
    payload = scan_available_features()
    print(f"\n{payload['n_combinations']} combinations present on this machine:")
    for key, info in sorted(payload["combinations"].items()):
        print(f"  {key:<62} {info['n_files']:>5} files  dim={info['expected_dim']}")
