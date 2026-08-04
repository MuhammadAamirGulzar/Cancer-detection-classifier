"""Configuration package for the MSI slide-classification pipeline.

See :mod:`slide_classification.config.paths` for the single source of truth for
every filesystem root used by the pipeline.
"""

from .paths import (  # noqa: F401
    MACHINE,
    ROOTS,
    REPO_ROOT,
    SLIDE_CLS_ROOT,
    COHORTS,
    CANONICAL_MODELS,
    AGGREGATION_METHODS,
    MODEL_BASE_DIMS,
    AGG_MULTIPLIERS,
    TASK_PREFIXES,
    expected_dim,
    feature_dir,
    labels_path,
    folds_path,
    results_root,
    cache_dir,
    is_combination_valid,
    scan_available_features,
    git_commit,
    run_stamp,
)
