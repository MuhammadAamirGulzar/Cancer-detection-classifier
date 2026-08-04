"""Experiment runners built on the single data layer (work order Task 1.0).

Every runner imports :mod:`data_layer`; no runner defines its own dataset class.

  * :mod:`runners.cv_runner`   - K-fold CV (TCGA-CV, SurGen-CV)
  * :mod:`runners.classifiers` - classifier dispatch and hyperparameter grids
  * :mod:`runners.results_io`  - merge-friendly, provenance-stamped result files
  * :mod:`runners.thresholds`  - tau_TCGA decision-threshold policy (section 1b)
  * :mod:`runners.runlog`      - logging, resumable progress, skip tracking
"""
