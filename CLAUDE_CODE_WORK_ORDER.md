# Work Order — MSI Prediction Pipeline (KSA_project2)

**For:** Claude Code, running inside `D:\Aamir Gulzar\KSA_project2`
**Owner:** maamir
**Date:** 4 August 2026
**Standard:** research-publication. Assume every number will be scrutinised by a reviewer. Do not "fix" anything by making a number look better; fix mechanisms and re-run.

---

## START HERE — unattended run

You are running with the owner unavailable for extended periods. **Work continuously through the phases in order. Do not stop to ask questions.**

```
1. Read Sections 0, 1, 1b, 1c in full before writing any code.     <- all decisions are pre-made there
2. Work phases in order: 0 -> 1 -> 2 -> 3 -> 4.  Phase 5 is OUT OF SCOPE.
3. Within a phase, follow the task order given in "Suggested execution order".
4. After EVERY task: run its acceptance check, paste the output, git commit.
5. After EVERY phase: write reports/PHASE_<n>_REPORT.md.
6. Update progress.json continuously so an interrupted run resumes cleanly.
```

**The only hard stop is Task 0.2** (the owner must revoke Hugging Face tokens). Write `TOKENS_TO_REVOKE.md`, flag it, and **carry on** — the code change is safe to make before revocation.

**If you hit an undocumented decision:** re-read this document first. If the answer genuinely is not here, append it to `OPEN_QUESTIONS.md`, choose the most conservative option, log your reasoning, and continue. Never idle.

**Halt only for:** a shape mismatch, a label mismatch, a failed leakage assertion, or a metric-parity check that fails after refactoring. Those are correctness failures and must not be worked around.

---

## 0. Read this first

### 0.1 What the project is

Prediction of **microsatellite instability (MSI-H vs non-MSI-H)** from H&E whole-slide images in colorectal cancer. Patches → foundation-model embeddings → **slide-level aggregation** → lightweight classifier. The experimental variable is the **aggregation strategy**, not the encoder.

Three cohorts: **TCGA** (primary), **PAIP** (external), **SurGen** (second external, largely unused so far).

### 0.2 Rules of engagement

1. **Never edit a results file by hand.** Results are only ever produced by re-running a script.
2. **Never delete an existing results folder.** Rename to `<name>_ARCHIVED_<date>` instead. Old numbers must remain auditable.
3. **Every task below has an acceptance check.** Do not mark a task done until its check passes and you have pasted the output.
4. **Work one phase at a time.** Do not start Phase 2 until Phase 1 is committed.
5. If a task's premise turns out to be wrong when you read the code, **stop and report** rather than improvising.

### 0.3 Verified ground truth — do not re-derive, and flag if you observe otherwise

| Fact | Value |
|---|---|
| TCGA slides with labels | 416 (61 MSI-H / 355 non-MSI-H) |
| TCGA slides with **both** label and features | **413** (60 MSI-H / 353 non-MSI-H) |
| TCGA labelled-but-no-features | `TCGA-AD-6895_MSIH`, `TCGA-AD-6899_nonMSIH`, `TCGA-CM-6680_nonMSIH` |
| TCGA feature-but-no-label (harmless) | `TCGA-D5-7000_nonMSIH`, `TCGA-EI-6917_nonMSIH`, `TCGA-F5-6814_nonMSIH`, `TCGA-T9-A92H_nonMSIH` |
| TCGA patients | 416 unique, **one slide per patient**, no fold leakage — verified clean |
| TCGA fold sizes (folds 1–4) | 103 / 98 / 112 / 103; MSI-H per fold 15 / 13 / 20 / 13 |
| PAIP slides with labels | 78 (19 MSI-H / 59 non-MSI-H) |
| PAIP slides with features | **73** (17 MSI-H / 56 non-MSI-H) |
| PAIP labelled-but-no-features | `training_data_19_nonMSIH`, `training_data_30_MSIH`, `training_data_41_nonMSIH`, `training_data_42_MSIH`, `training_data_46_nonMSIH` |
| PAIP official train split | 47 slides (`paip_47slides.csv`) → **42 have features** (10 MSI-H) |
| PAIP official test split | 31 slides (`paip_31slides_labels.csv`) → **31 have features** (7 MSI-H), complete |
| SurGen slides in label file | 1020 (564 label-0, 60 label-1, 396 label −1 = unknown) |
| SurGen usable slides | **624** from **554 unique cases**; 70 cases contribute 2 slides each |
| SurGen aggregated features present **locally** | **622** slides, and **only for `conch1-5` and `virchow2`** |
| Foundation model dims | H-Optimus-1 1536 · UNI2 1536 · Conch1_5 768 · Virchow2 2560 · ConchV1 512 · PRISM 1280 · TITAN 768 |
| Aggregation row counts | Averaging 1 · Tissue_Type_Clustering 9 · Caption_based 14 · Caption_based_15 15 · TITAN 1 · PRISM 1 |

---

## 1. Canonical experiment taxonomy — USE THESE NAMES EVERYWHERE

The current naming is inconsistent (`TCGA-IV` appears where cross-validation is meant, `paip_kfolds_71.csv` has 78 rows and no folds). Adopt this vocabulary in **code, folder names, Excel sheet names, plot titles and the report**, and change nothing else about naming.

| Canonical name | Train on | Test on | Protocol | Status |
|---|---|---|---|---|
| **TCGA-CV** | TCGA folds | TCGA held-out fold | 4-fold CV, averaged | Exists — rename only |
| **PAIP-IV** | PAIP official train (42) | PAIP official test (31) | Single fixed split | **MISSING — build** |
| **PAIP-EV** | **TCGA-FULL** (see Task 3.0) | All 73 PAIP slides | External | Exists but **must be rebuilt** on new artifact |
| **SurGen-CV** | SurGen folds | SurGen held-out fold | 4-fold CV, **case-grouped** | Exists but **leaks** — see Task 2.1 |
| **SurGen-EV** | **TCGA-FULL** (see Task 3.0) | All SurGen slides | External | **MISSING — build** |

**Design principle:** use the data provider's own split when one exists (PAIP), use K-fold when none does (TCGA, SurGen). A `PAIP-CV` experiment previously existed and has been **retired** — see Task 2.2.

**Definitions to put in the report, verbatim:**

- **CV (cross-validation)** — the cohort is split into K folds; each fold serves as test once; reported metric is the mean across folds.
- **IV (internal validation)** — a single train/test split defined by the dataset provider; no averaging.
- **EV (external validation)** — trained entirely on cohort A, tested on the whole of cohort B, with zero cohort-B data in training. **The TCGA-trained artifact is defined in Task 3.0.** Three variants are reported side by side so they can be compared directly: TCGA-FULL (primary), 4-fold ensemble, and the legacy 4-fold-average convention. None is discarded.

---

## 1b. Decision-threshold policy — owner-confirmed

**Problem this solves.** On PAIP-EV, logistic regression produced the confusion matrix `[[56, 0], [15, 2]]` — non-MSI-H for almost every slide — despite an AUROC of 0.862. The ranking transferred; the operating point did not. A TCGA-calibrated cut-point applied to a cohort with different prevalence collapses balanced accuracy. This is a calibration artifact, and if left unaddressed a reviewer will read it as the method failing to generalise.

**Policy:**

1. **AUROC is the primary metric in every experiment.** It is threshold-free and therefore immune to this failure. Lead with it in every summary table and every radar plot.
2. **Internal experiments** (TCGA-CV, PAIP-IV, SurGen-CV) report at the default threshold **0.5**. Train and test distributions are matched, so no correction is warranted.
3. **External experiments** (PAIP-EV, SurGen-EV) report at a threshold **τ_TCGA fixed on TCGA and applied unchanged**. Never tune a threshold on PAIP or SurGen — that would silently convert external validation into internal validation.
4. **How to compute τ_TCGA:** pool the **out-of-fold** predicted probabilities from TCGA-CV (each of the 413 slides is predicted exactly once, by a model that did not see it), then set

   ```
   τ_TCGA = argmax_τ ( TPR(τ) − FPR(τ) )     # Youden's J
   ```

   This is a genuinely out-of-sample TCGA estimate and involves no external data. Compute one τ per (aggregation method × foundation model × classifier) and persist it to `thresholds_TCGA.json` alongside the model artifacts.
5. **Label every externally-reported operating-point metric** in the table header as `@τ_TCGA`, with the τ value recorded. A reviewer must be able to see the threshold was not fitted to the test cohort.
6. Keep the raw `0.5`-threshold numbers in the per-run CSV for audit. They are not headline figures, but they must remain reproducible.

> **Existing special case — decision already made, do not ask.** The random forest carries a hardcoded `prediction_threshold = 0.3` with `class_weight = {0: 1, 1: 10}`. **Ruling: τ_TCGA replaces the 0.3 for external experiments, and RF keeps `class_weight` unchanged.** Rationale: two competing threshold mechanisms cannot both be defended in a paper, and τ_TCGA is derived out-of-sample whereas 0.3 was chosen by hand. Additionally record `RF @ 0.3` in the audit CSV so the old behaviour stays reproducible. Do not pause for confirmation.

---

## 1c. Engineering requirements — apply to every task below

These are **cross-cutting**. Implement them in Phase 1 as part of the refactor, not bolted on later.

### 1c.1 Performance — the pipeline is currently doing ~20x more disk I/O than it needs to

Concrete inefficiencies found in the existing code. Fix all of them; they compound.

| # | Problem | Where | Fix |
|---|---|---|---|
| 1 | **Every `.pt` file is re-loaded from disk ~20 times per (method, model).** `run_k_fold_cross_validation` is called once per classifier (5), and each of its 4 fold iterations builds 3 fresh `WSIDataset` objects, each re-running `torch.load` over the directory. | `WSIDataset._load_data`, `run_k_fold_cross_validation` | **Load once, index many.** Build a single `(N, D)` float32 tensor + label vector + ID list per (cohort, method, model), hold it in RAM, and derive train/val/test by **index slicing**. Sizes are trivial: TCGA Virchow2 caption-15 is 413 × 38400 × 4B ≈ **63 MB**; SurGen ≈ 95 MB. Everything fits comfortably. |
| 2 | **O(n²) label lookup.** `if wsi_id in folds_df['WSI_Id'].values` then `folds_df.loc[folds_df['WSI_Id'] == wsi_id, ...]` — two full linear scans of the DataFrame per file. | `WSIDataset._load_data` (all ~6 copies) | Build `dict(zip(df.WSI_Id, df.label))` **once**, outside the loop. O(1) lookup. |
| 3 | **`DataLoader` is used and then immediately undone.** Data is batched at `batch_size=4`, then `get_feats_labels` iterates the whole loader and `torch.cat`s it back into one tensor. The batching accomplishes nothing but overhead. | `train_and_evaluate` | Delete the `DataLoader` layer entirely for these classifiers. Pass tensors directly. The ANN is already full-batch. |
| 4 | **`write_data_in_excel` is quadratic.** Every call opens the workbook, reads **all** sheets into DataFrames, then rewrites **all** of them. Called once per experiment. | `utility.py` | Accumulate results in memory; write the workbook **once** at the end of a run. Keep the existing function for backwards compatibility but stop calling it in loops. |
| 5 | Redundant re-computation across seeds/variants | Task 3.0 | Cache the assembled matrix to `cache/<cohort>_<method>_<model>.npz` (features, labels, ids). Subsequent runs load **one** file instead of hundreds. Invalidate on source-directory mtime. |
| 6 | Single-threaded sklearn | `knn.py`, `r_forest.py` | Set `n_jobs=-1` on `KNeighborsClassifier` and `RandomForestClassifier`. |
| 7 | ANN on CPU | `ann.py` | Use CUDA when available; the model is tiny and full-batch, so this is nearly free. Keep a CPU fallback. |

**Parallelism.** `(method × model)` combinations are independent. Parallelise them with a process pool for the sklearn classifiers (`lin`, `knn`, `proto`, `rf`). Keep **ANN runs sequential** to avoid GPU contention. Make the worker count a config value, defaulting to `min(8, cpu_count() - 2)`.

**Acceptance:** benchmark one `(method, model)` end-to-end before and after, and report wall-clock for both. Target is a **≥5x** reduction. If the speedup is under 2x, stop and report — something in the refactor did not land.

> Correctness outranks speed. Every optimisation above must be metric-neutral. After refactoring, re-run **one** configuration under the old and new code paths and confirm the metrics match to ~1e-6. Paste that comparison. If they differ, the refactor is wrong — do not proceed.

### 1c.2 Cross-machine operation — features are split across two machines

Aggregated features for some foundation models exist only on the remote server (`/media/dp-psau/Datum/Aamir/Azfaar`). Confirmed locally present: SurGen has **only `conch1-5` and `virchow2`**. The **same code must run unmodified on both machines**, and results must merge cleanly afterwards.

1. **Single path configuration.** Create `slide_classification/config/paths.py` resolving all roots from one `MACHINE` environment variable (`local` | `server`), with a dict per machine. **Remove every hardcoded `D:\Aamir Gulzar\...` and `/media/dp-psau/...` path** from every script and notebook. There are many; find them all.
2. **Auto-detect availability.** On startup, scan the feature tree and write `available_features.json` listing every `(cohort, method, model)` combination physically present, with its file count. Never assume — always scan.
3. **Skip, never fail.** If a combination is unavailable, log it to `skipped_combinations.csv` with a reason and continue. A missing model on one machine must never abort a run.
4. **Merge-friendly output.** Write **one result file per `(experiment, method, model, variant)`** rather than one large shared workbook. Merging across machines then becomes a file copy. Stamp every result file with `machine`, `timestamp`, `git_commit` and `n_test`.
5. **Provide `merge_results.py`.** Scans one or more result trees, detects duplicate `(experiment, method, model, variant)` keys, and on conflict keeps the newer `git_commit` while logging the collision. It then builds the consolidated workbook. This runs after results from both machines are in one place.
6. **Report coverage on every run:** which combinations ran here, which were skipped, and which are still missing globally.

**Acceptance:** `grep -rn "D:\\\\Aamir\|/media/dp-psau" --include=*.py --include=*.ipynb` returns nothing outside `config/paths.py`. A run on the local machine completes with SurGen H-Optimus-1/UNI2/ConchV1 listed as skipped, not crashed.

### 1c.3 Unattended operation — this will run in agent mode with the owner unavailable

Assume **no human is watching**. Design accordingly.

1. **Only one hard stop exists in this entire document: Task 0.2**, which needs the owner to revoke Hugging Face tokens. Reach that point, write the file list to `TOKENS_TO_REVOKE.md`, report it, and **continue with Phase 1** — the token replacement code change is safe to make before revocation happens.
2. **Every other decision is pre-made.** If you find yourself wanting to ask a question, first re-read this document — the answer is probably here. If it genuinely is not, record it in `OPEN_QUESTIONS.md`, pick the most conservative option, log the choice with your reasoning, and keep going. Never idle waiting for input.
3. **Resumability.** Maintain `progress.json` keyed by `(phase, task, experiment, method, model, variant, seed)`. Mark entries complete as they finish. On startup, skip completed entries unless `--force` is passed. Long runs must survive interruption without redoing finished work.
4. **Logging.** Append to `logs/run_<timestamp>.log`: timestamps, the combination in progress, wall-clock per combination, and full tracebacks. Never swallow an exception silently — the existing bare `except` blocks are part of why the missing-slide problem went unnoticed.
5. **Fail-fast on correctness, fail-soft on availability.** A shape mismatch, a label mismatch or a leakage-assertion failure must **halt** the run loudly. A missing feature directory must only skip.
6. **Phase reports.** At the end of each phase write `reports/PHASE_<n>_REPORT.md` containing: what changed, what was run, before/after metrics where applicable, acceptance-check output, anything skipped, and anything added to `OPEN_QUESTIONS.md`.
7. **Commit after every completed task**, message format `phase<N>.<task>: <summary>`. If something later proves wrong, the owner can bisect.
8. **Never delete.** Archive with a dated suffix. This is repeated because it matters.

**Acceptance:** kill a run mid-Phase-3, restart it, and confirm from the log that it resumes rather than restarting.

---

## PHASE 0 — Protect the work (do first, ~1.5 hours)

Nothing else in this document is safe to start until Phase 0 is done.

### Task 0.1 — Put the project under version control

Five of eight top-level directories are entirely untracked, including all of `slide_aggregation/` and `slide_classification/`. The last commit predates essentially all current work.

1. Write `.gitignore` at `Cancer-detection-classifier/.gitignore` excluding: `*.pt`, `*.pth`, `*.pkl`, `*.png` under patch/feature dirs, `__pycache__/`, `.ipynb_checkpoints/`, `_otf_tmp/`, `*.svs`, `*.czi`, `.env`.
   - **Do keep** `*.xlsx` and `*.csv` result files tracked — they are the record of what was run.
2. `git add -A && git commit` with message `checkpoint: full project state prior to remediation`.
3. Confirm result spreadsheets are tracked.

**Acceptance:** `git status --short` returns no `??` lines for `.py`, `.ipynb`, `.xlsx`, `.csv`. `git ls-files slide_classification | wc -l` > 0.

### Task 0.2 — Purge hardcoded credentials

**21 files** contain **4 distinct** Hugging Face tokens in plaintext, including `Slide_Classification.ipynb`, `Slide_Classification_TTC_exp.ipynb`, `Slide_Classification_best_k_labels_exp_updated.ipynb`, all `Complete_Pipeline/surgen_processing_*.py`, and several `Analysis_and_Visualization` notebooks.

1. Report the full file list to the user; **the user must revoke all four tokens on huggingface.co before you proceed.**
2. Replace every `login("hf_...")` with:
   ```python
   import os
   from dotenv import load_dotenv
   from huggingface_hub import login
   load_dotenv()
   login(os.environ["HF_TOKEN"])
   ```
   `.env` already exists at `Cancer-detection-classifier/.env` with an `HF_TOKEN` entry and is already gitignored.
3. Clear notebook outputs on any notebook that echoed a token.

**Acceptance:** `grep -r "hf_[A-Za-z0-9]\{30,\}" --include=*.py --include=*.ipynb Cancer-detection-classifier/` returns nothing.

> **Note:** the tokens remain in Git history from Task 0.1's commit. That is acceptable *provided the tokens are revoked*. Do not attempt a history rewrite.

---

## PHASE 1 — Correctness fixes (do before re-running anything, ~1.5 days)

These change *numbers*. Every result produced after Phase 1 supersedes the equivalent result produced before it.

### Task 1.0 — [P0] Refactor the data layer first

Implement **Section 1c.1 items 1–4** and **Section 1c.2 items 1–3** before touching the classifier logic. Every subsequent task edits this code, so doing it first avoids applying the same fix six times across the duplicated `WSIDataset` copies.

Consolidate the ~6 duplicated `WSIDataset` implementations into **one** module, `slide_classification/data_layer.py`, exposing:

```python
load_cohort(cohort, method, model)   -> (feats: Tensor[N,D], labels: Tensor[N], ids: list[str])
build_splits(ids, fold_map)          -> dict of index arrays
```

Every runner imports from it. No script defines its own dataset class afterwards.

**Acceptance:** exactly one `class WSIDataset` (or equivalent) remains in the active codebase; the metric-parity check from Section 1c.1 passes.

### Task 1.1 — [P0] Stop selecting ANN hyperparameters on the test fold

**Where:** `slide_classification/Slide_Classification.ipynb` cells 6 and 8; `slide_classification/eval_patch_features/ann.py`; and the same pattern in `windows_slide_classification_surgen.py`, `linux_slide_classification_surgen.py`, `slide_classification_surgen.py`, `Slide_Classification_TTC_exp.ipynb`, `Slide_Classification_best_k_labels_exp_updated.ipynb`, `Tissue_Type_Combinations_Exp/TTC_combination_experiment.py`.

**The bug:** the ANN grid tries 4 configurations (`hidden_dim1` ∈ {128,256} × `hidden_dim2` ∈ {64,128}). The winner is chosen by `eval_metrics['ann_macro_f1']`, but `eval_ANN` computes metrics **only on the test set**. So the best of four is selected using test performance and then reported as a test result. This is optimistic bias.

**Confirm the scope before changing anything:** `lin` (`C=[10], max_iter=[300]`), `knn` (`k=[3]`), `proto` (no grid) and `rf` (single combination) all have single-point grids, so no selection occurs and **their metrics are unbiased**. Verify this is still true in each file; if any of them has gained a real grid, apply the same fix.

**Fix:**
1. In `ann.py`, add a `valid_metrics` return: after `classifier.fit(...)`, evaluate on `valid_feats`/`valid_labels` and return metrics under a `val_` prefix alongside the test metrics.
2. In every calling loop, change the selection criterion from `eval_metrics['ann_macro_f1']` to the validation macro-F1.
3. Add an inline comment marking this as a deliberate correction, with the date.

**Acceptance:** grep every driver file; no selection comparison reads a test-set metric. Print, for one fold, the 4 validation scores and the selected configuration, and show the selected configuration is the argmax of the *validation* scores.

### Task 1.2 — [P0] Save the ANN checkpoint that was actually selected

**Where:** `eval_patch_features/ann.py`, and the calling loops.

**The bug:** `torch.save` is called **inside** the hyperparameter loop, to `fold{fold}_trained_ann_model_{input_dim}.pth`. `input_dim` is identical across the 4 configurations, so all four write to the same path and the file on disk is the **last** configuration tried, not the selected one. Verified: every saved TCGA ANN checkpoint has `hidden_dim1=256, hidden_dim2=128` — the final grid entry.

Consequence: **PAIP-EV runs a different network than the one whose TCGA-CV metrics are printed beside it.**

**Fix:**
1. Remove the `torch.save` from inside `eval_ANN`; return the model object instead.
2. Save once, after the grid completes, using the selected configuration.
3. Change the filename to `fold{fold}_ann_{input_dim}_{h1}_{h2}.pth`, and write a sibling `fold{fold}_ann_config.json` recording `input_dim`, `h1`, `h2`, `max_iter` and the selection metric.
4. Update every loader to read the config JSON. Keep the existing state-dict-shape-inference fallback in `Slide_Classification.ipynb` cell 14 as a safety net.

**Acceptance:** for one configuration, the saved checkpoint's `0.weight` and `4.weight` shapes match the `config.json`, and both match the configuration reported as selected.

### Task 1.3 — [P1] Remove the double softmax in the ANN

**Where:** `eval_patch_features/ann.py`, class `ANNBinaryClassifier`.

**The bug:** the network ends in `nn.Softmax(dim=1)` and is trained with `nn.CrossEntropyLoss`, which applies log-softmax internally and expects raw logits. Softmax is applied twice, flattening gradients and likely leaving the network underfitted. **ANN results are probably a floor, not a ceiling.**

**Fix:**
1. Remove the final `nn.Softmax` from the training graph.
2. Apply `torch.softmax` explicitly in `predict_proba` only.
3. Update **every** inference path that assumes the loaded model already outputs probabilities — including `Slide_Classification.ipynb` cell 14 (comment reads `# Model already has Softmax, so out is already probabilities`) and `external_validation_script.py` around line 401.

**Because this changes the architecture, every ANN checkpoint must be regenerated and every ANN result re-run.** Sequence it with Phase 3 so ANN is trained once, not twice.

**Acceptance:** train one fold before and after; report both balanced accuracy and AUROC. Expect improvement or parity — if results degrade materially, stop and report rather than proceeding.

### Task 1.4 — [P1] Fail loudly when a labelled slide has no features

**Where:** the `WSIDataset._load_data` method — it is duplicated in ~6 places; fix all of them.

**The bug:** the loader iterates files on disk and skips anything not in `fold_ids`. A labelled slide with no feature file silently never appears. No warning, no count check. This is why TCGA is 413 and not 416, and PAIP 73 and not 78 — correct results, wrong reported N.

**Fix:** after loading, compare `len(self.data)` against the number of requested `fold_ids`. If they differ, print a `[WARN]` listing the missing IDs, and write a per-run `missing_slides.csv` into the Output folder.

Additionally: in the `BRAF`/`KRAS`/`TP53`/`CIMP` branches, `label` is left **undefined** when `wsi_id` is not found in `folds_df`, causing a `NameError` swallowed by the bare `except`. Initialise `label = None` and skip explicitly with a warning. This is latent now but will bite in Phase 5.

**Acceptance:** running TCGA-CV prints a warning naming exactly the 3 known missing slides and reports `N=413`.

### Task 1.5 — [P2] Retire the stale external-validation script

`slide_classification/external_validation_script.py` hardcodes the ANN architecture as `hidden_dim=256, hidden_dim2=64`. Saved checkpoints are `256/128`, so it would fail with a shape mismatch. The equivalent cell in `Slide_Classification.ipynb` infers dimensions from the state dict and is correct — **it is the notebook that produced the current PAIP-EV results.**

Move the script to `slide_classification/misc/` with a header comment stating it is superseded and by what. Do the same for `dataloader.py` (imported nowhere, stale Windows paths, derives labels from filename substrings) and `plot_surgen_ttc_results.py` (reads a remote Linux path that does not exist locally and expects filenames from a different script).

**Acceptance:** no active code path imports any of the three.

---

## PHASE 2 — Fix the validation protocols (~1 day)

### Task 2.1 — [P0] Rebuild SurGen folds at case level

**Where:** `build_runtime_folds` in `windows_slide_classification_surgen.py` (and the linux/older variants).

**The bug:** folds are built by shuffling **slide IDs** and dealing them round-robin. But 70 of the 554 labelled cases contribute **two slides each**. Sibling slides land in different folds roughly 3 times in 4, putting the same patient in train and test. Two sections from one tumour are far more alike than two different tumours, so this inflates every SurGen number.

TCGA is immune because it has exactly one slide per patient — this code was safe there and was carried over unchanged.

**Fix:**
1. Derive `case_id` by stripping the trailing `_NN` from the slide ID (e.g. `SR1482_40X_HE_T004_01` → `SR1482_40X_HE_T004`). **Verify this regex against the real IDs before relying on it** and report how many cases it produces — expect **554**.
2. Assign **whole cases** to folds, stratified on the case-level label. Verified: no case has conflicting labels across its slides, so the case label is unambiguous.
3. Keep `seed=42` and the exclusion of the 396 label −1 slides (that part is already correct).
4. Print per-fold slide count, case count and MSI-H count.
5. Assert no `case_id` appears in more than one fold; raise if violated.

**Acceptance:** the assertion passes, and per-fold case counts sum to 554.

> **Expect SurGen-CV numbers to drop after this.** That is the correction working, not a regression. Archive the old results per rule 2 and report the before/after delta explicitly — a reviewer will ask.

### Task 2.2 — [P0] Retire PAIP-CV

**Decision (owner, confirmed):** PAIP-CV is dropped from the study. Only **PAIP-IV** and **PAIP-EV** are reported for the PAIP cohort. Do not repair it, do not re-run it.

**Why it is being retired — record this in the report, because a reviewer comparing against the previous result set will ask.**

The existing PAIP "cross-validation" in `Slide_Classification.ipynb` cell 8 builds folds by **sequential slicing of CSV order** — no shuffle, no stratification:

```python
fold_size = num_total_slides // num_folds
fold_ids = valid_wsi_ids[start_idx: start_idx + fold_size]
```

The CSV is ordered `training_data_01 … training_data_47, validation_data_01 …`, so the folds track PAIP's own acquisition split rather than a random partition:

| Fold | n | MSI-H | from `training_data_*` | from `validation_data_*` |
|---|---|---|---|---|
| 1 | 18 | 4 | 18 | 0 |
| 2 | 18 | 4 | 18 | 0 |
| 3 | 18 | 5 | 6 | 12 |
| 4 | 19 | 4 | 0 | 19 |

Folds 1 and 2 are entirely the provider's training set; fold 4 is entirely its validation set. Class balance came out even **by luck of ordering**, not by design, and is not reproducible under any reordering.

**But the deciding argument is not the bug — it is that the experiment is strictly dominated.** Even with fold construction repaired, PAIP-CV would train on ~36 slides (2 of 4 folds) and test on 18 with ~4 positives. PAIP-IV trains on **42** and tests on **31**. PAIP-CV therefore trains on *less* data and tests on *smaller* sets while answering the same question. An AUROC computed on 14 negatives and 4 positives cannot separate models; the per-fold confusion matrices already in `best_of_all_exps_metric.xlsx` (e.g. `[[14, 0], [1, 3]]`) show this directly.

**Actions:**
1. Rename `slide_classification/PAIP_Results/` → `slide_classification/PAIP_Results_ARCHIVED_<date>/`. **Do not delete** — those numbers may already have been circulated.
2. Remove the PAIP k-fold block from `Slide_Classification.ipynb` cell 8, replaced by the PAIP-IV runner from Task 3.1. Leave a markdown cell recording the retirement and the reason.
3. Drop the `PAIP` sheet from the consolidated workbook and the PAIP radar plots keyed to it (Phase 4 handles this).

**If uncertainty estimates on PAIP are wanted**, use bootstrap confidence intervals over the 31 PAIP-IV test slides (resample with replacement, 1000 iterations, report 95% CI on BalAcc and AUROC). That is the correct tool at this sample size — not K-fold. Implement it as part of Task 3.1.

**Acceptance:** no active code path builds PAIP folds; `PAIP_Results_ARCHIVED_<date>/` exists and is intact.

### Task 2.3 — [P2] Rename the misleading fold/label files

- `slide_classification/paip_kfolds_71.csv` — contains **78 rows** and **no fold column**. It is a label table. Rename to `paip_78_labels.csv` and update all references.
- `paip_data/labels/paip_78slides_labels.csv` — **malformed**. The `fold` column contains a mix of fold indices and label strings: `{'nonMSIH': 24, '1': 10, '0': 10, '2': 9, '3': 9, '4': 9, 'MSIH': 7}`. The 31 `validation_data_*` rows have their columns shifted, so their label has landed in the `fold` column. Regenerate this file cleanly from `paip_47slides.csv` + `paip_31slides_labels.csv`, with an explicit `split` column (`train`/`test`) rather than an overloaded `fold`.

**Acceptance:** the regenerated file has 78 rows, a clean `split` column with 47/31, and label counts matching 19 MSI-H / 59 non-MSI-H.

---

## PHASE 3 — Missing experiments (~3–4 days)

Run these **only after Phases 1 and 2 are committed**, so every number comes from corrected code.

### Task 3.0 — [P0] Build the TCGA-FULL artifact — **do this before 3.1 and 3.2**

**Owner decision:** external validation uses a **single model retrained on all TCGA** as the primary artifact, with a **4-fold probability ensemble** reported as a robustness check.

**Why a new artifact is needed — and why the old one is kept.** PAIP-EV currently applies the 4 TCGA fold-models separately and averages their four metric sets. But each fold-model was trained on only 2 of 4 folds:

| Test fold | Val fold | Train n | Train MSI-H |
|---|---|---|---|
| 1 | 2 | 213 | 32 |
| 2 | 3 | 204 | 27 |
| 3 | 4 | 200 | 28 |
| 4 | 1 | 209 | 33 |

So the existing external validation demonstrates transfer from models trained on **~208 slides (≈28 MSI-H)** when **413 slides (60 MSI-H)** are available. Training data and positive count both roughly double under TCGA-FULL.

**Owner decision: keep the existing 2-fold-trained results as a third reported variant.** They are not retired. The comparison "half the training data vs all of it" is itself a result worth showing — it quantifies how much the extra ~200 slides buy, which is a question a reviewer may well ask. Preserve the existing `TCGA_PAIP_EV_Results/` numbers and re-run the same convention wherever coverage is being extended, so all three variants exist for every combination.

#### 3.0a — Primary artifact: `TCGA-FULL`

1. **Fix hyperparameters first, from TCGA-CV only.** For each (aggregation method × foundation model × classifier), take the configuration selected most often across the 4 folds; break ties by best mean **validation** score. Persist to `tcga_full_hparams.json`. **No external data may influence this step.**
2. **Train one final model per combination on all 413 TCGA slides.**
   - `lin`, `knn`, `proto`, `rf`: train on all **413**.
   - `ann`: needs a held-out set for early stopping — carve a **stratified 15%** (seed-fixed) from the 413, train on ~351, early-stop on ~62. Document this asymmetry in the report; do not hide it.
3. **Repeat over 5 seeds** (42–46) to obtain a variance estimate. Report **mean ± SD** across seeds. Note that `lin` and `knn` are effectively deterministic here, so their SD will be ~0 — that is expected, not a bug.
4. Save to `slide_classification/TCGA_FULL_Models/<method>/<model>/1-MSIH/seed<NN>/`, each with its `config.json` (per Task 1.2) and the `thresholds_TCGA.json` from Section 1b.

#### 3.0b — Robustness check: 4-fold ensemble

Average the **predicted probabilities** of the 4 existing fold-models into one prediction per slide, then apply τ_TCGA once. This is one prediction per slide, not four metric sets averaged. Report alongside TCGA-FULL as a secondary row.

#### 3.0c — Legacy variant: 4-fold-average (keep)

The existing convention: apply each of the 4 fold-models separately to the full external cohort, average the four metric sets. Retain it and extend it to any new combination, so all three variants are available everywhere.

#### 3.0d — Reporting

Every EV summary table carries **three** rows, clearly labelled, in this order:

| Variant | Train n | Role |
|---|---|---|
| `TCGA-FULL (mean ± SD, 5 seeds)` | 413 (ANN 351+62) | **Primary** |
| `TCGA 4-fold ensemble` | 4 × ~208, pooled probabilities | Robustness check |
| `TCGA 4-fold average (legacy)` | 4 × ~208, averaged metrics | Comparison / continuity with earlier results |

Add a short `Δ(FULL − legacy)` column on BalAcc and AUROC. That delta is the answer to "how much did doubling the training data buy?" and is worth reporting explicitly. If TCGA-FULL does **not** beat the legacy variant, report that honestly — it is an informative negative result, not a failure.

**Acceptance:** `tcga_full_hparams.json` exists and every value is traceable to a TCGA-CV validation score. No PAIP or SurGen file is read anywhere in this task — verify by grep. Training-set size printed as 413 (or 351 + 62 for the ANN).

### Task 3.1 — [P0] Build PAIP-IV

The provider's official split. Data is already present:

- Train: `paip_data/labels/paip_47slides.csv` → 47 IDs, of which **42 have features** (10 MSI-H)
- Test: `paip_data/labels/paip_31slides_labels.csv` → 31 IDs, **all 31 have features** (7 MSI-H)

Build a single-split runner (no fold loop, no averaging). Carve a stratified validation subset out of the 42 training slides for ANN early stopping — do **not** touch the 31 test slides for any tuning decision. Sweep all 6 aggregation methods × 5 models × 5 classifiers.

Output to `slide_classification/PAIP_IV_Results/<method>/<model>/1-MSIH/{Output,models}/`.

Additionally implement the **bootstrap confidence intervals** described in Task 2.2 (1000 resamples over the 31 test slides; 95% CI on BalAcc and AUROC) and write them alongside the point estimates.

**Acceptance:** test-set N is exactly 31 in every confusion matrix. Report the 42/31 counts prominently — they are not 47/31, and the report must say so. Every PAIP-IV metric carries a CI.

### Task 3.2 — [P0] Build SurGen-EV — runs partially on this machine, by design

Apply all three Task 3.0 variants (TCGA-FULL, 4-fold ensemble, 4-fold average) to all SurGen slides, at τ_TCGA. Directly analogous to PAIP-EV, reusing the corrected `Slide_Classification.ipynb` cell 14 logic.

**Expected partial coverage — this is not an error.** Per Section 1c.2, the run proceeds with whatever is present and logs the rest. SurGen aggregated features exist **locally only for `conch1-5` and `virchow2`** (622 slides each, under `surgen_data/surgen_processed/<model>/features/slide_aggregation/<method>/<model>/`). `SurGen_Results/` contains results for five models, so H-Optimus-1, UNI2 and ConchV1 SurGen features were produced on the remote machine (`/media/dp-psau/Datum/Aamir/Azfaar`) and are **not on this workstation**.

**Do not fabricate or re-extract them. Instead:**
1. Run SurGen-EV now for **Conch1_5 and Virchow2**, across Averaging / Tissue_Type_Clustering / Caption_based_aggregation, for all three Task 3.0 variants.
2. Log the missing combinations to `skipped_combinations.csv`. The identical code, run on the server with `MACHINE=server`, will pick them up; `merge_results.py` then combines both trees.
3. Note the folder-naming mismatch: local dirs are `conch1-5` / `virchow2`, results dirs use `Conch1_5` / `Virchow2`. Put an explicit mapping in `config/paths.py` — do **not** rely on case-insensitive path resolution, since it works on Windows and fails on Linux.

Also: **622 local feature files vs 624 usable labelled slides** — identify the 2 missing and report them.

Output to `slide_classification/SurGen_EV_Results/<method>/<model>/1-MSIH/Output/`.

**Acceptance:** confusion matrices sum to the number of SurGen slides with both a label and features. No SurGen slide appears in any training set.

### Task 3.3 — [P1] Complete the coverage gaps

Fill in, using corrected code:

| Experiment | Missing |
|---|---|
| TCGA-CV | TITAN and PRISM are commented out of `AGGREGATION_METHODS` — re-enable and run |
| PAIP-IV | TITAN and PRISM |
| PAIP-EV | TITAN and PRISM — **plus TCGA-FULL and ensemble variants (Task 3.0)** for every combination. Existing 4-fold-average results are kept as the legacy variant, not overwritten |
| SurGen-CV | `Caption_based_aggregation_15_classes`; TITAN and PRISM |

TITAN runs only with Conch1_5; PRISM is model-agnostic. Both guards already exist in the loops — preserve them.

**Acceptance:** a coverage matrix (experiment × method × model) printed with ✓ / ✗ / blocked, and no unexplained ✗.

---

## PHASE 4 — Reporting layer (~1 day)

### Task 4.1 — Regenerate the consolidated results workbook

`Analysis_and_Visualization/Analysis.ipynb` cell 1 already walks the results tree into `slide_classification/best_of_all_exps_metric.xlsx`, currently with sheets `PAIP`, `SurGen`, `TCGA_PAIP_EV`, `TCGA`.

Update it to emit **one sheet per canonical experiment**, named exactly:

`TCGA-CV` · `PAIP-IV` · `PAIP-EV` · `SurGen-CV` · `SurGen-EV`

That is **five** sheets. The old `PAIP` sheet is dropped with PAIP-CV (Task 2.2).

### Task 4.2 — Add a sorted summary table to every experiment

Each sheet gets a `Summary` block, **one row per (aggregation method × foundation model × classifier)** with columns: `Method`, `Model`, `Classifier`, `BalAcc`, `AUROC`, `Acc`, `MacroF1`, `N_test`.

**Sort order, exactly as specified by the owner:** descending by BalAcc **and** AUROC jointly; **on a tie, AUROC wins**. Implement as: sort by `AUROC` descending, then stable-sort by `BalAcc` descending — this yields BalAcc primary with AUROC as the tie-break.

Summary tables are currently missing for PAIP-EV and SurGen. All five experiments must have one.

### Task 4.3 — Separate TITAN and PRISM tables

TITAN and PRISM are **slide-level encoders with no patch aggregation step**. They are baselines, not aggregation methods, and mixing them into the same table invites a reviewer to read them as comparable rows.

Give each experiment a dedicated `TITAN_PRISM` block, and add a one-line note in the report explaining why they are tabulated apart. `Analysis.ipynb` cell 4 already has a TITAN-vs-PRISM plotting routine — extend rather than duplicate it.

### Task 4.4 — Regenerate radar plots

`Analysis.ipynb` cell 3 currently plots 3 aggregation methods with `Caption_based_aggregation_15_classes` commented out, into `Radar_Plots/{3_agg_methods,4_agg_methods}/`.

Regenerate for all five experiments, for both `AUC` and `BalAcc`, into `Radar_Plots/<experiment-canonical-name>/`. Archive the existing `3_agg_methods` / `4_agg_methods` folders — their underlying numbers predate every Phase 1 and 2 correction.

**Acceptance:** every plot title carries the canonical experiment name and the test-set N.

### Task 4.5 — Write `RESULTS.md`

A single document with, per experiment: definition, cohort, N, split description, coverage matrix, summary table, and known caveats. Include a top-level table stating **N for every experiment** — TCGA-CV 413 · PAIP-IV 42 train / 31 test · PAIP-EV 73 · SurGen-CV 624 · SurGen-EV 622 — since these differ from the raw label-file counts and reviewers will check. Add a short subsection recording that PAIP-CV was run previously and retired, with the reason from Task 2.2.

---

## PHASE 5 — Deferred, do not start yet

Per the owner: complete MSI end-to-end before opening these.

- **BRAF / KRAS / TP53 / CIMP.** Infrastructure is fully wired in every config dict; only `task_names = ["MSIH"]` gates it. TCGA labels exist (`kfold_BRAF.csv`, `kfold_KRAS.csv`, `kfold_TP53.csv`, `kfold_CIMP.csv`, all sharing TCGA's fold assignment). Fix Task 1.4's undefined-`label` bug before enabling.
  > **Open question for the owner — the plan assumes SurGen carries these labels, but I could not verify it.** `surgen_labels.csv` has only `WSI_Id` and `label_desc` (MSI only). The upstream files `SR386_labels.csv` and `SR1482_labels.csv` live on the remote machine and are not in this workspace. **Confirm SurGen mutation labels exist before scheduling TCGA→SurGen mutation experiments.**
- **Tissue-classifier attribution.** `TissueClassifier_CRC100K/model_metrics/eval_metrics.csv` holds one **unlabelled** row (Acc 0.701, BalAcc 0.675, MacroF1 0.639, AUROC 0.948) against **nine** trained classifiers. Regenerate with a model-name column. The owner's TTC/best-k experiments are the evidence base for choosing among them.

---

## Known issues carried forward — mention in the report, no code change needed

1. **Training set is 50% of the cohort.** The rotation is test = fold *i*, val = fold *i+1*, train = remaining two. Confirmed deliberate by the owner (2 train / 1 val / 1 test). State it explicitly in Methods — a reviewer comparing against 5-fold work will otherwise assume 75% training data.
2. **`combine_trainval=True`** for `lin`/`knn`/`proto`/`rf` merges the validation fold into training; only the ANN uses it as a true validation set. Not wrong, but "validation fold" means two different things depending on classifier. Document it.
3. **Empty tissue rows.** In Tissue_Type_Clustering, an absent tissue class yields an all-zero row. Across 417 TCGA slides: lymphocyte empty in **109**, adipose 69, smooth muscle 51, normal mucosa 51. The classifier cannot distinguish "absent" from "zero-valued". Relevant when interpreting which classes the top-k experiments select.
4. **Notebook section headings do not always match their code paths** — one section labelled TCGA loads PAIP data. Cosmetic, but fix the headings while you are in those files.
5. **Fold numbering.** Folds are 1–4 in the CSVs and reported as `Fold1..Fold4`, but checkpoints are saved `fold0..fold3`. Consistent, but easy to misread — document the offset.

---

## Suggested execution order

```
Phase 0  (0.1 → 0.2)                    ~1.5 h    protect + secure
Phase 1  (1.0 → 1.1 → 1.2 → 1.4 → 1.5 → 1.3)  ~1.5 d  1.0 first (all later tasks edit it); 1.3 last (forces retraining)
Phase 2  (2.1 → 2.2 → 2.3)              ~1 day    protocol
Phase 3  (3.0 → 3.1 → 3.2 → 3.3)        ~3-4 days experiments; 3.0 gates both EV experiments
Phase 4  (4.1 → 4.5)                    ~1 day    reporting
```

**Report to the owner after each phase.** For Phases 1 and 2 specifically, present a **before/after table of affected metrics** — the SurGen-CV and ANN numbers will move, and that movement is the evidence the corrections worked.
