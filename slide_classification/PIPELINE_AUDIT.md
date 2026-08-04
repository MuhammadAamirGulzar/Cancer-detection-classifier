# Slide Classification Pipeline Audit

Read-only audit of `slide_classification/`. All files listed were read in full (notebooks cell-by-cell). Citations reference the actual file/cell/line where a claim is grounded. Anything not directly confirmed from code is marked **INFERENCE**.

---

## 1. Per-file findings

### 1.1 `Slide_Classification.ipynb`

Multi-section notebook containing several **independent** pipelines under one file (each preceded by a markdown header), not a single coherent pipeline:

| Cell | Markdown label | Actual cohort trained/evaluated | Data source |
|---|---|---|---|
| Cell 6 | "Pipeline for TCGA Data" | **TCGA**, internal 4-fold CV | `K_FOLDS_PATH = os.path.join(PROJECT_ROOT, "kfolds_IDARS_fixed.csv")`; `DATA_PATH = os.path.join(BASE_ROOT, "dataset", "slide_aggregation", AGGREGATION_METHOD, MODEL_NAME)` |
| Cell 8 | "Pipeline for PAIP Dataset" | **PAIP**, internal 4-fold CV | `K_FOLDS_PATH = os.path.join(PROJECT_ROOT, "paip_kfolds_71.csv")`; `DATA_PATH = os.path.join(BASE_ROOT, "paip_data", "slide_aggregation", AGGREGATION_METHOD, MODEL_NAME)` |
| Cell 13 | (standalone PAIP split) | **PAIP**, string-matched train/val split (not cross-cohort) | `train_ids = labels_df[labels_df["WSI_Id"].str.contains("training", case=False)]...`; `DATA_PATH = r"...\paip_data\slide_aggregation\Caption_Based_Clustering_FiveCrop\conch_CC_fivecrop"` |
| Cell 15 | "Pipeline for TCGA MultiClass classification (full)" | **Actually PAIP** — label is misleading | `K_FOLDS_PATH = r"D:\...\paip_data\Results\paip_kfolds_71.csv"`; `DATA_PATH = r"D:\...\paip_data\slide_aggregation\conch_average_fivecrop"` |

**Held out?** No section performs cross-cohort validation — every pipeline here is k-fold (or split) CV **within a single cohort's own data**. Cell 15's "TCGA MultiClass" name does not match its data paths, which point at PAIP — flagged as likely copy/paste naming leftover (**INFERENCE** on cause, but the path mismatch itself is directly read).

**Tasks active:** `task_names = ["MSIH"]` (Cell 6, Cell 8). BRAF/KRAS/TP53 label-loading branches exist in `WSIDataset._load_data` but are unreached dead code since `task_names` never includes them; no CIMP branch present in this notebook's own dataset class.

**Aggregation methods / foundation models (Cell 6, mirrored in Cell 8):**
```python
AGGREGATION_METHODS = ["Caption_based_aggregation_15_classes"]  # ["Averaging", "Caption_based_aggregation", "Caption_based_aggregation_15_classes", "Tissue_Type_Clustering", "TITAN", "PRISM"]
MODEL_NAMES = ["H-Optimus-1", "Conch1_5", "UNI2", "Virchow2", "ConchV1"]
```
Classifiers per fold: `model_types = ['lin', 'ann', 'knn', 'proto', 'rf']` (logistic regression, ANN, KNN, ProtoNet, Random Forest). "Aggregation" here means pre-computed slide-level feature aggregation strategy (Averaging / Caption-based clustering / Tissue-Type clustering / TITAN / PRISM), not an in-model attention aggregator.

---

### 1.2 `external_validation_script.py`

**Purpose:** "Universal External Validation — TCGA to PAIP." Pure inference script — no training.

**Trained on:** Nothing trained here; loads pre-existing TCGA-trained model artifacts:
```python
TCGA_MODELS_BASE = os.path.join(PROJECT_ROOT, "TCGA_Results_Updated", AGGREGATION_METHOD, MODEL_NAME, TASK, "models")
```
(lines 48-49)

**Evaluated on:** PAIP:
```python
PAIP_DATA_BASES = [
    os.path.join(BASE_ROOT, "paip_data", "slide_aggregation", "Caption_Based_Clustering_FiveCrop"),
    os.path.join(BASE_ROOT, "paip_data", "slide_aggregation", AGGREGATION_METHOD, MODEL_NAME),
    ...
]
GROUND_TRUTH_PATH = os.path.join(BASE_ROOT, "paip_data", "labels", "paip_kfolds_71.csv")
```
(lines 51-60)

**Held out?** Fully held out — cross-cohort: TCGA-trained models applied to PAIP features/labels with zero PAIP data in training.

**Task:** `TASK = "1-MSIH"` (line 42); comment shows the full intended list (`# 1-MSIH, 2-BRAF, 3-KRAS, 4-TP53, 5-CIMP`) but only MSIH is selected.

**Model/aggregation:**
```python
MODEL_NAME = "H-Optimus-1"  # H-Optimus-1, Conch1_5, UNI2
AGGREGATION_METHOD = "Caption_based_aggregation"
```
(lines 40-41). Classifiers run: `'classifiers_to_run': ['logistic_regression', 'protonet', 'ann']` (lines 217-221); `knn`/`random_forest` templates exist but aren't in the active run list.

---

### 1.3 `TCGA_PAIP_Validation.ipynb`

**Purpose:** Batch/looping version of the same TCGA→PAIP external validation pattern as 1.2 — pure inference, no training code anywhere in the notebook.

**Trained on:** Nothing trained; loads TCGA classifier artifacts:
```python
if aggregation_method == "PRISM":
    tcga_models_base = os.path.join(project_root, "TCGA_Results", "PRISM", task, "models")
else:
    tcga_models_base = os.path.join(project_root, "TCGA_Results", aggregation_method, model_name, task, "models")
```
(Cell 7)

**Evaluated on:** PAIP:
```python
GROUND_TRUTH_PATH = os.path.join(BASE_ROOT, "paip_data", "labels", "paip_kfolds_71.csv")
paip_data_base = get_paip_data_path(base_root, aggregation_method, model_name)
```
(Cell 1, Cell 7); `get_paip_data_path` (Cell 2) → `os.path.join(base_root, "paip_data", "slide_aggregation", aggregation_method)`.

**Held out?** Fully held out, same as 1.2.

**Active run:**
```python
AGGREGATION_METHODS = ["PRISM"]
MODEL_NAMES = ["PRISM"]
TASKS = ["1-MSIH"]
```
(Cell 1) — only MSIH active; the full intended sweep across `H-Optimus-1/Conch1_5/UNI2/Virchow2/ConchV1` × `Caption_based_aggregation/Tissue_Type_Clustering/Averaging/TITAN` is commented out. `FOLDS = [0,1,2,3]` (Cell 2) are the 4 TCGA-trained fold models, each applied to the full PAIP set.

---

### 1.4 `slide_classification_surgen.py` — **SurGen protocol resolved**

**CRITICAL FINDING:** This script does **NOT** train on TCGA and evaluate on SurGen. It trains **and** evaluates entirely within **SurGen**, via k-fold CV internal to that cohort — the same "own-cohort CV" pattern as `Slide_Classification.ipynb`'s TCGA/PAIP sections, not the external-validation pattern of `TCGA_PAIP_Validation.ipynb`/`external_validation_script.py`.

Proof — single data directory, no TCGA path anywhere in the file:
```python
BASE_ROOT = "/media/dp-psau/Datum/Aamir/Azfaar"
PROJECT_ROOT = os.path.join(BASE_ROOT, "surgen_processing")
DATA_PATH = os.path.join(BASE_ROOT, "dataset", "slide_aggregation", AGGREGATION_METHOD, MODEL_NAME)
RESULT_ROOT = os.path.join(PROJECT_ROOT, RESULT_FOLDER, AGGREGATION_METHOD, MODEL_NAME)
```
(lines 79-80, 133-134; `RESULT_FOLDER = "SurGen_Results"`, line 36)

Folds are built at runtime from slide IDs found inside that same directory (not a pre-split TCGA/SurGen partition):
```python
all_slide_ids = [os.path.splitext(f)[0] for f in os.listdir(DATA_PATH) if f.endswith('.pt')]
folds = build_runtime_folds(all_slide_ids, num_folds=NUM_FOLDS, seed=RANDOM_SEED)
```
(lines 437-443), then rotated train/val/test **within these SurGen folds** in `run_k_fold_cross_validation` (lines 395-433):
```python
test_ids = folds[i]
val_ids = folds[(i + 1) % num_folds]
train_ids = [... other folds ...]
```
called as `run_k_fold_cross_validation(DATA_PATH, folds, model_type=model)` (line 460) — `DATA_PATH` is the single SurGen feature directory. Labels: `LABELS_CSV_PATH = "/media/dp-psau/Datum/Aamir/Azfaar/surgen_processed/surgen_labels.csv"` (line 76).

**Tasks:** `task_names = ["MSIH"]` (line 65). BRAF/KRAS/TP53 branches exist in `WSIDataset._load_data` (lines 196-222) but unreached; no CIMP branch.

**Aggregation/models:**
```python
AGGREGATION_METHODS = ["Averaging","Tissue_Type_Clustering"]
MODEL_NAMES = ["H-Optimus-1"]
```
(lines 34-35), classifiers `['lin','ann','knn','proto','rf']`.

---

### 1.5 `slide_classification_surgen_best_k_labels_exp.py`

**Purpose:** SurGen version of the tissue-class-selection experiment. Phase 1 scores each of 8 non-background tissue classes individually (balanced accuracy) across foundation models; Phase 2 fixes the top-k globally-best classes (k ∈ {3,5,8}) and re-trains/evaluates that combination — same two-phase design as §1.6 below, applied to SurGen instead of TCGA.

Also confirms the same-cohort finding of §1.4: single data directory, runtime-built folds, no TCGA reference:
```python
BASE_ROOT       = "/media/dp-psau/Datum/Aamir/Azfaar"
PROJECT_ROOT    = os.path.join(BASE_ROOT, "surgen_processing")
LABELS_CSV_PATH = "/home/mle/Aamir/Azfaar/surgen_processing/surgen_labels.csv"
RESULT_FOLDER   = "Surgen_TTC_exp"
DATA_PATH = os.path.join(BASE_ROOT, "surgen_processed", MODEL_NAME, "features", "slide_aggregation", AGGREGATION_METHOD, MODEL_NAME)
```
(lines 114-119, 764-767), folds:
```python
all_slide_ids = [os.path.splitext(f)[0] for f in os.listdir(first_data_path) if f.endswith(".pt") and os.path.splitext(f)[0] in valid_ids]
folds = build_runtime_folds(all_slide_ids, num_folds=NUM_FOLDS, seed=RANDOM_SEED)
```
(lines 793-798).

**Tissue classes swept:**
```python
TISSUE_CLASS_NAMES = ["adipose","background","debris","lymphocyte","mucin","smooth_muscle","normal_mucosa","stroma","tumor"]
BACKGROUND_IDX       = TISSUE_CLASS_NAMES.index("background")   # → 1
ACTIVE_CLASS_INDICES = [i for i in range(NUM_TISSUE_CLASSES) if i != BACKGROUND_IDX]
```
(lines 85-102); `k_values = [3, 5, 8]` (line 108). Only ANN classifier used: `model_types = ["ann"]` (line 107), only `Tissue_Type_Clustering` aggregation, `MODEL_NAMES = ["h-optimus-1"]` (line 60). **Tasks:** `task_names = ["MSIH"]` (line 106); `SURGEN_TASK_CONFIG` defines MSIH/BRAF/KRAS/TP53, no CIMP entry, only MSIH active.

---

### 1.6 `Slide_Classification_best_k_labels_exp_updated.ipynb`

**Purpose:** TCGA-only notebook running two tissue-class-selection ("best k labels") experiments — one for `Tissue_Type_Clustering` (9 classes, cells 5–13), one for `Caption_based_aggregation` (14 classes, cells 14–21). Each `.pt` per WSI is a `(NUM_TISSUE_CLASSES × feature_dim)` matrix, one row per tissue/caption cluster. The "k labels" experiment selects a subset of `k` class-rows to concatenate into the classifier's input feature vector — i.e., hard top-k feature/row selection over tissue clusters, not attention pooling.

**Method 1** (cells 7, 15): tier-filtered exhaustive combination search via `itertools.combinations`.
**Method 2 — "best k only"** (cells 11, 19, the file's namesake): global top-k selection.

- 9-class candidates (cell 11): `TISSUE_CLASS_NAMES = ["adipose","background","debris","lymphocyte","mucin","smooth_muscle","normal_mucosa","stroma","tumor"]`; background excluded from candidacy:
  ```python
  BACKGROUND_IDX = TISSUE_CLASS_NAMES.index("background")  # → 1
  EVALUABLE_INDICES = [i for i in range(NUM_TISSUE_CLASSES) if i != BACKGROUND_IDX]
  ```
- 14-class candidates (cell 15): adipose, debris, lymphocyte, plasma_cell, lymphoid_agg, mucin, smooth_muscle, normal_mucosa, adenoma, stroma, mesenchymal, carcinoma, poor_diff_carcinoma, signet_ring — no background bucket exists in this label set at all.
- **k sweep:** `k_values = [3, 5, 7, 8]` (9-class, cell 11); `k_values_14 = [3, 5, 7, 10, 14]` (14-class, cell 19).
- **Selection logic:** Phase 1 evaluates each active class alone across all 5 foundation models (`MODEL_NAMES = ["H-Optimus-1","Conch1_5","UNI2","Virchow2","ConchV1"]`) with an ANN classifier, computing `df["avg_bacc"] = df[bacc_cols].mean(axis=1, skipna=True)` and sorting descending. Phase 2 takes `top_k_rows = ranked_df.head(k)` — the single globally-best k-subset (by average balanced accuracy across models) is frozen and re-tested, per k, across all 5 models. This is **not** an exhaustive search over k-sized combinations — only one (the globally top-ranked) combination is tested per k, in contrast to Method 1's exhaustive tiered search.
- **Ranking metric:** balanced accuracy (`bacc`), `model_types = ["ann"]`.

**Cohort:** TCGA only — `TASK_FOLD_CONFIG` (cells 7/11/15/19) points at local TCGA fold CSVs (`kfolds_IDARS_fixed.csv`, `kfold_BRAF.csv`, `kfold_KRAS.csv`, `kfold_TP53.csv`, `kfold_CIMP.csv`); `DATA_PATH = os.path.join(BASE_ROOT, "dataset", "slide_aggregation", AGGREGATION_METHOD, MODEL_NAME)`, `BASE_ROOT = r"D:\Aamir Gulzar\KSA_project2"`. Neither PAIP nor SurGen loaded in this notebook.

**Tasks active:** `task_names = ["MSIH"]` in every driving cell, despite `TASK_FOLD_CONFIG` fully defining BRAF/KRAS/TP53/CIMP.

**Note:** Cell 3 hardcodes a live-looking Hugging Face token via `login("hf_***REDACTED***")` — a credential-hygiene issue outside the audit's original scope but worth flagging; same token appears in `Slide_Classification_TTC_exp.ipynb` cell 3.

---

### 1.7 `Slide_Classification_TTC_exp.ipynb`

**Purpose:** Earlier/parallel Tissue-Type-Combination (TTC) experiment notebook. Cell 6 ("Blind test all TTC combinations") runs the baseline `lin/ann/knn/proto/rf` CV pipeline across `AGGREGATION_METHODS = ["Averaging","Caption_based_aggregation","Tissue_Type_Clustering","TITAN"]`, and only when `AGGREGATION_METHOD == "Tissue_Type_Clustering"` also calls `run_tissue_combination_search(...)` with `k_values = [3,5]`. Cell 8 ("Method 2: Check individual results then match") is the tier-based exhaustive-search predecessor of cell 7 in the best-k notebook (§1.6).

**Ranking/selection logic (cell 8):** Phase 1 scores each class alone; tiers by `bacc`: `STRONG_THRESHOLD = 0.65`, `NEUTRAL_THRESHOLD = 0.58`. Phase 2's `_smart_combinations()` filters `itertools.combinations(all_indices, k)` by `MAX_WEAK_PERCENT = 0.45`, `MAX_NEUTRAL_PERCENT = 0.70`, `MIN_STRONG_COUNT = 1`, sorted by `bacc` descending.

**Background exclusion — DISCREPANCY vs. §1.6:** This notebook does **not** exclude background anywhere. `TISSUE_CLASS_NAMES` still includes `"background"` at index 1, Phase 1 iterates `for class_idx, class_name in enumerate(TISSUE_CLASS_NAMES):` over all 9 classes, and Phase 2's `all_indices = class_df["class_idx"].tolist()` includes background unless it fails a tier threshold on its own merit. There is no `BACKGROUND_IDX`/`EVALUABLE_INDICES` construct in this file — that exclusion mechanism was added only in the later "updated" best-k notebook (§1.6).

**Cohort:** TCGA only (same `TASK_FOLD_CONFIG`/`DATA_PATH` pattern as §1.6). **Tasks:** `task_names = ["MSIH"]` (cell 6, cell 8), despite the comment listing `# "MSIH","BRAF","KRAS","TP53","CIMP"`.

---

### 1.8 `plot_surgen_ttc_results.py`

Plots SurGen "fixed top-k combo" results, reading from:
```python
BASE_SURGEN = os.path.join(
    "/media/dp-psau/Datum/Aamir/Azfaar",
    "surgen_processing", "Surgen_TTC_exp", "Tissue_Type_Clustering", "Global", "1-MSIH",
)
fname = f"tissue_topk_summary_surgen_k{k_val}.xlsx"
```
This path/filename pattern matches the output of `slide_classification_surgen_best_k_labels_exp.py` (§1.5: `RESULT_FOLDER = "Surgen_TTC_exp"`, `k_values = [3,5,8]`) — i.e., this plots the SurGen analogue of the fixed-top-k ("best-k") experiment, not the tier-based exhaustive search. `K_VALUES_SHOW = [3,5,8]`, `MODEL_NAMES_CHART = ["h-optimus-1"]`, `CLASSIFIER = "ann"`, `RANK_METRIC = "bacc"`.

**Note:** the path it reads (`/media/dp-psau/Datum/Aamir/Azfaar/...`) is a remote Linux path, not the local `SurGen_Results/` folder in this repo — this plotting script cannot run as-is from this working directory; its data source lives on the remote training machine and does not match the local `SurGen_Results/` naming (see discrepancy #4 below).

---

## 2. Active tasks — confirmed answer

**Only MSIH is active across every training/eval file audited.** BRAF/KRAS/TP53 (and, in some files, CIMP) are fully wired into config dictionaries (`TASK_FOLD_CONFIG`, `SURGEN_TASK_CONFIG`) and dataset-loading branches, but every driving `task_names`/`TASKS` list in every file is hardcoded to `["MSIH"]` / `"1-MSIH"`. This is corroborated independently by the results-folder inventory (§3): no `2-BRAF`, `3-KRAS`, `4-TP53`, or `5-CIMP` folder exists anywhere in `TCGA_Results/`, `PAIP_Results/`, `TCGA_PAIP_EV_Results/`, or `SurGen_Results/` — only `1-MSIH`.

---

## 3. Results folder inventory

All four folders share the shape `<AGGREGATION_METHOD>/<FOUNDATION_MODEL>/1-MSIH/{Output,models}` (or a flatter variant for the EV folder).

### `TCGA_Results/`
- **Aggregation methods:** `Averaging`, `Caption_based_aggregation`, `Caption_based_aggregation_15_classes`, `PRISM`, `TITAN`, `Tissue_Type_Clustering`.
- **Foundation models:** `Conch1_5, ConchV1, H-Optimus-1, UNI2, Virchow2` (5) for most methods; `Caption_based_aggregation_15_classes` additionally has a `PRISM` sub-model; `PRISM`/`TITAN` methods don't split by model the same way.
- **Leaf files** (e.g. `Tissue_Type_Clustering/H-Optimus-1/1-MSIH/`): `Output/cross_valid_avg_eval_metrics.xlsx`, `Output/cross_valid_avg_probs_all.xlsx`, `models/fold{0-3}_{knn,logistic_regression,protonet,random_forest}.pkl`, `models/fold{0-3}_trained_ann_model_13824.pth`.
- **Source:** `Slide_Classification.ipynb` (baseline pipeline, Cell 6) — file/folder shape matches that cell's output exactly, and there are no `tissue_combo_summary`/`tissue_topk_summary` files here (those live in `Tissue_Type_Combinations_Exp/`, see below).
- **Protocol:** internal 4-fold CV on TCGA only. **Tasks:** MSIH only.

### `PAIP_Results/`
- Same aggregation-method and model set as `TCGA_Results/`.
- **Source: INFERENCE** — not opened in this audit pass; file/folder naming symmetry with `TCGA_Results/` strongly implies a PAIP-cohort counterpart of the same baseline pipeline (`Slide_Classification.ipynb` Cell 8 or an equivalent script), i.e. internal CV on PAIP.
- **Protocol (INFERENCE):** internal CV on PAIP only. **Tasks:** MSIH only (confirmed structurally — only `1-MSIH` folders present).

### `TCGA_PAIP_EV_Results/`
- **Aggregation methods:** `Averaging`, `Caption_based_aggregation`, `PRISM`, `TITAN`, `Tissue_Type_Clustering` — **missing** `Caption_based_aggregation_15_classes` (present in TCGA/PAIP results).
- **Structure:** flat, no `/Output` or `/models` subfolders — leaves are `Tissue_Type_Clustering/H-Optimus-1/1-MSIH/external_validation_PAIP_individual.csv` and `external_validation_PAIP_summary.csv`.
- **Source:** `external_validation_script.py` — filenames match exactly (`individual_file = os.path.join(output_base, "external_validation_PAIP_individual.csv")`, similarly for `_summary.csv`).
- **Protocol:** external validation — train on TCGA, evaluate on PAIP. **Tasks:** MSIH only.

### `SurGen_Results/`
- **Foundation models:** only `H-Optimus-1`, `UNI2` (2, vs. 5 in TCGA/PAIP).
- **Aggregation methods:** only `Averaging`, `Caption_based_aggregation`, `Tissue_Type_Clustering` — no `PRISM`, `TITAN`, or `Caption_based_aggregation_15_classes`.
- **Leaf files** (`Tissue_Type_Clustering/H-Optimus-1/1-MSIH/Output/`): `cross_valid_avg_eval_metrics.xlsx`, `tissue_combo_probs_k{3,5,8}_best.xlsx`, `tissue_combo_summary_k{3,5,8}.xlsx`, `tissue_individual_class_results.xlsx` — this naming matches the **tier-based exhaustive-search** pattern (§1.7-style Method 1), not the fixed-top-k pattern.
- **Source:** `slide_classification_surgen.py` (`RESULT_FOLDER = "SurGen_Results"`).
- **Protocol:** internal CV within SurGen only (confirmed in §1.4 — no TCGA involvement). **Tasks:** MSIH only.

### `Tissue_Type_Combinations_Exp/` (bonus — related to TTC/best-k experiments)
```
Tissue_Type_Combinations_Exp/
  TCGA_TTC_Result/
    Caption_based_aggregation/{Conch1_5,ConchV1,H-Optimus-1,UNI2,Virchow2}/1-MSIH/{Output,models}
    Caption_based_aggregation/Global/1-MSIH/
        tissue_individual_class_results_14_global.xlsx
        tissue_topk_probs_14cls_k{3,5,7,9,10,14}.xlsx
        tissue_topk_summary_14cls_k{3,5,7,9,10,14}.xlsx
    Tissue_Type_Clustering/{Conch1_5,ConchV1,H-Optimus-1,UNI2,Virchow2}/1-MSIH/{Output,models}
    Tissue_Type_Clustering/Global/1-MSIH/
        tissue_individual_class_results_9_global.xlsx
        tissue_topk_probs_9cls_k{3,5,7,8}.xlsx
        tissue_topk_summary_9cls_k{3,5,7,8}.xlsx
    across_k_*.png, by_model_*.png, k_comparison_best_combo*.png, top5_combos_*.png
  TTC_combination_experiment.py
  __pycache__/
```
This is the **TCGA output of the "best k labels" (Method 2, fixed-top-k) experiment** — filenames (`tissue_topk_summary_9cls_k{k}.xlsx`, `tissue_individual_class_results_9_global.xlsx`, `Global/1-MSIH/`) match `Slide_Classification_best_k_labels_exp_updated.ipynb` cells 11 (9-class) and 15/19 (14-class, `tissue_topk_summary_14cls_k{k}.xlsx`) exactly. Task folder is `1-MSIH` only.

**Note:** on-disk k-values (`k=3,5,7,8` for 9-class; `k=3,5,7,9,10,14` for 14-class) are a superset of the k-lists currently declared in the notebook (`k_values=[3,5,7,8]` cell 11 matches; but `k_values_14=[3,5,7,10,14]` in cell 19 is missing `k=9`, which exists on disk) — implies an earlier run used a different `k_values_14` list than what's currently saved in the notebook, or the notebook was edited after that run (**INFERENCE** on cause).

---

## 4. Discrepancies flagged across folders/experiments

1. **Foundation-model coverage gap:** TCGA/PAIP/TCGA_PAIP_EV all have 5 models (`H-Optimus-1, Conch1_5, UNI2, Virchow2, ConchV1`) plus `PRISM`/`TITAN`; `SurGen_Results/` has only 2 (`H-Optimus-1`, `UNI2`) and lacks `PRISM`/`TITAN`/`Caption_based_aggregation_15_classes` entirely.
2. **Background-exclusion logic differs between the two TTC notebooks:** present (`BACKGROUND_IDX`/`EVALUABLE_INDICES`) only in `Slide_Classification_best_k_labels_exp_updated.ipynb` (§1.6); absent in `Slide_Classification_TTC_exp.ipynb` (§1.7), where background can in principle be selected into a combination if it clears the tier threshold on its own.
3. **Task coverage:** all four result folders and both TTC/best-k notebooks are MSIH-only in practice, despite BRAF/KRAS/TP53/CIMP being fully wired into config dicts across every file — confirmed dead/unused code paths (§2).
4. **`plot_surgen_ttc_results.py` reads from a path that doesn't exist locally** (`/media/dp-psau/Datum/Aamir/Azfaar/...`); the local `SurGen_Results/` folder's file naming (`tissue_combo_summary_k{k}.xlsx`, from `slide_classification_surgen.py`) does **not** match what the plot script expects (`tissue_topk_summary_surgen_k{k}.xlsx`, from `slide_classification_surgen_best_k_labels_exp.py`) — these come from two different SurGen scripts running two different experiment types (exhaustive-tier-search vs. fixed-top-k), and only the latter's remote output feeds the plotting script.
5. **`Caption_based_aggregation_15_classes`** appears in `TCGA_Results/` and `PAIP_Results/` (including a `PRISM` sub-model) but not in `TCGA_PAIP_EV_Results/` or `SurGen_Results/` — this 15-class caption variant was never carried into external validation or SurGen.
6. **Hardcoded credential:** a live-looking Hugging Face token `hf_***REDACTED***` appears in plaintext in `Slide_Classification_best_k_labels_exp_updated.ipynb` (cell 3) and `Slide_Classification_TTC_exp.ipynb` (cell 3). Outside the original audit scope but flagged for credential hygiene.

---

## 5. Direct answers to the critical questions

- **SurGen protocol:** `slide_classification_surgen.py` does **not** train on TCGA and evaluate on SurGen. It trains and evaluates entirely within SurGen via k-fold CV internal to that cohort (folds built at runtime from SurGen slide IDs found in one directory; no TCGA path/model/CSV referenced anywhere in the file). Same is true of `slide_classification_surgen_best_k_labels_exp.py`. The only cross-cohort external-validation pathway found anywhere in this audit is **TCGA → PAIP**, implemented by `external_validation_script.py` and `TCGA_PAIP_Validation.ipynb`.
- **"Best k labels" experiments:** They select a subset of `k` tissue/caption-cluster classes (rows of the per-slide aggregated feature tensor) to feed into the ANN classifier, choosing the single globally top-ranked k-class combination (by cross-model-averaged balanced accuracy) rather than exhaustively searching all combinations of size k. Candidate classes are the 9 `Tissue_Type_Clustering` classes (minus background) or the 14 `Caption_based_aggregation` classes (no background bucket exists there); k is swept over `[3,5,7,8]` (9-class) or `[3,5,7,10,14]` (14-class).
- **TTC experiment:** Ranks individual tissue classes by balanced accuracy into tiers (strong ≥0.65, neutral ≥0.58, else weak), then searches combinations via `itertools.combinations` filtered by tier-composition constraints (max 45% weak, max 70% neutral, ≥1 strong), ranked by `bacc`. **Background exclusion is NOT present** in `Slide_Classification_TTC_exp.ipynb` — that exclusion mechanism was only added in the later `..._best_k_labels_exp_updated.ipynb`.
- **Active tasks:** MSIH only, everywhere. BRAF/KRAS/TP53/CIMP are defined in config but never actually run, confirmed both in code (`task_names = ["MSIH"]` hardcoded in every file) and in results-folder structure (only `1-MSIH` folders exist).
