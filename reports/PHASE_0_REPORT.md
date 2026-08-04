# Phase 0 Report — Protect the Work

**Date:** 2026-08-04
**Status:** Complete

## What changed

### Task 0.1 — Version control
- Confirmed the repo already had 8 prior commits and an `origin` remote (`github.com/fahamin5149/Cancer-detection-classifier`, a collaborator's account) that local `master` matched exactly — left untouched.
- Per live owner instruction, created a **new private repo** at `github.com/MuhammadAamirGulzar/Cancer-detection-classifier` and added it as remote `aamir`.
- Rewrote `.gitignore`. Beyond the doc's literal list, a pre-commit size audit (`git diff --cached` + `stat`) found several files the doc's rules didn't anticipate:
  - `*_labeled_df*.csv` and `classification_results_*.csv` — raw per-patch classification dumps, **100MB–1.9GB each**, several duplicated across two directory locations. Not the kind of aggregate "result" the doc's "keep `*.csv` tracked" rule was written for.
  - `Complete_Pipeline/logs/*.log` + `crash_diagnostics/*.log` — 200MB+ of run logs.
  - `data_cleaning/iframe_figures/*.html` — large embedded-data Plotly exports.
  - Also added `*.tif`/`*.tiff`/`*.h5`/`*.bin` alongside the doc's `*.pt`/`*.pth`/`*.pkl`/`*.png` (same "large regenerable binary" category, confirmed via a repo-wide extension census: 181,915 PNGs, 13,851 `.pt` files, etc., against a ~107GB working tree).
  - Full reasoning logged in `OPEN_QUESTIONS.md` #1.
- Checkpoint commit `be0cbf7`: 777 files, ~318MB, largest single file 35MB.
- First push attempt **hung indefinitely** — diagnosed via Windows process CPU stats (`git`/`git-remote-https` at ~0 CPU, `git-credential-manager` at 699+ CPU-seconds and climbing) as a stuck credential resolution, not slow compression as initially assumed. Root cause: `gh auth login` had authenticated but never wired itself into git's credential chain — only the generic system-wide Windows `manager` helper was configured, which had no working credential for the new account and hung instead of failing cleanly. Fixed with `gh auth setup-git`; retry pushed immediately.
- Follow-up commit `93d591f` removed 9 `.ipynb_checkpoints` files that were tracked from the original pre-session history (predating this work entirely) and survived the new `.gitignore` because gitignore doesn't retroactively untrack already-committed files. One (`Titan_Project-checkpoint.ipynb`) still held a live token that the Task 0.2 scanner had missed because it explicitly skips checkpoint directories.

### Task 0.2 — Credential purge (code only; revocation deferred per live owner instruction)
- Re-verified the doc's "21 files / 4 tokens" claim against actual code rather than trusting it: found **22 files, 3 distinct tokens** (a source-cell/text-only scanner, to avoid false positives from base64 image blobs in notebook outputs — the doc's literal `grep` pattern alone produces false positives for exactly this reason).
- Patterns found and fixed, all replaced with `os.environ["HF_TOKEN"]` + `load_dotenv()`:
  - Bare `login("hf_...")` calls (5 notebooks)
  - Keyword `login(token="hf_...")` calls (2 files)
  - Plain variable assignment `HF_TOKEN = "hf_..."` (8 `.py` files + 1 notebook)
  - Dict-config value `hf_token = "hf_...",` (3 notebooks)
  - Default-parameter value `token: str = "hf_..."` (2 files, 3 occurrences)
  - One notebook (`Feature_Extraction_Prism2_fivecrop.ipynb`) had a `print(hf_token)` debug line that had leaked the live token into cell **outputs** (not just source) — removed the print, cleared that notebook's outputs.
  - `slide_classification/PIPELINE_AUDIT.md` quoted a token as evidence text — redacted in place.
- **Sequencing deviation from the doc:** the doc's Task 0.1 commits first, Task 0.2 purges after (accepting tokens in local git history). Since this session was also pushing to a new remote for the first time, purge was done *before* the first commit instead, so no plaintext token ever entered history on the `aamir` remote at all. Reasoning in `OPEN_QUESTIONS.md` #2–3.
- Revocation itself explicitly deferred by the live owner until project completion; `.env` (already present, already gitignored) continues to supply `HF_TOKEN` for all downstream work. `TOKENS_TO_REVOKE.md` written (token values redacted even there) as the checklist for when revocation happens.

## Acceptance checks (doc's exact commands, output pasted)

```
$ git status --short | grep -E '^\?\?.*\.(py|ipynb|xlsx|csv)$' | wc -l
0

$ git ls-files slide_classification | wc -l
499

$ git ls-files | grep -E '\.(py|ipynb)$' | xargs grep -l "hf_[A-Za-z0-9]\{30,\}"
(no output — 0 matches, re-checked after the checkpoint-file cleanup commit)
```

## Skipped / not applicable
- Nothing skipped. Both tasks fully complete.

## Added to OPEN_QUESTIONS.md
1. `.gitignore` `*.png`/`*.tif` scope — blanket exclude instead of the doc's literal "under patch/feature dirs" scoping (181,915 files repo-wide).
2. Push destination — new repo under the live owner's own account rather than the pre-existing `fahamin5149` remote.
3. Token revocation timing — deferred to project completion per live instruction.

## Verified ground-truth corrections (for the record, not blocking)
An Explore-agent audit of the data-loading code (staged for Phase 1 Task 1.0) found the work order's own file lists have a few small errors, logged here rather than in OPEN_QUESTIONS.md since they don't require a decision, just a correction before Phase 1 starts:
- `WSIDataset`-family duplication is closer to **11–12 copies**, not "~6" — a second family (`TissueCombinationDataset`/`TissueCombinationDataset14`, 5 more copies) does the identical job and wasn't counted. That family uses a dict-indexed lookup with no stale-label hazard — it's the better pattern to standardize the new `data_layer.py` on, not the `WSIDataset` copies' O(n) `.values`/`.loc` scan.
- Task 1.1's file list is missing `slide_classification_surgen_best_k_labels_exp.py`, which has the identical test-set-metric ANN-selection leakage.
- §1c.1 item 6 names `r_forest.py` for the `n_jobs` fix, but that file is dead code (never imported anywhere); the live module `r_forest_eval.py` already has `n_jobs=-1` set on both `RandomForestClassifier` instantiations.
- The undefined-`label` bug (Task 1.4) is worse than described: only the *first* unmatched WSI in a given load raises the swallowed exception; every subsequent unmatched WSI after any prior match instead **silently inherits the previous WSI's label** with no error at all. Confirmed dormant for the currently-active MSIH task (folds are built from the same CSV the labels come from), so no existing published result is affected — but real exposure the moment BRAF/KRAS/TP53/CIMP (Phase 5) get enabled.

Full detail in the agent's findings, which will drive Phase 1 Task 1.0 directly.
