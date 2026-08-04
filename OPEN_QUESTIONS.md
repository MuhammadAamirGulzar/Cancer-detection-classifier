# Open Questions / Judgment Calls Log

Per the work order's instruction: undocumented decisions are recorded here with the reasoning, the most conservative option is chosen, and work continues without stopping to ask. Each entry is resolved unless marked OUTSTANDING.

---

### 1. `.gitignore` scope for `*.png` — RESOLVED, deviated from literal doc wording

The work order says to gitignore `*.png` "under patch/feature dirs" (implying a scoped exclusion, not blanket). A repo-wide extension census found **181,915 PNG files** — overwhelmingly patch-tile dumps, not analysis-result plots. Enumerating every patch/feature directory individually would be error-prone (miss one, and a future `git add` silently stages tens of thousands of tiles). Chose a blanket `*.png` gitignore instead, on the reasoning that the doc's own goal (don't track patch imagery) is better served by being over-inclusive here than under-inclusive, and the doc's stated exception (keep `*.xlsx`/`*.csv` results tracked) already covers the actual audit trail. Also added `*.tif`/`*.tiff` (2,114 files, same reasoning) and `*.h5`/`*.bin` (patch-feature and model-checkpoint binaries) which the doc's literal list didn't name but are clearly the same category as `*.pt`/`*.pth`/`*.pkl`.

**Consequence:** if Phase 4's radar plots or the `Tissue_Type_Combinations_Exp/TCGA_TTC_Result/*.png` summary charts need to be tracked in git (they read like results, similar to the `.xlsx` summaries beside them), they'll need an explicit `!path/to/file.png` whitelist rule added to `.gitignore` at that point. Not done pre-emptively because the exact output paths don't exist yet pre-Phase-4.

### 2. Repo push destination and sequencing — RESOLVED via live user clarification

The work order assumes a single project directory with one git remote. In practice `origin` was already linked to a *different* GitHub account (`fahamin5149`, a collaborator, not the project owner) with local `master` already in sync with `origin/main`. The live user (chat, 2026-08-04) clarified they want a **new** repo created under their own account, `github.com/MuhammadAamirGulzar` — done (private repo, remote named `aamir`; `origin` left untouched). Sequencing was also adjusted: Task 0.2 (credential purge) was done *before* the first commit/push rather than after, so the checkpoint commit that reaches GitHub never contains the plaintext tokens even in history. This is a stricter reading of the doc's own principle ("the code change is safe to make before revocation") applied one step earlier, since the tokens are now going to a remote host and revocation is being deferred indefinitely rather than immediately.

### 3. Token revocation timing — RESOLVED via live user instruction, deviates from doc's Task 0.2 framing

The doc frames Task 0.2 revocation as a near-immediate hard stop ("the user must revoke... before you proceed" for that task, though not blocking the rest of the run). The live user explicitly said not to worry about revocation now and will revoke "at the end after project completed." Proceeding on that basis: continuing to use the existing `HF_TOKEN` in `.env` (gitignored, never committed) for the rest of this run. `TOKENS_TO_REVOKE.md` is written now (redacted) so the checklist is ready whenever revocation happens.

---

### 4. Hardcoded-path sweep scope: classification pipeline now, upstream stages deferred — RESOLVED, narrower than the doc's literal acceptance

Work order §1c.2 item 1 says to remove every hardcoded `D:\Aamir Gulzar\...` / `/media/dp-psau/...` path from **every** script and notebook, with a repo-wide grep as the acceptance check. A census found **34 files** carrying such roots, split across two very different groups:

- **Classification pipeline (11 files, in `slide_classification/`)** — this is what Phases 1–4 actually execute. Swept: all active drivers now resolve roots through `config/paths.py`. Three of the offenders were retired to `misc/` by Task 1.5; the three SurGen CV scripts are superseded by `runners/cv_runner.py` in Task 2.1.
- **Upstream feature-extraction / aggregation (23 files, in `Complete_Pipeline/`, `slide_aggregation/`, `data_preprocessing/`, `Analysis_and_Visualization/`)** — these produce the `.pt` features that the classification pipeline consumes. They are **not re-run** by this remediation; Phases 1–4 read their output as given.

**Decision:** sweep the classification pipeline now; defer the upstream 23. Reasoning: converting them is mechanical but touches code that is not being executed or validated in this run, so a typo there would be undetectable until someone next re-extracts features — the change would carry risk with no offsetting benefit inside Phases 1–4. Making them depend on `slide_classification/config/paths.py` would also invert the dependency direction (feature extraction importing from slide classification).

**Consequence / when this must be revisited:** the upstream sweep becomes necessary the moment SurGen features for H-Optimus-1 / UNI2 / ConchV1 need to be generated on the server (work order Task 3.2 anticipates exactly this). At that point the aggregation scripts should either import a repo-root-level shared config or take their roots as CLI arguments. Until then, the repo-wide grep in §1c.2's acceptance check does **not** pass — it passes for `slide_classification/` only. Flagged rather than quietly claimed.

### 5. CIMP is not a binary task — RESOLVED 2026-08-04, decided from the data

**Original problem (kept for the record):** `kfold_CIMP.csv` carries `HypermethylationCategory` with **four** levels, so it needed dichotomising before it could be used. The owner was unsure which split was right and asked for either a clearer framing or a decision.

**Resolved by cross-tabulating the four levels against MSI status** across the 416 TCGA slides, rather than by appealing to convention:

| Category | n | % MSI-H |
|---|---|---|
| **CIMP-H** | 54 | **68.5%** |
| CRC CIMP-L | 178 | 5.6% |
| Non-CIMP | 182 | 6.6% |
| GEA CIMP-L | 2 | 100% |

CIMP-L is statistically indistinguishable from Non-CIMP (5.6% vs 6.6% MSI-H). Only CIMP-H is a distinct biological group, which is expected — CIMP-H arises via MLH1 promoter hypermethylation, the same mechanism that produces sporadic MSI-H.

**Decision: `CIMP-H` vs everything else** (54 positives, 13%). Three reasons:

1. It is the only split the data supports. Folding CIMP-L into the positive class would bury 178 Non-CIMP-like slides in with 54 genuinely distinct ones and destroy the signal.
2. It is the standard framing in the colorectal literature — CIMP-H is the recognised clinical entity; CIMP-L's definition varies by marker panel.
3. Its 13% prevalence closely matches MSI-H's 15%, so the pipeline's class weights, τ policy and fold structure carry over without retuning.

Implemented in `data_layer._label_map_tcga`, with the reasoning in the code. **Phase 5 is no longer blocked on this.**

**Data-quality note worth raising separately:** two slides are labelled `GEA CIMP-L` — *Gastroesophageal* Adenocarcinoma — in a colorectal cohort. Almost certainly a TCGA pan-cancer annotation artifact. Both are MSI-H. Under this decision they are negatives. Two slides out of 416 is negligible either way, but if CIMP is ever reported, they are worth checking or excluding explicitly.

### 5b. (superseded) CIMP is not a binary task — OUTSTANDING (Phase 5 blocker, no action needed yet)

`kfold_CIMP.csv` carries `HypermethylationCategory` with **four** levels (`Non-CIMP` 182, `CRC CIMP-L` 178, `CIMP-H` 54, `GEA CIMP-L` 2) and a matching 4-level `label_id`. Every other task (MSIH/BRAF/KRAS/TP53) is binary. The work order lists CIMP alongside them in Phase 5 without saying how to dichotomise it — `CIMP-H` vs rest, or `{CIMP-H, CRC CIMP-L, GEA CIMP-L}` vs `Non-CIMP`, are both defensible and give very different positive rates (54/416 vs 234/416).

**Decision:** `data_layer.load_label_map` raises `NotImplementedError` for CIMP with an explicit message rather than guessing. Phase 5 is deferred and out of scope, so this blocks nothing today. **Needs an owner decision before CIMP is enabled.**

---

*(New entries append below as they come up.)*
