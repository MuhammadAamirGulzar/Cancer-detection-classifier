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

*(New entries append below as they come up.)*
