# Hugging Face tokens to revoke

**Status: revocation deferred by owner until project completion (2026-08-04).** The tokens below remained in use via `.env` (local, gitignored) for the remainder of this work. This file is the checklist for when revocation happens — go to https://huggingface.co/settings/tokens and revoke by name/listing; you don't need the raw string to revoke, just to know how many distinct ones are live and roughly where they were used.

Token values are intentionally **not** reproduced here in full, even though this file's directory is gitignored from the `.env` itself — no reason to write live secrets into any new file if a redacted form is enough to act on.

## Distinct tokens found (3, code-verified — not 4 as the work order estimated)

| Token (redacted) | Used in |
|---|---|
| `hf_EGzv...jqea` | `Complete_Pipeline/*.py` (7 files: combined_pipeline_final_*, surgen_processing_*), `Complete_Pipeline/surgen_processing_pipeline_final.ipynb`, `Analysis_and_Visualization/*.ipynb` (3 files), `feature_extraction/Feature_Extraction_Prism2_fivecrop.ipynb` (was leaking into cell *output*, not just source) |
| `hf_BHaM...qXEn` | `slide_classification/Slide_Classification.ipynb`, `Slide_Classification_TTC_exp.ipynb`, `Slide_Classification_best_k_labels_exp_updated.ipynb`, `misc/Slide_Classification_Old.ipynb`, `TissueClassifier_CRC100K/Crc100K_Model_Training_Conch_UNI2.ipynb`, and quoted as evidence text in `slide_classification/PIPELINE_AUDIT.md` (now redacted there too) |
| `hf_YSFa...UjkP` | `feature_extraction/misc/Titan_Project_Old.ipynb`, `slide_aggregation/titan/Titan_Project.ipynb`, `slide_aggregation/caption_generation/misc/Label_matching_old.ipynb` |

## What was already fixed (code-only, no revocation needed for this part)

All 22 files above now read `os.environ["HF_TOKEN"]` (via `load_dotenv()`) instead of a hardcoded literal. `.env` already existed locally with `HF_TOKEN` set and was already in `.gitignore`. Verified zero remaining literal `hf_...` token strings anywhere in tracked source (notebook cell outputs included) as of 2026-08-04.

## When you're ready to revoke

1. Go to https://huggingface.co/settings/tokens, revoke all tokens matching the three prefixes above (or just revoke everything and issue one fresh token if simpler).
2. Issue a new token, put it in `.env` as `HF_TOKEN=hf_...` (already gitignored, nothing else to change).
3. Delete this file, or leave it — it contains no live secret.
