# TITAN Slide-Embedding Pipeline (`surgen_titan.py`)

This document describes how TITAN (Mahmood Lab, `MahmoodLab/TITAN` on Hugging
Face) is used to produce one slide-level embedding per whole-slide image (WSI)
in the SurGen cohort, as implemented in `surgen_titan.py`. It is written to
serve as a methods reference for later write-up.

## 1. Overview

The script turns each raw `.czi` whole-slide image into a single 768-dimensional
slide-level feature vector, saved as one `.pt` file per slide. Processing runs
in four stages:

| Stage | Name | Purpose |
|---|---|---|
| A-1 | RGB stats | Tile each slide into patches and record per-patch pixel statistics |
| A-2 | Tissue filtering | Discard background/white patches, keep tissue patches |
| QC | Visual inspection (optional) | Save PNG previews of kept/discarded patches for manual review |
| C | TITAN slide embedding | Encode tissue patches with CoNCH v1.5, aggregate into one slide embedding with TITAN |

Stages A-1/A-2/QC are shared, model-independent preprocessing steps common to
every feature-extraction script in this pipeline family (equivalent scripts
exist for UNI2-h, H-optimus-1, Virchow2, and CONCH v1 as the patch encoder).
Only Stage C is specific to TITAN.

## 2. Data and labelling

Two SurGen cohorts are processed: **SR386** and **SR1482**, distinguished by
filename prefix. Each `.czi` filename encodes a case ID (`_T\d{3}_` pattern),
which is matched against a cohort-specific label CSV to assign a binary
microsatellite-instability label:

- **SR386**: `mmr_loss_binary` column (1 → `msih`, 0 → `nonmsih`).
- **SR1482**: free-text `MSI` column (`"MSI high"` → `msih`, `"No MSI"` →
  `nonmsih`). Slides labelled `"MSI Low"`, `"Not performed"`, `"Insufficient"`,
  or `"Failed"` are excluded from downstream processing (ambiguous/unusable
  ground truth).

Slides whose case ID cannot be resolved to a usable label are skipped before
any tiling work is done.

## 3. Patch tiling

Each slide is read via `pylibCZIrw`. The native scanner magnification is read
from CZI metadata (defaulting to 40x if unavailable), and a downsample factor
is computed to reach the desired working magnification (`TARGET_MAG = 20x` by
default). Patches are square tiles of `PATCH_SIZE = 512` px at the target
magnification, laid out on a regular non-overlapping grid starting at the
slide's bounding-box origin.

Reads are done in row-strips (`ROW_BATCH` rows of patches per CZI call) on a
dedicated I/O thread, overlapped with per-strip statistics computation on the
main thread. If a strip read fails, the pipeline falls back to reading that
strip's rows individually and, failing that, skips the row (recorded, not
silently dropped).

For every patch, the pipeline records: mean and standard deviation per RGB
channel, and the fraction of near-black and near-white pixels.

## 4. Tissue segmentation (background removal)

Patches are classified as tissue or background/white in two stages:

1. **Pixel-rule pre-filter** (fast, deterministic): a patch is flagged white if
   any of the following hold — `black_pixel_ratio > 0.20` (scan edge / fold
   artifact), `white_pixel_ratio > 0.90` (mostly glass), or all three channel
   means exceed 210 with all three channel standard deviations below 12
   (uniformly pale, no texture).
2. **SVM classifier** (learned, applied only to patches that pass stage 1): a
   pretrained SVM (`svm_model.pkl`) operating on `[avg_G, avg_B, std_R]`
   assigns the remaining ambiguous patches to tissue or background.

The result is a per-slide, per-patch `white_label`, from which a "non-white"
(tissue) patch table is derived — this is the patch set that Stage C
processes. This tissue-vs-background decision is shared across every
model in the pipeline family, so all models see the same tissue mask.

## 5. Patch-level feature extraction (CoNCH v1.5)

TITAN does not define its own patch-level visual encoder; instead each
whole-slide patch is embedded with **CoNCH v1.5**, obtained directly from the
loaded TITAN model via `titan.return_conch()`. This guarantees the patch
encoder and its preprocessing exactly match the one TITAN was trained/aligned
against.

- Each 512×512 px tissue patch is transformed with CoNCH v1.5's own
  evaluation transform (`Resize(448) → CenterCrop(448) → ToTensor →
  Normalize`) and passed through CoNCH v1.5 to obtain a single 768-dimensional
  feature vector per patch.
- This differs from the sibling scripts for other backbones (UNI2-h,
  H-optimus-1, Virchow2, CONCH v1), which extract five 224×224 crops per patch
  for their own encoders — TITAN's aggregator expects exactly one vector per
  patch, so no five-crop scheme is used here.
- Patches are encoded in batches (`PATCH_BATCH_SIZE = 32` per forward pass) on
  GPU under fp16 autocast, with patch reads pipelined on a separate I/O thread
  so CZI decoding overlaps with GPU inference.

## 6. Slide-level aggregation (TITAN)

All per-patch CoNCH v1.5 features for a slide are aggregated into one
768-dimensional slide embedding by `titan.encode_slide_from_patch_features()`.

**Coordinate handling.** TITAN's aggregator needs each patch's true level-0
(full native-resolution) pixel coordinate, paired with the patch size at
level 0 (`patch_size_lv0 = PATCH_SIZE × downsample`, e.g. 1024 px on a 40x
slide tiled at 20x). Internally, TITAN floor-divides each coordinate by
`patch_size_lv0` to place patches on an integer grid and scatter-adds their
features into it (`preprocess_features`); passing coordinates in the wrong
frame relative to `patch_size_lv0` does not raise an error — it silently
collapses multiple neighbouring patches into the same grid cell. This
pipeline always converts patch coordinates back to true level-0 pixels before
calling TITAN, and explicitly asserts that every patch maps to a distinct grid
cell (no collapsed cells) before proceeding.

**Architecture.** TITAN's slide encoder is a Vision Transformer (embedding
dimension 768, 12 attention heads) that treats each patch feature as
one token. Rather than learned absolute position embeddings, it uses
**ALiBi** (Attention with Linear Biases): for every pair of patches, a
distance-dependent bias (per attention head) is added to the attention
scores, computed directly from the patches' true spatial grid coordinates.
This lets the model reason about the relative spatial arrangement of tissue
across the whole slide in a single global self-attention pass, rather than
processing patches independently or in fixed local windows. A background mask
derived from the patch grid ensures the encoder only attends over grid cells
that actually contain a tissue patch. The final pooled representation from
this transformer (via TITAN's attentional pooling head, `no_proj=True`) is the
768-dimensional slide embedding.

Because attention is computed globally over every patch in a slide
simultaneously, the aggregation step's cost scales with the **square** of the
number of tissue patches — independent of `PATCH_BATCH_SIZE`, which only
controls the batch size of the earlier, per-patch CoNCH encoding step.

**Memory-efficient attention implementation.** In TITAN's released
implementation the ALiBi bias is materialised as a dense
`(1, num_heads, N+1, N+1)` float32 tensor before being passed to
`scaled_dot_product_attention` as an explicit float mask. For a slide with
N ≈ 22,500 tissue patches this single tensor requires
`12 × 22,501² × 4 B ≈ 22.6 GB`, which exceeds the available memory of the
24 GB GPU used here — even though the same slide can be encoded in ~13 s using
3.6 GB once the attention is tiled. The cost is therefore an artifact of how
the bias is *stored*, not of the computation itself.

This pipeline replaces that with an equivalent **tiled** formulation. Since
the released code computes the bias as `distance_matrix × per-head slope`, the
dense tensor is simply `num_heads` scalar multiples of a single pairwise
Euclidean distance matrix, and is fully determined by the token grid
coordinates (N × 2) plus `num_heads` scalars. Attention is therefore evaluated
in chunks of query rows, with each chunk's distance bias computed on the fly.
Each chunk computes complete score rows over all keys, so every softmax is
taken over the full key set and the result is **mathematically identical** to
the released implementation — this is a change of memory layout, not an
approximation. Peak memory becomes linear rather than quadratic in patch
count (`O(chunk × N)`), allowing every slide in the cohort to be aggregated
over its complete patch set on a single 24 GB GPU. Equivalence is verified
numerically by `verify_titan_patch.py`, which compares per-head slopes, the
reconstructed dense bias, and end-to-end slide embeddings against the
unmodified model.

## 7. Output

One `.pt` file per slide is written to
`<OUTPUT_ROOT>/features/slide_aggregation/TITAN/conch1-5/<slide_stem>.pt`,
containing a single `(1, 768)` tensor — the slide-level TITAN embedding. This
follows the same `slide_aggregation/<method>/<model>` directory convention
used for other embedding methods (e.g. mean-pooled averaging) elsewhere in
this project, allowing TITAN's embeddings to be consumed by the same
downstream classification/evaluation code as those methods. Writes are atomic
(written to a temporary file, then renamed), and a slide is skipped entirely
on any re-run once its output file exists, making the full pipeline
resumable at slide granularity after an interruption.

## 8. Reproducibility and safeguards

**Pinned model revision.** TITAN is loaded with `trust_remote_code=True`,
meaning its Python implementation is fetched from the Hugging Face Hub
alongside the weights. The pipeline pins an explicit `revision` commit hash
(`TITAN_REVISION`) so that both the weights and the model code are fixed for
every run, and so the tiled-attention implementation described in §6 always
applies to the source it was validated against. The patch installs itself
against the loaded model's own classes and raises immediately if the expected
structure is absent, so a run can never silently fall back to unpatched
behaviour.

**Fallback for memory exhaustion.** Because the aggregation GPU may be shared
with other processes, available memory can change between the point where the
pipeline sizes its workload and the point where the allocation is made. If
aggregation runs out of memory despite the linear-memory implementation, the
pipeline retries with progressively **decimated** patch sets — keeping every
*s*-th patch along both grid axes, leaving `patch_size_lv0` and all spatial
coordinates unchanged so retained patches keep their true relative geometry
and remain in-distribution for TITAN. With the tiled implementation active
this path is a safety net that is not expected to trigger; the per-slide log
records the patch count and stride actually used, so any slide where it did
occur is identifiable after the fact and can be reported as using a subsample
of its tissue rather than the complete patch set.
