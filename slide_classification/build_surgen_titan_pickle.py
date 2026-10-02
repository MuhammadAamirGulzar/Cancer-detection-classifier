"""Consolidate per-slide SurGen TITAN .pt embeddings into one pickle.

Reads every `.pt` file under the local TITAN/conch1-5 feature tree (one
(1, 768) tensor per slide, as produced by Complete_Pipeline/surgen_titan.py)
and writes a single pickle containing case_id, slide_id, cohort, and the
768-d feature vector for each slide.

Run: python slide_classification/build_surgen_titan_pickle.py
"""

import re
from pathlib import Path

import pandas as pd
import torch

# pandas >= 3.0 defaults to an Arrow/numpy-string-backed extension dtype for
# str columns, which older pandas (e.g. 2.x, used by downstream notebooks/
# envs) cannot unpickle (NotImplementedError in NDArrayBacked.__setstate__).
# Force classic object-dtype strings so the pickle stays readable everywhere.
pd.set_option("future.infer_string", False)

from config.paths import feature_dir

OUTPUT_PATH = Path(__file__).resolve().parent / "surgen_titan_features.pkl"

CASE_ID_RE = re.compile(r"_T(\d{3})_")
COHORT_RE = re.compile(r"^(SR\d+)_", re.IGNORECASE)


def main() -> None:
    feat_dir = feature_dir("surgen", "TITAN", "Conch1_5")
    pt_files = sorted(feat_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found under {feat_dir}")

    rows = []
    for pt_path in pt_files:
        slide_id = pt_path.stem

        cohort_match = COHORT_RE.match(slide_id)
        cohort = cohort_match.group(1) if cohort_match else None

        case_match = CASE_ID_RE.search(slide_id)
        if not case_match:
            print(f"  [SKIP] {slide_id}: could not extract case ID")
            continue
        case_id = f"{cohort}_{case_match.group(1)}" if cohort else case_match.group(1)

        tensor = torch.load(pt_path, map_location="cpu")
        features = tensor.squeeze(0).numpy()

        rows.append({
            "case_id": case_id,
            "slide_id": slide_id,
            "cohort": cohort,
            "features": features,
        })

    df = pd.DataFrame(rows)
    df.to_pickle(OUTPUT_PATH)
    print(f"Wrote {len(df)} slides -> {OUTPUT_PATH}")
    print(f"  Feature dim  : {df['features'].iloc[0].shape}")
    print(f"  Cohorts      : {df['cohort'].value_counts().to_dict()}")
    print(f"  Unique cases : {df['case_id'].nunique()}")


if __name__ == "__main__":
    main()
