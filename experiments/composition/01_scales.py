"""Scale of the appended composition block, per arm x aggregation x encoder.

The block is sqrt(share) (length 1, since the shares sum to 1) times s, where s = median L2 norm of a non-empty
group row of that arm's TCGA slide features. The composition then has the length of one typical group row: it
neither vanishes next to G x D embedding values (kNN / proto L2-normalise the whole vector; LR is L2-regularised)
nor dominates them. Computed on TCGA only and without labels, so one fixed transform serves every cohort.

Writes scales.json {arm: {model: {method: s}}}.
"""

import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = REPO.parent
SLIDES = {"fivecrop": BASE / "dataset" / "slide_aggregation",
          "nocrop": REPO / "experiments" / "nocrop_vs_fivecrop" / "features" / "tcga"}
METHODS = ["Caption_based_aggregation", "Caption_based_aggregation_15_classes", "Tissue_Type_Clustering"]
MODELS = ["UNI2", "H-Optimus-1", "ConchV1", "Conch1_5", "Virchow2"]


def main():
    out = {}
    for arm, root in SLIDES.items():
        for model in MODELS:
            for method in METHODS:
                files = sorted((root / method / model).glob("*.pt"))
                assert len(files) == 417, (arm, method, model, len(files))
                norms = []
                for f in files:
                    t = torch.load(f, map_location="cpu").float()
                    n = t.reshape(t.shape[0], -1).norm(dim=1)
                    norms.append(n[n > 0].numpy())
                s = float(np.median(np.concatenate(norms)))
                out.setdefault(arm, {}).setdefault(model, {})[method] = s
                print(f"{arm:8s} {model:12s} {method:38s} s = {s:.3f}")
    (HERE / "scales.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
