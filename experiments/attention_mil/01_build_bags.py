"""Slide "bags" for attention-based MIL: one (N_patches, D) matrix per slide.

Each patch is the mean of its five crop vectors, from the same five-crop patch features as the benchmark
(the published Averaging slide feature is exactly the mean of these rows). Entirely zero-filled patch files are
left out, as the published slide features left them out; any other unreadable file stops that slide.

Output: features/<cohort>/<model>/<slide>.pt = {"x": float16 (N, D), "names": [patch stems]}, written atomically.

  <conda>/envs/exaonepath/python.exe experiments/attention_mil/01_build_bags.py --model H-Optimus-1
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
FEATURES = {
    "tcga": {"H-Optimus-1": "H_Optimus_1_fivecrop_Updated", "UNI2": "Uni_2h_fivecrop_Updated"},
    "paip": {"H-Optimus-1": "H_Optimus_1_fivecrop_Updated", "UNI2": "Uni_2h_fivecrop_Updated"},
}
COHORT_ROOT = {"tcga": BASE / "dataset", "paip": BASE / "paip_data"}
DIM = {"H-Optimus-1": 1536, "UNI2": 1536}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(DIM))
    ap.add_argument("--cohorts", default="paip,tcga")
    a = ap.parse_args()
    for cohort in a.cohorts.split(","):
        src = COHORT_ROOT[cohort] / "Features" / FEATURES[cohort][a.model]
        out = HERE / "features" / cohort / a.model
        out.mkdir(parents=True, exist_ok=True)
        slides = sorted(d.name for d in os.scandir(src) if d.is_dir())
        todo = [s for s in slides if not (out / f"{s}.pt").exists()]
        print(f"{cohort} {a.model}: {len(slides)} slides, {len(todo)} to build", flush=True)
        skipped, failed = 0, []
        with ThreadPoolExecutor(16) as pool:
            for s in tqdm(todo, desc=f"{cohort} {a.model}", unit="slide"):
                d = src / s
                stems = sorted(f[:-3] for f in os.listdir(d) if f.endswith(".pt"))

                def load(stem):
                    try:
                        return torch.load(d / f"{stem}.pt", map_location="cpu").reshape(-1, DIM[a.model]).float().mean(0)
                    except Exception as e:
                        return e
                vecs = list(pool.map(load, stems))
                bad = [(st, v) for st, v in zip(stems, vecs) if isinstance(v, Exception)]
                if any((d / f"{st}.pt").read_bytes().strip(b"\0") for st, _ in bad):
                    failed.append(s)
                    continue
                skipped += len(bad)
                keep = [i for i, v in enumerate(vecs) if not isinstance(v, Exception)]
                bag = {"x": torch.stack([vecs[i] for i in keep]).half(), "names": [stems[i] for i in keep]}
                tmp = out / f"{s}.pt.tmp"
                torch.save(bag, tmp)
                os.replace(tmp, out / f"{s}.pt")
        print(f"{cohort} {a.model}: done; {skipped} zero-filled patch files skipped; unreadable slides: {failed}", flush=True)
        if failed:
            sys.exit(1)


if __name__ == "__main__":
    main()
