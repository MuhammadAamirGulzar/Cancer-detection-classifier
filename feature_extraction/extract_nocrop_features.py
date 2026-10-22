"""No-crop patch features for several encoders in one pass over the patches.

Each non-white 512x512 patch is read and decoded once and handed whole (no crop) to every
selected encoder, through that encoder's own transform, so the drive is read once for all of them.

Per encoder, model loading, transform and precision are those of its five-crop run. They were
verified on 2026-10-09 by recomputing stored five-crop features of both cohorts with this setup
(cosine >= 0.9999996):

  encoder      input                                             precision          output
  uni2         timm create_transform: Resize 224 (bilinear)      fp32, no_grad      1536
  h_optimus_1  Resize((224,224)) bilinear, H-optimus mean/std    fp16 autocast      1536
  conch_v1     CONCH preprocess: Resize 448 (bicubic), CLIP norm fp32, no_grad      512  (encode_image, no projection)
  conch_v1_5   TITAN return_conch(): Resize 448 (bilinear)       fp32, no_grad      768
  virchow2     Resize((224,224)) bilinear, ImageNet mean/std     fp16 autocast      2560 (class token + mean patch token)

A whole 512x512 patch goes through each transform without losing any border (square input).

Output, per encoder: <cohort>/Features/<folder>/<slide>/<patch>.pt, one float32 tensor (D,).
Only the non-white patches of <cohort>/final_merged_without_white.csv are processed.
Resumable per patch and encoder; files are written atomically (.tmp, then rename).

Usage (exaonepath env):
  python feature_extraction/extract_nocrop_features.py --cohort tcga --models conch_v1,conch_v1_5,virchow2
  python feature_extraction/extract_nocrop_features.py --cohort paip --models all
  python feature_extraction/extract_nocrop_features.py --cohort paip --models all --check   # counts only, no GPU

Speed (RTX 4090, all encoders resident): GPU-bound at about 38 patches/s for conch_v1 + conch_v1_5
+ virchow2 and 23 patches/s for all five; CONCH v1.5 in fp32 at 448 px is most of the cost.
"""

from __future__ import annotations

import os

# The gated models load from the local Hugging Face cache. This must be set before
# torchvision / timm / transformers import huggingface_hub.
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import contextlib
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                                   # repo-local `conch` package
sys.path.insert(0, str(HERE.parent / "slide_classification"))   # config.paths
from config import paths as P  # noqa: E402

CONCH_V1_CKPT = HERE / "checkpoints" / "CONCH" / "pytorch_model.bin"

COHORTS = {
    "tcga": P.BASE_ROOT / "dataset",
    "paip": P.BASE_ROOT / "paip_data",
}


# --------------------------------------------------------------------------
# Encoders
# --------------------------------------------------------------------------


@dataclass
class Encoder:
    name: str
    dim: int
    fp16: bool           # fp16 autocast + inference_mode; otherwise fp32 + no_grad
    net: torch.nn.Module
    transform: Callable  # PIL image -> (3, H, W)
    call: Callable       # (net, batch) -> raw output, run inside the precision context
    post: Callable       # raw output -> (B, dim), run after it

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.to(DEVICE, non_blocking=True)
        if self.fp16:
            with torch.inference_mode():
                with torch.autocast(device_type=DEVICE.type, dtype=torch.float16):
                    out = self.call(self.net, batch)
                out = self.post(out)
        else:
            with torch.no_grad():
                out = self.post(self.call(self.net, batch))
        return out.float().cpu()


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _identity(x):
    return x


def load_uni2() -> Encoder:
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    kw = dict(img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5, embed_dim=1536,
              mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked,
              act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True)
    net = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **kw).to(DEVICE).eval()
    tf = create_transform(**resolve_data_config(net.pretrained_cfg, model=net))
    return Encoder("uni2", 1536, False, net, tf, lambda n, x: n(x), _identity)


def load_h_optimus_1() -> Encoder:
    import timm
    from torchvision import transforms

    net = timm.create_model("hf-hub:bioptimus/H-optimus-1", pretrained=True,
                            init_values=1e-5, dynamic_img_size=False).to(DEVICE).eval()
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517)),
    ])
    return Encoder("h_optimus_1", 1536, True, net, tf, lambda n, x: n(x), _identity)


def load_conch_v1() -> Encoder:
    from conch.open_clip_custom import create_model_from_pretrained

    net, preprocess = create_model_from_pretrained("conch_ViT-B-16", str(CONCH_V1_CKPT))
    return Encoder("conch_v1", 512, False, net.to(DEVICE).eval(), preprocess,
                   lambda n, x: n.encode_image(x, proj_contrast=False, normalize=False), _identity)


def load_conch_v1_5() -> Encoder:
    from transformers import AutoModel

    titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    net, tf = titan.return_conch()
    return Encoder("conch_v1_5", 768, False, net.to(DEVICE).eval(), tf,
                   lambda n, x: n(x), _identity)


def load_virchow2() -> Encoder:
    import timm
    from timm.data import resolve_data_config
    from timm.layers import SwiGLUPacked
    from torchvision import transforms

    net = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True,
                            mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU).to(DEVICE).eval()
    cfg = resolve_data_config(net.pretrained_cfg, model=net)
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"]),
    ])

    def post(out):  # (B, 261, 1280): token 0 = class, 1-4 = registers, 5+ = patches
        return torch.cat([out[:, 0], out[:, 5:].mean(dim=1)], dim=-1)

    return Encoder("virchow2", 2560, True, net, tf, lambda n, x: n(x), post)


LOADERS: Dict[str, Callable[[], Encoder]] = {
    "uni2": load_uni2,
    "h_optimus_1": load_h_optimus_1,
    "conch_v1": load_conch_v1,
    "conch_v1_5": load_conch_v1_5,
    "virchow2": load_virchow2,
}
OUTPUT_FOLDERS = {"uni2": "UNI2H_nocrop", "h_optimus_1": "H_Optimus_1_nocrop", "conch_v1": "Conch_v1_nocrop",
                  "conch_v1_5": "Conch_v1_5_nocrop", "virchow2": "Virchow2_nocrop"}


# --------------------------------------------------------------------------
# Work list
# --------------------------------------------------------------------------


@dataclass
class Job:
    patch_path: str
    save_paths: Dict[str, str]   # encoder -> .pt path, only for encoders still missing this patch


def build_jobs(cohort_root: Path, out_root: Path, models: List[str], slide_names=None):
    """One job per non-white patch that at least one selected encoder still lacks."""
    names = pd.read_csv(cohort_root / "final_merged_without_white.csv", usecols=["patch_name"])
    valid = set(names["patch_name"].astype(str))
    patch_root = cohort_root / "patch_data"
    slides = sorted(d.name for d in os.scandir(patch_root) if d.is_dir())
    if slide_names is not None:
        missing = set(slide_names) - set(slides)
        if missing:
            raise SystemExit(f"unknown slide(s): {sorted(missing)}")
        slides = [s for s in slides if s in set(slide_names)]

    jobs, stats = [], {m: {"expected": 0, "done": 0} for m in models}
    for slide in tqdm(slides, desc="Listing slides", unit="slide"):
        pngs = sorted(f.name for f in os.scandir(patch_root / slide) if f.name in valid)
        if not pngs:
            tqdm.write(f"  no non-white patches for {slide}")
            continue
        existing = {}
        for m in models:
            d = out_root / OUTPUT_FOLDERS[m] / slide
            d.mkdir(parents=True, exist_ok=True)
            existing[m] = {f.name[:-3] for f in os.scandir(d) if f.name.endswith(".pt")}
        for png in pngs:
            stem = png[:-4]
            need = {}
            for m in models:
                stats[m]["expected"] += 1
                if stem in existing[m]:
                    stats[m]["done"] += 1
                else:
                    need[m] = str(out_root / OUTPUT_FOLDERS[m] / slide / f"{stem}.pt")
            if need:
                jobs.append(Job(str(patch_root / slide / png), need))
    return jobs, stats


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def save_atomic(tensor: torch.Tensor, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(tensor, tmp)
    os.replace(tmp, path)


def forward_safe(enc: Encoder, batch: torch.Tensor) -> torch.Tensor:
    """Forward a batch; on CUDA out-of-memory, retry in two halves (recursively)."""
    try:
        return enc.forward(batch)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(batch) == 1:
            raise
        tqdm.write(f"  [OOM] {enc.name}: batch of {len(batch)} split in two")
        half = len(batch) // 2
        return torch.cat([forward_safe(enc, batch[:half]), forward_safe(enc, batch[half:])])


def extract(jobs: List[Job], encoders: Dict[str, Encoder], batch_size: int, num_workers: int) -> Dict[str, int]:
    failed = {m: 0 for m in encoders}

    def load(job: Job) -> Dict[str, torch.Tensor]:
        img = Image.open(job.patch_path).convert("RGB")   # decoded once for all encoders
        return {m: encoders[m].transform(img) for m in job.save_paths}

    prefetch = batch_size * 4
    with ThreadPoolExecutor(num_workers) as loader, ThreadPoolExecutor(2) as saver:
        pending, saving = deque(), deque()
        job_iter = iter(jobs)

        def submit_next():
            job = next(job_iter, None)
            if job is not None:
                pending.append((job, loader.submit(load, job)))

        def collect_saves(max_queued):
            while saving and (saving[0][2].done() or len(saving) > max_queued):
                m, path, fut = saving.popleft()
                try:
                    fut.result()
                except Exception as e:
                    failed[m] += 1
                    tqdm.write(f"  [save failed] {m} {path}: {e}")

        for _ in range(prefetch):
            submit_next()

        pbar = tqdm(total=len(jobs), desc="Extracting (" + ", ".join(encoders) + ")", unit="patch")
        while pending:
            batch = []   # (job, {encoder: tensor})
            while pending and len(batch) < batch_size:
                job, fut = pending.popleft()
                submit_next()
                try:
                    batch.append((job, fut.result()))
                except Exception as e:
                    for m in job.save_paths:
                        failed[m] += 1
                    pbar.update(1)
                    tqdm.write(f"  [load failed] {job.patch_path}: {e}")
            if not batch:
                continue

            for m, enc in encoders.items():
                rows = [(job, x[m]) for job, x in batch if m in job.save_paths]
                if not rows:
                    continue
                try:
                    feats = forward_safe(enc, torch.stack([x for _, x in rows]))   # (b, D)
                except Exception as e:
                    failed[m] += len(rows)
                    tqdm.write(f"  [forward failed] {m}, {len(rows)} patches: {e}")
                    continue
                for (job, _), f in zip(rows, feats):
                    # clone(): a row view would otherwise save the whole (b, D) batch storage
                    saving.append((m, job.save_paths[m], saver.submit(save_atomic, f.clone(), job.save_paths[m])))
            collect_saves(max_queued=prefetch * len(encoders))
            pbar.update(len(batch))

        collect_saves(max_queued=0)
        pbar.close()
    return failed


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="No-crop patch features, several encoders in one pass")
    ap.add_argument("--cohort", required=True, choices=sorted(COHORTS))
    ap.add_argument("--models", required=True,
                    help="comma-separated subset of " + ",".join(LOADERS) + ", or 'all'")
    ap.add_argument("--slides", default=None, help="comma-separated slide folders (default: all)")
    ap.add_argument("--out-root", default=None,
                    help="where the <encoder>_nocrop folders go (default: <cohort>/Features)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=16, help="patch read/decode threads")
    ap.add_argument("--check", action="store_true", help="only report completeness, load no model")
    a = ap.parse_args()

    models = list(LOADERS) if a.models == "all" else [m.strip() for m in a.models.split(",")]
    unknown = [m for m in models if m not in LOADERS]
    if unknown:
        raise SystemExit(f"unknown model(s) {unknown}; choose from {list(LOADERS)}")

    cohort_root = COHORTS[a.cohort]
    out_root = Path(a.out_root) if a.out_root else cohort_root / "Features"
    slides = [s.strip() for s in a.slides.split(",")] if a.slides else None

    jobs, stats = build_jobs(cohort_root, out_root, models, slides)
    for m, s in stats.items():
        print(f"  {m:12s} -> {out_root / OUTPUT_FOLDERS[m]}: {s['done']:,} of {s['expected']:,} patches done")
    print(f"{len(jobs):,} patches still needed by at least one encoder")
    if a.check or not jobs:
        return

    encoders = {}
    for m in models:
        if any(m in j.save_paths for j in jobs):
            print(f"loading {m} ...", flush=True)
            encoders[m] = LOADERS[m]()
    if DEVICE.type == "cuda":
        print(f"models on GPU: {torch.cuda.memory_allocated() / 2**30:.1f} GB", flush=True)

    failed = extract(jobs, encoders, a.batch_size, a.workers)
    for m in encoders:
        n = sum(m in j.save_paths for j in jobs)
        print(f"  {m:12s}: {n - failed[m]:,} saved, {failed[m]:,} failed")
    if any(failed.values()):
        print("Re-run the same command to retry the failures.")


if __name__ == "__main__":
    main()
