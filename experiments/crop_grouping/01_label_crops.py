"""CONCH v1 zero-shot caption labels for every five-crop crop, by the published labelling protocol.

The published Caption-14/15 labels (caption_generation/Results/classification_results_<cohort>[_15_classes].csv)
come from Zero_Shot_Label_Matching_Conch.ipynb: CONCH v1 (caption checkpoint, force_image_size=224), the whole
512 px patch through its preprocess (bicubic resize to 224), encode_image() @ zero-shot weights, argmax.

Here each patch is scored as 6 views with exactly that model and preprocess:
  views 0-4  torchvision FiveCrop(256) of the 512 px patch (top-left, top-right, bottom-left, bottom-right, centre),
             the same crops and order as the stored five-crop feature rows
  view  5    the whole patch, i.e. the published protocol; it is compared with the published CSV on every patch

Output, one file per slide (written atomically, so an interrupted run resumes):
  <out>/<cohort>/<slide>.npz  names (P,), sim14 (P,6,14) / sim15 (P,6,15) float16 similarities,
                              lab14 / lab15 (P,6) int8 argmax (taken in fp32), and the per-slide agreement of view 5
                              with the published labels (agree14/n14, agree15/n15, maxdiff14/maxdiff15 of the top score)

  python experiments/crop_grouping/01_label_crops.py --cohort tcga
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
BASE = REPO.parent
CAP = REPO / "slide_aggregation" / "caption_generation"
sys.path.insert(0, str(REPO / "feature_extraction"))
from conch.open_clip_custom import create_model_from_pretrained  # noqa: E402

PATCH_ROOT = {"tcga": BASE / "dataset" / "patch_data", "paip": BASE / "paip_data" / "patch_data"}
CSV = {"tcga": {14: "classification_results_tcga.csv", 15: "classification_results_tcga_15_classes.csv"},
       "paip": {14: "classification_results_paip.csv", 15: "classification_results_paip_15_classes.csv"}}
WEIGHTS = {14: "Conch_zeroshot_weights.pt", 15: "Conch_zeroshot_weights_15_classes.pt"}
# Column order of the weight matrices = key order of the prompt configs (Custom_prompts_all_per_class.json,
# config_15_classes.json) = group order of the published slide tensors.
CLASSES = {14: ['ADI', 'DEB', 'LYM', 'PLC', 'LYA', 'MUC', 'MUS', 'NORM', 'ADE', 'STR', 'MES', 'CAR', 'PDC', 'SIG'],
           15: ['ADI', 'DEB', 'TIL', 'PLC', 'LYA', 'LYM', 'MUS', 'MUC', 'NORM', 'ADE', 'STR', 'CAR', 'PDC', 'SIG',
                'BACK']}
FIVE = transforms.FiveCrop(256)


def slide_of(name, cohort):
    parts = name.split("_")
    return "_".join(parts[:2] if cohort == "tcga" else parts[:4])


class Views(Dataset):
    def __init__(self, items, pre):
        self.items, self.pre = items, pre

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path = self.items[i]
        try:
            img = Image.open(path).convert("RGB")
            if img.size != (512, 512):
                raise ValueError(f"size {img.size}, expected 512x512")
            return torch.stack([self.pre(v) for v in (*FIVE(img), img)]), i, True
        except Exception as e:                  # damaged image: its slide is not written, the run goes on
            print(f"UNREADABLE {path}: {e}", flush=True)
            return torch.zeros(6, 3, 224, 224), i, False


def save_npz(path, **arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(buf.getvalue())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=sorted(PATCH_ROOT))
    ap.add_argument("--out", default=str(BASE / "crop_captions"))
    ap.add_argument("--slides", default=None, help="comma-separated subset")
    ap.add_argument("--batch", type=int, default=64, help="patches per batch (x6 views)")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    out = Path(a.out) / a.cohort
    out.mkdir(parents=True, exist_ok=True)

    pub = {k: pd.read_csv(CAP / "Results" / CSV[a.cohort][k]).set_index("patch_name") for k in (14, 15)}
    names = sorted(set(pub[14].index) | set(pub[15].index))
    by_slide = {}
    for n in names:
        by_slide.setdefault(slide_of(n, a.cohort), []).append(n)
    slides = sorted(by_slide)
    if a.slides:
        slides = [s for s in slides if s in set(a.slides.split(","))]

    def done(s):
        try:
            with np.load(out / f"{s}.npz") as z:
                return len(z["names"]) == len(by_slide[s])
        except Exception:
            return False
    todo = [s for s in slides if not done(s)]
    print(f"{a.cohort}: {len(slides)} slides, {sum(len(by_slide[s]) for s in slides)} patches; "
          f"{len(slides) - len(todo)} slides already done, {sum(len(by_slide[s]) for s in todo)} patches to label")
    if not todo:
        return

    device = "cuda"
    model, pre = create_model_from_pretrained(
        "conch_ViT-B-16", str(CAP / "checkpoints" / "CONCH" / "pytorch_model.bin"), device=device,
        force_image_size=224)
    model.eval()
    W = {k: torch.load(CAP / "classifier_weights" / WEIGHTS[k], map_location=device).float() for k in (14, 15)}
    for k in (14, 15):
        assert W[k].shape == (512, k), W[k].shape

    items = [str(PATCH_ROOT[a.cohort] / s / n) for s in todo for n in by_slide[s]]
    owner = [(s, n) for s in todo for n in by_slide[s]]
    loader = DataLoader(Views(items, pre), batch_size=a.batch, num_workers=a.workers, pin_memory=True,
                        shuffle=False, persistent_workers=False, prefetch_factor=4)

    buf = {s: {14: [], 15: []} for s in todo}
    unreadable = {}
    cursor, pbar = 0, tqdm(total=len(items), unit="patch", smoothing=0.05)
    with torch.inference_mode():                        # published labelling ran in fp32 under inference_mode
        for x, idx, ok in loader:
            b = x.shape[0]
            for i in idx[~ok].tolist():
                unreadable.setdefault(owner[i][0], []).append(items[i])
            f = model.encode_image(x.view(-1, 3, 224, 224).to(device, non_blocking=True))   # normalised, projected
            for k in (14, 15):
                for s_i, sim in zip(idx.tolist(), (f @ W[k]).view(b, 6, k).cpu()):
                    buf[owner[s_i][0]][k].append(sim)
            pbar.update(b)
            # flush every slide whose patches are all scored (loader order is sequential)
            while cursor < len(todo) and len(buf[todo[cursor]][14]) == len(by_slide[todo[cursor]]):
                s = todo[cursor]
                if s in unreadable:                     # never write a slide with a missing patch
                    tqdm.write(f"  {s}: NOT written, {len(unreadable[s])} unreadable patch(es)")
                    del buf[s]
                    cursor += 1
                    continue
                sims = {k: torch.stack(buf[s][k]) for k in (14, 15)}          # (P, 6, k) fp32
                labs = {k: sims[k].argmax(dim=2).to(torch.int8) for k in (14, 15)}
                check = {}
                for k in (14, 15):
                    sel = [i for i, n in enumerate(by_slide[s]) if n in pub[k].index]
                    ref = pub[k].loc[[by_slide[s][i] for i in sel]]
                    mine = [CLASSES[k][j] for j in labs[k][sel, 5].tolist()]
                    top = sims[k][sel, 5].max(dim=1).values.double().numpy()
                    check[f"agree{k}"] = int((np.array(mine) == ref["label"].to_numpy()).sum())
                    check[f"n{k}"] = len(sel)
                    check[f"maxdiff{k}"] = float(np.abs(top - ref["score"].to_numpy()).max()) if sel else 0.0
                save_npz(out / f"{s}.npz", names=np.array(by_slide[s]),
                         sim14=sims[14].half().numpy(), sim15=sims[15].half().numpy(),
                         lab14=labs[14].numpy(), lab15=labs[15].numpy(), **check)
                if check["agree14"] != check["n14"] or check["agree15"] != check["n15"]:
                    tqdm.write(f"  {s}: whole-patch labels differ from published: 14-class "
                               f"{check['agree14']}/{check['n14']}, 15-class {check['agree15']}/{check['n15']}")
                del buf[s]
                cursor += 1
    pbar.close()
    print(f"done: {cursor - len(unreadable)} slides written to {out}")
    if unreadable:
        print(f"UNREADABLE patches in {len(unreadable)} slide(s); repair them and rerun (finished slides are kept):")
        for s, ps in unreadable.items():
            for q in ps:
                print(f"  {q}")
        sys.exit(1)


if __name__ == "__main__":
    main()
