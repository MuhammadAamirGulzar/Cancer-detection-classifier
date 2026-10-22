"""Slide-level features for the four aggregation methods, in one pass over a slide's patch features.

Each method follows the published TCGA/PAIP notebooks exactly:

  Averaging                             (1, D)  mean over each patch's rows (crops), then over patches
                                                (slide_aggregation/averaging/Averaging_Aggregation_*.ipynb)
  Caption_based_aggregation             (14, D) patch = mean of its rows; patches grouped by the CONCH
  Caption_based_aggregation_15_classes  (15, D) zero-shot caption label of the whole patch; mean per group,
                                                zeros for an empty group (caption_generation/Caption_Based_Aggregation.ipynb)
  Tissue_Type_Clustering                (9, D)  rows (crops) grouped by the encoder's CRC-100K tissue ANN; mean per
                                                class, zeros for an empty class. Two labelling rules exist (--ttc-rule):
                                                  top_left  every crop of a patch takes the label of its top-left crop.
                                                            This is what the published TCGA slides and the original 73
                                                            PAIP slides contain (tissue_type/*_fivecrop.ipynb looks the
                                                            crop labels up by patch name and falls back to crop 1).
                                                  per_crop  every crop is grouped by its own label. Used by the 5
                                                            gap-filled PAIP slides and by *_TTC_SurGen.py.

A patch file may hold (5, D) five-crop rows or a single (D,) no-crop vector; a no-crop patch is one row.

--composition-out ROOT also writes composition-aware variants of the grouping methods (Caption-14/15, TTC):
  ROOT/append/<method>/<model>/<slide>.pt   flattened group features followed by sqrt(share) * scale, (G*D + G,)
  ROOT/only/<method>/<model>/<slide>.pt     sqrt(share) alone, (G,)
share = fraction of the slide's grouped units in each group (patches for Caption, rows/crops for TTC, after the
TTC labelling rule); sqrt = Hellinger transform of proportions; scale (--composition-scale, a JSON {method: s})
makes the share block as long as a typical group row, so it is not drowned by thousands of embedding values.

--caption-level crop (five-crop features only) groups every CROP by its own CONCH v1 zero-shot label instead of
the whole patch's label, from experiments/crop_grouping/01_label_crops.py (same model, preprocess and weights as
the published labels; row i of a patch file is FiveCrop view i). The patch set is unchanged: only patches that have
a published label for that caption variant are used. Group mean over crops, zeros for an empty group.

Use --check-against to rebuild slides whose published slide features exist and compare them exactly
before writing anything new.

  python slide_aggregation/build_slide_features.py --cohort tcga --model Conch1_5 \
      --features dataset/Features/Conch_v1_5_fivecrop_256 --out dataset/slide_aggregation
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "slide_classification"))
from config import paths as P  # noqa: E402

COHORT_DIRS = {"tcga": "dataset", "paip": "paip_data"}
CAPTION_CSV = {
    "tcga": {"Caption_based_aggregation": "classification_results_tcga.csv",
             "Caption_based_aggregation_15_classes": "classification_results_tcga_15_classes.csv"},
    "paip": {"Caption_based_aggregation": "classification_results_paip.csv",
             "Caption_based_aggregation_15_classes": "classification_results_paip_15_classes.csv"},
}
# Group order of the published slide tensors (rows), from Caption_Based_Aggregation.ipynb.
CAPTION_GROUPS = {
    "Caption_based_aggregation": ['ADI', 'DEB', 'LYM', 'PLC', 'LYA', 'MUC', 'MUS', 'NORM', 'ADE', 'STR', 'MES',
                                  'CAR', 'PDC', 'SIG'],
    "Caption_based_aggregation_15_classes": ['ADI', 'DEB', 'TIL', 'PLC', 'LYA', 'LYM', 'MUS', 'MUC', 'NORM', 'ADE',
                                             'STR', 'CAR', 'PDC', 'SIG', 'BACK'],
}
TISSUE_ANN = {"Conch1_5": "ann_crc100k_conch15.pth", "ConchV1": "ann_crc100k_ConchV1.pth",
              "H-Optimus-1": "ann_crc100k_Hoptimus.pth", "UNI2": "ann_crc100k_uni2.pth",
              "Virchow2": "ann_crc100k_Virchow2.pth"}
METHODS = ["Averaging", "Caption_based_aggregation", "Caption_based_aggregation_15_classes",
           "Tissue_Type_Clustering"]
N_TISSUE = 9


def caption_labels(cohort):
    """{method: {slide: {patch_stem: label}}}"""
    out = {}
    for method, name in CAPTION_CSV[cohort].items():
        df = pd.read_csv(HERE / "caption_generation" / "Results" / name, usecols=["patch_name", "label"])
        stem = df["patch_name"].str.replace(".png", "", regex=False)
        if cohort == "tcga":
            slide = df["patch_name"].str.split("_").str[:2].str.join("_")
        else:  # PAIP slide names have four underscore parts (training_data_01_MSIH)
            slide = df["patch_name"].str.split("_").str[:4].str.join("_")
        lab = {}
        for s, st, l in zip(slide, stem, df["label"]):
            lab.setdefault(s, {})[st] = l
        out[method] = lab
    return out


def crop_caption_labels(root, slide):
    """{14|15: {patch_stem: (5,) crop labels}} from 01_label_crops.py; class index = CAPTION_GROUPS position."""
    with np.load(Path(root) / f"{slide}.npz") as z:
        stems = [n[:-4] for n in z["names"]]
        return {k: dict(zip(stems, z[f"lab{k}"][:, :5].astype(np.int64))) for k in (14, 15)}


def build_slide(slide_dir, labels, tissue_ann, device, pool, dim, methods, ttc_rule, crop_labels=None,
                skip_zero_filled=False):
    stems = sorted(f[:-3] for f in os.listdir(slide_dir) if f.endswith(".pt"))

    def load(s):
        try:
            return torch.load(slide_dir / f"{s}.pt", map_location="cpu")
        except Exception as e:                       # damaged file: report it, never skip it silently
            return e
    tensors = list(pool.map(load, stems))
    bad = [(slide_dir / f"{s}.pt", f"{type(t).__name__}: {t}") for s, t in zip(stems, tensors) if isinstance(t, Exception)]
    if bad and skip_zero_filled and all(not any(p.read_bytes()) for p, _ in bad):
        # entirely zero-filled files: the published notebooks skipped these, so leave them out the same way
        for p, _ in bad:
            print(f"  skipped zero-filled patch feature {p}", flush=True)
        keep = [i for i, t in enumerate(tensors) if not isinstance(t, Exception)]
        stems, tensors, bad = [stems[i] for i in keep], [tensors[i] for i in keep], []
    if bad:
        return None, bad, None
    rows = [t.reshape(-1, dim).float() for t in tensors]          # (n_crops, D) per patch
    patch_mean = torch.stack([r.mean(dim=0) for r in rows])         # (P, D)
    out, counts = {}, {}
    if "Averaging" in methods:
        out["Averaging"] = torch.cat([r.mean(dim=0, keepdim=True) for r in rows]).mean(dim=0, keepdim=True)

    slide = slide_dir.name
    for method, groups in CAPTION_GROUPS.items():
        if method not in methods:
            continue
        lab = labels[method].get(slide, {})
        feats = []
        if crop_labels is not None:                                 # crop-level grouping
            keep = [i for i, s in enumerate(stems) if s in lab]
            assert all(len(rows[i]) == 5 for i in keep), f"{slide}: crop-level captions need five-crop features"
            X = torch.cat([rows[i] for i in keep]) if keep else torch.zeros(0, dim)
            y = np.concatenate([crop_labels[len(groups)][stems[i]] for i in keep]) if keep else np.zeros(0, int)
            for g in range(len(groups)):
                feats.append(X[y == g].mean(dim=0, keepdim=True) if (y == g).any() else torch.zeros(1, dim))
            out[method] = torch.cat(feats)
            counts[method] = np.array([(y == g).sum() for g in range(len(groups))], dtype=np.float64)
            continue
        n = []
        for g in groups:
            idx = [i for i, s in enumerate(stems) if lab.get(s) == g]
            feats.append(patch_mean[idx].mean(dim=0, keepdim=True) if idx else torch.zeros(1, dim))
            n.append(len(idx))
        out[method] = torch.cat(feats)
        counts[method] = np.array(n, dtype=np.float64)

    if "Tissue_Type_Clustering" not in methods:
        return out, [], counts
    crops = torch.cat(rows)                                         # (all crops, D)
    with torch.no_grad():
        pred = torch.cat([tissue_ann(crops[i:i + 4096].to(device)).argmax(dim=1).cpu()
                          for i in range(0, len(crops), 4096)]).numpy()
    if ttc_rule == "top_left":                                      # label of each patch's first row, for all its rows
        first = np.cumsum([0] + [len(r) for r in rows[:-1]])
        pred = np.repeat(pred[first], [len(r) for r in rows])
    c = crops.numpy().astype(np.float32)
    g = np.zeros((N_TISSUE, dim), dtype=np.float32)
    for k in range(N_TISSUE):
        if (pred == k).any():
            g[k] = c[pred == k].mean(axis=0)
    out["Tissue_Type_Clustering"] = torch.from_numpy(g)
    counts["Tissue_Type_Clustering"] = np.array([(pred == k).sum() for k in range(N_TISSUE)], dtype=np.float64)
    return out, [], counts


def composition(counts):
    """sqrt of each group's share of the slide's grouped units (zeros if nothing was grouped)."""
    tot = counts.sum()
    return torch.from_numpy(np.sqrt(counts / tot) if tot > 0 else counts).float()


def save_pt(t, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(t, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=sorted(COHORT_DIRS))
    ap.add_argument("--model", required=True, choices=sorted(TISSUE_ANN))
    ap.add_argument("--features", required=True, help="patch-feature folder, relative to KSA_project2 or absolute")
    ap.add_argument("--out", help="root holding <method>/<model>/<slide>.pt (relative to KSA_project2 or absolute)")
    ap.add_argument("--slides", default=None, help="comma-separated subset")
    ap.add_argument("--methods", default=",".join(METHODS), help="comma-separated subset of " + ",".join(METHODS))
    ap.add_argument("--ttc-rule", choices=["top_left", "per_crop"], default=None,
                    help="required when Tissue_Type_Clustering is built (see the module docstring)")
    ap.add_argument("--caption-level", choices=["patch", "crop"], default="patch",
                    help="patch: published (one label per patch); crop: every crop by its own label")
    ap.add_argument("--crop-labels", default=None,
                    help="<root>/<cohort> of 01_label_crops.py output; required for --caption-level crop")
    ap.add_argument("--skip-zero-filled", action="store_true",
                    help="leave out patch files that are entirely zero bytes (as the published notebooks did); "
                         "any other unreadable file still stops its slide")
    ap.add_argument("--check-against", default=None,
                    help="root of published slide features; compare instead of writing (with --composition-out: "
                         "compare the group features AND write the composition variants)")
    ap.add_argument("--composition-out", default=None, help="root for the append/ and only/ composition variants")
    ap.add_argument("--composition-scale", default=None, help="JSON {method: scale} for this model")
    a = ap.parse_args()

    methods = a.methods.split(",")
    assert all(m in METHODS for m in methods), methods
    if "Tissue_Type_Clustering" in methods and a.ttc_rule is None:
        ap.error("--ttc-rule is required for Tissue_Type_Clustering")
    if a.caption_level == "crop" and not a.crop_labels:
        ap.error("--crop-labels is required for --caption-level crop")
    comp_scale = None
    if a.composition_out:
        import json
        if not a.composition_scale:
            ap.error("--composition-scale is required with --composition-out")
        comp_scale = json.loads(Path(a.composition_scale).read_text())
    base = P.BASE_ROOT
    feat_root = Path(a.features) if os.path.isabs(a.features) else base / a.features
    dim = P.MODEL_BASE_DIMS[a.model]
    slides = sorted(d.name for d in os.scandir(feat_root) if d.is_dir())
    if a.slides:
        slides = [s for s in slides if s in set(a.slides.split(","))]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tissue_ann = torch.load(HERE.parent / "TissueClassifier_CRC100K" / "models" / TISSUE_ANN[a.model],
                            map_location=device, weights_only=False).to(device).eval()
    labels = caption_labels(a.cohort)

    worst = {m: 0.0 for m in methods}
    mismatch = {m: [] for m in methods}
    unreadable = []
    with ThreadPoolExecutor(16) as pool:
        for slide in tqdm(slides, desc=f"{a.cohort} {a.model}", unit="slide"):
            crop_labels = crop_caption_labels(a.crop_labels, slide) if a.caption_level == "crop" else None
            res, bad, counts = build_slide(feat_root / slide, labels, tissue_ann, device, pool, dim, methods,
                                           a.ttc_rule, crop_labels, a.skip_zero_filled)
            if bad:                                   # slide not written: its patch features must be repaired first
                unreadable += bad
                continue
            for m, t in res.items():
                if a.check_against:
                    ref_root = Path(a.check_against) if os.path.isabs(a.check_against) else base / a.check_against
                    ref = torch.load(ref_root / m / a.model / f"{slide}.pt", map_location="cpu").float().reshape(t.shape)
                    w = (t - ref).abs().max().item()
                    worst[m] = max(worst[m], w)
                    if w >= 1e-4:
                        mismatch[m].append(slide)
                elif a.out:
                    out_root = Path(a.out) if os.path.isabs(a.out) else base / a.out
                    save_pt(t, out_root / m / a.model / f"{slide}.pt")
                if comp_scale is not None and m in counts:
                    root = Path(a.composition_out)
                    c = composition(counts[m])
                    save_pt(torch.cat([t.flatten().float(), c * float(comp_scale[m])]),
                            root / "append" / m / a.model / f"{slide}.pt")
                    save_pt(c, root / "only" / m / a.model / f"{slide}.pt")
    if unreadable:
        print(f"UNREADABLE PATCH FEATURES: {len(unreadable)} files in "
              f"{len({str(p.parent) for p, _ in unreadable})} slide(s); those slides were NOT written:")
        for p, e in unreadable:
            print(f"  {p}  ({e})")
    if a.check_against:
        for m, w in worst.items():
            print(f"  {m:38s} max abs diff vs published: {w:.2e}  {'MATCH' if w < 1e-4 else 'DIFFERENT'}")
            if mismatch[m]:
                print(f"  MISMATCH {m}: {','.join(mismatch[m])}")


if __name__ == "__main__":
    main()

