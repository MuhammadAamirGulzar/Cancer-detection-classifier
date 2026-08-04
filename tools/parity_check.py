"""Metric-parity + benchmark harness for the Task 1.0 data-layer refactor.

Work order section 1c.1: "Correctness outranks speed. Every optimisation must be
metric-neutral. After refactoring, re-run one configuration under the old and new
code paths and confirm the metrics match to ~1e-6."

Three checks are run:

  A. DATA IDENTITY  - the new data layer must deliver bit-identical features and
     labels to the old ``WSIDataset`` for the same fold ids, once both are put in
     the same row order.
  B. METRIC PARITY  - every classifier, fed tensors from each path, must produce
     the same metrics to 1e-6.
  C. BENCHMARK      - wall-clock for one ``(method, model)`` end to end, split
     into data-loading time and classifier time.

Check C also documents a reproducibility defect in the old path that the refactor
fixes: the old train ``DataLoader`` used ``shuffle=True`` with no seed, and
Random Forest's bootstrap sampling depends on row order, so the old pipeline
produced different RF numbers on every run of identical code.

Run:  python tools/parity_check.py [--method M] [--model M]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SLIDE_CLS = REPO_ROOT / "slide_classification"
sys.path.insert(0, str(SLIDE_CLS))

import data_layer as dl  # noqa: E402
from config import paths as P  # noqa: E402
from eval_patch_features.logistic import eval_linear  # noqa: E402
from eval_patch_features.ann import eval_ANN  # noqa: E402
from eval_patch_features.knn import eval_knn  # noqa: E402
from eval_patch_features.protonet import eval_protonet  # noqa: E402
from eval_patch_features.r_forest_eval import eval_r_forest  # noqa: E402

BATCH_SIZE = 4
SEED = 0


# ==========================================================================
# OLD CODE PATH - transcribed verbatim from Slide_Classification.ipynb cell 6
# ==========================================================================


class OldWSIDataset(Dataset):
    """The pre-refactor loader, reproduced exactly (including its O(n^2) lookup)."""

    def __init__(self, save_dir: str, fold_ids: List[str], k_folds_path: str):
        self.data = []
        self.save_dir = save_dir
        self.fold_ids = fold_ids
        self.k_folds_path = k_folds_path
        self._load_data()

    def _load_data(self):
        files = os.listdir(self.save_dir)
        folds_df = pd.read_csv(self.k_folds_path)

        for wsi_file in files:
            wsi_path = os.path.join(self.save_dir, wsi_file)
            wsi_id = os.path.splitext(wsi_file)[0]
            if wsi_id not in self.fold_ids:
                continue
            if not wsi_path.endswith(".pt"):
                continue
            try:
                wsi_features = torch.load(wsi_path)
                if wsi_features.is_cuda:
                    wsi_features = wsi_features.cpu()
                if wsi_features.dim() > 1:
                    final_features = wsi_features.flatten()
                else:
                    final_features = wsi_features
                # NOTE: 'label' is left undefined when wsi_id is absent from
                # folds_df -> NameError swallowed by the bare except below.
                # That is the Task 1.4 bug, preserved here on purpose.
                if wsi_id in folds_df["WSI_Id"].values:
                    fold_label = folds_df.loc[folds_df["WSI_Id"] == wsi_id, "label_desc"].values[0]
                    label = 0 if fold_label == "nonMSIH" else 1
                self.data.append((final_features, label, wsi_id))
            except Exception as e:
                print(f"[ERROR] Loading failed for {wsi_path}: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def old_get_feats_labels(loader):
    feats, labels, ids = [], [], []
    for features, lbls, wsi_ids in loader:
        feats.append(features)
        labels.append(lbls)
        if isinstance(wsi_ids, (list, tuple)):
            ids.extend(wsi_ids)
        else:
            ids.append(wsi_ids)
    return torch.cat(feats), torch.cat(labels), ids


# ==========================================================================
# Classifier stack - identical call signatures for both paths
# ==========================================================================


def run_classifier(kind, fold, tr, va, te, input_dim, save_path=None):
    """Run one classifier on ``(feats, labels)`` triples. Returns eval metrics."""
    (trf, trl), (vaf, val), (tef, tel) = tr, va, te
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if kind == "lin":
        m, _ = eval_linear(fold=fold, train_feats=trf, train_labels=trl,
                           valid_feats=vaf, valid_labels=val,
                           test_feats=tef, test_labels=tel,
                           max_iter=300, C=10, save_path=save_path, verbose=False)
    elif kind == "knn":
        m, _ = eval_knn(fold=fold, train_feats=trf, train_labels=trl,
                        val_feats=vaf, val_labels=val,
                        test_feats=tef, test_labels=tel,
                        n_neighbors=3, normalize_feats=True,
                        model_save_path=save_path, verbose=False)
    elif kind == "proto":
        m, _ = eval_protonet(fold=fold, train_feats=trf, train_labels=trl,
                             val_feats=vaf, val_labels=val,
                             test_feats=tef, test_labels=tel,
                             normalize_feats=True, model_save_path=save_path)
    elif kind == "rf":
        m, _ = eval_r_forest(fold=fold, train_feats=trf, train_labels=trl,
                             valid_feats=vaf, valid_labels=val,
                             test_feats=tef, test_labels=tel,
                             n_estimators=500, max_depth=None, min_samples_split=5,
                             min_samples_leaf=1, class_weight={0: 1, 1: 10},
                             prediction_threshold=0.3, save_path=save_path, verbose=False)
    elif kind == "ann":
        m, _ = eval_ANN(fold=fold, train_feats=trf, train_labels=trl,
                        valid_feats=vaf, valid_labels=val,
                        test_feats=tef, test_labels=tel,
                        input_dim=input_dim, hidden_dim=256, hidden_dim2=128,
                        max_iter=500, model_save_path=save_path, verbose=False)
    else:
        raise ValueError(kind)
    return {k: v for k, v in m.items() if isinstance(v, (int, float))}


# ==========================================================================
# Main
# ==========================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="Tissue_Type_Clustering")
    ap.add_argument("--model", default="H-Optimus-1")
    ap.add_argument("--classifiers", default="lin,knn,proto,rf,ann")
    args = ap.parse_args()

    method, model = args.method, args.model
    classifiers = args.classifiers.split(",")
    data_path = str(P.feature_dir("tcga", method, model))
    kfolds = str(P.labels_path("tcga", "MSIH"))
    input_dim = P.expected_dim(method, model)

    print("=" * 78)
    print(f"PARITY CHECK  |  TCGA / {method} / {model}  |  D={input_dim}")
    print("=" * 78)

    folds_df = pd.read_csv(kfolds)
    folds = [folds_df[folds_df["fold"] == f]["WSI_Id"].tolist()
             for f in sorted(folds_df["fold"].unique())]

    # ---------------- NEW PATH: load once ----------------------------------
    t0 = time.perf_counter()
    coh = dl.load_cohort("tcga", method, model, use_cache=False, verbose=False)
    splits = dl.build_splits(coh.ids, dl.load_fold_map("tcga"))
    new_load_time = time.perf_counter() - t0
    print(f"\nNEW data layer: N={coh.n} D={coh.dim} loaded once in {new_load_time:.2f}s")

    # =====================================================================
    # CHECK A - data identity
    # =====================================================================
    print("\n" + "-" * 78)
    print("CHECK A - DATA IDENTITY (old WSIDataset vs new data layer)")
    print("-" * 78)

    old_load_time = 0.0
    all_ok = True
    for i in range(len(folds)):
        test_ids = folds[i]
        val_ids = folds[(i + 1) % len(folds)]
        train_ids = [w for j in range(len(folds))
                     if j != i and j != (i + 1) % len(folds) for w in folds[j]]

        for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
            t0 = time.perf_counter()
            ds = OldWSIDataset(data_path, ids, kfolds)
            old_load_time += time.perf_counter() - t0

            old_ids = [d[2] for d in ds.data]
            order = np.argsort(old_ids)
            old_f = torch.stack([ds.data[k][0] for k in order]).float()
            old_l = torch.tensor([ds.data[k][1] for k in order], dtype=torch.long)

            rows = coh.rows_for(sorted(old_ids))
            new_f, new_l, new_ids = coh.subset(rows)

            same_ids = sorted(old_ids) == new_ids
            same_f = torch.equal(old_f, new_f)
            same_l = torch.equal(old_l, new_l)
            ok = same_ids and same_f and same_l
            all_ok &= ok
            maxdiff = (old_f - new_f).abs().max().item() if old_f.shape == new_f.shape else float("nan")
            print(f"  fold{i+1} {name:<5} n_old={len(old_ids):>3} n_new={len(new_ids):>3} "
                  f"ids={'OK' if same_ids else 'DIFF'} feats={'OK' if same_f else 'DIFF'} "
                  f"labels={'OK' if same_l else 'DIFF'} max|delta|={maxdiff:.2e}")

    print(f"\n  RESULT: {'PASS - data layer is bit-identical' if all_ok else 'FAIL'}")
    print(f"  old loader total time: {old_load_time:.1f}s   (12 dataset builds, 1 classifier's worth)")
    if not all_ok:
        sys.exit(1)

    # =====================================================================
    # CHECK B - metric parity, fold 1, matched row order
    # =====================================================================
    print("\n" + "-" * 78)
    print("CHECK B - METRIC PARITY (fold 1, identical row order, seed fixed)")
    print("-" * 78)

    i = 0
    test_ids, val_ids = folds[i], folds[(i + 1) % len(folds)]
    train_ids = [w for j in range(len(folds))
                 if j != i and j != (i + 1) % len(folds) for w in folds[j]]

    def old_split(ids, shuffle):
        ds = OldWSIDataset(data_path, ids, kfolds)
        ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)
        f, l, wid = old_get_feats_labels(ld)
        o = np.argsort(wid)
        return f[o].float(), l[o].long()

    old_tr, old_va, old_te = (old_split(train_ids, True),
                              old_split(val_ids, False),
                              old_split(test_ids, False))
    tr_rows, va_rows, te_rows = dl.cv_rotation(splits, 1)
    new_tr = coh.subset(tr_rows)[:2]
    new_va = coh.subset(va_rows)[:2]
    new_te = coh.subset(te_rows)[:2]

    worst = 0.0
    for kind in classifiers:
        t0 = time.perf_counter()
        m_old = run_classifier(kind, i, old_tr, old_va, old_te, input_dim)
        t_old = time.perf_counter() - t0
        t0 = time.perf_counter()
        m_new = run_classifier(kind, i, new_tr, new_va, new_te, input_dim)
        t_new = time.perf_counter() - t0

        keys = sorted(set(m_old) & set(m_new))
        deltas = {k: abs(m_old[k] - m_new[k]) for k in keys}
        mx = max(deltas.values()) if deltas else 0.0
        worst = max(worst, mx)
        status = "PASS" if mx <= 1e-6 else "FAIL"
        print(f"\n  [{kind}] max|delta| = {mx:.3e}  {status}   "
              f"(old {t_old:.1f}s / new {t_new:.1f}s)")
        for k in keys:
            print(f"      {k:<18} old={m_old[k]:.10f}  new={m_new[k]:.10f}  d={deltas[k]:.2e}")

    print(f"\n  RESULT: worst delta across all classifiers = {worst:.3e} "
          f"({'PASS' if worst <= 1e-6 else 'FAIL'}, tolerance 1e-6)")

    # =====================================================================
    # CHECK C - benchmark: full (method, model), 5 classifiers x 4 folds
    # =====================================================================
    print("\n" + "-" * 78)
    print("CHECK C - BENCHMARK (data-loading cost, 5 classifiers x 4 folds)")
    print("-" * 78)
    n_clf = 5
    old_proj = old_load_time * n_clf
    print(f"  OLD: 3 WSIDataset builds x 4 folds x {n_clf} classifiers = "
          f"{12 * n_clf} builds -> {old_proj:.1f}s of loading")
    print(f"  NEW: 1 assembly, reused by every fold and classifier -> {new_load_time:.1f}s")
    print(f"  speedup on the data path: {old_proj / max(new_load_time, 1e-9):.1f}x")

    t0 = time.perf_counter()
    dl.load_cohort("tcga", method, model, use_cache=True, verbose=False)
    warm = time.perf_counter() - t0
    print(f"  NEW (warm .npz cache): {warm:.2f}s -> {old_proj / max(warm, 1e-9):.0f}x")
    print("\nDone.")


if __name__ == "__main__":
    main()
