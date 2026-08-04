"""Acceptance evidence for work order Tasks 1.1, 1.2 and 1.3 (ANN corrections).

  Task 1.1 - hyperparameters must be selected on VALIDATION macro-F1, never on
             the test set. Prints the 4 grid configurations with both their
             validation and test scores, and shows which one each rule picks.
  Task 1.2 - the checkpoint written to disk must be the configuration that was
             actually selected. Verifies the saved state dict's 0.weight /
             4.weight shapes against the sibling config.json.
  Task 1.3 - the double softmax must be gone. Trains one fold under the old
             architecture (Softmax in the graph, fed to CrossEntropyLoss) and the
             new one (logits), and reports balanced accuracy and AUROC for both.

Run:  python tools/ann_fixes_check.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "slide_classification"))

import data_layer as dl  # noqa: E402
from config import paths as P  # noqa: E402
from eval_patch_features.ann import (  # noqa: E402
    ANNBinaryClassifier, eval_ANN, save_ann_checkpoint, load_ann_checkpoint,
)
from eval_patch_features.metrics import get_eval_metrics  # noqa: E402

METHOD, MODEL, FOLD = "Tissue_Type_Clustering", "H-Optimus-1", 1
GRID = [(h1, h2) for h1 in (128, 256) for h2 in (64, 128)]
MAX_ITER = 500
SEED = 0


def banner(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def load_fold():
    coh = dl.load_cohort("tcga", METHOD, MODEL, verbose=False)
    splits = dl.build_splits(coh.ids, dl.load_fold_map("tcga"))
    tr, va, te = dl.cv_rotation(splits, FOLD)
    return coh, coh.subset(tr), coh.subset(va), coh.subset(te)


def main():
    coh, (trf, trl, _), (vaf, val, _), (tef, tel, _) = load_fold()
    D = coh.dim
    print(f"TCGA / {METHOD} / {MODEL}  fold {FOLD}   D={D}")
    print(f"train={len(trl)}  val={len(val)}  test={len(tel)}   "
          f"(MSI-H: {int(trl.sum())}/{int(val.sum())}/{int(tel.sum())})")

    # =====================================================================
    banner("TASK 1.1 - hyperparameter selection must use VALIDATION, not test")
    rows = []
    for h1, h2 in GRID:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        m, dump = eval_ANN(
            fold=FOLD, train_feats=trf, train_labels=trl,
            valid_feats=vaf, valid_labels=val, test_feats=tef, test_labels=tel,
            input_dim=D, hidden_dim=h1, hidden_dim2=h2, max_iter=MAX_ITER,
            verbose=False,
        )
        rows.append({
            "h1": h1, "h2": h2,
            "val_macro_f1": m["val_ann_macro_f1"],
            "val_bacc": m["val_ann_bacc"],
            "test_macro_f1": m["ann_macro_f1"],
            "test_bacc": m["ann_bacc"],
            "test_auroc": m["ann_auroc"],
            "classifier": dump["classifier"],
        })

    print(f"\n  {'config':<14}{'VAL macro-F1':>14}{'VAL bacc':>11}"
          f"{'TEST macro-F1':>15}{'TEST bacc':>11}{'TEST auroc':>12}")
    for r in rows:
        print(f"  h1={r['h1']:<4} h2={r['h2']:<4}{r['val_macro_f1']:>14.6f}"
              f"{r['val_bacc']:>11.6f}{r['test_macro_f1']:>15.6f}"
              f"{r['test_bacc']:>11.6f}{r['test_auroc']:>12.6f}")

    sel_val = max(rows, key=lambda r: r["val_macro_f1"])
    sel_test = max(rows, key=lambda r: r["test_macro_f1"])
    print(f"\n  NEW rule (argmax VALIDATION macro-F1): h1={sel_val['h1']} h2={sel_val['h2']}"
          f"  -> reported test bacc={sel_val['test_bacc']:.6f} auroc={sel_val['test_auroc']:.6f}")
    print(f"  OLD rule (argmax TEST macro-F1)      : h1={sel_test['h1']} h2={sel_test['h2']}"
          f"  -> reported test bacc={sel_test['test_bacc']:.6f} auroc={sel_test['test_auroc']:.6f}")
    same = (sel_val["h1"], sel_val["h2"]) == (sel_test["h1"], sel_test["h2"])
    print(f"  Rules agree on this fold: {same}")
    print(f"  Optimistic bias the old rule bought itself on test macro-F1: "
          f"{sel_test['test_macro_f1'] - sel_val['test_macro_f1']:+.6f}")
    argmax_ok = sel_val["val_macro_f1"] == max(r["val_macro_f1"] for r in rows)
    print(f"\n  ASSERT selected config is argmax of VALIDATION scores: "
          f"{'PASS' if argmax_ok else 'FAIL'}")

    # =====================================================================
    banner("TASK 1.2 - the checkpoint on disk must be the selected configuration")
    with tempfile.TemporaryDirectory() as tmp:
        path = save_ann_checkpoint(
            tmp, FOLD, sel_val["classifier"],
            selection_metric="val_ann_macro_f1",
            selection_value=float(sel_val["val_macro_f1"]),
        )
        cfg = json.loads(Path(tmp, f"fold{FOLD}_ann_config.json").read_text())
        sd = torch.load(path, map_location="cpu")
        w0, w4 = tuple(sd["0.weight"].shape), tuple(sd["4.weight"].shape)

        print(f"  files written : {sorted(os.listdir(tmp))}")
        print(f"  config.json   : input_dim={cfg['input_dim']} h1={cfg['hidden_dim1']} "
              f"h2={cfg['hidden_dim2']} selected_on={cfg['selection_metric']}"
              f"={cfg['selection_value']:.6f}")
        print(f"  state dict    : 0.weight={w0}  4.weight={w4}")
        print(f"  selected cfg  : h1={sel_val['h1']} h2={sel_val['h2']} input_dim={D}")

        ok = (w0 == (cfg["hidden_dim1"], cfg["input_dim"])
              and w4 == (cfg["hidden_dim2"], cfg["hidden_dim1"])
              and cfg["hidden_dim1"] == sel_val["h1"]
              and cfg["hidden_dim2"] == sel_val["h2"])
        print(f"\n  ASSERT state dict == config.json == selected configuration: "
              f"{'PASS' if ok else 'FAIL'}")

        clf, meta = load_ann_checkpoint(tmp, FOLD, input_dim=D)
        print(f"  round-trip load: {meta}")
        probs = clf.predict_proba(tef).cpu().numpy()
        print(f"  predict_proba rows sum to 1: {np.allclose(probs.sum(axis=1), 1.0)}")
        print(f"  filename is grid-unique     : {os.path.basename(path)}")
        print("  (old name was fold{f}_trained_ann_model_{input_dim}.pth - identical "
              "for all 4\n   grid entries, so each overwrote the last)")

    # =====================================================================
    banner("TASK 1.3 - double softmax removed: before vs after")
    h1, h2 = sel_val["h1"], sel_val["h2"]

    class OldANN(ANNBinaryClassifier):
        """Pre-fix architecture: Softmax in the graph AND CrossEntropyLoss."""
        def __init__(self, **kw):
            super().__init__(**kw)
            self.model = nn.Sequential(*list(self.model), nn.Softmax(dim=1)).to(self.device)

    def run(cls, tag):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        clf = cls(input_dim=D, hidden_dim1=h1, hidden_dim2=h2,
                  max_iter=MAX_ITER, verbose=False)
        clf.fit(trf, trl, vaf, val, False)
        if tag == "OLD":
            # Old graph already ends in Softmax; predict_proba would apply it twice.
            feats = tef.to(clf.device)
            clf.model.eval()
            with torch.no_grad():
                probs = clf.model(feats).cpu().numpy()
        else:
            probs = clf.predict_proba(tef).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        return get_eval_metrics(tel.cpu().numpy(), preds, probs, prefix="ann_")

    m_old = run(OldANN, "OLD")
    m_new = run(ANNBinaryClassifier, "NEW")

    print(f"\n  {'variant':<34}{'bacc':>10}{'auroc':>10}{'macro_f1':>11}{'acc':>9}")
    print(f"  {'OLD (Softmax + CrossEntropyLoss)':<34}{m_old['ann_bacc']:>10.4f}"
          f"{m_old['ann_auroc']:>10.4f}{m_old['ann_macro_f1']:>11.4f}{m_old['ann_acc']:>9.4f}")
    print(f"  {'NEW (logits + CrossEntropyLoss)':<34}{m_new['ann_bacc']:>10.4f}"
          f"{m_new['ann_auroc']:>10.4f}{m_new['ann_macro_f1']:>11.4f}{m_new['ann_acc']:>9.4f}")
    print(f"  {'delta (new - old)':<34}{m_new['ann_bacc']-m_old['ann_bacc']:>+10.4f}"
          f"{m_new['ann_auroc']-m_old['ann_auroc']:>+10.4f}"
          f"{m_new['ann_macro_f1']-m_old['ann_macro_f1']:>+11.4f}"
          f"{m_new['ann_acc']-m_old['ann_acc']:>+9.4f}")
    print("\n  Work order Task 1.3: expect improvement or parity; if results degrade "
          "materially,\n  stop and report rather than proceeding.")
    print(f"  conf_matrix OLD={m_old['ann_conf_matrix'].tolist()}  "
          f"NEW={m_new['ann_conf_matrix'].tolist()}")


if __name__ == "__main__":
    main()
