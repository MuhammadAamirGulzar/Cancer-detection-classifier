"""Freeze the final ANN artefact: ann_old weights, configs and a manifest.

The owner's decision is that **ann_old is the final ANN configuration** for the
PAIP-IV study. This copies its 22 trained networks out of the results tree into a
stable location that no sweep will archive or overwrite, and writes a manifest
recording exactly what each one is.

Why this is needed: every ``iv_runner.py --force`` run renames the live results
tree to ``PAIP_IV_Results_ARCHIVED_<date>``, and the cleanup performed earlier in
the study deleted 16 such trees. A checkpoint that lives only inside a results
tree is one sweep away from being archived under an unpredictable name, and a few
sweeps away from being deleted. The final artefact should not be reachable only
by remembering which archive it landed in.

Source is verified before anything is copied - the tree must carry the ann_old
signature (single hidden layer, dropout 0.5, 512 units, 1000 iterations) or the
script refuses to run.

Run:  python tools/save_final_ann.py
      python tools/save_final_ann.py --source PAIP_IV_Results_ARCHIVED_20260812_3
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SLIDE_CLS = REPO_ROOT / "slide_classification"
DEFAULT_SOURCE = "PAIP_IV_Results_ARCHIVED_20260812_3"
OUT = SLIDE_CLS / "final_models" / "ann_old"

#: The configuration ann_old must have. Checked per combination before copying -
#: a tree that does not match is not ann_old, whatever its directory is called.
EXPECTED = {"n_hidden_layers": 1, "hidden_dim1": 512, "hidden_dim2": None,
            "dropout": 0.5, "patience": 10, "max_iter": 1000,
            "lr": 0.0001, "weight_decay": 0.0001, "softmax_in_graph": False}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(tree: Path) -> List[dict]:
    """Every (method, model) in the tree, with its config and result row."""
    entries = []
    for cfg_path in sorted(tree.rglob("fold0_ann_config.json")):
        combo = cfg_path.parents[1]                     # .../<Method>/<Model>/1-MSIH
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        mismatch = {k: (cfg.get(k), v) for k, v in EXPECTED.items()
                    if cfg.get(k) != v}
        if mismatch:
            raise SystemExit(
                f"ABORT - {cfg_path.relative_to(tree)} is not ann_old.\n"
                f"  mismatched: {mismatch}\n"
                f"  This tree does not hold the final configuration."
            )

        summary = combo / "Output" / "summary_iv.csv"
        row = next((r for r in csv.DictReader(open(summary, newline="",
                                                   encoding="utf-8"))
                    if r["Classifier"] == "ann"), None)
        if row is None:
            raise SystemExit(f"ABORT - no ann row in {summary}")

        parts = combo.relative_to(tree).parts          # (Method, Model, 1-MSIH)
        entries.append({
            "method": row["Method"], "model": row["Model"],
            "dir": combo, "config": cfg, "row": row,
            "weights": combo / "models" / cfg["checkpoint"],
        })
    if not entries:
        raise SystemExit(f"ABORT - no ANN checkpoints under {tree}")
    return entries


def freeze(tree: Path, out: Path) -> Path:
    entries = verify(tree)
    print(f"verified {len(entries)} checkpoints all carry the ann_old "
          f"configuration\n")

    if out.exists():
        shutil.rmtree(out)
    (out / "weights").mkdir(parents=True)

    manifest: Dict[str, object] = {
        "artefact": "PAIP-IV final ANN",
        "configuration": "ann_old",
        "decided": "owner decision - ann_old is the final ANN configuration",
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_tree": tree.name,
        "hyperparameters": dict(EXPECTED),
        "provenance": (
            "Architecture, dropout, width, iteration count and patience are taken "
            "verbatim from the collaborator's ann_old.py, as that script behaves "
            "when driven through its own eval_ANN with nothing overridden. TCGA-CV "
            "was not consulted. No hyperparameter was tuned on PAIP."
        ),
        "not_reproduced": (
            "ann_old.py places nn.Softmax inside the training graph while using "
            "CrossEntropyLoss, which applies log-softmax again. That is a defect, "
            "not a hyperparameter; these networks emit raw logits and softmax is "
            "applied only in predict_proba."
        ),
        "training": {
            "cohort": "PAIP", "split": "provider 47 train / 31 test",
            "trained_on": 47, "held_back_for_tuning": 0,
            "protocol": "full - no search, no carve-out",
            "threshold": 0.5, "seed": 42,
        },
        "models": [],
    }

    ba, au = [], []
    for e in entries:
        stem = f"{e['method']}__{e['model']}"
        dst_w = out / "weights" / f"{stem}.pth"
        dst_c = out / "weights" / f"{stem}.json"
        shutil.copy2(e["weights"], dst_w)
        dst_c.write_text(json.dumps(e["config"], indent=2), encoding="utf-8")

        ba.append(float(e["row"]["BalAcc"]))
        au.append(float(e["row"]["AUROC"]))
        manifest["models"].append({
            "method": e["method"], "model": e["model"],
            "weights": f"weights/{stem}.pth",
            "config": f"weights/{stem}.json",
            "input_dim": e["config"]["input_dim"],
            "hidden_dim1": e["config"]["hidden_dim1"],
            "sha256": sha256(dst_w),
            "BalAcc": float(e["row"]["BalAcc"]),
            "AUROC": float(e["row"]["AUROC"]),
            "ConfMatrix": e["row"]["ConfMatrix"],
        })
        print(f"  {e['method']}/{e['model']:<12} -> weights/{stem}.pth")

    manifest["results"] = {
        "n_combinations": len(entries),
        "mean_BalAcc": round(st.mean(ba), 4),
        "median_BalAcc": round(st.median(ba), 4),
        "mean_AUROC": round(st.mean(au), 4),
        "best_BalAcc": round(max(ba), 4),
    }

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                       encoding="utf-8")
    print(f"\n  manifest -> {(out / 'manifest.json').relative_to(REPO_ROOT)}")
    print(f"  {len(entries)} networks, mean BalAcc {manifest['results']['mean_BalAcc']}, "
          f"mean AUROC {manifest['results']['mean_AUROC']}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Freeze the final ann_old ANN artefact")
    ap.add_argument("--source", default=DEFAULT_SOURCE,
                    help="results tree holding the ann_old run")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    tree = SLIDE_CLS / args.source
    if not tree.is_dir():
        raise SystemExit(f"ABORT - no such tree: {tree}")
    p = freeze(tree, args.out)
    print(f"\nfrozen -> {p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
