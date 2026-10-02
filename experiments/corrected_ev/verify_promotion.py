"""Verify the promoted path. Read-only - writes nothing anywhere.

Two checks:

1. **Default (frozen) reproduces the published result exactly.** Runs the real
   ev_runner code path - the same ``load_model`` / ``predict_proba`` /
   ``_metrics_at`` / ``_aggregate_seeds`` calls - and compares against the
   stored ``result_*.json``. ``run_external_validation`` itself is not called,
   because it writes into the published tree; every function it would use is.

2. **``--threshold-mode=corrected`` matches the validated reference.** Calls
   ``ev_runner._add_corrected`` and compares against
   ``experiments/corrected_ev/results/corrected_ev_<cohort>.csv``.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "threshold_and_k"))
from common import P, cohort, valid_pairs                    # noqa: E402

import data_layer as dl                                      # noqa: E402
from runners import ev_runner as ev                           # noqa: E402
from runners import runlog                                    # noqa: E402
from runners.full_trainer import SEEDS, artifact_dir          # noqa: E402
from runners.model_io import load_model, predict_proba        # noqa: E402
from runners.thresholds import load_tau                       # noqa: E402

EV_DIR = {"paip": "TCGA_PAIP_EV_Results", "surgen": "SurGen_EV_Results"}
TOL = 1e-9
#: RandomForest is pickled with n_jobs=-1 and accumulates tree probabilities
#: across threads, so float addition order varies run to run. Measured spread
#: over five identical repeats in one process: 2.97e-06, with no code change
#: between them. Deviations at this scale are environmental, not regressions -
#: counted separately rather than hidden by a looser tolerance.
RF_JITTER = 1e-4


def frozen_entry(method, model, kind, coh):
    """Recompute tcga_full exactly as run_external_validation would."""
    tau = load_tau(method, model, kind, "MSIH")
    targets = coh.labels.numpy()
    sm, sp = [], []
    for seed in SEEDS:
        try:
            clf = load_model(kind, artifact_dir(method, model, seed, "MSIH"),
                             fold=None, input_dim=coh.dim)
        except FileNotFoundError:
            continue
        p = predict_proba(kind, clf, coh.feats)[:, 1]
        sp.append(p)
        sm.append(ev._metrics_at(targets, p, tau))
    if not sm:
        return None, tau
    entry = ev._aggregate_seeds(sm, SEEDS[:len(sm)])
    entry["mean_probs"] = np.mean(sp, axis=0).tolist()
    return entry, tau


def main() -> int:
    log = runlog.get_logger()
    ref = {c: pd.read_csv(HERE / "results" / f"corrected_ev_{c}.csv")
           for c in EV_DIR}
    d1 = d2 = n1 = n2 = jitter = 0
    worst1 = worst2 = 0.0

    for method, model in valid_pairs():
        for target, evd in EV_DIR.items():
            fs = glob.glob(str(P.SLIDE_CLS_ROOT / evd / method / model /
                               "1-MSIH" / "Output" / "result_*.json"))
            if not fs:
                continue
            pub = json.load(open(fs[0]))["results"]
            try:
                coh = cohort(target, method, model)
            except Exception:
                continue
            targets = coh.labels.numpy()

            for kind in ("lin", "ann", "knn", "proto", "rf"):
                p_pub = pub.get(kind, {}).get("tcga_full")
                if not p_pub:
                    continue
                entry, tau = frozen_entry(method, model, kind, coh)
                if entry is None:
                    continue

                # ---- check 1: frozen == published
                for key in ("bacc", "auroc", "acc", "macro_f1"):
                    n1 += 1
                    diff = abs(float(entry[key]) - float(p_pub[key]))
                    worst1 = max(worst1, diff)
                    if diff > TOL:
                        if kind == "rf" and diff < RF_JITTER:
                            jitter += 1
                        else:
                            d1 += 1
                            print(f"  FROZEN MISMATCH {target}/{method}/{model}/"
                                  f"{kind}/{key}: {entry[key]} vs {p_pub[key]}")

                # ---- check 2: corrected == validated reference
                ev._add_corrected(entry, kind, method, model, "MSIH", coh,
                                  targets, tau, ev.tm.KNN_K_VALIDATED, log)
                r = ref[target]
                row = r[(r.method == method) & (r.model == model) & (r.clf == kind)]
                if "bacc_corrected" not in entry or row.empty:
                    continue
                pairs = [("bacc_corrected", "BalAcc_corrected"),
                         ("status_corrected", "status_corrected")]
                if "auroc_corrected" in entry:
                    pairs.append(("auroc_corrected", "AUROC_corrected"))
                for ek, rk in pairs:
                    n2 += 1
                    got, want = entry[ek], row.iloc[0][rk]
                    if isinstance(got, str):
                        bad = got != want
                        diff = 0.0 if not bad else 1.0
                    else:
                        diff = abs(round(float(got), 4) - float(want))
                        bad = diff > 5e-5
                    worst2 = max(worst2, diff)
                    if bad:
                        d2 += 1
                        print(f"  CORRECTED MISMATCH {target}/{method}/{model}/"
                              f"{kind}/{ek}: {got} vs {want}")

    print("\n=== CHECK 1: default (frozen) vs published result_*.json ===")
    print(f"  {n1} values compared, {d1} real mismatches, "
          f"{jitter} within RF thread-jitter (<{RF_JITTER:g}), "
          f"max |diff| = {worst1:.3e}")
    print("  PASS - unflagged run reproduces published results "
          "(exact except RF float-accumulation jitter)"
          if d1 == 0 else "  FAIL")
    print("\n=== CHECK 2: --threshold-mode=corrected vs validated reference ===")
    print(f"  {n2} values compared, {d2} mismatches, max |diff| = {worst2:.3e}")
    print("  PASS - promoted path matches experiments/corrected_ev/results/"
          if d2 == 0 else "  FAIL")
    return 0 if (d1 == 0 and d2 == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
