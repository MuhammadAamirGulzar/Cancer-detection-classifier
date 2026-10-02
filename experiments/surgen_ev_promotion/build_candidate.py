"""SurGen-EV PROMOTED candidate — built under experiments/, published tree untouched.

Mirrors PAIP-EV's 2 Sep promotion exactly: KNN re-scored at k=35 with a tau
refitted on TCGA out-of-fold probabilities at that k (read from
``knn_k35_thresholds.json``, not recomputed), RF on a rate-matched quantile
threshold, and LR/ANN/ProtoNet left on the frozen tau_TCGA, untouched.

Why this lives in experiments/, not runners.ev_runner
-------------------------------------------------------
The owner asked for SurGen-EV to be "experimented using the experiments
folder" rather than applied straight to the published tree the way PAIP-EV
was. So this script never calls ``ev_runner.run_external_validation`` (which
writes into ``P.results_root(...)``, i.e. ``slide_classification/
SurGen_EV_Results/``). It reads that tree read-only and writes a candidate
tree under ``experiments/surgen_ev_promotion/candidate/`` in the same layout,
so it can be reviewed - and, later, applied - without ever putting the
published tree at risk.

Why re-use ev_runner._add_corrected / _promote instead of reimplementing them
-------------------------------------------------------------------------------
Those two functions are the exact code PAIP-EV's promotion ran through and
that was verified there (bit-identical headline for the non-promoted heads,
KNN/RF deltas matching the validated reference). Calling them here means the
SurGen-EV candidate is produced by literally the same logic, not a second
implementation that could drift from it. They operate on a plain ``entry``
dict and mutate it in place; this script deep-copies each published entry
before handing it in, so nothing written here can reach back into what was
read.

What is NOT recomputed
-----------------------
``mean_probs`` is read from the published record - it is already the
seed-42..46 average that a live TCGA-FULL apply would produce, and re-deriving
it would mean re-loading five seeds of saved models for no numerical
difference. ``_add_corrected``'s KNN branch does load its own artifact
(the k=35 rescoring needs the saved reference matrix), exactly as it does
in ev_runner.

fold_ensemble / fold_average are copied across unmodified - promotion applies
to tcga_full only, on both cohorts.

Run:  python experiments/surgen_ev_promotion/build_candidate.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SLIDE_CLS = REPO / "slide_classification"
SRC = SLIDE_CLS / "SurGen_EV_Results"
DST = HERE / "candidate"
RESULTS = HERE / "results"

sys.path.insert(0, str(SLIDE_CLS))
import data_layer as dl                                # noqa: E402
from runners import ev_runner as ev                     # noqa: E402
from runners import runlog                              # noqa: E402
from runners import threshold_modes as tm               # noqa: E402
from runners.thresholds import load_tau                 # noqa: E402

PROMOTE = ("knn", "rf")
CLFS = ("lin", "ann", "knn", "proto", "rf")
VARIANTS = ("tcga_full", "fold_ensemble", "fold_average")


def n_pos_pred(cm) -> int:
    return int(cm[0][1]) + int(cm[1][1])


def n_total(cm) -> int:
    return int(cm[0][0]) + int(cm[0][1]) + int(cm[1][0]) + int(cm[1][1])


def main() -> int:
    if not SRC.is_dir():
        sys.exit(f"no SurGen-EV tree at {SRC}")
    log = runlog.get_logger()
    DST.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    files = sorted(SRC.rglob("result_SurGen-EV_*_ev.json"))
    print(f"source: {SRC.name}  ({len(files)} combinations, opened read-only)")

    policy = ev._promoted_policy(list(PROMOTE), list(CLFS), tm.KNN_K_VALIDATED)
    rows, failures = [], []

    for f in files:
        published = json.loads(f.read_text(encoding="utf-8"))
        stamp, results = published["stamp"], published["results"]
        method, model = stamp["method"], stamp["model"]

        coh = dl.load_cohort("surgen", method, model, task="MSIH", verbose=False)
        targets = coh.labels.numpy()

        out_results = {}
        for clf in CLFS:
            per_variant = results.get(clf, {})
            new_pv = {}
            for var in VARIANTS:
                m = per_variant.get(var)
                if not isinstance(m, dict) or "bacc" not in m:
                    new_pv[var] = m
                    continue
                entry = copy.deepcopy(m)
                if var == "tcga_full" and clf in PROMOTE:
                    tau_frozen = load_tau(method, model, clf, "MSIH")
                    ev._add_corrected(entry, clf, method, model, "MSIH", coh,
                                      targets, tau_frozen, tm.KNN_K_VALIDATED, log)
                    ok = ev._promote(entry, clf, log)
                    if not ok:
                        failures.append((method, model, clf))
                        continue
                new_pv[var] = entry
            out_results[clf] = new_pv

            tf = new_pv.get("tcga_full")
            if tf:
                rows.append(dict(
                    method=method, model=model, clf=clf,
                    auroc=round(tf["auroc"], 4), bacc=round(tf["bacc"], 4),
                    threshold=round(tf["threshold"], 6),
                    threshold_scheme=tf.get("threshold_scheme", "frozen_tau_TCGA"),
                    n_pos_pred=n_pos_pred(tf["conf_matrix"]),
                    status=tm.health(n_pos_pred(tf["conf_matrix"]), n_total(tf["conf_matrix"])),
                    promoted=bool(tf.get("promoted")),
                    auroc_prev_frozen=(round(tf["auroc_frozen"], 4)
                                       if tf.get("promoted") else None),
                    bacc_prev_frozen=(round(tf["bacc_frozen"], 4)
                                      if tf.get("promoted") else None),
                    threshold_prev_frozen=(round(tf["threshold_frozen"], 6)
                                           if tf.get("promoted") else None),
                ))

        new_stamp = dict(stamp)
        new_stamp["threshold_mode"] = "promoted"
        new_stamp["promoted_classifiers"] = list(PROMOTE)
        new_stamp["knn_k_corrected"] = tm.KNN_K_VALIDATED
        new_stamp["threshold_policy"] = policy
        new_stamp["derived_from"] = f"{SRC.name}/{f.relative_to(SRC).as_posix()}"
        new_stamp["derivation"] = (
            "CANDIDATE built under experiments/surgen_ev_promotion/ via "
            "ev_runner._add_corrected/_promote on a copy of the published "
            "frozen record. Published SurGen_EV_Results/ is untouched.")

        out = DST / f.relative_to(SRC)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"stamp": new_stamp, "results": out_results},
                                  indent=2, default=str), encoding="utf-8")
        print(f"  {method}/{model}")

    df = pd.DataFrame(rows).sort_values(["clf", "method", "model"])
    df.to_csv(RESULTS / "surgen_ev_promoted_candidate.csv", index=False)

    print(f"\nwrote {DST.relative_to(REPO)}/  ({len(files)} files)")
    print(f"wrote {(RESULTS / 'surgen_ev_promoted_candidate.csv').relative_to(REPO)}"
          f"  ({len(df)} rows)")
    if failures:
        print(f"\nFAILED to promote {len(failures)} head(s): {failures}")
    print(f"\npublished tree at {SRC.relative_to(REPO)}/ was opened read-only; "
          f"nothing was written to it.")

    kr = df[df.clf.isin(PROMOTE)]
    print("\nSanity check (mean over combinations where promotion succeeded):")
    for clf in PROMOTE:
        s = kr[kr.clf == clf]
        print(f"  {clf:<4} n={len(s):<3} AUROC {s.auroc.mean():.4f}  "
              f"BalAcc {s.bacc.mean():.4f}  dead/near_dead="
              f"{s.status.str.contains('dead').sum()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
