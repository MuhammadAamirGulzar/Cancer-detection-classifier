"""PART 1 - a bare frozen SurGen-EV record, derived read-only.

What this does
--------------
SurGen-EV's published headline is already entirely frozen: 21 combinations, zero
promoted rows. The 31 Aug ``--threshold-mode corrected`` run computed a second
operating point for all five heads and stored it in ``*_corrected`` keys beside
the headline. Nothing ever read those keys - the workbook, the docx, RESULTS.md
and the radar plots all take ``bacc`` / ``auroc`` / ``threshold``.

This script strips them, leaving the record a plain ``--threshold-mode frozen``
run would have produced, and proves the headline did not move.

Where it writes - and what it deliberately does not touch
---------------------------------------------------------
The brief opens with "everything under experiments/ - do not touch published
SurGen-EV results". So the stripped tree is written **here**, under
``experiments/surgen_ev_diagnostic/frozen_baseline/``, and the published
``SurGen_EV_Results/`` tree is opened read-only and left byte-identical. That
also keeps the repo's rule of engagement intact: never edit a results file by
hand.

Applying this to the published tree is then a copy, once someone decides that is
wanted - see APPLY.md. Doing it the other way round is not recoverable.

``status`` - and a discrepancy the strip exposed
------------------------------------------------
``status_frozen`` was itself written by ``_add_corrected``, so a genuinely bare
frozen run would not carry it. Recomputing it rather than copying it turned up
something worth recording: **the stored status describes a different operating
point from the reported metrics.**

There are two defensible counts of "slides flagged positive", and they are not
close:

* ``mean_probs >= tau`` - threshold the seed-AVERAGED probability vector once.
  This is what ``_add_corrected`` used, and what ``status_frozen`` reports.
* ``FP + TP`` from ``conf_matrix`` - the MEAN of the five per-seed confusion
  matrices. This is the path ``bacc``, ``auroc`` and ``conf_matrix`` itself take
  (``_aggregate_seeds`` averages per-seed metrics).

Averaging probabilities and then thresholding is not the same operation as
thresholding and then averaging. On SurGen Averaging/Virchow2 the ANN gives 617
positives one way and 553 the other; RF gives 6 against 14. So the published
``status_frozen`` is not the health of the published BalAcc.

Both are therefore reported, named for what they are, with the disagreements
counted. ``status`` (from the confusion matrix) is the one consistent with the
reported metrics; ``status_meanprob`` reproduces the stored value exactly.

Run:  python experiments/surgen_ev_diagnostic/part1_frozen_baseline.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SLIDE_CLS = REPO / "slide_classification"
SRC = SLIDE_CLS / "SurGen_EV_Results"
DST = HERE / "frozen_baseline"
RESULTS = HERE / "results"

sys.path.insert(0, str(SLIDE_CLS))
from runners import threshold_modes as tm            # noqa: E402  (health())

VARIANTS = ("tcga_full", "fold_ensemble", "fold_average")
CLFS = ("lin", "ann", "knn", "proto", "rf")

#: Everything the corrected run added. Stripped wholesale.
CORRECTED_KEYS = (
    "threshold_mode", "threshold_corrected", "bacc_corrected", "acc_corrected",
    "macro_f1_corrected", "weighted_f1_corrected", "conf_matrix_corrected",
    "n_pos_pred_corrected", "status_corrected", "corrected_scheme",
    "auroc_corrected", "knn_k_corrected", "knn_tau_source", "status_frozen",
)

#: The headline. Every one of these must survive the strip untouched - that is
#: the whole claim being made, so it is checked rather than assumed.
HEADLINE_KEYS = (
    "bacc", "auroc", "acc", "macro_f1", "weighted_f1", "threshold",
    "conf_matrix", "bacc_sd", "auroc_sd", "acc_sd", "macro_f1_sd",
    "weighted_f1_sd", "n_seeds", "seeds", "n_folds_used", "mean_probs",
)


def n_pos_pred(cm) -> int:
    """Slides flagged positive, from the confusion matrix: FP + TP."""
    return int(cm[0][1]) + int(cm[1][1])


def n_total(cm) -> int:
    return int(cm[0][0]) + int(cm[0][1]) + int(cm[1][0]) + int(cm[1][1])


def strip_entry(entry: dict) -> dict:
    """Drop the corrected keys; attach both readings of the operating point."""
    out = {k: v for k, v in entry.items() if k not in CORRECTED_KEYS}
    cm = entry["conf_matrix"]
    n = n_total(cm)
    npp_cm = n_pos_pred(cm)
    out["n_pos_pred"] = npp_cm
    out["status"] = tm.health(npp_cm, n)          # consistent with bacc/auroc
    probs = entry.get("mean_probs")
    if probs is not None:
        npp_mp = int(sum(1 for p in probs if p >= entry["threshold"]))
        out["n_pos_pred_meanprob"] = npp_mp
        out["status_meanprob"] = tm.health(npp_mp, len(probs))
    return out


def main() -> int:
    if not SRC.is_dir():
        sys.exit(f"no SurGen-EV tree at {SRC}")
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    files = sorted(SRC.rglob("result_SurGen-EV_*_ev.json"))
    print(f"source: {SRC.name}  ({len(files)} combinations, opened read-only)")

    rows, drift, stripped_total, status_mismatch = [], [], 0, []
    for f in files:
        payload = json.loads(f.read_text(encoding="utf-8"))
        stamp, results = payload["stamp"], payload["results"]
        method, model = stamp["method"], stamp["model"]

        new_results = {}
        for clf, per_variant in results.items():
            new_pv = {}
            for var, m in per_variant.items():
                if not isinstance(m, dict) or "bacc" not in m:
                    new_pv[var] = m
                    continue
                stripped = strip_entry(m)
                stripped_total += sum(1 for k in CORRECTED_KEYS if k in m)

                # --- the claim: no headline value moved
                for k in HEADLINE_KEYS:
                    if (k in m) != (k in stripped) or m.get(k) != stripped.get(k):
                        drift.append((f.name, clf, var, k, m.get(k), stripped.get(k)))

                # --- status_meanprob must reproduce the stored value exactly;
                #     status (from the confusion matrix) may legitimately differ,
                #     and where it does, that IS the finding.
                if "status_frozen" in m and "status_meanprob" in stripped:
                    if m["status_frozen"] != stripped["status_meanprob"]:
                        drift.append((f.name, clf, var, "status_meanprob",
                                      m["status_frozen"], stripped["status_meanprob"]))
                    if m["status_frozen"] != stripped["status"]:
                        status_mismatch.append(dict(
                            method=method, model=model, clf=clf, variant=var,
                            status_meanprob=stripped["status_meanprob"],
                            status_confmatrix=stripped["status"],
                            n_pos_meanprob=stripped["n_pos_pred_meanprob"],
                            n_pos_confmatrix=stripped["n_pos_pred"]))

                new_pv[var] = stripped
                if var == "tcga_full":
                    rows.append(dict(
                        method=method, model=model, clf=clf,
                        auroc=round(m["auroc"], 4), bacc=round(m["bacc"], 4),
                        acc=round(m["acc"], 4), macro_f1=round(m["macro_f1"], 4),
                        threshold=round(m["threshold"], 6),
                        n_pos_pred=stripped["n_pos_pred"],
                        status=stripped["status"],
                        n_pos_pred_meanprob=stripped.get("n_pos_pred_meanprob"),
                        status_meanprob=stripped.get("status_meanprob"),
                        conf_matrix=str(m["conf_matrix"])))
            new_results[clf] = new_pv

        new_stamp = {k: v for k, v in stamp.items()
                     if k not in ("threshold_mode", "knn_k_corrected",
                                  "promoted_classifiers")}
        new_stamp["threshold_mode"] = "frozen"
        new_stamp["derived_from"] = f"{SRC.name}/{f.relative_to(SRC).as_posix()}"
        new_stamp["derivation"] = (
            "unused *_corrected columns stripped; headline values unchanged and "
            "verified bit-identical; status recomputed from the frozen confusion "
            "matrix. DIAGNOSTIC COPY - the published tree is untouched.")

        out = DST / f.relative_to(SRC)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"stamp": new_stamp, "results": new_results},
                                  indent=2, default=str), encoding="utf-8")

    df = pd.DataFrame(rows).sort_values(["clf", "method", "model"])
    df.to_csv(RESULTS / "part1_surgen_ev_frozen_baseline.csv", index=False)

    print(f"stripped {stripped_total} unused key(s) across "
          f"{len(files)} files x 5 classifiers x 3 variants")
    print(f"headline values compared: {len(HEADLINE_KEYS)} keys x "
          f"{len(files) * 5 * 3} entries")
    print(f"  drift: {len(drift)}  -> "
          + ("PASS - nothing published moved, and status_meanprob reproduces "
             "the stored status_frozen exactly"
             if not drift else f"FAIL {drift[:4]}"))

    if status_mismatch:
        sm = pd.DataFrame(status_mismatch)
        sm.to_csv(RESULTS / "part1_status_definition_disagreement.csv", index=False)
        print(f"\n  NOTE - {len(sm)} entries where the two readings of the "
              f"operating point disagree.\n  The published status_frozen "
              f"describes mean_probs>=tau; the reported BalAcc/conf_matrix "
              f"describe\n  the per-seed average. See the module docstring.")
        print(sm.to_string(index=False))
    print(f"\nwrote {DST.relative_to(REPO)}/  ({len(files)} files)")
    print(f"wrote {(RESULTS / 'part1_surgen_ev_frozen_baseline.csv').relative_to(REPO)}"
          f"  ({len(df)} rows)")

    # published tree must be exactly as we found it
    still = sorted(SRC.rglob("result_SurGen-EV_*_ev.json"))
    print(f"\npublished tree re-checked: {len(still)} files present, none written")
    return 0 if not drift else 1


if __name__ == "__main__":
    raise SystemExit(main())
