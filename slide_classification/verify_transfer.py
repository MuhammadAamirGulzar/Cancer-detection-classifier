"""Verify transferred SurGen aggregated features against the labelled cohort.

Run from slide_classification/ after moving encoders into surgen_processed.
Exit 0 only if every combination is at 622, or at a KNOWN-short count.
"""
import os
import sys

sys.path.insert(0, ".")
import pandas as pd

from config import paths as P

ENCODERS = ["UNI2", "H-Optimus-1", "ConchV1", "Conch1_5", "Virchow2"]
METHODS = ["Averaging", "Caption_based_aggregation",
           "Caption_based_aggregation_15_classes", "Tissue_Type_Clustering"]

# H-Optimus-1 lost nine SR386 negatives (T573-T583) at the aggregation stage.
# Confirmed on the server, not a transfer fault.
KNOWN_SHORT = {("Averaging", "H-Optimus-1"): 613,
               ("Tissue_Type_Clustering", "H-Optimus-1"): 613}

lab = pd.read_csv("surgen_labels.csv")
need = set(lab[lab.label_desc.isin([0, 1])].WSI_Id)
print(f"labelled cohort: {len(need)}; 2 slides have no features anywhere -> expect 622\n")

bad = 0
print("%-38s %-13s %7s %8s  %s" % ("method", "encoder", "covers", "missing", "status"))
for m in METHODS:
    for e in ENCODERS:
        d = P.feature_dir("surgen", m, e)
        if not d.is_dir():
            print("%-38s %-13s %7s %8s  %s" % (m, e, "-", "-", "NO DIR"))
            bad += 1
            continue
        ids = {os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".pt")}
        cov = len(need & ids)
        if cov == 622:
            status = "OK"
        elif KNOWN_SHORT.get((m, e)) == cov:
            status = "KNOWN-SHORT (expected)"
        else:
            status = "UNEXPECTED"
            bad += 1
        print("%-38s %-13s %7d %8d  %s" % (m, e, cov, len(need - ids), status))

print()
if bad:
    print(f"FAIL: {bad} combination(s) wrong. Do not run EV until resolved.")
    sys.exit(1)
print("PASS: all combinations present (two H-Optimus-1 at the known 613).")
