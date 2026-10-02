"""SurGen-EV kNN k-report — DIAGNOSTIC ONLY, built from the existing sweep.

=============================================================================
 DIAGNOSTIC. No k is selected here. Nothing is promoted, and the SurGen-EV
 candidate at experiments/surgen_ev_promotion/ keeps its fixed k=35
 regardless of what this report shows.
=============================================================================

Reruns nothing. ``part2_k_sweep.py`` (3 Sep) already computed kNN at
k in {15, 20, 25, 35, 50, 75, 100} under all three threshold schemes for every
SurGen-EV combination; this script reads those CSVs and reshapes them into the
report requested: kNN broken out per k, the other four heads left at their
existing configuration and reported under the ONE scheme each will actually
use if promoted - frozen tau_TCGA for LR/ANN/ProtoNet, rate-matched quantile
for RF - rather than both schemes side by side as the exploratory diagnostic
did. That matches the policy already fixed for PAIP-EV and the SurGen-EV
promotion candidate; showing both here would just be noise now that the
per-head scheme is no longer an open question.

Run:  python experiments/surgen_ev_diagnostic/report_knn_k_surgen.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

METHODS_AGG = ["Averaging", "Caption_based_aggregation",
               "Caption_based_aggregation_15_classes", "Tissue_Type_Clustering"]
SCHEME_NAME = {"a_frozen": "frozen_tau_TCGA", "b_refit_k": "refit_tau_at_k",
              "c_quantile": "rate_matched_quantile"}
OTHER_SCHEME = {"lin": "a_frozen", "ann": "a_frozen", "proto": "a_frozen",
                "rf": "c_quantile"}


def main() -> None:
    knn = pd.read_csv(RESULTS / "part2_DIAGNOSTIC_knn_k_sweep.csv")
    other = pd.read_csv(RESULTS / "part2_DIAGNOSTIC_other_heads.csv")

    k_s = knn[knn.cohort == "surgen"].copy()
    k_s["scheme"] = k_s["scheme"].map(SCHEME_NAME)
    k_s["is_agg"] = k_s.method.isin(METHODS_AGG)

    # -------- per-(combo x k x scheme), full detail --------------------------
    detail_cols = ["DIAGNOSTIC", "method", "model", "k", "scheme", "metric",
                   "tau", "auroc", "bacc", "bacc_oracle", "n_pos_pred",
                   "n_distinct_scores", "status", "quantile_r",
                   "quantile_degenerate"]
    detail = k_s[detail_cols].sort_values(["method", "model", "k", "scheme"])
    detail.to_csv(RESULTS / "surgen_knn_k_report_detail.csv", index=False)

    # -------- per-k summary, 20 aggregation combinations only ----------------
    agg = k_s[k_s.is_agg]
    summ = agg.groupby(["k", "scheme"]).agg(
        auroc_mean=("auroc", "mean"), auroc_min=("auroc", "min"),
        auroc_max=("auroc", "max"),
        bacc_mean=("bacc", "mean"),
        oracle_mean=("bacc_oracle", "mean"),
        n_dead=("status", lambda s: s.str.contains("dead").sum()),
        n_configs=("method", "size"),
    ).round(4).reset_index()
    summ.insert(0, "DIAGNOSTIC", "not-a-reportable-config")
    summ.to_csv(RESULTS / "surgen_knn_k_report_summary.csv", index=False)

    # -------- other four heads, ONE scheme per head (the policy already set) -
    o_s = other[(other.cohort == "surgen")].copy()
    keep = pd.concat([o_s[(o_s.clf == c) & (o_s.scheme == s)]
                      for c, s in OTHER_SCHEME.items()])
    keep = keep[keep.method.isin(METHODS_AGG)]
    keep_cols = ["DIAGNOSTIC", "method", "model", "clf", "scheme", "tau",
                "auroc", "bacc", "bacc_oracle", "status"]
    keep[keep_cols].sort_values(["clf", "method", "model"]).to_csv(
        RESULTS / "surgen_other_heads_at_policy_scheme.csv", index=False)
    other_summ = keep.groupby("clf").agg(
        auroc_mean=("auroc", "mean"), bacc_mean=("bacc", "mean"),
        oracle_mean=("bacc_oracle", "mean"),
        n_dead=("status", lambda s: s.str.contains("dead").sum()),
    ).round(4)

    # -------------------------------------------------------------- console --
    print("=" * 78)
    print("SurGen-EV kNN, k = 15..100 - DIAGNOSTIC, no k selected")
    print("Other four heads at the policy scheme only: frozen for LR/ANN/Proto, "
          "quantile for RF")
    print("=" * 78)
    print(f"\n20 aggregation combinations, mean over combos, by (k, scheme):\n")
    print(summ.drop(columns="DIAGNOSTIC").to_string(index=False))

    print(f"\nOther heads at their policy scheme (mean over 20 agg combos):\n")
    print(other_summ.to_string())

    print(f"\nwrote {(RESULTS/'surgen_knn_k_report_detail.csv').name}          "
          f"({len(detail)} rows - full combo x k x scheme detail)")
    print(f"wrote {(RESULTS/'surgen_knn_k_report_summary.csv').name}         "
          f"({len(summ)} rows - per k x scheme, mean over 20 combos)")
    print(f"wrote {(RESULTS/'surgen_other_heads_at_policy_scheme.csv').name}  "
          f"({len(keep)} rows)")
    print(f"\nNo k has been selected. The SurGen-EV promotion candidate at "
          f"experiments/surgen_ev_promotion/ still uses k=35, unchanged by "
          f"this report.")


if __name__ == "__main__":
    main()
