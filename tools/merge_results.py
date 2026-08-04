"""Merge result trees produced on different machines (work order 1c.2 item 5).

Aggregated features are split across a local Windows workstation and a remote
Linux server, so a full sweep is produced in two halves. Every runner writes one
provenance-stamped JSON per ``(experiment, method, model, variant)``, which makes
merging a file copy plus this script.

What it does:
  1. Scans one or more result trees for ``result_*.json`` files.
  2. Keys them by ``(experiment, method, model, variant)``.
  3. On a duplicate key, keeps the record with the **newer git commit** (falling
     back to timestamp when commits are unordered or unknown) and logs the
     collision rather than silently overwriting.
  4. Writes the consolidated workbook and a coverage matrix showing what ran
     where, what is missing, and what collided.

Run:
    python tools/merge_results.py TREE [TREE ...] [--out FILE.xlsx]

Typical use, once the server's results have been copied next to the local ones:
    python tools/merge_results.py slide_classification server_results \\
        --out slide_classification/best_of_all_exps_metric.xlsx
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "slide_classification"))

CANONICAL_EXPERIMENTS = ["TCGA-CV", "PAIP-IV", "PAIP-EV", "SurGen-CV", "SurGen-EV"]


def _commit_time(commit: str) -> Optional[int]:
    """Committer timestamp, so 'newer commit' is well defined. None if unknown."""
    if not commit or commit == "unknown":
        return None
    try:
        out = subprocess.run(["git", "show", "-s", "--format=%ct", commit],
                             cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=15)
        return int(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip() else None
    except Exception:
        return None


def discover(trees: List[Path]) -> List[dict]:
    """Every result JSON under the given trees, with its provenance parsed."""
    found = []
    for tree in trees:
        if not tree.exists():
            print(f"[WARN] tree does not exist, skipping: {tree}")
            continue
        for path in sorted(tree.rglob("result_*.json")):
            if "_ARCHIVED_" in str(path):
                continue          # archived pre-correction results are not merged
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                print(f"[WARN] unreadable {path}: {exc}")
                continue
            stamp = payload.get("stamp", {})
            found.append({
                "path": path,
                "payload": payload,
                "experiment": stamp.get("experiment", "?"),
                "method": stamp.get("method", "?"),
                "model": stamp.get("model", "?"),
                "variant": stamp.get("variant", "default"),
                "machine": stamp.get("machine", "?"),
                "timestamp": stamp.get("timestamp", ""),
                "git_commit": stamp.get("git_commit", "unknown"),
                "n_test": stamp.get("n_test"),
            })
    return found


def resolve(records: List[dict]) -> Tuple[Dict[Tuple, dict], List[dict]]:
    """Deduplicate by key, keeping the newer commit. Returns (kept, collisions)."""
    by_key: Dict[Tuple, List[dict]] = defaultdict(list)
    for r in records:
        by_key[(r["experiment"], r["method"], r["model"], r["variant"])].append(r)

    kept, collisions = {}, []
    for key, group in by_key.items():
        if len(group) == 1:
            kept[key] = group[0]
            continue
        ranked = sorted(
            group,
            key=lambda r: (_commit_time(r["git_commit"]) or -1, r["timestamp"]),
            reverse=True,
        )
        winner, losers = ranked[0], ranked[1:]
        kept[key] = winner
        collisions.append({
            "key": "|".join(key),
            "kept_machine": winner["machine"],
            "kept_commit": winner["git_commit"],
            "kept_timestamp": winner["timestamp"],
            "kept_path": str(winner["path"]),
            "discarded": "; ".join(
                f"{l['machine']}@{l['git_commit']}({l['timestamp']})" for l in losers),
            "n_duplicates": len(group),
        })
        print(f"[COLLISION] {'|'.join(key)}: kept {winner['machine']}@"
              f"{winner['git_commit']}, discarded {len(losers)} other(s)")
    return kept, collisions


def summary_rows(kept: Dict[Tuple, dict]) -> Dict[str, pd.DataFrame]:
    """One DataFrame per experiment, from each record's own summary block."""
    per_exp: Dict[str, List[dict]] = defaultdict(list)
    for (exp, method, model, variant), rec in sorted(kept.items()):
        payload = rec["payload"]
        base = {"Method": method, "Model": model, "Variant": variant,
                "Machine": rec["machine"], "GitCommit": rec["git_commit"],
                "N_test": rec["n_test"]}

        if "summary" in payload:                       # CV-style record
            for row in payload["summary"]:
                per_exp[exp].append({**base, **row})
        elif "results" in payload:                     # EV / IV-style record
            for clf, val in payload["results"].items():
                if isinstance(val, dict) and any(
                        isinstance(v, dict) for v in val.values()):
                    for vname, m in val.items():       # EV: nested by variant
                        if isinstance(m, dict) and "bacc" in m:
                            per_exp[exp].append({**base, "Variant": vname,
                                                 "Classifier": clf, **_flat(m)})
                elif isinstance(val, dict):            # IV: flat per classifier
                    per_exp[exp].append({**base, "Classifier": clf, **_flat(val)})

    out = {}
    for exp, rows in per_exp.items():
        df = pd.DataFrame(rows)
        # Work order Task 4.2 sort: BalAcc primary, AUROC as tie-break. Sort by
        # AUROC first, then stable-sort by BalAcc.
        for bacc, auroc in (("bacc", "auroc"), ("BalAcc", "AUROC")):
            if bacc in df.columns and auroc in df.columns:
                df = df.sort_values(auroc, ascending=False, kind="mergesort")
                df = df.sort_values(bacc, ascending=False, kind="mergesort")
                break
        out[exp] = df.reset_index(drop=True)
    return out


def _flat(m: dict) -> dict:
    keep = ("bacc", "auroc", "acc", "macro_f1", "weighted_f1", "threshold",
            "bacc_sd", "auroc_sd", "n_seeds", "n_folds_used")
    row = {k: m[k] for k in keep if k in m}
    if "conf_matrix" in m:
        row["conf_matrix"] = str(m["conf_matrix"])
    if "bootstrap" in m and isinstance(m["bootstrap"], dict):
        b = m["bootstrap"]
        row["BalAcc_CI"] = str(b.get("bacc_ci"))
        row["AUROC_CI"] = str(b.get("auroc_ci"))
    return row


def coverage(kept: Dict[Tuple, dict]) -> pd.DataFrame:
    rows = [{"Experiment": e, "Method": me, "Model": mo, "Variant": v,
             "Machine": r["machine"], "GitCommit": r["git_commit"],
             "Timestamp": r["timestamp"], "N_test": r["n_test"],
             "Path": str(r["path"])}
            for (e, me, mo, v), r in sorted(kept.items())]
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Merge result trees across machines")
    ap.add_argument("trees", nargs="+", type=Path)
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "slide_classification" / "best_of_all_exps_metric.xlsx")
    args = ap.parse_args()

    records = discover(args.trees)
    print(f"discovered {len(records)} result file(s) across {len(args.trees)} tree(s)")
    if not records:
        print("nothing to merge")
        return

    kept, collisions = resolve(records)
    print(f"kept {len(kept)} unique (experiment, method, model, variant) key(s); "
          f"{len(collisions)} collision(s)")

    sheets = summary_rows(kept)
    cov = coverage(kept)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.out, engine="openpyxl", mode="w") as xw:
        for exp in CANONICAL_EXPERIMENTS:
            if exp in sheets:
                sheets[exp].to_excel(xw, sheet_name=exp[:31], index=False)
        for exp, df in sheets.items():
            if exp not in CANONICAL_EXPERIMENTS:
                df.to_excel(xw, sheet_name=str(exp)[:31], index=False)
        cov.to_excel(xw, sheet_name="Coverage", index=False)
        if collisions:
            pd.DataFrame(collisions).to_excel(xw, sheet_name="Collisions", index=False)

    print(f"wrote {args.out}")
    print("\ncoverage by experiment:")
    for exp, df in sorted(sheets.items()):
        machines = ", ".join(sorted(set(cov[cov.Experiment == exp]["Machine"])))
        print(f"  {exp:<12} {len(df):>4} rows   machines: {machines}")


if __name__ == "__main__":
    main()
