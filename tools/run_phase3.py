"""Orchestrate the rest of Phase 3 as one resumable unattended run.

Order matters and is not arbitrary:

  1. hparams      TCGA-FULL hyperparameters, derived from TCGA-CV alone. Must
                  come after TCGA-CV and before any TCGA-FULL training.
  2. tcga_full    train the TCGA-FULL artifact on all 413 slides, seeds 42-46.
                  Gates both EV experiments (work order Task 3.0).
  3. surgen_cv    SurGen-CV with the corrected case-level folds (Task 2.1).
  4. paip_iv      PAIP-IV on the provider's 42/31 split + bootstrap CIs (3.1).
  5. paip_ev      PAIP-EV, three variants at tau_TCGA (3.0/3.3).
  6. surgen_ev    SurGen-EV, three variants at tau_TCGA (3.2). Partial coverage
                  on this machine by design - only Conch1_5 and Virchow2 have
                  SurGen features locally.

Each stage records completion in progress.json, so an interrupted run resumes
where it left off instead of redoing finished work. Stages are independent
processes in sequence; a stage that fails is reported and the run continues to
the next, because a failure in (say) SurGen-EV should not cost PAIP-EV.

Run:  python tools/run_phase3.py [--stages a,b,c] [--force]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SLIDE_CLS = REPO_ROOT / "slide_classification"
PYTHON = sys.executable

STAGES = {
    "hparams":   ["-m", "runners.hparams"],
    "tcga_full": ["-m", "runners.full_trainer"],
    "surgen_cv": ["-m", "runners.cv_runner", "--experiment", "SurGen-CV"],
    "paip_iv":   ["-m", "runners.iv_runner"],
    "paip_ev":   ["-m", "runners.ev_runner", "--experiment", "PAIP-EV"],
    "surgen_ev": ["-m", "runners.ev_runner", "--experiment", "SurGen-EV"],
}
DEFAULT_ORDER = ["hparams", "tcga_full", "surgen_cv", "paip_iv", "paip_ev", "surgen_ev"]


def run_stage(name: str, force: bool) -> tuple[bool, float]:
    cmd = [PYTHON, "-u"] + STAGES[name]
    if force and name != "hparams":
        cmd.append("--force")
    print(f"\n{'=' * 78}\nSTAGE {name}\n  {' '.join(cmd[1:])}\n{'=' * 78}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(SLIDE_CLS))
    dt = time.perf_counter() - t0
    ok = proc.returncode == 0
    print(f"\nSTAGE {name}: {'OK' if ok else f'FAILED (rc={proc.returncode})'} "
          f"in {dt / 60:.1f} min", flush=True)
    return ok, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=",".join(DEFAULT_ORDER))
    ap.add_argument("--force", action="store_true",
                    help="re-run combinations already marked done in progress.json")
    args = ap.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        sys.exit(f"unknown stage(s): {unknown}. Known: {list(STAGES)}")

    results = {}
    t_all = time.perf_counter()
    for name in stages:
        ok, dt = run_stage(name, args.force)
        results[name] = (ok, dt)

    print(f"\n{'=' * 78}\nPHASE 3 SUMMARY  (total {(time.perf_counter() - t_all) / 60:.1f} min)"
          f"\n{'=' * 78}")
    for name, (ok, dt) in results.items():
        print(f"  {name:<12} {'OK    ' if ok else 'FAILED'}  {dt / 60:>6.1f} min")
    failed = [n for n, (ok, _) in results.items() if not ok]
    if failed:
        print(f"\n{len(failed)} stage(s) failed: {failed}")
        print("Check logs/run_<timestamp>.log for tracebacks and "
              "slide_classification/skipped_combinations.csv for what was skipped.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
