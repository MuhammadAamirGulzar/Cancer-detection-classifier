"""Run experiments A-D. Read-only against the pipeline; writes only under
``experiments/threshold_and_k/results/``.

    python experiments/threshold_and_k/run_all.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp_a_threshold_health as A          # noqa: E402
import exp_b_refit_tau as B                 # noqa: E402
import exp_c_quantile as C                  # noqa: E402
import exp_d_weighting as D                 # noqa: E402
from common import RESULTS                  # noqa: E402


def main():
    for name, fn in [("A", A.main), ("B", B.main), ("C", C.main)]:
        print("\n" + "=" * 78)
        fn()
    print("\n" + "=" * 78)
    D.part1_weighting()
    D.part2_selection()
    print("\n" + "=" * 78)
    print("outputs:")
    for p in sorted(RESULTS.iterdir()):
        print(f"  {p.name:32s} {p.stat().st_size/1024:8.1f} KB")


if __name__ == "__main__":
    main()
