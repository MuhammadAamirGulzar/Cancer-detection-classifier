"""Unattended-run plumbing: logging, resumable progress, skip tracking.

Work order section 1c.3. The pipeline is expected to run for hours with nobody
watching, so it must (a) leave a full trace of what it did, (b) resume without
redoing finished work, and (c) distinguish a *correctness* failure - which must
halt loudly - from an *availability* failure, which must only skip.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from config import paths as P

REPO_ROOT = P.REPO_ROOT
LOG_DIR = REPO_ROOT / "logs"
PROGRESS_PATH = REPO_ROOT / "progress.json"
SKIPPED_PATH = P.SLIDE_CLS_ROOT / "skipped_combinations.csv"

_logger: Optional[logging.Logger] = None


def get_logger(name: str = "pipeline") -> logging.Logger:
    """Logger writing to both stdout and ``logs/run_<timestamp>.log``."""
    global _logger
    if _logger is not None:
        return _logger

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"run_{stamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info(f"log file: {log_path}")
    logger.info(f"machine={P.MACHINE} base_root={P.BASE_ROOT} commit={P.git_commit()}")
    _logger = logger
    return logger


@contextmanager
def timed(label: str, logger: Optional[logging.Logger] = None):
    """Log wall-clock for a block, and never swallow a traceback."""
    log = logger or get_logger()
    t0 = time.perf_counter()
    log.info(f"START  {label}")
    try:
        yield
    except Exception:
        log.error(f"FAILED {label} after {time.perf_counter() - t0:.1f}s")
        log.error(traceback.format_exc())
        raise
    else:
        log.info(f"DONE   {label}  ({time.perf_counter() - t0:.1f}s)")


# --------------------------------------------------------------------------
# Resumable progress (1c.3 item 3)
# --------------------------------------------------------------------------


def _load_progress() -> Dict[str, Any]:
    if PROGRESS_PATH.exists():
        try:
            return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            get_logger().error(f"{PROGRESS_PATH} is corrupt; starting a fresh 'runs' block")
    return {}


def progress_key(experiment: str, method: str, model: str,
                 variant: str = "default", seed: int = 42) -> str:
    return f"{experiment}|{method}|{model}|{variant}|seed{seed}"


def is_done(key: str, force: bool = False) -> bool:
    """Whether this combination already completed (skip unless ``--force``)."""
    if force:
        return False
    return _load_progress().get("runs", {}).get(key, {}).get("status") == "done"


def mark(key: str, status: str, **extra) -> None:
    """Record a combination's outcome. Written immediately so a kill is safe."""
    data = _load_progress()
    data.setdefault("runs", {})[key] = {
        "status": status,
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **extra,
    }
    data["last_updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tmp = PROGRESS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, PROGRESS_PATH)


# --------------------------------------------------------------------------
# Skip tracking (1c.2 item 3)
# --------------------------------------------------------------------------

_SKIP_FIELDS = ["timestamp", "machine", "experiment", "cohort", "method",
                "model", "variant", "reason", "detail"]


def record_skip(experiment: str, cohort: str, method: str, model: str,
                reason: str, detail: str = "", variant: str = "default") -> None:
    """Append to ``skipped_combinations.csv``. A missing model must never abort."""
    SKIPPED_PATH.parent.mkdir(parents=True, exist_ok=True)
    new = not SKIPPED_PATH.exists()
    with open(SKIPPED_PATH, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_SKIP_FIELDS)
        if new:
            w.writeheader()
        w.writerow({
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "machine": P.MACHINE, "experiment": experiment, "cohort": cohort,
            "method": method, "model": model, "variant": variant,
            "reason": reason, "detail": detail[:400],
        })
    get_logger().warning(f"SKIP {experiment} {cohort}/{method}/{model}: {reason}")


def coverage_report(experiment: str, ran: Iterable[tuple], skipped: Iterable[tuple],
                    logger: Optional[logging.Logger] = None) -> None:
    """Print which combinations ran here and which did not (1c.2 item 6)."""
    log = logger or get_logger()
    ran, skipped = list(ran), list(skipped)
    log.info("=" * 70)
    log.info(f"COVERAGE - {experiment} on machine={P.MACHINE}")
    log.info(f"  ran     ({len(ran)}): " + (", ".join(f"{m}/{n}" for m, n in ran) or "none"))
    log.info(f"  skipped ({len(skipped)}): " + (", ".join(f"{m}/{n}" for m, n in skipped) or "none"))
    if skipped:
        log.info(f"  -> rerun the identical code with MACHINE=server to pick these up, "
                 f"then merge with tools/merge_results.py")
    log.info("=" * 70)
