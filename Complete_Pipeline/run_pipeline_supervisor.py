"""
Auto-restart supervisor for combined_pipeline_final.py
====================================================
The pipeline is resume-safe (skips already-downloaded CZIs, already-computed
per-slide CSVs, and already-extracted .pt feature files), so if the child
process dies — e.g. from the native pylibCZIrw access-violation crashes seen
repeatedly in Windows Event Viewer (Application Error, python.exe faulting in
_pylibCZIrw.cp312-win_amd64.pyd) — simply relaunching it lets it pick up
exactly where it left off instead of losing the rest of an unattended run.

A crash is NOT the same as the pipeline's own deliberate low-disk-space halt
(see check_disk_space_or_halt in combined_pipeline_final.py) — that's a graceful
sys.exit(1) with a "[HALT] Less than ... GB free" message, and restarting
won't fix a full disk. The supervisor detects that case and stops instead of
looping.

Usage:
    python run_pipeline_supervisor.py
"""

import collections
import datetime
import os
import subprocess
import sys
import time

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
PIPELINE_SCRIPT = os.path.join(SCRIPT_DIR, "combined_pipeline_final_error_checks_v6.py")
LOG_DIR         = os.path.join(SCRIPT_DIR, "logs")

RESTART_COOLDOWN_SECONDS = 6     # let GPU/OS resources settle before relaunch
MAX_CRASHES_IN_WINDOW    = 5      # crash-loop guard
CRASH_WINDOW_SECONDS     = 240
DISK_HALT_MARKER         = "[HALT] Less than"
TAIL_LINES_KEPT          = 200    # enough to reliably catch the halt marker


def _log_path():
    os.makedirs(LOG_DIR, exist_ok=True)
    return os.path.join(LOG_DIR, f"pipeline_{datetime.date.today():%Y%m%d}.log")

def _stamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def supervisor_log(log_fh, msg):
    line = f"[SUPERVISOR {_stamp()}] {msg}"
    print(line, flush=True)
    log_fh.write(line + "\n")
    log_fh.flush()

def run_once(log_fh):
    """Launch the pipeline once, streaming its output to console + log file.
    Returns (returncode, tail_lines)."""
    supervisor_log(log_fh, f"Launching: {sys.executable} {PIPELINE_SCRIPT}")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    proc = subprocess.Popen(
        [sys.executable, "-u", PIPELINE_SCRIPT],
        cwd=SCRIPT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )

    tail = collections.deque(maxlen=TAIL_LINES_KEPT)
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            print(line, flush=True)
            log_fh.write(line + "\n")
            tail.append(line)
        log_fh.flush()
    except KeyboardInterrupt:
        supervisor_log(log_fh, "Ctrl+C received — terminating child process ...")
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise

    proc.wait()
    return proc.returncode, tail


def main():
    log_fh = open(_log_path(), "a", encoding="utf-8")
    supervisor_log(log_fh, "=" * 70)
    supervisor_log(log_fh, "Pipeline supervisor started.")

    crash_times = collections.deque()

    try:
        while True:
            returncode, tail = run_once(log_fh)

            if returncode == 0:
                supervisor_log(log_fh, "Pipeline exited cleanly (all batches complete). Supervisor stopping.")
                break

            if any(DISK_HALT_MARKER in line for line in tail):
                supervisor_log(
                    log_fh,
                    f"Pipeline halted itself due to low disk space (exit code {returncode}). "
                    f"This is a deliberate stop, not a crash — free up space on the monitored "
                    f"drive and rerun the supervisor. Not auto-restarting."
                )
                break

            now = time.time()
            crash_times.append(now)
            while crash_times and now - crash_times[0] > CRASH_WINDOW_SECONDS:
                crash_times.popleft()

            supervisor_log(
                log_fh,
                f"Pipeline exited with code {returncode} (crash #{len(crash_times)} "
                f"in the last {CRASH_WINDOW_SECONDS // 60} min)."
            )

            if len(crash_times) >= MAX_CRASHES_IN_WINDOW:
                supervisor_log(
                    log_fh,
                    f"{MAX_CRASHES_IN_WINDOW} crashes within {CRASH_WINDOW_SECONDS // 60} minutes — "
                    f"this looks like more than an occasional native crash. Stopping the supervisor "
                    f"to avoid a tight crash loop; investigate before rerunning."
                )
                break

            supervisor_log(log_fh, f"Restarting in {RESTART_COOLDOWN_SECONDS}s ...")
            time.sleep(RESTART_COOLDOWN_SECONDS)
    except KeyboardInterrupt:
        supervisor_log(log_fh, "Supervisor stopped by user (Ctrl+C).")
    finally:
        log_fh.close()


if __name__ == "__main__":
    main()
