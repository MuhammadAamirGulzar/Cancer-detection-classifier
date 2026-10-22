"""Check every file under runs/<arm> for crash damage: an unclean shutdown leaves recent files full of zero bytes.

Exit 1 (and list the files) if any file is zero-filled or does not parse (json / csv / joblib / torch).

  <conda>/envs/exaonepath/python.exe experiments/composition/03_scan_runs.py comp_fivecrop
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent


def check(p):
    b = p.read_bytes()
    if b and not any(b):
        return "all zero bytes"
    try:
        if p.suffix == ".json":
            json.loads(b.decode("utf-8"))
        elif p.suffix == ".csv":
            pd.read_csv(p)
        elif p.suffix in (".pkl", ".joblib"):
            joblib.load(p)
        elif p.suffix in (".pt", ".pth"):
            torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:80]}"
    return None


def main():
    root = HERE / "runs" / sys.argv[1]
    files = [p for p in root.rglob("*") if p.is_file()]
    bad = [(p, e) for p in files if (e := check(p))]
    print(f"{sys.argv[1]}: {len(files)} files checked, {len(bad)} damaged")
    for p, e in bad:
        print(f"  {p.relative_to(root)}: {e}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
