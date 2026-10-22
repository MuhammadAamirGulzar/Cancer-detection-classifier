"""Regenerate unreadable no-crop patch-feature files listed by 03b_build_missing_slides.py.

Per (cohort, encoder, slide): re-extract the slide with Feature_Extraction_nocrop_clean.ipynb's own code into
a temporary folder, require the slide's intact files to come out identical (so the regenerated ones are what
was there), then atomically overwrite only the damaged files and remove the temporary folder.
The source PNGs must decode; a slide whose PNG is unreadable is reported, not touched.

  <conda>/envs/exaonepath/python.exe experiments/nocrop_vs_fivecrop/03c_repair_unreadable.py
"""

import json
import os
import shutil
import sys
import types
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NB = REPO / "feature_extraction" / "Feature_Extraction_nocrop_clean.ipynb"
NAME = {"UNI2": "uni2", "H-Optimus-1": "h_optimus_1", "ConchV1": "conch_v1", "Conch1_5": "conch_v1_5", "Virchow2": "virchow2"}


def main():
    todo = defaultdict(list)          # (cohort, encoder, slide) -> [damaged .pt paths]
    for f in sorted((HERE / "results").glob("unreadable_*.txt")):
        _, cohort, model = f.stem.split("_", 2)
        for line in filter(None, f.read_text(encoding="utf-8").splitlines()):
            p = Path(line)
            todo[(cohort, model, p.parent.name)].append(p)
    if not todo:
        print("nothing to repair")
        return

    cells = {c["id"]: "".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"]}
    m = types.ModuleType("__nb__"); sys.modules["__nb__"] = m; g = m.__dict__
    for cid in ("c02", "c04", "c06", "c13"):
        exec(compile(cells[cid], cid, "exec"), g)

    for (cohort, model, slide), damaged in todo.items():
        live = damaged[0].parent
        tmp = live.parents[1] / "_repair_tmp"
        print(f"\n{cohort} {model} {slide}: {len(damaged)} damaged file(s)", flush=True)
        g["run_nocrop"](cohort, [NAME[model]], slides=[slide], out_root=tmp)
        new = tmp / live.parent.name / slide
        names = {p.name for p in damaged}
        intact = [f for f in sorted(os.listdir(live)) if f.endswith(".pt") and f not in names][:20]
        diff = max((torch.load(new / f) - torch.load(live / f)).abs().max().item() for f in intact)
        if diff > 1e-3:
            print(f"  STOP: regenerated intact files differ from the originals (max {diff:.2e}); nothing replaced")
            continue
        for p in damaged:
            shutil.copyfile(new / p.name, str(p) + ".tmp")
            os.replace(str(p) + ".tmp", p)
        still = 0
        for f in os.listdir(live):
            try:
                torch.load(live / f)
            except Exception:
                still += 1
        shutil.rmtree(tmp)
        print(f"  intact files identical (max diff {diff:.1e}); replaced {len(damaged)}; unreadable left in slide: {still}")


if __name__ == "__main__":
    main()
