"""Run PAIP-IV four times and collect every result into a single CSV.

Each run is exactly::

    python runners/iv_runner.py --force [any flags you passed through]

Flags this script does not recognise are forwarded verbatim to the runner, so
the repeats can be done under any configuration::

    python run_iv_repeats.py --ann-protocol full --classifiers ann

With no forwarded flags the runner's own defaults apply - seed 42 and the
``legacy`` ANN protocol. ``--force`` is always added, and only bypasses the
resume log so the run happens rather than being skipped.

Whatever configuration you choose is held FIXED across the repeats: the point is
to repeat one configuration, not to compare several.

What this measures
------------------
A reproducibility check. The runner seeds ``torch`` and ``numpy`` before every
fit and draws the early-stopping carve-out with a seeded stratified split, so
with the configuration held fixed the four runs are expected to come out
identical. The script verifies that cell by cell rather than assuming it.

If they are identical, every figure the pipeline reports is exact rather than
one sample from a distribution - which is worth being able to state, and also
rules out GPU nondeterminism (not guaranteed for BatchNorm, and not something
the pipeline configures away).

If they are not, the difference is real nondeterminism and its size is the
precision floor for any quoted result.

This cannot tell you how much a result would move under a different random
draw - that would need the seed varied, which changes a setting, which is
exactly what this script does not do.

Output
------
One file: ``paip_iv_repeats.csv``. Every row of every run, with a ``run`` column
and the full hyperparameter configuration that produced it. No per-run directory
tree is kept; the last run stays as the live ``PAIP_IV_Results/`` so its
predictions and checkpoints remain available.

Run:  python run_iv_repeats.py                      # 4 runs, all five heads
      python run_iv_repeats.py --classifiers ann    # 4 runs, ANN only
      python run_iv_repeats.py --runs 6
      python run_iv_repeats.py --from-existing      # rebuild CSV, no re-run
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
LIVE = HERE / "PAIP_IV_Results"
OUT_CSV = HERE / "paip_iv_repeats.csv"
LEGACY_REPEATS = HERE / "PAIP_IV_Results_ARCHIVED_repeats"

#: Columns compared between runs - everything a result is ever quoted from.
COMPARED = ("BalAcc", "BalAcc_CI_low", "BalAcc_CI_high",
            "AUROC", "AUROC_CI_low", "AUROC_CI_high",
            "Acc", "MacroF1", "Threshold", "ConfMatrix")

Key = Tuple[str, str, str]   # (method, model, classifier)


# ------------------------------------------------------- configuration source
#
# Some hyperparameters never reach any output file - they are literals inside
# nn.Sequential and fit(). Transcribing them into this script would mean the CSV
# keeps reporting 0.3 after someone changes the dropout, i.e. silently wrong.
#
# So they are READ FROM THE SOURCE at runtime instead:
#
#   GRIDS                 imported from runners.classifiers
#   dropout / depth       introspected from a throwaway ANNBinaryClassifier
#   lr / wd / patience    parsed out of ANNBinaryClassifier.fit's own source
#   seed / carve-out      read from runners.iv_runner's signature and constants
#
# Anything that cannot be determined is recorded as "?" rather than guessed.
# If the project environment is not importable (running this outside the conda
# env, say) the whole block degrades to "?" with a loud warning - never to a
# stale literal.

UNKNOWN = "?"


def _introspect() -> Dict[str, Dict[str, object]]:
    """Read the fixed hyperparameters out of the code that implements them."""
    import inspect
    import re

    sys.path.insert(0, str(HERE))
    import torch.nn as nn
    from runners.classifiers import GRIDS
    from eval_patch_features.ann import ANNBinaryClassifier
    from runners import iv_runner

    # --- ANN architecture: build a tiny instance and look at the real modules.
    probe = ANNBinaryClassifier(input_dim=4, hidden_dim1=3, hidden_dim2=2, verbose=False)
    layers = list(probe.model)
    dropouts = sorted({m.p for m in layers if isinstance(m, nn.Dropout)})
    n_linear = sum(isinstance(m, nn.Linear) for m in layers)
    has_softmax = any(isinstance(m, nn.Softmax) for m in layers)

    # --- Optimiser and stopping: literals inside fit(), so parse its source.
    #
    # Two traps here, both hit while testing this:
    #   * ann.py keeps a commented-out `# opt = optim.Adam(..., lr=1e-4)` above
    #     the live one, and a naive search matches the dead line first. Comments
    #     are stripped before matching.
    #   * the live call contains `self.model.parameters()`, so a `[^)]*` bridge
    #     from "optim.Adam(" to "lr=" stops at that inner ")". Matched on the
    #     assignment itself instead, which has no such problem.
    raw = inspect.getsource(ANNBinaryClassifier.fit)
    src = "\n".join(re.sub(r"#.*$", "", ln) for ln in raw.splitlines())

    def grab(pattern, cast=float):
        m = re.search(pattern, src)
        return cast(m.group(1)) if m else UNKNOWN

    # Prefer the instance attributes. Once these became constructor parameters
    # the literals vanished from fit() and source-matching started returning "?".
    lr = getattr(probe, "lr", None)
    wd = getattr(probe, "weight_decay", None)
    patience = getattr(probe, "patience", None)
    if lr is None:
        lr = grab(r"\blr\s*=\s*([\d.eE+\-]+)")
    if wd is None:
        wd = grab(r"\bweight_decay\s*=\s*([\d.eE+\-]+)")
    if patience is None:
        patience = grab(r"patience,\s*epochs_no_improve\s*=\s*(\d+)", int)
    batched = "DataLoader" in src

    g = GRIDS["ann"]
    space = (f"h1{{{','.join(map(str, g['hidden_dim1']))}}} x "
             f"h2{{{','.join(map(str, g['hidden_dim2']))}}} x "
             f"iter{{{','.join(map(str, g['max_iter']))}}}")
    grid_size = len(g["hidden_dim1"]) * len(g["hidden_dim2"]) * len(g["max_iter"])

    def flat(kind):
        return ", ".join(f"{k}={v[0]}" for k, v in GRIDS[kind].items()) or "none"

    return {
        "ann": {"hp_arch": f"{n_linear - 1} hidden layer(s)"
                           + (" [SOFTMAX IN GRAPH]" if has_softmax else ""),
                "hp_dropout": dropouts[0] if len(dropouts) == 1 else dropouts,
                "hp_patience": patience, "hp_lr": lr, "hp_weight_decay": wd,
                "hp_batching": "mini-batch" if batched else "full-batch",
                "hp_grid_size": grid_size, "hp_search_space": space},
        "lin": {"hp_arch": "logistic regression", "hp_other": flat("lin"),
                "hp_grid_size": 1, "hp_search_space": "single point"},
        "knn": {"hp_arch": "k-nearest neighbours",
                "hp_other": f"{flat('knn')}, L2-normalised; eval_knn runs its own "
                            f"30-point GridSearchCV on the training data",
                "hp_grid_size": 30, "hp_search_space": "nested CV on train"},
        "proto": {"hp_arch": "class-mean prototypes",
                  "hp_other": "L2-normalised, euclidean", "hp_grid_size": 0,
                  "hp_search_space": "none"},
        "rf": {"hp_arch": "random forest", "hp_other": flat("rf"),
               "hp_grid_size": 1, "hp_search_space": "single point"},
        "_meta": {"seed": inspect.signature(iv_runner.run_paip_iv)
                                 .parameters["seed"].default,
                  "val_frac": getattr(iv_runner, "ANN_VAL_FRAC", UNKNOWN)},
    }


try:
    CODE_FIXED: Dict[str, Dict[str, object]] = _introspect()
    RUNNER_DEFAULT_SEED = CODE_FIXED["_meta"]["seed"]
    _SOURCED = True
except Exception as exc:                                  # pragma: no cover
    print(f"[WARN] could not read the configuration from source ({type(exc).__name__}: "
          f"{exc}).\n       hp_* columns will be '{UNKNOWN}' rather than guessed - "
          f"run this inside the project environment to populate them.")
    CODE_FIXED = defaultdict(lambda: {c: UNKNOWN for c in (
        "hp_arch", "hp_dropout", "hp_patience", "hp_lr", "hp_weight_decay",
        "hp_batching", "hp_grid_size", "hp_search_space", "hp_other")})
    RUNNER_DEFAULT_SEED = UNKNOWN
    _SOURCED = False

HP_COLS = ("hp_seed", "hp_protocol", "hp_preset", "hp_arch", "hp_hidden_dim1", "hp_hidden_dim2", "hp_max_iter",
           "hp_dropout", "hp_patience", "hp_lr", "hp_weight_decay",
           "hp_batching", "hp_input_dim", "hp_grid_size", "hp_search_space",
           "hp_selected_on", "hp_selection_value", "hp_n_train_fitted",
           "hp_n_held_out", "hp_other")


def hparams_for(tree: Path, method: str, model: str, clf: str) -> Dict[str, object]:
    """The configuration that produced one row: code-fixed values plus whatever
    the run recorded - for the ANN, which grid point was selected and on what."""
    hp: Dict[str, object] = dict(CODE_FIXED.get(clf, {}))
    hp["hp_seed"] = RUNNER_DEFAULT_SEED

    # PRISM sits one level shallower than the rest - see config/paths.results_root.
    parts = ("PRISM",) if method == "PRISM" else (method, model)
    out = tree.joinpath(*parts, "1-MSIH")
    jsons = list((out / "Output").glob("result_PAIP-IV_*.json"))
    if not jsons:
        return hp

    entry = json.loads(jsons[0].read_text(encoding="utf-8")).get("results", {}).get(clf, {})
    n_train, n_val = entry.get("n_train"), entry.get("n_ann_early_stop")

    # Trust the runner's own n_train_effective when present - it knows which
    # protocol ran. Only fall back to arithmetic for older result files that
    # predate that field, where the ANN always held the carve-out back.
    eff = entry.get("n_train_effective")
    if eff is not None:
        hp["hp_n_train_fitted"] = eff
    elif clf == "ann" and n_train is not None and n_val is not None:
        hp["hp_n_train_fitted"] = n_train - n_val
    else:
        hp["hp_n_train_fitted"] = n_train
    hp["hp_n_held_out"] = (n_val or 0) if clf == "ann" else 0
    hp["hp_protocol"] = entry.get("ann_protocol") if clf == "ann" else None

    sel = entry.get("selection") or {}
    chosen = sel.get("selected") or {}
    hp["hp_hidden_dim1"] = chosen.get("hidden_dim1")
    hp["hp_hidden_dim2"] = chosen.get("hidden_dim2")
    hp["hp_max_iter"] = chosen.get("max_iter")
    hp["hp_selected_on"] = sel.get("selection_metric")
    hp["hp_selection_value"] = sel.get("selection_value")
    if sel.get("grid_trace"):
        hp["hp_grid_size"] = len(sel["grid_trace"])

    # The checkpoint config is written BY THE RUN, so it is the authority on what
    # was actually trained - architecture, dropout, optimiser, the lot. The
    # introspected CODE_FIXED values are only defaults read from a throwaway
    # probe; they describe the source, not this run. Anything recorded here wins.
    #
    # This matters: a --ann-arch shallow or --ann-preset run trains a
    # single-hidden-layer network with a different dropout, and reporting the
    # probe's "2 hidden layers / 0.3" for it is simply wrong.
    cfgs = list((out / "models").glob(f"fold0_{clf}_config.json"))
    if cfgs:
        cfg = json.loads(cfgs[0].read_text(encoding="utf-8"))
        hp["hp_input_dim"] = cfg.get("input_dim")
        for key, col in (("n_hidden_layers", "hp_arch"), ("dropout", "hp_dropout"),
                         ("patience", "hp_patience"), ("lr", "hp_lr"),
                         ("weight_decay", "hp_weight_decay"),
                         ("hidden_dim1", "hp_hidden_dim1"),
                         ("hidden_dim2", "hp_hidden_dim2"),
                         ("max_iter", "hp_max_iter")):
            if cfg.get(key) is not None:
                hp[col] = (f"{cfg[key]} hidden layer(s)" if col == "hp_arch"
                           else cfg[key])
        if cfg.get("softmax_in_graph"):
            hp["hp_arch"] = str(hp.get("hp_arch", "")) + " [SOFTMAX IN GRAPH]"

    # The stamp records the protocol and preset the sweep was invoked with.
    stamp = json.loads(jsons[0].read_text(encoding="utf-8")).get("stamp", {})
    if clf == "ann":
        hp["hp_protocol"] = stamp.get("ann_protocol", hp.get("hp_protocol"))
        hp["hp_preset"] = stamp.get("ann_preset") or ""
        if stamp.get("ann_held_back_for_tuning") is not None:
            hp["hp_n_held_out"] = stamp["ann_held_back_for_tuning"]
    return hp


def harvest(tree: Path, run: int) -> List[dict]:
    """Every summary row in one tree, tagged with run number and configuration."""
    rows = []
    for f in sorted(tree.rglob("summary_iv.csv")):
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                hp = hparams_for(tree, r["Method"], r["Model"], r["Classifier"])
                rows.append({"run": run, **r, **{c: hp.get(c, "") for c in HP_COLS}})
    return rows


def write_csv(rows: List[dict], path: Path) -> None:
    """Write one CSV, atomically.

    Two hard-won details:

    * Field names are the UNION of every row's keys, not just the first row's.
      Trees written by different revisions carry different columns (older ones
      have no AnnProtocol / N_train_effective), and DictWriter raises on any key
      it was not told about.
    * The rows go to a temporary file which is then renamed over the target. The
      previous version opened the destination with "w" - truncating it - and
      then raised on the mismatched columns, destroying a completed run's
      results. A write that fails must leave the old file untouched.
    """
    if not rows:
        raise SystemExit("no rows to write - did any run produce results?")

    cols, seen = ["run"], {"run"}
    for r in rows:
        for c in r:
            if c not in seen:
                seen.add(c)
                cols.append(c)
    rows = sorted(rows, key=lambda r: (r.get("Method", ""), r.get("Model", ""),
                                       r.get("Classifier", ""), r["run"]))

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)
    print(f"\n  wrote {len(rows)} rows x {len(cols)} columns -> {path.name}")


def run_once(n: int, extra: List[str]) -> None:
    """One full PAIP-IV sweep, exactly as the CLI would do it."""
    cmd = [sys.executable, "runners/iv_runner.py", "--force", *extra]
    print(f"\n{'=' * 72}\n  RUN {n}:  {' '.join(cmd)}\n{'=' * 72}", flush=True)
    if subprocess.run(cmd, cwd=HERE).returncode != 0:
        raise SystemExit(f"run {n} failed")
    if not LIVE.is_dir():
        raise SystemExit(f"run {n} produced no {LIVE.name}/")


# ---------------------------------------------------------------- reporting

def by_run(rows: List[dict]) -> Dict[int, Dict[Key, dict]]:
    out: Dict[int, Dict[Key, dict]] = defaultdict(dict)
    for r in rows:
        out[r["run"]][(r["Method"], r["Model"], r["Classifier"])] = r
    return out


def compare(rows: List[dict]) -> bool:
    """Compare every run against the first. True if all are identical."""
    runs = by_run(rows)
    ids = sorted(runs)
    print(f"\n{'=' * 72}\n  COMPARISON\n{'=' * 72}")

    base = runs[ids[0]]
    print(f"  run{ids[0]}: {len(base)} rows "
          f"({len({k[:2] for k in base})} combinations x "
          f"{len({k[2] for k in base})} classifier(s))")

    identical = True
    for i in ids[1:]:
        other = runs[i]
        missing, extra = set(base) - set(other), set(other) - set(base)
        diffs = [(k, c, base[k][c], other[k][c])
                 for k in sorted(set(base) & set(other))
                 for c in COMPARED
                 if c in base[k] and base[k][c] != other[k][c]]

        if not (missing or extra or diffs):
            print(f"  run{i}: identical to run{ids[0]}  "
                  f"({len(other)} rows x {len(COMPARED)} columns)")
            continue

        identical = False
        print(f"  run{i}: DIFFERS from run{ids[0]}")
        if missing:
            print(f"          {len(missing)} row(s) only in run{ids[0]}")
        if extra:
            print(f"          {len(extra)} row(s) only in run{i}")
        for (m, mo, clf), col, a, b in diffs[:10]:
            print(f"          {m}/{mo}/{clf:<5} {col:<14} {a}  ->  {b}")
        if len(diffs) > 10:
            print(f"          ... and {len(diffs) - 10} more")

    # The selected ANN configuration is part of the result, so check it too.
    g: Dict[Key, List[dict]] = defaultdict(list)
    for r in rows:
        g[(r["Method"], r["Model"], r["Classifier"])].append(r)
    varied = [k for k, rs in g.items() if k[2] == "ann" and
              len({(r["hp_hidden_dim1"], r["hp_hidden_dim2"], r["hp_max_iter"])
                   for r in rs}) > 1]
    n_ann = sum(1 for k in g if k[2] == "ann")
    if n_ann:
        print(f"\n  ANN configuration selected identically in "
              f"{n_ann - len(varied)} of {n_ann} combinations")
    return identical


def per_run_means(rows: List[dict]) -> None:
    runs = by_run(rows)
    classifiers = sorted({k[2] for r in runs.values() for k in r})
    print(f"\n  mean BalAcc per classifier, per run")
    print("    " + "run".ljust(8) + "".join(f"{c:>10}" for c in classifiers))
    for i in sorted(runs):
        vals: Dict[str, List[float]] = defaultdict(list)
        for (_, _, clf), r in runs[i].items():
            vals[clf].append(float(r["BalAcc"]))
        line = f"    run{i}".ljust(12)
        for c in classifiers:
            v = vals.get(c)
            line += f"{(sum(v) / len(v)):>10.4f}" if v else f"{'-':>10}"
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run PAIP-IV N times unchanged; collect all results into one CSV.")
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--classifiers", default=None,
                    help="passed through to iv_runner.py; selects which heads run, "
                         "does not change any of them")
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    ap.add_argument("--from-existing", action="store_true",
                    help="do not run anything; rebuild the CSV from trees on disk")
    ap.add_argument("--keep-trees", action="store_true",
                    help="do not delete each run's tree after harvesting it")
    # Anything else is a runner flag - forwarded verbatim, unparsed, so this
    # script never has to track iv_runner's CLI.
    args, forwarded = ap.parse_known_args()

    rows: List[dict] = []

    if args.from_existing:
        for t in sorted(LEGACY_REPEATS.glob("run*"),
                        key=lambda p: int(p.name.removeprefix("run") or 0)):
            n = int(t.name.removeprefix("run"))
            got = harvest(t, n)
            print(f"  harvested {len(got):>4} rows from {t.name}")
            rows += got
        if LIVE.is_dir():
            n = max((r["run"] for r in rows), default=0) + 1
            got = harvest(LIVE, n)
            sig = {(r["Method"], r["Model"], r["Classifier"]):
                   tuple(r[c] for c in COMPARED) for r in got}
            dupe = next((i for i, prev in by_run(rows).items()
                         if {k: tuple(v[c] for c in COMPARED)
                             for k, v in prev.items()} == sig), None)
            if got and dupe is not None:
                print(f"  skipped {LIVE.name}: identical to run{dupe}")
            elif got:
                print(f"  harvested {len(got):>4} rows from {LIVE.name} (as run{n})")
                rows += got
        if not rows:
            raise SystemExit("nothing to harvest")
    else:
        if args.runs < 2:
            raise SystemExit("--runs must be at least 2 for a comparison to mean anything")
        extra = (["--classifiers", args.classifiers] if args.classifiers else []) + forwarded

        print(f"PAIP-IV x {args.runs}, one configuration repeated.")
        if forwarded:
            print(f"  forwarded to runner : {' '.join(forwarded)}")
        else:
            print(f"  nothing forwarded   : runner defaults (seed "
                  f"{RUNNER_DEFAULT_SEED}, ANN protocol 'legacy')")
        print(f"  command per run : python runners/iv_runner.py --force {' '.join(extra)}")
        print(f"  output          : {args.out.name}")
        print(f"  expectation     : with the configuration fixed, identical results.")

        for n in range(1, args.runs + 1):
            run_once(n, extra)
            got = harvest(LIVE, n)
            rows += got
            print(f"  harvested {len(got)} rows from run {n}")
            # Remove the tree so the next sweep has nothing to archive; keep the last.
            if n < args.runs and not args.keep_trees:
                shutil.rmtree(LIVE)

    write_csv(rows, args.out)
    print(f"  hyperparameters read from source: {'yes' if _SOURCED else 'NO - columns are ' + UNKNOWN}")
    identical = compare(rows)
    per_run_means(rows)

    n_runs = len({r["run"] for r in rows})
    print(f"\n{'=' * 72}")
    if identical:
        print(f"  RESULT: all {n_runs} runs are bit-identical.")
        print("          The pipeline is reproducible at fixed configuration; any")
        print("          number quoted from it is exact, not an average over runs.")
    else:
        print(f"  RESULT: the {n_runs} runs are NOT identical.")
        print("          With the configuration fixed this is genuine nondeterminism")
        print("          (GPU reduction order / cuDNN algorithm choice). The spread")
        print("          above is the precision floor for any quoted result.")
    if LIVE.is_dir():
        print(f"\n  {LIVE.name}/ holds the final run - predictions and checkpoints included.")
    print("=" * 72)


if __name__ == "__main__":
    main()
