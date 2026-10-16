"""Sella TS optimisation started from generated TS geometries.

Does a sample converge to a first-order saddle, how fast, and is it the same
saddle that the reference TS converges to? The IRC from these saddles is
irc_check.py.

The surface is calculator=<name> (calculators.py): "aimnet", the ensemble mean of
AIMNet2-rxn (default), or "gxtb".

Every run is started on the same test reactions (all, or a random subset), together
with the RDB7 reference TS, which is optimised the same way and gives the
"correct" saddle of each reaction on that surface.

    # one array task: optimise its share of the (run, reaction) pairs
    PYTHONPATH=. .venv/bin/python sella_check.py run <out_dir> <task> <n_tasks> label=pkl ...
    # after all tasks: per-run summary against the reference
    PYTHONPATH=. .venv/bin/python sella_check.py summarize <out_dir>

An input reference_from=<earlier out_dir> skips the reference optimisations
and makes summarize reuse that check's; its settings must be the same.
data=<pkl> takes the reference TSs from that pickle without optimising its
samples, so a check with only data= optimises just the references.
n_reactions=<k> takes a random subset of k reactions (seed 0), the same one
irc_check.py takes.
"""

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

from calculators import ENSEMBLE_MEMBERS, make_calculator

SUBSET_SEED = 0
FMAX = 0.01  # eV/A
MAX_STEPS = 300
IMAG_CUTOFF = 50.0  # cm^-1; smaller imaginary modes are numerical noise
# sqrt(eV / (amu A^2)) in cm^-1
EV_AMU_A2_TO_CM1 = 521.47
POS_FIT_IDX = 1  # configs/train.yaml: the TS geometry in data.pos


def subset(rxn_indices, n_reactions=None):
    """Every reaction, or a random subset of n_reactions (seed SUBSET_SEED)."""
    if n_reactions is None:
        return sorted(rxn_indices)
    rng = np.random.default_rng(SUBSET_SEED)
    return sorted(rng.choice(sorted(rxn_indices), size=n_reactions, replace=False).tolist())


def kabsch_rmsd(a, b, allow_reflection=False):
    a, b = a - a.mean(0), b - b.mean(0)
    U, _, Vt = np.linalg.svd(a.T @ b)
    d = 1.0 if allow_reflection else np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    return float(np.sqrt(((a @ R - b) ** 2).sum(-1).mean()))


def n_imaginary(hessian, masses):
    """Imaginary modes above IMAG_CUTOFF of the mass-weighted Hessian (eV/A^2)."""
    m = np.repeat(masses, 3) ** -0.5
    w = np.linalg.eigvalsh(hessian * m[:, None] * m[None, :])
    freqs = np.sign(w) * np.sqrt(np.abs(w)) * EV_AMU_A2_TO_CM1
    return int((freqs < -IMAG_CUTOFF).sum()), float(freqs.min())


def settings(calculator="aimnet", n_reactions=None):
    out = dict(
        calculator=calculator,
        fmax=FMAX,
        max_steps=MAX_STEPS,
        n_reactions=n_reactions,
        subset_seed=SUBSET_SEED,
    )
    if calculator == "aimnet":
        out["ensemble_members"] = list(ENSEMBLE_MEMBERS)
    return out


def read_settings(check_dir):
    return json.loads((Path(check_dir) / "settings.json").read_text())


def read_jsonl(files):
    """All records of the given task*.jsonl files, one dict per line."""
    rows = []
    for f in files:
        with open(f) as fh:
            rows += [json.loads(line) for line in fh]
    return rows


def check_reference_settings(ref_dir, current):
    ref = read_settings(ref_dir)
    if ref != current:
        raise ValueError(f"reference_from={ref_dir} used other settings: {ref} vs {current}")


def optimise(numbers, positions, ase_calc):
    from ase import Atoms
    from sella import Sella

    atoms = Atoms(numbers=numbers, positions=positions)
    atoms.calc = ase_calc
    # curvature at the start: 0 imaginary modes means a minimum-like start,
    # which a low force alone would also reward
    try:
        n_imag_start, lowest_start = n_imaginary(
            np.asarray(ase_calc.get_hessian(atoms)).reshape(3 * len(atoms), -1), atoms.get_masses()
        )
    except Exception as e:  # e.g. a g-xTB SCF failure at the start geometry
        return dict(
            converged=False,
            steps=0,
            fmax=np.nan,
            n_imag=-1,
            lowest_freq=np.nan,
            n_imag_start=-1,
            lowest_freq_start=np.nan,
            seconds=0.0,
            error=f"start Hessian: {type(e).__name__}: {e}"[:200],
        ), positions
    opt = Sella(
        atoms,
        order=1,
        internal=True,
        hessian_function=ase_calc.get_hessian,
        logfile=None,
        trajectory=None,
    )
    t0 = time.time()
    try:
        converged = bool(opt.run(fmax=FMAX, steps=MAX_STEPS))
        error = ""
    except Exception as e:  # a failed optimisation is a result, not a crash
        converged, error = False, f"{type(e).__name__}: {e}"[:200]
    final = atoms.get_positions()
    fmax = float(np.sqrt((atoms.get_forces() ** 2).sum(-1)).max()) if not error else np.nan
    try:
        n_imag, lowest = (
            n_imaginary(
                np.asarray(ase_calc.get_hessian(atoms)).reshape(3 * len(atoms), -1),
                atoms.get_masses(),
            )
            if not error
            else (-1, np.nan)
        )
    except Exception as e:
        n_imag, lowest, error = -1, np.nan, f"final Hessian: {type(e).__name__}: {e}"[:200]
    return dict(
        converged=converged,
        steps=int(opt.nsteps),
        fmax=fmax,
        n_imag=n_imag,
        lowest_freq=lowest,
        n_imag_start=n_imag_start,
        lowest_freq_start=lowest_start,
        seconds=time.time() - t0,
        error=error,
    ), final


def run(out_dir, task, n_tasks, inputs):
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 2)))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs, source, reference_from, calculator, n_reactions = {}, None, None, "aimnet", None
    for item in inputs:
        label, path = item.split("=", 1)
        if label == "calculator":
            calculator = path
            continue
        if label == "n_reactions":
            n_reactions = int(path)
            continue
        if label == "reference_from":
            reference_from = path
            continue
        if label == "data":
            source = {int(d.rxn_index): d for d in pickle.load(open(path, "rb"))}
            continue
        runs[label] = {int(d.rxn_index): d for d in pickle.load(open(path, "rb"))}
    current = settings(calculator, n_reactions)
    (out_dir / "settings.json").write_text(json.dumps(current))
    ase_calc = make_calculator(calculator)
    # the reference TS is the same in every pickle; take it from data= or the first run,
    # unless an earlier check already optimised it (summarize reads it from there)
    first = source if source is not None else next(iter(runs.values()))
    chosen = subset(first.keys(), n_reactions)
    if reference_from is not None:
        check_reference_settings(reference_from, current)
        (out_dir / "reference_from.txt").write_text(str(Path(reference_from).resolve()))
        units = []
    else:
        units = [("reference", r) for r in chosen]
    units += [(lab, r) for lab in runs for r in chosen]
    mine = units[task::n_tasks]

    rows, finals = [], {}
    for label, r in mine:
        d = first[r] if label == "reference" else runs[label][r]
        start = (d.pos[:, POS_FIT_IDX] if label == "reference" else d.pos_gen[-1]).double().numpy()
        res, final = optimise(d.atom_type.numpy(), start, ase_calc)
        res.update(label=label, rxn_index=r, moved=kabsch_rmsd(start, final))
        rows.append(res)
        finals[f"{label}|{r}"] = final
        print(json.dumps(res), flush=True)
    with open(out_dir / f"task{task:03d}.jsonl", "w") as fh:
        fh.writelines(json.dumps(x) + "\n" for x in rows)
    np.savez(out_dir / f"task{task:03d}_final.npz", **finals)


def summarize(out_dir):
    import pandas as pd

    out_dir = Path(out_dir)
    df = pd.DataFrame(read_jsonl(sorted(out_dir.glob("task*.jsonl"))))
    finals = {}
    for f in sorted(out_dir.glob("task*_final.npz")):
        finals.update(dict(np.load(f)))
    ref_file = out_dir / "reference_from.txt"
    if ref_file.exists():
        # reference optimisations of an earlier check with the same settings
        ref_dir = Path(ref_file.read_text().strip())
        ref = pd.DataFrame(read_jsonl(sorted(ref_dir.glob("task*.jsonl"))))
        df = pd.concat([ref[ref.label == "reference"], df], ignore_index=True)
        for f in sorted(ref_dir.glob("task*_final.npz")):
            finals.update({k: v for k, v in np.load(f).items() if k.startswith("reference|")})

    ref_ok = df[(df.label == "reference") & df.converged & (df.n_imag == 1)].set_index("rxn_index")
    same, same_mirror = [], []
    for _, x in df.iterrows():
        key = x.rxn_index
        ok = x.converged and x.n_imag == 1 and x.label != "reference" and key in ref_ok.index
        if ok:
            a, b = finals[f"{x.label}|{key}"], finals[f"reference|{key}"]
            same.append(kabsch_rmsd(a, b))
            same_mirror.append(kabsch_rmsd(a, b, allow_reflection=True))
        else:
            same.append(np.nan)
            same_mirror.append(np.nan)
    df["rmsd_to_ref_saddle"] = same
    df["rmsd_to_ref_saddle_mirror"] = same_mirror
    df.to_csv(out_dir / "sella_results.csv", index=False)

    def stats(g):
        conv = g.converged & (g.n_imag == 1)
        return pd.Series(
            {
                "n": len(g),
                "converged_%": 100 * g.converged.mean(),
                "first_order_saddle_%": 100 * conv.mean(),
                "steps_median": g.loc[g.converged, "steps"].median(),
                "moved_median_A": g.loc[g.converged, "moved"].median(),
                "start_0_imag_%": 100 * (g.n_imag_start == 0).mean(),
                "start_1_imag_%": 100 * (g.n_imag_start == 1).mean(),
                "start_2plus_imag_%": 100 * (g.n_imag_start >= 2).mean(),
                "same_saddle_%": 100 * (g.rmsd_to_ref_saddle < 0.1).sum() / len(g),
                "same_saddle_up_to_mirror_%": 100
                * (g.rmsd_to_ref_saddle_mirror < 0.1).sum()
                / len(g),
                "errors": int((g.error != "").sum()),
            }
        )

    order = ["reference"] + [lab for lab in df.label.unique() if lab != "reference"]
    table = df.groupby("label").apply(stats).loc[order]
    pd.set_option("display.width", 200)
    print(table.round(1).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5:])
    else:
        summarize(sys.argv[2])
