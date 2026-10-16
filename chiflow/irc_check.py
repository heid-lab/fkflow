"""IRC from the Sella saddles of sella_check.py: does a saddle connect the reactant and product?

This is the criterion of a DFT Opt-TS + IRC check, and it is weaker than the
"same saddle as the reference" of sella_check.py: another conformer of the TS of
the same reaction also passes.

From every first-order saddle of a run (converged, one imaginary mode), a
mass-weighted IRC (Sella) goes forward and reverse on the surface of the Sella
check (calculator=<name>, default "aimnet"; checked against the Sella settings).
Each end is then minimised. The bond graph of each minimum
(distance below BOND_SCALE times the sum of the covalent radii) is compared with
the graphs of the RDB7 reactant and product geometries, built with the same rule.
When an end has the reactant or product graph, its stereo (R/S and E/Z from 3D,
RDKit) is compared with that of the RDB7 geometry.

    # one array task: its share of the (run, reaction) saddles
    PYTHONPATH=. .venv/bin/python irc_check.py run <out_dir> <task> <n_tasks> \
        data=<samples pkl> [calculator=<name>] [n_reactions=<k>] <label>=<sella check dir> ...
    # after all tasks: success by the IRC criterion and where the misses go
    PYTHONPATH=. .venv/bin/python irc_check.py summarize <out_dir> [stereo_counts.csv]

The reference saddles are the label "reference" of the check that optimised them.
n_reactions=<k> takes the same random subset of k reactions (seed 0) for every
label, for a pilot; the summary then covers only that subset.
paired=1 takes only the saddles that the Sella task with the same index found
(task<k>.jsonl), so a worker can run the IRC right after its own Sella task;
both must use the same number of tasks.
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
from sella_check import read_jsonl, read_settings

IRC_DX = (0.1, 0.05)  # amu^1/2 A per IRC step; the smaller one retries a failed direction
IRC_FMAX = 0.1  # eV/A; the ends are minimised afterwards
IRC_STEPS = 200
MIN_FMAX = 0.05
MIN_STEPS = 300
BOND_SCALE = 1.25  # bond if distance < BOND_SCALE * (r_cov,i + r_cov,j); 1.2 misses H2


def settings(calculator="aimnet"):
    out = dict(
        calculator=calculator,
        irc_dx=list(IRC_DX),
        irc_fmax=IRC_FMAX,
        irc_steps=IRC_STEPS,
        min_fmax=MIN_FMAX,
        min_steps=MIN_STEPS,
        bond_scale=BOND_SCALE,
    )
    if calculator == "aimnet":
        out["ensemble_members"] = list(ENSEMBLE_MEMBERS)
    return out


def bond_graph(numbers, positions):
    from ase.data import covalent_radii

    r = covalent_radii[numbers]
    d = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
    i, j = np.where(np.triu(d < BOND_SCALE * (r[:, None] + r[None]), k=1))
    return frozenset(zip(i.tolist(), j.tolist(), strict=True))


def rdmol_graph(mol):
    """Bond set of an RDB7 mapped molecule in data atom order (atom map number - 1)."""
    idx = [a.GetAtomMapNum() - 1 for a in mol.GetAtoms()]
    return frozenset(
        tuple(sorted((idx[b.GetBeginAtomIdx()], idx[b.GetEndAtomIdx()]))) for b in mol.GetBonds()
    )


def stereo_smiles(mol, positions):
    """Canonical isomeric SMILES of a mapped molecule, stereo assigned from positions."""
    from rdkit import Chem

    m = Chem.Mol(mol)
    m.RemoveAllConformers()
    conf = Chem.Conformer(m.GetNumAtoms())
    for a in m.GetAtoms():
        conf.SetAtomPosition(a.GetIdx(), positions[a.GetAtomMapNum() - 1].tolist())
    m.AddConformer(conf, assignId=True)
    Chem.AssignStereochemistryFrom3D(m)
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(Chem.RemoveHs(m))


def graph_str(g):
    return " ".join(f"{i}-{j}" for i, j in sorted(g))


def minimise(atoms, ase_calc):
    from sella import Sella

    opt = Sella(
        atoms,
        order=0,
        internal=True,
        hessian_function=ase_calc.get_hessian,
        logfile=None,
        trajectory=None,
    )
    return bool(opt.run(fmax=MIN_FMAX, steps=MIN_STEPS)), int(opt.nsteps)


def irc_ends(numbers, saddle, ase_calc):
    """Forward and reverse IRC from a saddle, each end minimised."""
    from ase import Atoms
    from sella import IRC

    atoms = Atoms(numbers=numbers, positions=saddle)
    atoms.calc = ase_calc
    e_ts = float(atoms.get_potential_energy())
    out, ends = {"e_ts": e_ts}, {}
    for direction in ("forward", "reverse"):
        k = direction[0]
        for dx in IRC_DX:
            atoms.positions = saddle
            try:
                irc = IRC(
                    atoms,
                    dx=dx,
                    hessian_function=ase_calc.get_hessian,
                    logfile=None,
                    trajectory=None,
                )
                out[f"{k}_irc_converged"] = bool(
                    irc.run(fmax=IRC_FMAX, steps=IRC_STEPS, direction=direction)
                )
                out[f"{k}_irc_steps"], out[f"{k}_irc_dx"] = int(irc.nsteps), dx
                end = Atoms(numbers=numbers, positions=atoms.get_positions())
                end.calc = ase_calc
                out[f"{k}_min_converged"], out[f"{k}_min_steps"] = minimise(end, ase_calc)
                out[f"{k}_fmax"] = float(np.linalg.norm(end.get_forces(), axis=1).max())
                out[f"{k}_barrier"] = e_ts - float(end.get_potential_energy())
                ends[k] = end.get_positions()
                out[f"{k}_error"] = ""
                break
            except Exception as e:  # a failed IRC is a result, not a crash
                out[f"{k}_error"] = f"{type(e).__name__}: {e}"[:200]
    return out, ends


def classify(numbers, d, ends):
    """Compare the IRC ends with the RDB7 reactant and product (graphs, then stereo)."""
    pos_r, pos_p = d.pos[:, 0].double().numpy(), d.pos[:, 2].double().numpy()
    g_r, g_p = bond_graph(numbers, pos_r), bond_graph(numbers, pos_p)
    mols = {"R": (d.rdmol[0], pos_r), "P": (d.rdmol[1], pos_p)}
    out = dict(
        ref_graphs_match_rdmol=bool(
            g_r == rdmol_graph(d.rdmol[0]) and g_p == rdmol_graph(d.rdmol[1])
        )
    )
    for k, pos in ends.items():
        g = bond_graph(numbers, pos)
        side = "R" if g == g_r else "P" if g == g_p else "X"
        out[f"{k}_side"] = side
        out[f"{k}_graph"] = graph_str(g)
        if side != "X":
            mol, pos_ref = mols[side]
            try:
                out[f"{k}_stereo_ok"] = stereo_smiles(mol, pos) == stereo_smiles(mol, pos_ref)
            except Exception:  # RDKit can fail on unusual valences
                out[f"{k}_stereo_ok"] = None
    return out


def run(out_dir, task, n_tasks, inputs):
    import pandas as pd

    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 2)))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data, checks, n_reactions, calculator, paired = None, {}, None, "aimnet", False
    for item in inputs:
        label, path = item.split("=", 1)
        if label == "paired":
            paired = path == "1"
        elif label == "calculator":
            calculator = path
        elif label == "n_reactions":
            n_reactions = int(path)
        elif label == "data":
            data_path = path
            data = {int(d.rxn_index): d for d in pickle.load(open(path, "rb"))}
        else:
            checks[label] = Path(path)
    for check in set(checks.values()):
        if read_settings(check)["calculator"] != calculator:
            raise ValueError(
                f"{check} ran on {read_settings(check)['calculator']}, not {calculator}"
            )
    (out_dir / "settings.json").write_text(json.dumps(settings(calculator)))
    ase_calc = make_calculator(calculator)
    (out_dir / "inputs.json").write_text(
        json.dumps({"data": data_path, **{k: str(v) for k, v in checks.items()}})
    )
    chosen = sorted(data)
    if n_reactions is not None:
        chosen = sorted(
            np.random.default_rng(0).choice(chosen, n_reactions, replace=False).tolist()
        )
        (out_dir / "subset.json").write_text(json.dumps(chosen))

    units = []
    for label, check in checks.items():
        files = [check / f"task{task:03d}.jsonl"] if paired else sorted(check.glob("task*.jsonl"))
        rows = pd.DataFrame(read_jsonl(f for f in files if f.exists()))
        if rows.empty:
            continue
        rows = rows[
            (rows.label == label)
            & rows.converged
            & (rows.n_imag == 1)
            & rows.rxn_index.isin(chosen)
        ]
        units += [(label, check, int(r)) for r in sorted(rows.rxn_index)]
    mine = units if paired else units[task::n_tasks]
    saddles = {}
    for check in {c for _, c, _ in mine}:
        npz = (
            [check / f"task{task:03d}_final.npz"]
            if paired
            else sorted(check.glob("task*_final.npz"))
        )
        for f in npz:
            saddles.update({f"{check}|{k}": v for k, v in np.load(f).items()})

    rows, finals = [], {}
    for label, check, r in mine:
        d = data[r]
        numbers = d.atom_type.numpy()
        t0 = time.time()
        res, ends = irc_ends(numbers, saddles[f"{check}|{label}|{r}"], ase_calc)
        res.update(classify(numbers, d, ends), label=label, rxn_index=r, seconds=time.time() - t0)
        rows.append(res)
        finals.update({f"{label}|{r}|{k}": v for k, v in ends.items()})
        print(json.dumps(res), flush=True)
    with open(out_dir / f"task{task:03d}.jsonl", "w") as fh:
        fh.writelines(json.dumps(x) + "\n" for x in rows)
    np.savez(out_dir / f"task{task:03d}_ends.npz", **finals)


def outcome(x):
    """One category per saddle, from its two IRC ends."""
    import pandas as pd

    if pd.isna(x.get("f_side")) or pd.isna(x.get("r_side")):
        return "irc_failed"
    sides = {x.f_side, x.r_side}
    if sides == {"R", "P"}:
        return (
            "connects"
            if x.f_stereo_ok is True and x.r_stereo_ok is True
            else "connects_wrong_stereo"
        )
    if sides in ({"R"}, {"P"}):
        return "same_side"  # a conformer or stereo TS on one side
    if len(sides) == 2:
        return "one_side"  # the right reactant or product, but another reaction
    return "unrelated"


def reaction_classes(d):
    """Simple classes from the RDB7 graphs: bonds changed, H transfer, ring change."""
    n = len(d.atom_type)
    g_r, g_p = rdmol_graph(d.rdmol[0]), rdmol_graph(d.rdmol[1])
    changed = g_r ^ g_p
    z = d.atom_type.numpy()

    def rings(g):  # cycle rank: bonds - atoms + connected components
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i, j in g:
            parent[find(i)] = find(j)
        return len(g) - n + len({find(i) for i in range(n)})

    ring_change = rings(g_p) - rings(g_r)
    return dict(
        n_heavy=int((z != 1).sum()),
        bonds_broken=len(g_r - g_p),
        bonds_formed=len(g_p - g_r),
        h_transfer=any(z[i] == 1 or z[j] == 1 for i, j in changed),
        ring=("closing" if ring_change > 0 else "opening" if ring_change < 0 else "none"),
    )


def summarize(out_dir, stereo_csv=None):
    import pandas as pd

    out_dir = Path(out_dir)
    df = pd.DataFrame(read_jsonl(sorted(out_dir.glob("task*.jsonl"))))
    inputs = json.loads((out_dir / "inputs.json").read_text())
    data = pickle.load(open(inputs.pop("data"), "rb"))
    checks = {k: Path(v) for k, v in inputs.items()}
    df["outcome"] = df.apply(outcome, axis=1)

    # every reaction of every run: the Sella result, the chirality-aware correct saddle,
    # the IRC outcome
    sella = []
    for label, check in checks.items():
        s = pd.read_csv(check / "sella_results.csv")
        sella.append(s[s.label == label].drop_duplicates("rxn_index"))
    all_df = pd.concat(sella, ignore_index=True)
    if (out_dir / "subset.json").exists():
        all_df = all_df[all_df.rxn_index.isin(json.loads((out_dir / "subset.json").read_text()))]
    all_df = all_df.merge(df.drop(columns=["seconds"]), on=["label", "rxn_index"], how="left")
    classes = pd.DataFrame([dict(rxn_index=int(d.rxn_index), **reaction_classes(d)) for d in data])
    all_df = all_df.merge(classes, on="rxn_index", how="left")
    if stereo_csv is not None:
        st = pd.read_csv(stereo_csv).set_index("rxn_index")
        all_df["n_stereo"] = all_df.rxn_index.map(st.n_stereo)
        all_df["gt_fail"] = all_df.rxn_index.map(st.gt_fail).astype(bool)
        chiral = all_df.n_stereo > 0
        # a mirror image counts only for reactions without stereocentres
        all_df["correct"] = (chiral & (all_df.rmsd_to_ref_saddle < 0.1)) | (
            ~chiral & (all_df.rmsd_to_ref_saddle_mirror < 0.1)
        )
    else:
        all_df["correct"] = all_df.rmsd_to_ref_saddle < 0.1
    first_order = all_df.converged & (all_df.n_imag == 1)
    all_df["category"] = np.where(first_order, all_df.outcome, "no_first_order_saddle")
    all_df.loc[all_df.correct & (all_df.label != "reference"), "category"] = "correct_saddle"
    all_df["irc_success"] = all_df.outcome == "connects"
    all_df.to_csv(out_dir / "irc_results.csv", index=False)

    pd.set_option("display.width", 200)
    order = [lab for lab in checks if lab == "reference"] + [
        lab for lab in checks if lab != "reference"
    ]

    def stats(g):
        return pd.Series(
            {
                "n": len(g),
                "first_order_%": 100 * (g.converged & (g.n_imag == 1)).mean(),
                "correct_saddle_%": 100 * g.correct.mean(),
                "irc_connects_%": 100 * g.irc_success.mean(),
                "connects_any_stereo_%": 100
                * g.outcome.isin(["connects", "connects_wrong_stereo"]).mean(),
                "correct_not_connects_%": 100 * (g.correct & ~g.irc_success).mean(),
            }
        )

    print("success by the IRC criterion (all reactions)")
    print(all_df.groupby("label").apply(stats).loc[order].round(1).to_string())
    print("\nwhere the saddles go (% of all reactions; correct_saddle first, then the IRC outcome)")
    cats = [
        "correct_saddle",
        "connects",
        "connects_wrong_stereo",
        "same_side",
        "one_side",
        "unrelated",
        "irc_failed",
        "no_first_order_saddle",
    ]
    tab = pd.crosstab(all_df.label, all_df.category, normalize="index") * 100
    print(
        tab.reindex(index=order, columns=[c for c in cats if c in tab.columns]).round(1).to_string()
    )

    groups = {
        "stereocentres": all_df.n_stereo.clip(upper=3) if "n_stereo" in all_df else None,
        "h_transfer": all_df.h_transfer,
        "ring": all_df.ring,
        "bonds_changed": (all_df.bonds_broken + all_df.bonds_formed).clip(upper=5),
    }
    for name, key in groups.items():
        if key is None:
            continue
        print(f"\nIRC connects % by {name} (n reactions in brackets)")
        g = all_df.assign(key=key).groupby(["key", "label"]).irc_success.agg(["mean", "size"])
        t = (100 * g["mean"]).round(1).astype(str) + " (" + g["size"].astype(str) + ")"
        print(t.unstack("label").reindex(columns=order).to_string())

    print(
        "\ngraph rule vs RDB7 SMILES graphs: agree on "
        f"{100 * df.drop_duplicates('rxn_index').ref_graphs_match_rdmol.mean():.1f} % of reactions"
    )
    n_err = int((df.f_error.fillna("") != "").sum() + (df.r_error.fillna("") != "").sum())
    print(f"IRC errors: {n_err}; median seconds per saddle {df.seconds.median():.0f}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5:])
    else:
        summarize(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
