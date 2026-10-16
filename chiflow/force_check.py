"""Forces on the generated TSs of a run, on g-xTB or AIMNet2-rxn.

The force is the one the steering potentials use, without the transform: the
RMS over all force components of the g-xTB forces or of the AIMNet2-rxn
ensemble mean. Also the largest atomic force and, for AIMNet2-rxn, the RMS
spread of the ensemble members. One single point per reaction, no optimisation.

    PYTHONPATH=. .venv/bin/python force_check.py <out.tsv> calculator=<name> [reference=1] <pkl>

reference=1 evaluates the RDB7 reference TSs of the pickle instead of the
generated ones. A failed g-xTB calculation gives NaN. Threads: SLURM_CPUS_PER_TASK.
"""

import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import torch

from calculators import ENSEMBLE_MEMBERS

MOLS_PER_CALL = 200
POS_FIT_IDX = 1  # configs/train.yaml: the TS geometry in data.pos


def gxtb_forces(geoms, n_workers):
    from flow_matching.gxtb_potential import gxtb_binary, gxtb_energy_forces

    binary = gxtb_binary()

    def one(g):
        z, pos = g
        try:
            f = gxtb_energy_forces(z, pos, binary)[1]
        except Exception:  # a failed calculation is a missing value, not a crash
            return np.nan, np.nan, np.nan
        return np.sqrt((f**2).mean()), np.linalg.norm(f, axis=1).max(), np.nan

    with ThreadPoolExecutor(n_workers) as pool:
        return list(pool.map(one, geoms))


def aimnet_forces(geoms):
    from flow_matching.aimnet_potential import AIMNetForcePotential

    potential = AIMNetForcePotential(ensemble_members=list(ENSEMBLE_MEMBERS))
    out = []
    for a in range(0, len(geoms), MOLS_PER_CALL):
        chunk = geoms[a : a + MOLS_PER_CALL]
        sizes = torch.tensor([len(z) for z, _ in chunk])
        batch = SimpleNamespace(
            atom_type=torch.cat([torch.as_tensor(z) for z, _ in chunk]),
            batch=torch.repeat_interleave(torch.arange(len(chunk)), sizes),
            num_graphs=len(chunk),
        )
        pos = torch.cat([torch.as_tensor(p) for _, p in chunk]).float()
        f_M_N_3 = potential.member_forces(pos, batch, "cpu").detach().double()
        mean, std = f_M_N_3.mean(0), f_M_N_3.std(0)
        for f, s in zip(mean.split(sizes.tolist()), std.split(sizes.tolist()), strict=True):
            out.append(
                (
                    float(f.pow(2).mean().sqrt()),
                    float(f.norm(dim=1).max()),
                    float(s.pow(2).mean().sqrt()),
                )
            )
    return out


def main(out_tsv, inputs):
    n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    torch.set_num_threads(n_threads)
    opts = dict(item.split("=", 1) for item in inputs[:-1])
    calculator, reference = opts.get("calculator", "aimnet"), opts.get("reference") == "1"
    samples = pickle.load(open(inputs[-1], "rb"))
    geoms = [
        (
            d.atom_type.numpy(),
            (d.pos[:, POS_FIT_IDX] if reference else d.pos_gen[-1]).double().numpy(),
        )
        for d in samples
    ]
    if calculator == "gxtb":
        rows = gxtb_forces(geoms, n_threads)
    elif calculator == "aimnet":
        rows = aimnet_forces(geoms)
    else:
        raise ValueError(f"calculator must be aimnet or gxtb, not {calculator}")
    with open(out_tsv, "w") as fh:
        fh.write("rxn_index\tn_atoms\trms_force\tmax_force\tforce_spread\n")
        for d, (z, _), (rms, fmax, spread) in zip(samples, geoms, rows, strict=True):
            fh.write(f"{int(d.rxn_index)}\t{len(z)}\t{rms:.5f}\t{fmax:.5f}\t{spread:.5f}\n")
    print(f"{out_tsv}: {len(rows)} reactions, {sum(np.isnan(r[0]) for r in rows)} failed")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
