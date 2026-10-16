"""AIMNet2-rxn RMS force of the generated TS of every reaction in a run.

The force is the ensemble mean of the four members (the default of
AIMNetForcePotential); aimnet_force.tsv also holds the RMS spread of the
members' forces, the ensemble's uncertainty.

Reads prediction pickles (samples_all.pkl), evaluates the final generated
geometry of each reaction and writes aimnet_force.tsv next to each pickle.
With several pickles it also prints a side-by-side summary, the first pickle
being the baseline.

Runs saved with model.save_final_particles=true also get particles.tsv: per
reaction, the force and RMSD of each of the S particles before the last
resampling. It answers whether the aggregation (the medoid) loses what the
steering found, and whether a lower force goes with a lower RMSD within a
reaction. RMSD here is a Kabsch RMSD in the saved atom order (no symmetry search), so it is
not the rmse column of test_results.csv.

    PYTHONPATH=. .venv/bin/python eval_aimnet_force.py <pkl> [<pkl> ...]
"""

import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.stats import spearmanr

from flow_matching.aimnet_potential import AIMNetForcePotential

MOLS_PER_CALL = 600
POS_FIT_IDX = 1  # configs/train.yaml: the TS geometry in data.pos


def rms_forces(pos_list, types_list, potential, device, with_spread=False):
    out = np.empty(len(pos_list))
    spread = np.empty(len(pos_list))
    for a in range(0, len(pos_list), MOLS_PER_CALL):
        pos_chunk, types_chunk = pos_list[a : a + MOLS_PER_CALL], types_list[a : a + MOLS_PER_CALL]
        pos = torch.cat(pos_chunk).float().to(device)
        batch = SimpleNamespace(
            atom_type=torch.cat(types_chunk).to(device),
            batch=torch.repeat_interleave(
                torch.arange(len(types_chunk)), torch.tensor([t.numel() for t in types_chunk])
            ).to(device),
            num_graphs=len(types_chunk),
        )
        rms, sd = potential.force_stats(pos, batch, device)
        out[a : a + len(types_chunk)] = rms.cpu().numpy()
        spread[a : a + len(types_chunk)] = sd.cpu().numpy()
    return (out, spread) if with_spread else out


def kabsch_rmsd(pred_S_N_3, gt_N_3):
    """RMSD of each of S geometries to gt after centring and the best proper rotation."""
    P = pred_S_N_3 - pred_S_N_3.mean(dim=1, keepdim=True)
    Q = (gt_N_3 - gt_N_3.mean(dim=0)).expand_as(P)
    U, _, Vt = torch.linalg.svd(P.transpose(1, 2) @ Q)
    d = torch.sign(torch.linalg.det(U @ Vt))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], dim=-1))
    P_rot = P @ (U @ D @ Vt)
    return ((P_rot - Q) ** 2).sum(-1).mean(-1).sqrt()


def particle_table(samples, potential, device):
    """One row per reaction: force and RMSD of the medoid and of the S pre-resampling particles."""
    S = samples[0].particles_pre.shape[0]
    # particles are atom-permuted only among equivalent atoms, so atom_type still fits
    pos = [p for d in samples for p in d.particles_pre]
    types = [d.atom_type for d in samples for _ in range(S)]
    f_R_S = rms_forces(pos, types, potential, device).reshape(len(samples), S)
    rows = []
    for d, f_S in zip(samples, f_R_S, strict=True):
        gt = d.pos[:, POS_FIT_IDX].float()
        rmsd_S = kabsch_rmsd(d.particles_pre.float(), gt).numpy()
        rmsd_med = float(kabsch_rmsd(d.pos_gen[-1].float()[None], gt)[0])
        rho = spearmanr(f_S, rmsd_S)[0] if np.ptp(f_S) > 0 and np.ptp(rmsd_S) > 0 else np.nan
        rows.append(
            (
                int(d.rxn_index),
                f_S.min(),
                np.median(f_S),
                rmsd_med,
                rmsd_S[f_S.argmin()],
                rmsd_S.min(),
                np.median(rmsd_S),
                rho,
            )
        )
    return rows


PARTICLE_COLS = [
    "rxn_index",
    "force_min",
    "force_median",
    "rmsd_medoid",
    "rmsd_at_force_min",
    "rmsd_min",
    "rmsd_median",
    "spearman_force_rmsd",
]


def main(paths):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    potential = AIMNetForcePotential()
    results = {}
    for path in map(Path, paths):
        samples = pickle.load(open(path, "rb"))
        f, sd = rms_forces(
            [d.pos_gen[-1] for d in samples],
            [d.atom_type for d in samples],
            potential,
            device,
            with_spread=True,
        )
        rxn = [int(d.rxn_index) for d in samples]
        with open(path.parent / "aimnet_force.tsv", "w") as fh:
            fh.write("rxn_index\trms_force\tforce_spread\n")
            fh.writelines(f"{r}\t{v:.5f}\t{u:.5f}\n" for r, v, u in zip(rxn, f, sd, strict=True))
        results[path] = (rxn, f)

        if hasattr(samples[0], "particles_pre"):
            rows = particle_table(samples, potential, device)
            with open(path.parent / "particles.tsv", "w") as fh:
                fh.write("\t".join(PARTICLE_COLS) + "\n")
                fh.writelines(
                    "\t".join([str(r[0])] + [f"{v:.5f}" for v in r[1:]]) + "\n" for r in rows
                )
            a = np.array([r[1:] for r in rows])
            print(
                f"{path.parent.name}: particles  force min/median {np.median(a[:, 0]):.3f}/"
                f"{np.median(a[:, 1]):.3f}  rmsd medoid {a[:, 2].mean():.4f}  "
                f"at force-min {a[:, 3].mean():.4f}  best {a[:, 4].mean():.4f}  "
                f"particle median {a[:, 5].mean():.4f}  "
                f"within-rxn rho(force, rmsd) {np.nanmedian(a[:, 6]):.3f}"
            )

    base_rxn, base_f = next(iter(results.values()))
    print(
        f"{'run':60s} {'median':>8s} {'mean':>8s} {'p10':>7s} {'p90':>7s} "
        f"{'ratio to base':>13s} {'lower than base':>15s}"
    )
    for path, (rxn, f) in results.items():
        assert rxn == base_rxn, f"{path}: reactions differ from the baseline"
        q10, q50, q90 = np.percentile(f, [10, 50, 90])
        ratio = np.median(f / base_f)
        lower = np.mean(f < base_f) * 100
        print(
            f"{str(path.parent)[-60:]:60s} {q50:8.3f} {f.mean():8.3f} {q10:7.3f} "
            f"{q90:7.3f} {ratio:13.3f} {lower:14.1f}%"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
