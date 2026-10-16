"""g-xTB force potential for FK steering, the semi-empirical counterpart of
AIMNetForcePotential: U = weight * transform(RMS force) per molecule.

Each molecule is one single point with gradient of the xtb 6.7.1 --gxtb binary,
run as a subprocess in a thread pool, one thread per molecule, so a
GPU job with several CPUs evaluates a batch in parallel. A molecule whose
calculation fails gets FAIL_RMS, so FK resampling drops it rather than the run.

The binary is the GXTB_BIN environment variable if set, otherwise `xtb` on the
PATH; it must be the g-xTB build of xtb 6.7.1 (README). It is resolved when
the potential is built, so a missing binary stops the run instead of failing
every particle.

    +fk_potentials=gxtb_force_chirality
"""

import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Batch

HARTREE_EV = 27.211386245988
BOHR_A = 0.529177210903
FAIL_RMS = 10.0  # eV/A, for a molecule whose calculation failed
SYMBOLS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S", 17: "Cl"}

TRANSFORMS = {
    "log1p": torch.log1p,
    "rms": lambda rms: rms,
}


def gxtb_binary() -> str:
    """Path of the g-xTB binary: GXTB_BIN, else `xtb` on the PATH."""
    binary = os.environ.get("GXTB_BIN") or shutil.which("xtb")
    if binary is None or not Path(binary).is_file():
        raise FileNotFoundError(
            f"g-xTB binary not found (GXTB_BIN={os.environ.get('GXTB_BIN')!r}, no xtb on PATH)"
        )
    return binary


def gxtb_energy_forces(numbers, positions, binary=None):
    """Energy (eV) and forces (eV/A) of a neutral singlet from one g-xTB call."""
    with tempfile.TemporaryDirectory() as tmp:
        xyz = [str(len(numbers)), ""] + [
            f"{SYMBOLS[int(z)]} {x:.10f} {y:.10f} {w:.10f}"
            for z, (x, y, w) in zip(numbers, positions, strict=True)
        ]
        (Path(tmp) / "in.xyz").write_text("\n".join(xyz) + "\n")
        env = {**os.environ, "OMP_NUM_THREADS": "1", "OMP_STACKSIZE": "1G"}
        p = subprocess.run(
            [binary or gxtb_binary(), "in.xyz", "--gxtb", "--grad", "--chrg", "0", "--uhf", "0"],
            cwd=tmp,
            env=env,
            capture_output=True,
            text=True,
        )
        grad_file = Path(tmp) / "gradient"
        if p.returncode != 0 or not grad_file.exists():
            raise RuntimeError(f"g-xTB failed: {p.stderr[-300:] or p.stdout[-300:]}")
        lines = grad_file.read_text().splitlines()
    n = len(numbers)
    # Turbomole gradient file: $grad, a cycle line with the energy,
    # n coordinate lines, n gradient lines
    cycle = next(i for i, line in enumerate(lines) if "cycle" in line)
    energy = float(lines[cycle].split("=")[2].split()[0])
    grad = np.array(
        [
            [float(v.replace("D", "E")) for v in line.split()]
            for line in lines[cycle + 1 + n : cycle + 1 + 2 * n]
        ]
    )
    return energy * HARTREE_EV, -grad * HARTREE_EV / BOHR_A


class GXTBForcePotential:
    """U = weight * transform(RMS g-xTB force), per molecule, eV/A before transform.

    Args:
        transform: "log1p" (log(1 + x)) or "rms" (x).
        weight: factor on this potential inside the potential sum.
        n_workers: parallel g-xTB processes; default SLURM_CPUS_PER_TASK or 1.
    """

    def __init__(self, transform: str = "log1p", weight: float = 1.0, n_workers: int | None = None):
        if transform not in TRANSFORMS:
            raise ValueError(f"transform must be one of {sorted(TRANSFORMS)}")
        self.transform = transform
        self.weight = weight
        self.n_workers = n_workers or int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
        self.binary = gxtb_binary()
        self.n_failed = 0

    def rms_force(self, pos_N_3: Tensor, batch: Batch, device) -> Tensor:
        pos = pos_N_3.detach().double().cpu().numpy()
        z = batch.atom_type.cpu().numpy()
        mol = batch.batch.cpu().numpy()

        def one(m):
            idx = mol == m
            try:
                _, forces = gxtb_energy_forces(z[idx], pos[idx], self.binary)
                return float(np.sqrt((forces**2).mean()))
            except Exception:  # a failed calculation is a bad particle, not a crash
                self.n_failed += 1
                return FAIL_RMS

        with ThreadPoolExecutor(self.n_workers) as pool:
            rms = list(pool.map(one, range(batch.num_graphs)))
        return torch.tensor(rms, dtype=torch.float32, device=device)

    def __call__(self, pos_N_3: Tensor, batch: Batch, device) -> Tensor:
        return self.weight * TRANSFORMS[self.transform](self.rms_force(pos_N_3, batch, device))
