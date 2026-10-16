"""ASE calculators for the Sella and IRC checks, each with energy, forces and a Hessian.

    make_calculator("aimnet")  AIMNet2-rxn, mean over ENSEMBLE_MEMBERS (the steering potential)
    make_calculator("gxtb")    g-xTB (flow_matching.gxtb_potential), finite-difference Hessian

All are neutral singlets. get_hessian(atoms) returns (3N, 3N) in eV/A^2.
"""

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

ENSEMBLE_MEMBERS = (0, 1, 2, 3)
CALCULATORS = ("aimnet", "gxtb")
FD_STEP = 0.005  # A, central differences of the forces


def fd_hessian(calc, atoms):
    """Hessian from central differences of the analytic forces, symmetrised."""
    a = atoms.copy()
    a.calc = calc
    x0 = a.get_positions()
    n = 3 * len(a)
    H = np.zeros((n, n))
    for k in range(n):
        for s in (1, -1):
            x = x0.copy()
            x.flat[k] += s * FD_STEP
            a.set_positions(x)
            H[k] -= s * a.get_forces().ravel() / (2 * FD_STEP)
    return (H + H.T) / 2


def ensemble_aimnet(members=ENSEMBLE_MEMBERS):
    """AIMNet2-rxn with energy, forces and Hessian averaged over the ensemble members."""
    from aimnet.calculators.aimnet2ase import AIMNet2ASE

    from flow_matching.aimnet_potential import AIMNetForcePotential

    subs = [
        AIMNet2ASE(c, charge=0, mult=1)
        for c in AIMNetForcePotential(ensemble_members=members)._calculators("cpu")
    ]

    class EnsembleAIMNet(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            properties = properties or ["energy"]
            super().calculate(atoms, properties, system_changes)
            self.results["energy"] = float(
                np.mean([c.get_potential_energy(self.atoms) for c in subs])
            )
            self.results["forces"] = np.mean([c.get_forces(self.atoms) for c in subs], axis=0)

        def get_hessian(self, atoms=None):
            return np.mean(
                [c.get_hessian(atoms if atoms is not None else self.atoms) for c in subs], axis=0
            ).reshape(3 * len(atoms if atoms is not None else self.atoms), -1)

    return EnsembleAIMNet()


class GXTB(Calculator):
    """g-xTB through the xtb binary: one single point with gradient per call."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, **kwargs):
        from flow_matching.gxtb_potential import gxtb_binary

        super().__init__(**kwargs)
        # resolved here, so a missing binary stops the check instead of failing every reaction
        self.binary = gxtb_binary()

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        from flow_matching.gxtb_potential import gxtb_energy_forces

        properties = properties or ["energy"]
        super().calculate(atoms, properties, system_changes)
        energy, forces = gxtb_energy_forces(self.atoms.numbers, self.atoms.positions, self.binary)
        self.results["energy"] = energy
        self.results["forces"] = forces

    def get_hessian(self, atoms=None):
        return fd_hessian(self, atoms if atoms is not None else self.atoms)


def make_calculator(name):
    if name == "aimnet":
        return ensemble_aimnet()
    if name == "gxtb":
        return GXTB()
    raise ValueError(f"calculator must be one of {CALCULATORS}")
