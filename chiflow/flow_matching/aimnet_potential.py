"""AIMNet2-rxn force potential for FK steering.

A transition state is a stationary point, so the potential rewards small
forces: U = weight * transform(RMS force) per molecule, with the forces taken
from the AIMNet2-rxn machine-learned interatomic potential at the particle's
coordinates. No reference geometry enters.

By default the forces are the mean over the four ensemble members (the
gradient of the ensemble-mean energy), so the steering does not exploit the
errors of one member. The spread of the members' forces is an uncertainty
estimate; uncertainty_weight adds it to U, which keeps the particles where the
members agree.

The potential has the same call signature as chiral_potential, so it slots into
FlowModule.fk_potential_fns. It is selected from the config, e.g.

    +fk_potentials=aimnet_force

and needs the optional dependency group:  uv sync --group aimnet
"""

from collections.abc import Sequence

import torch
from torch import Tensor
from torch_geometric.data import Batch

AIMNET_MODEL = "isayevlab/aimnet2-rxn"
# Pinned Hugging Face revision of the model repository.
AIMNET_REVISION = "13e9de20a1e878a18a371e94dfb2176629281cb6"

_CALCULATORS = {}


def aimnet_calculators(model, revision, members, device):
    """AIMNet2 calculators, one per ensemble member, loaded once per process."""
    import warp
    from aimnet.calculators import AIMNet2Calculator

    # warp's NVRTC build of the neighbour-list kernel fails for sm_86
    # (A40) but not sm_80; sm_80 PTX runs on all newer GPUs via the driver JIT.
    warp.config.ptx_target_arch = 80
    out = []
    for m in members:
        key = (model, revision, int(m), str(device))
        if key not in _CALCULATORS:
            _CALCULATORS[key] = AIMNet2Calculator(
                model,
                ensemble_member=int(m),
                revision=revision,
                device=str(device),
                compile_model=False,
            )
        out.append(_CALCULATORS[key])
    return out


TRANSFORMS = {
    # log keeps the weights sane when early one-shot estimates have huge forces
    "log1p": torch.log1p,
    "rms": lambda rms: rms,
}


class AIMNetForcePotential:
    """U = weight * transform(RMS force + uncertainty_weight * RMS force spread).

    Per molecule, in eV/A before transform. The force is the mean over the
    ensemble members; the spread is the standard deviation of the members'
    forces, per atom and component, as an RMS over the molecule.

    Args:
        transform: "log1p" (log(1 + x)) or "rms" (x).
        weight: factor on this potential inside the potential sum.
        model: Hugging Face model repository.
        revision: pinned revision of that repository.
        ensemble_members: the ensemble members (0-3) to average; an int for one.
        uncertainty_weight: factor on the RMS force spread; needs two or more members.
    """

    def __init__(
        self,
        transform: str = "log1p",
        weight: float = 1.0,
        model: str = AIMNET_MODEL,
        revision: str = AIMNET_REVISION,
        ensemble_members: int | Sequence[int] = (0, 1, 2, 3),
        uncertainty_weight: float = 0.0,
    ):
        if transform not in TRANSFORMS:
            raise ValueError(f"transform must be one of {sorted(TRANSFORMS)}")
        if isinstance(ensemble_members, int):
            ensemble_members = [ensemble_members]
        ensemble_members = [int(m) for m in ensemble_members]
        if not ensemble_members:
            raise ValueError("ensemble_members must not be empty")
        if uncertainty_weight and len(ensemble_members) < 2:
            raise ValueError("uncertainty_weight needs two or more ensemble members")
        self.transform = transform
        self.weight = weight
        self.model = model
        self.revision = revision
        self.ensemble_members = ensemble_members
        self.uncertainty_weight = uncertainty_weight
        self._calcs = None

    def _calculators(self, device):
        # Loaded on first use, so importing this module needs no GPU or download.
        if self._calcs is None:
            self._calcs = aimnet_calculators(
                self.model, self.revision, self.ensemble_members, device
            )
        return self._calcs

    def member_forces(self, pos_N_3: Tensor, batch: Batch, device) -> Tensor:
        """Forces of each ensemble member, shape (M, N, 3), eV/A."""
        forces_M_N_3 = []
        for calc in self._calculators(device):
            # The test loop runs in inference mode; the forces are autograd
            # derivatives inside the calculator, so leave it for this call.
            with torch.inference_mode(False), torch.enable_grad():
                out = calc(
                    {
                        "coord": pos_N_3.detach().clone(),
                        "numbers": batch.atom_type,
                        "mol_idx": batch.batch,
                        "charge": torch.zeros(batch.num_graphs, device=device),
                    },
                    forces=True,
                )
                forces_M_N_3.append(out["forces"].detach().to(device=device, dtype=torch.float32))
        return torch.stack(forces_M_N_3)

    @staticmethod
    def _rms_per_molecule(x_N_3: Tensor, batch: Batch, device) -> Tensor:
        sq_Nm = torch.zeros(batch.num_graphs, device=device).index_add_(
            0, batch.batch, (x_N_3**2).sum(dim=-1)
        )
        n_atoms_Nm = torch.bincount(batch.batch, minlength=batch.num_graphs).to(device)
        return torch.sqrt(sq_Nm / (3 * n_atoms_Nm))

    def force_stats(self, pos_N_3: Tensor, batch: Batch, device) -> tuple[Tensor, Tensor]:
        """RMS of the ensemble-mean force and RMS force spread per molecule, each (num_graphs,).

        The spread is zero for one member.
        """
        forces_M_N_3 = self.member_forces(pos_N_3, batch, device)
        rms_Nm = self._rms_per_molecule(forces_M_N_3.mean(dim=0), batch, device)
        if len(forces_M_N_3) == 1:
            return rms_Nm, torch.zeros_like(rms_Nm)
        spread_Nm = self._rms_per_molecule(forces_M_N_3.std(dim=0), batch, device)
        return rms_Nm, spread_Nm

    def rms_force(self, pos_N_3: Tensor, batch: Batch, device) -> Tensor:
        """RMS of the ensemble-mean force per molecule, shape (num_graphs,)."""
        return self.force_stats(pos_N_3, batch, device)[0]

    def __call__(self, pos_N_3: Tensor, batch: Batch, device) -> Tensor:
        rms_Nm, spread_Nm = self.force_stats(pos_N_3, batch, device)
        return self.weight * TRANSFORMS[self.transform](
            rms_Nm + self.uncertainty_weight * spread_Nm
        )
