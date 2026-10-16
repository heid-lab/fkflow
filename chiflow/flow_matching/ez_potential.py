"""E/Z potential for FK steering: the double bonds of reactant and product keep
their E/Z configuration in the transition state.

For each double bond C1=C2 with an E/Z tag (+1 Z, -1 E; from preprocessing,
utils/double_bond_utils.py):

    2 substituents on each atom, [H1, L1, H2, L2] (higher, lower CIP rank):
        n1 = (H1 - C1) x (L1 - C1),  n2 = (H2 - C2) x (L2 - C2),
        cos = n1.n2 / (|n1| |n2|)
    1 substituent on one atom, [H, L, S, -1]: the components of H (on the
    2-substituent atom) and S perpendicular to the bond,
        cos = H_perp.S_perp / (|H_perp| |S_perp|)

    loss = relu(1 + tag * cos)  in [0, 2],  summed per molecule.

The sign is the one that vanishes on the RDB7 reactant and product geometries.

At the reference TS, bonds whose bond order changes in the reaction can violate
the constraint (partly formed or broken double bonds are twisted).
persistent_only keeps only the double bonds that are double on both sides.

TSData does not shift these index fields when batching (their names do not
contain "index"), so the offsets come from the batch's slices here.

    +fk_potentials=gxtb_force_chirality_ez
"""

from functools import partial

import torch
from torch import Tensor
from torch_geometric.data import Batch

_EPS = 1e-8


def _bonds(batch: Batch, prefix: str):
    """Tagged double bonds of r_ or p_ with batch-global atom indices, cached per batch."""
    # in the instance __dict__, outside PyG's data store, so it is no data key
    per_batch = batch.__dict__.get("_ez_cache")
    if per_batch is None:
        per_batch = {}
        object.__setattr__(batch, "_ez_cache", per_batch)
    if prefix in per_batch:
        return per_batch[prefix]
    key = f"{prefix}db_c1c2_D_2"
    c1c2 = getattr(batch, key, None)
    out = None
    if c1c2 is not None and c1c2.shape[0] > 0:
        subs = getattr(batch, f"{prefix}db_substituents_D_4").long()
        tags = getattr(batch, f"{prefix}db_ez_tags_D").float()
        case = getattr(batch, f"{prefix}db_sub_case_D").long()
        slices = batch._slice_dict[key]
        counts = (slices[1:] - slices[:-1]).long()
        mol = torch.repeat_interleave(torch.arange(len(counts)), counts)
        offset = batch.ptr[:-1].cpu()[mol]
        c1c2 = c1c2.long().cpu() + offset[:, None]
        subs = subs.cpu()
        subs = torch.where(subs >= 0, subs + offset[:, None], subs)
        out = dict(c1c2=c1c2, subs=subs, tags=tags.cpu(), case=case.cpu(), mol=mol)
    per_batch[prefix] = out
    return out


def _persistent_mask(batch: Batch, prefix: str) -> Tensor:
    """Bonds of prefix that are also double bonds on the other side of the reaction."""
    mine, other = _bonds(batch, prefix), _bonds(batch, "p_" if prefix == "r_" else "r_")
    if mine is None:
        return torch.zeros(0, dtype=torch.bool)
    if other is None:
        return torch.zeros(len(mine["tags"]), dtype=torch.bool)
    pairs = {tuple(sorted(p)) for p in other["c1c2"].tolist()}
    return torch.tensor(
        [tuple(sorted(p)) in pairs for p in mine["c1c2"].tolist()], dtype=torch.bool
    )


def _cosines(pos: Tensor, c1c2: Tensor, subs: Tensor, case: Tensor) -> tuple[Tensor, Tensor]:
    c1, c2 = pos[c1c2[:, 0]], pos[c1c2[:, 1]]
    cos = torch.zeros(len(c1c2), device=pos.device, dtype=pos.dtype)
    valid = torch.zeros(len(c1c2), device=pos.device, dtype=torch.bool)
    two = case == 0
    if two.any():
        s = subs[two]
        n1 = torch.cross(pos[s[:, 0]] - c1[two], pos[s[:, 1]] - c1[two], dim=1)
        n2 = torch.cross(pos[s[:, 2]] - c2[two], pos[s[:, 3]] - c2[two], dim=1)
        norm = n1.norm(dim=1) * n2.norm(dim=1)
        cos[two] = (n1 * n2).sum(1) / norm.clamp_min(_EPS)
        valid[two] = norm > _EPS
    one = case > 0
    if one.any():
        s, first = subs[one], (case[one] == 1)[:, None]
        a2 = torch.where(first, c1[one], c2[one])  # the atom with two substituents
        a1 = torch.where(first, c2[one], c1[one])
        b = a1 - a2
        b = b / b.norm(dim=1, keepdim=True).clamp_min(_EPS)
        vh, vs = pos[s[:, 0]] - a2, pos[s[:, 2]] - a1
        vh = vh - (vh * b).sum(1, keepdim=True) * b
        vs = vs - (vs * b).sum(1, keepdim=True) * b
        norm = vh.norm(dim=1) * vs.norm(dim=1)
        cos[one] = (vh * vs).sum(1) / norm.clamp_min(_EPS)
        valid[one] = norm > _EPS
    return cos, valid


def ez_potential(
    pos_N_3: Tensor, batch: Batch, device, persistent_only: bool = False, weight: float = 1.0
) -> Tensor:
    """Per-molecule E/Z penalty, sum over tagged double bonds of relu(1 + tag * cos)."""
    total = torch.zeros(batch.num_graphs, device=device, dtype=pos_N_3.dtype)
    for prefix in ("r_", "p_"):
        b = _bonds(batch, prefix)
        if b is None:
            continue
        keep = b["tags"] != 0
        if persistent_only:
            keep &= _persistent_mask(batch, prefix)
        if not keep.any():
            continue
        dev = pos_N_3.device
        cos, valid = _cosines(
            pos_N_3, b["c1c2"][keep].to(dev), b["subs"][keep].to(dev), b["case"][keep].to(dev)
        )
        loss = torch.relu(1.0 + b["tags"][keep].to(dev) * cos) * valid
        total = total.index_add(0, b["mol"][keep].to(dev), loss.to(total.dtype))
    return weight * total


def make_ez_potential(persistent_only: bool = False, weight: float = 1.0):
    """For Hydra: a potential with the FlowModule call signature (pos_N_3, batch, device)."""
    return partial(ez_potential, persistent_only=persistent_only, weight=weight)
