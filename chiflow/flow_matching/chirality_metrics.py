"""
Chirality evaluation metrics for flow matching models.

The metric functions return per-reaction binary error indicators:
0.0 means no evaluated chiral center violates the criterion, 1.0 means at
least one evaluated chiral center violates it. Dataset-level means therefore
match the paper's "fraction of reactions with a chirality error" definition.

The neighbours of each center are stored in decreasing CIP order and the R/S tag
is -1 for R and +1 for S. After multiplying the normalized signed volume by the
tag, a correctly oriented center is negative, so a value at or above the
threshold is an error.
"""

import torch
from torch_geometric.data import Data


def _empty_chiral_indices() -> torch.Tensor:
    return torch.empty((0, 4), dtype=torch.long)


def _empty_rs_tags() -> torch.Tensor:
    return torch.empty(0, dtype=torch.float)


def _has_chiral_centers(batch: Data) -> bool:
    """Return whether reactant or product chiral-center annotations are present."""
    r_chi_center_C_4_index = getattr(
        batch, "r_cip_decr_chi_nbrs_C_4_index", _empty_chiral_indices()
    )
    p_chi_center_C_4_index = getattr(
        batch, "p_cip_decr_chi_nbrs_C_4_index", _empty_chiral_indices()
    )
    return r_chi_center_C_4_index.shape[0] > 0 or p_chi_center_C_4_index.shape[0] > 0


def evaluate_chirality_number(pos_N_3, batch: Data):
    """Return 1 if the reaction has annotated chiral centers, otherwise 0."""
    return 1 if _has_chiral_centers(batch) else 0


def _normalized_signed_tetrahedral_volumes(
    pos_N_3: torch.Tensor,
    indices_C_4: torch.Tensor,
    rs_tags_C: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute R/S-signed normalized scalar triple products for chiral centers."""
    device = pos_N_3.device
    indices_C_4 = indices_C_4.to(device).long()
    rs_tags_C = rs_tags_C.to(device).float()

    in_bounds_C = ((indices_C_4 >= 0) & (indices_C_4 < pos_N_3.shape[0])).all(dim=1)
    if not in_bounds_C.any():
        empty_volumes = torch.empty(0, device=device, dtype=pos_N_3.dtype)
        empty_valid = torch.empty(0, device=device, dtype=torch.bool)
        return empty_volumes, empty_valid

    indices_C_4 = indices_C_4[in_bounds_C]
    rs_tags_C = rs_tags_C[in_bounds_C]

    pos_indexed_C_4_3 = pos_N_3[indices_C_4]
    tetrahedral_sides_C_3_3 = pos_indexed_C_4_3[:, 1:, :] - pos_indexed_C_4_3[:, 0, :].unsqueeze(1)
    cross_product_C_3 = torch.cross(
        tetrahedral_sides_C_3_3[:, 0, :],
        tetrahedral_sides_C_3_3[:, 1, :],
        dim=-1,
    )
    scalar_triple_products_C = torch.sum(
        tetrahedral_sides_C_3_3[:, 2, :] * cross_product_C_3,
        dim=1,
    )

    bond_lengths_C_3 = torch.linalg.vector_norm(tetrahedral_sides_C_3_3, dim=2)
    cube_volumes_C = torch.prod(bond_lengths_C_3, dim=1)
    valid_C = cube_volumes_C > 1e-8

    normalized_volumes_C = torch.zeros_like(scalar_triple_products_C)
    normalized_volumes_C[valid_C] = scalar_triple_products_C[valid_C] / cube_volumes_C[valid_C]
    return normalized_volumes_C * rs_tags_C, valid_C


def evaluate_chirality_correctness(
    pos_N_3: torch.Tensor, batch: Data, threshold: float = 0.25
) -> float:
    """
    Evaluate the thresholded chirality error indicator.

    After applying the stored R/S tag, a signed normalized volume at or above
    ``threshold`` is counted as a chirality error. The default threshold follows
    the paper value theta = 0.25, so a flattened center (volume between 0 and
    theta, e.g. near the TS) is tolerated. See
    evaluate_persistent_chirality_correctness for the strict sign test.
    """
    r_chi_center_C_4_index = getattr(
        batch, "r_cip_decr_chi_nbrs_C_4_index", _empty_chiral_indices()
    )
    p_chi_center_C_4_index = getattr(
        batch, "p_cip_decr_chi_nbrs_C_4_index", _empty_chiral_indices()
    )
    r_rs_tag_C = getattr(batch, "r_rs_tag_C", _empty_rs_tags())
    p_rs_tag_C = getattr(batch, "p_rs_tag_C", _empty_rs_tags())

    wrong_predictions = 0
    for indices_C_4, rs_tags_C in (
        (r_chi_center_C_4_index, r_rs_tag_C),
        (p_chi_center_C_4_index, p_rs_tag_C),
    ):
        if indices_C_4.shape[0] == 0:
            continue

        signed_volumes_C, valid_C = _normalized_signed_tetrahedral_volumes(
            pos_N_3, indices_C_4, rs_tags_C
        )
        wrong_predictions += int(((signed_volumes_C >= threshold) & valid_C).sum().item())

    return 0.0 if wrong_predictions == 0 else 1.0


def evaluate_persistent_chirality_correctness(
    pos_N_3: torch.Tensor, batch: Data, threshold: float = 0.0
) -> float:
    """
    Evaluate persistent-center chirality error indicator.

    Only centers that appear in reactants and products with the same center atom,
    neighbor set, and R/S tag are evaluated. Such a center keeps its configuration
    through the reaction, so the default threshold 0.0 counts any wrong sign as an
    error (no tolerance band, unlike the 0.25 of the general metric). Centers whose
    tag differs between reactant and product (inversion) are not evaluated.
    """
    r_chi_center_C_4_index = getattr(
        batch, "r_cip_decr_chi_nbrs_C_4_index", _empty_chiral_indices()
    )
    p_chi_center_C_4_index = getattr(
        batch, "p_cip_decr_chi_nbrs_C_4_index", _empty_chiral_indices()
    )
    r_rs_tag_C = getattr(batch, "r_rs_tag_C", _empty_rs_tags())
    p_rs_tag_C = getattr(batch, "p_rs_tag_C", _empty_rs_tags())

    if r_chi_center_C_4_index.shape[0] == 0 or p_chi_center_C_4_index.shape[0] == 0:
        return 0.0

    # (center atom, neighbor set) occurs at most once among the product centers, so
    # the first match is the only candidate: `break` ends the search for this reactant
    # center whether or not the tags agree (different tags = inversion, not persistent).
    persistent_centers = []
    persistent_tags = []
    for i, r_center in enumerate(r_chi_center_C_4_index):
        r_center_atom = r_center[0].item()
        r_neighbors = set(r_center[1:].tolist())

        for j, p_center in enumerate(p_chi_center_C_4_index):
            p_center_atom = p_center[0].item()
            p_neighbors = set(p_center[1:].tolist())

            if r_center_atom == p_center_atom and r_neighbors == p_neighbors:
                if r_rs_tag_C[i] == p_rs_tag_C[j]:
                    persistent_centers.append(r_center)
                    persistent_tags.append(r_rs_tag_C[i])
                break

    if not persistent_centers:
        return 0.0

    persistent_indices_C_4 = torch.stack(persistent_centers)
    persistent_tags_C = torch.stack(persistent_tags).to(dtype=torch.float)
    signed_volumes_C, valid_C = _normalized_signed_tetrahedral_volumes(
        pos_N_3, persistent_indices_C_4, persistent_tags_C
    )
    wrong_predictions = int(((signed_volumes_C >= threshold) & valid_C).sum().item())
    return 0.0 if wrong_predictions == 0 else 1.0
