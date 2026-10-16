"""
Extract E/Z stereochemistry information for non-aromatic double bonds.

This module detects double bonds with specified stereochemistry and extracts
the substituent ordering based on Cahn-Ingold-Prelog (CIP) priority rules.

Supports all double bond types (C=C, C=N, N=N, C=O, N=O, etc.) and handles both:
- 2-sub case: Both atoms have 2 substituents (cross product of normals)
- 1-sub case: One atom has 2 subs, other has 1 sub (projection method)

For 2-sub case (A1=A2 with 2 substituents each):
- A1 has substituents H1 (high priority), L1 (low priority)
- A2 has substituents H2 (high priority), L2 (low priority)
- Normal vectors: n1 = (H1 - A1) x (L1 - A1), n2 = (H2 - A2) x (L2 - A2)

For 1-sub case (2-sub atom = A1, 1-sub atom = A2):
- A1 has H1 (high priority), L1 (low priority)
- A2 has S (single substituent)
- Project H1 and S into plane perpendicular to bond

Definitions:
-----------
- db: double bond
- ez: E/Z stereochemistry
- subs: substituents
- mn: map number
- mnidx: index in the list of atoms sorted by ascending map number
- sub_case: 0 = 2-2 subs, 1 = first atom has 2 subs, 2 = second atom has 2 subs
"""

import numpy as np
import torch
from rdkit import Chem


def get_double_bond_stereo_data(mol, perm_inv):
    """
    Extract E/Z stereochemistry data for non-aromatic double bonds.

    Supports all double bond types (C=C, C=N, N=N, etc.) and handles both
    2-substituent and 1-substituent cases.

    Double bonds without E/Z stereo are kept with tag 0.

    Parameters:
        mol: RDKit molecule with atom map numbers assigned
        perm_inv: Inverse permutation to map back to canonical ordering
                  (array of indices that inverts the atom ordering)

    Returns:
        db_c1c2_D_2: (D, 2) tensor with indices [A1, A2] for D double bonds
        db_substituents_D_4: (D, 4) tensor with substituent indices
            - For 2-2 case: [H1, L1, H2, L2] (high/low priority on each atom)
            - For 2-1 case: [H1, L1, S, -1] (H1/L1 from 2-sub atom, S from 1-sub)
            - For 1-2 case: [H2, L2, S, -1] (H2/L2 from 2-sub atom, S from 1-sub)
        db_ez_tags_D: (D,) tensor with +1 for Z (cis), -1 for E (trans), 0 for non-stereo
        db_sub_case_D: (D,) tensor with case indicator:
            - 0: Both atoms have 2 substituents
            - 1: First atom (A1) has 2 subs, second (A2) has 1 sub
            - 2: First atom (A1) has 1 sub, second (A2) has 2 subs
    """
    Chem.AssignStereochemistry(mol, cleanIt=False, force=True, flagPossibleStereoCenters=True)

    mn_to_mnidx = {
        atom.GetAtomMapNum(): i for i, atom in enumerate(np.array(mol.GetAtoms())[perm_inv])
    }

    db_c1c2_list = []
    db_substituents_list = []
    db_ez_tags_list = []
    db_sub_case_list = []

    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue

        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()

        if bond.GetIsAromatic():
            continue

        a1_neighbors = [n for n in atom1.GetNeighbors() if n.GetIdx() != atom2.GetIdx()]
        a2_neighbors = [n for n in atom2.GetNeighbors() if n.GetIdx() != atom1.GetIdx()]

        n_subs = (len(a1_neighbors), len(a2_neighbors))

        if n_subs == (2, 2):
            sub_case = 0
        elif n_subs == (2, 1):
            sub_case = 1
        elif n_subs == (1, 2):
            sub_case = 2
        else:
            continue

        stereo = bond.GetStereo()
        has_stereo = stereo in [Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ]

        all_atoms = [atom1, atom2] + a1_neighbors + a2_neighbors

        if not all(atom.GetAtomMapNum() for atom in all_atoms):
            raise ValueError("All atoms must have AtomMapNum property set.")

        try:
            if sub_case == 0:
                a1_ranks = [int(n.GetProp("_CIPRank")) for n in a1_neighbors]
                a2_ranks = [int(n.GetProp("_CIPRank")) for n in a2_neighbors]
                a1_sorted = sorted(zip(a1_ranks, a1_neighbors, strict=True))
                a2_sorted = sorted(zip(a2_ranks, a2_neighbors, strict=True))

                H1 = a1_sorted[0][1]
                L1 = a1_sorted[1][1]
                H2 = a2_sorted[0][1]
                L2 = a2_sorted[1][1]

                subs_indices = [
                    mn_to_mnidx[H1.GetAtomMapNum()],
                    mn_to_mnidx[L1.GetAtomMapNum()],
                    mn_to_mnidx[H2.GetAtomMapNum()],
                    mn_to_mnidx[L2.GetAtomMapNum()],
                ]

            elif sub_case == 1:
                a1_ranks = [int(n.GetProp("_CIPRank")) for n in a1_neighbors]
                a1_sorted = sorted(zip(a1_ranks, a1_neighbors, strict=True))
                H1 = a1_sorted[0][1]
                L1 = a1_sorted[1][1]
                S = a2_neighbors[0]

                subs_indices = [
                    mn_to_mnidx[H1.GetAtomMapNum()],
                    mn_to_mnidx[L1.GetAtomMapNum()],
                    mn_to_mnidx[S.GetAtomMapNum()],
                    -1,
                ]

            else:
                a2_ranks = [int(n.GetProp("_CIPRank")) for n in a2_neighbors]
                a2_sorted = sorted(zip(a2_ranks, a2_neighbors, strict=True))
                H2 = a2_sorted[0][1]
                L2 = a2_sorted[1][1]
                S = a1_neighbors[0]

                subs_indices = [
                    mn_to_mnidx[H2.GetAtomMapNum()],
                    mn_to_mnidx[L2.GetAtomMapNum()],
                    mn_to_mnidx[S.GetAtomMapNum()],
                    -1,
                ]

        except KeyError:
            continue

        if has_stereo:
            if stereo == Chem.BondStereo.STEREOZ:
                ez_tag = +1.0
            else:
                ez_tag = -1.0
        else:
            ez_tag = 0.0

        db_c1c2_list.append(
            [mn_to_mnidx[atom1.GetAtomMapNum()], mn_to_mnidx[atom2.GetAtomMapNum()]]
        )
        db_substituents_list.append(subs_indices)
        db_ez_tags_list.append(ez_tag)
        db_sub_case_list.append(sub_case)

    if len(db_c1c2_list) == 0:
        return (
            torch.empty((0, 2), dtype=torch.long),
            torch.empty((0, 4), dtype=torch.long),
            torch.empty(0, dtype=torch.float),
            torch.empty(0, dtype=torch.long),
        )

    db_c1c2_torch = torch.tensor(db_c1c2_list, dtype=torch.long)
    db_subs_torch = torch.tensor(db_substituents_list, dtype=torch.long)
    db_ez_tags_torch = torch.tensor(db_ez_tags_list, dtype=torch.float)
    db_sub_case_torch = torch.tensor(db_sub_case_list, dtype=torch.long)

    return (db_c1c2_torch, db_subs_torch, db_ez_tags_torch, db_sub_case_torch)
