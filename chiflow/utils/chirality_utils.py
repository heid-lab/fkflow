"""
This module computes, for each chiral center in an RDKit molecule, a 4x3x2 encoding of
its stereochemistry based on the Cahn-Ingold-Prelog (CIP) priority rules.

CIP: A system that assigns priority ranks to substituents around a stereocenter based on
atomic number and connectivity (e.g., higher atomic number = higher priority).
- After ranking, orient the lowest-priority substituent away from the viewer.
- Trace a path from highest to 2nd to 3rd priority: clockwise => R (rectus); counterclockwise => S (sinister).

The 4x3x2 matrix for each center:
 - axis 0 (size 4): choice of which neighbor points “away” from the viewer.
 - axis 1 (size 3): choice of the first “inward” neighbor.
 - axis 2 (size 2): two neighbor atom indices in clockwise order (first, second).
"""

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sympy.combinatorics import Permutation


def get_chiral_matrix(mol, perm_inv):
    """
    Build a (C,4,3,2) tensor for each chiral center:
      C: number of R/S centers
      4: choice of "away" neighbor
      3: choice of first "inward" neighbor
      2: (first, second) inward neighbors in CW order
    
    Count inversions in a permutation to determine parity (even vs. odd).
    An inversion is any pair of positions (i, j) with i < j such that
    permutation_list[i] > permutation_list[j]. The total number of such
    pairs defines the permutation's parity.

    Definitions:
    --------
    - chi: chiral
    - mn: map number
    - nbrs: neighbors
    - mnidx: the index of the list of atoms that is sorted by ascending map number
    """
    # 1. Assign stereochemical labels and CIP ranks
    AllChem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    stereo_centers_C = Chem.FindMolChiralCenters(mol, includeCIP=True, useLegacyImplementation=False)

    mn_to_mnidx = {atom.GetAtomMapNum():i for i, atom in enumerate(np.array(mol.GetAtoms())[perm_inv])}

    matrices_C_4_3_2_2 = []
    chi_center_nbrs_C_4 = []

    for center_idx, cip_label in stereo_centers_C:
        if cip_label not in ('R', 'S'):
            # skip non‐R/S or undefined centers
            continue
        chi_atom = mol.GetAtomWithIdx(center_idx)
        chi_map_num = chi_atom.GetAtomMapNum()

        # 2. Gather tetrahedral neighbors
        tetra_atoms_4 = list(chi_atom.GetNeighbors())
        if len(tetra_atoms_4) != 4: continue
        
        # Check if all neighbors have map numbers before proceeding
        if not all(atom.GetAtomMapNum() for atom in tetra_atoms_4):
            raise ValueError("All neighbor atoms must have AtomMapNum property set.")
        
        tetra_atom_to_list_idx = {atom: i for i, atom in enumerate(tetra_atoms_4)}

        # 3. Sort neighbors by CIP rank
        ranks_4 = [int(n.GetProp('_CIPRank')) for n in tetra_atoms_4]
        tetra_atoms_sorted_4 = [nbr for _, nbr in sorted(zip(ranks_4, tetra_atoms_4))]

        # 4. Define reference order: [highest, lowest, 2nd, 3rd]
        ref_order_4 = [tetra_atoms_sorted_4[3], tetra_atoms_sorted_4[0], tetra_atoms_sorted_4[1], tetra_atoms_sorted_4[2]]
        ref_idx_4 = [tetra_atom_to_list_idx[n] for n in ref_order_4]
        ref_even = Permutation(ref_idx_4).is_even
        cw_ref = (cip_label == 'R')  # R = clockwise reference

        # 5. Build the 4×3×2 block for this center
        matrix_4_3_2_2 = np.zeros((4, 3, 2, 2), dtype=int)
        chi_center_nbrs_4 = [-1] * 4
        for away_i, away_atom in enumerate(tetra_atoms_4):  # axis=0 (4 choices)
            chi_center_nbrs_4[away_i] = mn_to_mnidx[away_atom.GetAtomMapNum()]
            # the three that point inward
            inward_candidates_3 = [n for n in tetra_atoms_4 if n is not away_atom]
            for first_i, first_atom in enumerate(inward_candidates_3):  # axis=1 (3)
                # the two remaining for the second inward position
                opt1, opt2 = [n for n in inward_candidates_3 if n is not first_atom]

                # test parity of the trial ordering
                trial_order_4 = [away_atom, first_atom, opt1, opt2]
                trial_idx_4 = [tetra_atom_to_list_idx[n] for n in trial_order_4]
                trial_even = Permutation(trial_idx_4).is_even

                # decide which inward pair follows CW sense
                same_sense = (trial_even == ref_even)
                pick_cw = cw_ref if same_sense else not cw_ref
                second_atom = opt1 if pick_cw else opt2  # axis=2 (CW/CCW)

                # record atom indices from chi center to atom for cross-product => shape (4, 3, 2, 2)
                matrix_4_3_2_2[away_i, first_i, 0, :] = [mn_to_mnidx[chi_map_num], mn_to_mnidx[first_atom.GetAtomMapNum()]]
                matrix_4_3_2_2[away_i, first_i, 1, :] = [mn_to_mnidx[chi_map_num], mn_to_mnidx[second_atom.GetAtomMapNum()]]
        
        matrices_C_4_3_2_2.append(matrix_4_3_2_2)
        chi_center_nbrs_C_4.append(chi_center_nbrs_4)

    matrices_torch_C_4_3_2_2 = torch.from_numpy(
        np.array(matrices_C_4_3_2_2, dtype=int)
    )
    chi_center_nbrs_torch_C_4 = torch.from_numpy(
        np.array(chi_center_nbrs_C_4, dtype=int)
    )
    if len(matrices_torch_C_4_3_2_2) == 0:
        matrices_torch_C_4_3_2_2 = torch.empty((0, 4, 3, 2, 2), dtype=torch.long)
        chi_center_nbrs_torch_C_4 = torch.empty((0, 4), dtype=torch.long)

    return (
        matrices_torch_C_4_3_2_2, 
        chi_center_nbrs_torch_C_4,
    )
