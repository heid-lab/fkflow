import pickle
import copy
import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from utils.chem import BOND_TYPES
from utils.chirality_utils import get_chiral_matrix


class TSData(Data):
    def __cat_dim__(self, key, item, *args, **kwargs):
            """
            Determines the concatenation dimension for a given attribute.
            """
            # List of attribute keys that should be concatenated along the first dimension (dim=0)
            # instead of the default last dimension for '*_index' suffixed tensors.
            keys_to_cat_dim_0 = {
                'r_cip_decr_chi_nbrs_C_4_index',
                'p_cip_decr_chi_nbrs_C_4_index',
            }

            if key in keys_to_cat_dim_0:
                # If the item is a tensor, concatenate along the first dimension.
                return 0
            
            # For all other attributes, fall back to the default PyG behavior.
            return super().__cat_dim__(key, item, *args, **kwargs)


def generate_ts_data2(
    r_smarts,
    p_smarts,
    energies=None,
    xyz_block=None,
    rxn_block=None,
    feat_dict={},
    only_sampling=True,
):
    r = Chem.MolFromSmarts(r_smarts)
    p = Chem.MolFromSmarts(p_smarts)
    Chem.SanitizeMol(r)
    Chem.SanitizeMol(p)

    N = r.GetNumAtoms()
    if rxn_block is not None:
        pos = torch.Tensor(rxn_block) # rxn_block_RTSP_N_3D
        pos = pos.transpose(0, 1)
        assert pos.shape[0] == N and p.GetNumAtoms() == N
    else:
        pos = torch.zeros(N,3)

    r_perm = np.array([a.GetAtomMapNum() for a in r.GetAtoms()]) - 1
    p_perm = np.array([a.GetAtomMapNum() for a in p.GetAtoms()]) - 1
    assert r_perm.min() >= 0 and np.unique(r_perm).size == N
    assert p_perm.min() >= 0 and np.unique(p_perm).size == N
    r_perm_inv = np.argsort(r_perm)
    p_perm_inv = np.argsort(p_perm)

    r_atomic_number, p_atomic_number = [], []
    r_feat, p_feat = [], []
    
    def get_closest_value(v, feat):
        return v.get(
            feat,
            v[min(v.keys(), key=lambda k: abs(int(k) - int(feat)))] # get closest key to 'feat' in 'v', return its value
        )

    # feat: len(v) done for one-hot encoding of feat based on len(v)
    for atom in np.array(r.GetAtoms())[r_perm_inv]:
        r_atomic_number.append(atom.GetAtomicNum())
        atomic_feat = []
        for k, v in feat_dict.items():
            feat = getattr(atom, k)()
            if not only_sampling:
                if feat not in v:
                    v.update({feat: len(v)})
            atomic_feat.append(get_closest_value(v, feat))
        r_feat.append(atomic_feat)

    for atom in np.array(p.GetAtoms())[p_perm_inv]:
        p_atomic_number.append(atom.GetAtomicNum())
        atomic_feat = []
        for k, v in feat_dict.items():
            feat = getattr(atom, k)()
            if not only_sampling:
                if feat not in v:
                    v.update({feat: len(v)})
            atomic_feat.append(get_closest_value(v, feat))
        p_feat.append(atomic_feat)

    assert r_atomic_number == p_atomic_number
    z = torch.tensor(r_atomic_number, dtype=torch.long)
    r_feat = torch.tensor(r_feat, dtype=torch.long)
    p_feat = torch.tensor(p_feat, dtype=torch.long)
    r_adj = Chem.rdmolops.GetAdjacencyMatrix(r)
    p_adj = Chem.rdmolops.GetAdjacencyMatrix(p)
    r_adj_perm = r_adj[r_perm_inv, :].T[r_perm_inv, :].T
    p_adj_perm = p_adj[p_perm_inv, :].T[p_perm_inv, :].T
    adj = r_adj_perm + p_adj_perm
    row, col = adj.nonzero()

    _nonbond = 0
    p_edge_type = []
    for i, j in zip(p_perm_inv[row], p_perm_inv[col]):
        b = p.GetBondBetweenAtoms(int(i), int(j))
        if b is not None:
            p_edge_type.append(BOND_TYPES[b.GetBondType()])
        elif b is None:
            p_edge_type.append(_nonbond)

    r_edge_type = []
    for i, j in zip(r_perm_inv[row], r_perm_inv[col]):
        b = r.GetBondBetweenAtoms(int(i), int(j))
        if b is not None:
            r_edge_type.append(BOND_TYPES[b.GetBondType()])
        elif b is None:
            r_edge_type.append(_nonbond)
    
    edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)
    
    r_nonzero = np.array(r_adj_perm.nonzero())
    r_edge_index = torch.tensor(r_nonzero, dtype=torch.long)
        
    p_nonzero = np.array(p_adj_perm.nonzero())
    p_edge_index = torch.tensor(p_nonzero, dtype=torch.long)
    
    r_edge_type = torch.tensor(r_edge_type)
    p_edge_type = torch.tensor(p_edge_type)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    r_perm_tensor = (r_edge_index[0] * N + r_edge_index[1]).argsort()
    p_perm_tensor = (p_edge_index[0] * N + p_edge_index[1]).argsort()

    edge_index = edge_index[:, perm]
    r_edge_index = r_edge_index[:, r_perm_tensor]
    p_edge_index = p_edge_index[:, p_perm_tensor]

    r_edge_type = r_edge_type[perm]
    p_edge_type = p_edge_type[perm]
    edge_type = r_edge_type * len(BOND_TYPES) + p_edge_type
        
    mapnum_to_r_index = torch.from_numpy(r_perm_inv)
    mapnum_to_p_index = torch.from_numpy(p_perm_inv)
    
    # Handle energies
    energies = torch.tensor(energies, dtype=torch.float32) if energies is not None else None

    # Chirality: Cross Product features ------------------------------------------------------------------
    r_node_cp_C_4_3_2_2_index, r_chi_center_nbrs_C_4 = get_chiral_matrix(r, r_perm_inv)
    p_node_cp_C_4_3_2_2_index, p_chi_center_nbrs_C_4 = get_chiral_matrix(p, p_perm_inv)

    # Chirality: CIP R/S features -------------------------------------------------------------------------
    # KM's wish
    r_cip_tag_N = get_cip_r_s_tag(r, r_perm_inv)
    p_cip_tag_N = get_cip_r_s_tag(p, p_perm_inv)
    # KM's wish
    r_cip_tetra_atoms_in_decreasing_order_4_N = get_cip_tetra_atoms_in_decreasing_order(r, r_perm_inv)
    p_cip_tetra_atoms_in_decreasing_order_4_N = get_cip_tetra_atoms_in_decreasing_order(p, p_perm_inv)

    data = TSData(
        atom_type=z,
        r_feat=r_feat,
        p_feat=p_feat,
        pos=pos,

        # Chirality: Cross Product features
        # used in gotennet
        # saves index of node array
        r_node_cp_C_4_3_2_2_index=r_node_cp_C_4_3_2_2_index,
        p_node_cp_C_4_3_2_2_index=p_node_cp_C_4_3_2_2_index,

        # Chirality: CIP R/S features
        r_cip_tag_N=r_cip_tag_N,
        p_cip_tag_N=p_cip_tag_N,
        r_cip_tetra_atoms_in_decreasing_order_4_N_index=r_cip_tetra_atoms_in_decreasing_order_4_N,
        p_cip_tetra_atoms_in_decreasing_order_4_N_index=p_cip_tetra_atoms_in_decreasing_order_4_N,

        # Neighbors of chiral centers.
        # used in gotennet
        r_chi_center_nbrs_C_4_index=r_chi_center_nbrs_C_4,
        p_chi_center_nbrs_C_4_index=p_chi_center_nbrs_C_4,
        
        edge_index=edge_index,
        r_edge_index=r_edge_index,
        p_edge_index=p_edge_index,
        edge_type=edge_type,
        rdmol=(copy.deepcopy(r), copy.deepcopy(p)),
        smiles=f"{r_smarts}>>{p_smarts}",
        energies=energies,

        mapnum_to_r_index=mapnum_to_r_index,
        mapnum_to_p_index=mapnum_to_p_index
    )
    return data, feat_dict


def get_cip_r_s_tag(mol, perm_inv):
    """
    Get the CIP R/S tag for each atom in the molecule.
    Ordered by ascending atom map number (with perm_inv).
    R -> 0, S -> 1, undefined -> -1
    :param mol: RDKit molecule object
    :param perm_inv: Inverse permutation array for atom map numbers
    :return: torch tensor of CIP tags
    """
    cip_tags_N = [-1] * mol.GetNumAtoms()
    for i, a in enumerate(np.array(mol.GetAtoms())[perm_inv]):
        if a.HasProp("_CIPCode"):
            cip_tag = a.GetProp("_CIPCode")
            if cip_tag == "R": cip_tags_N[i] = 0
            elif cip_tag == "S": cip_tags_N[i] = 1
            else: cip_tags_N[i] = -1
        else:
            cip_tags_N[i] = -1
    
    return torch.tensor(cip_tags_N, dtype=torch.long)


def get_cip_tetra_atoms_in_decreasing_order(mol, perm_inv):
    """
    Get the tetrahedral atoms map numbers in decreasing order of their CIP rank.
    Ordered by ascending atom map number (with perm_inv).
    :param mol: RDKit molecule object
    :param perm_inv: Inverse permutation array for atom map numbers
    :return: torch tensor of tetrahedral atoms map numbers
    """
    mn_to_mnidx = {atom.GetAtomMapNum():i for i, atom in enumerate(np.array(mol.GetAtoms())[perm_inv])}
    rank_sorted_mapnums_4_N = np.full((4, mol.GetNumAtoms()), -1, dtype=int)
    for i, a in enumerate(np.array(mol.GetAtoms())[perm_inv]):
        if a.HasProp("_CIPCode"):
            cip_tag = a.GetProp("_CIPCode")
            if cip_tag in ['R', 'S']:
                tetra_atoms_4 = list(a.GetNeighbors())
                if len(tetra_atoms_4) != 4: continue
                cip_ranks_4 = np.array([int(n.GetProp('_CIPRank')) for n in tetra_atoms_4])
                mapnums_4 = np.array([mn_to_mnidx[n.GetAtomMapNum()] for n in tetra_atoms_4])
                rank_sorted_mapnums_4_N[:, i] = mapnums_4[np.argsort(cip_ranks_4)]
    
    return torch.tensor(rank_sorted_mapnums_4_N, dtype=torch.long)


def get_chi_center_nbrs(mol, perm_inv):
    # have the nbrs in the list by ascending mapnum
    mapnum_to_list_idx = {mapnum:i for i, mapnum in enumerate(mol.GetAtoms()[perm_inv])}
    mapnums_C4 = []
    for i, a in enumerate(np.array(mol.GetAtoms())[perm_inv]):
        if not a.HasProp("_CIPCode"): continue
        cip_tag = a.GetProp("_CIPCode")
        if cip_tag not in ['R', 'S']: continue
        tetra_atoms_4 = list(a.GetNeighbors())
        if len(tetra_atoms_4) != 4: continue
        mapnums_C4.extend([n.GetAtomMapNum() for n in tetra_atoms_4])
    
    mapnums_ordered_C4 = [-1] * len(mapnums_C4)
    for i in range(len(mapnums_C4)):
        mapnums_ordered_C4[mapnum_to_list_idx[mapnums_C4[i]]] = mapnums_C4[i]
    
    return torch.tensor(mapnums_ordered_C4, dtype=torch.long)


class ConformationDataset(Dataset):
    def __init__(self, path, transform=None):
        with open(path, "rb") as f:
            self.data = pickle.load(f)
        self.transform = transform
        self.atom_types = self._atom_types()
        self.edge_types = self._edge_types()

    def __getitem__(self, idx):
        data = self.data[idx].clone()
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.data)

    def _atom_types(self):
        """All atom types."""
        atom_types = set()
        for graph in self.data:
            atom_types.update(graph.atom_type.tolist())
        return sorted(atom_types)

    def _edge_types(self):
        """All edge types."""
        edge_types = set()
        for graph in self.data:
            edge_types.update(graph.edge_type.tolist())
        return sorted(edge_types)
