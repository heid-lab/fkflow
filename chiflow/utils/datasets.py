import pickle
import copy
import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

from utils.chem import BOND_TYPES


def generate_ts_data2(
    r_smarts,
    p_smarts,
    energies=None,
    add_chiral_edges=True,
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

    r_cip_decr_chi_nbrs_C_4, r_chi_center_C, r_rs_tag_C = get_cip_tetra_atoms_in_decreasing_order(r, r_perm_inv)
    p_cip_decr_chi_nbrs_C_4, p_chi_center_C, p_rs_tag_C = get_cip_tetra_atoms_in_decreasing_order(p, p_perm_inv)

    # create directed edges for each chiral center (4 per center), 
    # in the order of Rectus (R) or Sinister (S) (saved in r_rs_tag_C)
    def _get_chi_edges(cip_nbrs_C_4, rs_tag_C):
        # build edges (4 for each chiral center)
        edges_4C_2 = []
        for i, tag in enumerate(rs_tag_C):
            nbrs_4 = cip_nbrs_C_4[i]
            order = [0, 1, 2, 3] if tag == -1 else [0, 3, 2, 1]
            for j in range(4):
                src = nbrs_4[order[j]]
                dst = nbrs_4[order[(j + 1) % 4]]
                edges_4C_2.append([src, dst])
        chi_edges_4C_2 = np.array(edges_4C_2, dtype=np.int64)

        # build adjacency
        chi_adj = np.zeros((N, N), dtype=np.int64)
        if len(chi_edges_4C_2) > 0:
            idx = chi_edges_4C_2.T
            chi_adj[idx[0], idx[1]] = 1

        return chi_adj

    r_chi_adj = _get_chi_edges(r_cip_decr_chi_nbrs_C_4, r_rs_tag_C)
    p_chi_adj = _get_chi_edges(p_cip_decr_chi_nbrs_C_4, p_rs_tag_C)

    if add_chiral_edges:
        adj += r_chi_adj + p_chi_adj
    row, col = adj.nonzero()

    NO_BOND = 0
    CHI_EDGE_TYPE = len(BOND_TYPES) + 1
    def _compute_edge_types(mol, perm_inv, chi_adj, rows, cols):
        edge_types = []
        for src, dst in zip(rows, cols):
            bond = mol.GetBondBetweenAtoms(int(perm_inv[src]), int(perm_inv[dst]))
            if add_chiral_edges and (chi_adj[src, dst] == 1):
                edge_types.append(CHI_EDGE_TYPE)
            elif bond is not None:
                edge_types.append(BOND_TYPES[bond.GetBondType()])
            else:
                edge_types.append(NO_BOND)
        return torch.tensor(edge_types)

    r_edge_type = _compute_edge_types(r, r_perm_inv, r_chi_adj, row, col)
    p_edge_type = _compute_edge_types(p, p_perm_inv, p_chi_adj, row, col)
    
    edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)

    perm = (edge_index[0] * N + edge_index[1]).argsort()

    edge_index = edge_index[:, perm]
    r_edge_type = r_edge_type[perm]
    p_edge_type = p_edge_type[perm]

    edge_type = r_edge_type * (CHI_EDGE_TYPE+1) + p_edge_type # +1 for potential no bond
    
    data = TSData(
        atom_type=z,
        r_feat=r_feat,
        p_feat=p_feat,
        pos=pos,

        r_chi_center_C_index=r_chi_center_C,
        p_chi_center_C_index=p_chi_center_C,
        r_cip_decr_chi_nbrs_C_4_index=r_cip_decr_chi_nbrs_C_4,
        p_cip_decr_chi_nbrs_C_4_index=p_cip_decr_chi_nbrs_C_4,
        r_rs_tag_C=r_rs_tag_C,
        p_rs_tag_C=p_rs_tag_C,

        edge_index=edge_index,
        edge_type=edge_type,
        rdmol=(copy.deepcopy(r), copy.deepcopy(p)),
        smiles=f"{r_smarts}>>{p_smarts}",
        #energies=energies,
    )
    return data, feat_dict


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
    Also returns a tensor with 0 for R and 1 for S for each chiral center.
    :param mol: RDKit molecule object
    :param perm_inv: Inverse permutation array for atom map numbers
    :return: (tensor of tetrahedral atoms map numbers, tensor of chiral center indices, tensor of R/S tags)
    """
    AllChem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    #AssignCIPLabels(mol)
    # stereo_centers_C = Chem.FindMolChiralCenters(mol, includeCIP=True, useLegacyImplementation=False)
    mn_to_mnidx = {atom.GetAtomMapNum():i for i, atom in enumerate(np.array(mol.GetAtoms())[perm_inv])}
    
    cip_decr_chi_nbrs_C_4 = []
    chi_center_C = []
    rs_tag_C = []
    
    #for center_idx, cip_label in stereo_centers_C:
    for chi_atom in mol.GetAtoms():
        cip_label = None
        if chi_atom.HasProp('_CIPCode'):
            cip_label = chi_atom.GetProp('_CIPCode')
            if cip_label not in ('R', 'S'):
                continue
        else:
            continue
        
        #chi_atom = mol.GetAtomWithIdx(center_idx)
        chi_map_num = chi_atom.GetAtomMapNum()
        
        # 2. Gather tetrahedral neighbors
        tetra_atoms_4 = list(chi_atom.GetNeighbors())
        if len(tetra_atoms_4) != 4: continue
        
        # Check if all neighbors have map numbers before proceeding
        if not all(atom.GetAtomMapNum() for atom in tetra_atoms_4):
            raise ValueError("All neighbor atoms must have AtomMapNum property set.")
                
        # 3. Sort neighbors by CIP rank
        ranks_4 = [int(n.GetProp('_CIPRank')) for n in tetra_atoms_4]
        assert len(set(ranks_4)) == 4
        tetra_atoms_sorted_4 = [nbr for _, nbr in sorted(zip(ranks_4, tetra_atoms_4), reverse=True)]
        
        # 4. Get the map numbers in sorted order
        rank_sorted_mapnums_4 = [n.GetAtomMapNum() for n in tetra_atoms_sorted_4]
        
        # 5. Convert map numbers to indices in the sorted list (ordered by ascending map number)
        cip_decr_chi_nbrs_4 = [mn_to_mnidx[n] for n in rank_sorted_mapnums_4]
        chi_mn_idx = mn_to_mnidx[chi_map_num]
        
        cip_decr_chi_nbrs_C_4.append(cip_decr_chi_nbrs_4)
        chi_center_C.append(chi_mn_idx)
        rs_tag_C.append(-1 if cip_label == 'R' else 1)
    
    # Convert to tensor
    if len(cip_decr_chi_nbrs_C_4) == 0:
        cip_decr_chi_nbrs_C_4 = torch.empty((0, 4), dtype=torch.int32)
        chi_center_C = torch.empty((0,), dtype=torch.int32)
        rs_tag_C = torch.empty((0,), dtype=torch.int32)
    else:
        cip_decr_chi_nbrs_C_4 = torch.from_numpy(np.array(cip_decr_chi_nbrs_C_4, dtype=np.int32))
        chi_center_C = torch.from_numpy(np.array(chi_center_C, dtype=np.int32))
        rs_tag_C = torch.from_numpy(np.array(rs_tag_C, dtype=np.int32))
    
    return cip_decr_chi_nbrs_C_4, chi_center_C, rs_tag_C


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
    def __init__(self, path=None, data_R=None, transform=None):
        if data_R is not None:
            self.data = data_R
        else:
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
