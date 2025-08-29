"""
Chirality evaluation metrics for flow matching models.

This module provides functions to evaluate the correctness of chiral centers
in molecular structures, using normalized chiral volumes and threshold-based
correctness checking.
"""

import torch
from torch_geometric.data import Data


def evaluate_chirality_correctness(pos_N_3, batch: Data, threshold: float = 0.1):
    """
    Evaluate chirality correctness with normalized volume and threshold checking.
    
    For each chiral center, compute normalized chiral volume and check if it's 
    in the correct direction (positive for R, negative for S) within threshold.
    
    Parameters:
        pos_N_3 (torch.Tensor): Atomic positions (N, 3) - should be TS positions (i=1)
        batch (Data): Batch data containing chiral center indices and R/S tags
        threshold (float): Tolerance threshold - volumes closer to 0 than this are considered wrong
    
    Returns:
        Dict[str, float]: Dictionary with chirality error percentage
    """
    device = pos_N_3.device
    Nm = int(batch.batch.max().item() + 1)  # number of molecules
    
    # Get chiral center data (using same structure as compute_chiral_volume)
    r_chi_center_C_4_index = getattr(batch, 'r_cip_decr_chi_nbrs_C_4_index', torch.empty((0, 4), dtype=torch.long))
    p_chi_center_C_4_index = getattr(batch, 'p_cip_decr_chi_nbrs_C_4_index', torch.empty((0, 4), dtype=torch.long))
    r_rs_tag_C = getattr(batch, 'r_rs_tag_C', torch.empty(0, dtype=torch.float))
    p_rs_tag_C = getattr(batch, 'p_rs_tag_C', torch.empty(0, dtype=torch.float))
    
    chi_center_C_4_index_2 = [r_chi_center_C_4_index, p_chi_center_C_4_index]
    rs_tag_C_2 = [r_rs_tag_C, p_rs_tag_C]
    
    total_chiral_centers = 0
    wrong_predictions = 0
    
    for i in [0, 1]:  # reactant and product chiral centers
        indices_C_4 = chi_center_C_4_index_2[i]
        rs_tags = rs_tag_C_2[i]
        
        if indices_C_4.shape[0] == 0:  # Skip if no chiral centers
            continue
            
        C = indices_C_4.shape[0]
        N = pos_N_3.shape[0]
        
        # Create one-hot encoding matrix for gradient-friendly indexing (same as compute_chiral_volume)
        indices_C_4 = indices_C_4.to(device).long()
        indices_flat = indices_C_4.flatten()
        
        # Create one-hot matrix
        one_hot = torch.zeros(C*4, N, device=device, dtype=pos_N_3.dtype)
        for j, idx in enumerate(indices_flat):
            if idx < N:  # Ensure index is valid
                one_hot[j, idx] = 1.0
        
        # Use matrix multiplication to gather positions
        pos_gathered_C4_3 = torch.mm(one_hot, pos_N_3)
        pos_indexed_C_4_3 = pos_gathered_C4_3.view(C, 4, 3)
        
        # Compute chiral volume (same as compute_chiral_volume)
        tetrahedral_sides_C_3_3 = pos_indexed_C_4_3[:, 1:,:] - pos_indexed_C_4_3[:, 0 ,:].unsqueeze(1)
        cross_product_C_3 = torch.cross(tetrahedral_sides_C_3_3[:,0, :], tetrahedral_sides_C_3_3[:,1,:])
        scalar_triple_products_C = torch.sum(tetrahedral_sides_C_3_3[:,2,:]*cross_product_C_3, dim = 1)
        
        # Normalize by "cube" volume (product of bond lengths)
        bond_lengths_C_3 = torch.norm(tetrahedral_sides_C_3_3, dim=2)  # (C, 3)
        cube_volumes_C = torch.prod(bond_lengths_C_3, dim=1)  # (C,)
        
        # Avoid division by zero
        valid_volumes = cube_volumes_C > 1e-8
        normalized_volumes_C = torch.zeros_like(scalar_triple_products_C)
        normalized_volumes_C[valid_volumes] = scalar_triple_products_C[valid_volumes] / cube_volumes_C[valid_volumes]
        
        # Apply R/S tags (same as compute_chiral_volume)
        rs_tag_float = rs_tags.float().to(device)
        signed_normalized_volumes_C = normalized_volumes_C * rs_tag_float
        
        # Check correctness: positive volumes above threshold are correct
        for j in range(C):
            if valid_volumes[j]:  # Only count valid chiral centers
                total_chiral_centers += 1
                volume = signed_normalized_volumes_C[j].item()
                
                # Volume should be positive and above threshold for correct chirality
                if volume <= threshold:
                    wrong_predictions += 1
    
    # Calculate error percentage
    if total_chiral_centers > 0:
        chirality_error_pct = (wrong_predictions / total_chiral_centers) * 100.0
    else:
        chirality_error_pct = 0.0
    
    return {"chirality_error_pct": round(chirality_error_pct, 4)}


def evaluate_persistent_chirality_correctness(pos_N_3, batch: Data, threshold: float = 0.1):
    """
    Evaluate chirality correctness ONLY for chiral centers that stay the same 
    between reactants and products (same center atom with same neighbors).
    
    Parameters:
        pos_N_3 (torch.Tensor): Atomic positions (N, 3) - should be TS positions (i=1)
        batch (Data): Batch data containing chiral center indices and R/S tags
        threshold (float): Tolerance threshold - volumes further than 0 are considered wrong
    
    Returns:
        Dict[str, float]: Dictionary with persistent chirality error percentage
    """
    device = pos_N_3.device
    
    # Get chiral center data
    r_chi_center_C_4_index = getattr(batch, 'r_cip_decr_chi_nbrs_C_4_index', torch.empty((0, 4), dtype=torch.long))
    p_chi_center_C_4_index = getattr(batch, 'p_cip_decr_chi_nbrs_C_4_index', torch.empty((0, 4), dtype=torch.long))
    r_rs_tag_C = getattr(batch, 'r_rs_tag_C', torch.empty(0, dtype=torch.float))
    p_rs_tag_C = getattr(batch, 'p_rs_tag_C', torch.empty(0, dtype=torch.float))
    
    if r_chi_center_C_4_index.shape[0] == 0 or p_chi_center_C_4_index.shape[0] == 0:
        return {"persistent_chirality_error_pct": 0.0}
    
    # Find persistent chiral centers (same center atom and same set of neighbors)
    persistent_centers = []
    persistent_tags = []
    
    for i, r_center in enumerate(r_chi_center_C_4_index):
        r_center_atom = r_center[0].item()
        r_neighbors = set(r_center[1:].tolist())
        
        # Look for matching center in products
        for j, p_center in enumerate(p_chi_center_C_4_index):
            p_center_atom = p_center[0].item()
            p_neighbors = set(p_center[1:].tolist())
            
            # Check if same center atom and same neighbors
            if r_center_atom == p_center_atom and r_neighbors == p_neighbors:
                # Check if both have the same R/S tag (chirality should be preserved)
                if r_rs_tag_C[i] == p_rs_tag_C[j]:
                    persistent_centers.append(r_center)
                    persistent_tags.append(r_rs_tag_C[i])
                break
    
    if len(persistent_centers) == 0:
        return {"persistent_chirality_error_pct": 0.0}
    
    # Convert to tensors
    persistent_indices = torch.stack(persistent_centers).to(device)
    persistent_rs_tags = torch.tensor(persistent_tags, dtype=torch.float, device=device)
    
    C = len(persistent_centers)
    N = pos_N_3.shape[0]
    
    # Create one-hot encoding matrix
    indices_flat = persistent_indices.flatten()
    one_hot = torch.zeros(C*4, N, device=device, dtype=pos_N_3.dtype)
    for j, idx in enumerate(indices_flat):
        if idx < N:
            one_hot[j, idx] = 1.0
    
    # Gather positions
    pos_gathered_C4_3 = torch.mm(one_hot, pos_N_3)
    pos_indexed_C_4_3 = pos_gathered_C4_3.view(C, 4, 3)
    
    # Compute chiral volume
    tetrahedral_sides_C_3_3 = pos_indexed_C_4_3[:, 1:,:] - pos_indexed_C_4_3[:, 0 ,:].unsqueeze(1)
    cross_product_C_3 = torch.cross(tetrahedral_sides_C_3_3[:,0, :], tetrahedral_sides_C_3_3[:,1,:])
    scalar_triple_products_C = torch.sum(tetrahedral_sides_C_3_3[:,2,:]*cross_product_C_3, dim = 1)
    
    # Normalize by "cube" volume
    bond_lengths_C_3 = torch.norm(tetrahedral_sides_C_3_3, dim=2)
    cube_volumes_C = torch.prod(bond_lengths_C_3, dim=1)
    
    # Avoid division by zero
    valid_volumes = cube_volumes_C > 1e-8
    normalized_volumes_C = torch.zeros_like(scalar_triple_products_C)
    normalized_volumes_C[valid_volumes] = scalar_triple_products_C[valid_volumes] / cube_volumes_C[valid_volumes]
    
    # Apply R/S tags
    signed_normalized_volumes_C = normalized_volumes_C * persistent_rs_tags
    
    # Check correctness
    total_persistent_centers = 0
    wrong_predictions = 0
    
    for j in range(C):
        if valid_volumes[j]:
            total_persistent_centers += 1
            volume = signed_normalized_volumes_C[j].item()
            
            # Volume should be positive and above threshold for correct chirality
            if volume <= threshold:
                wrong_predictions += 1
    
    # Calculate error percentage
    if total_persistent_centers > 0:
        persistent_chirality_error_pct = (wrong_predictions / total_persistent_centers) * 100.0
    else:
        persistent_chirality_error_pct = 0.0
    
    return {"persistent_chirality_error_pct": round(persistent_chirality_error_pct, 4)}