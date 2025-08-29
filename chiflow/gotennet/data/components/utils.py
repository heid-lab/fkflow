import numpy as np
import torch
from pytorch_lightning.utilities import rank_zero_warn
import pickle
from torch_geometric.data import Dataset


class ConformationDataset(Dataset):
    def __init__(self, data_file, data_indices, transform=None):
        """
        Select data according to rxn_index attribute.
        
        Parameters:
        - data_file: Path to the pickle file containing all data samples
        - data_indices: Indices matching the rxn_index attribute in data objects
        - transform: Optional transforms to apply to each data sample
        """
        super().__init__()
        
        # Load the full dataset
        with open(data_file, "rb") as f:
            all_data = pickle.load(f)
        
        # Create a mapping from rxn_index to the data object
        rxn_index_to_data = {}
        for data_obj in all_data:
            key = data_obj.rxn_index if isinstance(data_obj.rxn_index, int) else data_obj.rxn_index.item()
            rxn_index_to_data[key] = data_obj
        
        # Get data by the provided rxn_indices
        self.data = [rxn_index_to_data[idx] for idx in data_indices]
        self.transform = transform
        
        # Cache atom and edge types
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
    
    
class CountNodesPerGraph(object):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data):
        if not hasattr(data, "__num_nodes__"):
            data.num_nodes = len(data.pos)
        data.num_nodes_per_graph = torch.LongTensor([data.num_nodes])
        return data


def train_val_test_split(dset_len, train_size, val_size, test_size, seed):
    assert (train_size is None) + (val_size is None) + (
            test_size is None) <= 1, "Only one of train_size, val_size, test_size is allowed to be None."

    is_float = (isinstance(train_size, float), isinstance(val_size, float), isinstance(test_size, float))

    train_size = round(dset_len * train_size) if is_float[0] else train_size
    val_size = round(dset_len * val_size) if is_float[1] else val_size
    test_size = round(dset_len * test_size) if is_float[2] else test_size

    if train_size is None:
        train_size = dset_len - val_size - test_size
    elif val_size is None:
        val_size = dset_len - train_size - test_size
    elif test_size is None:
        test_size = dset_len - train_size - val_size

    if train_size + val_size + test_size > dset_len:
        if is_float[2]:
            test_size -= 1
        elif is_float[1]:
            val_size -= 1
        elif is_float[0]:
            train_size -= 1

    assert train_size >= 0 and val_size >= 0 and test_size >= 0, (
        f"One of training ({train_size}), validation ({val_size}) or "
        f"testing ({test_size}) splits ended up with a negative size."
    )

    total = train_size + val_size + test_size
    assert dset_len >= total, f"The dataset ({dset_len}) is smaller than the combined split sizes ({total})."

    if total < dset_len:
        rank_zero_warn(f"{dset_len - total} samples were excluded from the dataset")

    idxs = np.arange(dset_len, dtype=np.int64)
    idxs = np.random.default_rng(seed).permutation(idxs)

    idx_train = idxs[:train_size]
    idx_val = idxs[train_size: train_size + val_size]
    idx_test = idxs[train_size + val_size: total]

    return np.array(idx_train), np.array(idx_val), np.array(idx_test)


def make_splits(dataset_len, train_size, val_size, test_size, seed, filename=None, splits=None):
    if splits is not None:
        splits = np.load(splits)
        idx_train = splits["idx_train"]
        idx_val = splits["idx_val"]
        idx_test = splits["idx_test"]
    else:
        idx_train, idx_val, idx_test = train_val_test_split(dataset_len, train_size, val_size, test_size, seed)

    if filename is not None:
        np.savez(filename, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

    return torch.from_numpy(idx_train), torch.from_numpy(idx_val), torch.from_numpy(idx_test)


class MissingLabelException(Exception):
    pass
