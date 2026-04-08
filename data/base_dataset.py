"""
base_dataset.py — Sliding window generator with causal graph construction.
Updated for UNIFIED PROCESSING UNIT: Yields full plant snapshots [N, W, 1] per timestep 't'.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

class BaseTimeSeriesDataset(Dataset):
    def __init__(self, signals: np.ndarray, labels: np.ndarray,
                 window: int = 30, stride: int = 1,
                 graph: Data | None = None,
                 norm_mean: np.ndarray | None = None,
                 norm_std: np.ndarray | None = None):
        super().__init__()
        assert signals.shape == labels.shape, "Signals and labels must match shape."

        T, N = signals.shape
        self.window = window
        self.stride = stride
        self.N = N
        self.T = T

        # Use provided mean/std (prevents val/test leakage)
        if norm_mean is not None and norm_std is not None:
            self.mean = norm_mean.reshape(1, -1)
            self.std  = norm_std.reshape(1, -1)
        else:
            self.mean = signals.mean(axis=0, keepdims=True)
            self.std  = signals.std(axis=0, keepdims=True) + 1e-8

        self.signals = (signals - self.mean) / self.std
        self.labels  = labels

        # Build semantic correlation graph
        self.graph = graph if graph is not None else \
            build_graph_from_correlation(self.signals, threshold=0.5)

        # 🛡️ FIX 1: Unit of execution is now TIMESTEP 't' across ALL nodes
        self._index_t = np.arange(window, T, stride, dtype=np.int32)
        self._len = len(self._index_t)

        self._precomputed_labels = self._vectorized_window_labels()
        
        # 🛡️ FIX 4: Correct Data Shape [num_samples, N, W, 1] applied here
        self._precomputed_windows = self._build_all_windows()

    def _vectorized_window_labels(self) -> np.ndarray:
        T, N = self.labels.shape
        W = self.window
        cum = np.zeros((T + 1, N), dtype=np.int64)
        cum[1:] = np.cumsum(self.labels, axis=0)
        
        t_arr = self._index_t
        # Get labels for ALL nodes at time t -> Shape: [num_samples, N]
        window_sums = cum[t_arr] - cum[t_arr - W]
        return (window_sums > 0).astype(np.int64)

    def _build_all_windows(self) -> torch.Tensor:
        """
        Extracts windows and forces [N, W, 1] shape per timestep.
        """
        from numpy.lib.stride_tricks import as_strided
        T, N = self.signals.shape
        W = self.window
        strides = self.signals.strides
        
        shape = (T - W + 1, W, N)
        new_strides = (strides[0], strides[0], strides[1])
        all_windows_view = as_strided(self.signals, shape=shape, strides=new_strides)
        
        start_indices = self._index_t - W
        
        # Extracted shape is [num_samples, W, N]
        windows = all_windows_view[start_indices]
        
        # Transpose to [num_samples, N, W] so each node has its own temporal sequence
        windows = np.transpose(windows, (0, 2, 1))
        
        # Add feature dimension -> [num_samples, N, W, 1]
        return torch.from_numpy(windows.copy().astype(np.float32)).unsqueeze(-1)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        # 🛡️ FIX: Returns the complete graph state at time 't'
        return {
            "t"    : int(self._index_t[idx]),
            "x"    : self._precomputed_windows[idx], # Shape [N, W, 1]
            "y"    : torch.tensor(self._precomputed_labels[idx], dtype=torch.long), # Shape [N]
            "graph": self.graph
        }

    def as_flat_list(self):
        # Used by curriculum scheduler to get timesteps and their corresponding label vectors
        return list(zip(self._index_t.tolist(), self._precomputed_labels.tolist()))


def build_graph_from_correlation(signals: np.ndarray, threshold: float = 0.5) -> Data:
    
    # 🛡️ FIX 2: Handle NaNs BEFORE graph build
    corr = np.corrcoef(signals.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    
    abs_corr = np.abs(corr)
    np.fill_diagonal(abs_corr, 0.0)
    
    mask = abs_corr >= threshold
    src, dst = np.where(mask)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    
    degree = torch.zeros(signals.shape[1], dtype=torch.long)
    if len(src) > 0:
        degree = torch.tensor(np.bincount(src, minlength=signals.shape[1]), dtype=torch.long)

    graph = Data(edge_index=edge_index, num_nodes=signals.shape[1])
    graph.degree = degree
    return graph
