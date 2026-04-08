"""
base_dataset.py — Sliding window generator with causal graph construction.
Updated for causality and normalization consistency.
"""

import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Data


class BaseTimeSeriesDataset(Dataset):
    """
    Generates (node_id, t, window, graph, label) tuples.
    
    FIXES:
    1. Causal Normalization: Uses provided mean/std to prevent future data leakage.
    2. Explicit Graph Semantics: Edges defined via Pearson correlation[cite: 109].
    """

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

        # --- CAUSAL NORMALIZATION  ---
        # If mean/std are provided (from training set), use them. 
        # Otherwise, calculate from current data (assumed to be training set).
        if norm_mean is not None and norm_std is not None:
            self.mean = norm_mean.reshape(1, -1)
            self.std  = norm_std.reshape(1, -1)
        else:
            self.mean = signals.mean(axis=0, keepdims=True)
            self.std  = signals.std(axis=0, keepdims=True) + 1e-8

        # Apply transformation
        self.signals = (signals - self.mean) / self.std
        self.labels  = labels

        # --- GRAPH CONSTRUCTION (Pearson Correlation) [cite: 106, 109, 110] ---
        self.graph = graph if graph is not None else \
            build_graph_from_correlation(self.signals, threshold=0.5)

        # Vectorized Indexing
        t_values = np.arange(window, T, stride, dtype=np.int32)
        n_times  = len(t_values)
        self._index_t = np.repeat(t_values, N)
        self._index_v = np.tile(np.arange(N, dtype=np.int32), n_times)
        self._len = len(self._index_t)

        # Pre-compute windows and labels
        self._precomputed_labels = self._vectorized_window_labels()
        self._precomputed_windows = self._build_all_windows()

        # Pre-compute forecast targets: signals[t] (one step AFTER window) 
        self._precomputed_targets = torch.from_numpy(
            self.signals[self._index_t, self._index_v].astype(np.float32)
        ).unsqueeze(-1)

    def _vectorized_window_labels(self) -> np.ndarray:
        """Anomaly in window if any timestamp t-W to t-1 is 1."""
        T, N = self.labels.shape
        W = self.window
        cum = np.zeros((T + 1, N), dtype=np.int64)
        cum[1:] = np.cumsum(self.labels, axis=0)
        
        t_arr = self._index_t
        v_arr = self._index_v
        window_sums = cum[t_arr, v_arr] - cum[t_arr - W, v_arr]
        return (window_sums > 0).astype(np.int64)

    def _build_all_windows(self) -> torch.Tensor:
        """Vectorized window extraction [n_samples, W, 1]."""
        from numpy.lib.stride_tricks import as_strided
        T, N = self.signals.shape
        W = self.window
        strides = self.signals.strides
        shape = (T - W + 1, W, N)
        new_strides = (strides[0], strides[0], strides[1])
        all_windows_view = as_strided(self.signals, shape=shape, strides=new_strides)
        
        start_indices = self._index_t - W
        node_indices = self._index_v
        windows = all_windows_view[start_indices, :, node_indices]
        return torch.from_numpy(windows.copy().astype(np.float32)).unsqueeze(-1)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return {
            "node_id" : int(self._index_v[idx]),
            "t"       : int(self._index_t[idx]),
            "x_window": self._precomputed_windows[idx],
            "graph"   : self.graph,
            "label"   : int(self._precomputed_labels[idx]),
        }

    def as_flat_list(self):
        """Used by curriculum scheduler[cite: 86, 205]."""
        return list(zip(
            self._index_v.tolist(),
            self._index_t.tolist(),
            self._precomputed_labels.tolist()
        ))


def build_graph_from_correlation(signals: np.ndarray, threshold: float = 0.5) -> Data:
    """
    Builds a static graph using Pearson correlation[cite: 109, 110].
    Edges are defined where |corr(i, j)| >= threshold.
    """
    corr = np.corrcoef(signals.T)
    corr = np.nan_to_num(corr, nan=0.0)
    abs_corr = np.abs(corr)
    np.fill_diagonal(abs_corr, 0.0)
    
    mask = abs_corr >= threshold
    src, dst = np.where(mask)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    
    # Store degree for H_struct calculations [cite: 142]
    degree = torch.zeros(signals.shape[1], dtype=torch.long)
    if len(src) > 0:
        degree = torch.tensor(np.bincount(src, minlength=signals.shape[1]), dtype=torch.long)

    graph = Data(edge_index=edge_index, num_nodes=signals.shape[1])
    graph.degree = degree
    return graph
