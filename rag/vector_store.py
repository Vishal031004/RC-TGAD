"""
vector_store.py — GPU-Accelerated FAISS store for RC-TGAD.
Supports Multi-GPU (T4 x2) on Kaggle for lightning-fast Hardness Eval.
"""

import faiss
import numpy as np
import torch
from typing import List, Dict


class VectorStore:
    """
    FAISS L2 index that stores embeddings + binary labels.
    Automatically moves to GPU if available.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        
        # 1. Initialize the CPU Index
        cpu_index = faiss.IndexFlatL2(dim)
        
        # 2. Check for GPU (Kaggle T4s)
        if torch.cuda.is_available():
            try:
                # We use a resource manager to speed up memory allocation
                self.res = faiss.StandardGpuResources()
                # Move index to GPU 0 (or use index_cpu_to_all_gpus for both T4s)
                self.index = faiss.index_cpu_to_gpu(self.res, 0, cpu_index)
                print(f"[VectorStore] Initialized FAISS-GPU on device {torch.cuda.current_device()}")
            except Exception as e:
                print(f"[VectorStore] GPU move failed, falling back to CPU: {e}")
                self.index = cpu_index
        else:
            self.index = cpu_index
            
        self.labels: List[int] = []            

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, z: np.ndarray, label: int) -> None:
        """Add one embedding to the store."""
        
        # 🛡️ Prevent Memory Bloat
        if self.index.ntotal > 50000:
            self.reset()
            
        z_np = _to_numpy(z).reshape(1, -1).astype("float32")
        if z_np.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self.dim}, got {z_np.shape[1]}"
            )
        self.index.add(z_np)
        self.labels.append(int(label))

    def add_batch(self, zs: np.ndarray, labels: List[int]) -> None:
        """Bulk add."""
        
        if self.index.ntotal + len(labels) > 50000:
            self.reset()
            
        zs_np = _to_numpy(zs).astype("float32")
        assert zs_np.shape[0] == len(labels), "zs and labels must have same length"
        self.index.add(zs_np)
        self.labels.extend([int(l) for l in labels])

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, z: np.ndarray, k: int = 10) -> List[Dict]:
        """Retrieve k nearest neighbors using GPU parallel search."""
        n_stored = self.index.ntotal
        if n_stored == 0:
            return []

        k_actual = min(k, n_stored)
        z_np = _to_numpy(z).reshape(1, -1).astype("float32")
        
        # This search is now happening on the GPU!
        distances, indices = self.index.search(z_np, k_actual)

        results = []
        for j, idx in enumerate(indices[0]):
            if idx == -1:          
                continue
            results.append({
                "label": self.labels[idx],
                "dist":  float(distances[0][j]),
            })
        return results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.index.ntotal

    def reset(self) -> None:
        """Clear the store."""
        self.index.reset()
        self.labels.clear()

    def save(self, path: str) -> None:
        # GPU indices must be moved back to CPU before saving to disk
        cpu_index = faiss.index_gpu_to_cpu(self.index)
        faiss.write_index(cpu_index, path)
        np.save(path + ".labels.npy", np.array(self.labels, dtype=np.int32))

    def load(self, path: str) -> None:
        cpu_index = faiss.read_index(path)
        if torch.cuda.is_available():
            self.res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(self.res, 0, cpu_index)
        else:
            self.index = cpu_index
        self.labels = np.load(path + ".labels.npy").tolist()


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)
