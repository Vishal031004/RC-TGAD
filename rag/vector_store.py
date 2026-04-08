"""
vector_store.py — FAISS-backed vector store for RC-TGAD (Person 2)
FIXED: Memory Cap implemented to prevent OOM and stale distributions.
"""

import faiss
import numpy as np
from typing import List, Dict


class VectorStore:
    """
    FAISS L2 index that stores embeddings + binary labels (0=normal, 1=anomaly).
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)   # exact L2 search
        self.labels: List[int] = []            

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, z: np.ndarray, label: int) -> None:
        """Add one embedding to the store."""
        
        # 🛡️ FIX: Prevent Memory Bloat and Stale Retrievals
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
        """Bulk add — slightly faster than calling add() in a loop."""
        
        # 🛡️ FIX: Prevent Memory Bloat and Stale Retrievals
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
        """Retrieve k nearest neighbors."""
        n_stored = self.index.ntotal
        if n_stored == 0:
            return []

        k_actual = min(k, n_stored)
        z_np = _to_numpy(z).reshape(1, -1).astype("float32")
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
        """Clear the store (useful between datasets / ablation runs)."""
        self.index.reset()
        self.labels.clear()

    def save(self, path: str) -> None:
        faiss.write_index(self.index, path)
        np.save(path + ".labels.npy", np.array(self.labels, dtype=np.int32))

    def load(self, path: str) -> None:
        self.index = faiss.read_index(path)
        self.labels = np.load(path + ".labels.npy").tolist()


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)
