"""
vector_store.py — Optimized CPU FAISS store for RC-TGAD.
Uses Inverted File Index (IVF) to accelerate search without needing GPU-specific installs.
"""

import faiss
import numpy as np
from typing import List, Dict, Union


class VectorStore:
    """
    FAISS IVF index that stores embeddings + binary labels.
    Uses clustering to achieve near-GPU speeds on standard CPU hardware.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        # nlist: Number of clusters. 100 is ideal for the 50k memory cap.
        self.nlist = 100 
        
        # The quantizer tells FAISS how to find the nearest cluster center
        quantizer = faiss.IndexFlatL2(dim)
        
        # IVF Index: Partitions the vector space into cells (Inverted File)
        self.index = faiss.IndexIVFFlat(quantizer, dim, self.nlist, faiss.METRIC_L2)
        
        self.labels: List[int] = []            

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, z: Union[np.ndarray, 'torch.Tensor'], label: int) -> None:
        """Add one embedding to the store with auto-training."""
        
        # 🛡️ Prevent Memory Bloat (Same logic as original)
        if self.index.ntotal > 50000:
            self.reset()
            
        z_np = _to_numpy(z).reshape(1, -1).astype("float32")
        
        # IVF requires a 'training' phase to establish cluster centroids.
        # We'll use the first point to initialize if not trained.
        if not self.index.is_trained:
            self.index.train(z_np)
            
        self.index.add(z_np)
        self.labels.append(int(label))

    def add_batch(self, zs: Union[np.ndarray, 'torch.Tensor'], labels: List[int]) -> None:
        """Bulk add to the store."""
        
        if self.index.ntotal + len(labels) > 50000:
            self.reset()
            
        zs_np = _to_numpy(zs).astype("float32")
        
        if not self.index.is_trained:
            self.index.train(zs_np)
            
        self.index.add(zs_np)
        self.labels.extend([int(l) for l in labels])

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, z: Union[np.ndarray, 'torch.Tensor'], k: int = 10) -> List[Dict]:
        """Retrieve k nearest neighbors using fast clustered search."""
        n_stored = self.index.ntotal
        
        # If we don't have enough data yet to perform a meaningful search
        if n_stored < 1:
            return []

        z_np = _to_numpy(z).reshape(1, -1).astype("float32")
        
        # nprobe: How many clusters to check. 
        # 1 = fastest (but less accurate), 10 = high accuracy/high speed.
        self.index.nprobe = 10 
        
        k_actual = min(k, n_stored)
        distances, indices = self.index.search(z_np, k_actual)

        results = []
        for j, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(self.labels):          
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
        """Clear the store and re-initialize the index."""
        # Note: IVF indices need a full reset to clear training data
        quantizer = faiss.IndexFlatL2(self.dim)
        self.index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, faiss.METRIC_L2)
        self.labels.clear()

    def save(self, path: str) -> None:
        """Serialize index and labels to disk."""
        faiss.write_index(self.index, path)
        np.save(path + ".labels.npy", np.array(self.labels, dtype=np.int32))

    def load(self, path: str) -> None:
        """Load index and labels from disk."""
        self.index = faiss.read_index(path)
        self.labels = np.load(path + ".labels.npy").tolist()


def _to_numpy(x) -> np.ndarray:
    """Helper to convert torch tensors or arrays to FAISS-compatible numpy."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)
