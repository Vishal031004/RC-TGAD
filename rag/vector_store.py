import faiss
import numpy as np
import torch
from typing import List, Dict, Union

class VectorStore:
    def __init__(self, dim: int = 64):
        self.dim = dim
        # 1. Start with a standard CPU Index
        cpu_index = faiss.IndexFlatL2(dim)
        
        # 2. Move to ALL available GPUs
        if torch.cuda.is_available():
            try:
                # This helper automatically handles the StandardGpuResources for you
                self.index = faiss.index_cpu_to_all_gpus(cpu_index)
                print(f"[VectorStore] 🚀 FAISS-GPU Active! Using {faiss.get_num_gpus()} GPUs.")
            except Exception as e:
                print(f"[VectorStore] GPU move failed, using CPU: {e}")
                self.index = cpu_index
        else:
            self.index = cpu_index
            
        self.labels: List[int] = []            

    def add(self, z: Union[np.ndarray, 'torch.Tensor'], label: int) -> None:
        if self.index.ntotal > 50000:
            self.reset()
        z_np = _to_numpy(z).reshape(1, -1).astype("float32")
        self.index.add(z_np)
        self.labels.append(int(label))

    def query(self, z: Union[np.ndarray, 'torch.Tensor'], k: int = 10) -> List[Dict]:
        n_stored = self.index.ntotal
        if n_stored < 1: return []

        z_np = _to_numpy(z).reshape(1, -1).astype("float32")
        distances, indices = self.index.search(z_np, min(k, n_stored))

        results = []
        for j, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(self.labels): continue
            results.append({"label": self.labels[idx], "dist": float(distances[0][j])})
        return results

    def reset(self) -> None:
        self.index.reset()
        self.labels.clear()

def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"): return x.detach().cpu().numpy()
    return np.asarray(x)
