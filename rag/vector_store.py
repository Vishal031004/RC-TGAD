"""
vector_store.py — Pure PyTorch VRAM Vector Store.
Bypasses FAISS entirely to eliminate CPU-GPU transfer bottlenecks.
"""
import torch
import numpy as np
from typing import List, Dict, Union

class VectorStore:
    def __init__(self, dim: int = 64, max_capacity: int = 50000):
        self.dim = dim
        self.max_capacity = max_capacity
        # Force CUDA if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n[VectorStore] 🚀 PURE PYTORCH VRAM BANK INITIALIZED ON {self.device.type.upper()}! 🚀\n")
        
        # Pre-allocate memory directly on the GPU
        self.memory = torch.zeros((max_capacity, dim), device=self.device, dtype=torch.float32)
        self.labels = torch.zeros(max_capacity, device=self.device, dtype=torch.long)
        self.ptr = 0

    def add(self, z: Union[np.ndarray, torch.Tensor], label: int) -> None:
        """Add one embedding to the GPU store."""
        if self.ptr >= self.max_capacity:
            self.reset()
            
        # Ensure input is a tensor on the correct device
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z, dtype=torch.float32, device=self.device)
        else:
            z = z.to(self.device).float()
            
        self.memory[self.ptr] = z.view(-1)
        self.labels[self.ptr] = int(label)
        self.ptr += 1

    def query(self, z: Union[np.ndarray, torch.Tensor], k: int = 10) -> List[Dict]:
        """Retrieve k nearest neighbors using native GPU math."""
        if self.ptr == 0: 
            return []

        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z, dtype=torch.float32, device=self.device)
        else:
            z = z.to(self.device).float()

        z = z.view(1, -1)
        
        # Slice only the valid memory
        valid_memory = self.memory[:self.ptr]
        
        # Native PyTorch Euclidean distance (Runs instantly on GPU cores)
        distances = torch.cdist(z, valid_memory) 
        
        k_actual = min(k, self.ptr)
        topk_dist, topk_idx = torch.topk(distances, k_actual, largest=False, dim=1)

        # Move only the final top-k answers to CPU for the logger
        topk_dist = topk_dist[0].cpu().tolist()
        topk_idx = topk_idx[0].cpu().tolist()
        
        results = []
        for j in range(k_actual):
            idx = topk_idx[j]
            results.append({
                "label": int(self.labels[idx].item()), 
                "dist": float(topk_dist[j])
            })
        return results

    def reset(self) -> None:
        """O(1) instant reset."""
        self.ptr = 0
        
    def __len__(self) -> int:
        return self.ptr
