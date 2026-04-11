import torch
import numpy as np
from typing import List, Dict, Union

class VectorStore:
    def __init__(self, dim: int = 64, max_capacity: int = 50000):
        self.dim = dim
        self.max_capacity = max_capacity
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.memory = torch.zeros((max_capacity, dim), device=self.device, dtype=torch.float32)
        self.labels = torch.zeros(max_capacity, device=self.device, dtype=torch.long)
        self.ptr = 0
        
        # We will use this to ensure the debug statement only prints once and doesn't spam you
        self._has_printed_debug = False 

    def add(self, z: Union[np.ndarray, torch.Tensor], label: int) -> None:
        if self.ptr >= self.max_capacity:
            self.reset()
            
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z, dtype=torch.float32, device=self.device)
        else:
            z = z.to(self.device).float()
            
        self.memory[self.ptr] = z.view(-1)
        self.labels[self.ptr] = int(label)
        self.ptr += 1

    def query(self, z: Union[np.ndarray, torch.Tensor], k: int = 10) -> List[Dict]:
        # 🚨 THE SCREAMING DEBUG STATEMENT 🚨
        if not self._has_printed_debug:
            print("\n" + "🔥"*20)
            print("🚨 YES! THE PURE PYTORCH GPU CODE IS EXECUTING! 🚨")
            print(f"🚨 RUNNING ON: {self.device.type.upper()} 🚨")
            print("🔥"*20 + "\n")
            self._has_printed_debug = True

        if self.ptr == 0: 
            return []

        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z, dtype=torch.float32, device=self.device)
        else:
            z = z.to(self.device).float()

        z = z.view(1, -1)
        valid_memory = self.memory[:self.ptr]
        
        distances = torch.cdist(z, valid_memory) 
        
        k_actual = min(k, self.ptr)
        topk_dist, topk_idx = torch.topk(distances, k_actual, largest=False, dim=1)

        topk_dist = topk_dist[0].cpu().tolist()
        topk_idx = topk_idx[0].cpu().tolist()

        
        results = []
        for j in range(k_actual):
            idx = topk_idx[j]
            results.append({"label": int(self.labels[idx].item()), "dist": float(topk_dist[j])})
        return results

    def reset(self) -> None:
        self.ptr = 0
        
    def __len__(self) -> int:
        return self.ptr


    # ------------------------------------------------------------------
    # Utility (Safe PyTorch Save/Load)
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Saves the GPU memory bank safely to the hard drive."""
        torch.save({
            'memory': self.memory[:self.ptr].cpu(),
            'labels': self.labels[:self.ptr].cpu(),
            'ptr': self.ptr,
            'dim': self.dim
        }, path + ".pt")

    def load(self, path: str) -> None:
        """Loads a saved memory bank directly back into the GPU."""
        data = torch.load(path + ".pt", map_location=self.device)
        self.ptr = data['ptr']
        self.dim = data['dim']
        self.memory[:self.ptr] = data['memory'].to(self.device)
        self.labels[:self.ptr] = data['labels'].to(self.device)

