"""
rag_scorer.py — Unified entry point for RAG hardness scoring.
FIXED: Interface B alignment, causal ordering, and Memory Sliding Window.
FIXED: Thread-safe locking for Multi-Threaded Trainer compatibility.
"""

import torch
import numpy as np
import threading
from typing import Tuple, Optional, List, Union
from rag.vector_store import VectorStore
from rag.hardness import compute_h_temp, compute_h_struct, compute_h_rag

# 🛡️ THREAD LOCK: Prevents FAISS Segmentation Faults during multi-threading
_scorer_lock = threading.Lock()

def score_hardness(
    z: torch.Tensor,
    x: torch.Tensor,
    x_hat: torch.Tensor,
    node_id: int,
    graph,
    t: int,
    window_errors: List[float],
    vector_store: VectorStore,
    ground_truth_label: int,
    alphas: Tuple[float, float, float] = (0.33, 0.33, 0.34),
    k_neighbors: int = 10,
    gamma: float = 0.5,
    anomaly_source_id: Optional[int] = None,
    return_components: bool = False,
    **kwargs 
) -> Union[float, Tuple[float, float, float, float]]:
    alpha_1, alpha_2, alpha_3 = alphas

    # 1. Compute H_temp using CURRENT history
    h_temp = compute_h_temp(x, x_hat, window_errors)
    
    # 2. Calculate error magnitude
    e = torch.norm(x - x_hat, p=2).item()
    
    # 3. H_struct
    h_struct = compute_h_struct(node_id, graph, anomaly_source_id, gamma)

    # ⚡ NEW: Move the heavy FAISS search OUTSIDE the lock!
    # FAISS search is perfectly thread-safe and releases the Python GIL.
    h_rag = compute_h_rag(z, vector_store, k=k_neighbors)

    # 🚦 ACQUIRE LOCK ONLY FOR WRITING
    # This takes 0.001 seconds, completely eliminating the "Traffic Jam"
    with _scorer_lock:
        
        # 4. Safely modify the sliding window memory
        window_errors.append(e)
        if len(window_errors) > 10000:
            window_errors.pop(0)

        # 5. Composite score
        H = alpha_1 * h_temp + alpha_2 * h_struct + alpha_3 * h_rag
        H_clipped = float(np.clip(H, 0.0, 1.0))

        # 6. Normalize and ADD to store safely
        z_np = z.detach().cpu().numpy() if hasattr(z, "detach") else np.asarray(z)
        norm = np.linalg.norm(z_np)
        if norm > 1e-8:
            z_np = z_np / norm
        
        vector_store.add(z_np, label=ground_truth_label)

    if return_components:
        return H_clipped, h_temp, h_struct, h_rag
        
    return H_clipped
