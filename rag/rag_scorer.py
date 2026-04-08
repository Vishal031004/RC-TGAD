"""
rag_scorer.py — Unified entry point for RAG hardness scoring.
FIXED: Interface B alignment and causal ordering.
"""

import torch
import numpy as np
from typing import Tuple, Optional, List
from rag.vector_store import VectorStore
from rag.hardness import compute_h_temp, compute_h_struct, compute_h_rag

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
    **kwargs # Accept extra args from Trainer gracefully
) -> float:
    alpha_1, alpha_2, alpha_3 = alphas

    # 1. Compute H_temp
    h_temp = compute_h_temp(x, x_hat, window_errors)
    
    # 2. Append error AFTER calculating h_temp to prevent lookahead bias
    e = torch.norm(x - x_hat, p=2).item()
    window_errors.append(e)

    # 3. H_struct
    h_struct = compute_h_struct(node_id, graph, anomaly_source_id, gamma)

    # 4. H_RAG (Normalizes internally)
    h_rag = compute_h_rag(z, vector_store, k=k_neighbors)

    # 5. Composite score
    H = alpha_1 * h_temp + alpha_2 * h_struct + alpha_3 * h_rag

    # 6. Normalize and ADD to store for future retrievals
    z_np = z.detach().cpu().numpy() if hasattr(z, "detach") else np.asarray(z)
    norm = np.linalg.norm(z_np)
    if norm > 1e-8:
        z_np = z_np / norm
    
    vector_store.add(z_np, label=ground_truth_label)

    return float(np.clip(H, 0.0, 1.0))
