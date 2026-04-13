import numpy as np
import networkx as nx
import torch
from typing import Optional, List

# ======================================================================
# GLOBAL GRAPH CACHE (The 50x Speed Fix)
# ======================================================================
_CACHED_GRAPH = None

def _pyg_to_nx(graph):
    global _CACHED_GRAPH
    if _CACHED_GRAPH is not None:
        return _CACHED_GRAPH
        
    G = nx.Graph()
    if hasattr(graph, 'edge_index') and graph.edge_index is not None:
        edges = graph.edge_index.cpu().numpy().T
        G.add_edges_from(edges)
        
    # Ensure all nodes are added even if they are isolated/disconnected
    if hasattr(graph, 'num_nodes'):
        G.add_nodes_from(range(graph.num_nodes))
        
    _CACHED_GRAPH = G
    return G

# ======================================================================
# Component 1 — H_temp (SOTA Percentile Mapping)
# ======================================================================
def compute_h_temp(x, x_hat, window_errors=None, eps=1e-8):
    x_flat = x.reshape(-1).float()
    x_hat_flat = x_hat.reshape(-1).float()
    
    e = torch.norm(x_flat - x_hat_flat, p=2).item()
    
    if not window_errors:
        return 0.5
        
    e_min = np.percentile(window_errors, 5)
    e_max = np.percentile(window_errors, 95)
    
    # Direct mapping: Low error -> Low Hardness (Easy)
    h_temp = (e - e_min) / (e_max - e_min + eps)
    return float(np.clip(h_temp, 0.0, 1.0))

# ======================================================================
# Component 2 — H_struct (SOTA Centrality + Dynamic Normalization)
# ======================================================================
def compute_h_struct(node_id, graph, anomaly_source_id=0, gamma=0.5):
    G = _pyg_to_nx(graph)
    
    try:
        # ⚡ RESTORED SOTA: Degree Centrality Term
        degrees = dict(G.degree())
        deg = degrees.get(node_id, 0)
        max_deg = max(degrees.values()) if degrees else 1
        centrality_term = 1.0 - (deg / max_deg) 
        
        # ⚡ NEW: Dynamic Depth Term (Bypasses NetworkX bug safely)
        if nx.has_path(G, node_id, anomaly_source_id):
            path_lengths = nx.single_source_shortest_path_length(G, anomaly_source_id)
            actual_max_dist = max(path_lengths.values()) if path_lengths else 1
            dist = path_lengths.get(node_id, 0)
            depth_term = dist / actual_max_dist
        else:
            depth_term = 1.0 
            
        # Blend exactly like the SOTA baseline
        h_struct = gamma * depth_term + (1 - gamma) * centrality_term
        return float(np.clip(h_struct, 0.0, 1.0))
        
    except Exception as e:
        print(f"Graph Math Error: {e}")
        return 0.6667

# ======================================================================
# Component 3 — H_RAG (SOTA Fallback Interface)
# ======================================================================
def compute_h_rag(z, vector_store, k: int = 10) -> float:
    # L2 Normalize embedding before FAISS query
    z_np = z.detach().cpu().numpy() if hasattr(z, "detach") else np.asarray(z)
    norm = np.linalg.norm(z_np)
    if norm > 1e-8:
        z_np = z_np / norm

    neighbors = vector_store.query(z_np, k=k)
    if not neighbors:
        return 0.0

    labels = [n["label"] for n in neighbors]
    p_hat = sum(labels) / len(labels)

    if p_hat == 0.0 or p_hat == 1.0:
        return 0.0

    entropy = -p_hat * np.log2(p_hat) - (1 - p_hat) * np.log2(1 - p_hat)
    return float(np.clip(entropy, 0.0, 1.0))
