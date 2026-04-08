"""
hardness.py — Three-component hardness scoring for RC-TGAD (Person 2)

Components:
  H_temp  — temporal / reconstruction-error hardness
  H_struct — structural / graph-topology hardness
  H_RAG   — retrieval-entropy hardness (the paper's core novelty)

All three return float in [0, 1].
High score = HARD sample.
"""

import numpy as np
import networkx as nx
import torch
from typing import Optional, List


# ======================================================================
# Component 1 — H_temp
# ======================================================================

def compute_h_temp(x, x_hat, window_errors, eps=1e-8):
    x_flat = x.reshape(-1).float()
    x_hat_flat = x_hat.reshape(-1).float()
    
    e = torch.norm(x_flat - x_hat_flat, p=2).item()
    
    if not window_errors:
        return 0.5
        
    # FIX: Use percentiles instead of absolute min/max for stability
    e_min = np.percentile(window_errors, 5)
    e_max = np.percentile(window_errors, 95)
    
    # Invert: low error relative to history -> high hardness (subtle)
    # Ensure clipping so the result stays in [0, 1]
    h_temp = 1.0 - (e - e_min) / (e_max - e_min + eps)
    return float(np.clip(h_temp, 0.0, 1.0))


# ======================================================================
# Component 2 — H_struct
# ======================================================================

def compute_h_struct(
    node_id: int,
    graph,                          # torch_geometric Data object
    anomaly_source_id: Optional[int] = None,
    gamma: float = 0.5,
) -> float:
    """
    Structural hardness based on graph topology.

    Two sub-terms:
      • centrality_term: low-degree nodes are harder (peripheral, less context).
      • depth_term:      nodes far from the anomaly source are harder
                         (propagation effect is diluted).

    Args:
        node_id:          Index of the node being scored.
        graph:            torch_geometric Data object.
                          Must have .edge_index and (optionally) .num_nodes.
        anomaly_source_id: The node where the anomaly originates, if known.
                           Pass None to use default depth_term = 0.5.
        gamma:            Weight between depth_term and centrality_term.
                          Default 0.5 per paper.

    Returns:
        float in [0, 1].
    """
    # Build a NetworkX graph for topology queries
    G = _pyg_to_nx(graph)

    # --- centrality term ---
    degrees = dict(G.degree())
    deg = degrees.get(node_id, 0)
    max_deg = max(degrees.values()) if degrees else 1
    centrality_term = 1.0 - (deg / max_deg)   # low degree → harder

    # --- depth term ---
    if anomaly_source_id is not None and G.has_node(anomaly_source_id):
        try:
            dist = nx.shortest_path_length(G, node_id, anomaly_source_id)
            diam = nx.diameter(G) if nx.is_connected(G) else _pseudo_diameter(G)
            diam = max(diam, 1)   # guard against diameter == 0
            depth_term = dist / diam
        except nx.NetworkXNoPath:
            depth_term = 1.0     # unreachable → maximally hard
    else:
        depth_term = 0.5         # unknown source → neutral prior

    h_struct = gamma * depth_term + (1 - gamma) * centrality_term
    return float(np.clip(h_struct, 0.0, 1.0))


# ======================================================================
# Component 3 — H_RAG  (THE NOVELTY)
# ======================================================================

def compute_h_rag(z, vector_store, k: int = 10) -> float:
    """
    Retrieval-entropy hardness.

    Retrieves k nearest neighbors from the FAISS store and measures
    label entropy among them.

    High entropy (p_hat ≈ 0.5) → mixed neighborhood → HARD to classify.
    Low entropy  (p_hat ≈ 0 or 1) → clean neighborhood → EASY.

    Args:
        z:            Query embedding — Tensor [d_z] or np.ndarray [d_z].
        vector_store: VectorStore instance (from vector_store.py).
        k:            Number of neighbors to retrieve.

    Returns:
        float in [0, 1].   Returns 0.0 if store is empty.
    """
    neighbors = vector_store.query(z, k=k)

    if not neighbors:
        return 0.0   # empty store → no information → treat as easy

    labels = [n["label"] for n in neighbors]
    p_hat = sum(labels) / len(labels)   # proportion of anomalies

    # Degenerate cases: zero entropy → easy sample
    if p_hat == 0.0 or p_hat == 1.0:
        return 0.0

    # Binary Shannon entropy (max = 1.0 when p_hat = 0.5)
    entropy = -p_hat * np.log2(p_hat) - (1 - p_hat) * np.log2(1 - p_hat)
    return float(np.clip(entropy, 0.0, 1.0))


# ======================================================================
# Internal helpers
# ======================================================================

def _pyg_to_nx(graph) -> nx.Graph:
    """Convert torch_geometric Data.edge_index to an undirected NetworkX graph."""
    G = nx.Graph()
    num_nodes = graph.num_nodes if hasattr(graph, "num_nodes") and graph.num_nodes is not None \
                else int(graph.edge_index.max().item()) + 1
    G.add_nodes_from(range(num_nodes))

    edge_index = graph.edge_index.cpu().numpy()
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    G.add_edges_from(edges)
    return G


def _pseudo_diameter(G: nx.Graph) -> int:
    """Return approximate diameter for disconnected graphs (max over components)."""
    best = 1
    for component in nx.connected_components(G):
        sub = G.subgraph(component)
        if len(sub) > 1:
            best = max(best, nx.diameter(sub))
    return best
