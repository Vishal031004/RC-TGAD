"""
backbone.py — Unified entry point for LSTM + GNN backbone.
Includes Critical Fixes:
1. Unified Processing Unit (Process [B, N, W, 1] simultaneously).
2. Proper PyG graph disjoint batching for GNN.
3. Fixed Interface A to prevent isolated node embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data

from .lstm_encoder import LSTMEncoder
from .gnn_reasoner import GNNReasoner


# ---------------------------------------------------------------------------
# Backbone module
# ---------------------------------------------------------------------------

class Backbone(nn.Module):
    """
    Combines LSTMEncoder (per-node) and GNNReasoner (cross-node).

    Dimensions (must be agreed with Person 2 on Day 1):
        LSTM hidden_size : 64
        GNN out_dim      : 64   ← FAISS index dim
    """

    def __init__(
        self,
        d_in        : int = 1,
        hidden_size : int = 64,
        gnn_out_dim : int = 64,
        num_nodes   : int = 51,
        window_size : int = 30,
        lstm_layers : int = 2,
        gat_heads   : int = 4,
        dropout     : float = 0.1,
    ):
        super().__init__()
        self.num_nodes   = num_nodes
        self.hidden_size = hidden_size
        self.gnn_out_dim = gnn_out_dim

        self.lstm = LSTMEncoder(
            d_in        = d_in,
            hidden_size = hidden_size,
            num_layers  = lstm_layers,
            dropout     = dropout,
        )

        self.gnn = GNNReasoner(
            in_dim  = hidden_size,
            out_dim = gnn_out_dim,
            heads   = gat_heads,
            dropout = dropout,
        )

    # ------------------------------------------------------------------
    def forward(self, x_windows: torch.Tensor,
                graph: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process a full batch of node windows.
        Supports single snapshot [N, W, d_in] or batched snapshots [B, N, W, d_in].
        """
        is_batched = (x_windows.dim() == 4)
        if is_batched:
            B, N_dim, W, d_in = x_windows.shape
            # 🛡️ Flatten batch and nodes for the LSTM
            x_flat = x_windows.view(B * N_dim, W, d_in)
        else:
            B = 1
            N_dim, W, d_in = x_windows.shape
            x_flat = x_windows

        # 1. TEMPORAL PHASE (LSTM) - Independent sequences
        h_all, x_hat_all = self.lstm(x_flat)    # [B*N, hidden], [B*N, d_in]

        # 2. SPATIAL PHASE (GNN) - Cross-node reasoning
        device = x_windows.device
        base_edge_index = graph.edge_index.to(device)
        
        if is_batched:
            # 🛡️ FIX: Create disjoint graphs for PyTorch Geometric
            # Example: Node 0 in Batch 1 becomes Node 51. Node 1 becomes Node 52.
            E = base_edge_index.shape[1]
            edge_index = base_edge_index.repeat(1, B)
            
            # Create offsets [0, 0, ..., 51, 51, ..., 102, 102, ...]
            offsets = torch.arange(B, device=device).repeat_interleave(E) * N_dim
            edge_index = edge_index + offsets
            
            edge_attr = graph.edge_attr.to(device).repeat(B, 1) if graph.edge_attr is not None else None
        else:
            edge_index = base_edge_index
            edge_attr = graph.edge_attr.to(device) if graph.edge_attr is not None else None

        # Pass through GNN
        z_all = self.gnn(h_all, edge_index, edge_attr)   # [B*N, gnn_out_dim]

        # Reshape back to batched format if needed
        if is_batched:
            z_all = z_all.view(B, N_dim, -1)
            x_hat_all = x_hat_all.view(B, N_dim, -1)

        return z_all, x_hat_all

    # ------------------------------------------------------------------
    def reconstruction_loss(self, x_windows: torch.Tensor,
                             x_hat_all: torch.Tensor) -> torch.Tensor:
        """MSE loss between last timestep and reconstruction for all nodes."""
        # Target is the last timestep: [B, N, d_in]
        if x_windows.dim() == 4:
            target = x_windows[:, :, -1, :] 
        else:
            target = x_windows[:, -1, :]
            
        return nn.MSELoss()(x_hat_all, target)


# ---------------------------------------------------------------------------
# Interface A — fixed contract for Person 2
# ---------------------------------------------------------------------------

_global_backbone: Backbone | None = None


def load_backbone(checkpoint_path: str, device: str = "cpu",
                  **kwargs) -> Backbone:
    """Load a trained backbone from a checkpoint file."""
    global _global_backbone
    model = Backbone(**kwargs)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    _global_backbone = model
    return model


@torch.no_grad()
def get_embedding(
    x_windows : torch.Tensor,
    graph     : Data,
    node_id   : int,
    t         : int,
    backbone  : Backbone | None = None,
    device    : str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Interface A — called by Person 2 (RAG scorer).
    
    🛡️ FIX: Now requires the full plant snapshot [N, W, d_in] to ensure 
    the GNN can perform message passing across neighbors. It computes the 
    whole graph, but returns only the requested node's data.

    Args:
        x_windows : Tensor [N, W, d_in]  ALL windows for time t
        graph     : torch_geometric Data object (static, shared)
        node_id   : int                  node index v to extract
        t         : int                  timestep (informational)
        backbone  : Backbone | None      if None, uses global loaded backbone
        device    : str

    Returns:
        z     : Tensor [d_z]   GNN-enhanced embedding for node v at t
        x_hat : Tensor [d_in]  LSTM reconstructed signal for node v at t
    """
    model = backbone or _global_backbone
    if model is None:
        raise RuntimeError(
            "No backbone available. Call load_backbone() first or pass backbone=.")

    model = model.to(device).eval()
    x_windows = x_windows.to(device)

    # Perform full spatial-temporal reasoning for time t
    z_all, x_hat_all = model(x_windows, graph)

    # Extract strictly the node we care about for RAG scoring
    z     = z_all[node_id].cpu()     # [d_z]
    x_hat = x_hat_all[node_id].cpu() # [d_in]

    return z, x_hat
