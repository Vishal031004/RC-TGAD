"""
trainer.py — Unified Training Loop for RC-TGAD.
Refactored for Unified Processing Unit [B, N, W, 1].
Includes Indexing Mapping and Scale Collapse Safeguard.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional
from curriculum.scheduler import get_batch_fast, pacing

# ─────────────────────────────────────────────────────────────────────────────
# MOCK CLASSES (Required by ablations.py imports)
# ─────────────────────────────────────────────────────────────────────────────

class MockBackbone(nn.Module):
    def __init__(self, d_in: int = 10, d_z: int = 64, num_nodes: int = 10):
        super().__init__()
        self.num_nodes = num_nodes

    def forward(self, x_windows, graph=None):
        pass

class MockRAGScorer:
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def score_hardness(self, *args, **kwargs) -> float:
        # Accepts any arguments (including the new ground_truth_label) and returns a dummy score
        return float(self.rng.random())

class MockTemporalGraphDataset(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        pass
    def __len__(self):
        return 0
    def __getitem__(self, idx):
        return {}

class Trainer:
    def __init__(
        self,
        backbone,
        rag_scorer,
        dataset,
        config: Dict,
        use_curriculum: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.raw_backbone   = backbone
        self.backbone       = backbone.to(device)
        
        # Multi-GPU support
        if device == "cuda" and torch.cuda.device_count() > 1:
            self.backbone = nn.DataParallel(self.backbone)
            print(f"[Trainer] Using {torch.cuda.device_count()} GPUs via DataParallel")
            
        self.rag_scorer     = rag_scorer
        self.dataset        = dataset
        self.config         = config
        self.use_curriculum = use_curriculum
        self.device         = device

        self.optimizer = torch.optim.Adam(
            self.backbone.parameters(),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 1e-5)
        )

        self.history = {
            "train_loss": [],
            "val_f1":     [],
            "val_auc_pr": [],
            "pct_data":   [],
        }

        print(f"[Trainer] Device    : {self.device}")
        print(f"[Trainer] Curriculum: {'ON' if use_curriculum else 'OFF'}")
        print(f"[Trainer] Timesteps : {len(dataset)}")

    @torch.no_grad()
    def _compute_hardness_from_loss(self) -> np.ndarray:
        """
        Computes hardness for all samples in the dataset.
        Now operates on snapshots [N, W, 1].
        """
        print("[Trainer] Computing hardness scores...")
        self.backbone.eval()
        ds = self.dataset
        n_timesteps = len(ds)
        N = self.raw_backbone.num_nodes
        batch_size = self.config.get("batch_size", 32)
        
        # We store scores as a flat array of [n_timesteps * N] to match scheduler expectations
        all_scores = np.zeros(n_timesteps * N, dtype=np.float32)
        
        for i in range(0, n_timesteps, batch_size):
            end_i = min(i + batch_size, n_timesteps)
            batch_data = [ds[j] for j in range(i, end_i)]
            
            # Stack into [B, N, W, 1]
            x = torch.stack([d["x"] for d in batch_data]).to(self.device)
            y = torch.stack([d["y"] for d in batch_data]) # [B, N]
            graph = batch_data[0]["graph"]
            
            z_all, x_hat_all = self.backbone(x, graph) 
            # z_all: [B, N, d_z], x_hat_all: [B, N, d_in]
            
            # Reconstruction target is the last value of each window: [B, N, 1]
            target = x[:, :, -1, :] 

            # Loop through batch to score via RAG
            B_real = x.shape[0]
            for b in range(B_real):
                t = batch_data[b]["t"]
                # 🛡️ FIX: Explicit Mapping. No longer assumes sequential indices.
                t_base_idx = (t - ds.window) // getattr(ds, 'stride', 1)
                
                for n in range(N):
                    global_idx = t_base_idx * N + n
                    
                    h = self.rag_scorer.score_hardness(
                        z=z_all[b, n],
                        x=target[b, n],
                        x_hat=x_hat_all[b, n],
                        node_id=n,
                        graph=graph,
                        t=t,
                        ground_truth_label=int(y[b, n])
                    )
                    all_scores[global_idx] = h

        # 🛡️ FIX: Safeguard against Scale Collapse
        score_min, score_max = all_scores.min(), all_scores.max()
        score_range = score_max - score_min
        if score_range < 1e-6:
            print("[Trainer] Hardness collapsed (Range < 1e-6). Using neutral 0.5.")
            all_scores = np.full_like(all_scores, 0.5)
        else:
            all_scores = np.clip((all_scores - score_min) / (score_range + 1e-8), 0, 1)
            
        self.backbone.train()
        return all_scores

    def _train_epoch(self, indices, batch_size: int) -> float:
        """
        Trains on a subset of (node, t) samples provided by the scheduler.
        """
        self.backbone.train()
        total_loss = 0.0
        n_steps = 0
        ds = self.dataset
        N = self.raw_backbone.num_nodes
        
        # Group selected indices by timestep for efficient GNN processing
        from collections import defaultdict
        t_groups = defaultdict(list)
        for idx in indices:
            t_idx = idx // N
            node_idx = idx % N
            t_groups[t_idx].append(node_idx)
            
        t_indices = sorted(list(t_groups.keys()))
        
        for i in range(0, len(t_indices), batch_size):
            end_i = min(i + batch_size, len(t_indices))
            current_t_batch = t_indices[i:end_i]
            
            # 🛡️ Unified Unit: We MUST load the full snapshot for the GNN
            batch_data = [ds[t_idx] for t_idx in current_t_batch]
            x = torch.stack([d["x"] for d in batch_data]).to(self.device)
            graph = batch_data[0]["graph"]
            
            self.optimizer.zero_grad()
            
            # Forward pass: [B, N, d_z], [B, N, d_in]
            z_all, x_hat_all = self.backbone(x, graph)
            target = x[:, :, -1, :] # Last value in window
            
            # 🛡️ Curriculum Masking: Only calculate loss on nodes selected by scheduler
            loss = 0.0
            nodes_count = 0
            for b, t_idx in enumerate(current_t_batch):
                selected_nodes = t_groups[t_idx]
                if len(selected_nodes) > 0:
                    loss += nn.MSELoss()(x_hat_all[b, selected_nodes], target[b, selected_nodes])
                    nodes_count += 1
            
            if nodes_count > 0:
                loss = loss / nodes_count
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item()
                n_steps += 1

        return total_loss / max(n_steps, 1)

    @torch.no_grad()
    def _validate(self, val_dataset) -> Tuple[float, float]:
        from utils.metrics import compute_f1, compute_auc_pr
        self.backbone.eval()
        
        all_scores, all_labels = [], []
        
        for i in range(len(val_dataset)):
            data = val_dataset[i]
            x = data["x"].unsqueeze(0).to(self.device) # [1, N, W, 1]
            y = data["y"] # [N]
            
            _, x_hat = self.backbone(x, data["graph"])
            target = x[:, :, -1, :]
            
            # Error per node
            score = torch.norm(x_hat.squeeze(0) - target.squeeze(0), dim=-1)
            all_scores.extend(score.cpu().tolist())
            all_labels.extend(y.tolist())
            
        return compute_f1(all_scores, all_labels), compute_auc_pr(all_scores, all_labels)

    def train(self, val_dataset=None, save_dir: str = "checkpoints"):
        os.makedirs(save_dir, exist_ok=True)
        epochs = self.config.get("epochs", 100)
        k_warmup = self.config.get("k_warmup", 30)
        batch_size = self.config.get("batch_size", 32)
        n_samples = len(self.dataset) * self.raw_backbone.num_nodes # Total (node, t) pairs
        
        if self.use_curriculum:
            hardness_array = self._compute_hardness_from_loss()
        else:
            hardness_array = np.zeros(n_samples, dtype=np.float32)

        for epoch in range(epochs):
            t_start = time.time()
            
            if self.use_curriculum:
                indices = get_batch_fast(hardness_array, epoch, k_warmup)
                if epoch > 0 and epoch % 10 == 0:
                    hardness_array = self._compute_hardness_from_loss()
            else:
                indices = np.arange(n_samples)

            train_loss = self._train_epoch(indices, batch_size)
            
            f1, auc_pr = 0.0, 0.0
            if val_dataset is not None and (epoch % 5 == 0 or epoch == epochs - 1):
                f1, auc_pr = self._validate(val_dataset)
                # Checkpointing logic here...

            print(f"Epoch {epoch} | Loss: {train_loss:.4f} | F1: {f1:.4f} | Time: {time.time()-t_start:.1f}s")

        return self.history
