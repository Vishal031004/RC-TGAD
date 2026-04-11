"""
trainer.py — Unified Training Loop for RC-TGAD.
Refactored for Unified Processing Unit [B, N, W, 1].
Includes DataParallel Graph-Wrapper Fix to prevent silent deadlocks.
Includes AMP (Mixed Precision) and Single-Threaded GPU optimization for maximum speed.
Includes explicit Garbage Collection to prevent Kaggle RAM spikes.
"""

import os
import time
import gc
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from curriculum.scheduler import get_batch_fast, pacing

# ─────────────────────────────────────────────────────────────────────────────
# MOCK CLASSES
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
        return float(self.rng.random())

class MockTemporalGraphDataset(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        pass
    def __len__(self):
        return 0
    def __getitem__(self, idx):
        return {}

# ─────────────────────────────────────────────────────────────────────────────
# THE DATAPARALLEL SHIELD
# ─────────────────────────────────────────────────────────────────────────────
class DPGraphWrapper:
    """
    Tricks nn.DataParallel into NOT slicing the graph in half.
    Since it's a custom object, DataParallel will pass references safely to all GPUs.
    """
    def __init__(self, data):
        self.edge_index = data.edge_index
        self.edge_attr = getattr(data, 'edge_attr', None)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────────────────────────────────────
class Trainer:
    def __init__(
        self,
        backbone,
        rag_scorer,
        dataset,
        config: Dict,
        use_curriculum: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        logger=None  # 🛡️ IEEE PAPER LOGGER
    ):
        self.raw_backbone   = backbone
        self.backbone       = backbone.to(device)
        
        if device == "cuda" and torch.cuda.device_count() > 1:
            self.backbone = nn.DataParallel(self.backbone)
            print(f"[Trainer] Using {torch.cuda.device_count()} GPUs via DataParallel")
            
        self.rag_scorer     = rag_scorer
        self.dataset        = dataset
        self.config         = config
        self.use_curriculum = use_curriculum
        self.device         = device
        self.logger         = logger

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

    @torch.no_grad()
    def _compute_hardness_from_loss(self) -> np.ndarray:
        # ⚡ UPDATED PRINT STATEMENT
        print("\n[Trainer] Computing hardness scores (Single-threaded GPU)...")
        self.backbone.eval()
        ds = self.dataset
        n_timesteps = len(ds)
        N = self.raw_backbone.num_nodes
        batch_size = self.config.get("batch_size", 32) * 2 
        
        all_scores = np.zeros(n_timesteps * N, dtype=np.float32)
        
        from tqdm import tqdm
        pbar = tqdm(total=n_timesteps, desc="Hardness Eval", unit="steps")
        
        for i in range(0, n_timesteps, batch_size):
            end_i = min(i + batch_size, n_timesteps)
            batch_data = [ds[j] for j in range(i, end_i)]
            
            x = torch.stack([d["x"] for d in batch_data]).to(self.device)
            y = torch.stack([d["y"] for d in batch_data])
            graph_safe = DPGraphWrapper(batch_data[0]["graph"])
            
            z_all, x_hat_all = self.backbone(x, graph_safe) 
            
            # Target the FUTURE step
            target = torch.stack([
                torch.tensor(ds.signals[d["t"]], dtype=torch.float32) 
                for d in batch_data
            ]).unsqueeze(-1).to(self.device)

            z_all_cpu = z_all.cpu()
            x_hat_all_cpu = x_hat_all.cpu()
            target_cpu = target.cpu()

            B_real = x.shape[0]
            
            # ⚡ TURBO FIX 1: Pure PyTorch Single-Threaded Loop
            # No threads = No deadlocks. The GPU handles the speed natively.
            for b in range(B_real):
                t = batch_data[b]["t"]
                t_base_idx = (t - ds.window) // getattr(ds, 'stride', 1)
                
                for n in range(N):
                    global_idx = t_base_idx * N + n
                    
                    h_val = self.rag_scorer.score_hardness(
                        z=z_all_cpu[b, n],
                        x=target_cpu[b, n],
                        x_hat=x_hat_all_cpu[b, n],
                        node_id=n,
                        graph=batch_data[0]["graph"],
                        t=t,
                        ground_truth_label=int(y[b, n])
                    )
                    
                    all_scores[global_idx] = h_val
            
            pbar.update(B_real)
            
        pbar.close()

        score_min, score_max = all_scores.min(), all_scores.max()
        score_range = score_max - score_min
        if score_range < 1e-6:
            all_scores = np.full_like(all_scores, 0.5)
        else:
            all_scores = np.clip((all_scores - score_min) / (score_range + 1e-8), 0, 1)
            
        self.backbone.train()
        return all_scores

    def _train_epoch(self, indices, batch_size: int) -> float:
        # ⚡ TURBO FIX 2: Automatic Mixed Precision (AMP)
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        
        self.backbone.train()
        total_loss = 0.0
        n_steps = 0
        ds = self.dataset
        N = self.raw_backbone.num_nodes
        
        is_full_dataset = (len(indices) == len(ds) * N)
        
        if is_full_dataset:
            t_indices = list(range(len(ds)))
        else:
            from collections import defaultdict
            t_groups = defaultdict(list)
            for idx in indices:
                t_idx = int(idx // N)
                node_idx = int(idx % N)
                t_groups[t_idx].append(node_idx)
            t_indices = sorted(list(t_groups.keys()))
        
        for i in range(0, len(t_indices), batch_size):
            end_i = min(i + batch_size, len(t_indices))
            current_t_batch = t_indices[i:end_i]
            
            batch_data = [ds[t_idx] for t_idx in current_t_batch]
            x = torch.stack([d["x"] for d in batch_data]).to(self.device)
            graph_safe = DPGraphWrapper(batch_data[0]["graph"])
            
            self.optimizer.zero_grad()
            
            # Run forward pass in 16-bit to double GPU speed
            with autocast():
                z_all, x_hat_all = self.backbone(x, graph_safe)
                
                target = torch.stack([
                    torch.tensor(ds.signals[d["t"]], dtype=torch.float32) 
                    for d in batch_data
                ]).unsqueeze(-1).to(self.device)
                
                if is_full_dataset:
                    loss = nn.MSELoss()(x_hat_all, target)
                else:
                    loss = 0.0
                    nodes_count = 0
                    for b, t_idx in enumerate(current_t_batch):
                        selected_nodes = t_groups[t_idx]
                        if len(selected_nodes) > 0:
                            loss += nn.MSELoss()(x_hat_all[b, selected_nodes], target[b, selected_nodes])
                            nodes_count += 1
                    if nodes_count > 0:
                        loss = loss / nodes_count
            
            # Safely scale gradients back up for backward pass
            if isinstance(loss, torch.Tensor):
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), 1.0)
                scaler.step(self.optimizer)
                scaler.update()
                
                total_loss += loss.item()
                n_steps += 1

        return total_loss / max(n_steps, 1)

    @torch.no_grad()
    def _validate(self, val_dataset) -> Tuple[float, float]:
        from utils.metrics import compute_f1, compute_auc_pr
        self.backbone.eval()
        all_scores, all_labels = [], []
        
        batch_size = self.config.get("batch_size", 32)
        n_val = len(val_dataset)
        
        for i in range(0, n_val, batch_size):
            end_i = min(i + batch_size, n_val)
            batch_data = [val_dataset[j] for j in range(i, end_i)]
            
            x = torch.stack([d["x"] for d in batch_data]).to(self.device) 
            y = torch.stack([d["y"] for d in batch_data]) 
            
            graph_safe = DPGraphWrapper(batch_data[0]["graph"])
            _, x_hat = self.backbone(x, graph_safe)
            
            target = torch.stack([
                torch.tensor(val_dataset.signals[d["t"]], dtype=torch.float32) 
                for d in batch_data
            ]).unsqueeze(-1).to(self.device)
            
            node_scores = torch.norm(x_hat.view(x_hat.shape[0], x_hat.shape[1], -1) - target.view(target.shape[0], target.shape[1], -1), dim=-1)
            system_scores = node_scores.mean(dim=1)
            system_labels = y[:, 0]
            
            all_scores.extend(system_scores.cpu().tolist())
            all_labels.extend(system_labels.tolist())
            
        return compute_f1(all_scores, all_labels), compute_auc_pr(all_scores, all_labels)

    def train(self, val_dataset=None, save_dir: str = "checkpoints"):
        os.makedirs(save_dir, exist_ok=True)
        epochs = self.config.get("epochs", 100)
        k_warmup = self.config.get("k_warmup", 30)
        batch_size = self.config.get("batch_size", 32)
        n_samples = len(self.dataset) * self.raw_backbone.num_nodes 
        
        if self.use_curriculum:
            hardness_array = self._compute_hardness_from_loss()
        else:
            hardness_array = np.zeros(n_samples, dtype=np.float32)

        print("\n[Trainer] Starting training...")
        print("-" * 60)

        for epoch in range(epochs):
            t_start = time.time()
            
            if self.use_curriculum:
                indices = get_batch_fast(hardness_array, epoch, k_warmup)
                
                # 🛡️ IEEE LOGGER: Record pacing details
                if self.logger:
                    current_k = len(indices)
                    max_hardness = float(np.max(hardness_array[indices])) if current_k > 0 else 0.0
                    self.logger.log_curriculum_pacing(epoch, current_k, n_samples, max_hardness)

                if epoch > 0 and epoch % 10 == 0:
                    hardness_array = self._compute_hardness_from_loss()
            else:
                indices = np.arange(n_samples)

            train_loss = self._train_epoch(indices, batch_size)
            
            f1, auc_pr = 0.0, 0.0
            if val_dataset is not None and (epoch % 5 == 0 or epoch == epochs - 1):
                f1, auc_pr = self._validate(val_dataset)

            epoch_time = time.time() - t_start
            print(f"Epoch {epoch} | Loss: {train_loss:.4f} | F1: {f1:.4f} | Time: {epoch_time:.1f}s")

            # 🛡️ IEEE LOGGER: Record epoch metrics
            if self.logger:
                self.logger.log_epoch(epoch, train_loss, f1, epoch_time)

            # 🧹 CRASH PREVENTION: Clear memory actively before next cycle
            try:
                del indices
            except NameError:
                pass
            gc.collect()
            torch.cuda.empty_cache()

        return self.history
