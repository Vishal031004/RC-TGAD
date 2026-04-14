"""
trainer.py — Unified Training Loop for RC-TGAD.
Includes AMP (Mixed Precision) and Node-Level Vectorized Processing.
"""

import os
import time
import gc
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from curriculum.scheduler import get_batch_fast, pacing

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

class DPGraphWrapper:
    def __init__(self, data):
        self.edge_index = data.edge_index
        self.edge_attr = getattr(data, 'edge_attr', None)

class Trainer:
    def __init__(self, backbone, rag_scorer, dataset, config: Dict, use_curriculum: bool = True, device: str = "cuda", logger=None):
        self.raw_backbone   = backbone
        self.backbone       = backbone.to(device)
        
        if device == "cuda" and torch.cuda.device_count() > 1:
            self.backbone = nn.DataParallel(self.backbone)
            
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

        self.history = {"train_loss": [], "val_f1": [], "val_auc_pr": [], "pct_data": [], "max_hardness": []}

    @torch.no_grad()
    def _compute_hardness_from_loss(self, current_epoch: int = 0) -> np.ndarray:
        print("\n[Trainer] Computing hardness scores (Node-Level GPU Batching)...")        
        self.backbone.eval()
        ds = self.dataset
        n_timesteps = len(ds)
        N = self.raw_backbone.num_nodes
        batch_size = self.config.get("batch_size", 32) * 2 
        
        all_scores = np.zeros(n_timesteps * N, dtype=np.float32)
        detailed_scores = [] 
        
        from tqdm import tqdm
        pbar = tqdm(total=n_timesteps, desc="Hardness Eval", unit="steps")
        
        try:
            for i in range(0, n_timesteps, batch_size):
                end_i = min(i + batch_size, n_timesteps)
                batch_data = [ds[j] for j in range(i, end_i)]
                
                x = torch.stack([d["x"] for d in batch_data]).to(self.device)
                y = torch.stack([d["y"] for d in batch_data]).to(self.device)
                graph_safe = DPGraphWrapper(batch_data[0]["graph"])
                
                z_all, x_hat_all = self.backbone(x, graph_safe) 
                
                target = torch.stack([
                    torch.tensor(ds.signals[d["t"]], dtype=torch.float32) 
                    for d in batch_data
                ]).unsqueeze(-1).to(self.device)

                rag_cfg = self.config.get("rag", {})
                a1 = rag_cfg.get("alpha_1", 0.33)
                a2 = rag_cfg.get("alpha_2", 0.33)
                a3 = rag_cfg.get("alpha_3", 0.34)
                
                h_results = self.rag_scorer.score_hardness(
                    z=z_all, x=target, x_hat=x_hat_all, y=y,
                    batch_data=batch_data, alphas=(a1, a2, a3)
                )
                
                # ⚡ Safely pull the distinct score for EVERY node
                for b_idx, node_res_list in enumerate(h_results):
                    t = batch_data[b_idx]["t"]
                    t_base_idx = (t - ds.window) // getattr(ds, 'stride', 1)
                    
                    for n in range(N):
                        global_idx = t_base_idx * N + n
                        res = node_res_list[n]
                        
                        all_scores[global_idx] = res["total"]
                        
                        if self.logger:
                            detailed_scores.append([
                                current_epoch, global_idx, 
                                round(res["temp"], 4), round(res["struct"], 4), round(res["rag"], 4), round(res["total"], 4)
                            ])
                
                pbar.update(x.shape[0])
                
        except KeyboardInterrupt:
            print("\n🛑 KAGGLE STOP BUTTON DETECTED! Salvaging data...")
            
        finally:
            pbar.close()
            if self.logger and detailed_scores:
                self.logger.log_curriculum_scores(detailed_scores)
                print(f"✅ SUCCESSFULLY SAVED {len(detailed_scores)} ROWS TO CSV!")
            
            if len(detailed_scores) < (n_timesteps * N * 0.9):
                raise RuntimeError("Pipeline safely halted for user inspection.")

        score_min, score_max = all_scores.min(), all_scores.max()
        score_range = score_max - score_min
        if score_range < 1e-6:
            all_scores = np.full_like(all_scores, 0.5)
        else:
            all_scores = np.clip((all_scores - score_min) / (score_range + 1e-8), 0, 1)
            
        self.backbone.train()
        return all_scores

    def _train_epoch(self, indices, batch_size: int) -> float:
        from torch.amp import autocast, GradScaler
        scaler = GradScaler('cuda')
        
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
            
            with autocast('cuda'):
                z_all, x_hat_all = self.backbone(x, graph_safe)
                
                target = torch.stack([
                    torch.tensor(ds.signals[d["t"]], dtype=torch.float32) 
                    for d in batch_data
                ]).unsqueeze(-1).to(self.device)
                
                if is_full_dataset:
                    loss = nn.MSELoss()(x_hat_all, target)
                else:
                    B_curr = len(current_t_batch)
                    mask = torch.zeros((B_curr, N), dtype=torch.bool)
                    for b, t_idx in enumerate(current_t_batch):
                        mask[b, t_groups[t_idx]] = True
                        
                    mask = mask.to(self.device)
                    valid_x_hat = x_hat_all[mask]
                    valid_target = target[mask]
                    
                    if valid_x_hat.numel() > 0:
                        loss = nn.MSELoss()(valid_x_hat, valid_target)
                    else:
                        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
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
            hardness_array = self._compute_hardness_from_loss(0)
        else:
            hardness_array = np.zeros(n_samples, dtype=np.float32)

        print("\n[Trainer] Starting training...")
        print("-" * 60)

        for epoch in range(epochs):
            t_start = time.time()
            max_h = 1.0 
            
            if self.use_curriculum:
                indices = get_batch_fast(hardness_array, epoch, k_warmup)
                current_k = len(indices)
                max_h = float(np.max(hardness_array[indices])) if current_k > 0 else 0.0
                
                if self.logger:
                    self.logger.log_curriculum_pacing(epoch, current_k, n_samples, max_h)

                if epoch > 0 and epoch % 10 == 0:
                    hardness_array = self._compute_hardness_from_loss(epoch)
            else:
                indices = np.arange(n_samples)

            train_loss = self._train_epoch(indices, batch_size)
            
            f1, auc_pr = 0.0, 0.0
            if val_dataset is not None and (epoch % 5 == 0 or epoch == epochs - 1):
                f1, auc_pr = self._validate(val_dataset)

            self.history["train_loss"].append(train_loss)
            self.history["val_f1"].append(f1)
            self.history["val_auc_pr"].append(auc_pr)
            self.history["pct_data"].append((len(indices) / n_samples) * 100)
            self.history["max_hardness"].append(max_h)

            epoch_time = time.time() - t_start
            print(f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | Val AUC-PR: {auc_pr:.4f} | Max Hardness: {max_h:.4f} | Time: {epoch_time:.1f}s")

            if self.logger:
                self.logger.log_epoch(epoch, train_loss, auc_pr, max_h, epoch_time)

            try:
                del indices
            except NameError:
                pass
            gc.collect()
            torch.cuda.empty_cache()

        if self.logger:
            self.logger.verify_disk_writes()

        return self.history
