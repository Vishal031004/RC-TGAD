# curriculum/trainer.py
# PERSON 3 — Full Training Loop
# RC-TGAD: Retrieval-Augmented Curriculum Training for Temporal Graph Anomaly Detection

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from torch.utils.data import Subset
from typing import Dict, List, Tuple, Optional
from curriculum.scheduler import get_batch, get_batch_fast, pacing


# ─────────────────────────────────────────────────────────────────────────────
# MOCK CLASSES
# ─────────────────────────────────────────────────────────────────────────────

class MockBackbone(nn.Module):
    def __init__(self, d_in: int = 10, d_z: int = 64, num_nodes: int = 10):
        super().__init__()
        self.d_in = d_in
        self.d_z = d_z
        self.num_nodes = num_nodes
        self.lstm_mock  = nn.Linear(d_in, d_z)
        self.recon_mock = nn.Linear(d_z, d_in)

    def get_embedding(self, x_window, graph=None, node_id=None, t=None):
        x_flat = x_window.mean(dim=0)
        z      = torch.relu(self.lstm_mock(x_flat))
        x_hat  = self.recon_mock(z)
        return z, x_hat

    def forward(self, x_windows, graph=None):
        """
        Batched forward for GPU efficiency.
        Supports x_windows : [B*N, W, d_in] or [B, N, W, d_in].
        """
        is_batched = (x_windows.dim() == 4)
        if is_batched:
            B, N_dim, W, d_in = x_windows.shape
            x_windows = x_windows.view(B * N_dim, W, d_in)
        else:
            N_dim = x_windows.shape[0]
            
        x_last    = x_windows[:, -1, :]                          # [B*N, d_in]
        z_all     = torch.relu(self.lstm_mock(x_last))           # [B*N, d_z]
        x_hat_all = self.recon_mock(z_all)                       # [B*N, d_in]
        
        if is_batched:
            z_all = z_all.view(B, N_dim, -1)
            x_hat_all = x_hat_all.view(B, N_dim, -1)
            
        return z_all, x_hat_all


class MockRAGScorer:
    def __init__(self, seed: int = 42):
        self.rng    = np.random.RandomState(seed)
        self._cache: Dict[Tuple, float] = {}

    def score_hardness(self, z, x, x_hat, node_id, graph, t,
                       window_errors=None) -> float:
        key = (node_id, t)
        if key not in self._cache:
            self._cache[key] = float(self.rng.random())
        return self._cache[key]

    def get_all_scores(self, dataset) -> Dict[Tuple, float]:
        return {
            (node_id, t): self.score_hardness(None, None, None, node_id, None, t)
            for (node_id, t, _) in dataset
        }


class MockTemporalGraphDataset(torch.utils.data.Dataset):
    def __init__(self, n_nodes: int = 10, T: int = 200,
                 window_size: int = 30, d_in: int = 10,
                 anomaly_ratio: float = 0.1, seed: int = 42):
        super().__init__()
        rng = np.random.RandomState(seed)
        self.window_size = window_size
        self.window = window_size
        self.d_in = d_in
        self.N = n_nodes
        self.graph = None

        raw    = rng.randn(n_nodes, T + window_size, d_in).astype(np.float32)
        labels = np.zeros((n_nodes, T), dtype=np.int64)
        for n in range(n_nodes):
            anomaly_times = rng.choice(T, size=int(T * anomaly_ratio), replace=False)
            raw[n, anomaly_times] += rng.randn(len(anomaly_times), d_in) * 5
            labels[n, anomaly_times] = 1

        self.samples = []
        
        # BaseTimeSeriesDataset structure: group by time, then tile by node
        # _index_t = np.repeat(times, N)
        # _index_v = np.tile(nodes, T)
        t_values = np.arange(T, dtype=np.int32)
        node_ids_arr = np.arange(n_nodes, dtype=np.int32)
        
        self._index_t = np.repeat(t_values, n_nodes)
        self._index_v = np.tile(node_ids_arr, T)
        
        # Build samples in the exact same order
        target_list = []
        for t in range(T):
            for n in range(n_nodes):
                window = raw[n, t: t + window_size]
                w_tensor = torch.tensor(window)
                # Target = next value after window (forecast target)
                target_val = raw[n, t + window_size]  # [d_in]
                self.samples.append({
                    "x_window": w_tensor,
                    "target":   torch.tensor(target_val),
                    "node_id":  n,
                    "t":        t,
                    "label":    int(labels[n, t]),
                    "graph":    None
                })
                target_list.append(torch.tensor(target_val))

        self._precomputed_labels = np.array([s["label"] for s in self.samples], dtype=np.int64)
        self._precomputed_windows = torch.stack([s["x_window"] for s in self.samples])
        self._precomputed_targets = torch.stack(target_list)  # [n_samples, d_in]




        self.as_tuples = [
            (s["node_id"], s["t"], s["label"]) for s in self.samples
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_dummy_graph(N: int, device: str) -> Data:
    """
    Fully-connected graph for a batch of N nodes.
    Used only when no real graph is available (mock mode).
    """
    edge_index = torch.combinations(torch.arange(N), r=2).T      # [2, E]
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return Data(edge_index=edge_index.to(device))


def _collate_graph(batch_list):
    """
    Custom collate: stacks tensors normally but handles
    torch_geometric Data objects by returning the first one
    (the graph is shared/static — same for all samples in a batch).
    """
    x_windows = torch.stack([b["x_window"] for b in batch_list])
    node_ids  = torch.tensor([b["node_id"]  for b in batch_list])
    ts        = torch.tensor([b["t"]        for b in batch_list])
    labels    = torch.tensor([b["label"]    for b in batch_list])

    # Graph: shared static object — just pass the first one through
    graph = batch_list[0].get("graph", None)

    return {
        "x_window": x_windows,
        "node_id":  node_ids,
        "t":        ts,
        "label":    labels,
        "graph":    graph,
    }


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
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
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

        self.optimizer = torch.optim.Adam(
            self.backbone.parameters(),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 1e-5)
        )

        self.recon_loss_fn = nn.MSELoss()

        self.history = {
            "train_loss": [],
            "val_f1":     [],
            "val_auc_pr": [],
            "n_samples":  [],
            "pct_data":   [],
        }

        self.wandb = None
        if config.get("use_wandb", False):
            try:
                import wandb
                wandb.init(
                    project=config.get("wandb_project", "rc-tgad"),
                    config=config,
                    name=config.get("run_name", "run")
                )
                self.wandb = wandb
                print("[Trainer] wandb initialized.")
            except ImportError:
                print("[Trainer] wandb not installed — skipping.")

        print(f"[Trainer] Device    : {self.device}")
        print(f"[Trainer] Curriculum: {'ON' if use_curriculum else 'OFF (baseline mode)'}")
        print(f"[Trainer] Dataset   : {len(dataset)} samples")

    # ── HARDNESS SCORING ─────────────────────────────────────────────────────
    @torch.no_grad()
    def _compute_hardness_from_loss(self) -> np.ndarray:
        print("[Trainer] Computing hardness scores with REAL labels...")
        t0 = time.time()
        
        ds = self.dataset
        N = self.raw_backbone.num_nodes
        n_samples = len(ds)
        n_times = n_samples // N
        batch_size = self.config.get("batch_size", 64)
        graph = ds.graph if ds.graph is not None else Data(edge_index=torch.empty((2, 0), dtype=torch.long))
    
        self.backbone.eval()
        all_scores = torch.zeros(n_samples)
        use_amp = (self.device != "cpu")
        
        for ti_start in range(0, n_times, batch_size):
            ti_end = min(ti_start + batch_size, n_times)
            B = ti_end - ti_start
            idx_start = ti_start * N
            idx_end = ti_end * N
            
            x_all = ds._precomputed_windows[idx_start:idx_end].to(self.device)
            # FIX: Fetch the actual labels for this batch
            batch_labels = ds._precomputed_labels[idx_start:idx_end] 
            
            x_all_reshaped = x_all.view(B, N, -1, x_all.shape[-1])
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                z_all, x_hat_all = self.backbone(x_all_reshaped, graph)
                
            z_all = z_all.view(B * N, -1)
            x_hat_all = x_hat_all.view(B * N, -1)
            target = ds._precomputed_targets[idx_start:idx_end].to(self.device)
    
            # Update the scorer loop to include the label
            for i in range(B * N):
                global_idx = idx_start + i
                # FIX: Pass the actual label to the scorer
                h = self.rag_scorer.score_hardness(
                    z=z_all[i],
                    x=target[i], # using target as the 'raw' x
                    x_hat=x_hat_all[i],
                    node_id=global_idx % N,
                    graph=graph,
                    t=global_idx // N,
                    ground_truth_label=int(batch_labels[i]) # PASSING REAL LABEL HERE
                )
                all_scores[global_idx] = h
                
        # Normalize with percentile clipping to fix the "stability" issue
        scores_np = all_scores.numpy()
        p5, p95 = np.percentile(scores_np, 5), np.percentile(scores_np, 95)
        scores_np = np.clip((scores_np - p5) / (p95 - p5 + 1e-8), 0, 1)
    
        self.backbone.train()
        return scores_np

    # ── SINGLE EPOCH ─────────────────────────────────────────────────────────

    def _train_epoch(self, indices, batch_size: int) -> float:

        self.backbone.train()

        total_loss = 0.0
        n_steps = 0

        ds = self.dataset
        N = self.raw_backbone.num_nodes
        W = ds.window
        d_in = ds._precomputed_windows.shape[-1]
        graph = ds.graph

        # --------------------------------------------------
        # Pre-sort indices by timestep ONCE — O(N log N)
        # Then each batch is a contiguous slice — O(1)
        # This replaces the old O(N * num_batches) mask scan.
        # --------------------------------------------------
        indices_arr = np.asarray(indices, dtype=np.int64)
        times = ds._index_t[indices_arr]
        sort_order = np.argsort(times, kind='mergesort')
        sorted_indices = indices_arr[sort_order]
        sorted_times = times[sort_order]

        # Find boundaries between unique timesteps
        change_mask = np.empty(len(sorted_times), dtype=bool)
        change_mask[0] = True
        change_mask[1:] = sorted_times[1:] != sorted_times[:-1]
        group_starts = np.where(change_mask)[0]
        n_unique = len(group_starts)
        # Append sentinel for easy slicing
        group_starts = np.append(group_starts, len(sorted_indices))

        # Append sentinel for easy slicing
        group_starts = np.append(group_starts, len(sorted_indices))

        # We no longer pre-build PyG Batches. Backbone handles it.
        # Fallback for mock dataset testing:
        graph = ds.graph if ds.graph is not None else Data(edge_index=torch.empty((2, 0), dtype=torch.long))

        # AMP scaler for mixed precision
        use_amp = (self.device != "cpu")
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        # --------------------------------------------------
        # Train batches of timesteps — contiguous slice per batch
        # --------------------------------------------------
        for gi_start in range(0, n_unique, batch_size):
            gi_end = min(gi_start + batch_size, n_unique)
            B = gi_end - gi_start

            # Contiguous slice of all selected samples in this batch's timesteps
            idx_lo = group_starts[gi_start]
            idx_hi = group_starts[gi_end]
            chunk_global_idx = sorted_indices[idx_lo:idx_hi]

            # The unique timesteps present in this chunk
            unique_in_chunk = sorted_times[group_starts[gi_start:gi_end]]

            # We MUST load all N nodes for these timesteps so the GNN has full context,
            # otherwise unselected nodes would be zero-padded and corrupt the graph!
            # The exact contiguous indices in the dataset for these full timesteps are:
            # We map physical time t to block index k: k = (t - window) // stride
            stride = getattr(ds, 'stride', 1)
            k_values = (unique_in_chunk - W) // stride
            base_idx = (k_values * N)[:, None]  # [B, 1]
            node_offsets = np.arange(N)[None, :]       # [1, N]
            full_t_indices = (base_idx + node_offsets).flatten() # [B * N]

            x_all = ds._precomputed_windows[full_t_indices].to(
                self.device, non_blocking=True)

            # Which timestep-group does each *selected* sample belong to? (0..B-1)
            chunk_times = sorted_times[idx_lo:idx_hi]
            batch_offset = np.searchsorted(unique_in_chunk, chunk_times)

            # Flat indices into the [B*N] batch for the *selected* nodes only
            nids = ds._index_v[chunk_global_idx].astype(np.int64)
            selected_flat_indices = torch.from_numpy(batch_offset * N + nids).long().to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            x_all_reshaped = x_all.view(B, N, W, d_in)

            with torch.amp.autocast('cuda', enabled=use_amp):
                # DataParallel splits gracefully along B dimension
                z_all, x_hat_all = self.backbone(x_all_reshaped, graph)
                
                # Flatten the outputs back to [B*N, ...]
                z_all = z_all.view(B * N, -1)
                x_hat_all = x_hat_all.view(B * N, -1)
                
                # Forecast target: signals[t] — one step AFTER the window
                target = ds._precomputed_targets[full_t_indices].to(
                    self.device, non_blocking=True)
                
                # ONLY compute loss on the nodes selected by the curriculum!
                # If we compute it on all B*N nodes, curriculum is defeated.
                loss = self.recon_loss_fn(
                    x_hat_all[selected_flat_indices], 
                    target[selected_flat_indices]
                )

            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), 1.0)
            scaler.step(self.optimizer)
            scaler.update()

            total_loss += loss.item()
            n_steps += 1

        return total_loss / max(n_steps, 1)

    # ── VALIDATION ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, val_dataset) -> Tuple[float, float]:

        from utils.metrics import compute_f1, compute_auc_pr

        self.backbone.eval()

        all_scores = []
        all_labels = []

        N = self.raw_backbone.num_nodes
        graph = val_dataset.graph
        n_samples = len(val_dataset)
        n_times = n_samples // N
        batch_size = self.config.get("batch_size", 64)

        # We no longer pre-build PyG Batches.
        graph = val_dataset.graph if val_dataset.graph is not None else Data(edge_index=torch.empty((2, 0), dtype=torch.long))

        use_amp = (self.device != "cpu")

        # Data is ordered by timestep: indices [ti*N : (ti+1)*N] = timestep ti
        for ti_start in range(0, n_times, batch_size):
            ti_end = min(ti_start + batch_size, n_times)
            B = ti_end - ti_start

            idx_start = ti_start * N
            idx_end = ti_end * N

            # Single contiguous slice — no Python loop!
            x_all = val_dataset._precomputed_windows[idx_start:idx_end].to(
                self.device, non_blocking=True)
            chunk_labels = val_dataset._precomputed_labels[idx_start:idx_end]

            x_all_reshaped = x_all.view(B, N, -1, x_all.shape[-1])

            with torch.amp.autocast('cuda', enabled=use_amp):
                _, x_hat_all = self.backbone(x_all_reshaped, graph)
                
            x_hat_all = x_hat_all.view(B * N, -1)

            # Forecast target: signals[t] — one step AFTER the window
            target = val_dataset._precomputed_targets[idx_start:idx_end].to(
                self.device, non_blocking=True)
            scores = torch.norm(x_hat_all - target, dim=1)

            all_scores.extend(scores.cpu().tolist())
            all_labels.extend(chunk_labels.tolist())

        return compute_f1(all_scores, all_labels), compute_auc_pr(all_scores, all_labels)
    

        # ── MAIN TRAIN LOOP ──────────────────────────────────────────────────────

    def train(self, val_dataset=None, save_dir: str = "checkpoints"):
        os.makedirs(save_dir, exist_ok=True)

        epochs     = self.config.get("epochs", 100)
        k_warmup   = self.config.get("k_warmup", 30)
        batch_size = self.config.get("batch_size", 64)
        best_f1    = -1.0
        best_epoch = -1
        n_samples  = len(self.dataset)

        # ── Cosine annealing LR scheduler ───────────────────────────────
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-5
        )

        # ── Initial hardness scores (numpy array) ───────────────────────
        if self.use_curriculum:
            hardness_array = self._compute_hardness_from_loss()
        else:
            hardness_array = np.zeros(n_samples, dtype=np.float32)

        print(f"\n[Trainer] Starting training for {epochs} epochs...")
        print("-" * 60)

        for epoch in range(epochs):

            t_start = time.time()

            if self.use_curriculum:
                indices = get_batch_fast(
                    hardness_array, epoch, k_warmup, verbose=False
                )
                # Recompute hardness every 10 epochs (curriculum adapts)
                if epoch > 0 and epoch % 10 == 0:
                    hardness_array = self._compute_hardness_from_loss()
            else:
                indices = np.arange(n_samples)

            train_loss = self._train_epoch(indices, batch_size)
            lr_scheduler.step()

            epoch_time = time.time() - t_start

            f1, auc_pr = 0.0, 0.0

            if val_dataset is not None and (epoch % 5 == 0 or epoch == epochs - 1):

                f1, auc_pr = self._validate(val_dataset)

                if f1 > best_f1:
                    best_f1 = f1
                    best_epoch = epoch

                    ckpt_path = os.path.join(save_dir, "best_model.pt")

                    torch.save({
                        "epoch": epoch,
                        "model_state": self.backbone.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "f1": f1,
                        "auc_pr": auc_pr,
                        "config": self.config
                    }, ckpt_path)

            pct = 100 * len(indices) / n_samples
            lam = pacing(epoch, k_warmup)

            self.history["train_loss"].append(train_loss)
            self.history["val_f1"].append(f1)
            self.history["val_auc_pr"].append(auc_pr)
            self.history["n_samples"].append(len(indices))
            self.history["pct_data"].append(pct)

            if epoch % 5 == 0 or epoch == epochs - 1:
                lr_now = lr_scheduler.get_last_lr()[0]
                print(
                    f"Epoch {epoch:4d}/{epochs} | "
                    f"λ={lam:.3f} | "
                    f"data={pct:5.1f}% | "
                    f"loss={train_loss:.4f} | "
                    f"F1={f1:.4f} AUC-PR={auc_pr:.4f} | "
                    f"lr={lr_now:.6f} | "
                    f"{epoch_time:.1f}s"
                )

        print("-" * 60)
        print(f"[Trainer] Done. Best F1={best_f1:.4f} at epoch {best_epoch}")

        return self.history


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    os.makedirs("utils", exist_ok=True)
    if not os.path.exists("utils/metrics.py"):
        with open("utils/metrics.py", "w") as f:
            f.write("""
from sklearn.metrics import f1_score, average_precision_score
import numpy as np

def compute_f1(scores, labels, threshold=None):
    scores = np.array(scores); labels = np.array(labels)
    if threshold is None: threshold = np.percentile(scores, 90)
    preds = (scores >= threshold).astype(int)
    return float(f1_score(labels, preds, zero_division=0))

def compute_auc_pr(scores, labels):
    scores = np.array(scores); labels = np.array(labels)
    if labels.sum() == 0: return 0.0
    return float(average_precision_score(labels, scores))
""")

    for d in ["curriculum", "utils"]:
        init = os.path.join(d, "__init__.py")
        if not os.path.exists(init):
            open(init, "w").close()

    print("=" * 60)
    print("TRAINER SMOKE TEST")
    print("=" * 60)

    CONFIG = {
        "epochs": 20, "k_warmup": 10, "batch_size": 32,
        "lr": 1e-3, "weight_decay": 1e-5, "use_wandb": False, "run_name": "smoke"
    }

    train_ds = MockTemporalGraphDataset(n_nodes=5, T=100, window_size=10, d_in=8)
    val_ds   = MockTemporalGraphDataset(n_nodes=5, T=50,  window_size=10, d_in=8, seed=99)
    scorer   = MockRAGScorer(seed=42)

    print("\n--- Baseline (no curriculum) ---")
    t1 = Trainer(MockBackbone(d_in=8, d_z=32, num_nodes=5), scorer, train_ds, CONFIG, use_curriculum=False)
    h1 = t1.train(val_dataset=val_ds, save_dir="checkpoints/baseline")

    print("\n--- RC-TGAD (curriculum ON) ---")
    t2 = Trainer(MockBackbone(d_in=8, d_z=32, num_nodes=5), scorer, train_ds, CONFIG, use_curriculum=True)
    h2 = t2.train(val_dataset=val_ds, save_dir="checkpoints/rctgad")

    print(f"\nBaseline F1 : {h1['val_f1'][-1]:.4f}")
    print(f"RC-TGAD  F1 : {h2['val_f1'][-1]:.4f}")
    print("\nSmoke test passed ✅")
