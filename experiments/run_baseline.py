# experiments/run_baseline.py
# PERSON 3 — Baseline Experiment Runner
# RC-TGAD: Runs the NO-CURRICULUM baseline (vanilla LSTM+GNN with no scheduling)

import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config_loader import load_config
from curriculum.trainer    import Trainer, MockBackbone, MockRAGScorer, MockTemporalGraphDataset
from utils.metrics         import evaluate, AblationTracker, smooth_scores


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="RC-TGAD Baseline Runner")
    parser.add_argument("--config",   type=str,  default="configs/default.yaml")
    parser.add_argument("--mock",     action="store_true", help="Use mock data/model")
    parser.add_argument("--seeds",    type=int,  nargs="+", default=[42, 43, 44])
    parser.add_argument("--dataset",  type=str,  default=None)
    parser.add_argument("--override", type=str,  nargs="*", default=[])
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# DATASET & MODEL LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(cfg, seed, mock=False):
    if mock:
        d_in       = cfg["model"]["d_in"]
        win        = cfg["model"]["window_size"]
        train_data = MockTemporalGraphDataset(n_nodes=10, T=500, window_size=win, d_in=d_in, seed=seed)
        val_data   = MockTemporalGraphDataset(n_nodes=10, T=150, window_size=win, d_in=d_in, seed=seed+100)
        test_data  = MockTemporalGraphDataset(n_nodes=10, T=200, window_size=win, d_in=d_in, seed=seed+200)
        return train_data, val_data, test_data

    dataset_name = cfg["data"]["dataset"]
    win    = cfg["model"]["window_size"]
    stride = cfg["data"].get("stride", 1)

    if dataset_name == "swat":
        from data.swat import load_swat
        train_data, val_data, test_data, _ = load_swat(
            data_dir  = cfg["data"]["data_dir"],
            window    = win,
            stride    = stride,
            val_ratio = cfg["data"]["val_split"],
        )
    elif dataset_name == "smap":
        from data.smap import load_smap
        train_data, val_data, test_data, _ = load_smap(
            data_dir  = cfg["data"].get("data_dir", "data/raw/smap"),
            window    = win,
            stride    = stride,
            val_ratio = cfg["data"]["val_split"],
        )
    elif dataset_name == "msl":
        from data.smap import load_msl      
        train_data, val_data, test_data, _ = load_msl(
            data_dir  = cfg["data"].get("data_dir", "data/raw/smap"),
            window    = win,
            stride    = stride,
            val_ratio = cfg["data"]["val_split"],
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    cfg["model"]["num_nodes"] = train_data.N
    print(f"[Dataset] Auto-detected num_nodes = {train_data.N}")

    train_data.as_tuples = train_data.as_flat_list()
    val_data.as_tuples   = val_data.as_flat_list()
    test_data.as_tuples  = test_data.as_flat_list()

    return train_data, val_data, test_data

def load_backbone(cfg, mock=False):
    if mock:
        return MockBackbone(
            d_in=cfg["model"]["d_in"],
            d_z=cfg["model"]["gnn_out_dim"],
            num_nodes=10   
        )
    from backbone.backbone import Backbone
    return Backbone(
        d_in        = cfg["model"]["d_in"],         
        hidden_size = cfg["model"]["lstm_hidden"],    
        gnn_out_dim = cfg["model"]["gnn_out_dim"],    
        num_nodes   = cfg["model"]["num_nodes"],      
        window_size = cfg["model"]["window_size"],    
        lstm_layers = cfg["model"]["lstm_layers"],    
        gat_heads   = cfg["model"]["gnn_heads"],      
        dropout     = cfg["model"]["dropout"],        
    )

def load_rag_scorer(cfg, mock=False):
    return MockRAGScorer()


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE & EVALUATION WITH DYNAMIC THRESHOLDING
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _run_inference(backbone, dataset, cfg, device):
    """Helper function to run pure inference on any dataset split."""
    from collections import defaultdict
    from torch.utils.data import DataLoader
    
    all_scores = []
    all_labels = []

    if isinstance(backbone, MockBackbone):
        def mock_collate(batch):
            return {
                "x_window": torch.stack([b["x_window"] for b in batch]),
                "target":   torch.stack([b["target"] for b in batch]) if "target" in batch[0] else torch.stack([b["x_window"][:, -1, :] for b in batch]),
                "label": torch.tensor([b["label"] for b in batch]),
                "node_id": [b["node_id"] for b in batch],
                "t": [b["t"] for b in batch],
                "graph": None,
            }
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, collate_fn=mock_collate)
        for batch in loader:
            x_window = batch["x_window"].to(device, non_blocking=True)
            target   = batch["target"].to(device, non_blocking=True) 
            labels   = batch["label"]
            _, x_hat_all = backbone(x_window, None)
            scores = torch.norm(x_hat_all - target, dim=1)
            all_scores.extend(scores.cpu().tolist())
            all_labels.extend(labels.tolist())
            
    else:
        from curriculum.trainer import DPGraphWrapper
        batch_size = cfg["training"].get("batch_size", 32) * 2 
        n_samples = len(dataset)
        
        for i in range(0, n_samples, batch_size):
            end_i = min(i + batch_size, n_samples)
            batch_data = [dataset[j] for j in range(i, end_i)]
            
            x = torch.stack([d["x"] for d in batch_data]).to(device)
            y = torch.stack([d["y"] for d in batch_data])
            
            graph_safe = DPGraphWrapper(batch_data[0]["graph"])
            _, x_hat = backbone(x, graph_safe)
            
            target = torch.stack([
                torch.tensor(dataset.signals[d["t"]], dtype=torch.float32) 
                for d in batch_data
            ]).unsqueeze(-1).to(device)
            
            node_scores = torch.norm(
                x_hat.view(x_hat.shape[0], x_hat.shape[1], -1) - 
                target.view(target.shape[0], target.shape[1], -1), 
                dim=-1
            )
            # 🛡️ FIX: Changed from max() to mean() to prevent single-sensor False Positives
            system_scores = node_scores.mean(dim=1)
            system_labels = y[:, 0]
            
            all_scores.extend(system_scores.cpu().tolist())
            all_labels.extend(system_labels.tolist())
            
    return np.array(all_scores), np.array(all_labels)


@torch.no_grad()
def evaluate_on_test(backbone, test_dataset, cfg, device, val_dataset=None) -> dict:
    """Evaluates the model, using a Mean+Sigma distribution to set the threshold."""
    
    dynamic_threshold = None
    
    # 1. Sweep Validation Set to calculate Statistical Threshold
    if val_dataset is not None:
        print("\n[Evaluate] Running inference on Validation Set for Dynamic Thresholding...")
        val_scores, _ = _run_inference(backbone, val_dataset, cfg, device)
        
        # Smooth to ignore point-noise, then calculate distribution
        smoothed_val = smooth_scores(val_scores, window_size=10)
        
        # ⚡ IEEE STANDARD: Mean + 3*Sigma (99.7% Confidence Interval)
        # This is much more robust than 'Max' for high-precision papers.
        v_mean = np.mean(smoothed_val)
        v_std = np.std(smoothed_val)
        
        # You can adjust '3' to '2' if you want even higher recall
        dynamic_threshold = float(v_mean + (3 * v_std)) 
        
        print(f"[Evaluate] Val Mean: {v_mean:.4f} | Val Std: {v_std:.4f}")
        print(f"[Evaluate] Statistical Threshold (Mean + 3σ): {dynamic_threshold:.4f}")

    # 2. Sweep Test Set
    print("[Evaluate] Running inference on Test Set...")
    test_scores, test_labels = _run_inference(backbone, test_dataset, cfg, device)

    # 3. Calculate metrics using our statistical cutoff
    return evaluate(test_scores, test_labels, threshold=dynamic_threshold, verbose=False)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE SEED RUN
# ─────────────────────────────────────────────────────────────────────────────

def run_single_seed(cfg, seed, mock, results_dir):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*55}")
    print(f"  BASELINE RUN  |  seed={seed}  |  dataset={cfg['data']['dataset']}")
    print(f"{'='*55}")

    cfg["training"]["seed"]    = seed
    cfg["logging"]["run_name"] = f"baseline_seed{seed}"

    train_data, val_data, test_data = load_dataset(cfg, seed, mock)
    backbone                        = load_backbone(cfg, mock)
    rag_scorer                      = load_rag_scorer(cfg, mock)

    device = cfg["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available, falling back to CPU")
        device = "cpu"

    trainer = Trainer(
        backbone=backbone,
        rag_scorer=rag_scorer,
        dataset=train_data,
        config={
            "epochs":        cfg["training"]["epochs"],
            "k_warmup":      cfg["curriculum"]["k_warmup"],
            "batch_size":    cfg["training"]["batch_size"],
            "lr":            cfg["training"]["lr"],
            "weight_decay":  cfg["training"]["weight_decay"],
            "use_wandb":     cfg["logging"]["use_wandb"],
            "run_name":      cfg["logging"]["run_name"],
            "wandb_project": cfg["logging"]["wandb_project"],
        },
        use_curriculum=False,
        device=device
    )

    history = trainer.train(
        val_dataset=val_data,
        save_dir=os.path.join(results_dir, f"seed{seed}")
    )

    print(f"\n[Baseline] Evaluating on test set (seed={seed})...")
    # Passed val_data here to activate the thresholding
    test_results = evaluate_on_test(backbone, test_data, cfg, device, val_dataset=val_data)

    print(f"\n  Test Results (seed={seed}):")
    print(f"    F1-PA    : {test_results['f1_pa']:.4f}")
    print(f"    AUC-PR   : {test_results['auc_pr']:.4f}")
    print(f"    AUC-ROC  : {test_results['auc_roc']:.4f}")
    print(f"    Precision: {test_results['precision']:.4f}")
    print(f"    Recall   : {test_results['recall']:.4f}")

    seed_path = os.path.join(results_dir, f"seed{seed}", "test_results.json")
    os.makedirs(os.path.dirname(seed_path), exist_ok=True)
    with open(seed_path, "w") as f:
        json.dump({**test_results, "history": history}, f, indent=2)

    return test_results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    overrides = args.override or []
    if args.dataset:
        overrides.append(f"data.dataset={args.dataset}")
    if args.mock:
        overrides += ["training.epochs=20", "training.device=cpu"]

    cfg = load_config(args.config, overrides if overrides else None)

    dataset_name = cfg["data"]["dataset"]
    results_dir  = os.path.join("results", "baseline", dataset_name)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\nRC-TGAD — BASELINE EXPERIMENT")
    print(f"Dataset  : {dataset_name}")
    print(f"Seeds    : {args.seeds}")
    print(f"Epochs   : {cfg['training']['epochs']}")
    print(f"Mock mode: {args.mock}")
    print(f"Results  : {results_dir}")

    all_results = []
    for seed in args.seeds:
        result = run_single_seed(cfg, seed, args.mock, results_dir)
        all_results.append(result)

    print(f"\n{'='*55}")
    print(f"  BASELINE FINAL RESULTS  ({dataset_name}, {len(args.seeds)} seeds)")
    print(f"{'='*55}")

    for metric in ["f1_pa", "auc_pr", "auc_roc", "precision", "recall"]:
        vals = [r[metric] for r in all_results]
        print(f"  {metric:<12}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    agg = {
        metric: {
            "mean": float(np.mean([r[metric] for r in all_results])),
            "std":  float(np.std([r[metric]  for r in all_results])),
            "runs": [r[metric] for r in all_results]
        }
        for metric in ["f1_pa", "auc_pr", "auc_roc", "precision", "recall"]
    }
    agg_path = os.path.join(results_dir, "aggregate.json")
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\n  Saved to {agg_path}")

if __name__ == "__main__":
    main()
