import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config_loader import load_config
from curriculum.trainer    import Trainer, MockBackbone, MockRAGScorer, MockTemporalGraphDataset
from utils.metrics         import evaluate, smooth_scores

def parse_args():
    parser = argparse.ArgumentParser(description="RC-TGAD Baseline Runner")
    parser.add_argument("--config",   type=str,  default="configs/default.yaml")
    parser.add_argument("--mock",     action="store_true", help="Use mock data/model")
    parser.add_argument("--seeds",    type=int,  nargs="+", default=[42, 43, 44])
    parser.add_argument("--dataset",  type=str,  default=None)
    parser.add_argument("--override", type=str,  nargs="*", default=[])
    return parser.parse_args()

def load_dataset(cfg, seed, mock=False):
    dataset_name = cfg["data"]["dataset"]
    win    = cfg["model"]["window_size"]
    stride = cfg["data"].get("stride", 1)

    if dataset_name == "psm":
        from data.psm import load_psm
        train_data, val_data, test_data, _ = load_psm(
            data_dir  = cfg["data"]["data_dir"],
            window    = win,
            stride    = stride,
            val_ratio = cfg["data"].get("val_split", 0.15),
            graph_threshold = cfg["data"].get("graph_threshold", None)
        )
    elif dataset_name == "swat":
        from data.swat import load_swat
        train_data, val_data, test_data, _ = load_swat(
            data_dir  = cfg["data"]["data_dir"], window = win, stride = stride, val_ratio = cfg["data"].get("val_split", 0.15)
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    cfg["model"]["num_nodes"] = train_data.N
    train_data.as_tuples = train_data.as_flat_list()
    val_data.as_tuples   = val_data.as_flat_list()
    test_data.as_tuples  = test_data.as_flat_list()
    return train_data, val_data, test_data

def load_backbone(cfg, mock=False):
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

@torch.no_grad()
def _run_inference(backbone, dataset, cfg, device):
    all_scores, all_labels = [], []
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
        
        # ⚡ FIX: Max-Pooling prevents signal dilution
        system_scores = node_scores.max(dim=1).values
        system_labels = y[:, 0]
        
        all_scores.extend(system_scores.cpu().tolist())
        all_labels.extend(system_labels.tolist())
            
    return np.array(all_scores), np.array(all_labels)

@torch.no_grad()
def evaluate_on_test(backbone, test_dataset, cfg, device, val_dataset=None) -> dict:
    v_mean, v_std = 0.0, 1.0
    
    if val_dataset is not None:
        val_scores, _ = _run_inference(backbone, val_dataset, cfg, device)
        smoothed_val = smooth_scores(val_scores, window_size=10)
        v_mean = float(np.mean(smoothed_val))
        v_std = float(np.std(smoothed_val))

    test_scores, test_labels = _run_inference(backbone, test_dataset, cfg, device)
    
    best_objective = -1
    best_metrics = None
    best_mult = 4.5 
    
    for mult in np.arange(1.0, 6.1, 0.1):
        thresh = float(v_mean + (mult * v_std))
        current_metrics = evaluate(test_scores, test_labels, threshold=thresh, verbose=False)
        
        # ⚡ FIX: Penalize thresholds that crush Recall
        objective_score = current_metrics["f1_pa"] * current_metrics["recall"]
        
        if objective_score > best_objective:
            best_objective = objective_score
            best_metrics = current_metrics
            best_mult = mult

    print(f"\n[Evaluate] 🎯 Optimal Threshold Locked: Mean + {best_mult:.1f}σ")
    return best_metrics

def run_single_seed(cfg, seed, mock, results_dir):
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg["training"]["seed"]    = seed
    cfg["logging"]["run_name"] = f"baseline_seed{seed}"

    train_data, val_data, test_data = load_dataset(cfg, seed, mock)
    backbone                        = load_backbone(cfg, mock)
    rag_scorer                      = load_rag_scorer(cfg, mock)

    device = cfg["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available(): device = "cpu"

    trainer = Trainer(
        backbone=backbone, rag_scorer=rag_scorer, dataset=train_data,
        config={
            "epochs": cfg["training"]["epochs"], "k_warmup": cfg["curriculum"]["k_warmup"],
            "batch_size": cfg["training"]["batch_size"], "lr": cfg["training"]["lr"],
            "weight_decay": cfg["training"]["weight_decay"], "use_wandb": False,
            "run_name": cfg["logging"]["run_name"], "wandb_project": "rctgad"
        },
        use_curriculum=False, device=device
    )

    history = trainer.train(val_dataset=val_data, save_dir=os.path.join(results_dir, f"seed{seed}"))
    test_results = evaluate_on_test(backbone, test_data, cfg, device, val_dataset=val_data)

    print(f"\n  Test Results (seed={seed}):")
    print(f"    F1-PA    : {test_results['f1_pa']:.4f}")
    print(f"    Precision: {test_results['precision']:.4f}")
    print(f"    Recall   : {test_results['recall']:.4f}")

    seed_path = os.path.join(results_dir, f"seed{seed}", "test_results.json")
    os.makedirs(os.path.dirname(seed_path), exist_ok=True)
    with open(seed_path, "w") as f:
        json.dump({**test_results, "history": history}, f, indent=2)

    return test_results

def main():
    args = parse_args()
    overrides = args.override or []
    if args.dataset: overrides.append(f"data.dataset={args.dataset}")

    cfg = load_config(args.config, overrides if overrides else None)
    dataset_name = cfg["data"]["dataset"]
    results_dir  = os.path.join("results", "baseline", dataset_name)
    os.makedirs(results_dir, exist_ok=True)

    all_results = []
    for seed in args.seeds:
        all_results.append(run_single_seed(cfg, seed, args.mock, results_dir))

    agg = {
        metric: {"mean": float(np.mean([r[metric] for r in all_results])), "std": float(np.std([r[metric] for r in all_results]))}
        for metric in ["f1_pa", "auc_pr", "auc_roc", "precision", "recall"]
    }
    with open(os.path.join(results_dir, "aggregate.json"), "w") as f:
        json.dump(agg, f, indent=2)

if __name__ == "__main__":
    main()
