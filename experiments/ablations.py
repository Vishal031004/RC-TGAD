import os
import sys
import json
import argparse
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config_loader    import load_config, get_ablation_configs
from curriculum.trainer       import Trainer, MockBackbone, MockTemporalGraphDataset
from utils.metrics            import AblationTracker
from experiments.run_baseline import load_dataset, load_backbone, evaluate_on_test

try:
    from rag.rag_scorer       import score_hardness
except ModuleNotFoundError:
    try:
        from curriculum.rag_scorer import score_hardness
    except ModuleNotFoundError:
        from rag_scorer       import score_hardness

from rag.vector_store         import VectorStore
from utils.logger             import DeepResearchLogger

def parse_args():
    parser = argparse.ArgumentParser(description="RC-TGAD Ablation Runner")
    parser.add_argument("--config",   type=str, default="configs/default.yaml")
    parser.add_argument("--mock",     action="store_true", help="Use mock data/model")
    parser.add_argument("--seeds",    type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--dataset",  type=str, default=None)
    parser.add_argument("--variants", type=str, nargs="*", default=None)
    parser.add_argument("--override", type=str, nargs="*", default=[])
    return parser.parse_args()

class RandomHardnessScorer:
    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)
    def score_hardness(self, **kwargs):
        if 'z' in kwargs and len(kwargs['z'].shape) == 3:
            B, N = kwargs['z'].shape[0], kwargs['z'].shape[1]
            return [[{"total": float(self.rng.random()), "temp": 0.0, "struct": 0.0, "rag": 0.0} for _ in range(N)] for _ in range(B)]
        return float(self.rng.random())

def run_variant_seed(variant_name, cfg, seed, mock, results_dir):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)

    safe_name = variant_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    variant_dir = os.path.join(results_dir, safe_name, f"seed{seed}")
    os.makedirs(variant_dir, exist_ok=True)

    result_path = os.path.join(variant_dir, "test_results.json")
    if os.path.exists(result_path):
        print(f"  [Skip] {variant_name} seed={seed} — already computed")
        with open(result_path) as f:
            saved = json.load(f)
            return {k: saved[k] for k in ["f1_pa", "auc_pr", "auc_roc", "precision", "recall"]}

    cfg["training"]["seed"]    = seed
    cfg["logging"]["run_name"] = f"{safe_name}_seed{seed}"

    paper_logger = DeepResearchLogger(save_dir=variant_dir, config_dict=cfg)
    train_data, val_data, test_data = load_dataset(cfg, seed, mock)
    backbone = load_backbone(cfg, mock)
    paper_logger.log_architecture(backbone, train_data)

    device = cfg["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    curriculum_enabled = cfg["curriculum"]["enabled"]

    if not curriculum_enabled or variant_name == "Random Curriculum":
        rag_scorer = RandomHardnessScorer(seed=seed)
    else:
        print("\n" + "="*60)
        print("🚀 [Ablations] Ultimate SOTA Fusion Engaged!")
        print("="*60)
        
        class RawScorerInjection:
            def __init__(self, vector_dim=128):
                import inspect
                params = inspect.signature(VectorStore.__init__).parameters
                if 'dim' in params: self.vector_store = VectorStore(dim=vector_dim)
                else:               self.vector_store = VectorStore(vector_dim)
                
                self.cached_h_struct = None
                self.window_errors = [] 
                self.rag_window_dists = [] 
                print(f"✅ Node-Level Injection active (Dim: {vector_dim})")
                
            def score_hardness(self, z, x, x_hat, y, batch_data, alphas):
                import torch.nn.functional as F
                B, N, D = z.shape
                a1, a2, a3 = alphas
                
                # =========================================================
                # 1. H_TEMP (SOTA Percentile Mapping)
                # =========================================================
                diff = (x_hat.view(B, N, -1) - x.view(B, N, -1)).cpu()
                raw_e = torch.norm(diff, dim=-1).numpy()
                
                self.window_errors.extend(raw_e.flatten().tolist())
                if len(self.window_errors) > 5000:
                    self.window_errors = self.window_errors[-5000:]
                
                if len(self.window_errors) > 10:
                    e_min = np.percentile(self.window_errors, 5)
                    e_max = np.percentile(self.window_errors, 95)
                    h_temp = (raw_e - e_min) / (e_max - e_min + 1e-8)
                    h_temp = np.clip(h_temp, 0.0, 1.0)
                else:
                    h_temp = np.full((B, N), 0.5)

                # =========================================================
                # 2. H_STRUCT (SOTA Centrality + Dynamic Normalization)
                # =========================================================
                if self.cached_h_struct is None:
                    self.cached_h_struct = np.zeros(N)
                    try:
                        from rag.hardness import compute_h_struct
                        graph = batch_data[0]["graph"]
                        for n in range(N):
                            # The function inside rag/hardness.py now natively computes centrality!
                            self.cached_h_struct[n] = compute_h_struct(n, graph, anomaly_source_id=0, gamma=0.5)
                    except Exception as e:
                        print(f"\n🚨 H_STRUCT CALCULATION FAILED: {e}")
                        self.cached_h_struct = np.full(N, 0.6667)
                h_struct = np.tile(self.cached_h_struct, (B, 1))
                
                # =========================================================
                # 3. H_RAG (Percentile Stretched Vector Distance)
                # =========================================================
                z_flat = z.view(B * N, D)
                z_norm = F.normalize(z_flat, p=2, dim=-1).to(self.vector_store.device)
                
                valid_memory = self.vector_store.memory[:self.vector_store.ptr]
                
                if valid_memory.shape[0] > 0:
                    distances = torch.cdist(z_norm, valid_memory)
                    k_actual = min(10, valid_memory.shape[0])
                    topk_dists, _ = torch.topk(distances, k_actual, largest=False, dim=1)
                    
                    raw_rag = topk_dists.mean(dim=1).cpu().numpy()
                    
                    self.rag_window_dists.extend(raw_rag.tolist())
                    if len(self.rag_window_dists) > 5000:
                        self.rag_window_dists = self.rag_window_dists[-5000:]
                        
                    if len(self.rag_window_dists) > 10:
                        r_min = np.percentile(self.rag_window_dists, 5)
                        r_max = np.percentile(self.rag_window_dists, 95)
                        h_rag = (raw_rag - r_min) / (r_max - r_min + 1e-8)
                        h_rag = np.clip(h_rag, 0.0, 1.0).reshape(B, N)
                    else:
                        h_rag = np.zeros((B, N))
                else:
                    h_rag = np.zeros((B, N))
                    
                # Auto-Populate VectorStore
                for b in range(B):
                    if int(y[b, 0].item()) == 0:
                        for n in range(N):
                            idx = b * N + n
                            self.vector_store.add(z_norm[idx], 0)

                # =========================================================
                # 4. Pack Results
                # =========================================================
                batch_results = []
                for b in range(B):
                    node_results = []
                    for n in range(N):
                        tot = a1 * h_temp[b, n] + a2 * h_struct[b, n] + a3 * h_rag[b, n]
                        node_results.append({
                            "temp": float(h_temp[b, n]),
                            "struct": float(h_struct[b, n]),
                            "rag": float(h_rag[b, n]),
                            "total": float(tot)
                        })
                    batch_results.append(node_results)
                    
                return batch_results
                
        v_dim = cfg.get("rag", {}).get("vector_dim", 128)
        rag_scorer = RawScorerInjection(vector_dim=v_dim)

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
        use_curriculum=curriculum_enabled,
        device=device,
        logger=paper_logger
    )

    history = trainer.train(val_dataset=val_data, save_dir=variant_dir)
    test_results = evaluate_on_test(backbone, test_data, cfg, device, val_dataset=val_data)

    with open(result_path, "w") as f:
        json.dump({**test_results, "history": history}, f, indent=2)

    return {k: test_results[k] for k in ["f1_pa", "auc_pr", "auc_roc", "precision", "recall"]}

def generate_latex_table(agg, dataset_name):
    metrics   = ["f1_pa", "auc_pr", "precision", "recall"]
    col_names = ["F1-PA", "AUC-PR", "Precision", "Recall"]
    variants  = list(agg.keys())
    best = {m: max(agg[v][m]["mean"] for v in variants) for m in variants} 

    lines = [
        r"\begin{table}[h]", r"\centering",
        r"\caption{Ablation study on " + dataset_name.upper() + r" dataset.}",
        r"\label{tab:ablation}", r"\begin{tabular}{lcccc}", r"\toprule",
        r"\textbf{Variant} & " + " & ".join(f"\\textbf{{{c}}}" for c in col_names) + r" \\",
        r"\midrule"
    ]
    for variant in variants:
        row_vals = []
        for m in metrics:
            mean, std = agg[variant][m]["mean"], agg[variant][m]["std"]
            val = f"{mean:.4f}$\\pm${std:.4f}"
            try:
                if abs(mean - best[m]) < 1e-6: val = f"\\textbf{{{val}}}"
            except KeyError:
                pass
            row_vals.append(val)
        safe_var = variant.replace("_", r"\_").replace("&", r"\&")
        lines.append(f"{safe_var} & " + " & ".join(row_vals) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)

def main():
    args = parse_args()
    overrides = list(args.override or [])
    if args.dataset: overrides.append(f"data.dataset={args.dataset}")
    if args.mock: overrides += ["training.epochs=20", "training.device=cpu"]

    base_cfg     = load_config(args.config, overrides if overrides else None)
    dataset_name = base_cfg["data"]["dataset"]
    results_dir  = os.path.join("results", "ablations", dataset_name)
    os.makedirs(results_dir, exist_ok=True)

    ablation_cfgs = get_ablation_configs(args.config)
    if overrides:
        from configs.config_loader import _apply_override
        for name in ablation_cfgs:
            for ov in overrides: _apply_override(ablation_cfgs[name], ov)
    if args.variants:
        ablation_cfgs = {k: v for k, v in ablation_cfgs.items() if k in args.variants}

    print(f"\nRC-TGAD — ABLATION SUITE")
    print(f"Dataset: {dataset_name} | Variants: {len(ablation_cfgs)} | Seeds: {args.seeds}")
    
    tracker = AblationTracker()
    all_agg = {}
    
    for variant_name, cfg in ablation_cfgs.items():
        variant_results = []
        for seed in args.seeds:
            res = run_variant_seed(variant_name, cfg, seed, args.mock, results_dir)
            variant_results.append(res)
            tracker.add(variant_name, [res["f1_pa"]], [res["auc_pr"]])
            
        all_agg[variant_name] = {
            m: {"mean": float(np.mean([r[m] for r in variant_results])), "std": float(np.std([r[m] for r in variant_results]))}
            for m in ["f1_pa", "auc_pr", "auc_roc", "precision", "recall"]
        }

    agg_path = os.path.join(results_dir, "aggregate.json")
    with open(agg_path, "w") as f: json.dump(all_agg, f, indent=2)

    tex_path = os.path.join(results_dir, "paper_table.tex")
    with open(tex_path, "w") as f: f.write(generate_latex_table(all_agg, dataset_name))
    print(f"\n✅ LaTeX table saved: {tex_path}")

if __name__ == "__main__":
    main()
