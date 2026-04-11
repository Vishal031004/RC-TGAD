# experiments/ablations.py
# PERSON 3 — Full Ablation Suite
# RC-TGAD: Runs all 6 ablation variants and produces the paper table

import os
import sys
import json
import argparse
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config_loader    import load_config, get_ablation_configs
from curriculum.trainer       import Trainer, MockBackbone, MockRAGScorer, MockTemporalGraphDataset
from utils.metrics            import evaluate, AblationTracker
from experiments.run_baseline import load_dataset, load_backbone, evaluate_on_test
from experiments.run_rctgad   import RealRAGScorer

# Import the ultimate IEEE logger we just created
from utils.logger             import DeepResearchLogger

# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="RC-TGAD Ablation Runner")
    parser.add_argument("--config",   type=str, default="configs/default.yaml")
    parser.add_argument("--mock",     action="store_true", help="Use mock data/model")
    parser.add_argument("--seeds",    type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--dataset",  type=str, default=None)
    parser.add_argument("--variants", type=str, nargs="*", default=None,
                        help="Run only specific variants by name (default: all 6)")
    parser.add_argument("--override", type=str, nargs="*", default=[])
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# RANDOM CURRICULUM SCORER
# ─────────────────────────────────────────────────────────────────────────────

class RandomHardnessScorer:
    """Assigns random hardness scores to prove the math matters."""
    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)

    def score_hardness(self, z, x, x_hat, node_id, graph, t, window_errors=None):
        return float(self.rng.random())

    def get_all_scores(self, dataset):
        return {(node_id, t): float(self.rng.random()) for (node_id, t, _) in dataset}


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE VARIANT + SINGLE SEED RUN
# ─────────────────────────────────────────────────────────────────────────────

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

    # 1. Initialize the Ultimate Paper Logger
    paper_logger = DeepResearchLogger(save_dir=variant_dir, config_dict=cfg)

    # Load Data & Model
    train_data, val_data, test_data = load_dataset(cfg, seed, mock)
    backbone = load_backbone(cfg, mock)

    # 2. Dump the architecture details for the methodology section
    paper_logger.log_architecture(backbone, train_data)

    device = cfg["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    curriculum_enabled = cfg["curriculum"]["enabled"]

    if not curriculum_enabled:
        rag_scorer = MockRAGScorer()
    elif variant_name == "Random Curriculum":
        rag_scorer = RandomHardnessScorer(seed=seed)
    else:
        try:
            rag_scorer = RealRAGScorer(cfg)
            rag_scorer.reset()
        except Exception:
            rag_scorer = MockRAGScorer(seed=seed)

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
        logger=paper_logger  # <--- PASS THE LOGGER TO THE TRAINER
    )

    history = trainer.train(val_dataset=val_data, save_dir=variant_dir)

    test_results = evaluate_on_test(backbone, test_data, cfg, device, val_dataset=val_data)

    with open(result_path, "w") as f:
        json.dump({**test_results, "history": history}, f, indent=2)

    return {k: test_results[k] for k in ["f1_pa", "auc_pr", "auc_roc", "precision", "recall"]}


# ─────────────────────────────────────────────────────────────────────────────
# LATEX TABLE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_latex_table(agg, dataset_name):
    metrics   = ["f1_pa", "auc_pr", "precision", "recall"]
    col_names = ["F1-PA", "AUC-PR", "Precision", "Recall"]
    variants  = list(agg.keys())

    best = {}
    for m in metrics:
        best[m] = max(agg[v][m]["mean"] for v in variants)

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation study on " + dataset_name.upper() + r" dataset.}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Variant} & " + " & ".join(f"\\textbf{{{c}}}" for c in col_names) + r" \\")
    lines.append(r"\midrule")

    for variant in variants:
        row_vals = []
        for m in metrics:
            mean = agg[variant][m]["mean"]
            std  = agg[variant][m]["std"]
            val  = f"{mean:.4f}$\\pm${std:.4f}"
            if abs(mean - best[m]) < 1e-6:
                val = f"\\textbf{{{val}}}"
            row_vals.append(val)
        safe_variant = variant.replace("_", r"\_").replace("&", r"\&")
        lines.append(f"{safe_variant} & " + " & ".join(row_vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    overrides = list(args.override or [])
    if args.dataset:
        overrides.append(f"data.dataset={args.dataset}")
    if args.mock:
        overrides += ["training.epochs=20", "training.device=cpu"]

    base_cfg     = load_config(args.config, overrides if overrides else None)
    dataset_name = base_cfg["data"]["dataset"]
    results_dir  = os.path.join("results", "ablations", dataset_name)
    os.makedirs(results_dir, exist_ok=True)

    ablation_cfgs = get_ablation_configs(args.config)

    if overrides:
        for name in ablation_cfgs:
            for ov in overrides:
                from configs.config_loader import _apply_override
                _apply_override(ablation_cfgs[name], ov)

    if args.variants:
        ablation_cfgs = {k: v for k, v in ablation_cfgs.items() if k in args.variants}

    print(f"\nRC-TGAD — ABLATION SUITE (PAPER READY)")
    print(f"Dataset   : {dataset_name}")
    print(f"Variants  : {len(ablation_cfgs)}")
    print(f"Seeds     : {args.seeds}")
    print(f"Epochs    : {base_cfg['training']['epochs']}")
    print(f"Results   : {results_dir}")
    print(f"\nVariants to run:")
    for i, name in enumerate(ablation_cfgs):
        cfg = ablation_cfgs[name]
        a1, a2, a3 = cfg['rag']['alpha_1'], cfg['rag']['alpha_2'], cfg['rag']['alpha_3']
        enabled = cfg['curriculum']['enabled']
        print(f"  {i+1}. {name:<38} curriculum={str(enabled):<5} α=({a1:.2f},{a2:.2f},{a3:.2f})")

    tracker    = AblationTracker()
    all_agg    = {}
    total_runs = len(ablation_cfgs) * len(args.seeds)
    run_count  = 0
    t_total    = time.time()

    for variant_name, cfg in ablation_cfgs.items():
        print(f"\n{'─'*55}")
        print(f"  Variant: {variant_name}")
        print(f"{'─'*55}")

        variant_results = []
        for seed in args.seeds:
            run_count += 1
            print(f"\n  [{run_count}/{total_runs}] seed={seed}")
            t0 = time.time()
            result = run_variant_seed(variant_name, cfg, seed, args.mock, results_dir)
            variant_results.append(result)
            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s  |  "
                  f"F1-PA={result['f1_pa']:.4f}  AUC-PR={result['auc_pr']:.4f}")

        # 🛡️ FIX: Pass ACTUAL results to the tracker
        for result in variant_results:
            tracker.add(
                variant_name,
                f1_list=[result["f1_pa"]],
                auc_list=[result["auc_pr"]] # Ensure your tracker.add takes this arg
            )

        all_agg[variant_name] = {
            metric: {
                "mean": float(np.mean([r[metric] for r in variant_results])),
                "std":  float(np.std([r[metric]  for r in variant_results])),
                "runs": [r[metric] for r in variant_results]
            }
            for metric in ["f1_pa", "auc_pr", "auc_roc", "precision", "recall"]
        }

    total_time = time.time() - t_total
    print(f"\n\nAll {total_runs} runs complete in {total_time/60:.1f} minutes")

    print(f"\n{'='*75}")
    print(f"  ABLATION RESULTS  —  {dataset_name.upper()}  ({len(args.seeds)} seeds)")
    print(f"{'='*75}")
    print(f"  {'Variant':<38} {'F1-PA':>12} {'AUC-PR':>10} {'Precision':>10} {'Recall':>8}")
    print(f"  {'─'*70}")

    for variant, agg in all_agg.items():
        f1   = agg['f1_pa']['mean']
        f1s  = agg['f1_pa']['std']
        ap   = agg['auc_pr']['mean']
        prec = agg['precision']['mean']
        rec  = agg['recall']['mean']
        print(f"  {variant:<38} {f1:.4f}±{f1s:.4f}  {ap:.4f}  {prec:.4f}  {rec:.4f}")

    print(f"{'='*75}")
    
    agg_path = os.path.join(results_dir, "aggregate.json")
    with open(agg_path, "w") as f:
        json.dump(all_agg, f, indent=2)

    latex   = generate_latex_table(all_agg, dataset_name)
    tex_path = os.path.join(results_dir, "paper_table.tex")
    with open(tex_path, "w") as f:
        f.write(latex)
    print(f"\n  LaTeX table saved: {tex_path}")

if __name__ == "__main__":
    main()
