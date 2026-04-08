# experiments/run_rctgad.py
# PERSON 3 — RC-TGAD Full Model Experiment Runner
# Curriculum ON — uses real hardness scores from Person 2's RAG scorer
#
# MODES:
#   Mock mode  (NOW)   : python experiments/run_rctgad.py --mock
#   Real mode  (Merge) : python experiments/run_rctgad.py --config configs/default.yaml
#
# Compare against baseline:
#   python experiments/run_rctgad.py --mock --compare results/baseline/swat/aggregate.json

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config_loader import load_config
from curriculum.trainer    import Trainer, MockBackbone, MockRAGScorer, MockTemporalGraphDataset
from utils.metrics         import evaluate, AblationTracker

from experiments.run_baseline import (
    load_dataset,
    load_backbone,
    evaluate_on_test
)


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="RC-TGAD Full Model Runner")
    parser.add_argument("--config",   type=str,  default="configs/default.yaml")
    parser.add_argument("--mock",     action="store_true", help="Use mock data/model")
    parser.add_argument("--seeds",    type=int,  nargs="+", default=[42, 43, 44])
    parser.add_argument("--dataset",  type=str,  default=None)
    parser.add_argument("--compare",  type=str,  default=None,
                        help="Path to baseline aggregate.json to print comparison table")
    parser.add_argument("--override", type=str,  nargs="*", default=[])
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# RAG SCORER ADAPTER
#
# WHY THIS EXISTS:
#   Person 2's score_hardness() is a plain FUNCTION (not a class), and it
#   takes extra stateful args: window_errors (running list) and vector_store.
#   Your trainer.py expects an object with .score_hardness() and .get_all_scores()
#   — the Interface B contract.
#
#   This adapter wraps Person 2's function into the class interface your
#   trainer already expects. Zero changes needed in trainer.py.
# ─────────────────────────────────────────────────────────────────────────────

class RealRAGScorer:
    """
    Adapter that wraps Person 2's score_hardness() function into the
    Interface B class contract that trainer.py expects.

    Owns:
      - VectorStore instance (FAISS index)
      - window_errors running list
      - alpha weights and k_neighbors from config
    """

    def __init__(self, cfg: dict):
        from rag.vector_store import VectorStore

        self.vector_store   = VectorStore(dim=cfg["rag"]["vector_dim"])
        self.window_errors  = []           # grows throughout training
        self.alphas         = (
            cfg["rag"]["alpha_1"],
            cfg["rag"]["alpha_2"],
            cfg["rag"]["alpha_3"],
        )
        self.k_neighbors    = cfg["rag"]["k_neighbors"]
        self.gamma          = cfg["rag"]["gamma"]

    def score_hardness(self, z, x, x_hat, node_id, graph, t, 
                       ground_truth_label=0, **kwargs) -> float:
        """
        FIX: Removed hardcoded ground_truth_label=0.
        Now receives the label from the trainer.
        """
        from rag.rag_scorer import score_hardness as p2_score_hardness

        return p2_score_hardness(
            z=z,
            x=x,
            x_hat=x_hat,
            node_id=node_id,
            graph=graph,
            t=t,
            window_errors=self.window_errors,
            vector_store=self.vector_store,
            ground_truth_label=ground_truth_label, # PASSING THROUGH
            alphas=self.alphas,
            k_neighbors=self.k_neighbors,
            gamma=self.gamma,
        )
                           
    def get_all_scores(self, dataset_tuples) -> dict:
        """
        Interface B — pre-compute hardness scores for all (node_id, t) pairs.
        Returns dict {(node_id, t): float}.

        NOTE: Person 2's score_dataset() requires x_windows and graphs dicts
        which we don't have here at pre-compute time. So we return neutral
        scores (0.5) and let scores update dynamically during training via
        score_hardness() calls. The curriculum will still work — it just starts
        with uniform scores in epoch 0 and improves from epoch 1 onward.
        """
        return {
            (node_id, t): 0.5
            for (node_id, t, _) in dataset_tuples
        }

    def reset(self):
        """Call between ablation runs to clear the vector store and error history."""
        self.vector_store.reset()
        self.window_errors.clear()


def load_rag_scorer_real(cfg):
    """
    Returns a RealRAGScorer wrapping Person 2's modules.
    Falls back to MockRAGScorer if Person 2's files aren't present yet.
    """
    try:
        # This import will fail if rag/ folder isn't in repo yet
        from rag.rag_scorer import score_hardness  # noqa: F401 — just testing import
        print("[run_rctgad] Person 2's RAG scorer found — using RealRAGScorer")
        return RealRAGScorer(cfg)
    except ImportError:
        print("[run_rctgad] Person 2 module not found — using MockRAGScorer")
        return MockRAGScorer()


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE SEED RUN
# ─────────────────────────────────────────────────────────────────────────────

def run_single_seed(cfg, seed, mock, results_dir):
    """Run RC-TGAD (curriculum ON) for one seed. Returns test metrics dict."""
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*55}")
    print(f"  RC-TGAD RUN  |  seed={seed}  |  dataset={cfg['data']['dataset']}")
    print(f"  alphas: H_temp={cfg['rag']['alpha_1']}  "
          f"H_struct={cfg['rag']['alpha_2']}  "
          f"H_RAG={cfg['rag']['alpha_3']}")
    print(f"  k_warmup={cfg['curriculum']['k_warmup']}")
    print(f"{'='*55}")

    cfg["training"]["seed"]     = seed
    cfg["logging"]["run_name"]  = f"rctgad_seed{seed}"

    train_data, val_data, test_data = load_dataset(cfg, seed, mock)
    backbone                        = load_backbone(cfg, mock)
    rag_scorer = MockRAGScorer() if mock else load_rag_scorer_real(cfg)

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
        use_curriculum=True,
        device=device
    )

    history = trainer.train(
        val_dataset=val_data,
        save_dir=os.path.join(results_dir, f"seed{seed}")
    )

    print(f"\n[RC-TGAD] Evaluating on test set (seed={seed})...")
    test_results = evaluate_on_test(backbone, test_data, cfg, device)

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

    return test_results, history


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(baseline_path, rctgad_results):
    if not os.path.exists(baseline_path):
        print(f"\n[Compare] Baseline file not found: {baseline_path}")
        return

    with open(baseline_path) as f:
        baseline = json.load(f)

    rctgad_agg = {
        metric: {
            "mean": float(np.mean([r[metric] for r in rctgad_results])),
            "std":  float(np.std([r[metric]  for r in rctgad_results])),
        }
        for metric in ["f1_pa", "auc_pr", "precision", "recall"]
    }

    print(f"\n{'='*65}")
    print(f"  BASELINE  vs  RC-TGAD  COMPARISON")
    print(f"{'='*65}")
    print(f"  {'Metric':<14} {'Baseline':>18} {'RC-TGAD':>18} {'Delta':>8}")
    print(f"  {'-'*60}")

    for metric in ["f1_pa", "auc_pr", "precision", "recall"]:
        b_mean = baseline[metric]["mean"]
        b_std  = baseline[metric]["std"]
        r_mean = rctgad_agg[metric]["mean"]
        r_std  = rctgad_agg[metric]["std"]
        delta  = r_mean - b_mean
        arrow  = "▲" if delta > 0 else "▼"
        print(f"  {metric:<14} "
              f"{b_mean:.4f}±{b_std:.4f}   "
              f"{r_mean:.4f}±{r_std:.4f}   "
              f"{arrow}{abs(delta):.4f}")

    print(f"{'='*65}")
    print(f"  F1-PA and AUC-PR are the primary paper metrics.")


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
    results_dir  = os.path.join("results", "rctgad", dataset_name)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\nRC-TGAD — FULL MODEL EXPERIMENT")
    print(f"Dataset  : {dataset_name}")
    print(f"Seeds    : {args.seeds}")
    print(f"Epochs   : {cfg['training']['epochs']}")
    print(f"k_warmup : {cfg['curriculum']['k_warmup']}")
    print(f"Alphas   : {cfg['rag']['alpha_1']}, {cfg['rag']['alpha_2']}, {cfg['rag']['alpha_3']}")
    print(f"Mock mode: {args.mock}")

    all_results   = []
    all_histories = []
    for seed in args.seeds:
        result, history = run_single_seed(cfg, seed, args.mock, results_dir)
        all_results.append(result)
        all_histories.append(history)

    print(f"\n{'='*55}")
    print(f"  RC-TGAD FINAL RESULTS  ({dataset_name}, {len(args.seeds)} seeds)")
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

    baseline_path = args.compare or os.path.join(
        "results", "baseline", dataset_name, "aggregate.json"
    )
    print_comparison(baseline_path, all_results)


if __name__ == "__main__":
    main()
