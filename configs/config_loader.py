import yaml
import copy
import argparse
from typing import Dict, List, Optional

def load_config(path: str, overrides: Optional[List[str]] = None) -> Dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if overrides:
        for override in overrides:
            _apply_override(cfg, override)

    _validate(cfg)
    return cfg

def _apply_override(cfg: Dict, override: str):
    assert "=" in override, f"Override must be 'key=value', got: {override}"
    key_path, value_str = override.split("=", 1)
    keys  = key_path.strip().split(".")
    value = _cast(value_str.strip())

    d = cfg
    for k in keys[:-1]:
        if k not in d:
            d[k] = {} # Safety: create section if it doesn't exist
        d = d[k]
    d[keys[-1]] = value
    print(f"[Config] Override: {key_path} = {value}")

def _cast(value: str):
    if value.lower() == "null" or value.lower() == "none":
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value 

def _validate(cfg: Dict):
    assert cfg["model"]["lstm_hidden"] == cfg["rag"]["vector_dim"], \
        f"model.lstm_hidden ({cfg['model']['lstm_hidden']}) must equal rag.vector_dim ({cfg['rag']['vector_dim']})"

    assert cfg["model"]["gnn_out_dim"] == cfg["rag"]["vector_dim"], \
        f"model.gnn_out_dim ({cfg['model']['gnn_out_dim']}) must equal rag.vector_dim ({cfg['rag']['vector_dim']})"

    alpha_sum = cfg["rag"]["alpha_1"] + cfg["rag"]["alpha_2"] + cfg["rag"]["alpha_3"]
    assert abs(alpha_sum - 1.0) < 1e-4, \
        f"rag alphas must sum to 1.0, got {alpha_sum:.4f}"

    assert cfg["training"]["seed"] is not None, "training.seed must be set for reproducibility"

def get_ablation_configs(base_config_path: str) -> Dict[str, Dict]:
    ablations = {
        "No Curriculum (Baseline)": [
            "curriculum.enabled=false"
        ],
        "Random Curriculum": [
            "curriculum.enabled=true",
            "rag.alpha_1=0.33",
            "rag.alpha_2=0.33",
            "rag.alpha_3=0.34",
        ],
        "H_temp only": [
            "curriculum.enabled=true",
            "rag.alpha_1=1.0",
            "rag.alpha_2=0.0",
            "rag.alpha_3=0.0",
        ],
        "H_struct only": [
            "curriculum.enabled=true",
            "rag.alpha_1=0.0",
            "rag.alpha_2=1.0",
            "rag.alpha_3=0.0",
        ],
        "H_temp + H_struct (no RAG)": [
            "curriculum.enabled=true",
            "rag.alpha_1=0.5",
            "rag.alpha_2=0.5",
            "rag.alpha_3=0.0",
        ],
        "Full RC-TGAD": [
            "curriculum.enabled=true",
            "rag.alpha_1=0.33",
            "rag.alpha_2=0.33",
            "rag.alpha_3=0.34",
        ],
        # 🛡️ THE FIX: ADDED NO GNN ABLATION
        "No GNN": [
            "curriculum.enabled=true",
            "model.gnn_heads=0",
            "rag.alpha_1=0.5",
            "rag.alpha_2=0.0", # H_struct makes no sense without a graph
            "rag.alpha_3=0.5",
        ],
    }

    configs = {}
    for name, overrides in ablations.items():
        configs[name] = load_config(base_config_path, overrides)
    return configs

def parse_args():
    parser = argparse.ArgumentParser(description="RC-TGAD Experiment Runner")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    return parser.parse_args()
