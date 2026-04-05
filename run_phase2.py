import torch
import os
import sys
from collections import OrderedDict
import backbone.backbone
from backbone.backbone import Backbone as OriginalBackbone

# ====================================================================
# 🛑 THE BRAKES: Standard Gradient Clipper 🛑
# Prevents layer-by-layer multiplication from exploding
# ====================================================================
orig_step = torch.optim.Optimizer.step
def safe_step(self, *args, **kwargs):
    params = []
    for group in self.param_groups:
        for p in group['params']:
            if p.grad is not None:
                # If a completely broken record slips through, ignore it
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                params.append(p)
                
    # THE SPEED LIMIT
    if params:
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        
    return orig_step(self, *args, **kwargs)

torch.optim.Optimizer.step = safe_step

# ====================================================================
# 🚀 PHASE 2 WEIGHT INJECTION
# ====================================================================
class Phase2Backbone(OriginalBackbone):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        weight_path = "results/rctgad/swat/seed42/phase1_expert.pt"
        
        if os.path.exists(weight_path):
            print("\n" + "="*50)
            print("🚀 PHASE 2 FORCE-LOAD: Injecting Phase 1 weights...")
            ckpt = torch.load(weight_path, map_location='cpu')
            state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            self.load_state_dict(new_state_dict)
            print("✅ EXPERT WEIGHTS INJECTED SUCCESSFULLY")
            print("="*50 + "\n")

backbone.backbone.Backbone = Phase2Backbone

import experiments.run_rctgad as run_script

if __name__ == "__main__":
    run_script.main()
