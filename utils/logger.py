import json
import csv
import datetime
from pathlib import Path

class DeepResearchLogger:
    """The ultimate IEEE paper logging suite. Leaves no black boxes."""
    def __init__(self, save_dir, config_dict):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # All 5 files required for the paper
        self.config_json = self.save_dir / "hyperparameters.json"
        self.metrics_csv = self.save_dir / "metrics.csv"
        self.events_log = self.save_dir / "training_events.log"
        self.arch_txt = self.save_dir / "architecture_details.txt"
        self.scores_csv = self.save_dir / "curriculum_scores.csv"
        
        # 1. Freeze Hyperparameters
        with open(self.config_json, 'w') as f:
            json.dump(config_dict, f, indent=4)
            
        # 2. Initialize Metrics CSV
        with open(self.metrics_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train_Loss', 'Val_F1', 'Time_Sec'])
            
        # 3. Initialize Curriculum Scores CSV
        with open(self.scores_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Sample_Idx', 'H_temp', 'H_struct', 'H_rag', 'H_total'])

        self.log_event("🚀 Deep Research Logger Initialized.")

    def log_event(self, text):
        """Timestamps and saves major events to the text log."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {text}"
        print(log_entry) # Still print to Kaggle console
        with open(self.events_log, 'a') as f:
            f.write(log_entry + "\n")

    def log_architecture(self, model, dataset):
        """Dumps the exact GNN/LSTM dimensions and math for the paper."""
        with open(self.arch_txt, 'w') as f:
            f.write("=== RC-TGAD SPATIO-TEMPORAL BACKBONE ===\n\n")
            f.write(f"Total Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
            d_in = dataset[0]['x'].shape[-1] if hasattr(dataset, '__getitem__') else "N/A"
            f.write(f"Input Node Features (d_in): {d_in}\n")
            f.write(f"Graph Nodes (Sensors): {getattr(dataset, 'N', 'N/A')}\n")
            f.write("\n=== DETAILED PYTORCH MODULE ===\n")
            f.write(str(model))
        self.log_event("💾 Architecture details saved.")
            
    def log_epoch(self, epoch, loss, val_f1, time_sec):
        """Writes the epoch metrics to the CSV for plotting later."""
        with open(self.metrics_csv, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f"{loss:.6f}", f"{val_f1:.4f}", f"{time_sec:.2f}"])

    def log_curriculum_update(self, epoch, h_temp, h_struct, h_rag, h_total, current_k, total_n):
        percentage = (current_k / total_n) * 100
        self.log_event(f"📈 [Curriculum Update] Epoch {epoch} | Unlocked {current_k}/{total_n} samples ({percentage:.1f}%) | Max Hardness: {float(np.max(h_total)):.4f}")
        
        # 🛡️ SAFETY: Move to CPU and convert to numpy once to prevent VRAM leakage
        h_temp_np = h_temp.detach().cpu().numpy() if torch.is_tensor(h_temp) else h_temp
        h_struct_np = h_struct.detach().cpu().numpy() if torch.is_tensor(h_struct) else h_struct
        h_rag_np = h_rag.detach().cpu().numpy() if torch.is_tensor(h_rag) else h_rag
        h_total_np = h_total.detach().cpu().numpy() if torch.is_tensor(h_total) else h_total

        with open(self.scores_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            for idx in range(0, len(h_total_np), 10): 
                writer.writerow([
                    epoch, 
                    idx, 
                    f"{float(h_temp_np[idx]):.4f}", 
                    f"{float(h_struct_np[idx]):.4f}", 
                    f"{float(h_rag_np[idx]):.4f}", 
                    f"{float(h_total_np[idx]):.4f}"
                ])
