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
            
        # 2. Initialize Metrics CSV (⚡ CHANGED F1 TO AUC-PR)
        with open(self.metrics_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train_Loss', 'Val_AUC_PR', 'Max_Hardness', 'Time_Sec'])
            
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
            
    def log_epoch(self, epoch, loss, val_auc_pr, max_hardness, time_sec):
        """Writes the epoch metrics to the CSV for plotting later."""
        with open(self.metrics_csv, 'a', newline='') as f:
            # ⚡ Saves exact floats to CSV so you can graph them in Python/Excel later
            csv.writer(f).writerow([epoch, f"{loss:.6f}", f"{val_auc_pr:.4f}", f"{max_hardness:.4f}", f"{time_sec:.2f}"])

    def log_curriculum_pacing(self, epoch, current_k, total_n, max_hardness):
        """Logs the pacing event for the Curriculum Scheduler."""
        percentage = (current_k / total_n) * 100
        self.log_event(f"📈 [Curriculum Update] Epoch {epoch} | Unlocked {current_k}/{total_n} samples ({percentage:.1f}%) | Max Hardness: {max_hardness:.4f}")
    
    def log_curriculum_scores(self, detailed_scores):
        """Bulk writes individual hardness scores to prevent Kaggle IO bottlenecks."""
        with open(self.scores_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            # detailed_scores is a list of lists: [[epoch, idx, h_temp, h_struct, h_rag, h_total], ...]
            writer.writerows(detailed_scores)
        # ⚡ Print statement moved to the correct function with the correct syntax
        print(f"💾 [Logger] Bulk saved {len(detailed_scores)} hardness rows to {self.scores_csv.name}")

    def verify_disk_writes(self):
        """Audits the hard drive to prove files were written and have data."""
        print("\n" + "="*60)
        print("💽 IEEE PAPER LOG AUDIT: VERIFYING DISK WRITES")
        print("="*60)
        
        files_to_check = {
            "Hyperparameters": self.config_json,
            "Architecture": self.arch_txt,
            "Events Log": self.events_log,
            "Epoch Metrics": self.metrics_csv,
            "Detailed Scores (Bulk)": self.scores_csv
        }
        
        all_good = True
        for name, filepath in files_to_check.items():
            if filepath.exists():
                size_kb = filepath.stat().st_size / 1024
                if size_kb > 0:
                    # If it's over 1000 KB, print in MB instead
                    if size_kb > 1000:
                        print(f"✅ {name:22} : {filepath.name} ({(size_kb/1024):.2f} MB)")
                    else:
                        print(f"✅ {name:22} : {filepath.name} ({size_kb:.2f} KB)")
                else:
                    print(f"⚠️ {name:22} : {filepath.name} (CREATED BUT EMPTY!)")
                    all_good = False
            else:
                print(f"❌ {name:22} : {filepath.name} (MISSING FROM DISK!)")
                all_good = False
                
        print("-" * 60)
        if all_good:
            print("🚀 SUCCESS: All files are securely written and contain data.")
        else:
            print("⚠️ WARNING: Some logs failed to write.")
        print("="*60 + "\n")
