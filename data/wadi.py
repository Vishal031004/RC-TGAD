import os
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from .base_dataset import BaseTimeSeriesDataset

def load_wadi(data_dir: str = "data/raw", window: int = 60, stride: int = 2, val_ratio: float = 0.2, **kwargs):
    """
    Direct-from-Raw WADI Loader.
    Architecturally cloned from swat.py, featuring Causal Smoothing, Dead-Sensor Dropping, 
    RobustScaling, Memory Rescue, and FOOLPROOF majority-class label detection.
    """
    clean_dir = Path(data_dir)
    normal_path = clean_dir / "WADI_14days_new.csv"
    attack_path = clean_dir / "WADI_attackdataLABLE.csv"

    print(f"\n🌊 [WADI Loader] Loading RAW Data | Val Ratio: {val_ratio}")

    # ==========================================
    # 🛡️ 1. ACTIVE HEADER HUNTER & LOAD
    # ==========================================
    normal_df = pd.read_csv(normal_path, sep=",", low_memory=False)
    clean_normal_cols = [str(c).replace('\\', '').strip().upper() for c in normal_df.columns]
    if '1_AIT_001_PV' not in clean_normal_cols:
        print("⚠️ Bad headers detected in Normal data. Shifting down 1 row...")
        normal_df = pd.read_csv(normal_path, sep=",", header=1, low_memory=False)
    
    attack_df = pd.read_csv(attack_path, sep=",", low_memory=False)
    clean_attack_cols = [str(c).replace('\\', '').strip().upper() for c in attack_df.columns]
    if '1_AIT_001_PV' not in clean_attack_cols:
        print("⚠️ Garbage top row detected in Attack data. Shifting headers down 1 row...")
        attack_df = pd.read_csv(attack_path, sep=",", header=1, low_memory=False)

    # ==========================================
    # 🛡️ 2. AGGRESSIVE COLUMN ALIGNMENT
    # ==========================================
    normal_df.columns = [str(c).replace('\\', '').strip().upper() for c in normal_df.columns]
    attack_df.columns = [str(c).replace('\\', '').strip().upper() for c in attack_df.columns]

    label_cols = [c for c in attack_df.columns if 'LABEL' in c or 'ATTACK' in c]
    label_col = label_cols[0] if label_cols else attack_df.columns[-1]
    
    drop_cols = ["ROW", "DATE", "TIME", label_col, "TIMESTAMP"]
    feature_cols = [c for c in normal_df.columns if c in attack_df.columns and c not in drop_cols]
    
    print(f"🔗 [WADI Alignment] Successfully locked onto {len(feature_cols)} sensor columns!")

    # ==========================================
    # 🛡️ 3. NaN SCRUBBER
    # ==========================================
    print("🩹 Patching offline WADI sensor gaps...")
    for df in [normal_df, attack_df]:
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        df[feature_cols] = df[feature_cols].ffill().bfill().fillna(0)

    # ==========================================
    # 🛡️ 4. FOOLPROOF LABEL EXTRACTION
    # ==========================================
    print(f"🎯 Scrubbing text from label column: {label_col}")
    clean_labels = pd.to_numeric(attack_df[label_col], errors='coerce')
    valid_labels = clean_labels.dropna().values
    
    # Dynamically find the majority class (Normal operation)
    unique_vals, counts = np.unique(valid_labels, return_counts=True)
    normal_val = unique_vals[np.argmax(counts)]
    print(f"🕵️ Detected '{normal_val}' as the Normal baseline. Everything else is an Attack!")

    clean_labels = clean_labels.fillna(normal_val).values
    row_labels_test = np.where(clean_labels != normal_val, 1, 0).astype(np.int64)
    print(f"🚨 Successfully found {row_labels_test.sum()} attack moments in the test set!")

    # ==========================================
    # 🛡️ 5. CAUSAL SMOOTHING
    # ==========================================
    print("🌊 Applying causal smoothing (Shifted Rolling Mean)...")
    normal_smooth = normal_df[feature_cols].rolling(window=5).mean().shift(1).fillna(0).values.astype(np.float32)
    attack_smooth = attack_df[feature_cols].rolling(window=5).mean().shift(1).fillna(0).values.astype(np.float32)

    # ==========================================
    # 🛡️ 6. MEMORY RESCUE (GC & DOWNSAMPLING)
    # ==========================================
    print("🧹 Triggering Garbage Collection & Downsampling to save Kaggle RAM...")
    del normal_df
    del attack_df
    gc.collect() 

    D_RATE = 5 
    normal_smooth = normal_smooth[::D_RATE]
    attack_smooth = attack_smooth[::D_RATE]
    row_labels_test = row_labels_test[::D_RATE]

    # ==========================================
    # 🛡️ 7. CHRONOLOGICAL SPLIT
    # ==========================================
    split_idx = int(len(normal_smooth) * (1 - val_ratio))
    
    train_sig = normal_smooth[:split_idx]
    train_lbl = np.zeros((len(train_sig), len(feature_cols)), dtype=np.int64)

    val_sig = normal_smooth[split_idx:]
    val_lbl = np.zeros((len(val_sig), len(feature_cols)), dtype=np.int64)

    test_sig = attack_smooth
    test_lbl = np.repeat(row_labels_test[:, None], test_sig.shape[1], axis=1)

    # ==========================================
    # 🛡️ 8. REMOVE ZERO-VARIANCE FEATURES
    # ==========================================
    print("🧠 Filtering zero-variance features...")
    stds = train_sig.std(axis=0)
    valid_idx = np.where(stds > 1e-8)[0]
    
    train_sig = train_sig[:, valid_idx]
    val_sig = val_sig[:, valid_idx]
    test_sig = test_sig[:, valid_idx]
    
    train_lbl = train_lbl[:, valid_idx]
    val_lbl = val_lbl[:, valid_idx]
    test_lbl = test_lbl[:, valid_idx]
    
    feature_cols = [feature_cols[i] for i in valid_idx]

    # ==========================================
    # 🛡️ 9. ROBUST SCALING & CLIPPING
    # ==========================================
    print("⚖️ Applying Robust Scaling and Hard Clipping [-5, 5]...")
    scaler = RobustScaler()
    train_sig = scaler.fit_transform(train_sig)
    val_sig   = scaler.transform(val_sig)
    test_sig  = scaler.transform(test_sig)

    train_sig = np.clip(train_sig, -5.0, 5.0).astype(np.float32)
    val_sig   = np.clip(val_sig, -5.0, 5.0).astype(np.float32)
    test_sig  = np.clip(test_sig, -5.0, 5.0).astype(np.float32)

    dummy_mean = np.zeros(train_sig.shape[1], dtype=np.float32)
    dummy_std = np.ones(train_sig.shape[1], dtype=np.float32)

    print("🚀 Building Torch Datasets (Memory usage stable)...")
    train_ds = BaseTimeSeriesDataset(
        train_sig, train_lbl, window=window, stride=stride,
        norm_mean=dummy_mean, norm_std=dummy_std
    )
    
    val_ds = BaseTimeSeriesDataset(
        val_sig, val_lbl, window=window, stride=stride,
        norm_mean=dummy_mean, norm_std=dummy_std,
        graph=train_ds.graph
    )
    
    test_ds = BaseTimeSeriesDataset(
        test_sig, test_lbl, window=window, stride=stride,
        norm_mean=dummy_mean, norm_std=dummy_std,
        graph=train_ds.graph
    )

    return train_ds, val_ds, test_ds, feature_cols
