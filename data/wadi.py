import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from .base_dataset import BaseTimeSeriesDataset

def load_wadi(data_dir: str = "data/raw", window: int = 60, stride: int = 2, val_ratio: float = 0.2, **kwargs):
    """
    Direct-from-Raw WADI Loader.
    Architecturally cloned from swat.py, featuring Causal Smoothing, Dead-Sensor Dropping, 
    and RobustScaling, while handling WADI's -1 labels and NaN dropouts.
    """
    clean_dir = Path(data_dir)
    normal_path = clean_dir / "WADI_14days_new.csv"
    attack_path = clean_dir / "WADI_attackdataLABLE.csv"

    print(f"\n🌊 [WADI Loader] Loading RAW Data | Val Ratio: {val_ratio}")

    # WADI sometimes has units in row 1, skipping it ensures clean float conversion
    try:
        normal_df = pd.read_csv(normal_path, sep=",", low_memory=False)
    except:
        normal_df = pd.read_csv(normal_path, sep=",", header=1, low_memory=False)
        
    try:
        attack_df = pd.read_csv(attack_path, sep=",", low_memory=False)
    except:
        attack_df = pd.read_csv(attack_path, sep=",", header=1, low_memory=False)

    normal_df.columns = normal_df.columns.str.strip()
    attack_df.columns = attack_df.columns.str.strip()

    # 1. Identify the weird WADI attack label column dynamically
    label_cols = [c for c in attack_df.columns if 'label' in str(c).lower() or 'attack' in str(c).lower()]
    label_col = label_cols[0] if label_cols else attack_df.columns[-1]
    
    # 2. Define Features (drop metadata)
    drop_cols = ["Row", "Date", "Time", label_col, "Timestamp"]
    feature_cols = [c for c in normal_df.columns if c in attack_df.columns and c not in drop_cols]

    # 🛡️ NaN Scrubber & Forward/Backward Fill (WADI has LOTS of these)
    print("🩹 Patching offline WADI sensor gaps...")
    for df in [normal_df, attack_df]:
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        df[feature_cols] = df[feature_cols].ffill().bfill().fillna(0)

    # ==========================================
    # 🛡️ CAUSAL SMOOTHING
    # ==========================================
    print("🌊 Applying causal smoothing (Shifted Rolling Mean)...")
    normal_smooth = normal_df[feature_cols].rolling(window=5).mean().shift(1).fillna(0).values.astype(np.float32)
    attack_smooth = attack_df[feature_cols].rolling(window=5).mean().shift(1).fillna(0).values.astype(np.float32)

    # ==========================================
    # 🛡️ CHRONOLOGICAL SPLIT
    # ==========================================
    split_idx = int(len(normal_smooth) * (1 - val_ratio))
    
    train_sig = normal_smooth[:split_idx]
    train_lbl = np.zeros((len(train_sig), len(feature_cols)), dtype=np.int64)

    val_sig = normal_smooth[split_idx:]
    val_lbl = np.zeros((len(val_sig), len(feature_cols)), dtype=np.int64)

    test_sig = attack_smooth
    
    # Handle WADI's -1 labels
    raw_labels_test = attack_df[label_col].values
    if -1 in raw_labels_test:
        row_labels_test = np.where(raw_labels_test == -1, 1, 0).astype(np.int64)
    else:
        row_labels_test = raw_labels_test.astype(np.int64)
        
    test_lbl = np.repeat(row_labels_test[:, None], test_sig.shape[1], axis=1)

    # ==========================================
    # 🛡️ REMOVE ZERO-VARIANCE FEATURES (Computed on Train ONLY)
    # ==========================================
    print("🧠 Filtering zero-variance features...")
    stds = train_sig.std(axis=0)
    valid_idx = np.where(stds > 1e-8)[0]
    
    print(f"Removed {len(feature_cols) - len(valid_idx)} dead sensors.")
    train_sig = train_sig[:, valid_idx]
    val_sig = val_sig[:, valid_idx]
    test_sig = test_sig[:, valid_idx]
    feature_cols = [feature_cols[i] for i in valid_idx]

    # ==========================================
    # 🛡️ ROBUST SCALING & CLIPPING (Fitted on Train ONLY)
    # ==========================================
    print("⚖️ Applying Robust Scaling and Hard Clipping [-5, 5]...")
    scaler = RobustScaler()
    train_sig = scaler.fit_transform(train_sig)
    val_sig   = scaler.transform(val_sig)
    test_sig  = scaler.transform(test_sig)

    train_sig = np.clip(train_sig, -5.0, 5.0).astype(np.float32)
    val_sig   = np.clip(val_sig, -5.0, 5.0).astype(np.float32)
    test_sig  = np.clip(test_sig, -5.0, 5.0).astype(np.float32)

    # Pass dummy normalizers to BaseDataset
    dummy_mean = np.zeros(train_sig.shape[1], dtype=np.float32)
    dummy_std = np.ones(train_sig.shape[1], dtype=np.float32)

    # Create Datasets
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

    print(f"[Dataset] Train samples: {len(train_ds):,}")
    print(f"[Dataset] Val samples:   {len(val_ds):,}")
    print(f"[Dataset] Test samples:  {len(test_ds):,}\n")

    return train_ds, val_ds, test_ds, feature_cols
