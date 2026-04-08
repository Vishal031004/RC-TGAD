import os
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from sklearn.preprocessing import RobustScaler
from .base_dataset import BaseTimeSeriesDataset

def load_swat(data_dir: str = "data/raw", window: int = 30, stride: int = 1, val_ratio: float = 0.2, **kwargs):
    """
    Direct-from-Raw SWaT Loader.
    No external preprocessing script needed. Handles everything causally in-memory.
    """
    clean_dir = Path(data_dir)
    # Load directly from the RAW files
    normal_path = clean_dir / "SWaT_Dataset_Normal_v1.csv"
    attack_path = clean_dir / "SWaT_Dataset_Attack_v0.csv"

    print(f"\n[SWaT Loader] Loading RAW Data | Val Ratio: {val_ratio}")

    # The raw files have a header on row 1 (0-indexed is row 1)
    normal_df = pd.read_csv(normal_path, sep=",", header=1, low_memory=False)
    attack_df = pd.read_csv(attack_path, sep=",", header=1, low_memory=False)

    normal_df.columns = normal_df.columns.str.strip()
    attack_df.columns = attack_df.columns.str.strip()

    feature_cols = [c for c in normal_df.columns if c not in ["Timestamp", "Normal/Attack"]]

    # 🛡️ NaN Scrubber & Forward/Backward Fill
    for df in [normal_df, attack_df]:
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        df[feature_cols] = df[feature_cols].ffill().bfill()

    # ==========================================
    # 🛡️ FIX: CAUSAL SMOOTHING
    # ==========================================
    print("🌊 Applying causal smoothing (Shifted Rolling Mean)...")
    normal_smooth = normal_df[feature_cols].rolling(window=5).mean().shift(1).fillna(0).values.astype(np.float32)
    attack_smooth = attack_df[feature_cols].rolling(window=5).mean().shift(1).fillna(0).values.astype(np.float32)

    # ==========================================
    # 🛡️ FIX: CHRONOLOGICAL SPLIT
    # ==========================================
    split_idx = int(len(normal_smooth) * (1 - val_ratio))
    
    train_sig = normal_smooth[:split_idx]
    train_lbl = np.zeros((len(train_sig), len(feature_cols)), dtype=np.int64)

    val_sig = normal_smooth[split_idx:]
    val_lbl = np.zeros((len(val_sig), len(feature_cols)), dtype=np.int64)

    test_sig = attack_smooth
    raw_labels_test = attack_df["Normal/Attack"].astype(str).str.strip().values
    row_labels_test = (raw_labels_test != "Normal").astype(np.int64)
    test_lbl = np.repeat(row_labels_test[:, None], test_sig.shape[1], axis=1)

    # ==========================================
    # 🛡️ FIX: REMOVE ZERO-VARIANCE FEATURES (Computed on Train ONLY)
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
    # 🛡️ FIX: ROBUST SCALING & CLIPPING (Fitted on Train ONLY)
    # ==========================================
    print("⚖️ Applying Robust Scaling and Hard Clipping [-5, 5]...")
    scaler = RobustScaler()
    train_sig = scaler.fit_transform(train_sig)
    val_sig   = scaler.transform(val_sig)
    test_sig  = scaler.transform(test_sig)

    train_sig = np.clip(train_sig, -5.0, 5.0)
    val_sig   = np.clip(val_sig, -5.0, 5.0)
    test_sig  = np.clip(test_sig, -5.0, 5.0)

    # Pass dummy normalizers to BaseDataset since we already scaled it perfectly here
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
