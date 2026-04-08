import os
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from .base_dataset import BaseTimeSeriesDataset

def load_swat(data_dir: str = "data/ready", window: int = 30, stride: int = 1, val_ratio: float = 0.2, **kwargs):
    """
    Updated SWaT Loader: Fixes Causal Leakage and Normalization stats.
    """
    norm_type = "Robust"   
    clean_dir = Path(data_dir)
    normal_path = clean_dir / f"SWaT_Normal_{norm_type}.csv"
    attack_path = clean_dir / f"SWaT_Attack_{norm_type}.csv"

    print(f"\n[Causal Load] Dataset: {norm_type.upper()} | Val Ratio: {val_ratio}")

    # Load raw dataframes
    normal_df = pd.read_csv(normal_path, low_memory=False)
    attack_df = pd.read_csv(attack_path, low_memory=False)

    feature_cols = [c for c in normal_df.columns if c not in ["Timestamp", "Normal/Attack"]]

    # ==========================================
    # 🛡️ FIX 1: CAUSAL SMOOTHING
    # ==========================================
    # We apply smoothing but shift it by 1 to ensure no future-lookahead.
    # This addresses the 'violation of causal constraints' feedback.
    normal_smooth = normal_df[feature_cols].rolling(window=5).mean().shift(1).fillna(0).values.astype(np.float32)
    attack_smooth = attack_df[feature_cols].rolling(window=5).mean().shift(1).fillna(0).values.astype(np.float32)

    # Scrubber for safety
    normal_smooth = np.nan_to_num(normal_smooth, nan=0.0)
    attack_smooth = np.nan_to_num(attack_smooth, nan=0.0)

    # ==========================================
    # 🛡️ FIX 2: CHRONOLOGICAL VALIDATION SPLIT
    # ==========================================
    # Split the normal data into Training and Validation segments sequentially.
    split_idx = int(len(normal_smooth) * (1 - val_ratio))
    
    train_sig = normal_smooth[:split_idx]
    train_lbl = np.zeros((len(train_sig), len(feature_cols)), dtype=np.int64)

    val_sig = normal_smooth[split_idx:]
    val_lbl = np.zeros((len(val_sig), len(feature_cols)), dtype=np.int64)

    # Test set uses the attack data
    test_sig = attack_smooth
    raw_labels_test = attack_df["Normal/Attack"].astype(str).str.strip().values
    row_labels_test = (raw_labels_test != "Normal").astype(np.int64)
    test_lbl = np.repeat(row_labels_test[:, None], test_sig.shape[1], axis=1)

    # ==========================================
    # 🛡️ FIX 3: LEAKAGE-FREE NORMALIZATION STATS
    # ==========================================
    # Calculate stats ONLY from the Training segment.
    train_mean = train_sig.mean(axis=0)
    train_std = train_sig.std(axis=0) + 1e-8

    # Create Datasets using identical normalization parameters
    train_ds = BaseTimeSeriesDataset(
        train_sig, train_lbl, window=window, stride=stride,
        norm_mean=train_mean, norm_std=train_std
    )
    
    val_ds = BaseTimeSeriesDataset(
        val_sig, val_lbl, window=window, stride=stride,
        norm_mean=train_mean, norm_std=train_std,
        graph=train_ds.graph # Share the Training graph for consistency
    )
    
    test_ds = BaseTimeSeriesDataset(
        test_sig, test_lbl, window=window, stride=stride,
        norm_mean=train_mean, norm_std=train_std,
        graph=train_ds.graph
    )

    print(f"[Dataset] Train samples: {len(train_ds):,}")
    print(f"[Dataset] Val samples:   {len(val_ds):,}")
    print(f"[Dataset] Test samples:  {len(test_ds):,}\n")

    return train_ds, val_ds, test_ds, feature_cols
