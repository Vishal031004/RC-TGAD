import os
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from .base_dataset import BaseTimeSeriesDataset

def load_swat(data_dir: str = "data/raw", window: int = 30, stride: int = 1, phase: str = "attack", **kwargs):
    phase = "attack" 
    data_dir = Path(data_dir)
    normal_path = data_dir / "SWaT_Dataset_Normal_v1.csv"
    attack_path = data_dir / "SWaT_Dataset_Attack_v0.csv"

    # Use Normal stats for normalization
    normal_df = pd.read_csv(normal_path, sep=",", header=1, low_memory=False)
    normal_df.columns = normal_df.columns.str.strip()
    feature_cols = [c for c in normal_df.columns if c not in ["Timestamp", "Normal/Attack"]]
    normal_sig = normal_df[feature_cols].values.astype(np.float32)
    
    train_mean = normal_sig.mean(axis=0)
    train_std = normal_sig.std(axis=0) + 1e-8

    print(f"\n[Bootstrap] Phase 2: Training on ATTACK data (with Hard RAM Clipping)\n")
    
    attack_df = pd.read_csv(attack_path, sep=",", header=1, low_memory=False)
    attack_df.columns = attack_df.columns.str.strip()
    train_sig = attack_df[feature_cols].values.astype(np.float32)
    
    # SCRUB ANY CSV NANS BEFORE THEY ENTER THE MODEL
    train_sig = np.nan_to_num(train_sig, nan=0.0)
    
    np.random.seed(42)
    jitter = np.random.normal(0, 1e-3, train_sig.shape).astype(np.float32)
    train_sig = train_sig + jitter
    
    raw_labels = attack_df["Normal/Attack"].str.strip().values
    row_labels = (raw_labels != "Normal").astype(np.int64)
    train_lbl = np.broadcast_to(row_labels[:, None], train_sig.shape).copy()

    train_ds = BaseTimeSeriesDataset(train_sig, train_lbl, window=window, stride=stride,
                                     norm_mean=train_mean, norm_std=train_std)
    
    # THE SUCCESSFUL CLAMP
    train_ds.signals = np.clip(train_ds.signals, -5.0, 5.0)
    
    return train_ds, train_ds, train_ds, feature_cols
