import os
import numpy as np
import pandas as pd
from data.base_dataset import BaseTimeSeriesDataset, build_graph_from_correlation

def _load_telemanom(data_dir, dataset="SMAP"):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Provided data_dir does not exist: {data_dir}")

    # Search for labeled_anomalies.csv
    anomaly_csv = os.path.join(data_dir, "labeled_anomalies.csv")
    if not os.path.exists(anomaly_csv):
        parent_csv = os.path.join(os.path.dirname(data_dir), "labeled_anomalies.csv")
        child_csv = os.path.join(data_dir, "data", "labeled_anomalies.csv")
        if os.path.exists(parent_csv):
            anomaly_csv = parent_csv
        elif os.path.exists(child_csv):
            anomaly_csv = child_csv
        else:
            root_dir = data_dir.split("/data")[0]
            root_csv = os.path.join(root_dir, "labeled_anomalies.csv")
            if os.path.exists(root_csv):
                anomaly_csv = root_csv
            else:
                raise FileNotFoundError(f"Could not find labeled_anomalies.csv in {data_dir} or nearby directories.")

    # Search for train/test directories
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    if not os.path.exists(train_dir):
        child_train = os.path.join(data_dir, "data", "train")
        child_test = os.path.join(data_dir, "data", "test")
        if os.path.exists(child_train):
            train_dir, test_dir = child_train, child_test

    print(f"📌 [SMAP Loader] Using CSV: {anomaly_csv}")
    print(f"📌 [SMAP Loader] Using Train Dir: {train_dir}")

    label_df = pd.read_csv(anomaly_csv)
    label_df = label_df[label_df["spacecraft"] == dataset]
    channels = label_df["chan_id"].tolist()

    train_list, test_list, test_lbl_list, channel_ids = [], [], [], []

    for chan in channels:
        tr_p = os.path.join(train_dir, f"{chan}.npy")
        te_p = os.path.join(test_dir, f"{chan}.npy")
        if not (os.path.exists(tr_p) and os.path.exists(te_p)):
            continue

        train_list.append(np.load(tr_p))
        test_list.append(np.load(te_p))
        
        row = label_df[label_df["chan_id"] == chan].iloc[0]
        indices = eval(row["anomaly_sequences"])
        lbl = np.zeros(len(test_list[-1]), dtype=np.int64)
        for start, end in indices:
            lbl[start : end + 1] = 1
        test_lbl_list.append(lbl)
        channel_ids.append(chan)

    print(f"🏁 Kept {len(channel_ids)}/{len(channels)} channels for {dataset}")
    
    if len(channel_ids) == 0:
        raise ValueError(f"No valid .npy channel files found in {train_dir}")

    def align(arrays, is_label=False):
        # Determine target sequence length along the time axis
        target = int(np.median([len(a) for a in arrays]))
        res = []
        for a in arrays:
            # Extract primary telemetry channel (column 0) if 2D array
            if not is_label and a.ndim > 1:
                a = a[:, 0]
                
            if len(a) >= target:
                res.append(a[:target])
            else:
                res.append(np.pad(a, (0, target - len(a)), "edge"))
        return np.stack(res, axis=1)

    train_sig = align(train_list, is_label=False)
    test_sig = align(test_list, is_label=False)
    
    # 🛡️ THE FIX: Keep the original node-level labels intact!
    # Removing the flattening/broadcasting logic so each sensor keeps its own ground truth.
    test_lbl_matrix = align(test_lbl_list, is_label=True)

    return train_sig, test_sig, test_lbl_matrix, channel_ids

def load_smap(data_dir: str, window: int = 30, stride: int = 1, val_ratio: float = 0.15, graph_threshold: float | None = None):
    train_sig, test_sig, test_lbl, _ = _load_telemanom(data_dir, "SMAP")
    
    split_idx = int(len(train_sig) * (1 - val_ratio))
    tr_signals = train_sig[:split_idx]
    va_signals = train_sig[split_idx:]

    tr_labels = np.zeros_like(tr_signals, dtype=np.int64)
    va_labels = np.zeros_like(va_signals, dtype=np.int64)

    train_set = BaseTimeSeriesDataset(tr_signals, tr_labels, window, stride, graph_threshold=graph_threshold)
    val_set = BaseTimeSeriesDataset(va_signals, va_labels, window, stride, graph=train_set.graph)
    test_set = BaseTimeSeriesDataset(test_sig, test_lbl, window, stride, graph=train_set.graph,
                                     norm_mean=train_set.mean, norm_std=train_set.std)

    return train_set, val_set, test_set, train_set.graph
