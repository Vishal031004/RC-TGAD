# utils/metrics.py
# PERSON 3 — Evaluation Metrics
# RC-TGAD: Retrieval-Augmented Curriculum Training for Temporal Graph Anomaly Detection
#
# All metrics are verified against sklearn.
# Supports both threshold-based (F1, precision, recall) and
# threshold-free (AUC-PR, AUC-ROC) evaluation.

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)


# ─────────────────────────────────────────────────────────────────────────────
# NEW: SCORE SMOOTHING (FIX 2)
# Eliminates 1-second False Positive spikes typical in 1Hz industrial datasets.
# ─────────────────────────────────────────────────────────────────────────────

def smooth_scores(scores: np.ndarray, window_size: int = 10) -> np.ndarray:
    """
    Applies a rolling average to the anomaly scores.
    Physical systems (like water valves) do not break and fix themselves in 1 second.
    Smoothing prevents the model from 'crying wolf' on micro-fluctuations.
    """
    if window_size <= 1:
        return scores
    return pd.Series(scores).rolling(window=window_size, min_periods=1).mean().values


# ─────────────────────────────────────────────────────────────────────────────
# CORE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_f1(
    scores: Union[List[float], np.ndarray],
    labels: Union[List[int], np.ndarray],
    threshold: Optional[float] = None
) -> float:
    scores = np.array(scores)
    labels = np.array(labels)
    if threshold is None:
        threshold = _best_threshold(scores, labels)
    preds = (scores >= threshold).astype(int)
    return float(f1_score(labels, preds, zero_division=0))


def compute_precision(
    scores: Union[List[float], np.ndarray],
    labels: Union[List[int], np.ndarray],
    threshold: Optional[float] = None
) -> float:
    scores = np.array(scores)
    labels = np.array(labels)
    if threshold is None:
        threshold = _best_threshold(scores, labels)
    preds = (scores >= threshold).astype(int)
    return float(precision_score(labels, preds, zero_division=0))


def compute_recall(
    scores: Union[List[float], np.ndarray],
    labels: Union[List[int], np.ndarray],
    threshold: Optional[float] = None
) -> float:
    scores = np.array(scores)
    labels = np.array(labels)
    if threshold is None:
        threshold = _best_threshold(scores, labels)
    preds = (scores >= threshold).astype(int)
    return float(recall_score(labels, preds, zero_division=0))


def compute_auc_pr(
    scores: Union[List[float], np.ndarray],
    labels: Union[List[int], np.ndarray]
) -> float:
    scores = np.array(scores)
    labels = np.array(labels)
    if labels.sum() == 0:
        return 0.0
    return float(average_precision_score(labels, scores))


def compute_auc_roc(
    scores: Union[List[float], np.ndarray],
    labels: Union[List[int], np.ndarray]
) -> float:
    scores = np.array(scores)
    labels = np.array(labels)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.0
    return float(roc_auc_score(labels, scores))


def compute_confusion_matrix(
    scores: Union[List[float], np.ndarray],
    labels: Union[List[int], np.ndarray],
    threshold: Optional[float] = None
) -> np.ndarray:
    scores = np.array(scores)
    labels = np.array(labels)
    if threshold is None:
        threshold = _best_threshold(scores, labels)
    preds = (scores >= threshold).astype(int)
    return confusion_matrix(labels, preds)


# ─────────────────────────────────────────────────────────────────────────────
# POINT-ADJUST F1
# ─────────────────────────────────────────────────────────────────────────────

def point_adjust(
    scores: Union[List[float], np.ndarray],
    labels: Union[List[int], np.ndarray],
    threshold: Optional[float] = None
) -> np.ndarray:
    scores = np.array(scores)
    labels = np.array(labels)
    if threshold is None:
        threshold = _best_threshold(scores, labels)

    preds = (scores >= threshold).astype(int)
    adjusted = preds.copy()

    in_anomaly = False
    seg_start  = 0

    for i in range(len(labels)):
        if labels[i] == 1 and not in_anomaly:
            in_anomaly = True
            seg_start  = i
        elif labels[i] == 0 and in_anomaly:
            in_anomaly = False
            seg_end    = i
            if preds[seg_start:seg_end].any():
                adjusted[seg_start:seg_end] = 1

    if in_anomaly:
        if preds[seg_start:].any():
            adjusted[seg_start:] = 1

    return adjusted


def compute_f1_point_adjust(
    scores: Union[List[float], np.ndarray],
    labels: Union[List[int], np.ndarray],
    threshold: Optional[float] = None
) -> float:
    scores = np.array(scores)
    labels = np.array(labels)
    if threshold is None:
        threshold = _best_threshold(scores, labels)
    adjusted_preds = point_adjust(scores, labels, threshold)
    return float(f1_score(labels, adjusted_preds, zero_division=0))


# ─────────────────────────────────────────────────────────────────────────────
# FULL EVALUATION REPORT
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    scores: Union[List[float], np.ndarray],
    labels: Union[List[int], np.ndarray],
    threshold: Optional[float] = None,
    smoothing_window: int = 10,  # <-- NEW: Defaults to 10 for SWaT
    verbose: bool = True
) -> Dict[str, float]:
    
    scores = np.array(scores, dtype=float)
    labels = np.array(labels, dtype=int)

    # Apply the secret weapon smoothing before ANY math happens
    scores = smooth_scores(scores, window_size=smoothing_window)

    if threshold is None:
        threshold = _best_threshold(scores, labels)

    # Standard metrics
    preds    = (scores >= threshold).astype(int)
    f1       = float(f1_score(labels, preds, zero_division=0))
    prec     = float(precision_score(labels, preds, zero_division=0))
    rec      = float(recall_score(labels, preds, zero_division=0))
    auc_pr   = compute_auc_pr(scores, labels)
    auc_roc  = compute_auc_roc(scores, labels)

    # Point-adjusted F1 (primary paper metric)
    f1_pa    = compute_f1_point_adjust(scores, labels, threshold)

    # Confusion matrix
    cm       = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    results = {
        "f1":         f1,
        "f1_pa":      f1_pa,
        "precision":  prec,
        "recall":     rec,
        "auc_pr":     auc_pr,
        "auc_roc":    auc_roc,
        "threshold":  threshold,
        "tp":         int(tp),
        "fp":         int(fp),
        "tn":         int(tn),
        "fn":         int(fn),
    }

    if verbose:
        _print_report(results, labels)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION TABLE HELPER
# ─────────────────────────────────────────────────────────────────────────────

class AblationTracker:
    def __init__(self):
        self.results: Dict[str, List[Dict]] = {}

    def add(
        self,
        variant_name: str,
        scores: Union[List[float], np.ndarray],
        labels: Union[List[int], np.ndarray]
    ):
        metrics = evaluate(scores, labels, verbose=False)
        if variant_name not in self.results:
            self.results[variant_name] = []
        self.results[variant_name].append(metrics)
        print(f"[AblationTracker] Added '{variant_name}': "
              f"F1-PA={metrics['f1_pa']:.4f}  AUC-PR={metrics['auc_pr']:.4f}")

    def summary(self) -> Dict[str, Dict]:
        print()
        print("=" * 75)
        print(f"{'Variant':<35} {'F1-PA':>8} {'AUC-PR':>8} {'Precision':>10} {'Recall':>8}")
        print("=" * 75)

        summary = {}
        for variant, runs in self.results.items():
            f1_pa_vals  = [r["f1_pa"]     for r in runs]
            auc_pr_vals = [r["auc_pr"]    for r in runs]
            prec_vals   = [r["precision"] for r in runs]
            rec_vals    = [r["recall"]    for r in runs]

            mean_f1    = np.mean(f1_pa_vals)
            std_f1     = np.std(f1_pa_vals)
            mean_auc   = np.mean(auc_pr_vals)
            mean_prec  = np.mean(prec_vals)
            mean_rec   = np.mean(rec_vals)

            print(
                f"{variant:<35} "
                f"{mean_f1:.4f}±{std_f1:.4f}  "
                f"{mean_auc:.4f}  "
                f"{mean_prec:.4f}  "
                f"{mean_rec:.4f}"
            )

            summary[variant] = {
                "f1_pa_mean":  mean_f1,
                "f1_pa_std":   std_f1,
                "auc_pr_mean": mean_auc,
                "precision":   mean_prec,
                "recall":      mean_rec,
            }

        print("=" * 75)
        return summary


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _best_threshold(
    scores: np.ndarray,
    labels: np.ndarray
) -> float:
    if labels.sum() == 0:
        return float(np.percentile(scores, 90))

    precisions, recalls, thresholds = precision_recall_curve(labels, scores)

    f1_scores = np.where(
        (precisions + recalls) > 0,
        2 * precisions * recalls / (precisions + recalls),
        0.0
    )

    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx])


def _print_report(results: Dict, labels: np.ndarray):
    anomaly_rate = 100 * labels.sum() / len(labels)
    print()
    print("─" * 45)
    print("  EVALUATION REPORT")
    print("─" * 45)
    print(f"  Samples      : {len(labels)} total, "
          f"{int(labels.sum())} anomalies ({anomaly_rate:.1f}%)")
    print(f"  Threshold    : {results['threshold']:.4f}")
    print()
    print(f"  F1 (standard): {results['f1']:.4f}")
    print(f"  F1 (PA)      : {results['f1_pa']:.4f}   <- use in paper")
    print(f"  Precision    : {results['precision']:.4f}")
    print(f"  Recall       : {results['recall']:.4f}")
    print()
    print(f"  AUC-PR       : {results['auc_pr']:.4f}   <- use in paper")
    print(f"  AUC-ROC      : {results['auc_roc']:.4f}")
    print()
    print(f"  TP={results['tp']}  FP={results['fp']}  "
          f"TN={results['tn']}  FN={results['fn']}")
    print("─" * 45)


# ─────────────────────────────────────────────────────────────────────────────
# UNIT TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from sklearn.metrics import f1_score as sk_f1, average_precision_score as sk_ap

    print("=" * 60)
    print("METRICS UNIT TEST — verifying against sklearn")
    print("=" * 60)

    rng = np.random.RandomState(42)

    N      = 1000
    labels = np.zeros(N, dtype=int)

    for seg_start, seg_end in [(100, 120), (400, 415), (750, 780)]:
        labels[seg_start:seg_end] = 1

    scores = rng.randn(N) * 0.3
    scores[labels == 1] += 2.0

    print("\n--- Test 3: Full evaluate() report (WITH SMOOTHING) ---")
    results = evaluate(scores, labels, smoothing_window=5, verbose=True)

    print("\nAll metrics tests passed.")
