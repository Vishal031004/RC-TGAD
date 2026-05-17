# RC-TGAD
## Retrieval-Augmented Curriculum Learning for Temporal Graph Anomaly Detection using LSTM-GNN Models

---

# Overview

RC-TGAD is a hybrid deep learning framework for multivariate time-series anomaly detection designed for highly complex Industrial Control Systems (ICS) and Multivariate Time Series (MTS) data.

The framework combines:

- Long Short-Term Memory (LSTM) networks for temporal feature extraction
- Graph Neural Networks (GNNs) for spatial sensor-correlation reasoning
- Retrieval-Augmented (RAG)-based hardness scoring using FAISS
- Curriculum learning for progressive training from easy to hard samples

RC-TGAD is specifically engineered to:

- detect subtle and structurally complex anomalies,
- model inter-sensor relationships,
- handle noisy industrial telemetry,
- and operate efficiently under constrained compute environments such as Kaggle.

The framework introduces a novel hardness-aware curriculum training pipeline that dynamically prioritizes easier samples during early training and gradually exposes the model to harder anomalous patterns.

---

# 🌟 Key Features & Innovations

## Dynamic Curriculum Learning
Gradually unlocks harder training samples based on hardness scores, preventing catastrophic forgetting and improving stability during exposure to edge-case operational physics.

---

## Retrieval-Augmented (RAG) Hardness Scoring
Uses historical embedding retrieval with k-nearest neighbors to distinguish between true cyberattacks and rare-but-normal operational mode transitions.

---

## Temporal Modeling with LSTM
Each sensor node is independently modeled using a two-layer LSTM to capture temporal evolution and sequential dependencies.

---

## Graph Attention-Based Reasoning
Uses Graph Attention Networks (GATs) to model cross-sensor interactions and anomaly propagation patterns.

---

## Composite Hardness Scoring
Defines sample hardness using:

- Temporal hardness
- Structural hardness
- Retrieval-augmented hardness

---

## Causal Smoothing (Lookahead Prevention)
Applies shifted rolling mean smoothing to prevent future telemetry leakage during preprocessing.

---

## Robust Scaling & Clipping
Uses RobustScaler with Interquartile Range (IQR) normalization and hard clipping to prevent gradient instability caused by extreme cyberattack spikes.

---

## Memory-Efficient Downsampling
Supports aggressive downsampling for massive ICS datasets to reduce memory footprint and enable training within limited RAM environments.

---

## Foolproof Label Extraction
Dynamically extracts anomaly labels from corrupted ICS CSV files using majority-class detection.

---

# Motivation

Traditional anomaly detection systems often:

- rely solely on reconstruction error,
- ignore graph structure,
- fail to model anomaly ambiguity,
- and treat all training samples equally.

As a result, they struggle with:

- subtle anomalies,
- sparse cyberattacks,
- and structurally complex operational failures.

RC-TGAD addresses these limitations by integrating:

1. Temporal modeling (LSTM)
2. Structural reasoning (GNN)
3. Retrieval-based ambiguity estimation (RAG)
4. Curriculum learning

into a unified anomaly detection framework.

---

# Architecture

The overall RC-TGAD pipeline:

```text
Input Windows
      ↓
Per-node LSTM Encoder
      ↓
Node Embeddings (H)
      ↓
Graph Attention Network (GAT)
      ↓
Joint Embeddings (Z)
      ↓
RAG-Based Hardness Scoring
      ↓
Curriculum Scheduler
      ↓
Progressive Training
      ↓
Anomaly Detection
```

---

# Methodology

---

## 1. Sliding Window Representation

For each node \(v\) and timestep \(t\), a sliding window of length \(W\) is constructed:

```text
x(v,t) = [X(t-W+1,v), ..., X(t,v)]
```

Features are normalized per sensor.

---

## 2. LSTM Encoder

Each node window is independently passed through a two-layer LSTM.

Outputs:

- Hidden embedding:

```text
h ∈ R^64
```

- Reconstruction:

```text
x_hat ∈ R^1
```

The reconstruction objective captures temporal consistency.

---

## 3. Graph Attention Network (GAT)

The node embeddings are refined using a two-layer GAT.

The GAT:

- aggregates neighborhood information,
- models sensor dependencies,
- and captures anomaly propagation patterns.

Final embedding:

```text
z ∈ R^64
```

These embeddings are stored inside the FAISS vector store.

---

## 4. Retrieval-Augmented Hardness Scoring

The core novelty of RC-TGAD is the composite hardness score:

```text
H(v,t) ∈ [0,1]
```

Higher values correspond to harder samples.

The score consists of three components:

---

### Temporal Hardness (H_temp)

Computed from reconstruction error:

```text
e(v,t) = ||x - x_hat||
```

Subtle anomalies with low reconstruction error are considered harder.

---

### Structural Hardness (H_struct)

Measures graph-based difficulty using:

- node centrality,
- graph connectivity,
- and distance from anomaly source.

Peripheral nodes and structurally isolated nodes are treated as harder samples.

---

### Retrieval-Augmented Hardness (H_RAG)

A FAISS vector store stores historical embeddings and labels.

For each incoming embedding:

1. Retrieve k nearest neighbors
2. Analyze neighborhood label distribution
3. Compute entropy-based uncertainty

Mixed neighborhoods imply ambiguous samples and therefore higher hardness.

This enables:

- memory-aware anomaly reasoning,
- ambiguity estimation,
- and subtle anomaly detection.

---

## 5. Curriculum Learning

RC-TGAD uses hardness-aware curriculum learning.

Training progresses from:

```text
Easy Samples → Hard Samples
```

using a pacing function:

```text
λ(k) = min(1.0, k / K_warmup)
```

where:

- \(k\) = current epoch
- \(K_warmup\) = curriculum warmup period

A sample is selected if:

```text
H(v,t) ≤ λ(k)
```

This stabilizes training and improves generalization.

---

# 📂 Project Structure

```plaintext
RC-TGAD/
│
├── configs/
│   └── default.yaml          # Master configuration for training & architecture
│
├── data/
│   ├── base_dataset.py       # Core PyTorch Dataset & graph builder
│   ├── swat.py               # Optimized SWaT data loader
│   └── wadi.py               # Hardened WADI data loader
│
├── backbone/
│   ├── backbone.py
│   ├── gnn_reasoner.py
│   └── lstm_encoder.py
│
├── curriculum/
│   ├── scheduler.py
│   └── trainer.py
│
├── rag/
│   ├── hardness.py
│   ├── rag_scorer.py
│   ├── vector_store.py
│   └── test_real_embeddings.py
│
├── experiments/
│   ├── run_baseline.py
│   └── ablations.py
│
├── utils/
│   └── metrics.py
│
├── results/
│
└── checkpoints/
```

---

# 💾 Datasets

The framework supports:

- SWaT (Secure Water Treatment)
- WADI (Water Distribution)
- MSL (planned)

The framework expects raw `.csv` files obtained from:

- iTrust (Singapore University of Technology and Design)

---

## Supported Datasets

### SWaT
- 51 sensors
- ~11 days of data

### WADI
- 127 sensors
- ~16 days of data

Custom data loaders automatically handle:

- NaNs
- header misalignment
- zero-variance sensors
- label extraction
- graph construction

---

# Installation

## Clone Repository

```bash
git clone <repo-url>
cd RC-TGAD
```

---

## Create Environment

```bash
conda create -n rctgad python=3.10
conda activate rctgad
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Required Libraries

Main dependencies:

```text
PyTorch
PyTorch Geometric
FAISS
NumPy
scikit-learn
networkx
PyYAML
```

---

# 🚀 Running the Project

---

# 1. Running the Full RC-TGAD Pipeline

```bash
python -u experiments/ablations.py \
    --dataset wadi \
    --seeds 42 \
    --variants "Full RC-TGAD" \
    --override \
    data.data_dir=/path/to/your/wadi/data/ \
    data.stride=2 \
    training.epochs=35 \
    training.batch_size=64 \
    training.weight_decay=1e-3 \
    model.dropout=0.3 \
    model.window_size=60 \
    model.lstm_hidden=128 \
    model.gnn_heads=4 \
    model.gnn_out_dim=128 \
    data.graph_threshold=0.1 \
    rag.vector_dim=128 \
    rag.k_neighbors=5 \
    rag.alpha_1=0.34 \
    rag.alpha_2=0.33 \
    rag.alpha_3=0.33
```

---

# 2. Running Baseline Ablations

```bash
python -u experiments/ablations.py \
    --dataset wadi \
    --seeds 42 \
    --variants "No Curriculum" "No RAG" \
    --override \
    data.data_dir=/path/to/your/wadi/data/ \
    data.stride=2 \
    training.epochs=35 \
    training.batch_size=64 \
    training.weight_decay=1e-3 \
    model.dropout=0.3 \
    model.window_size=60 \
    model.lstm_hidden=128 \
    model.gnn_heads=4 \
    model.gnn_out_dim=128 \
    data.graph_threshold=0.1
```

---

# 📊 Evaluation Metrics

The framework evaluates anomaly detection performance using:

- F1 Score
- Point-Adjusted F1 (F1-PA)
- AUC-PR
- AUC-ROC

---

## Point-Adjusted F1 (F1-PA)

Industrial cyberattacks are continuous events spanning multiple timesteps.

Standard F1 heavily penalizes delayed detections.

Point-adjusted F1 considers an anomaly segment correctly detected if any point within the segment is identified.

This better reflects real-world operational requirements.

---

# Experimental Setup

## Default Hyperparameters

| Parameter | Value |
|---|---|
| Window Size | 30 |
| LSTM Hidden Size | 64 |
| GNN Output Dimension | 64 |
| Attention Heads | 4 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Warmup Epochs | 30 |
| k Neighbors | 10 |

---

# Results

RC-TGAD achieves close to state-of-the-art performance on multivariate time-series anomaly detection benchmarks while improving robustness on:

- subtle anomalies,
- ambiguous operational transitions,
- and structurally complex attack propagation patterns.

Results including:

- AUC-ROC
- AUC-PR
- F1
- F1-PA

are automatically logged to:

```text
results/ablations/{dataset}/
```

and formatted into paper-ready LaTeX tables.

---

# Research Contributions

RC-TGAD introduces:

1. Retrieval-augmented hardness scoring for anomaly detection
2. Entropy-based ambiguity estimation using embedding neighborhoods
3. Hardness-aware curriculum learning
4. Joint temporal and structural reasoning using LSTM + GNN
5. Memory-aware anomaly detection using FAISS retrieval

---

# Future Work

Potential future directions include:

- Dynamic hardness re-scoring during training
- Adaptive weighting of hardness components
- Dynamic graph construction
- Evaluation on additional datasets
- Memory-efficient vector stores
- Online streaming anomaly detection

---

# Citation

```bibtex
@article{rctgad2026,
  title={RC-TGAD: Retrieval-Augmented Curriculum Learning for Temporal Graph Anomaly Detection using LSTM-GNN Models},
  author={Vishal P and Shaikh Saniya Ali and Sagar R Bhat and Shylaja S S},
  journal={Work in Progress},
  year={2026}
}
```

---

# Authors

- Vishal P
- Shaikh Saniya Ali
- Sagar R Bhat
- Dr. Shylaja S S

Department of Computer Science and Engineering  
PES University, RR Campus, Bangalore

---

# License

This project is intended for academic and research purposes.

---

# Acknowledgements

We thank the Department of Computer Science and Engineering at PES University for support and guidance throughout this work.
