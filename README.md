# RFM-TransE: Relational Fuzzy Membership Knowledge Graph Embedding

This repository contains the official implementation of **RFM-TransE** (Relational Fuzzy Membership TransE), an uncertainty-aware knowledge graph embedding model that maps translation distances into bounded, continuous fuzzy membership values $[0, 1]$.

---

## 🚀 Key Features

* **Relational Fuzzy Membership**: Extends translation-based embeddings by learning relation-specific sharpness ($\beta_r > 0$) and tolerance boundary radius ($\gamma_r > 0$) parameters using smooth $\text{softplus}$ parameterization.
* **Continuous Truth Functions**: Supports multiple membership formulations:
  * **Sigmoid**: $\sigma\big(\beta_r \cdot (\gamma_r - d) + b_0\big)$
  * **Exponential**: $\exp\big(-\beta_r \cdot \max(0, d - \gamma_r)\big)$
  * **Gaussian**: $\exp\big(-\beta_r \cdot (\max(0, d - \gamma_r))^2\big)$
* **Numerically Stable Training**: Direct logit-space optimization via `BCEWithLogitsLoss` combined with margin-ranking distance loss.
* **Comprehensive Multi-Seed Benchmark Suite**: Automated evaluation across Link Prediction (MRR, Hits@3, Hits@5), Triple Classification (F1), Fuzzy Regression (MSE, MAE), and Calibration Quality (ECE).

---

## 📁 Repository Structure

```text
├── RFM_TransE.py                # Core RFM-TransE model implementation
├── transe.py                    # Standard TransE baseline model
├── utils.py                     # Dataset loading, caching, labeling & evaluation metrics
├── benchmark.py                 # Multi-seed experimental benchmark runner
├── benchmark_results.md         # Detailed empirical benchmark results
└── benchmark_results_rerun.md   # Extended multi-seed distribution and ablation audit
```

---

## 🛠️ Installation & Requirements

Ensure you have Python 3.8+ and PyTorch installed:

```bash
pip install torch numpy
```

---

## 📊 Running Benchmarks

Datasets (**CoDEx-S**, **CoDEx-M**, **FB15k**, **FB15k-237**, **WN18RR**) are automatically downloaded and cached in `data/` on the first run.

### Run on a single dataset:
```bash
python3 benchmark.py --dataset CoDEx-S --epochs 30 --seeds 42 100 2024
```

### Run on all benchmark datasets:
```bash
python3 benchmark.py --dataset all --uncertainty all --epochs 30 --output_file results.md
```

### Command-line Options:
* `--dataset`: Dataset name (`CoDEx-S`, `CoDEx-M`, `FB15k`, `FB15k-237`, `WN18RR`, or `all`).
* `--uncertainty`: Synthetic uncertainty regime (`low`, `medium`, `high`, or `all`).
* `--seeds`: List of random seeds for multi-seed aggregation (default: `42 100 2024`).
* `--embedding_dim`: Dimensionality of vector embeddings (default: `50`).
* `--margin`: Margin for translation distance ranking loss (default: `2.0`).
* `--lambda_fuzzy`: Weight scaling factor for fuzzy membership loss (default: `0.1`).
