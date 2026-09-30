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
└── benchmark.py                 # Multi-seed experimental benchmark runner
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

Run commands from this repository directory. By default, **100% of each official
training, validation, and test split** is used. The previous default limits of
10,000 training triples and 500 validation/test triples have been removed.
The original split boundaries are preserved: validation is used to select
classification thresholds and test is used for final metrics. Neighbourhoods for
structural targets are built only from the selected training triples. Known
triples across all splits are still used for filtered evaluation and evaluation
negative rejection, as in the existing benchmark protocol.

Full runs take substantially longer, especially filtered ranking, which scores
every entity as a possible head and tail for every selected test triple. Lower
`--eval_batch_size` to reduce ranking memory usage without reducing coverage.
Default reports are saved as `benchmark_results_full.md`, keeping the old default
report filename separate. Existing thesis numbers are not results of this new
protocol; rerun the experiments before updating them.

### Run on a single dataset:
```bash
python3 benchmark.py --dataset CoDEx-S --epochs 30 --seeds 42 100 2024
```

### Run on all benchmark datasets:
```bash
python3 benchmark.py --dataset all --uncertainty all --epochs 30 --output_file results.md
```

### Command-line Options:
* `--train_fraction`, `--val_fraction`, `--test_fraction`: Proportion of each official split to use, in `(0, 1]` (default: `1.0`). These are sampling fractions, not new train/validation/test split ratios.
* `--train_sample_size`: Optional training cap (default: `0`, unlimited).
* `--eval_sample_size`: Optional shared validation/test cap (default: `0`, unlimited).
* `--val_sample_size`, `--test_sample_size`: Override the shared cap for the respective split; `0` explicitly requests no cap.
* `--eval_batch_size`: Number of ranking queries processed together (default: `16`).
* Fractions apply first, then positive caps. Nonpositive caps mean unlimited. Selections are reproducible per seed and do not mutate the source splits. All models share the same selected data within each run.
* `--dataset`: Dataset name (`CoDEx-S`, `CoDEx-M`, `FB15k`, `FB15k-237`, `WN18RR`, or `all`).
* `--uncertainty`: Synthetic uncertainty regime (`low`, `medium`, `high`, or `all`).
* `--models`: Select implementations to run from `standard`, `sigmoid`, `exponential`, and `gaussian`. The final thesis comparison uses `--models standard sigmoid`.
* `--seeds`: List of random seeds for multi-seed aggregation (default: `42 100 2024`).
* `--embedding_dim`: Dimensionality of vector embeddings (default: `50`).
* `--margin`: Margin for translation distance ranking loss (default: `2.0`).
* `--lambda_fuzzy`: Weight scaling factor for fuzzy membership loss (default: `0.1`).

### A shorter run with broader training coverage

Use half of each dataset's training split and all validation data, with at most
5,000 test triples. Unlike a fixed 10,000 limit, training size grows with the dataset:

```bash
python3 benchmark.py --dataset all --train_fraction 0.5 --test_sample_size 5000 --output_file benchmark_results_half_train.md
```

### Small verification run

```bash
python3 benchmark.py --dataset CoDEx-S --epochs 1 --seeds 42 --train_sample_size 100 --eval_sample_size 20 --output_file smoke_results.md
python3 -m unittest discover -s tests -v
```

Using the old limits is still possible with `--train_sample_size 10000
--eval_sample_size 500`. This reproduces the old sample sizes, not necessarily
identical historical results: split selection now uses independent random streams
and training no longer modifies the cached source order.
