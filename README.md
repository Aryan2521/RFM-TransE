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

---

## Results after changing the benchmark coverage

The results below were produced after changing the benchmark runner from the
earlier fixed limits of at most **10,000 training triples and 500
validation/test triples** to **100% of every official training, validation, and
test split**. Consequently, these values differ from the results printed in the
thesis and must not be interpreted as a direct architecture-only replication of
the earlier tables. The model implementations and main hyperparameters remain
the same, but the models now learn from substantially more triples and are
evaluated on the complete official test sets.

All new values are reported as **mean ± sample standard deviation** over seeds
**42, 100, and 2024**, using 30 epochs, embedding dimension 50, learning rate
0.005, margin 2.0, and fuzzy-loss weight 0.1. MRR, Hits@3, Hits@5, and F1 are
higher-is-better metrics. MSE, MAE, and ECE are lower-is-better metrics. Training
time is the recorded CPU training time and excludes evaluation.

### All model variants under the medium uncertainty regime

This table provides every recorded benchmark metric for Standard TransE and all
three membership mappings under medium uncertainty.

| Dataset | Model | MRR | Hits@3 | Hits@5 | F1 | MSE | MAE | ECE | Train time |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CoDEx-S | Standard TransE | 0.2113 ± 0.0043 | 0.2302 ± 0.0054 | 0.2966 ± 0.0023 | 0.8804 ± 0.0048 | 0.2063 ± 0.0039 | 0.3801 ± 0.0008 | 0.2491 ± 0.0070 | 16.0s ± 2.0s |
| CoDEx-S | RFM-TransE (Sigmoid) | 0.2223 ± 0.0031 | 0.2468 ± 0.0084 | 0.3170 ± 0.0048 | 0.8982 ± 0.0012 | 0.0824 ± 0.0037 | 0.2544 ± 0.0109 | 0.1605 ± 0.0102 | 17.5s ± 0.5s |
| CoDEx-S | RFM-TransE (Exponential) | 0.2202 ± 0.0031 | 0.2484 ± 0.0051 | 0.3190 ± 0.0092 | 0.8894 ± 0.0030 | 0.0867 ± 0.0020 | 0.2650 ± 0.0053 | 0.1663 ± 0.0060 | 18.9s ± 2.0s |
| CoDEx-S | RFM-TransE (Gaussian) | 0.2170 ± 0.0068 | 0.2494 ± 0.0042 | 0.3234 ± 0.0061 | 0.9009 ± 0.0062 | 0.0940 ± 0.0181 | 0.2221 ± 0.0128 | 0.1314 ± 0.0243 | 18.1s ± 1.1s |
| CoDEx-M | Standard TransE | 0.1250 ± 0.0015 | 0.1362 ± 0.0017 | 0.1819 ± 0.0004 | 0.8670 ± 0.0016 | 0.1867 ± 0.0040 | 0.3763 ± 0.0006 | 0.2039 ± 0.0113 | 206.7s ± 0.7s |
| CoDEx-M | RFM-TransE (Sigmoid) | 0.1325 ± 0.0042 | 0.1443 ± 0.0051 | 0.1892 ± 0.0045 | 0.8904 ± 0.0039 | 0.0671 ± 0.0014 | 0.1736 ± 0.0022 | 0.0814 ± 0.0016 | 232.7s ± 1.5s |
| CoDEx-M | RFM-TransE (Exponential) | 0.1314 ± 0.0024 | 0.1422 ± 0.0013 | 0.1847 ± 0.0010 | 0.8861 ± 0.0007 | 0.0895 ± 0.0018 | 0.2638 ± 0.0041 | 0.1501 ± 0.0047 | 218.9s ± 3.0s |
| CoDEx-M | RFM-TransE (Gaussian) | 0.1355 ± 0.0033 | 0.1474 ± 0.0043 | 0.1924 ± 0.0027 | 0.8926 ± 0.0016 | 0.0748 ± 0.0024 | 0.1969 ± 0.0021 | 0.0910 ± 0.0025 | 232.9s ± 1.4s |
| FB15k | Standard TransE | 0.1453 ± 0.0031 | 0.1545 ± 0.0061 | 0.1958 ± 0.0047 | 0.9162 ± 0.0010 | 0.2301 ± 0.0010 | 0.3855 ± 0.0002 | 0.2823 ± 0.0014 | 347.8s ± 71.7s |
| FB15k | RFM-TransE (Sigmoid) | 0.1707 ± 0.0041 | 0.1843 ± 0.0058 | 0.2326 ± 0.0059 | 0.9394 ± 0.0002 | 0.0374 ± 0.0004 | 0.1211 ± 0.0004 | 0.0677 ± 0.0008 | 369.0s ± 76.5s |
| FB15k | RFM-TransE (Exponential) | 0.1708 ± 0.0018 | 0.1847 ± 0.0015 | 0.2317 ± 0.0008 | 0.9331 ± 0.0016 | 0.0719 ± 0.0019 | 0.2349 ± 0.0030 | 0.1651 ± 0.0022 | 357.1s ± 71.7s |
| FB15k | RFM-TransE (Gaussian) | 0.1607 ± 0.0011 | 0.1738 ± 0.0004 | 0.2185 ± 0.0004 | 0.9301 ± 0.0002 | 0.1454 ± 0.0033 | 0.2381 ± 0.0035 | 0.1880 ± 0.0038 | 338.1s ± 98.1s |
| FB15k-237 | Standard TransE | 0.1529 ± 0.0009 | 0.1618 ± 0.0024 | 0.2054 ± 0.0011 | 0.9117 ± 0.0020 | 0.2164 ± 0.0005 | 0.3803 ± 0.0001 | 0.2663 ± 0.0013 | 200.0s ± 1.3s |
| FB15k-237 | RFM-TransE (Sigmoid) | 0.1585 ± 0.0011 | 0.1668 ± 0.0024 | 0.2097 ± 0.0024 | 0.9285 ± 0.0006 | 0.0461 ± 0.0008 | 0.1352 ± 0.0012 | 0.0711 ± 0.0012 | 232.2s ± 2.7s |
| FB15k-237 | RFM-TransE (Exponential) | 0.1644 ± 0.0023 | 0.1725 ± 0.0026 | 0.2166 ± 0.0025 | 0.9232 ± 0.0022 | 0.0765 ± 0.0028 | 0.2414 ± 0.0061 | 0.1592 ± 0.0010 | 215.9s ± 1.0s |
| FB15k-237 | RFM-TransE (Gaussian) | 0.1584 ± 0.0038 | 0.1678 ± 0.0037 | 0.2121 ± 0.0027 | 0.9258 ± 0.0006 | 0.0990 ± 0.0058 | 0.2010 ± 0.0064 | 0.1309 ± 0.0074 | 227.5s ± 4.0s |
| WN18RR | Standard TransE | 0.0725 ± 0.0128 | 0.1071 ± 0.0234 | 0.1474 ± 0.0294 | 0.7663 ± 0.0087 | 0.1999 ± 0.0088 | 0.3907 ± 0.0015 | 0.2115 ± 0.0174 | 28.8s ± 3.3s |
| WN18RR | RFM-TransE (Sigmoid) | 0.1295 ± 0.0003 | 0.2273 ± 0.0020 | 0.2898 ± 0.0014 | 0.7904 ± 0.0051 | 0.1533 ± 0.0014 | 0.2462 ± 0.0018 | 0.2132 ± 0.0021 | 38.2s ± 2.1s |
| WN18RR | RFM-TransE (Exponential) | 0.1349 ± 0.0072 | 0.2409 ± 0.0188 | 0.2989 ± 0.0162 | 0.7877 ± 0.0029 | 0.1401 ± 0.0030 | 0.2934 ± 0.0056 | 0.1758 ± 0.0025 | 38.2s ± 4.9s |
| WN18RR | RFM-TransE (Gaussian) | 0.1372 ± 0.0020 | 0.2445 ± 0.0023 | 0.3080 ± 0.0077 | 0.7855 ± 0.0012 | 0.1452 ± 0.0022 | 0.2584 ± 0.0013 | 0.2000 ± 0.0032 | 42.3s ± 12.0s |

### Sigmoid RFM-TransE across all uncertainty regimes

The following comparison isolates the final sigmoid mapping and shows how its
ranking, classification, continuous-error, calibration, and runtime measurements
change across the low, medium, and high target regimes.

| Dataset | Regime | MRR | Hits@3 | Hits@5 | F1 | MSE | MAE | ECE | Train time |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CoDEx-S | Low | 0.2213 ± 0.0045 | 0.2452 ± 0.0062 | 0.3190 ± 0.0042 | 0.8979 ± 0.0049 | 0.1273 ± 0.0068 | 0.3303 ± 0.0138 | 0.2235 ± 0.0135 | 15.2s ± 1.1s |
| CoDEx-S | Medium | 0.2223 ± 0.0031 | 0.2468 ± 0.0084 | 0.3170 ± 0.0048 | 0.8982 ± 0.0012 | 0.0824 ± 0.0037 | 0.2544 ± 0.0109 | 0.1605 ± 0.0102 | 17.5s ± 0.5s |
| CoDEx-S | High | 0.2211 ± 0.0060 | 0.2428 ± 0.0071 | 0.3114 ± 0.0052 | 0.8958 ± 0.0060 | 0.0444 ± 0.0013 | 0.1758 ± 0.0065 | 0.0997 ± 0.0038 | 21.0s ± 1.2s |
| CoDEx-M | Low | 0.1339 ± 0.0025 | 0.1468 ± 0.0025 | 0.1923 ± 0.0007 | 0.8927 ± 0.0005 | 0.0878 ± 0.0017 | 0.1775 ± 0.0023 | 0.0787 ± 0.0020 | 266.3s ± 33.6s |
| CoDEx-M | Medium | 0.1325 ± 0.0042 | 0.1443 ± 0.0051 | 0.1892 ± 0.0045 | 0.8904 ± 0.0039 | 0.0671 ± 0.0014 | 0.1736 ± 0.0022 | 0.0814 ± 0.0016 | 232.7s ± 1.5s |
| CoDEx-M | High | 0.1317 ± 0.0020 | 0.1431 ± 0.0011 | 0.1893 ± 0.0011 | 0.8890 ± 0.0017 | 0.0416 ± 0.0005 | 0.1472 ± 0.0012 | 0.0764 ± 0.0006 | 235.0s ± 10.4s |
| FB15k | Low | 0.1732 ± 0.0019 | 0.1866 ± 0.0028 | 0.2364 ± 0.0031 | 0.9419 ± 0.0008 | 0.0427 ± 0.0002 | 0.1059 ± 0.0005 | 0.0418 ± 0.0003 | 460.0s ± 77.1s |
| FB15k | Medium | 0.1707 ± 0.0041 | 0.1843 ± 0.0058 | 0.2326 ± 0.0059 | 0.9394 ± 0.0002 | 0.0374 ± 0.0004 | 0.1211 ± 0.0004 | 0.0677 ± 0.0008 | 369.0s ± 76.5s |
| FB15k | High | 0.1658 ± 0.0068 | 0.1798 ± 0.0069 | 0.2280 ± 0.0070 | 0.9371 ± 0.0011 | 0.0267 ± 0.0003 | 0.1141 ± 0.0004 | 0.0695 ± 0.0005 | 632.9s ± 531.9s |
| FB15k-237 | Low | 0.1592 ± 0.0025 | 0.1663 ± 0.0046 | 0.2107 ± 0.0052 | 0.9315 ± 0.0009 | 0.0548 ± 0.0005 | 0.1241 ± 0.0009 | 0.0495 ± 0.0010 | 266.3s ± 37.7s |
| FB15k-237 | Medium | 0.1585 ± 0.0011 | 0.1668 ± 0.0024 | 0.2097 ± 0.0024 | 0.9285 ± 0.0006 | 0.0461 ± 0.0008 | 0.1352 ± 0.0012 | 0.0711 ± 0.0012 | 232.2s ± 2.7s |
| FB15k-237 | High | 0.1568 ± 0.0044 | 0.1658 ± 0.0016 | 0.2082 ± 0.0028 | 0.9268 ± 0.0011 | 0.0310 ± 0.0002 | 0.1225 ± 0.0003 | 0.0723 ± 0.0006 | 220.3s ± 9.1s |
| WN18RR | Low | 0.1303 ± 0.0012 | 0.2252 ± 0.0023 | 0.2913 ± 0.0038 | 0.7873 ± 0.0050 | 0.2143 ± 0.0010 | 0.2695 ± 0.0007 | 0.2407 ± 0.0010 | 47.1s ± 5.8s |
| WN18RR | Medium | 0.1295 ± 0.0003 | 0.2273 ± 0.0020 | 0.2898 ± 0.0014 | 0.7904 ± 0.0051 | 0.1533 ± 0.0014 | 0.2462 ± 0.0018 | 0.2132 ± 0.0021 | 38.2s ± 2.1s |
| WN18RR | High | 0.1308 ± 0.0013 | 0.2321 ± 0.0015 | 0.2919 ± 0.0016 | 0.7892 ± 0.0020 | 0.0892 ± 0.0004 | 0.1990 ± 0.0005 | 0.1642 ± 0.0005 | 39.8s ± 4.7s |

### Medium-regime sigmoid comparison before and after the coverage change

The **Full-split Sigmoid** rows use the complete official splits. The **Earlier
Sampled Sigmoid** rows reproduce the aggregate values reported in the thesis
before the coverage change, when training was capped at 10,000 triples and
validation/test evaluation at 500 triples. Because the dataset coverage differs,
this is a comparison of benchmark protocols rather than a controlled model
ablation.

| Dataset | Version | MRR | Hits@3 | Hits@5 | F1 | MSE | MAE | ECE | Train time |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CoDEx-S | Full-split Sigmoid | 0.2223 ± 0.0031 | 0.2468 ± 0.0084 | 0.3170 ± 0.0048 | 0.8982 ± 0.0012 | 0.0824 ± 0.0037 | 0.2544 ± 0.0109 | 0.1605 ± 0.0102 | 17.5s ± 0.5s |
| CoDEx-S | Earlier Sampled Sigmoid | 0.1519 ± 0.0015 | 0.1643 ± 0.0029 | 0.2193 ± 0.0045 | 0.7997 ± 0.0125 | 0.1421 ± 0.0004 | 0.3738 ± 0.0005 | 0.1246 ± 0.0137 | 1.7s ± 0.1s |
| CoDEx-M | Full-split Sigmoid | 0.1325 ± 0.0042 | 0.1443 ± 0.0051 | 0.1892 ± 0.0045 | 0.8904 ± 0.0039 | 0.0671 ± 0.0014 | 0.1736 ± 0.0022 | 0.0814 ± 0.0016 | 232.7s ± 1.5s |
| CoDEx-M | Earlier Sampled Sigmoid | 0.0529 ± 0.0110 | 0.0520 ± 0.0108 | 0.0757 ± 0.0155 | 0.7132 ± 0.0057 | 0.1525 ± 0.0009 | 0.3826 ± 0.0012 | 0.1216 ± 0.0068 | 1.5s ± 0.0s |
| FB15k | Full-split Sigmoid | 0.1707 ± 0.0041 | 0.1843 ± 0.0058 | 0.2326 ± 0.0059 | 0.9394 ± 0.0002 | 0.0374 ± 0.0004 | 0.1211 ± 0.0004 | 0.0677 ± 0.0008 | 369.0s ± 76.5s |
| FB15k | Earlier Sampled Sigmoid | 0.0461 ± 0.0037 | 0.0473 ± 0.0021 | 0.0637 ± 0.0015 | 0.5645 ± 0.0119 | 0.1603 ± 0.0006 | 0.3918 ± 0.0006 | 0.0887 ± 0.0056 | 1.5s ± 0.0s |
| FB15k-237 | Full-split Sigmoid | 0.1585 ± 0.0011 | 0.1668 ± 0.0024 | 0.2097 ± 0.0024 | 0.9285 ± 0.0006 | 0.0461 ± 0.0008 | 0.1352 ± 0.0012 | 0.0711 ± 0.0012 | 232.2s ± 2.7s |
| FB15k-237 | Earlier Sampled Sigmoid | 0.0945 ± 0.0009 | 0.1030 ± 0.0050 | 0.1287 ± 0.0040 | 0.6272 ± 0.0397 | 0.1555 ± 0.0012 | 0.3871 ± 0.0016 | 0.1015 ± 0.0074 | 1.5s ± 0.0s |
| WN18RR | Full-split Sigmoid | 0.1295 ± 0.0003 | 0.2273 ± 0.0020 | 0.2898 ± 0.0014 | 0.7904 ± 0.0051 | 0.1533 ± 0.0014 | 0.2462 ± 0.0018 | 0.2132 ± 0.0021 | 38.2s ± 2.1s |
| WN18RR | Earlier Sampled Sigmoid | 0.0064 ± 0.0009 | 0.0047 ± 0.0012 | 0.0087 ± 0.0035 | 0.6573 ± 0.0111 | 0.1641 ± 0.0006 | 0.3953 ± 0.0005 | 0.0793 ± 0.0047 | 2.9s ± 0.6s |

The full-split results show substantially higher ranking and classification
scores than the earlier sampled experiment, especially on the larger datasets.
The sigmoid model also reduces MSE and MAE relative to the new Standard TransE
reference on every dataset under medium uncertainty. ECE improves relative to
the new Standard TransE reference on CoDEx-S, CoDEx-M, FB15k, and FB15k-237;
WN18RR is the exception, where medium-regime ECE is 0.2132 for sigmoid and
0.2115 for the fixed Standard TransE mapping. These calibration values measure
agreement with the constructed targets and should not be interpreted as
empirical real-world confidence.
