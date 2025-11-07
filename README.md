# MAPP: Maximum Auction-to-Posted Price

This repository contains the code and experiments for our paper:

**[Learn then Decide: A Learning Approach for Designing Data Marketplaces](https://arxiv.org/abs/2503.10773)**
Yingqi Gao, Jin Zhou, Hua Zhou, Yong Chen, Xiaowu Dai
*Submitted to Journal of the American Statistical Association, 2025*

---

## Overview

We propose the **Maximum Auction-to-Posted Price (MAPP)** mechanism, a two-stage approach for pricing in data marketplaces:

1. **Learning Stage (Auction):** Collect bids from initial buyers to learn the value distribution
2. **Decision Stage (Posted Price):** Use learned distribution to set optimal posted prices for subsequent sales

**Key Properties:**
- ✅ Incentive Compatible (leave-one-out pricing ensures truthful bidding)
- ✅ Individually Rational (buyers only pay if value ≥ price)
- ✅ Revenue Optimal (maximizes seller revenue given learned distribution)

**Pricing Methods Evaluated:**
- **ECDF**: Empirical CDF (non-parametric baseline)
- **KDE**: Kernel Density Estimation
- **RDE**: Repeated Density Estimation (FPCA-based, proposed in paper)
- **Myerson**: Theoretical benchmark (assumes true distribution known)
- **MyersonNet**: Deep learning-based optimal auction mechanism

---

## Getting Started

### Requirements

- **Python 3.11** (for reproducibility)
- **R 4.0+** (4.2+ recommended)
- **macOS/Linux/Windows** (macOS requires Xcode command line tools)

### One-Command Setup

```bash
git clone https://github.com/yingqi-gao/MAPP.git
cd MAPP
./setup.sh
```

The setup script will:
- ✅ Check R installation (≥4.0, recommends ≥4.2)
- ✅ Install R packages: `fdapace` (CRAN) + [`densityFPCA`](https://github.com/jiamingqiu/densityFPCA) (GitHub)
- ✅ Create Python virtual environment (`venv/`)
- ✅ Install all Python dependencies with pinned versions
- ✅ Test Python-R integration

**For manual installation or troubleshooting**, see the [Installation Guide](#detailed-installation) below.

### Activate Environment

```bash
source venv/bin/activate
```

---

## Reproducing Paper Results

### Simulated Data Experiments

Open and run the notebook:
```bash
jupyter notebook notebooks/simulated_data.ipynb
```

This notebook:
- Generates synthetic auction data from 4 distribution families (truncnorm, truncexpon, beta, truncpareto)
- Trains RDE and MyersonNet models on separate training data
- Evaluates all pricing methods across different data sparsity levels
- Produces k-fold sensitivity plots and regret histograms

**Experimental Design:**
- 4 distributions × 4 bid counts (10, 50, 100, 200) × 3 k-folds (2, 5, 10) = 48 experiments
- 1000 runs per experiment for statistical significance
- Automatic caching (rerun uses cached data/models)

### Real Data Experiments

Open and run the notebook:
```bash
jupyter notebook notebooks/real_data.ipynb
```

Applies MAPP to real-world FCC spectrum auction data.

---

## Repository Structure

The code is organized into a Python package (`mapp/`) and Jupyter notebooks (`notebooks/`):

- **`mapp/core/`** - Core auction logic and experiment runner
- **`mapp/data/`** - Synthetic data generation
- **`mapp/methods/`** - All pricing methods (ECDF, KDE, RDE, Myerson, MyersonNet)
- **`mapp/experiments/`** - Experiment orchestration with caching
- **`mapp/utils/`** - Plotting, result loading, global configuration
- **`notebooks/`** - Jupyter notebooks for reproducing paper results

Browse the full structure on [GitHub](https://github.com/yingqi-gao/MAPP).

---

## Key Concepts

- **CDF (Cumulative Distribution Function):** Characterizes buyer value distribution, used to compute optimal price
- **Leave-One-Group-Out (k-fold):** Ensures incentive compatibility - each buyer's bid is excluded when calculating their price
- **Regret:** `Ideal Revenue - Actual Revenue` (lower is better)
  - **Ideal Revenue:** Revenue using oracle price with true CDF
  - **Actual Revenue:** Revenue using price from estimated CDF
- **Training Data:** Separate dense auction data (200×200) used to train RDE and MyersonNet models
- **Test Data:** Sparser auction data (10-200 bids) used to evaluate methods

---

## Configuration

All global settings are in `mapp/utils/constants.py`:

```python
# Value bounds for data generation
VALUE_LOWER_BOUND = 1.0
VALUE_UPPER_BOUND = 10.0

# Optimization bounds for pricing methods
OPTIMIZATION_LOWER_BOUND = 1.0
OPTIMIZATION_UPPER_BOUND = 10.0
```

**Important:** All experiments use consistent bounds. Modify `constants.py` to change bounds globally.

---

## Caching and Reproducibility

All expensive operations are automatically cached:

| Data Type | Location | Naming Pattern |
|-----------|----------|----------------|
| Test Data | `workspace/{dist}/data/` | `{dist}_a{auctions}_b{bids}_r{runs}_s{seed}_test.json` |
| Train Data | `workspace/{dist}/data/` | `{dist}_a{auctions}_b{bids}_s{seed}_train.json` |
| RDE Models | `workspace/{dist}/rde_models/` | `{dist}_N{N_train}_n{n_train}_l{lower}_u{upper}.pkl` |
| MyersonNet | `workspace/{dist}/myerson_net_models/` | `{dist}_N{N}_n{n}_agents{a}_epochs{e}.pt` |
| Results | `workspace/{dist}/regrets/` | `{dist}_b{bids}_k{k}_{method}.pkl` |

**Reproducibility:**
- `requirements.txt` - Pinned dependencies for exact version control
- All random seeds fixed in notebooks
- Cached data ensures identical results across runs

To regenerate from scratch, delete `workspace/` and rerun notebooks.

---

## Detailed Installation

### Step 1: Install R

**macOS:**
```bash
brew install r  # or download from https://cran.r-project.org/bin/macosx/
```

**Ubuntu:**
```bash
sudo apt-get install r-base r-base-dev
```

**Windows:**
Download from https://cran.r-project.org/bin/windows/base/

**Verify installation:**
```bash
R --version  # Should show 4.0+ (4.2+ recommended)
```

### Step 2: Install R Packages

```r
# In R console:
install.packages("fdapace")
install.packages("remotes")
remotes::install_github("jiamingqiu/densityFPCA")
```

**macOS Troubleshooting:** If compilation fails, install gfortran:
```bash
brew install gcc
```

### Step 3: Set Up Python Environment

```bash
# Create virtual environment with Python 3.11
python3.11 -m venv venv

# Activate environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Configure R_HOME

**macOS:**
```bash
export R_HOME="/Library/Frameworks/R.framework/Resources"
# Add to venv/bin/activate for persistence
echo 'export R_HOME="/Library/Frameworks/R.framework/Resources"' >> venv/bin/activate
```

**Linux:**
```bash
export R_HOME=$(R RHOME)
```

### Step 5: Test Installation

```bash
# Test Python-R bridge
python mapp/methods/cdf_based/estimation/rbridge.py

# Test MyersonNet
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
```

---

## Dependencies

### Python (requirements.txt)

- **Core:** numpy, pandas, scipy
- **Visualization:** matplotlib, seaborn
- **ML:** scikit-learn, torch (for MyersonNet)
- **Notebook:** jupyter, ipykernel
- **R Integration:** rpy2

### R

- **fdapace** (CRAN): Functional principal component analysis
- **[densityFPCA](https://github.com/jiamingqiu/densityFPCA)**: Repeated density estimation

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{gao2025learn,
  title={Learn then Decide: A Learning Approach for Designing Data Marketplaces},
  author={Gao, Yingqi and Zhou, Jin and Zhou, Hua and Chen, Yong and Dai, Xiaowu},
  journal={arXiv preprint arXiv:2503.10773},
  year={2025}
}
```

---

## Contact

For questions or issues:
- Open a GitHub issue
- Contact: yqg36@g.ucla.edu

---

## Acknowledgments

- **MyersonNet** implementation adapted from ["Optimal Auctions through Deep Learning" (Dütting et al., 2019)](https://arxiv.org/abs/1706.03459)
- **[densityFPCA](https://github.com/jiamingqiu/densityFPCA)** package by Jiaming Qiu
