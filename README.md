# 📄 Learning to Price ML Datasets via Posted Price and Auction

This repository contains the code and experiments for our paper:

**📄 [Learn then Decide: A Learning Approach for Designing Data Marketplaces](https://arxiv.org/abs/2503.10773)**  
Yingqi Gao, Jin Zhou, Hua Zhou, Yong Chen, Xiaowu Dai\
*Submitted to Journal of the American Statistical Association, 2025*

---

## 🧾 Overview

We propose a two-stage mechanism for selling machine learning datasets that combines posted pricing with auctions. This repo includes:

- Implementation of all mechanisms described in the paper
- Scripts to simulate multiple settings
- Tools for evaluating regret, revenue, and fairness
- Reproduction of all tables and figures in the main text and appendix

---

## 🚀 Getting Started

This project requires both a Python environment (managed via `conda`) and an R environment (version 4.2) with access to the `densityFPCA` package.

### 🔧 1. Clone the Repository and Set Up the Python Environment

```bash
git clone https://github.com/yingqi-gao/MAPP.git
cd MAPP

# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate mapp

# (Optional) Register as a Jupyter kernel
python -m ipykernel install --user --name mapp --display-name "Python (mapp)"
```

### 🧬 2. Set Up the R Environment

This project uses R 4.2. On macOS with the CRAN binary, you may need to set the R_HOME environment variable so Python (via rpy2) can locate the correct R installation.

```bash
export R_HOME="/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources"
```

### 📦 3. Install Required R Packages

Open R 4.2 in your terminal and run the following:

```r
install.packages("devtools")  # if not already installed
devtools::install_github("jiamingqiu/densityFPCA")
install.packages("tidyverse") # if not already installed
```

📌 For details, see the densityFPCA GitHub repo.

💡 If compilation errors occur, ensure that your C/C++ toolchain (e.g., Homebrew gcc-15) is installed and configured. You may need to edit your ~/.R/Makevars file to specify the correct compiler paths.


## 📊 Reproducing Results

To reproduce the figures and tables from the paper:

### ✅ Simulated Data

- Open and run [`simulations.ipynb`](simulations.ipynb)  
  This notebook generates all synthetic datasets, runs the pricing mechanisms, and produces the figures and tables corresponding to our simulation studies.

### ✅ Real Data

- Open and run [`real.ipynb`](real.ipynb)  
  This notebook reproduces the results from our real-world dataset experiments.

---

### 🧠 Code Components Overview

Our method integrates both Python and R components. The core modules include:

- [`_density_estimation.r`](./_density_estimation.r)  
  Adapts the repeated density estimation method from [Qiu et al. (densityFPCA)](https://github.com/jiamingqiu/densityFPCA) in R.

- [`_price_optimization.py`](./_price_optimization.py)  
  Implements the price optimization routine based on estimated densities.

- [`_simulations.py`](./_simulations.py)  
  Contains utilities for generating synthetic data under various scenarios.

- [`_plot.py`](./_plot.py)  
  Provides helper functions for visualizing results and generating figures.

These components are used directly within the two Jupyter notebooks and require no separate configuration.
