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

## 🔧 Installation

We recommend using a Python virtual environment:


```bash
git clone https://github.com/yingqi-gao/MAPP.git
cd MAPP
conda env create -f environment.yml
conda activate mapp
```

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
