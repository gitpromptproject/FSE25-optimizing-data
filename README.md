# 📦 FSE'25 Replication Package: Data-Centric Optimization in Code Summarization Dataset

This repository contains the complete replication package for the ```Data-Centric Optimization in Code Summarization Dataset``` paper:


---

## 📚 Overview

Our study explores **data-centric optimization strategies** for improving efficiency and reliability in training large language models (LLMs) for code summarization. We investigate three **token-level reduction techniques**—Abstract Syntax Tree (AST), Function Signature (FS), and CrystalBLEU (CBLUE) guided filtering—combined with **semantic filtering via the SIDE metric**. Experiments span Java and Python, including a newly retrained semantic alignment metric for Python (SIDE-py).

This replication package includes:

- Token-based optimization (Study-1)
- Training and validation of SIDE-py (Study-2)
- Cross-language replication (Study-3)

---

## 🗂️ Repository Structure

### 🔹 `study-1/` – Java Token-Level Optimization

| Path | Description |
|------|-------------|
| `codereval/` | Evaluation data and metric scripts for CodeEval (Java) |
| `training_and_inference/` | Training and inference scripts using CodeT5+ |
| `statistical-tests/Java/` | Wilcoxon + Cliff’s Delta R scripts and results |

### 🔹 `study-2/` – Training `SIDE-py` for Python

| Path | Description |
|------|-------------|
| `data-files/` | Triplet data for SIDE-py training |
| `training-sidep/` | MPNet-based training scripts for SIDE-py |
| `scripts_for_benchmark/` | Scripts for logistic regression and plotting |

### 🔹 `study-3/` – Python Generalization Evaluation

| Path | Description |
|------|-------------|
| `evaluation-benchmark-data/` | Annotated benchmark data (Cupri et al.) |
| `inference-results/` | Predictions from all optimization variants |
| `scripts/` | SIDE-py filtering and model evaluation |
| `statistical-tests/Python/` | Regression and correlation R scripts |

---

## ⚙️ Environment Setup

We provide a pre-configured Conda environment to ensure full reproducibility.

### 📌 Step-by-step Setup

```bash
# 1. Create and activate Conda environment
conda env create -f environment.yml
conda activate fse25-optimize

# 2. [Optional] Verify installation
python -m torch --version