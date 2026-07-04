# Version-Context, Cross-Ecosystem Malicious Package Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Overview
This repository contains the complete codebase and data reproducibility artifacts for the research manuscript **"Version-Context, Cross-Ecosystem Detection"**. 

It provides a fully reproducible pipeline for detecting software supply chain attacks across multiple ecosystems (npm and PyPI). The core methodology relies on extracting delta-features between consecutive package versions to capture malicious intent accurately.

## ✨ Key Research Contributions
This pipeline is designed to address five outstanding challenges in supply-chain security:
1. **Ecosystem Balance**: Diagnostic and per-ecosystem evaluation to provide transparent reporting of npm vs. PyPI performance.
2. **Ablation Studies**: Measures and isolates the contribution of each feature group (e.g., static signals, AST size, entropy).
3. **Semantic Embedding Layer**: Optional CodeBERT integration for deep embeddings of changed code.
4. **Temporal Generalization**: Employs a strict temporal hold-out split (training on pre-cutoff incidents, testing on newer ones) to evaluate the model's ability to catch unseen attacks.
5. **Statistical Rigor**: Reports 95% Confidence Intervals (CI) for all metrics and paired t-tests for statistical significance between models.

## 📁 Repository Structure
```text
├── data/
│   └── merged_dataset_usable.csv   # The primary dataset (ensure this is placed here before running)
├── cross_ecosystem_pipeline.py     # The main end-to-end execution script
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules for large data and cache files
└── README.md                       # Project documentation
