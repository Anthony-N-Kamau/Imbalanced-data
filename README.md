# Imbalanced Data: Handling Class Imbalance in Classification

This repository contains a Jupyter notebook exploring techniques for
handling the **class imbalance problem** in machine learning — where one
class in a classification task is dramatically underrepresented compared to
another. The notebook demonstrates the problem on a synthetic 2D dataset
and compares three strategies for addressing it: SMOTE oversampling,
GAN-based synthetic data generation, and cost-sensitive classification.

## Overview

A synthetic imbalanced dataset is generated with 10,000 samples split
99%/1% between two classes (9,900 majority-class samples vs. 100
minority-class samples). A logistic regression baseline is trained on this
raw imbalanced data, then three separate remediation techniques are applied
and compared against the baseline using F1-Score, Accuracy, and Balanced
Accuracy.

## Contents

| Section | Description |
|---|---|
| **Handling the Imbalanced Data Problem** | Generates a synthetic 2D dataset (`make_classification`) with a 99:1 class imbalance and visualizes the class distribution |
| **Baseline model** | Splits the data 70/30 (stratified), trains a `LogisticRegression` classifier on the raw imbalanced training set, and evaluates it with a confusion matrix, F1-Score, Accuracy, and Balanced Accuracy |
| **SMOTE for Balancing Data** | Applies Synthetic Minority Oversampling Technique (SMOTE) to the training set only, generating synthetic minority-class examples until both classes are balanced (6,930 vs. 6,930), then retrains and re-evaluates the model |
| **C-GAN: Generate Synthetic Data** | Uses a Conditional Tabular GAN (`CTGAN` via the `sdv` library) to learn the data distribution and generate new minority-class samples under a specified condition, balancing the dataset, then retrains and re-evaluates the model |
| **Cost Sensitive Classification** | Trains a `LogisticRegression` model directly on the original imbalanced data, but assigns a much higher misclassification cost to the minority class via `class_weight`, and evaluates the resulting model |

## Key findings

- On the raw imbalanced data, the baseline logistic regression achieves a
  very high **99.4% accuracy** but a much lower **73.3% balanced accuracy**
  and a weak minority-class F1-score of **0.60** — a textbook illustration
  of how overall accuracy is a misleading metric under class imbalance,
  since the model mostly just predicts the majority class.
- **SMOTE oversampling** raises balanced accuracy substantially (to
  **89.1%**), at the cost of overall accuracy dropping (to 91.4%) and more
  false positives, since the classifier now pays much more attention to the
  minority class.
- **CTGAN-generated synthetic data** achieves similar or slightly better
  balanced accuracy (**91.7%**) compared to SMOTE, generating minority-class
  samples via a learned generative model rather than simple geometric
  interpolation.
- **Cost-sensitive classification** (adjusting `class_weight` without
  generating any new data) achieves comparable balanced accuracy (**88.3%**)
  to the resampling methods, without needing to alter the training data at
  all — a useful alternative when synthetic data generation is undesirable
  or infeasible.
- All three remediation methods trade some overall accuracy for
  substantially better minority-class detection and balanced accuracy
  compared to the untreated baseline.

## Requirements

This project uses Python with the following packages:

```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib sdv ctgan
```

- `numpy` / `pandas` — data handling
- `scikit-learn` — synthetic data generation (`make_classification`),
  train/test splitting, `LogisticRegression`, and evaluation metrics
  (confusion matrix, accuracy, balanced accuracy, F1-score)
- `imbalanced-learn` (`imblearn`) — SMOTE oversampling
- `matplotlib` — visualizing class distributions before/after balancing
- `sdv` / `ctgan` — Conditional Tabular GAN (CTGAN) for synthetic minority-class data generation

## Repository structure

```
.
├── imbalanced_data.ipynb    # Jupyter notebook source
├── imbalanced_data.pdf       # Rendered PDF export of the notebook
└── README.md
```

## Usage

Open `imbalanced_data.ipynb` in Jupyter or Google Colab and run all cells
top to bottom. The notebook was authored and exported from Google Colab,
using `nbconvert` and `xelatex` to produce the accompanying PDF.
