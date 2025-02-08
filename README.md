# Loan Default Prediction Using Machine Learning

This project predicts loan default using various machine learning techniques, focusing on feature selection, handling class imbalance, dimensionality reduction, and implementing classifiers such as SVMs and Dual SVMs. The pipeline is designed to preprocess data, evaluate features, and train models while ensuring scalability and interpretability.

---

## Table of Contents

- [Features](#features)
- [Data](#data)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Pipeline Overview](#pipeline-overview)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

---

## Features

- **Data Preprocessing:**
  - Exploratory Data Analysis (EDA).
  - Feature selection using Recursive Feature Elimination (RFE).
  - Correlation-based feature optimization.
  - Handling class imbalance using SMOTE.
  - Data normalization.

- **Custom Algorithms:**
  - **LinearSVM:** Gradient-based linear SVM implementation.
  - **LinearSVM_v2:** Enhanced SVM with hyperparameter tuning and early stopping.
  - **DualSVM:** Custom kernel-based SVM supporting linear and RBF kernels.

- **Dimensionality Reduction:**
  - Custom PCA implementation to reduce dimensionality.

- **Model Evaluation:**
  - Confusion Matrix, Classification Reports, and F1 Scores for imbalanced datasets.

---

## Data

The dataset used in this project is `loan_dataset.csv`, containing features such as loan amount, annual income, interest rate, and the target variable `loan_condition_cat`:
- `0`: Defaulted loan.
- `1`: Non-defaulted loan.
https://www.kaggle.com/datasets/mrferozi/loan-data-for-dummy-bank/data
---

## Requirements

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

---

## Setup

1. Place `loan_dataset.csv` in the project directory.
2. Run the script to preprocess the dataset and train models.
3. Modify parameters in the script as needed to experiment with different configurations.

---

## Usage

### Step 1: Run the Script

Execute the script:

```bash
python loan_prediction.py
```

### Step 2: Review Results

Examine the confusion matrix, classification report, and feature selection results.

### Step 3: Experiment

Adjust hyperparameters, enable SMOTE, or modify features to experiment with different configurations.

---

## Pipeline Overview

1. **Data Preprocessing:**
   - Data exploration (EDA) and null value checks.
   - Feature selection using RFE and correlation heatmaps.

2. **Class Imbalance Handling:**
   - Optional SMOTE implementation for balancing classes.

3. **Normalization:**
   - Normalize features for improved model performance.

4. **Dimensionality Reduction:**
   - Apply custom PCA for reducing dimensionality.

5. **Model Training:**
   - Train SVM models (LinearSVM, LinearSVM_v2, DualSVM).

6. **Evaluation:**
   - Evaluate model performance using classification metrics.

---

## Results

### Sample Output:

- **Confusion Matrix:**
  ```
  [[340  45]
   [ 50 565]]
  ```

- **Classification Report:**
  ```
               precision    recall  f1-score   support

            0       0.87      0.88      0.87       385
            1       0.93      0.92      0.92       615

       accuracy                           0.91      1000
      macro avg       0.90      0.90      0.90      1000
   weighted avg       0.91      0.91      0.91      1000
  ```

- **Training Time:** Approximately ~2.5 seconds (depends on dataset size and hardware).

---


##### NOTES: some lines of code are commented to avoid unwanted and large runtimes to perform some feature, read carefully and uncomment them according to your needs. Also, this project requires a plenty of computational capacity to run.

## Acknowledgments

- Inspired by machine learning workflows for imbalanced classification problems.
- Combines theoretical concepts and practical implementations of SVMs, PCA, and feature selection techniques.


## Authors

- [@Gilber Hernandez](https://www.github.com/gilbermhb)
- [@Saira Guzman]( )

