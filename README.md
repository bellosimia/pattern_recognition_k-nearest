# KNN Classification: Breast Cancer Diagnosis

A K-Nearest Neighbor classifier built to distinguish malignant from benign breast tumors, using the Breast Cancer Wisconsin (Diagnostic) dataset. This project was built as a pattern recognition mini-project covering the full instance-based learning workflow: preprocessing, hyperparameter tuning via cross-validation, and evaluation with multiple classification metrics.

## Project Overview

KNN classifies a new data point based on the majority class among its k nearest neighbors in feature space, where "nearest" is determined by a distance metric (Euclidean, by default). It's a simple algorithm conceptually, but it's also a useful lens for understanding why feature scaling matters in machine learning: features on a smaller numeric scale get drowned out by features on a larger scale unless everything is standardized first. That point is demonstrated directly in this project with a real before/after accuracy comparison, not just stated as a rule of thumb.

## Dataset

**Breast Cancer Wisconsin (Diagnostic)**, loaded via `sklearn.datasets.load_breast_cancer`.

- 569 samples, 30 numeric features
- Features are computed from digitized images of fine needle aspirate (FNA) biopsies of breast masses, describing characteristics of cell nuclei (radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, fractal dimension, each given as mean, standard error, and "worst"/largest value)
- Binary target: malignant (212 samples) or benign (357 samples)
- No missing values

## Project Structure

```
knn-classification/
├── knn_classification.ipynb   # Main notebook, all cells executed
├── README.md                  # This file
└── requirements.txt           # Python dependencies
```

## Workflow

1. **Load and explore** the dataset, check for missing values, examine class balance and feature distributions
2. **Train/test split** (80/20), stratified to preserve class balance in both sets
3. **Feature scaling** using `StandardScaler`, fit on training data only to avoid data leakage
4. **Baseline model** trained with k=5 as an initial reference point
5. **Hyperparameter tuning**: 5-fold cross-validation across k = 1, 3, 5, 7, 9, 11 to find the best-performing k
6. **Final evaluation** on the held-out test set using accuracy, precision, recall, F1-score, and a confusion matrix
7. **Scaling impact comparison**: same model, same split, scaled vs. unscaled features, to show the effect directly rather than just asserting it
8. **Sample predictions** on individual test points with class probabilities, to show how KNN's confidence behaves at low k

## Key Findings

- Cross-validation identified **k=3** as the best-performing and most consistent choice among the tested values.
- The final model reached **98.25% accuracy**, **97.3% precision**, **100% recall**, and an **F1-score of 0.986** on the test set, with 2 misclassifications out of 114 samples (both false negatives).
- Feature scaling accounted for a **5.26 percentage point** accuracy difference (98.25% scaled vs. 92.98% unscaled) with everything else held constant, which is a direct demonstration of why distance-based algorithms like KNN need standardized inputs.
- In a diagnostic context, the false negatives (malignant cases predicted as benign) are the more consequential error type and are worth tracking closely if this model were extended.

## Setup

```bash
pip install -r requirements.txt
jupyter notebook knn_classification.ipynb
```

## Requirements

See `requirements.txt`. Core dependencies: scikit-learn, pandas, numpy, matplotlib, seaborn.
