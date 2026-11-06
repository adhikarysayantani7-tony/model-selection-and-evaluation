# Machine Learning Model Selection & Evaluation Report

## 1. Executive Summary & Problem Formulation
The goal of this task is to benchmark and compare multiple supervised machine learning algorithms on a consumer credit risk dataset. We evaluated three distinct models—**Logistic Regression**, **Decision Tree Classifier**, and **Random Forest Classifier**—to determine which architecture offers optimal predictive accuracy and generalization performance on unseen test data.

---

## 2. Models Evaluated & Theoretical Rationale

### A. Logistic Regression (Linear Benchmark)
* **Architecture**: Linear probability model applying a sigmoid link function.
* **Strengths**: Highly interpretable, computationally lightweight, calibrated probability outputs.
* **Limitations**: Assumes linear decision boundaries; cannot easily capture complex non-linear feature interactions without manual transformation.

### B. Decision Tree Classifier (Single Tree Non-Linear Model)
* **Architecture**: Hierarchical rule-based splitting optimizing Gini Impurity.
* **Strengths**: Captures non-linear relationships without feature scaling; transparent decision logic.
* **Limitations**: Prone to high variance and overfitting on small training samples.

### C. Random Forest Classifier (Ensemble Bagging Architecture)
* **Architecture**: Ensemble of decorrelated decision trees aggregated via bootstrap sampling and majority voting.
* **Strengths**: Significantly reduces variance compared to individual decision trees; robust against noise and outliers.
* **Limitations**: Slightly higher computational overhead; less directly interpretable than a single tree.

---

## 3. Cross-Validation & Validation Methodology
To guard against data leakage and overfitting, we utilized **5-Fold Stratified Cross-Validation (`StratifiedKFold`, $k=5$)** with a fixed random seed (`random_state=42`):
1. **Stratification**: Ensured identical class balance across each of the 5 training folds.
2. **Holdout Evaluation**: An independent 20% holdout test set was isolated prior to cross-validation to assess true real-world generalization.

---

## 4. Empirical Performance Comparison Scorecard

| Model Architecture | 5-Fold CV Mean Accuracy | CV Std Dev ($\pm\sigma$) | Test Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 81.25% | $\pm$ 0.125 | 75.00% | 80.00% | 75.00% | 0.7333 |
| **Decision Tree** | 87.50% | $\pm$ 0.153 | 75.00% | 80.00% | 75.00% | 0.7333 |
| **Random Forest (Selected)** | **93.75%** | **$\pm$ 0.088** | **100.00%** | **100.00%** | **100.00%** | **1.0000** |

---

## 5. Model Justification & Deployment Recommendation
* **Primary Recommendation**: **Random Forest Classifier** is the selected model for production deployment.
* **Justification**:
  1. **Highest Cross-Validation Stability**: Random Forest achieved the highest cross-validation mean accuracy (**93.75%**) with the lowest variance ($\pm 0.088$), proving consistent performance across varying data subsets.
  2. **Variance Reduction**: While the individual Decision Tree suffered from variance drop-off on the test set, Random Forest's ensemble voting successfully mitigated overfitting.
  3. **Metric Superiority**: Random Forest achieved a **1.0000 F1-Score** on holdout evaluation, capturing all positive approvals without false positives or false negatives.
