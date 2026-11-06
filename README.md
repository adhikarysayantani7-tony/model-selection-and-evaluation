# Model Selection and Evaluation

## Project Overview
This repository implements a rigorous machine learning model selection and evaluation pipeline applied to credit risk assessment. We benchmark Logistic Regression, Decision Tree, and Random Forest classifiers using 5-Fold Stratified Cross-Validation, analyze bias-variance trade-offs, and justify the optimal model for production deployment.

## Repository Architecture
- README.md
- requirements.txt
- data/dataset.csv
- src/model_selection.py
- notebooks/model_selection_and_evaluation.ipynb
- docs/model_comparison_report.md

## Models Evaluated
1. Logistic Regression: Linear probability baseline providing high interpretability.
2. Decision Tree Classifier: Non-linear tree model showing susceptibility to variance.
3. Random Forest Classifier (Selected): Ensemble bagging architecture achieving superior generalization and zero test error.

## Model Performance Scorecard
* Logistic Regression: 5-Fold CV Mean = 81.25%, Test Accuracy = 75.00%, F1-Score = 0.7333
* Decision Tree: 5-Fold CV Mean = 87.50%, Test Accuracy = 75.00%, F1-Score = 0.7333
* Random Forest (Champion): 5-Fold CV Mean = 93.75%, Test Accuracy = 100.00%, F1-Score = 1.0000

## Documentation & Analysis
A comprehensive technical comparison covering cross-validation stability, overfitting mitigation, and deployment justification is detailed in `docs/model_comparison_report.md`.

## How to Run
1. Clone this repository:
   git clone https://github.com/adhikarysayantani7-tony/model-selection-and-evaluation.git
2. Install dependencies:
   pip install -r requirements.txt
3. Run the evaluation script:
   python src/model_selection.py
4. Open the interactive notebook:
   jupyter notebook notebooks/model_selection_and_evaluation.ipynb
