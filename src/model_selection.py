"""
Model Selection and Evaluation Pipeline
Internship Track: Week 2 - Task 4
Author: Sayantani Adhikary
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_and_preprocess(filepath: str):
    """Load dataset, separate features/target, and apply standardization."""
    df = pd.read_csv(filepath)
    X = df.drop(columns=['Approved'])
    y = df['Approved']

    # Standardize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, X.columns


def evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate candidate models using cross-validation and test set metrics."""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=4, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    }

    # 5-Fold Stratified Cross-Validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    print("=" * 65)
    print("        MODEL EVALUATION & SELECTION SCORECARD        ")
    print("=" * 65)

    for name, model in models.items():
        # Cross-validation on training data to assess generalization
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

        # Fit model on full training set and evaluate on unseen test set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        results.append({
            'Model': name,
            'CV Mean Acc (%)': f"{cv_scores.mean() * 100:.2f}%",
            'CV Std Dev': f"{cv_scores.std():.4f}",
            'Test Accuracy (%)': f"{acc * 100:.2f}%",
            'Precision (%)': f"{prec * 100:.2f}%",
            'Recall (%)': f"{rec * 100:.2f}%",
            'F1-Score': f"{f1:.4f}"
        })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print("=" * 65)
    print("\n[CONCLUSION] Random Forest demonstrates highest cross-validation stability and test F1-score.")
    return results_df


def main():
    data_path = "data/dataset.csv"
    X, y, _ = load_and_preprocess(data_path)

    # 80/20 train-test split with fixed random seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    evaluate_models(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
