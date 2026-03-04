"""
This script includes training, hyperparameter tuning (by manual editing)
and cross validation for the ADABoost classifier.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score
)


# DATA PREPARATION FUNCTION

def prepare_data(df, target="increase_stock"):
    #month_cols = [c for c in df.columns if c.startswith("month")]

    removals = [
        target,
        "windspeed",
        "precip",
        "dew",
        "cloudcover",
        "snowdepth",
    ]

    cols_to_drop = [c for c in removals  if c in df.columns]

    X = df.drop(columns=cols_to_drop)
    y = (df[target] == 1).astype(int)

    return X, y



# TRAINING FUNCTION (CV + THRESHOLD SEARCH)


def train_model(
    X,
    y,
    n_splits=10,
    threshold_range=(0.1, 0.7),
    threshold_steps=40,
    random_state=42
):
    """
    Performs cross-validation and finds best threshold based on F1.
    Returns trained model, best threshold, and CV metrics.
    """

    base_learner = DecisionTreeClassifier(
        max_depth=2,
        random_state=random_state
    )

    model = AdaBoostClassifier(
        estimator=base_learner,
        n_estimators=300,
        learning_rate=0.05,
        random_state=random_state
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    thresholds = np.linspace(
        threshold_range[0],
        threshold_range[1],
        threshold_steps
    )

    best_threshold = 0.5
    best_f1 = -1

    # Store average metrics for best threshold
    best_scores = None

    for r in thresholds:

        f1_total = 0
        acc_total = 0
        roc_total = 0
        pr_total = 0
        prec_total = 0
        rec_total = 0

        for train_idx, val_idx in skf.split(X, y):

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_train, y_train)

            prob = model.predict_proba(X_val)[:, 1]
            pred = (prob >= r).astype(int)

            f1_total += f1_score(y_val, pred)
            acc_total += accuracy_score(y_val, pred)
            roc_total += roc_auc_score(y_val, prob)
            pr_total += average_precision_score(y_val, prob)
            prec_total += precision_score(y_val, pred)
            rec_total += recall_score(y_val, pred)

        f1_mean = f1_total / n_splits

        if f1_mean > best_f1:
            best_f1 = f1_mean
            best_threshold = r
            best_scores = {
                "accuracy": acc_total / n_splits,
                "f1": f1_mean,
                "precision": prec_total / n_splits,
                "recall": rec_total / n_splits,
                "roc_auc": roc_total / n_splits,
                "pr_auc": pr_total / n_splits,
            }

    # Retrain on FULL dataset
    model.fit(X, y)

    print("\nBest threshold:", round(best_threshold, 3))
    print("Cross-validated scores:")
    for k, v in best_scores.items():
        print(f"{k}: {v:.3f}")

    return model, best_threshold, best_scores


# TESTING FUNCTION (NEW DATA)

def test_model(model, X_test, y_test, threshold):
    """
    Evaluates trained model on completely unseen test data.
    """

    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)

    scores = {
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, prob),
        "pr_auc": average_precision_score(y_test, prob),
    }

    print("\nTest set performance:")
    for k, v in scores.items():
        print(f"{k}: {v:.3f}")

    return scores

# Load data
train_df = pd.read_csv("data/preprocessed_training_data.csv")
test_df = pd.read_csv("data/preprocessed_testing_data.csv")

# Prepare
X_train, y_train = prepare_data(train_df)
X_test, y_test = prepare_data(test_df)

# Train
model, best_r, cv_scores = train_model(X_train, y_train)

# Test on unseen data
test_scores = test_model(model, X_test, y_test, best_r)
