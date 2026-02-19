import pandas as pd
import numpy as np
import sklearn.discriminant_analysis as skl_da 
import sklearn.model_selection as skl_ms
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    recall_score,
    precision_score,
    auc
)
from matplotlib import pyplot as plt

pd.options.display.max_columns = None
pd.options.display.max_rows = None

# -----------------------------
# 1) READ DATA
# -----------------------------

df = pd.read_csv('strat_preprocessed_training_data.csv')

#define potential removals
month = [c for c in df.columns if c.startswith("month")]
day_of_week = [c for c in df.columns if c.startswith("day_of_week")]
hour_of_day = [c for c in df.columns if c.startswith("hour_of_day")]

# Remove features "summertime", "dew", "snowdepth", "holiday", "weekday", "visibility", r: 0.26

removals = ["increase_stock",
              #"windspeed",
              #"summertime",
              #"precip",
              #"humidity",
              #"dew",
              #"temp",
              #"cloudcover",
              #"snowdepth",
              #"holiday",
              "weekday",
              "visibility"
              ]

cols_to_drop = removals

existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
x = df.drop(columns=existing_cols_to_drop)

# define output
y = df['increase_stock']
y = (y == 1).astype(int)   # convert -1 → 0

# -----------------------------
# 2) DEFINE MODEL
# -----------------------------

model = skl_da.LinearDiscriminantAnalysis() # LDA
# model = skl_da.QuadraticDiscriminantAnalysis(reg_param=0.1) # QDA

# -----------------------------
# 3) PLOTTING FUNCTION 
# -----------------------------

def plot_auc_curves(tprs, aucs, precisions, aps, mean_fpr, mean_recall):
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_tpr[-1] = 1.0

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    plt.figure()
    plt.plot(mean_fpr, mean_tpr,
             label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})")
    plt.fill_between(
        mean_fpr,
        mean_tpr - std_tpr,
        mean_tpr + std_tpr,
        alpha=0.2
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    

    mean_precision = np.mean(precisions, axis=0)
    std_precision = np.std(precisions, axis=0)

    mean_ap = np.mean(aps)
    std_ap = np.std(aps)

    baseline = y.mean()

    plt.figure()
    plt.plot(
        mean_recall,
        mean_precision,
        label=f"Mean PR (AP = {mean_ap:.3f} ± {std_ap:.3f})"
    )
    plt.fill_between(
        mean_recall,
        mean_precision - std_precision,
        mean_precision + std_precision,
        alpha=0.2
    )

    plt.hlines(
        baseline, 0, 1, linestyles="--",
        label=f"Baseline (p={baseline:.2f})"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    

# -----------------------------
# 4) K-FOLD LOOP 
# -----------------------------

def k_fold_loop(model, x, y, r=0.5, n_splits=10, plot_curves=False):

    accuracy = 0
    f1 = 0
    roc_auc = 0
    pr_auc = 0
    pr = 0
    rec = 0

    tprs = []
    aucs = []
    precisions = []
    aps = []

    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(x):

        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(x_train, y_train)

        prob = model.predict_proba(x_test)[:, 1]
        pred = (prob >= r).astype(int)

        f1 += f1_score(y_test, pred)
        accuracy += accuracy_score(y_test, pred)
        roc_auc += roc_auc_score(y_test, prob)
        pr_auc += average_precision_score(y_test, prob)
        pr += precision_score(y_test, pred)
        rec += recall_score(y_test, pred)

        if plot_curves:
            fpr, tpr, _ = roc_curve(y_test, prob)
            roc_auc_for_plt = auc(fpr, tpr)

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0

            tprs.append(interp_tpr)
            aucs.append(roc_auc_for_plt)

            precision, recall, _ = precision_recall_curve(y_test, prob)
            ap = average_precision_score(y_test, prob)

            interp_precision = np.interp(
                mean_recall,
                recall[::-1],
                precision[::-1]
            )

            precisions.append(interp_precision)
            aps.append(ap)

    scores = {
        "accuracy": accuracy / n_splits,
        "f1": f1 / n_splits,
        "precision": pr / n_splits,
        "recall": rec / n_splits,
        "roc_auc": roc_auc / n_splits,
        "pr_auc": pr_auc / n_splits
    }

    if plot_curves:
        plot_auc_curves(tprs, aucs, precisions, aps, mean_fpr, mean_recall)

    return scores

# -----------------------------
# 5) THRESHOLD GRID SEARCH 
# -----------------------------

def grid_search_r(model, x, y, start=0.15, stop=0.95, num=80):

    rs = np.linspace(start, stop, num)

    accuracies = []
    f1s = []
    precisions = []
    recalls = []

    for r in rs:
        scores = k_fold_loop(model, x, y, r)

        accuracies.append(scores["accuracy"])
        f1s.append(scores["f1"])
        precisions.append(scores["precision"])
        recalls.append(scores["recall"])

        print(f"r: {r:.2f}")
        print(f"Accuracy: {scores['accuracy']:.3f} "
             f"F1: {scores['f1']:.3f} "
             f"Precision: {scores['precision']:.3f} "
             f"Recall: {scores['recall']:.3f} ")

    plt.plot(rs, accuracies, label="accuracy")
    plt.plot(rs, f1s, label="f1")
    plt.plot(rs, precisions, label="precision")
    plt.plot(rs, recalls, label="recall")
    plt.legend()
    plt.xlabel("r")
    plt.show()

# -----------------------------
# 6) FINAL EVALUATION
# -----------------------------

r = 0.33  # From grid search, this is the best threshold for F1 score

def get_roc_pr_auc(model, x, y, r):
    scores = k_fold_loop(model, x, y, r, plot_curves=False)

    print(f"REMOVALS: {removals}")
    print(f"r: {r} | "
          f"ROC AUC: {scores['roc_auc']:.3f} | "
          f"PR-AUC: {scores['pr_auc']:.3f} | "
          f"Accuracy: {scores['accuracy']:.3f} | "
          f"F1: {scores['f1']:.3f} | "
          f"Precision: {scores['precision']:.3f} | "
          f"Recall: {scores['recall']:.3f} ")

# RUN EVERYTHING
# grid_search_r(model, x, y)
# get_roc_pr_auc(model, x, y, r)


# -----------------
# 7) PERFORMANCE EVALUATION
# -----------------

# _____LDA_____

# All features (NOT STRATIFIED)
# r: 0.24 | ROC AUC: 0.921 | PR-AUC: 0.744 | Accuracy: 0.863 | F1: 0.673 | Precision: 0.597 | Recall: 0.783

# REMOVE (NOT STRATIFIED): ['increase_stock', 'summertime', 'precip', 'dew', 'snowdepth', 'visibility']
# r: 0.25 | ROC AUC: 0.920 | PR-AUC: 0.732 | Accuracy: 0.870 | F1: 0.688 | Precision: 0.611 | Recall: 0.798 

# REMOVE (STRATIFIED):['increase_stock', 'weekday', 'visibility']
# r: 0.33 | ROC AUC: 0.918 | PR-AUC: 0.736 | Accuracy: 0.874 | F1: 0.670 | Precision: 0.634 | Recall: 0.720 

# ----------------
# 8) TEST
# ----------------

model = skl_da.LinearDiscriminantAnalysis().fit(x, y)

def test_model(model):
    bikes = pd.read_csv('strat_preprocessed_testing_data.csv')
    X_test = bikes.drop(columns=cols_to_drop)
    y_test = bikes['increase_stock']
    y_test = (y_test == 1).astype(int)
    probabilities = model.predict_proba(X_test)[:,1]
    pred = (probabilities >= r).astype(int)
    f1_test = f1_score(y_test, pred)
    accuracy_test = accuracy_score(y_test, pred)
    roc_auc_test = roc_auc_score(y_test, probabilities)
    pr_auc_test = average_precision_score(y_test, probabilities)
    pr_test = precision_score(y_test, pred)
    rec_test = recall_score(y_test, pred)
    
    scores = {
    "accuracy": accuracy_test,
    "f1": f1_test,
    "precision": pr_test,
    "recall": rec_test,
    "roc_auc": roc_auc_test,
    "pr_auc": pr_auc_test
    }

    return scores


scores = test_model(model)

print("TEST RUN")
print(f"r: {r} | "
        f"ROC AUC: {scores['roc_auc']:.3f} | "
        f"PR-AUC: {scores['pr_auc']:.3f} | "
        f"Accuracy: {scores['accuracy']:.3f} | "
        f"F1: {scores['f1']:.3f} | "
        f"Precision: {scores['precision']:.3f} | "
        f"Recall: {scores['recall']:.3f} ")


# _____LDA_____

# REMOVE (NOT STRATIFIED): ['increase_stock', 'summertime', 'precip', 'dew', 'snowdepth', 'visibility']
# r: 0.25 | ROC AUC: 0.892 | PR-AUC: 0.633 | Accuracy: 0.831 | F1: 0.585 | Precision: 0.487 | Recall: 0.731

# REMOVE (STRATIFIED): ['increase_stock', 'weekday', 'visibility']
# r: 0.33 | ROC AUC: 0.900 | PR-AUC: 0.693 | Accuracy: 0.856 | F1: 0.639 | Precision: 0.581 | Recall: 0.709 

