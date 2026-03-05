"""
This script includes training, hyperparameter tuning & feature selection (by manual editing)
and cross validation for the Random Forest classifier.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score,
                             f1_score,
                             roc_auc_score,
                             roc_curve,
                             precision_recall_curve,
                             average_precision_score,
                             recall_score,
                             precision_score,
                             auc)
from matplotlib import pyplot as plt
pd.options.display.max_columns = None
pd.options.display.max_rows = None

#read data
df = pd.read_csv('data/preprocessed_training_data.csv', )
df_test = pd.read_csv('data/preprocessed_testing_data.csv', )
#define potential removals
month = [c for c in df.columns if c.startswith("month")]
day_of_week = [c for c in df.columns if c.startswith("day_of_week")]
hour_of_day = [c for c in df.columns if c.startswith("hour_of_day")]

removals = (["increase_stock",
              "windspeed",
              #"summertime",
              "precip",
              #"humidity",
              "dew",
              #"temp",
              "cloudcover",
              #"snowdepth",
              "holiday",
              "weekday",
              #"visibility"
              ]
            + month
            #+ day_of_week
            #+ hour_of_day
            )

cols_to_drop = removals
#filter inputs
x = df.drop(columns=cols_to_drop)
#define output
y = df['increase_stock']
#convert -1 to 0
y = (y == 1).astype(int)

#define model
model = RandomForestClassifier(
    criterion="entropy",
    min_samples_split=2,
    min_samples_leaf=1,
    n_estimators=100,
    max_features="sqrt",
    random_state=42
)

def k_fold_loop(model, x, y, r = 0.5, n_splits = 10, plot_curves = False):
    #set evaluation metrics
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
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for i, (train_index, test_index) in enumerate(kf.split(x, y)):

        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(x_train, y_train)

        # predict probabilities
        prob = model.predict_proba(x_test)[:, 1]
        pred = (prob >= r).astype(int)

        # get scores
        f1_s = f1_score(y_test, pred)
        acc_s = accuracy_score(y_test, pred)
        roc_auc_s = roc_auc_score(y_test, prob)
        pr_auc_s = average_precision_score(y_test, prob)
        pr_s = precision_score(y_test, pred)
        rec_s = recall_score(y_test, pred)

        # accumulate scores
        f1 += f1_s
        accuracy += acc_s
        roc_auc += roc_auc_s
        pr_auc += pr_auc_s
        pr += pr_s
        rec += rec_s

    scores = {
        "accuracy": accuracy / n_splits,
        "f1": f1 / n_splits,
        "precision": pr / n_splits,
        "recall": rec / n_splits,
        "roc_auc": roc_auc / n_splits,
        "pr_auc": pr_auc / n_splits
    }

    return scores

def grid_search_r(model, x, y, start = 0.15, stop = 0.64, num = 50):
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
        print(f"Accurancy: {scores['accuracy']:.3f} "
              f"F1 score: {scores['f1']:.3f} "
              f"Precision: {scores['precision']:.3f} "
              f"Recall: {scores['recall']:.3f} ")

    plt.plot(rs, accuracies, label="accuracy")
    plt.plot(rs, f1s, label="f1")
    plt.plot(rs, precisions, label="precision")
    plt.plot(rs, recalls, label="recall")
    plt.legend()
    plt.xlabel("r")
    plt.show()

r = 0.37 # r-value found in grid-search
def get_roc_pr_auc(model, x, y, r):
    scores = k_fold_loop(model, x, y, r, plot_curves=True)
    print(f"K-fold CV scores:"
          f"r: {r} | "
          f"ROC AUC: {scores['roc_auc']:.3f} | "
          f"PR-AUC: {scores['pr_auc']:.3f} | "
          f"Accurancy: {scores['accuracy']:.3f} | "
          f"F1 score: {scores['f1']:.3f} | "
          f"Precision: {scores['precision']:.3f} | "
          f"Recall: {scores['recall']:.3f} ")

#grid_search_r(model, x, y, num = 25)
get_roc_pr_auc(model, x, y, r)

