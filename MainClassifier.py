"""
This script includes the full splitting, preprocessing, training and validation pipeline
for the final RandomForestClassifier after it was chosen as the final model.
The hyperparameter tuning and kFold-tuning-validation for the models are viewed in
Random Forest: EdvinRForest.py
KNN: knn/
LDA: Emil.py
ADABoost: Joakim.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (accuracy_score,
                             f1_score,
                             roc_auc_score,
                             average_precision_score,
                             recall_score,
                             precision_score
                             )
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

"""
====================================
******** COMMON FUNCTIONS **********
====================================
"""

# k-fold
def k_fold_loop(model, x, y, r = 0.5, n_splits = 10):
    #set evaluation metrics
    accuracy = 0
    f1 = 0
    roc_auc = 0
    pr_auc = 0
    pr = 0
    rec = 0

    # KFold Loop
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for i, (train_index, test_index) in enumerate(kf.split(x, y)):

        x_train_kf, x_test_kf = x.iloc[train_index], x.iloc[test_index]
        y_train_kf, y_test_kf = y.iloc[train_index], y.iloc[test_index]
        model.fit(x_train_kf, y_train_kf)

        # predict probabilities
        prob = model.predict_proba(x_test_kf)[:, 1]
        pred = (prob >= r).astype(int)

        # get scores
        f1_s = f1_score(y_test_kf, pred)
        acc_s = accuracy_score(y_test_kf, pred)
        roc_auc_s = roc_auc_score(y_test_kf, prob)
        pr_auc_s = average_precision_score(y_test_kf, prob)
        pr_s = precision_score(y_test_kf, pred)
        rec_s = recall_score(y_test_kf, pred)

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

# Example function for hyper paramater tuning. Can easily be modified for hyperparamater other than r
# For other hyperparamaters for Random Forest, manual/random search was used.
# Other Models used slightly different approaches and of course other paramateres.
# Please see the repo for further details.

def grid_search_r(model, x, y, start = 0.15, stop = 0.64, num = 50):
    rs = np.linspace(start, stop, num)
    for r in rs:
        scores = k_fold_loop(model, x, y, r)
        print(f"Accurancy: {scores['accuracy']:.3f} "
              f"F1 score: {scores['f1']:.3f} "
              f"Precision: {scores['precision']:.3f} "
              f"Recall: {scores['recall']:.3f} ")

"""
====================================
***** COMMON DATA EXTRACTION, ******
*** PREPROCESSING AND SPLITTING  ***
====================================
"""

# Read the dataset
df = pd.read_csv('data/training_data_VT2026.csv')

mapping = {"low_bike_demand": 0, "high_bike_demand": 1}
df["increase_stock"] = df["increase_stock"].map(mapping)

# Drop snow feature due to missing data
df = df.drop(["snow"], axis=1)

# Turn snow depth into a categorical feature
df["snowdepth"] = df["snowdepth"].apply(lambda x: 1 if x > 0 else 0)

# Create dummy variables for categorical features
cat_cols = ["month", "day_of_week", "hour_of_day"]
df = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols, dtype=int)

# Test / Train split (30/70), straitified on output feature
train_df, test_df = train_test_split(df,
                                     test_size=0.3,
                                     random_state=42,
                                     stratify=df["increase_stock"])

# Scale all numerical features [0,1]
scaling_columns = ["dew", "temp", "humidity", "precip", "windspeed", "cloudcover", "visibility"]
for column in scaling_columns:
    min_val = train_df[column].min()
    max_val = train_df[column].max()

    # Scaling based on training data only for model to be representative in testing
    train_df[column] = (train_df[column] - min_val) / (max_val - min_val)
    test_df[column] = (test_df[column] - min_val) / (max_val - min_val)

# Group dummy variables for easier feature selection
month = [c for c in train_df.columns if c.startswith("month")]
day_of_week = [c for c in train_df.columns if c.startswith("day_of_week")]
hour_of_day = [c for c in train_df.columns if c.startswith("hour_of_day")]

# Define output
y_train = train_df['increase_stock']
y_test = test_df['increase_stock']

"""
====================================
MODELS AND THEIR RESPECTIVE FEATURES
FROM INDIVIDUAL FEATURE SELECTIONS
====================================
"""
#############################
# RANDOM FOREST CLASSIFIER: #
#############################

# Decide features to drop.
# uncommented features produced best f1 scores, backward selection was used for RF
removals_RF = (["increase_stock",
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

# Drop features
x_train_RF = train_df.drop(columns=removals_RF, inplace=False)
x_test_RF = test_df.drop(columns=removals_RF, inplace=False)

# Define model
RF = RandomForestClassifier(
    criterion="entropy",
    min_samples_split=2,
    min_samples_leaf=1,
    n_estimators=100,
    max_features="sqrt",
    random_state=42
)

# EXAMPLE HYPER PARAMETER TUNING FOR RANDOM FORREST
# THIS IS UNCOMMENTED SINCE IT PRODUCES LOTS OF PRINT OUTS
#grid_search_r(RF, x_train_RF, x_test_RF)

# r = 0.37 maximized f-score in grid search for Random Forest

# RUN K_FOLD VALIDATION LOOP:

scores_RF = k_fold_loop(RF, x_train_RF, y_train, r=0.37, n_splits=10)
print("K_FOLD VALIDATION SCORES FOR Random Forest:")
print(f"Accurancy: {scores_RF['accuracy']:.3f} "
      f"F1 score: {scores_RF['f1']:.3f} "
      f"Precision: {scores_RF['precision']:.3f} "
      f"Recall: {scores_RF['recall']:.3f} ")

###################
# kNN CLASSIFIER: #
###################

removals_KNN = (
    ["increase_stock"]
    + month
)

x_train_KNN = train_df.drop(columns=removals_KNN, inplace=False)
x_test_KNN = test_df.drop(columns=removals_KNN, inplace=False)

KNN = KNeighborsClassifier(n_neighbors=3)

scores_KNN = k_fold_loop(KNN, x_train_KNN, y_train, r=0.37, n_splits=10)
print("K_FOLD VALIDATION SCORES FOR KNN:")
print(f"Accurancy: {scores_KNN['accuracy']:.3f} "
      f"F1 score: {scores_KNN['f1']:.3f} "
      f"Precision: {scores_KNN['precision']:.3f} "
      f"Recall: {scores_KNN['recall']:.3f} ")

################################
# LINEAR DISCRIMINANT ANALYSIS #
################################

removals_LDA = ["increase_stock",
              "weekday",
              "visibility"
              ]

x_train_LDA = train_df.drop(columns=removals_LDA, inplace=False)
x_test_LDA = test_df.drop(columns=removals_LDA, inplace=False)
LDA = LinearDiscriminantAnalysis()

scores_LDA = k_fold_loop(LDA, x_train_LDA, y_train, r=0.37, n_splits=10)
print("K_FOLD VALIDATION SCORES FOR LDA:")
print(f"Accurancy: {scores_LDA['accuracy']:.3f} "
      f"F1 score: {scores_LDA['f1']:.3f} "
      f"Precision: {scores_LDA['precision']:.3f} "
      f"Recall: {scores_LDA['recall']:.3f} ")

########################
# ADA BOOST CLASSIFIER #
########################

removals_ADA = [
    "increase_stock",
    "windspeed",
    "precip",
    "dew",
    "cloudcover",
    "snowdepth"
]

x_train_ADA = train_df.drop(columns=removals_ADA, inplace=False)
x_test_ADA = test_df.drop(columns=removals_ADA, inplace=False)

base_learner = DecisionTreeClassifier(
    max_depth=2,
    random_state=42
)

ADA = AdaBoostClassifier(
    estimator=base_learner,
    n_estimators=300,
    learning_rate=0.05,
    random_state=42
)

scores_ADA = k_fold_loop(ADA, x_train_ADA, y_train, r=0.37, n_splits=10)
print("K_FOLD VALIDATION SCORES FOR ADA:")
print(f"Accurancy: {scores_ADA['accuracy']:.3f} "
      f"F1 score: {scores_ADA['f1']:.3f} "
      f"Precision: {scores_ADA['precision']:.3f} "
      f"Recall: {scores_ADA['recall']:.3f} ")


# --- FINAL TEST COMPARING THE MODELS ---

models = [
    (RF, x_train_RF, x_test_RF, 0.37),
    (KNN, x_train_KNN, x_test_KNN, 0.303),
    (LDA, x_train_LDA, x_test_LDA, 0.67),
    (ADA, x_train_ADA, x_test_ADA, 0.423)

]

for (model, x_train_m, x_test_m, model_r) in models:
    # Fit Model on ENTIRE 70% Split.
    model.fit(x_train_m, y_train)

    # Get predicted class probability and prediction on test data
    prob = model.predict_proba(x_test_m)[:, 1]
    pred = (prob >= model_r).astype(int)

    # Calculate and show metrics
    print(f"FINAL TEST SCORES FOR {model.__class__.__name__}:")
    print(f"r: {model_r} | "
          f"ROC AUC: {roc_auc_score(y_test, prob):.3f} | "
          f"PR-AUC: {average_precision_score(y_test, prob):.3f} | "
          f"Accurancy: {accuracy_score(y_test, pred):.3f} | "
          f"F1 score: {f1_score(y_test, pred):.3f} | "
          f"Precision: {precision_score(y_test, pred):.3f} | "
          f"Recall: {recall_score(y_test, pred):.3f} ")
