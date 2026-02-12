import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score
)

# -----------------------------
# 1) READ DATA
# -----------------------------
train_df = pd.read_csv("full_preprocessed_training_data.csv")
test_df  = pd.read_csv("full_preprocessed_testing_data.csv")

y_train = (train_df["increase_stock"] == 1).astype(int)
y_test  = (test_df["increase_stock"] == 1).astype(int)

# -----------------------------
# 2) ONLY GOOD FEATURE SCENARIOS
# -----------------------------
scenarios = {
    "baseline": [],
    "no_temp": ["temp"],
    "no_temp_humidity": ["temp", "humidity"],
    "no_temp_humidity_snow": ["temp", "humidity", "snowdepth"],
}

# -----------------------------
# 3) CV + GRID SETTINGS
# -----------------------------
kf = KFold(n_splits=10, shuffle=True, random_state=42)

k_values = range(3, 31, 2)
r_values = np.linspace(0.1, 0.9, 25)

global_best = {"scenario": None, "k": None, "r": None, "f1": -1}

# -----------------------------
# 4) GRID SEARCH ON TRAIN ONLY
# -----------------------------
for scenario, cols_to_remove in scenarios.items():

    X = train_df.drop(columns=["increase_stock"] + cols_to_remove, errors="ignore")

    for k in k_values:
        for r in r_values:

            f1_scores = []

            for train_idx, val_idx in kf.split(X):

                X_train_fold = X.iloc[train_idx]
                X_val_fold   = X.iloc[val_idx]
                y_train_fold = y_train.iloc[train_idx]
                y_val_fold   = y_train.iloc[val_idx]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_fold)
                X_val_scaled   = scaler.transform(X_val_fold)

                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train_scaled, y_train_fold)

                y_prob = knn.predict_proba(X_val_scaled)[:, 1]
                y_pred = (y_prob >= r).astype(int)  

                f1_scores.append(f1_score(y_val_fold, y_pred, zero_division=0))

            mean_f1 = np.mean(f1_scores)

            if mean_f1 > global_best["f1"]:
                global_best.update({
                    "scenario": scenario,
                    "k": k,
                    "r": r,
                    "f1": mean_f1
                })

print("\nBEST MODEL FROM CV")
print(global_best)

# -----------------------------
# 5) FINAL TRAIN ON FULL TRAIN
# -----------------------------
cols_to_remove = scenarios[global_best["scenario"]]

X_train_full = train_df.drop(columns=["increase_stock"] + cols_to_remove, errors="ignore")
X_test_final = test_df.drop(columns=["increase_stock"] + cols_to_remove, errors="ignore")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled  = scaler.transform(X_test_final)

knn = KNeighborsClassifier(n_neighbors=global_best["k"])
knn.fit(X_train_scaled, y_train)

# -----------------------------
# 6) REAL FINAL TEST
# -----------------------------
y_prob = knn.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_prob >= global_best["r"]).astype(int)

print("\nFINAL TEST PERFORMANCE")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
print(f"F1 score : {f1_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall   : {recall_score(y_test, y_pred):.3f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.3f}")
