import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score,
    recall_score, roc_auc_score, average_precision_score
)

# Laod traindata
df = pd.read_csv("data/preprocessed_training_data.csv")

# train / test split
train_df, test_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    stratify=df["increase_stock"]
)


month = [c for c in train_df.columns if c.startswith("month")]
day_of_week = [c for c in train_df.columns if c.startswith("day_of_week")]
hour_of_day = [c for c in train_df.columns if c.startswith("hour_of_day")]


# feature removals
removals = (
    ["increase_stock"]
    + month
    #+ hour_of_day
    #+ day_of_week
)

# Drop features
x_train = train_df.drop(columns=removals)
x_test = test_df.drop(columns=removals)

# Output
y_train = (train_df["increase_stock"] == 1).astype(int)
y_test = (test_df["increase_stock"] == 1).astype(int)


# kNN-MODEL
k = 3 #from hyperparameter tuning
knn_model = KNeighborsClassifier(n_neighbors=k)

# best threshold 
r = 0.305

# Train
knn_model.fit(x_train, y_train)

# Predictions
prob = knn_model.predict_proba(x_test)[:, 1]
pred = (prob >= r).astype(int)

#Scores
f1 = f1_score(y_test, pred)
accuracy = accuracy_score(y_test, pred)
roc_auc = roc_auc_score(y_test, prob)
pr_auc = average_precision_score(y_test, prob)
pr = precision_score(y_test, pred)
rec = recall_score(y_test, pred)

print(
    f"FINAL TEST: "
    f"r: {r} | "
    f"ROC AUC: {roc_auc:.3f} | "
    f"PR-AUC: {pr_auc:.3f} | "
    f"Accuracy: {accuracy:.3f} | "
    f"F1 score: {f1:.3f} | "
    f"Precision: {pr:.3f} | "
    f"Recall: {rec:.3f}"
)
