import pandas as pd
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, average_precision_score
)

from sulletrain import knn_model, r, removals

# test data 
test_data= "strat_preprocessed_testing_data.csv"

# load test data
df = pd.read_csv(test_data)

# drop same features as training (except target)
df = df.drop(columns=[c for c in removals if c != "increase_stock"], errors="ignore")

y_test = (df["increase_stock"] == 1).astype(int)
X_test = df.drop(columns=["increase_stock"])

# predictions
y_prob = knn_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= r).astype(int)

# scores
print("Results on test data:")

print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
print(f"F1-score : {f1_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}")
print(f"Recall   : {recall_score(y_test, y_pred):.3f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.3f}")
print(f"PR-AUC   : {average_precision_score(y_test, y_prob):.3f}")