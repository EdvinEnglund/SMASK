import pandas as pd
import random
df_test = pd.read_csv('strat_preprocessed_testing_data.csv',)
y_test = df_test['increase_stock']
y_test = (y_test == 1).astype(int)
from sklearn.metrics import (accuracy_score,
                             f1_score,
                             roc_auc_score,
                             roc_curve,
                             precision_recall_curve,
                             average_precision_score,
                             recall_score,
                             precision_score,
                             auc)
pred = []
prob = []
for i in range(len(y_test)):
    rand = random.random()
    prob.append(rand)
    if rand < 0.1875:
        pred.append(1)
    else:
        pred.append(0)


f1 = f1_score(y_test, pred)
accuracy = accuracy_score(y_test, pred)
roc_auc = roc_auc_score(y_test, prob)
pr_auc = average_precision_score(y_test, prob)
pr = precision_score(y_test, pred)
rec = recall_score(y_test, pred)

print(f"FINAL TEST: "
      f"ROC AUC: {roc_auc:.3f} | "
      f"PR-AUC: {pr_auc:.3f} | "
      f"Accurancy: {accuracy:.3f} | "
      f"F1 score: {f1:.3f} | "
      f"Precision: {pr:.3f} | "
      f"Recall: {rec:.3f} ")