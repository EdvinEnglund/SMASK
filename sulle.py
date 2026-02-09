import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
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

df = pd.read_csv("preprocessed_training_data.csv")
#Separera data
X = df.drop("increase_stock", axis=1)
y = (df["increase_stock"] == 1).astype(int)

#cross validation step
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_k = 13 #hade testat olika k-värden och 13 gav bäst resultat på valideringsdata.

#Lagring för fold-metrics
accuracies = []
f1s = []
precisions_scores = []
recalls = []
roc_aucs = []
pr_aucs = []

mean_fpr = np.linspace(0, 1, 100) #linspace används för att skapa en jämnt fördelad array av 100 punkter mellan 0 och 1, vilket används för att interpolera TPR-värdena på ROC-kurvan. Detta gör att vi kan plotta en smidig genomsnittlig ROC-kurva över alla foldar. Samma koncept gäller för mean_recall i PR-kurvan.
mean_recall = np.linspace(0, 1, 100)

tprs = [] #tprs är en lista som kommer att lagra de interpolerade TPR-värdena för varje fold i ROC-kurvan. Genom att interpolera TPR-värdena på en gemensam uppsättning FPR-punkter (mean_fpr) kan vi sedan beräkna en genomsnittlig ROC-kurva över alla foldar.
precisions_curve = []


#runnar kNN med 5-fold och beräknar alla metrics i varje fold.
for train_index, test_index in kf.split(X):

    # Split data using row positions (.iloc)#.iloc används för att indexera rader i DataFrame baserat på deras position (index) snarare än etiketter. I detta fall används train_index och test_index, som genereras av KFold, för att dela upp X och y i tränings- och testuppsättningar för varje fold. Detta säkerställer att vi korrekt delar data utan att blanda ihop raderna, vilket är viktigt för att undvika data leakage och få rätt resultat i varje fold.
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Scale features inside each fold (avoid data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train kNN with chosen best_k
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = knn.predict(X_test_scaled)
    y_prob = knn.predict_proba(X_test_scaled)[:, 1]

    # basic metrics
    accuracies.append(accuracy_score(y_test, y_pred))
    f1s.append(f1_score(y_test, y_pred))
    precisions_scores.append(precision_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))

    # roc_auc
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0

    tprs.append(interp_tpr)
    roc_aucs.append(roc_auc)

    #precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])

    precisions_curve.append(interp_precision)
    pr_aucs.append(pr_auc)

#plottar ROC-kurvan
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0

plt.figure()
plt.plot(mean_fpr, mean_tpr, label=f"ROC (AUC = {np.mean(roc_aucs):.3f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("kNN ROC Curve (5-fold CV)")
plt.legend()
plt.show()


#plottar PR-kurvan
mean_precision_curve = np.mean(precisions_curve, axis=0)

plt.figure()
plt.plot(mean_recall, mean_precision_curve, label=f"PR (AP = {np.mean(pr_aucs):.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("kNN Precision-Recall Curve (5-fold CV)")
plt.legend()
plt.show()


#printar alla scores
print("\nFINAL kNN PERFORMANCE (5-fold CV)")
print(f"Accuracy : {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
print(f"F1 score : {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
print(f"Precision: {np.mean(precisions_scores):.3f} ± {np.std(precisions_scores):.3f}")
print(f"Recall   : {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
print(f"ROC-AUC  : {np.mean(roc_aucs):.3f} ± {np.std(roc_aucs):.3f}")
print(f"PR-AUC   : {np.mean(pr_aucs):.3f} ± {np.std(pr_aucs):.3f}")
