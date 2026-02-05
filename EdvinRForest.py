import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (accuracy_score,
                             f1_score,
                             roc_auc_score,
                             roc_curve,
                             precision_recall_curve,
                             average_precision_score,
                             recall_score,
                             precision_score)
from matplotlib import pyplot as plt
pd.options.display.max_columns = None
pd.options.display.max_rows = None

#read data
df = pd.read_csv('preprocessed_training_data.csv',)
#define potential removals
month = [c for c in df.columns if c.startswith("month")]
day_of_week = [c for c in df.columns if c.startswith("day_of_week")]
hour_of_day = [c for c in df.columns if c.startswith("hour_of_day")]

removals = ["increase_stock",
              "windspeed",
              #"summertime",
              #"precip",
              #"humidity",
              "dew",
              #"temp",
              "cloudcover",
              #"snowdepth",
              "holiday",
              "weekday",
              "visibility"
              ]

cols_to_drop = removals + month
#filter inputs
x = df.drop(columns=cols_to_drop)
#define output
y = df['increase_stock']
#convert -1 to 0
y = (y == 1).astype(int)

#define and fit model
model = RandomForestClassifier(
    criterion="entropy",
    min_samples_leaf=1,
    n_estimators=500
)

#split data
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#create K fold indicies
n_splits = 10
kf = KFold(n_splits=10, shuffle=True, random_state=42)
#adjust probability threshhold
r = 0.35

#set evaluation metrics
accuracy = 0
f1 = 0
roc_auc = 0
pr_auc =0
precision = 0
recall = 0

for train_index, test_index in kf.split(x):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(x_train, y_train)

    # predict probabilities
    prob = model.predict_proba(x_test)[:, 1]
    pred = (prob >= r).astype(int)

    # accumulate scores
    f1 += f1_score(y_test, pred)
    accuracy += accuracy_score(y_test, pred)
    roc_auc += roc_auc_score(y_test, pred)
    pr_auc += average_precision_score(y_test, pred)
    precision += precision_score(y_test, pred)
    recall += recall_score(y_test, pred)

print(f"Accuracy (r={r}): {accuracy / n_splits}")
print(f"F1 (r={r}):  {f1 / n_splits}")
print(f"Precision (r={r}):  {precision / n_splits}")
print(f"Recall (r={r}): {recall / n_splits}")
print(f"ROC AUC: ", roc_auc / n_splits)
print(f"PR AUC: ", pr_auc / n_splits)


#plot au-roc
#fpr, tpr, _ = roc_curve(y_test, prob, pos_label = 1)
#plt.plot(fpr, tpr)

"""
#create confusion matrix
cf = pd.crosstab(pred, y_test, rownames=['Actual'], colnames=['Predicted'])
print(f"missclassification rate: {1 - accuracy_score(y_test, pred)}")
print(f"f1 score = {f1_score(y_test, pred)}")
print(f"roc auc = {roc_auc_score(y_test, pred)}")
print(f"pr-auc = {average_precision_score(y_test, pred)}")
print(f"precision = {precision_score(y_test, pred)}")
print(f"recall = {recall_score(y_test, pred)}")
"""

plt.show()
