import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.model_selection as skl_ms
import sklearn.discriminant_analysis as skl_da


# Data
bikes = pd.read_csv('preprocessed_training_data.csv')
X = bikes.drop(columns='increase_stock')
y = bikes['increase_stock']


# Define model
def create_discriminant_model(model_type):
  if model_type == 'LDA':
    model = skl_da.LinearDiscriminantAnalysis()
  elif model_type == 'QDA':
    model = skl_da.QuadraticDiscriminantAnalysis(reg_param=0.1)
  else:
    model = skl_da.LinearDiscriminantAnalysis()
    print("ALERT: No discriminant model specified, defaulting to LDA")  

  return model


# Cross-validation
def kfold_cv(model, X, y, n_fold=10):
  cv = skl_ms.StratifiedKFold(n_splits=n_fold, random_state=42, shuffle=True)
  thresholds = np.arange(0.01, 0.99, 0.01)
  precisions = np.zeros(len(thresholds))
  recalls = np.zeros(len(thresholds))
  
  for train_index, val_index in cv.split(X, y):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    model.fit(X_train, y_train)
    P = np.sum(y_val == 1)

    for j, r in enumerate(thresholds):
      predict_prob = model.predict_proba(X_val)
      predictions = np.where(predict_prob[:,1] >= r, 1, -1)
      TP = np.sum((predictions == 1) & (y_val == 1))
      P_star = np.sum(predictions == 1)
      precision = TP/P_star
      recall = TP/P
      precisions[j] += precision
      recalls[j] += recall

  precisions /= n_fold
  recalls /= n_fold

  return precisions, recalls

# Plot precision-recall curve
def plot_PR_curve(x, y):
  plt.plot(x, y)
  plt.title("Precision-recall curve")
  plt.xlabel("Recall")
  plt.ylabel("Precision")
  plt.show()

model = create_discriminant_model('QDA')
prec, rec = kfold_cv(model, X, y)
plot_PR_curve(prec, rec)
