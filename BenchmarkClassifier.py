import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    recall_score,
    precision_score,
)
from matplotlib import pyplot as plt

# Data
bikes = pd.read_csv('strat_preprocessed_testing_data.csv')
y = bikes['increase_stock']

# Convert negative class labels in validation data set from "-1" to "0"
# This ensures that scikit-learn metrics methods work properly
y_val = (y == 1).astype(int) 


# Generate random positive class probabilities
def get_benchmark_probas(y):
    rng = np.random.default_rng(seed=42)
    probas = rng.random((len(y), 1)) # Vector with dimensions: (no of data points x 1)
    bench_proba = np.round(probas, decimals=3) # round to three decimals
    return bench_proba


# Calculate prediction errors for a given decision threshold
def evaluate_model(model_proba, y_val, r):
    if (len(model_proba) != len(y)):
        print(f"Error, unexpected model input, expected {len(y_val)} values, received {len(model_proba)}")
        return
    
    model_pred = np.zeros(len(y_val))
    for i in range(len(model_proba)):
        if model_proba[i] > r:
            model_pred[i] = 1
        else:
            model_pred[i] = 0   
  
    eval_scores = {'acc': round(accuracy_score(y_val, model_pred), 3), 
                   'prec': round(precision_score(y_val, model_pred), 3), 
                   'recall': round(recall_score(y_val, model_pred), 3), 
                   'f1': round(f1_score(y_val, model_pred), 3)
                   }

    return eval_scores


# Get ROC-AUC and PR-AUC scores
def get_aucs(model_proba, y_val):
    roc_auc = roc_auc_score(y_val, model_proba)
    pr_auc = average_precision_score(y_val, model_proba)
    aucs = {'roc_auc': round(roc_auc, 3), 'pr_auc': round(pr_auc, 3)}
    
    return aucs


######## RUN SCRIPT ########## 

threshold = 0.8125 # Set decision threshold to reflect problem class imbalance

benchmark_probas = get_benchmark_probas(y)
eval_scores = evaluate_model(benchmark_probas, y_val, threshold)
auc_scores = get_aucs(benchmark_probas, y_val)

roc_auc = auc_scores['roc_auc']
pr_auc = auc_scores['pr_auc']
f1 = eval_scores['f1']
prec = eval_scores['prec']
recall = eval_scores['recall']
acc = eval_scores['acc']

print(f"BENCHMARK MODEL SCORES: "
      f"ROC AUC: {roc_auc} | "
      f"PR-AUC: {pr_auc} | "
      f"Accuracy: {acc} | "
      f"F1 score: {f1} | "
      f"Precision: {prec} | "
      f"Recall: {recall} ")
