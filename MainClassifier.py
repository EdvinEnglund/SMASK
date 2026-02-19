import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import (accuracy_score,
                             f1_score,
                             roc_auc_score,
                             average_precision_score,
                             recall_score,
                             precision_score
                             )
from sklearn.model_selection import train_test_split

df = pd.read_csv('training_data_VT2026.csv')

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

# Decide features to drop.
# (uncommented features produced best f1 scores)
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

# Drop features
x_train = train_df.drop(columns=removals)
x_test = test_df.drop(columns=removals)

# Define output
y_train = train_df['increase_stock']
y_test = test_df['increase_stock']

# Define model
model = RandomForestClassifier(
    criterion="entropy",
    min_samples_split=2,
    min_samples_leaf=1,
    n_estimators=100,
    max_features="sqrt",
    random_state=42
)

# --- TEST ---
# r = 0.37 maximized f-score in grid search
r = 0.37

# Fit model to training data
model.fit(x_train, y_train)

# Get predicted class probability and prediction on test data
prob = model.predict_proba(x_test)[:, 1]
pred = (prob >= r).astype(int)

# Calculate metrics
f1 = f1_score(y_test, pred)
accuracy = accuracy_score(y_test, pred)
roc_auc = roc_auc_score(y_test, prob)
pr_auc = average_precision_score(y_test, prob)
pr = precision_score(y_test, pred)
rec = recall_score(y_test, pred)

print(f"FINAL TEST: "
      f"r: {r} | "
      f"ROC AUC: {roc_auc:.3f} | "
      f"PR-AUC: {pr_auc:.3f} | "
      f"Accurancy: {accuracy:.3f} | "
      f"F1 score: {f1:.3f} | "
      f"Precision: {pr:.3f} | "
      f"Recall: {rec:.3f} ")
