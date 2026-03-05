# Script for predicting demand based on test data

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Preprocessing based on entire training data
df = pd.read_csv('data/training_data_VT2026.csv')
test_df = pd.read_csv('data/test_data_VT2026.csv')

mapping = {"low_bike_demand": 0, "high_bike_demand": 1}
df["increase_stock"] = df["increase_stock"].map(mapping)

df = df.drop(["snow"], axis=1)
test_df = test_df.drop(["snow"], axis=1)

df["snowdepth"] = df["snowdepth"].apply(lambda x: 1 if x > 0 else 0)
test_df["snowdepth"] = test_df["snowdepth"].apply(lambda x: 1 if x > 0 else 0)

cat_cols = ["month", "day_of_week", "hour_of_day"]
df = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols, dtype=int)
test_df = pd.get_dummies(test_df, columns=cat_cols, prefix=cat_cols, dtype=int)

# Scale ENTIRE data this time
scaling_columns = ["dew", "temp", "humidity", "precip", "windspeed", "cloudcover", "visibility"]

# Scaling based on the training data
for column in scaling_columns:
    min_val = df[column].min()
    max_val = df[column].max()
    df[column] = (df[column] - min_val) / (max_val - min_val)
    test_df[column] = (test_df[column] - min_val) / (max_val - min_val)

# Group dummy variables for easier feature selection
month = [c for c in df.columns if c.startswith("month")]
day_of_week = [c for c in df.columns if c.startswith("day_of_week")]
hour_of_day = [c for c in df.columns if c.startswith("hour_of_day")]

# Remove features based on feature selection in EdvinRForest
# Increase stock is removed seperately since it is missing from test data
removals = ([
              "windspeed",
              "precip",
              "dew",
              "cloudcover",
              "holiday",
              "weekday"]
            + month
            )

# Define the training data
x_train = df.drop(columns = removals)
x_train = x_train.drop(columns = "increase_stock")
y_train = df["increase_stock"]

# Define the test data
x_test = test_df.drop(columns = removals)

# Define the model, hyperparameters selected from EdvinRForest
model = RandomForestClassifier(
    criterion="entropy",
    min_samples_split=2,
    min_samples_leaf=1,
    n_estimators=100,
    max_features="sqrt",
    random_state=42
)

# Fit and predict
model.fit(x_train, y_train)
predictions = model.predict_proba(x_test)[:, 1]
r = 0.37 # r = 0.37 was found in grid search in EdvinRForest
predictions = (predictions >= r).astype(int)
predictions = pd.DataFrame([predictions])
predictions.to_csv('predictions.csv', index=False, header=False)



