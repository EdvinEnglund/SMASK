import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('training_data_VT2026.csv')

mapping = {"low_bike_demand": -1, "high_bike_demand": 1}
df["increase_stock"] = df["increase_stock"].map(mapping)

df = df.drop(["snow"], axis=1)

df["snowdepth"] = df["snowdepth"].apply(lambda x: 1 if x > 0 else 0)

cat_cols = ["month", "day_of_week", "hour_of_day"]
df = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols, dtype=int)

# SPLIT FIRST
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["increase_stock"])

# SCALE USING TRAIN STATISTICS ONLY
scaling_columns = ["dew", "temp", "humidity", "precip", "windspeed", "cloudcover", "visibility"]

for column in scaling_columns:
    min_val = train_df[column].min()
    max_val = train_df[column].max()

    train_df[column] = (train_df[column] - min_val) / (max_val - min_val)
    test_df[column] = (test_df[column] - min_val) / (max_val - min_val)

train_df.to_csv('strat_preprocessed_training_data.csv', index=False)
test_df.to_csv('strat_preprocessed_testing_data.csv', index=False)
