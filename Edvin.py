import pandas as pd
from sklearn.model_selection import train_test_split

pd.options.display.max_columns = None
pd.options.display.max_rows = None
df = pd.read_csv('training_data_VT2026.csv')

#rebrand output labels
mapping = {"low_bike_demand": -1, "high_bike_demand": 1}
df["increase_stock"] = df["increase_stock"].map(mapping)

#drop unnecessary features
df = df.drop(["snow", "visibility", "holiday", "weekday"], axis = 1)

#make snowdepth categorical:
df["snowdepth"] = df["snowdepth"].apply(lambda x: 1 if x > 0 else 0)

#one-hot encoding dummy variables
cat_cols = ["month", "day_of_week", "hour_of_day"]
df = pd.get_dummies(
    df,
    columns=cat_cols,
    prefix=cat_cols,
    dtype=int
)

#min-max scale numerical features
def min_max_scale(df, columns):
    for column in columns:
        min_val = df[column].min()
        max_val = df[column].max()
        df[column] = (df[column] - min_val) / (max_val - min_val)

scaling_columns = ["dew", "temp", "humidity", "precip", "windspeed", "cloudcover"]
min_max_scale(df, scaling_columns)

train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)


train_df.to_csv('preprocessed_training_data.csv', index = False)
test_df.to_csv('preprocessed_testing_data.csv', index = False)