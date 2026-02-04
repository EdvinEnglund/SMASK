import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('training_data_VT2026.csv')

mapping = {"low_bike_demand": -1, "high_bike_demand": 1}
df["snowdepth"] = df["snowdepth"].apply(lambda x: 0 if x == 0 else 1)

mapping = {"low_bike_demand": -1, "high_bike_demand": 1}

df["increase_stock"] = df["increase_stock"].map(mapping)

cleaned_df = df.drop(columns=["snow", "visibility", "holiday", "weekday"])

cleaned_df.to_csv('cleaned_training_data_VT2026.csv', index=False)