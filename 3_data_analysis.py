import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.max_columns = None
pd.options.display.max_rows = None
df = pd.read_csv('preprocessed_training_data.csv')

positives = 0
negatives = 0

for d in df["increase_stock"]:
    if d > 0:
        positives += 1
    else:
        negatives += 1

print(f"positives: {positives}")
print(f"negatives: {negatives}")
print(f"ratio: {positives/(positives + negatives)}")


#mapping = {"low_bike_demand": -1, "high_bike_demand": 1}

#df["increase_stock"] = df["increase_stock"].map(mapping)

"""
corr = df.corr()

plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()
"""


