import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Load data
df = pd.read_csv("data/training_data_VT2026.csv")
df["increase_stock"] = df["increase_stock"].astype("category")

features = [
    "hour_of_day","day_of_week","month","holiday","weekday",
    "summertime","temp","dew","humidity","precip","snow",
    "snowdepth","windspeed","cloudcover","visibility"
]

categorical_features = [
    "hour_of_day","day_of_week","month",
    "holiday","weekday","summertime","snow"
]

n_features = len(features)
n_cols = 4
n_rows = math.ceil(n_features / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
axes = axes.flatten()

for i, feature in enumerate(features):
    ax = axes[i]

    if feature in categorical_features:
        # Use original integer categories
        temp_df = df.copy()
        temp_df["bin"] = temp_df[feature]

    else:
        # Round to nearest integer
        temp_df = df.copy()
        temp_df["bin"] = temp_df[feature].round().astype(int)

    grouped = (
        temp_df.groupby(["bin", "increase_stock"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    # Convert to fractions per bin
    fractions = grouped.div(grouped.sum(axis=1), axis=0)
    x_labels = fractions.index.astype(str)
    bottom = np.zeros(len(fractions))

    for category in fractions.columns:
        ax.bar(x_labels, fractions[category], bottom=bottom)
        bottom += fractions[category].values

    ax.set_title(feature)
    ax.set_ylabel("Fraction")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)

# Remove unused axes
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right")

plt.tight_layout()
plt.show()