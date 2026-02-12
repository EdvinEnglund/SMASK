import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

# -----------------------------
# 1) READ DATA
# -----------------------------
df = pd.read_csv("full_preprocessed_training_data.csv")

y = (df["increase_stock"] == 1).astype(int)

# -----------------------------
# 2) DEFINE FEATURE SCENARIOS
# -----------------------------
scenarios = {
    "baseline": [],
    "no_temp": ["temp"],
    "no_humidity": ["humidity"],
    "no_snowdepth": ["snowdepth"],
    "no_time": [
        "weekday", "holiday", "summertime",
        *[c for c in df.columns if c.startswith("month")],
        *[c for c in df.columns if c.startswith("day_of_week")],
        *[c for c in df.columns if c.startswith("hour_of_day")]
    ],
}

k_values = range(3, 31, 2)

# -----------------------------
# 3) LOOP SCENARIOS
# -----------------------------
for scenario, cols_to_remove in scenarios.items():

    X = df.drop(columns=["increase_stock"] + cols_to_remove, errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    best_k = None
    best_f1 = -1

    # ----- test different k -----
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)
        f1 = f1_score(y_test, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_k = k

    print(f"\nScenario: {scenario}")
    print(f"Features: {X.shape[1]}")
    print(f"Best k  : {best_k}")
    print(f"Best F1 : {best_f1:.3f}")
