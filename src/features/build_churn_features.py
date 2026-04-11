import pandas as pd
from datetime import datetime
import numpy as np

df = pd.read_parquet("../../data/processed/retailrocket_standardized.parquet")
print("Loaded data shape:", df.shape)
# 1. Calc Churn Label (30 days)
# last date in the dataset as "today" for now
today = df["interaction_timestamp"].max()
# Recency: days since last activity per user
last_activity = df.groupby("user_id")["interaction_timestamp"].max().reset_index()
last_activity["days_since_last_activity"] = (
    today - last_activity["interaction_timestamp"]
).dt.days

# Churn = 1, if inactive for 30
last_activity["churn"] = (last_activity["days_since_last_activity"] >= 30).astype(int)

print("Mean Churn rate (30+ days inactive):", last_activity["churn"].mean().round(3))

# 2. Basic RFM + Aggregation Features (MUST HAVE)
features = last_activity[["user_id", "days_since_last_activity", "churn"]].copy()

# Frequency & Aggregation
agg = (
    df.groupby("user_id")
    .agg(
        {
            "interaction_type": "count",  # total interactions
            "product_id": "nunique",  # unique products viewed/interacted
            "interaction_weight": "sum",  # total weighted interactions
        }
    )
    .reset_index()
)

agg = agg.rename(
    columns={
        "interaction_type": "total_interactions",
        "product_id": "unique_products",
        "interaction_weight": "total_weight",
    }
)

# Specific counts
view_count = (
    df[df["interaction_type"] == "view"].groupby("user_id").size().reset_index(name="view_count")
)
cart_count = (
    df[df["interaction_type"] == "addtocart"]
    .groupby("user_id")
    .size()
    .reset_index(name="cart_count")
)
purchase_count = (
    df[df["interaction_type"] == "transaction"]
    .groupby("user_id")
    .size()
    .reset_index(name="purchase_count")
)

# Merge everything
features = features.merge(agg, on="user_id", how="left")
features = features.merge(view_count, on="user_id", how="left")
features = features.merge(cart_count, on="user_id", how="left")
features = features.merge(purchase_count, on="user_id", how="left")

# Fill NaN with 0 (users who never did that action)
features = features.fillna(0)

# 3. Important Ratio Features (Very Important for Churn)

features["view_to_purchase_ratio"] = features["view_count"] / (
    features["purchase_count"] + 1
)  # +1 to avoid division by zero
features["cart_to_purchase_ratio"] = features["cart_count"] / (
    features["purchase_count"] + 1
)
features["cart_abandonment_rate"] = features["cart_count"] / (
    features["cart_count"] + features["purchase_count"] + 1
)

print(" First 5 rows of features: ")
print(features.head().to_string())

print(" Features created:", features.columns.tolist())

# Save for model training
features.to_parquet("../../data/processed/churn_features.parquet", index=False)
print("Churn features saved to ../../data/processed/churn_features.parquet")
