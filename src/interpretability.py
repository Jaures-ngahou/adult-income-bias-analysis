# ==========================================================
# Import Libraries
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt

import sys

sys.path.append("../src")
from training import (
    split_features_target,
    standardize_features
)

from evaluation import load_model


# ==========================================================
# Load Dataset
# ==========================================================

TRAIN_PATH = "../data/processed/train.csv"
TEST_PATH = "../data/processed/test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train, X_test, y_train, y_test = split_features_target(
    train_df,
    test_df
)

X_train_scaled, X_test_scaled, scaler = standardize_features(
    X_train,
    X_test
)


# ==========================================================
# Load Trained Models
# ==========================================================

logistic_model = load_model("../models/logistic_regression.pkl")

rf_model = load_model("../models/random_forest.pkl")


# ==========================================================
# Logistic Regression Coefficients
# ==========================================================

coefficients = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": logistic_model.coef_[0]
})

coefficients



# ==========================================================
# Top Positive Coefficients
# ==========================================================

coefficients.sort_values(
    by="Coefficient",
    ascending=False
).head(10)


# ==========================================================
# Most Influential Features
# ==========================================================

coefficients["Absolute Coefficient"] = coefficients["Coefficient"].abs()

coefficients.sort_values(
    by="Absolute Coefficient",
    ascending=False
).head(15)


# ==========================================================
# Random Forest Feature Importance
# ==========================================================

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

feature_importance.head(15)

# ==========================================================
# Plot Feature Importance
# ==========================================================

top_features = feature_importance.head(15)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.gca().invert_yaxis()

plt.xlabel("Feature Importance")
plt.ylabel("Feature")

plt.title("Top 15 Feature Importances - Random Forest")

plt.tight_layout()

plt.show()