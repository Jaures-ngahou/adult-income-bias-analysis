import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib



# ======================================================
# Load processed datasets
# ======================================================
train_path = "../data/processed/train.csv"
test_path = "../data/processed/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# ==========================================================
# 1. Split Features and Target
# ==========================================================

def split_features_target(train_df, test_df):
    """
    Split the training and testing datasets into
    features (X) and target (y).
    """

    print("Splitting features and target...")

    X_train = train_df.drop(columns="income")
    y_train = train_df["income"]

    X_test = test_df.drop(columns="income")
    y_test = test_df["income"]

    print("Features and target successfully separated.\n")

    return X_train, X_test, y_train, y_test


# ==========================================================
# 2. Standardize Features
# ==========================================================

def standardize_features(X_train, X_test):
    """
    Standardize the input features using StandardScaler.

    The scaler is fitted only on the training set
    to avoid data leakage.
    """

    print("Standardizing features...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Feature standardization completed.\n")

    return X_train_scaled, X_test_scaled, scaler


# ==========================================================
# 3. Train Logistic Regression
# ==========================================================

def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression classifier.
    """

    print("Training Logistic Regression model...")

    model = LogisticRegression(
        random_state=42,
        max_iter=2000
    )

    model.fit(X_train, y_train)

    print("Logistic Regression training completed.\n")

    return model


# ==========================================================
# 4. Train Random Forest
# ==========================================================

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier.
    """

    print("Training Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("Random Forest training completed.\n")

    return model


# ==========================================================
# 5. Save Model
# ==========================================================

def save_model(model, file_path):
    """
    Save a trained model to disk.
    """

    print(f"Saving model to {file_path}...")

    joblib.dump(model, file_path)

    print("Model saved successfully.\n")
    
    
# ==========================================================
# 6. Load Model
# ==========================================================

def load_model(file_path):
    """
    Load a trained model from disk.
    """

    print(f"Loading model from {file_path}...")

    model = joblib.load(file_path)

    print("Model loaded successfully.\n")

    return model