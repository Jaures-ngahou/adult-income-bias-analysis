import pandas as pd
from sklearn.model_selection import train_test_split


# ==========================================================
# Paths
# ==========================================================

DATA_PATH = "../data/raw/adult.csv"
TRAIN_PATH = "../data/processed/train.csv"
TEST_PATH = "../data/processed/test.csv"


# ==========================================================
# 1. Load Dataset
# ==========================================================

def load_dataset(path):
    """
    Load the Adult Census Income dataset.
    """
    print("Loading dataset...")

    df = pd.read_csv(path)

    print(f"Dataset loaded successfully ({df.shape[0]} rows, {df.shape[1]} columns).\n")

    return df


# ==========================================================
# 2. Handle Missing Values
# ==========================================================

def handle_missing_values(df):
    """
    Clean categorical variables and replace missing values.
    """
    print("Handling missing values...")

    df_clean = df.copy()

    # Remove leading/trailing spaces
    for column in df_clean.select_dtypes(include="object").columns:
        df_clean[column] = df_clean[column].str.strip()

    # Replace missing values represented by '?'
    df_clean.replace("?", "Unknown", inplace=True)

    print("Missing values handled successfully.\n")

    return df_clean


# ==========================================================
# 3. One-Hot Encoding
# ==========================================================

def encode_categorical_features(df):
    """
    Apply One-Hot Encoding to all categorical input features.
    """

    print("Applying One-Hot Encoding...")

    categorical_columns = df.select_dtypes(include="object").columns.tolist()

    # Remove target variable
    categorical_columns.remove("income")

    df_encoded = pd.get_dummies(
        df,
        columns=categorical_columns,
        dtype=int
    )

    print(f"Encoding completed ({df_encoded.shape[1]} columns).\n")

    return df_encoded


# ==========================================================
# 4. Split Dataset
# ==========================================================

def split_dataset(df):
    """
    Convert the target variable and split the dataset
    into training and testing sets.
    """

    print("Splitting dataset...")

    # Convert target to binary values
    df["income"] = df["income"].map({
        "<=50K": 0,
        ">50K": 1
    })

    # Features
    X = df.drop(columns="income")

    # Target
    y = df["income"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Dataset successfully split.\n")

    return X_train, X_test, y_train, y_test


# ==========================================================
# 5. Save Datasets
# ==========================================================

def save_datasets(X_train, X_test, y_train, y_test):
    """
    Save the training and testing datasets.
    """

    print("Saving processed datasets...")

    train_df = X_train.copy()
    train_df["income"] = y_train.values

    test_df = X_test.copy()
    test_df["income"] = y_test.values

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"Training dataset saved: {TRAIN_PATH}")
    print(f"Testing dataset saved : {TEST_PATH}\n")


# ==========================================================
# Main Preprocessing Pipeline
# ==========================================================

def preprocess_data():
    """
    Execute the complete preprocessing pipeline.
    """

    df = load_dataset(DATA_PATH)

    df = handle_missing_values(df)

    df = encode_categorical_features(df)

    X_train, X_test, y_train, y_test = split_dataset(df)

    save_datasets(X_train, X_test, y_train, y_test)

    print("Preprocessing completed successfully!")


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    preprocess_data()