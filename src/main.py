import pandas as pd

import sys

sys.path.append("./src")

from preprocessing import preprocess_data

from training import (
    split_features_target,
    standardize_features,
    train_logistic_regression,
    train_random_forest,
    save_model
)

from evaluation import evaluate_model

from fairness import (
    compare_groups,
    compute_disparate_impact,
    compute_equal_opportunity_difference
)


# ==========================================================
# Main Program
# ==========================================================

def main():

    print("=" * 60)
    print("Adult Census Income Bias Analysis")
    print("=" * 60)

    # ======================================================
    # Phase A - Preprocessing
    # ======================================================

    preprocess_data()

    # ======================================================
    # Load processed datasets
    # ======================================================

    train_df = pd.read_csv("../data/processed/train.csv")
    test_df = pd.read_csv("../data/processed/test.csv")

    # ======================================================
    # Split features and target
    # ======================================================

    X_train, X_test, y_train, y_test = split_features_target(
        train_df,
        test_df
    )

    # ======================================================
    # Standardize data for Logistic Regression
    # ======================================================

    X_train_scaled, X_test_scaled, scaler = standardize_features(
        X_train,
        X_test
    )

    # ======================================================
    # Train Logistic Regression
    # ======================================================

    logistic_model = train_logistic_regression(
        X_train_scaled,
        y_train
    )

    save_model(
        logistic_model,
        "../models/logistic_regression.pkl"
    )

    # ======================================================
    # Train Random Forest
    # ======================================================

    random_forest_model = train_random_forest(
        X_train,
        y_train
    )

    save_model(
        random_forest_model,
        "../models/random_forest.pkl"
    )

    # ======================================================
    # Model Evaluation
    # ======================================================

    print("\n========== Logistic Regression ==========\n")

    logistic_predictions = evaluate_model(
        logistic_model,
        X_test_scaled,
        y_test
    )

    print("\n========== Random Forest ==========\n")

    random_forest_predictions = evaluate_model(
        random_forest_model,
        X_test,
        y_test
    )

    # ======================================================
    # Fairness Analysis
    # ======================================================

    gender_columns = [
        "sex_Female",
        "sex_Male"
    ]

    race_columns = [
        column for column in X_test.columns
        if column.startswith("race_")
    ]

    print("\n========== Gender Analysis ==========\n")

    print(
        compare_groups(
            X_test,
            y_test,
            logistic_predictions,
            gender_columns
        )
    )

    print("\n========== Race Analysis ==========\n")

    print(
        compare_groups(
            X_test,
            y_test,
            logistic_predictions,
            race_columns
        )
    )
    """
    print("\n========== Disparate Impact ==========\n")

    compute_disparate_impact(
        X_test,
        logistic_predictions,
        "sex_Female",
        "sex_Male"
    )

    print("\n========== Equal Opportunity Difference ==========\n")

    compute_equal_opportunity_difference(
        X_test,
        y_test,
        logistic_predictions,
        "sex_Female",
        "sex_Male"
    )

    print("\nProgram completed successfully.")
    """
    # ======================================================
    # Disparate Impact
    # ======================================================

    print("\n========== Disparate Impact ==========\n")

    di_results = compute_disparate_impact(
        X_test,
        logistic_predictions,
        "sex_Female",
        "sex_Male"
    )

    for key, value in di_results.items():
        print(f"{key}: {value}")

    # ======================================================
    # Equal Opportunity Difference
    # ======================================================

    print("\n========== Equal Opportunity Difference ==========\n")

    eod_results = compute_equal_opportunity_difference(
        X_test,
        y_test,
        logistic_predictions,
        "sex_Female",
        "sex_Male"
    )

    for key, value in eod_results.items():
        print(f"{key}: {value}")

    print("\nProgram completed successfully.")

# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()