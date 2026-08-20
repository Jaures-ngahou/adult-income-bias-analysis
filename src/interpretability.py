import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# 1. Logistic Regression Coefficients
# ==========================================================

def get_logistic_coefficients(model, feature_names):
    """
    Return the coefficients of a trained Logistic Regression model.
    """

    coefficients = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_[0]
    })

    return coefficients


# ==========================================================
# 2. Most Influential Logistic Regression Features
# ==========================================================

def get_most_influential_features(coefficients, top_n=15):
    """
    Return the features with the largest absolute coefficients.
    """

    coefficients = coefficients.copy()

    coefficients["Absolute Coefficient"] = (
        coefficients["Coefficient"].abs()
    )

    return coefficients.sort_values(
        by="Absolute Coefficient",
        ascending=False
    ).head(top_n)


# ==========================================================
# 3. Random Forest Feature Importance
# ==========================================================

def get_feature_importance(model, feature_names, top_n=15):
    """
    Return the most important features of a trained Random Forest model.
    """

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    return feature_importance.sort_values(
        by="Importance",
        ascending=False
    ).head(top_n)


# ==========================================================
# 4. Plot Feature Importance
# ==========================================================

def plot_feature_importance(feature_importance):
    """
    Plot Random Forest feature importance.
    """

    plt.figure(figsize=(10, 6))

    plt.barh(
        feature_importance["Feature"],
        feature_importance["Importance"]
    )

    plt.gca().invert_yaxis()

    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")

    plt.title("Top Feature Importances - Random Forest")

    plt.tight_layout()

    plt.show()