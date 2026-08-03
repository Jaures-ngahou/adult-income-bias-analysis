from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt
import joblib


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


# ==========================================================
# 1. Make Predictions
# ==========================================================

def make_predictions(model, X_test):
    """
    Generate predictions using a trained model.
    """

    print("Generating predictions...")

    y_pred = model.predict(X_test)

    print("Predictions generated successfully.\n")

    return y_pred


# ==========================================================
# 2. Compute Accuracy
# ==========================================================

def compute_accuracy(y_test, y_pred):
    """
    Compute the accuracy score.
    """

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}\n")

    return accuracy


# ==========================================================
# 3. Classification Report
# ==========================================================

def display_classification_report(y_test, y_pred):
    """
    Display precision, recall and F1-score.
    """

    print("Classification Report:\n")

    report = classification_report(
        y_test,
        y_pred
    )

    print(report)

    return report


# ==========================================================
# 4. Confusion Matrix
# ==========================================================

def plot_confusion_matrix(y_test, y_pred):
    """
    Display the confusion matrix.
    """

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "<=50K",
            ">50K"
        ]
    )

    cm_display.plot()

    plt.title("Confusion Matrix")

    plt.show()

    return cm


# ==========================================================
# 5. Complete Evaluation Pipeline
# ==========================================================

def evaluate_model(model, X_test, y_test):
    """
    Execute the complete evaluation pipeline.
    """

    y_pred = make_predictions(
        model,
        X_test
    )

    compute_accuracy(
        y_test,
        y_pred
    )

    display_classification_report(
        y_test,
        y_pred
    )

    plot_confusion_matrix(
        y_test,
        y_pred
    )

    return y_pred

# ==========================================================
# Compute Evaluation Metrics to be used in calculating the bias
# ==========================================================

def compute_metrics(y_test, y_pred):
    """
    Compute the main evaluation metrics.
    """

    metrics = {
        "Accuracy": compute_accuracy(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred)
    }

    return metrics