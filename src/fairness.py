import pandas as pd

from sklearn.metrics import (
    recall_score
)
import sys

sys.path.append("../src")
from evaluation import (
    compute_metrics
)

# ==========================================================
# 1. Compute Group Metrics
# ==========================================================

def compute_group_metrics(X_test, y_test, y_pred, group_column):
    """
    Compute evaluation metrics for a specific demographic group.
    """

    mask = X_test[group_column] == 1

    metrics = compute_metrics(
        y_test[mask],
        y_pred[mask]
    )

    results = {
        "Group": group_column,
        "Samples": mask.sum(),
        **metrics
    }

    return results


# ==========================================================
# 2. Compare Groups
# ==========================================================

def compare_groups(X_test, y_test, y_pred, group_columns):
    """
    Compare model performance across demographic groups.
    """

    results = []

    for group in group_columns:

        results.append(
            compute_group_metrics(
                X_test,
                y_test,
                y_pred,
                group
            )
        )

    return pd.DataFrame(results)


# ==========================================================
# 3. Compute Disparate Impact
# ==========================================================

def compute_disparate_impact(
    X_test,
    y_pred,
    protected_group,
    reference_group
):
    """
    Compute the Disparate Impact (DI).
    """

    protected_mask = X_test[protected_group] == 1
    reference_mask = X_test[reference_group] == 1

    protected_rate = y_pred[protected_mask].mean()
    reference_rate = y_pred[reference_mask].mean()

    di = protected_rate / reference_rate

    return {
        "Protected Group": protected_group,
        "Reference Group": reference_group,
        "Protected Positive Rate": protected_rate,
        "Reference Positive Rate": reference_rate,
        "Disparate Impact": di
    }


# ==========================================================
# 4. Compute Equal Opportunity Difference
# ==========================================================

def compute_equal_opportunity_difference(
    X_test,
    y_test,
    y_pred,
    protected_group,
    reference_group
):
    """
    Compute the Equal Opportunity Difference (EOD).
    """

    protected_mask = X_test[protected_group] == 1
    reference_mask = X_test[reference_group] == 1

    protected_tpr = recall_score(
        y_test[protected_mask],
        y_pred[protected_mask]
    )

    reference_tpr = recall_score(
        y_test[reference_mask],
        y_pred[reference_mask]
    )

    eod = protected_tpr - reference_tpr

    return {
        "Protected Group": protected_group,
        "Reference Group": reference_group,
        "Protected TPR": protected_tpr,
        "Reference TPR": reference_tpr,
        "Equal Opportunity Difference": eod
    }