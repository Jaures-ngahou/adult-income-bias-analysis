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

    The function selects the samples belonging to the specified
    group and computes the model performance metrics for that group.
    """

    # Select only the samples belonging to the specified group
    mask = X_test[group_column] == 1

    # Compute the evaluation metrics for the selected group
    metrics = compute_metrics(
        y_test[mask],
        y_pred[mask]
    )

    # Store the group name, number of samples and computed metrics
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

    The function computes the evaluation metrics separately
    for each demographic group and returns the results as a DataFrame.
    """

    results = []

    # Compute metrics for each demographic group
    for group in group_columns:

        results.append(
            compute_group_metrics(
                X_test,
                y_test,
                y_pred,
                group
            )
        )

    # Convert the results into a DataFrame for easier analysis
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

    DI compares the positive prediction rate of the protected
    group with that of a reference group.
    """

    # Create masks to identify the two demographic groups
    protected_mask = X_test[protected_group] == 1
    reference_mask = X_test[reference_group] == 1

    # Compute the proportion of positive predictions for each group
    protected_rate = y_pred[protected_mask].mean()
    reference_rate = y_pred[reference_mask].mean()

    # Calculate the ratio between the two positive prediction rates
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

    EOD compares the True Positive Rate (TPR) of the protected
    group with that of a reference group.
    """

    # Create masks to identify the two demographic groups
    protected_mask = X_test[protected_group] == 1
    reference_mask = X_test[reference_group] == 1

    # Compute the True Positive Rate for the protected group
    protected_tpr = recall_score(
        y_test[protected_mask],
        y_pred[protected_mask]
    )

    # Compute the True Positive Rate for the reference group
    reference_tpr = recall_score(
        y_test[reference_mask],
        y_pred[reference_mask]
    )

    # Calculate the difference between the two TPRs
    eod = protected_tpr - reference_tpr

    return {
        "Protected Group": protected_group,
        "Reference Group": reference_group,
        "Protected TPR": protected_tpr,
        "Reference TPR": reference_tpr,
        "Equal Opportunity Difference": eod
    }