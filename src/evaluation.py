"""
Evaluation Module
=================
Classification and optimisation metrics for the convergence study.
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

from .logistic_regression import predict, predict_proba


def classification_metrics(
    X: np.ndarray,
    y_true: np.ndarray,
    w: np.ndarray,
    b: float,
    threshold: float = 0.5,
) -> dict:
    """Compute a full set of classification metrics.

    Positive class: y = 1 (bad credit).

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc,
        confusion_matrix, classification_report (str).
    """
    y_pred = predict(X, w, b, threshold=threshold)
    y_prob = predict_proba(X, w, b)

    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": cm,
        "report": classification_report(y_true, y_pred, digits=4),
    }
    return metrics


def optimisation_summary(result) -> dict:
    """Extract key optimisation diagnostics from a GDResult."""
    return {
        "converged": result.converged,
        "iterations": result.iterations,
        "final_loss": result.loss_history[-1] if result.loss_history else None,
        "final_grad_norm": (
            result.grad_norm_history[-1] if result.grad_norm_history else None
        ),
        "elapsed_sec": result.elapsed_sec,
        "condition_number": result.condition_number,
    }


def compare_with_sklearn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    w_gd: np.ndarray,
    b_gd: float,
    lam: float = 0.0,
    max_iter_sk: int = 10_000,
) -> dict:
    """Fit scikit-learn LogisticRegression and compare coefficients.

    Parameters
    ----------
    lam : float
        The λ used in our implementation.  sklearn's C = 1/λ (or very
        large when λ = 0).

    Returns
    -------
    dict with sklearn metrics, coefficient comparison, and max absolute
        difference in predicted probabilities on the test set.
    """
    from sklearn.linear_model import LogisticRegression

    C_val = 1.0 / lam if lam > 0 else 1e8  # near-unregularised

    clf = LogisticRegression(
        penalty="l2",
        C=C_val,
        solver="lbfgs",
        max_iter=max_iter_sk,
        fit_intercept=True,
        tol=1e-8,
    )
    clf.fit(X_train, y_train)

    sk_w = clf.coef_.ravel()
    sk_b = float(clf.intercept_[0])

    # Coefficient comparison
    w_diff = np.abs(w_gd - sk_w)

    # Predicted-probability comparison on test set
    gd_probs = 1.0 / (1.0 + np.exp(-(X_test @ w_gd + b_gd)))
    sk_probs = clf.predict_proba(X_test)[:, 1]
    prob_diff = np.abs(gd_probs - sk_probs)

    return {
        "sklearn_accuracy": float(clf.score(X_test, y_test)),
        "sklearn_w": sk_w,
        "sklearn_b": sk_b,
        "max_w_diff": float(w_diff.max()),
        "mean_w_diff": float(w_diff.mean()),
        "max_prob_diff": float(prob_diff.max()),
        "mean_prob_diff": float(prob_diff.mean()),
    }
