"""
Logistic Regression — Mathematical Core
========================================
Implements the sigmoid, binary cross-entropy loss, gradient and Hessian
for the convergence-analysis project.

All functions operate on NumPy arrays.  The parameter vector θ is stored
as a single array of shape (p+1,) where θ[-1] is the intercept b and
θ[:p] is the weight vector w.  This simplifies the Gradient Descent loop.

Notation (matches the blueprint):
    m  — number of training observations
    p  — number of features (after preprocessing)
    X  — design matrix  (m, p)
    y  — target vector   (m,)   values in {0, 1}
    w  — weight vector   (p,)
    b  — intercept       scalar
    z  — linear scores   Xw + b   (m,)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Sigmoid
# ---------------------------------------------------------------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid σ(z) = 1 / (1 + exp(-z))."""
    # Clip to avoid overflow in exp
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Return P(y=1 | X) for each row."""
    z = X @ w + b
    return sigmoid(z)


def predict(X: np.ndarray, w: np.ndarray, b: float,
            threshold: float = 0.5) -> np.ndarray:
    """Binary predictions."""
    return (predict_proba(X, w, b) >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Objective (loss) function
# ---------------------------------------------------------------------------

_EPS = 1e-15  # clipping bound for log-stability


def binary_cross_entropy(
    X: np.ndarray, y: np.ndarray,
    w: np.ndarray, b: float,
    lam: float = 0.0,
) -> float:
    """Compute J(w, b) — binary cross-entropy with optional L2 penalty.

    J = -(1/m) Σ [y·log(p̂) + (1-y)·log(1-p̂)]  +  (λ/2m)‖w‖²

    Parameters
    ----------
    lam : float
        L2 regularisation strength (λ).  0 = unregularised.
    """
    m = len(y)
    p_hat = predict_proba(X, w, b)
    p_hat = np.clip(p_hat, _EPS, 1 - _EPS)

    loss = -(1.0 / m) * (y @ np.log(p_hat) + (1 - y) @ np.log(1 - p_hat))

    if lam > 0:
        loss += (lam / (2 * m)) * np.dot(w, w)

    return float(loss)


# ---------------------------------------------------------------------------
# Gradient
# ---------------------------------------------------------------------------

def gradient(
    X: np.ndarray, y: np.ndarray,
    w: np.ndarray, b: float,
    lam: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Compute ∇_w J and ∂J/∂b.

    Returns
    -------
    grad_w : np.ndarray (p,)
    grad_b : float
    """
    m = len(y)
    p_hat = predict_proba(X, w, b)
    error = p_hat - y                       # (m,)

    grad_w = (1.0 / m) * (X.T @ error)     # (p,)
    grad_b = float((1.0 / m) * error.sum())

    if lam > 0:
        grad_w = grad_w + (lam / m) * w     # do not penalise b

    return grad_w, grad_b


def gradient_norm(grad_w: np.ndarray, grad_b: float) -> float:
    """L2 norm of the full gradient vector [grad_w; grad_b]."""
    return float(np.sqrt(np.dot(grad_w, grad_w) + grad_b ** 2))


# ---------------------------------------------------------------------------
# Hessian
# ---------------------------------------------------------------------------

def hessian(
    X: np.ndarray,
    w: np.ndarray, b: float,
    lam: float = 0.0,
) -> np.ndarray:
    """Compute the Hessian of J with respect to w (the p×p weight block).

    H = (1/m) Xᵀ W X  +  (λ/m) I

    where W = diag(p̂_i (1 − p̂_i)).

    For numerical studies on the credit dataset (p ~ 40–60 after encoding)
    the full Hessian is small enough to compute and eigen-decompose.
    """
    m = X.shape[0]
    p_hat = predict_proba(X, w, b)
    s = p_hat * (1 - p_hat)                 # diagonal weights (m,)

    # H = (1/m) Xᵀ diag(s) X
    # Efficient: scale rows of X then multiply
    Xs = X * s[:, np.newaxis]               # (m, p) each row scaled
    H = (1.0 / m) * (Xs.T @ X)             # (p, p)

    if lam > 0:
        H += (lam / m) * np.eye(X.shape[1])

    return H


def hessian_eigenvalues(H: np.ndarray) -> np.ndarray:
    """Return sorted eigenvalues (ascending) of a symmetric matrix."""
    eigvals = np.linalg.eigvalsh(H)
    return eigvals


def condition_number(H: np.ndarray) -> float:
    """Condition number κ(H) = λ_max / λ_min for a positive-definite H.

    Returns np.inf if λ_min ≤ 0 (not positive definite).
    """
    eigvals = hessian_eigenvalues(H)
    lam_min = eigvals[0]
    lam_max = eigvals[-1]
    if lam_min <= 0:
        return np.inf
    return float(lam_max / lam_min)
