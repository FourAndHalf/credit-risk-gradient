"""
Batch Gradient Descent Implementation
======================================
The core optimisation loop, with full iteration history for convergence
analysis.  This is a from-scratch implementation — no library optimiser.

The history dict recorded at every iteration is the basis for all
convergence plots and tables in the dissertation.
"""

import time
import numpy as np
from dataclasses import dataclass, field

from .logistic_regression import (
    binary_cross_entropy,
    gradient,
    gradient_norm,
    hessian,
    hessian_eigenvalues,
    condition_number,
)


@dataclass
class GDResult:
    """Container for a single Gradient Descent run."""

    w: np.ndarray                           # final weights
    b: float                                # final intercept
    converged: bool                         # True if gradient norm < tol
    iterations: int                         # total iterations executed
    elapsed_sec: float                      # wall-clock seconds

    # Per-iteration history
    loss_history: list[float] = field(default_factory=list)
    grad_norm_history: list[float] = field(default_factory=list)
    w_history: list[np.ndarray] = field(default_factory=list)
    b_history: list[float] = field(default_factory=list)

    # Hessian diagnostics (computed once at the end by default)
    hessian_eigenvalues: np.ndarray | None = None
    condition_number: float | None = None


def batch_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float = 0.01,
    lam: float = 0.0,
    max_iter: int = 10_000,
    tol: float = 1e-6,
    record_params: bool = False,
    compute_hessian: bool = True,
    verbose: bool = False,
    print_every: int = 500,
) -> GDResult:
    """Run batch Gradient Descent on the logistic-regression objective.

    Stopping rule
    -------------
    Stop when ‖∇J‖ < tol  OR  iteration count reaches max_iter.

    Parameters
    ----------
    X : (m, p) design matrix.
    y : (m,)   binary targets.
    learning_rate : η — the constant step size.
    lam : L2 regularisation parameter (λ).  0 = unregularised.
    max_iter : upper bound on iterations.
    tol : gradient-norm tolerance for convergence.
    record_params : if True, store w and b at each iteration (memory
        intensive for large p — leave False for production runs).
    compute_hessian : if True, compute the Hessian, its eigenvalues and
        the condition number at the final parameter values.
    verbose : print progress.
    print_every : iteration interval for verbose output.

    Returns
    -------
    GDResult
    """
    m, p = X.shape

    # Initialise parameters to zero
    w = np.zeros(p, dtype=np.float64)
    b = 0.0

    loss_history: list[float] = []
    grad_norm_history: list[float] = []
    w_history: list[np.ndarray] = []
    b_history: list[float] = []

    converged = False
    t_start = time.perf_counter()

    for k in range(1, max_iter + 1):
        # --- Forward pass & loss ---
        loss = binary_cross_entropy(X, y, w, b, lam=lam)
        loss_history.append(loss)

        # --- Gradient ---
        gw, gb = gradient(X, y, w, b, lam=lam)
        gnorm = gradient_norm(gw, gb)
        grad_norm_history.append(gnorm)

        if record_params:
            w_history.append(w.copy())
            b_history.append(b)

        # --- Convergence check ---
        if gnorm < tol:
            converged = True
            if verbose:
                print(f"  Converged at iteration {k}  "
                      f"(loss={loss:.8f}, ‖∇‖={gnorm:.2e})")
            break

        # --- Divergence guard ---
        if not np.isfinite(loss):
            if verbose:
                print(f"  Diverged at iteration {k} (loss={loss})")
            break

        # --- Update ---
        w = w - learning_rate * gw
        b = b - learning_rate * gb

        if verbose and k % print_every == 0:
            print(f"  iter {k:>6d}  loss={loss:.8f}  ‖∇‖={gnorm:.2e}")

    elapsed = time.perf_counter() - t_start

    # --- Hessian diagnostics at the final point ---
    h_eigvals = None
    kappa = None
    if compute_hessian:
        H = hessian(X, w, b, lam=lam)
        h_eigvals = hessian_eigenvalues(H)
        kappa = condition_number(H)

    return GDResult(
        w=w, b=b,
        converged=converged,
        iterations=len(loss_history),
        elapsed_sec=elapsed,
        loss_history=loss_history,
        grad_norm_history=grad_norm_history,
        w_history=w_history if record_params else [],
        b_history=b_history if record_params else [],
        hessian_eigenvalues=h_eigvals,
        condition_number=kappa,
    )


def run_experiment_grid(
    X: np.ndarray,
    y: np.ndarray,
    learning_rates: list[float],
    *,
    lam: float = 0.0,
    max_iter: int = 10_000,
    tol: float = 1e-6,
    verbose: bool = True,
) -> dict[float, GDResult]:
    """Run GD for several learning rates and return results keyed by η."""
    results = {}
    for lr in learning_rates:
        if verbose:
            print(f"\n--- η = {lr} ---")
        results[lr] = batch_gradient_descent(
            X, y,
            learning_rate=lr,
            lam=lam,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
        )
    return results
