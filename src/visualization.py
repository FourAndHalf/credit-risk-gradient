"""
Visualisation Module
====================
Plotting utilities for the convergence-analysis project.
All figures are saved to ``results/figures/`` and also returned for
inline display in notebooks.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path

FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Consistent style
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


def _save(fig: Figure, name: str) -> None:
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Loss curve
# ---------------------------------------------------------------------------

def plot_loss_curve(
    loss_history: list[float],
    title: str = "Objective Value vs Iteration",
    save_name: str | None = "loss_curve",
    ax: plt.Axes | None = None,
) -> Figure:
    """Plot J(w,b) against iteration number."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(range(1, len(loss_history) + 1), loss_history, linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss  J(w, b)")
    ax.set_title(title)

    if save_name:
        _save(fig, save_name)
    return fig


def plot_loss_curves_comparison(
    results: dict,
    title: str = "Loss Curves — Learning-Rate Comparison",
    save_name: str | None = "loss_curves_lr_comparison",
    max_iters_plot: int | None = None,
) -> Figure:
    """Overlay loss curves for multiple GDResults keyed by label."""
    fig, ax = plt.subplots()
    for label, res in results.items():
        hist = res.loss_history
        if max_iters_plot:
            hist = hist[:max_iters_plot]
        ax.plot(range(1, len(hist) + 1), hist, label=str(label), linewidth=1.2)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss  J(w, b)")
    ax.set_title(title)
    ax.legend(title="η")

    if save_name:
        _save(fig, save_name)
    return fig


# ---------------------------------------------------------------------------
# Gradient-norm curve
# ---------------------------------------------------------------------------

def plot_grad_norm(
    grad_norm_history: list[float],
    title: str = "Gradient Norm vs Iteration",
    save_name: str | None = "grad_norm",
    log_scale: bool = True,
) -> Figure:
    fig, ax = plt.subplots()
    ax.plot(range(1, len(grad_norm_history) + 1), grad_norm_history, linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("‖∇J‖")
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")

    if save_name:
        _save(fig, save_name)
    return fig


def plot_grad_norms_comparison(
    results: dict,
    title: str = "Gradient Norm — Learning-Rate Comparison",
    save_name: str | None = "grad_norms_lr_comparison",
    log_scale: bool = True,
    max_iters_plot: int | None = None,
) -> Figure:
    fig, ax = plt.subplots()
    for label, res in results.items():
        hist = res.grad_norm_history
        if max_iters_plot:
            hist = hist[:max_iters_plot]
        ax.plot(range(1, len(hist) + 1), hist, label=str(label), linewidth=1.2)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("‖∇J‖")
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")
    ax.legend(title="η")

    if save_name:
        _save(fig, save_name)
    return fig


# ---------------------------------------------------------------------------
# Hessian eigenvalue spectrum
# ---------------------------------------------------------------------------

def plot_eigenvalue_spectrum(
    eigenvalues: np.ndarray,
    title: str = "Hessian Eigenvalue Spectrum",
    save_name: str | None = "eigenvalue_spectrum",
) -> Figure:
    fig, ax = plt.subplots()
    ax.bar(range(len(eigenvalues)), np.sort(eigenvalues)[::-1], width=0.8)
    ax.set_xlabel("Eigenvalue Index (sorted descending)")
    ax.set_ylabel("Eigenvalue λ")
    ax.set_title(title)

    if save_name:
        _save(fig, save_name)
    return fig


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str] | None = None,
    title: str = "Confusion Matrix",
    save_name: str | None = "confusion_matrix",
) -> Figure:
    if labels is None:
        labels = ["Good (0)", "Bad (1)"]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")

    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            colour = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=colour,
                    fontsize=14)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if save_name:
        _save(fig, save_name)
    return fig


# ---------------------------------------------------------------------------
# Scaling comparison (side-by-side)
# ---------------------------------------------------------------------------

def plot_scaling_comparison(
    result_unscaled,
    result_scaled,
    save_name: str | None = "scaling_comparison",
) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    axes[0].plot(result_unscaled.loss_history, label="Unscaled", linewidth=1.2)
    axes[0].plot(result_scaled.loss_history, label="Scaled", linewidth=1.2)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()

    # Gradient norms
    axes[1].plot(result_unscaled.grad_norm_history, label="Unscaled", linewidth=1.2)
    axes[1].plot(result_scaled.grad_norm_history, label="Scaled", linewidth=1.2)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("‖∇J‖")
    axes[1].set_title("Gradient Norm")
    axes[1].set_yscale("log")
    axes[1].legend()

    fig.suptitle("Effect of Feature Scaling on Convergence", fontsize=13)
    fig.tight_layout()

    if save_name:
        _save(fig, save_name)
    return fig


# ---------------------------------------------------------------------------
# Regularisation comparison
# ---------------------------------------------------------------------------

def plot_regularisation_comparison(
    result_unreg,
    result_reg,
    lam: float,
    save_name: str | None = "regularisation_comparison",
) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(result_unreg.loss_history, label="λ = 0", linewidth=1.2)
    axes[0].plot(result_reg.loss_history, label=f"λ = {lam}", linewidth=1.2)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()

    axes[1].plot(result_unreg.grad_norm_history, label="λ = 0", linewidth=1.2)
    axes[1].plot(result_reg.grad_norm_history, label=f"λ = {lam}", linewidth=1.2)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("‖∇J‖")
    axes[1].set_title("Gradient Norm")
    axes[1].set_yscale("log")
    axes[1].legend()

    fig.suptitle("Effect of L2 Regularisation on Convergence", fontsize=13)
    fig.tight_layout()

    if save_name:
        _save(fig, save_name)
    return fig
