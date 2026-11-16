"""
Run every experiment in the study and regenerate all tables and figures.

Usage (from the project root):
    python run_all_experiments.py

Outputs
    results/tables/*.csv        one file per table in the report
    results/figures/*.png       one file per figure in the report
    results/results_summary.md  every table in one readable file

The script reads whatever is in data/raw/. Replace the placeholder file with
the real UCI South German Credit data and re-run to refresh every number.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

from src.data_preparation import (CATEGORICAL_COLS, NUMERICAL_COLS, TARGET_COL,
                                  load_raw_data, prepare_data)
from src.gradient_descent import batch_gradient_descent
from src.logistic_regression import predict_proba

ROOT = Path(__file__).parent
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

TOL = 1e-6
MAX_ITER = 20_000

# ---- chart style -----------------------------------------------------------
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "legend.frameon": False, "axes.axisbelow": True,
})


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)


# ---- mathematics helpers ---------------------------------------------------
def full_hessian(X, w, b, lam=0.0):
    """Hessian over all parameters (weights and intercept)."""
    m = X.shape[0]
    Xa = np.hstack([X, np.ones((m, 1))])
    p_hat = predict_proba(X, w, b)
    s = p_hat * (1 - p_hat)
    H = (Xa * s[:, None]).T @ Xa / m
    reg = np.eye(Xa.shape[1]) * (lam / m)
    reg[-1, -1] = 0.0                      # intercept is not penalised
    return H + reg


def spectrum(H):
    e = np.linalg.eigvalsh(H)
    return e, float(e[-1]), float(e[0]), float(e[-1] / e[0])


def status_of(res):
    loss = np.asarray(res.loss_history)
    if res.converged:
        return "Converged"
    if not np.all(np.isfinite(loss)) or np.any(np.diff(loss) > 1e-9):
        return "Unstable (loss increases)"
    return "Stable, tolerance not reached"


def tail_rate(res, window=200):
    g = np.asarray(res.grad_norm_history)
    if len(g) < window + 1:
        return np.nan
    return float((g[-1] / g[-1 - window]) ** (1.0 / window))


def metrics_at(y, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
    }


def main():
    out = []                                # markdown summary blocks

    def table(df, name, title):
        df.to_csv(TAB / f"{name}.csv", index=False)
        out.append(f"### {title}\n\n{df.to_markdown(index=False)}\n")
        print(f"\n{title}\n{df.to_string(index=False)}")

    # ======================= DATA ==========================================
    df = load_raw_data(ROOT / "data" / "raw")
    scaled = prepare_data(df, scale=True)
    unscaled = prepare_data(df, scale=False)
    X, Xt = scaled["X_train"], scaled["X_test"]
    y, yt = scaled["y_train"], scaled["y_test"]
    Xu = unscaled["X_train"]
    m, p = X.shape

    counts = df[TARGET_COL].value_counts().sort_index()
    table(pd.DataFrame({
        "Class": ["Good credit (y = 0)", "Bad credit (y = 1)"],
        "Full data": counts.values,
        "Share (%)": (100 * counts.values / len(df)).round(1),
        "Training set": [int((y == 0).sum()), int((y == 1).sum())],
        "Test set": [int((yt == 0).sum()), int((yt == 1).sum())],
    }), "data_class_distribution", "Class distribution")

    d = df[NUMERICAL_COLS].describe().T
    d["skew"] = df[NUMERICAL_COLS].skew()
    table(d[["mean", "std", "min", "25%", "50%", "75%", "max", "skew"]]
          .round(2).reset_index().rename(columns={"index": "Variable"}),
          "data_numeric_summary", "Descriptive statistics of numerical predictors")

    table(pd.DataFrame({
        "Variable": CATEGORICAL_COLS,
        "Levels": [df[c].nunique() for c in CATEGORICAL_COLS],
        "Dummy columns created": [df[c].nunique() - 1 for c in CATEGORICAL_COLS],
        "Largest level share (%)": [round(100 * df[c].value_counts(normalize=True).max(), 1)
                                    for c in CATEGORICAL_COLS],
    }), "data_categorical_encoding", "Categorical predictors and encoding expansion")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(["Good credit", "Bad credit"], counts.values, color=BLUE, width=0.5)
    ax.bar_label(bars, labels=[f"{v} ({100 * v / len(df):.0f}%)" for v in counts.values],
                 padding=3, color=INK)
    ax.set_ylabel("Number of applicants")
    ax.set_title("Class distribution of the credit outcome")
    ax.grid(axis="x", visible=False)
    save(fig, "fig01_class_distribution.png")

    # ======================= CURVATURE AT THE START ========================
    zero = np.zeros(p)
    _, L, mu0, k0 = spectrum(full_hessian(X, zero, 0.0))
    eig_u, Lu, muu, ku = spectrum(full_hessian(Xu, zero, 0.0))
    eig_s = spectrum(full_hessian(X, zero, 0.0))[0]
    eta_max = 2.0 / L

    # Reference optimum (tight tolerance) for rate predictions
    ref = batch_gradient_descent(X, y, learning_rate=1.0, max_iter=200_000,
                                 tol=1e-10, compute_hessian=False)
    eig_star, L_star, mu_star, k_star = spectrum(full_hessian(X, ref.w, ref.b))

    table(pd.DataFrame({
        "Quantity": ["Parameters (weights + intercept)", "L = largest eigenvalue of H(0)",
                     "Step-size ceiling 2/L", "Largest eigenvalue of H(w*)",
                     "Smallest eigenvalue of H(w*)", "Condition number at w*",
                     "Minimum training loss J(w*)"],
        "Value": [p + 1, round(L, 4), round(eta_max, 3), round(L_star, 4),
                  f"{mu_star:.5f}", round(k_star, 1), f"{ref.loss_history[-1]:.6f}"],
    }), "curvature_summary", "Curvature of the scaled problem")

    # ======================= EXPERIMENT A ==================================
    etas = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 2.5, 3.0, 4.0]
    runs = {e: batch_gradient_descent(X, y, learning_rate=e, max_iter=MAX_ITER,
                                      tol=TOL, compute_hessian=False) for e in etas}
    rows = []
    for e, r in runs.items():
        st = status_of(r)
        theory = float(np.max(np.abs(1 - e * eig_star)))
        rows.append({
            "Learning rate": e, "Ratio to 2/L": round(e / eta_max, 3),
            "Iterations": r.iterations, "Final loss": round(r.loss_history[-1], 6),
            "Final gradient norm": f"{r.grad_norm_history[-1]:.2e}",
            "Predicted rate": round(theory, 5) if theory < 1 else "> 1",
            "Observed rate": round(tail_rate(r), 5) if st != "Unstable (loss increases)" else "-",
            "Status": st,
        })
    table(pd.DataFrame(rows), "expA_learning_rate", "Experiment A: learning-rate sweep")

    stable = [e for e in etas if status_of(runs[e]) != "Unstable (loss increases)"]
    stable_plot = stable[-len(BLUE_RAMP):]
    for name, key, ylabel, title in [
        ("fig02_loss_by_learning_rate.png", "loss_history", "Training loss",
         "Training loss for stable learning rates"),
        ("fig03_gradient_norm_by_learning_rate.png", "grad_norm_history",
         "Gradient norm", "Gradient norm for stable learning rates"),
    ]:
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        for c, e in zip(BLUE_RAMP, stable_plot):
            v = getattr(runs[e], key)
            ax.plot(np.arange(1, len(v) + 1), v, color=c, label=f"η = {e}")
        ax.set_xscale("log")
        if key == "grad_norm_history":
            ax.set_yscale("log")
            ax.axhline(TOL, color=MUTED, linewidth=1, linestyle="--")
            ax.text(1.1, TOL * 1.4, "stopping tolerance", color=MUTED, fontsize=8)
        ax.set_xlabel("Iteration (log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(title="Learning rate", loc="upper left", bbox_to_anchor=(1.01, 1))
        save(fig, name)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    edge = [e for e in etas if e >= 2.0]
    for c, e in zip([BLUE, AQUA, YELLOW, ORANGE], edge):
        v = runs[e].loss_history[:60]
        ax.plot(np.arange(1, len(v) + 1), v, color=c,
                label=f"η = {e}  ({e / eta_max:.2f} × ceiling)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training loss")
    ax.set_title(f"Behaviour around the step-size ceiling 2/L = {eta_max:.2f}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    save(fig, "fig04_step_size_ceiling.png")

    # ======================= EXPERIMENT B ==================================
    rs = batch_gradient_descent(X, y, learning_rate=1 / L, max_iter=MAX_ITER,
                                tol=TOL, compute_hessian=False)
    ru = batch_gradient_descent(Xu, y, learning_rate=1 / Lu, max_iter=MAX_ITER,
                                tol=TOL, compute_hessian=False)

    def at(r, k):
        return round(r.loss_history[min(k, r.iterations) - 1], 4)

    table(pd.DataFrame({
        "Quantity": ["Largest eigenvalue L", "Smallest eigenvalue", "Condition number",
                     "Step-size ceiling 2/L", "Step size used (1/L)", "Loss after 100 iterations",
                     "Loss after 1,000 iterations", "Loss at stop", "Iterations run",
                     "Final gradient norm", "Status"],
        "Scaled features": [f"{L:.4f}", f"{mu0:.5f}", f"{k0:.1f}", f"{2 / L:.3f}", f"{1 / L:.3f}",
                            at(rs, 100), at(rs, 1000), at(rs, MAX_ITER), rs.iterations,
                            f"{rs.grad_norm_history[-1]:.2e}", status_of(rs)],
        "Unscaled features": [f"{Lu:.3e}", f"{muu:.5f}", f"{ku:.3e}", f"{2 / Lu:.2e}", f"{1 / Lu:.2e}",
                              at(ru, 100), at(ru, 1000), at(ru, MAX_ITER), ru.iterations,
                              f"{ru.grad_norm_history[-1]:.2e}", status_of(ru)],
    }), "expB_scaling", "Experiment B: scaled versus unscaled features")

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    idx = np.arange(1, len(eig_s) + 1)
    ax.plot(idx, eig_s[::-1], color=BLUE, marker="o", markersize=4, label="Scaled features")
    ax.plot(idx, eig_u[::-1], color=ORANGE, marker="s", markersize=4, label="Unscaled features")
    ax.set_yscale("log")
    ax.set_xlabel("Eigenvalue rank (largest first)")
    ax.set_ylabel("Eigenvalue of H(0), log scale")
    ax.set_title("Hessian eigenvalue spectrum at the starting point")
    ax.legend()
    save(fig, "fig05_eigenvalue_spectrum.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(np.arange(1, rs.iterations + 1), rs.loss_history, color=BLUE,
            label=f"Scaled (κ = {k0:.0f})")
    ax.plot(np.arange(1, ru.iterations + 1), ru.loss_history, color=ORANGE,
            label=f"Unscaled (κ = {ku:.1e})")
    ax.set_xscale("log")
    ax.set_xlabel("Iteration (log scale)")
    ax.set_ylabel("Training loss")
    ax.set_title("Convergence with each problem run at its own step size 1/L")
    ax.legend()
    save(fig, "fig06_scaled_vs_unscaled.png")

    # ======================= EXPERIMENT C ==================================
    lams = [0.0, 0.1, 1.0, 10.0, 100.0]
    rows, reg_runs = [], {}
    for lam in lams:
        r = batch_gradient_descent(X, y, learning_rate=1.0, lam=lam, max_iter=MAX_ITER,
                                   tol=TOL, compute_hessian=False)
        reg_runs[lam] = r
        _, Lr, mur, kr = spectrum(full_hessian(X, r.w, r.b, lam))
        rows.append({
            "Lambda": lam, "Shift added to eigenvalues (lambda/m)": f"{lam / m:.5f}",
            "Smallest eigenvalue": f"{mur:.5f}", "Condition number": round(kr, 1),
            "Iterations to tolerance": r.iterations if r.converged else f"> {MAX_ITER}",
            "Weight norm": round(float(np.linalg.norm(r.w)), 3),
            "Training loss (unpenalised)": round(float(
                -np.mean(y * np.log(predict_proba(X, r.w, r.b)) +
                         (1 - y) * np.log(1 - predict_proba(X, r.w, r.b)))), 5),
            "Test AUC": round(roc_auc_score(yt, predict_proba(Xt, r.w, r.b)), 4),
        })
    reg = pd.DataFrame(rows)
    table(reg, "expC_regularisation", "Experiment C: L2 regularisation")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    labels = [str(l) for l in lams]
    for ax, col, title in [(axes[0], "Condition number", "Condition number at the solution"),
                           (axes[1], "Iterations to tolerance", "Iterations to reach tolerance")]:
        vals = [r.iterations if col.startswith("Iter") else v
                for r, v in zip(reg_runs.values(), reg["Condition number"])]
        bars = ax.bar(labels, vals, color=BLUE, width=0.55)
        ax.bar_label(bars, labels=[f"{v:,.0f}" for v in vals], padding=3, color=INK, fontsize=8)
        ax.set_xlabel("Regularisation strength λ")
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    save(fig, "fig07_regularisation.png")

    # ======================= EXPERIMENT D ==================================
    rows, agree = [], None
    for lam in [0.0, 10.0]:
        r = reg_runs[lam] if lam > 0 else ref
        clf = LogisticRegression(C=(1 / lam if lam > 0 else 1e10), solver="lbfgs",
                                 max_iter=50_000, tol=1e-12).fit(X, y)
        pg, ps = predict_proba(Xt, r.w, r.b), clf.predict_proba(Xt)[:, 1]
        if lam == 0:
            agree = (ps, pg)
        rows.append({
            "Lambda": lam,
            "Max coefficient difference": f"{np.abs(r.w - clf.coef_.ravel()).max():.2e}",
            "Intercept difference": f"{abs(r.b - clf.intercept_[0]):.2e}",
            "Max probability difference": f"{np.abs(pg - ps).max():.2e}",
            "Mean probability difference": f"{np.abs(pg - ps).mean():.2e}",
            "Predicted labels that differ": int(((pg >= .5) != (ps >= .5)).sum()),
        })
    table(pd.DataFrame(rows), "expD_sklearn", "Experiment D: agreement with scikit-learn")

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="Exact agreement")
    ax.scatter(agree[0], agree[1], s=22, color=BLUE, edgecolor=SURFACE, linewidth=0.6,
               label="Test applicants")
    ax.set_xlabel("scikit-learn probability of bad credit")
    ax.set_ylabel("From-scratch probability of bad credit")
    ax.set_title("Predicted probabilities: from-scratch vs scikit-learn")
    ax.legend()
    save(fig, "fig10_sklearn_agreement.png")

    # ======================= CLASSIFICATION ================================
    ptr, pte = predict_proba(X, ref.w, ref.b), predict_proba(Xt, ref.w, ref.b)
    rows = []
    for name, yy, pp in [("Training", y, ptr), ("Test", yt, pte)]:
        rows.append({"Set": name, **{k: round(v, 4) for k, v in metrics_at(yy, pp).items()},
                     "ROC-AUC": round(roc_auc_score(yy, pp), 4)})
    table(pd.DataFrame(rows), "classification_metrics",
          "Classification performance at threshold 0.5")

    rows = []
    for thr in [0.3, 0.4, 0.5]:
        cm = confusion_matrix(yt, (pte >= thr).astype(int))
        rows.append({"Threshold": thr, **{k: round(v, 4) for k, v in metrics_at(yt, pte, thr).items()},
                     "Bad loans caught": f"{cm[1, 1]} of {cm[1].sum()}",
                     "Good applicants refused": f"{cm[0, 1]} of {cm[0].sum()}"})
    table(pd.DataFrame(rows), "classification_thresholds", "Effect of the decision threshold (test set)")

    cm = confusion_matrix(yt, (pte >= 0.5).astype(int))
    table(pd.DataFrame(cm, columns=["Predicted good", "Predicted bad"])
          .assign(Actual=["Actual good", "Actual bad"])[["Actual", "Predicted good", "Predicted bad"]],
          "confusion_matrix", "Confusion matrix (test set, threshold 0.5)")

    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#cde2fb", "#0d366b"]))
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=18,
                    color="white" if cm[i, j] > cm.max() / 2 else INK)
    ax.set_xticks([0, 1], ["Predicted good", "Predicted bad"])
    ax.set_yticks([0, 1], ["Actual good", "Actual bad"])
    ax.set_title("Confusion matrix on the test set")
    ax.grid(False)
    save(fig, "fig08_confusion_matrix.png")

    fpr, tpr, _ = roc_curve(yt, pte)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="No-skill line")
    ax.plot(fpr, tpr, color=BLUE, label=f"Model (AUC = {roc_auc_score(yt, pte):.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve on the test set")
    ax.legend(loc="lower right")
    save(fig, "fig09_roc_curve.png")

    (ROOT / "results" / "results_summary.md").write_text(
        "# Results summary\n\nGenerated by run_all_experiments.py\n\n" + "\n".join(out))
    print("\nAll tables and figures regenerated.")


if __name__ == "__main__":
    main()
