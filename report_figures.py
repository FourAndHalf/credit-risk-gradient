"""
Figures for the report chapters (methodology diagrams and theory illustrations).

Usage (from the project root):
    python report_figures.py

Writes PNG files to results/figures/report/. Figures that use data are computed
from the same pipeline as run_all_experiments.py (seed 42, 80/20 split).
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from src.data_preparation import load_raw_data, prepare_data

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "figures" / "report"
OUT.mkdir(parents=True, exist_ok=True)

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BLUE_LIGHT, BLUE_DARK = "#cde2fb", "#104281"
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "legend.frameon": False,
    "axes.axisbelow": True,
})


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- diagrams
def box(ax, x, y, w, h, text, fill=BLUE_LIGHT, edge=BLUE, size=9, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fill, ec=edge, lw=1.2))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=INK,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, text=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=12, color=MUTED, lw=1.2))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, text, ha="center",
                fontsize=7.5, color=MUTED)


def blank(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def fig_research_design():
    fig, ax = blank(10, 3.4)
    xs = [1.1, 3.1, 5.1, 7.1, 9.0]
    labels = ["Optimisation and\nstatistical theory",
              "Testable\npredictions\nP1 to P5",
              "Curvature quantities\ncomputed from data\n(L, μ, κ, ρ)",
              "Controlled\nexperiments\nA to J",
              "Comparison:\nprediction vs\nobservation"]
    for x, t in zip(xs, labels):
        box(ax, x, 2.0, 1.75, 1.15, t)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, a + 0.9, 2.0, b - 0.9, 2.0)
    ax.text(2.1, 0.75, "Deduction", ha="center", color=MUTED, fontsize=9, style="italic")
    ax.text(6.1, 0.75, "Measurement and testing", ha="center", color=MUTED, fontsize=9, style="italic")
    ax.plot([0.25, 4.0], [1.0, 1.0], color=GRID, lw=2)
    ax.plot([4.2, 9.9], [1.0, 1.0], color=GRID, lw=2)
    ax.text(0.1, 3.15, "Research design: theory → prediction → measurement → test",
            fontsize=12, fontweight="bold")
    save(fig, "fig3_1_research_design.png")


def fig_data_pipeline(n_train, n_test, n_num, n_dummy):
    fig, ax = blank(11.6, 3.9)
    ax.text(0.1, 3.65, "Data preparation pipeline", fontsize=12, fontweight="bold")
    box(ax, 1.1, 1.85, 1.9, 1.3, "Raw data\n1,000 applicants\n20 predictors\n+ outcome")
    box(ax, 3.3, 1.85, 1.9, 1.3, "Recode outcome\ny = 1: bad credit\ny = 0: good credit")
    box(ax, 5.5, 1.85, 1.9, 1.3, f"Stratified split\n(seed 42)\nTrain {n_train} | Test {n_test}\n70/30 kept in both")
    box(ax, 7.95, 2.65, 2.3, 0.9, f"{n_num} numerical → standardise\n(training mean and SD)", size=8.5)
    box(ax, 7.95, 1.05, 2.3, 0.9, f"13 categorical → one-hot\n(drop reference) = {n_dummy} columns", size=8.5)
    box(ax, 10.55, 1.85, 1.8, 1.5, f"Design matrix\nX̃: {n_train} × {n_num + n_dummy + 1}\n\n{n_num + n_dummy} features\n+ intercept",
        fill="#e8f5ee", edge=AQUA, size=8.8, bold=True)
    arrow(ax, 2.05, 1.85, 2.35, 1.85)
    arrow(ax, 4.25, 1.85, 4.55, 1.85)
    arrow(ax, 6.45, 2.1, 6.8, 2.55)
    arrow(ax, 6.45, 1.6, 6.8, 1.15)
    arrow(ax, 9.1, 2.55, 9.65, 2.15)
    arrow(ax, 9.1, 1.15, 9.65, 1.55)
    save(fig, "fig3_2_data_pipeline.png")


def fig_algorithm():
    fig, ax = blank(6.6, 7.4)
    ax.set_ylim(1.0, 7.4)
    ax.text(0.1, 7.15, "Batch Gradient Descent (Algorithm 3.1)", fontsize=12, fontweight="bold")
    x = 2.6
    steps = [(6.35, "Start: θ₀ = 0, k = 0"),
             (5.35, "Predict: p = σ(X̃θₖ)\nfor all 800 applicants"),
             (4.25, "Compute loss J(θₖ) and\ngradient gₖ = ∇J(θₖ); record both"),
             (3.05, "Is ‖gₖ‖ < ε = 10⁻⁶ ?"),
             (1.75, "Update: θₖ₊₁ = θₖ − η gₖ\nk ← k + 1")]
    for y, t in steps:
        fill = "#fff4d6" if "?" in t else BLUE_LIGHT
        edge = YELLOW if "?" in t else BLUE
        box(ax, x, y, 3.6, 0.8, t, fill=fill, edge=edge, size=9)
    for (y1, _), (y2, _) in zip(steps[:-1], steps[1:]):
        arrow(ax, x, y1 - 0.42, x, y2 + 0.42)
    ax.text(x + 0.12, 2.33, "No", fontsize=8.5, color=MUTED)
    box(ax, 5.65, 3.05, 1.55, 0.8, "Stop:\nconverged", fill="#e8f5ee", edge=AQUA, size=9)
    arrow(ax, x + 1.82, 3.05, 4.85, 3.05)
    ax.text(4.62, 3.18, "Yes", fontsize=8.5, color=MUTED, ha="center")
    ax.plot([x - 1.8, x - 2.25, x - 2.25, x - 1.8], [1.75, 1.75, 5.35, 5.35], color=MUTED, lw=1.2)
    ax.text(0.18, 3.55, "repeat (at most 20,000 iterations)", fontsize=8, color=MUTED,
            rotation=90, ha="center", va="center")
    ax.annotate("", xy=(x - 1.8, 5.35), xytext=(x - 2.25, 5.35),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
    save(fig, "fig3_3_algorithm_flowchart.png")


def fig_experiments():
    fig, ax = blank(10, 4.6)
    ax.text(0.1, 4.35, "Experimental design: one factor varied at a time", fontsize=12, fontweight="bold")
    groups = [
        (2.6, "Optimisation experiments", BLUE_LIGHT, BLUE,
         ["A  Learning rate η", "B  Feature scaling", "C  Reference level of categories",
          "D  Regularisation strength λ", "E  Predicting iteration counts",
          "F  Newton, momentum, line search", "G  Verification vs libraries"]),
        (7.4, "Statistical and decision analysis", "#e8f5ee", AQUA,
         ["H  Standard errors, odds ratios, tests", "I  Performance, cross-validation,",
          "    bootstrap, 50 random splits", "J  Calibration and cost-based thresholds"]),
    ]
    for xc, title, fill, edge, items in groups:
        ax.add_patch(FancyBboxPatch((xc - 2.25, 0.25), 4.5, 3.75,
                                    boxstyle="round,pad=0.02,rounding_size=0.1", fc=fill, ec=edge, lw=1.2))
        ax.text(xc, 3.7, title, ha="center", fontweight="bold", fontsize=10)
        for i, t in enumerate(items):
            ax.text(xc - 2.0, 3.2 - i * 0.42, t, fontsize=9.2, color=INK)
    save(fig, "fig3_4_experimental_design.png")


def fig_split(y_train, y_test):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    groups = ["Training set (800)", "Test set (200)"]
    good = [int((y_train == 0).sum()), int((y_test == 0).sum())]
    bad = [int((y_train == 1).sum()), int((y_test == 1).sum())]
    xpos = np.arange(2)
    b1 = ax.bar(xpos - 0.18, good, 0.34, color=BLUE, label="Good credit (y = 0)")
    b2 = ax.bar(xpos + 0.18, bad, 0.34, color=ORANGE, label="Bad credit (y = 1)")
    for bars in (b1, b2):
        ax.bar_label(bars, padding=3, color=INK, fontsize=9)
    ax.set_xticks(xpos, groups)
    ax.set_ylabel("Number of applicants")
    ax.set_title("Stratified 80/20 split keeps the 70/30 class ratio")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(good) * 1.15)
    save(fig, "fig3_5_train_test_split.png")


# ---------------------------------------------------------- theory: 4.1
def fig_convexity():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.linspace(-2.5, 2.5, 400)
    f = np.log1p(np.exp(x)) - 0.3 * x
    ax = axes[0]
    ax.plot(x, f, color=BLUE, label="Convex function f")
    x0 = 0.8
    f0 = np.log1p(np.exp(x0)) - 0.3 * x0
    g0 = 1 / (1 + np.exp(-x0)) - 0.3
    ax.plot(x, f0 + g0 * (x - x0), color=MUTED, ls="--", lw=1.3, label="Tangent line (below f)")
    a, b = -1.8, 1.9
    fa, fb = np.log1p(np.exp(a)) - 0.3 * a, np.log1p(np.exp(b)) - 0.3 * b
    ax.plot([a, b], [fa, fb], color=ORANGE, lw=1.6, label="Chord (above f)")
    ax.scatter([x0], [f0], color=INK, s=18, zorder=3)
    ax.set_title("Convexity: tangent below, chord above")
    ax.set_xlabel("θ")
    ax.set_ylabel("f(θ)")
    ax.legend(loc="upper center", fontsize=8.5)
    ax.set_ylim(0.0, 2.6)
    ax = axes[1]
    Lc = 0.25
    ax.plot(x, f, color=BLUE, label="f")
    ax.plot(x, f0 + g0 * (x - x0) + Lc / 2 * (x - x0) ** 2, color=ORANGE, lw=1.6,
            label="Quadratic upper bound (descent lemma)")
    ax.plot(x, f0 + g0 * (x - x0), color=MUTED, ls="--", lw=1.3, label="Tangent line")
    ax.scatter([x0], [f0], color=INK, s=18, zorder=3)
    ax.set_title("Smoothness: f below a quadratic bowl")
    ax.set_xlabel("θ")
    ax.legend(loc="upper center", fontsize=8.5)
    ax.set_ylim(0.0, 2.6)
    save(fig, "fig4_1_convexity_smoothness.png")


def gd_path(A, x0, eta, n):
    pts = [x0]
    x = x0.copy()
    for _ in range(n):
        x = x - eta * A @ x
        pts.append(x.copy())
    return np.array(pts)


def fig_conditioning_contours():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    g = np.linspace(-3, 3, 300)
    Xg, Yg = np.meshgrid(g, g)
    for ax, (l1, l2), title in [(axes[0], (1.0, 0.7), "Well conditioned (κ ≈ 1.4)"),
                                (axes[1], (1.0, 0.05), "Ill conditioned (κ = 20)")]:
        c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
        Q = np.array([[c, -s], [s, c]])
        A = Q @ np.diag([l1, l2]) @ Q.T
        Z = 0.5 * (A[0, 0] * Xg ** 2 + 2 * A[0, 1] * Xg * Yg + A[1, 1] * Yg ** 2)
        ax.contour(Xg, Yg, Z, levels=np.geomspace(0.01, 6, 12), colors=[GRID], linewidths=1)
        path = gd_path(A, np.array([-2.6, 1.9]), 1.8 / l1, 25)
        ax.plot(path[:, 0], path[:, 1], color=ORANGE, lw=1.4, marker="o", ms=3.5)
        ax.scatter([0], [0], color=INK, marker="*", s=90, zorder=4)
        ax.set_aspect("equal")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_title(title)
        ax.set_xlabel("θ₁")
        ax.set_ylabel("θ₂")
        ax.grid(False)
    axes[1].text(-2.9, -2.8, "25 steps, η = 1.8/L: zig-zag across the valley,\nslow progress along it",
                 fontsize=8.5, color=MUTED)
    axes[0].text(-2.9, -2.8, "25 steps, η = 1.8/L: reaches the minimum quickly", fontsize=8.5, color=MUTED)
    save(fig, "fig4_2_conditioning_contours.png")


# ---------------------------------------------------------- theory: 4.2
def fig_sigmoid():
    z = np.linspace(-8, 8, 500)
    s = 1 / (1 + np.exp(-z))
    d1 = s * (1 - s)
    d2 = d1 * (1 - 2 * s)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    ax.plot(z, s, color=BLUE)
    ax.axhline(0.5, color=MUTED, lw=1, ls="--")
    ax.set_title("Sigmoid σ(z)")
    ax.set_xlabel("Linear score z")
    ax.set_ylabel("Probability of bad credit")
    ax = axes[1]
    ax.plot(z, d1, color=BLUE, label="σ′(z) = σ(1 − σ)")
    ax.plot(z, d2, color=ORANGE, label="σ″(z)")
    ax.axhline(0.25, color=MUTED, lw=1, ls="--")
    ax.text(-7.8, 0.233, "max σ′ = 1/4 at z = 0", color=MUTED, fontsize=8.5)
    m = 1 / (6 * np.sqrt(3))
    ax.axhline(m, color=MUTED, lw=0.8, ls=":")
    ax.axhline(-m, color=MUTED, lw=0.8, ls=":")
    ax.text(3.2, m + 0.008, "|σ″| ≤ 1/(6√3) ≈ 0.096", color=MUTED, fontsize=8.5)
    ax.set_title("Derivatives of the sigmoid")
    ax.set_xlabel("Linear score z")
    ax.legend(loc="center right", fontsize=8.5, bbox_to_anchor=(1.0, 0.33))
    save(fig, "fig4_3_sigmoid.png")


def fig_loss_terms():
    z = np.linspace(-6, 6, 400)
    sp = np.log1p(np.exp(z))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(z, sp - z, color=ORANGE, label="Bad credit (y = 1): ln(1 + e⁻ᶻ)")
    ax.plot(z, sp, color=BLUE, label="Good credit (y = 0): ln(1 + eᶻ)")
    ax.set_xlabel("Linear score z = x̃ᵀθ  (higher = model predicts more risk)")
    ax.set_ylabel("Loss contribution ℓᵢ")
    ax.set_title("Each applicant's contribution to the loss")
    ax.legend(loc="upper center")
    ax.annotate("Bad credit given a low score\n(confident and wrong):\nloss grows linearly",
                xy=(-5.0, 5.0), xytext=(-5.9, 1.4), fontsize=8.5, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax.text(2.6, 0.55, "Confident and right:\nloss near zero", fontsize=8.5, color=MUTED)
    save(fig, "fig4_4_loss_per_applicant.png")


def newton_fit(Xa, y, iters=50):
    t = np.zeros(Xa.shape[1])
    m = len(y)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xa @ t))
        g = Xa.T @ (p - y) / m
        if np.linalg.norm(g) < 1e-12:
            break
        H = (Xa * (p * (1 - p))[:, None]).T @ Xa / m
        t = t - np.linalg.solve(H, g)
    return t


def fig_curvature_weights(Xa, theta):
    p = 1 / (1 + np.exp(-Xa @ theta))
    w = p * (1 - p)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(w, bins=np.linspace(0, 0.25, 26), color=BLUE, edgecolor=SURFACE, linewidth=1.5,
            label="At the optimum θ*")
    ax.axvline(0.25, color=ORANGE, lw=2, label="At the start θ = 0 (all 800 at 1/4)")
    ax.set_xlabel("Curvature weight pᵢ(1 − pᵢ)")
    ax.set_ylabel("Number of training applicants")
    ax.set_title("Curvature weights shrink as the model learns")
    ax.legend(loc="upper left")
    ax.text(0.01, ax.get_ylim()[1] * 0.62, f"Mean at θ*: {w.mean():.3f}\n(vs 0.250 at the start)",
            fontsize=9, color=MUTED)
    save(fig, "fig4_5_curvature_weights.png")
    return w


def fig_spectrum_start_vs_opt(Xa, theta):
    m = Xa.shape[0]
    H0 = Xa.T @ Xa / (4 * m)
    p = 1 / (1 + np.exp(-Xa @ theta))
    Hs = (Xa * (p * (1 - p))[:, None]).T @ Xa / m
    e0 = np.sort(np.linalg.eigvalsh(H0))[::-1]
    es = np.sort(np.linalg.eigvalsh(Hs))[::-1]
    idx = np.arange(1, len(e0) + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(idx, e0, color=ORANGE, marker="o", ms=3.5, label=f"Start H(0): L = {e0[0]:.4f}")
    ax.plot(idx, es, color=BLUE, marker="s", ms=3.5,
            label=f"Optimum H*: λmax = {es[0]:.4f}, μ = {es[-1]:.2e}")
    ax.set_yscale("log")
    ax.set_xlabel("Eigenvalue rank (largest first)")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_title("Hessian eigenvalues at the start and at the optimum")
    ax.legend(loc="lower left")
    save(fig, "fig4_6_spectrum_start_vs_optimum.png")
    return e0, es, Hs


# ---------------------------------------------------------- theory: 4.3
def fig_contraction_regimes():
    t = np.linspace(0, 2.6, 400)
    r = np.abs(1 - t)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.axvspan(0, 1, color="#e8f5ee", lw=0)
    ax.axvspan(1, 2, color="#fff4d6", lw=0)
    ax.axvspan(2, 2.6, color="#fde3d8", lw=0)
    ax.plot(t, r, color=BLUE)
    ax.axhline(1, color=MUTED, lw=1, ls="--")
    ax.text(0.5, 1.45, "Shrinks steadily", ha="center", fontsize=9)
    ax.text(1.5, 1.45, "Shrinks, flips sign\n(zig-zag)", ha="center", fontsize=9)
    ax.text(2.3, 1.45, "Grows:\ndiverges", ha="center", fontsize=9)
    ax.set_xlim(0, 2.6)
    ax.set_ylim(0, 1.75)
    ax.set_xlabel("ηλᵢ  (step size × curvature in that direction)")
    ax.set_ylabel("|1 − ηλᵢ|  (factor per step)")
    ax.set_title("What one Gradient Descent step does to each eigen-direction")
    ax.grid(False)
    save(fig, "fig4_7_contraction_regimes.png")


def fig_decrease_parabola(L):
    eta = np.linspace(0, 2 / L * 1.08, 300)
    dec = eta * (1 - L * eta / 2)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(eta, dec, color=BLUE)
    ax.axhline(0, color=MUTED, lw=1)
    ax.axvline(1 / L, color=MUTED, lw=1, ls="--")
    ax.axvline(2 / L, color=ORANGE, lw=1.5, ls="--")
    ax.text(1 / L + 0.02, dec.max() * 0.15, f"1/L = {1 / L:.3f}\nlargest guaranteed\ndecrease", fontsize=8.5,
            color=MUTED)
    ax.text(2 / L + 0.02, dec.max() * 0.6, f"2/L = {2 / L:.3f}\nguarantee\nends", fontsize=8.5, color=ORANGE)
    ax.set_xlabel("Step size η")
    ax.set_ylabel("Guaranteed decrease per unit ‖g‖²")
    ax.set_title("Sufficient decrease: η(1 − Lη/2)")
    save(fig, "fig4_8_sufficient_decrease.png")


def fig_stepsize_map(L, lmax_star):
    fig, ax = plt.subplots(figsize=(10, 2.6))
    g, loc = 2 / L, 2 / lmax_star
    ax.axvspan(0, g, color="#e8f5ee", lw=0)
    ax.axvspan(g, loc, color="#fff4d6", lw=0)
    ax.axvspan(loc, 4.3, color="#fde3d8", lw=0)
    ax.axvline(g, color=AQUA, lw=2)
    ax.axvline(loc, color=ORANGE, lw=2)
    tested = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 2.5, 3.0, 4.0]
    ax.scatter(tested, [0.5] * len(tested), color=INK, s=22, zorder=3)
    for t in [0.5, 1.0, 2.0, 2.5, 3.0, 4.0]:
        ax.text(t, 0.62, f"{t}", ha="center", fontsize=8.5)
    ax.text(0.05, 0.9, "Global guarantee:\nloss falls every step", fontsize=9, va="top")
    ax.text(g + 0.05, 0.9, "Locally stable only:\nmay or may not converge", fontsize=9, va="top")
    ax.text(loc + 0.05, 0.9, "Optimum unstable:\ncannot converge", fontsize=9, va="top")
    ax.text(g, 0.06, f" 2/L = {g:.3f}", color=AQUA, fontsize=9, fontweight="bold")
    ax.text(loc, 0.06, f" 2/λmax(H*) = {loc:.3f}", color=ORANGE, fontsize=9, fontweight="bold")
    ax.set_xlim(0, 4.3)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Learning rate η  (dots: values tested in Experiment A)")
    ax.set_title("Two step-size limits for this study's data")
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    save(fig, "fig4_9_stepsize_limits.png")


def fig_modal_components(Hs, theta, eta=1.0, tol=1e-6):
    ev, V = np.linalg.eigh(Hs)
    c = V.T @ (-theta)
    ks = np.unique(np.geomspace(1, 30000, 400).astype(int))
    comp = np.abs(ev[:, None] * c[:, None] * (1 - eta * ev[:, None]) ** ks[None, :])
    total = np.sqrt((comp ** 2).sum(axis=0))
    slow = comp[0]
    fast = np.sqrt((comp[1:] ** 2).sum(axis=0))
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(ks, total, color=INK, lw=2, label="Predicted gradient norm (all directions)")
    ax.plot(ks, fast, color=BLUE, lw=1.6, label="All other directions combined")
    ax.plot(ks, slow, color=ORANGE, lw=1.6, label=f"Flattest direction only (μ = {ev[0]:.2e})")
    ax.axhline(tol, color=MUTED, lw=1, ls="--")
    ax.text(1.2, tol * 1.6, "stopping tolerance 10⁻⁶", fontsize=8.5, color=MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration k (log scale)")
    ax.set_ylabel("Gradient norm (log scale)")
    ax.set_title("Modal prediction for η = 1: the flattest direction takes over late")
    ax.legend(loc="lower left", fontsize=8.5)
    save(fig, "fig4_10_modal_components.png")


def main():
    df = load_raw_data(ROOT / "data" / "raw")
    d = prepare_data(df, scale=True)
    X, y = d["X_train"], d["y_train"]
    Xa = np.hstack([X, np.ones((X.shape[0], 1))])
    n_num = 7
    n_dummy = X.shape[1] - n_num

    fig_research_design()
    fig_data_pipeline(len(y), len(d["y_test"]), n_num, n_dummy)
    fig_algorithm()
    fig_experiments()
    fig_split(y, d["y_test"])

    fig_convexity()
    fig_conditioning_contours()

    fig_sigmoid()
    fig_loss_terms()
    theta = newton_fit(Xa, y)
    fig_curvature_weights(Xa, theta)
    e0, es, Hs = fig_spectrum_start_vs_opt(Xa, theta)

    fig_contraction_regimes()
    fig_decrease_parabola(e0[0])
    fig_stepsize_map(e0[0], es[0])
    fig_modal_components(Hs, theta)
    print(f"L={e0[0]:.4f} lmax*={es[0]:.4f} mu*={es[-1]:.3e}")


if __name__ == "__main__":
    main()
