"""
Extended experiments (C, E, F, H, I, J and the exploratory analysis).

Usage (from the project root):
    python run_extended_experiments.py

Writes CSV tables to results/tables/extended/ and figures to
results/figures/extended/. Uses the same data pipeline, split (seed 42),
start point (zero), tolerance (1e-6) and iteration budget as
run_all_experiments.py. The reference optimum is computed with Newton's method.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_preparation import (CATEGORICAL_COLS, NUMERICAL_COLS, TARGET_COL,
                                  load_raw_data, prepare_data)

ROOT = Path(__file__).parent
FIG = ROOT / "results" / "figures" / "extended"
TAB = ROOT / "results" / "tables" / "extended"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

TOL, MAX_ITER, SEED = 1e-6, 20_000, 42

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
PURPLE, RED = "#8a5cd1", "#d03b3b"
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

ENGLISH = {
    "laufkont": "status", "laufzeit": "duration", "moral": "credit_history",
    "verw": "purpose", "hoehe": "amount", "sparkont": "savings",
    "beszeit": "employment_duration", "rate": "installment_rate",
    "famges": "personal_status_sex", "buerge": "other_debtors",
    "wohnzeit": "present_residence", "verm": "property", "alter": "age",
    "weitkred": "other_installment_plans", "wohn": "housing",
    "bishkred": "number_credits", "beruf": "job", "pers": "people_liable",
    "telef": "telephone", "gastarb": "foreign_worker",
}


def nice(feature):
    """'cat__laufkont_4' -> 'status = 4'; 'num__hoehe' -> 'amount (per SD)'."""
    if feature == "intercept":
        return "intercept"
    kind, rest = feature.split("__", 1)
    if kind == "num":
        return f"{ENGLISH.get(rest, rest)} (per SD)"
    var, level = rest.rsplit("_", 1)
    return f"{ENGLISH.get(var, var)} = {level}"


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", name)


def table(df, name):
    df.to_csv(TAB / f"{name}.csv", index=False)
    print(f"\n[{name}]\n{df.to_string(index=False)}")


sig = lambda z: 1 / (1 + np.exp(-z))


# ------------------------------------------------------------------ core maths
def loss(A, y, th, lam=0.0):
    z = A @ th
    return np.mean(np.logaddexp(0, z) - y * z) + lam / (2 * len(y)) * np.sum(th[:-1] ** 2)


def grad(A, y, th, lam=0.0):
    m = len(y)
    g = A.T @ (sig(A @ th) - y) / m
    r = lam / m * th
    r[-1] = 0.0
    return g + r


def hess(A, th, lam=0.0):
    m = A.shape[0]
    q = sig(A @ th)
    R = np.eye(A.shape[1]) * lam / m
    R[-1, -1] = 0.0
    return (A * (q * (1 - q))[:, None]).T @ A / m + R


def newton(A, y, lam=0.0, tol=1e-12, maxit=100, record=False):
    th = np.zeros(A.shape[1])
    hist = []
    for k in range(maxit):
        g = grad(A, y, th, lam)
        hist.append(np.linalg.norm(g))
        if hist[-1] < tol:
            break
        th = th - np.linalg.solve(hess(A, th, lam), g)
    return (th, hist) if record else th


def gd(A, y, eta, tol=TOL, maxit=MAX_ITER, record=True):
    th = np.zeros(A.shape[1])
    hist = []
    for k in range(maxit):
        g = grad(A, y, th)
        n = np.linalg.norm(g)
        hist.append(n)
        if n < tol:
            return th, hist, k + 1, True
        th = th - eta * g
    return th, hist, maxit, False


def augment(X):
    return np.hstack([X, np.ones((X.shape[0], 1))])


def main():
    df = load_raw_data(ROOT / "data" / "raw")
    d = prepare_data(df, scale=True)
    X, Xt, y, yt = d["X_train"], d["X_test"], d["y_train"], d["y_test"]
    names = list(d["feature_names"]) + ["intercept"]
    A, At = augment(X), augment(Xt)
    m, n1 = A.shape

    th_star = newton(A, y)
    H0, Hs = hess(A, np.zeros(n1)), hess(A, th_star)
    L = np.linalg.eigvalsh(H0)[-1]
    ev, V = np.linalg.eigh(Hs)
    mu, Lstar = ev[0], ev[-1]
    print(f"L={L:.4f} 2/L={2 / L:.3f} lmax*={Lstar:.4f} 2/lmax*={2 / Lstar:.3f} mu={mu:.3e} kappa*={Lstar / mu:.1f}")

    # ============================================================ EDA
    yb = df[TARGET_COL].values
    rows = []
    work = df.copy()
    for c in CATEGORICAL_COLS + ["laufzeit", "hoehe", "alter"]:
        col = c
        if c in ("laufzeit", "hoehe", "alter"):
            col = c + "_q"
            work[col] = pd.qcut(work[c], 5, duplicates="drop")
        t = pd.crosstab(work[col], yb)
        good, bad = t[0] / t[0].sum(), t[1] / t[1].sum()
        iv = float(((good - bad) * np.log(good / bad)).sum())
        chi2, pval = stats.chi2_contingency(t)[:2]
        rows.append({"Variable": ENGLISH[c] + (" (quintiles)" if c in ("laufzeit", "hoehe", "alter") else ""),
                     "Information value": round(iv, 3),
                     "Chi-square p-value": f"{pval:.1e}"})
    ivt = pd.DataFrame(rows).sort_values("Information value", ascending=False)
    table(ivt, "eda_information_value")

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    s = ivt.iloc[::-1]
    colors = [BLUE if v >= 0.1 else "#9cc3ef" if v >= 0.02 else GRID for v in s["Information value"]]
    ax.barh(s["Variable"], s["Information value"], color=colors)
    for v in (0.02, 0.1, 0.3):
        ax.axvline(v, color=MUTED, lw=0.8, ls=":")
    ax.text(0.103, -0.9, "medium", fontsize=8, color=MUTED)
    ax.text(0.303, -0.9, "strong", fontsize=8, color=MUTED)
    ax.set_xlabel("Information value (separation of good vs bad credit)")
    ax.set_title("Which variables separate good from bad credit?")
    ax.grid(axis="y", visible=False)
    save(fig, "eda_information_value.png")

    br = df.groupby("laufkont")[TARGET_COL].agg(["mean", "size"]).reset_index()
    table(br.rename(columns={"laufkont": "Checking-account status code", "mean": "Bad-credit rate", "size": "Applicants"})
          .round(3), "eda_badrate_status")
    fig, ax = plt.subplots(figsize=(6.2, 4))
    bars = ax.bar(br["laufkont"].astype(str), 100 * br["mean"], color=BLUE, width=0.55)
    ax.bar_label(bars, labels=[f"{100 * v:.0f}%\n(n={s})" for v, s in zip(br["mean"], br["size"])], padding=3, fontsize=9)
    ax.axhline(30, color=MUTED, lw=1, ls="--")
    ax.text(3.45, 31.5, "overall 30%", fontsize=8.5, color=MUTED, ha="right")
    ax.set_ylim(0, 65)
    ax.set_xlabel("Checking-account status (dataset code)")
    ax.set_ylabel("Bad-credit rate (%)")
    ax.set_title("Bad-credit rate by checking-account status")
    ax.grid(axis="x", visible=False)
    save(fig, "eda_badrate_status.png")

    # ============================================================ E: modal prediction
    c = V.T @ (-th_star)
    g0 = np.linalg.norm(grad(A, y, np.zeros(n1)))
    rows = []
    obs = {}
    for eta in (1.0, 2.0, 2.5):
        _, hist, k, conv = gd(A, y, eta)
        obs[eta] = (hist, k)
        # vectorised search for the modal prediction
        kgrid = np.arange(1, 60000)
        gn = np.sqrt(((ev[:, None] * c[:, None]) ** 2 * (1 - eta * ev[:, None]) ** (2 * kgrid[None, :])).sum(axis=0))
        pred = int(kgrid[np.argmax(gn < TOL)])
        rho = max(abs(1 - eta * ev[0]), abs(1 - eta * ev[-1]))
        worst = np.log(g0 / TOL) / -np.log(rho)
        rows.append({"Learning rate": eta, "Observed iterations": k, "Modal prediction": pred,
                     "Modal error (%)": round(100 * (pred - k) / k, 1),
                     "Worst-case prediction": int(round(worst)),
                     "Worst-case / observed": round(worst / k, 2)})
    modal = pd.DataFrame(rows)
    table(modal, "expE_iteration_prediction")

    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    x = np.arange(3)
    w = 0.26
    b1 = ax.bar(x - w, modal["Observed iterations"], w, color=INK, label="Observed")
    b2 = ax.bar(x, modal["Modal prediction"], w, color=BLUE, label="Eigenvalue (modal) prediction")
    b3 = ax.bar(x + w, modal["Worst-case prediction"], w, color=GRID, edgecolor=MUTED, label="Worst-case bound")
    for b in (b1, b2, b3):
        ax.bar_label(b, labels=[f"{v:,.0f}" for v in b.datavalues], padding=2, fontsize=8)
    ax.set_xticks(x, [f"η = {e}" for e in modal["Learning rate"]])
    ax.set_ylabel("Iterations to ‖∇J‖ < 10⁻⁶")
    ax.set_title("Eigenvalues predict the iteration count to within 3%")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    save(fig, "expE_iteration_prediction.png")

    # tolerance sweep (eta = 1)
    rows = []
    th = np.zeros(n1)
    tols = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    ti = 0
    for k in range(60000):
        gn = np.linalg.norm(grad(A, y, th))
        while ti < len(tols) and gn < tols[ti]:
            rows.append({"Tolerance": tols[ti], "Iterations": k + 1,
                         "Distance to optimum": float(np.linalg.norm(th - th_star)),
                         "Certified bound ε/μ": tols[ti] / mu,
                         "Loss gap": float(loss(A, y, th) - loss(A, y, th_star)),
                         "Certified bound ε²/(2μ)": tols[ti] ** 2 / (2 * mu)})
            ti += 1
        if ti == len(tols):
            break
        th = th - 1.0 * grad(A, y, th)
    tol_t = pd.DataFrame(rows)
    tol_t["Extra iterations"] = tol_t["Iterations"].diff()
    out = tol_t.copy()
    for col in ["Distance to optimum", "Certified bound ε/μ", "Loss gap", "Certified bound ε²/(2μ)"]:
        out[col] = out[col].map(lambda v: f"{v:.2e}")
    table(out, "expE_tolerance_sweep")

    # curvature along each run's own path, compared with its stability threshold 2/eta
    rec_rows = []
    for eta in (2.0, 2.5, 3.0):
        th = np.zeros(n1)
        lm = []
        for k in range(3001):
            if k in (0, 1) or k >= 2900:
                lm.append(np.linalg.eigvalsh(hess(A, th))[-1])
            th = th - eta * grad(A, y, th)
        rec_rows.append({"Learning rate": eta, "Threshold 2/η": round(2 / eta, 3),
                         "λmax at start": round(lm[0], 4), "λmax after 1 step": round(lm[1], 4),
                         "λmax late in run (max of last 100 its)": round(max(lm[2:]), 4),
                         "Stable near the end?": "yes" if max(lm[2:]) < 2 / eta else "no"})
    cur = pd.DataFrame(rec_rows)
    table(cur, "expE_curvature_along_path")
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    xs = np.arange(3)
    late = cur["λmax late in run (max of last 100 its)"].values
    thr = cur["Threshold 2/η"].values
    colors = [AQUA if l < t else ORANGE for l, t in zip(late, thr)]
    bars = ax.bar(xs, late, 0.5, color=colors)
    ax.bar_label(bars, labels=[f"λmax\n{v:.3f}" for v in late], label_type="center", color="white", fontsize=10, fontweight="bold")
    for x_, t in zip(xs, thr):
        ax.hlines(t, x_ - 0.38, x_ + 0.38, color=INK, lw=2.2)
        ax.text(x_ + 0.4, t, f"2/η = {t:.2f}", va="center", fontsize=9)
    ax.axhline(L, color=MUTED, lw=1, ls="--")
    ax.text(2.62, L + 0.02, f"curvature at the start L = {L:.3f}", ha="right", fontsize=8.5, color=MUTED)
    ax.set_xticks(xs, [f"η = {e}\n{'converges' if c == AQUA else 'oscillates'}" for e, c in zip(cur["Learning rate"], colors)])
    ax.set_ylim(0, 1.55)
    ax.set_xlim(-0.6, 2.9)
    ax.set_ylabel("Largest Hessian eigenvalue")
    ax.set_title("Late in the run, stability needs λmax below 2/η")
    ax.grid(axis="x", visible=False)
    save(fig, "expE_curvature_along_path.png")

    # ============================================================ C: reference level
    Xd = df.drop(columns=[TARGET_COL])
    yy = df[TARGET_COL].values.astype(float)
    Xtr, Xte, ytr, yte = train_test_split(Xd, yy, test_size=0.2, random_state=SEED, stratify=yy)

    def encode(drop):
        ct = ColumnTransformer([("num", StandardScaler(), NUMERICAL_COLS),
                                ("cat", OneHotEncoder(drop=drop, sparse_output=False), CATEGORICAL_COLS)])
        return augment(ct.fit_transform(Xtr)), augment(ct.transform(Xte))

    mf = [Xtr[cn].value_counts().idxmax() for cn in CATEGORICAL_COLS]
    rows, probs = [], {}
    for label, drop in [("First level (default)", "first"), ("Most frequent level", mf)]:
        Aa, Ate = encode(drop)
        ts = newton(Aa, ytr)
        e = np.linalg.eigvalsh(hess(Aa, ts))
        L0 = np.linalg.eigvalsh(hess(Aa, np.zeros(Aa.shape[1])))[-1]
        _, _, k, conv = gd(Aa, ytr, 1.0)
        probs[label] = sig(Ate @ ts)
        rows.append({"Reference level": label, "L": round(L0, 4), "2/L": round(2 / L0, 3),
                     "μ = λmin(H*)": f"{e[0]:.2e}", "κ*": round(e[-1] / e[0], 1),
                     "GD iterations (η = 1)": k,
                     "Log-likelihood": round(-len(ytr) * loss(Aa, ytr, ts), 3)})
    ref = pd.DataFrame(rows)
    ref["Max test-probability difference"] = ["—", f"{np.max(np.abs(probs['First level (default)'] - probs['Most frequent level'])):.1e}"]
    table(ref, "expC_reference_level")

    load_idx = np.argsort(-np.abs(V[:, 0]))[:6]
    vm = pd.DataFrame({"Parameter": [nice(names[i]) for i in load_idx], "Loading": V[load_idx, 0].round(3)})
    table(vm, "expC_flattest_direction")
    job_counts = df["beruf"].value_counts().sort_index()
    table(pd.DataFrame({"job code": job_counts.index, "Applicants": job_counts.values}), "expC_job_levels")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    ax = axes[0]
    vm_plot = vm.iloc[::-1]
    ax.barh(vm_plot["Parameter"], vm_plot["Loading"].abs(), color=[ORANGE if "job" in p or p == "intercept" else BLUE for p in vm_plot["Parameter"]])
    ax.set_xlabel("|Loading| on the flattest eigenvector")
    ax.set_title("The flattest direction: intercept + job dummies")
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    kap = ref["κ*"].values
    its = ref["GD iterations (η = 1)"].values
    xs = np.arange(2)
    b = ax.bar(xs, kap, 0.5, color=[ORANGE, AQUA], label="Condition number κ*")
    ax.bar_label(b, labels=[f"{v:,.0f}" for v in kap], padding=2, fontsize=9)
    ax.set_xticks(xs, ["Rarest level as\nreference (default)", "Most frequent level\nas reference"])
    ax.set_ylabel("κ* at the optimum")
    ax.set_title("Same model, κ falls by 69%")
    ax.grid(axis="x", visible=False)
    ax.text(0.98, 0.96, f"GD iterations (η = 1):\n{its[0]:,} → {its[1]:,}\nProbabilities identical\n(difference {ref.iloc[1, -1]})",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color=INK)
    save(fig, "expC_reference_level.png")

    # ============================================================ F: methods
    traces, rows = {}, {}
    t0 = time.perf_counter()
    _, h, k, _ = gd(A, y, 1.0)
    rows["Gradient Descent, η = 1"] = (k, time.perf_counter() - t0)
    traces["Gradient Descent, η = 1"] = h
    t0 = time.perf_counter()
    _, h, k, _ = gd(A, y, 2.5)
    rows["Gradient Descent, η = 2.5"] = (k, time.perf_counter() - t0)
    traces["Gradient Descent, η = 2.5"] = h

    def momentum(eta, beta, nesterov):
        th = np.zeros(n1)
        v = np.zeros(n1)
        hist = []
        for k in range(MAX_ITER * 3):
            gn = np.linalg.norm(grad(A, y, th))
            hist.append(gn)
            if gn < TOL:
                return hist, k + 1
            gg = grad(A, y, th + beta * v) if nesterov else grad(A, y, th)
            v = beta * v - eta * gg
            th = th + v
        return hist, k

    kap = Lstar / mu
    t0 = time.perf_counter()
    h, k = momentum(4 / (np.sqrt(Lstar) + np.sqrt(mu)) ** 2, ((np.sqrt(kap) - 1) / (np.sqrt(kap) + 1)) ** 2, False)
    rows["Heavy-ball (theory-optimal η, β)"] = (k, time.perf_counter() - t0)
    traces["Heavy-ball (theory-optimal η, β)"] = h
    t0 = time.perf_counter()
    h, k = momentum(1 / Lstar, (np.sqrt(kap) - 1) / (np.sqrt(kap) + 1), True)
    rows["Nesterov (η = 1/λmax(H*))"] = (k, time.perf_counter() - t0)
    traces["Nesterov (η = 1/λmax(H*))"] = h

    def armijo(eta0=4.0, cc=1e-4, tau=0.5):
        th = np.zeros(n1)
        hist = []
        for k in range(MAX_ITER * 3):
            g = grad(A, y, th)
            gn = np.linalg.norm(g)
            hist.append(gn)
            if gn < TOL:
                return hist, k + 1
            eta, f = eta0, loss(A, y, th)
            while loss(A, y, th - eta * g) > f - cc * eta * g @ g:
                eta *= tau
            th = th - eta * g
        return hist, k

    t0 = time.perf_counter()
    h, k = armijo()
    rows["Armijo line search"] = (k, time.perf_counter() - t0)
    traces["Armijo line search"] = h

    t0 = time.perf_counter()
    _, h = newton(A, y, tol=TOL, record=True)
    rows["Newton / IRLS"] = (len(h) - 1, time.perf_counter() - t0)
    traces["Newton / IRLS"] = h
    base = rows["Gradient Descent, η = 1"][0]
    meth = pd.DataFrame([{"Method": k_, "Iterations": v[0], "Speed-up vs GD (η = 1)": round(base / max(v[0], 1), 1),
                          "Time (ms)": round(1000 * v[1], 1)} for k_, v in rows.items()])
    table(meth, "expF_methods")
    table(pd.DataFrame({"Newton iteration": range(len(traces["Newton / IRLS"])),
                        "Gradient norm": [f"{v:.2e}" for v in traces["Newton / IRLS"]]}), "expF_newton_trace")

    fig, ax = plt.subplots(figsize=(8, 4.6))
    style = {"Gradient Descent, η = 1": (INK, "-"), "Gradient Descent, η = 2.5": (MUTED, "-"),
             "Armijo line search": (YELLOW, "-"), "Heavy-ball (theory-optimal η, β)": (BLUE, "-"),
             "Nesterov (η = 1/λmax(H*))": (PURPLE, "-"), "Newton / IRLS": (AQUA, "-")}
    for k_, h in traces.items():
        colr, ls = style[k_]
        ax.plot(np.arange(1, len(h) + 1), h, color=colr, ls=ls, lw=1.8 if "Newton" not in k_ else 2.5,
                marker="o" if "Newton" in k_ else None, ms=4, label=f"{k_}: {rows[k_][0]:,} it.")
    ax.axhline(TOL, color=MUTED, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration (log scale)")
    ax.set_ylabel("Gradient norm (log scale)")
    ax.set_title("Same problem, same answer: very different numbers of iterations")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5)
    save(fig, "expF_methods.png")

    # ============================================================ H: inference
    cov = np.linalg.inv(m * Hs)
    se = np.sqrt(np.diag(cov))
    z = th_star / se
    pv = 2 * stats.norm.sf(np.abs(z))
    coef = pd.DataFrame({"Parameter": [nice(nm) for nm in names], "Coefficient": th_star, "SE": se, "z": z,
                         "p-value": pv, "Odds ratio": np.exp(th_star),
                         "OR 95% low": np.exp(th_star - 1.96 * se), "OR 95% high": np.exp(th_star + 1.96 * se)})
    smr = sm.Logit(y, A).fit(disp=0, method="newton", maxiter=100)
    check = {"max |coef − statsmodels|": float(np.max(np.abs(smr.params - th_star))),
             "max |SE − statsmodels|": float(np.max(np.abs(smr.bse - se)))}
    out = coef.copy()
    for col in ["Coefficient", "SE", "z", "Odds ratio", "OR 95% low", "OR 95% high"]:
        out[col] = out[col].round(3)
    out["p-value"] = out["p-value"].map(lambda v: f"{v:.2g}")
    table(out, "expH_coefficients")
    llf, ll0 = smr.llf, smr.llnull
    lr_rows = [{"Test": "All predictors (model vs intercept only)", "LR statistic": round(2 * (llf - ll0), 1),
                "df": n1 - 1, "p-value": f"{stats.chi2.sf(2 * (llf - ll0), n1 - 1):.1e}"}]
    for var in CATEGORICAL_COLS:
        idx = [i for i, nm in enumerate(names) if nm.startswith(f"cat__{var}_")]
        keep = [i for i in range(n1) if i not in idx]
        th_r = newton(A[:, keep], y)
        llr = -m * loss(A[:, keep], y, th_r)
        stat = 2 * (llf - llr)
        lr_rows.append({"Test": f"Drop {ENGLISH[var]}", "LR statistic": round(stat, 2), "df": len(idx),
                        "p-value": f"{stats.chi2.sf(stat, len(idx)):.1e}"})
    lr = pd.DataFrame(lr_rows)
    table(lr, "expH_likelihood_ratio")
    var_dir = pd.DataFrame({"Direction": ["Flattest eigenvector (μ)", "Steepest eigenvector (λmax)"],
                            "Eigenvalue": [f"{mu:.2e}", f"{Lstar:.3f}"],
                            "Variance of estimate along it, 1/(mλ)": [f"{1 / (m * mu):.3f}", f"{1 / (m * Lstar):.2e}"],
                            "Standard deviation": [f"{np.sqrt(1 / (m * mu)):.3f}", f"{np.sqrt(1 / (m * Lstar)):.4f}"]})
    table(var_dir, "expH_variance_by_direction")
    stats_summary = pd.DataFrame([
        {"Quantity": "Coefficients significant at 5% (excluding intercept)", "Value": int((pv[:-1] < 0.05).sum())},
        {"Quantity": "McFadden pseudo-R²", "Value": round(1 - llf / ll0, 3)},
        {"Quantity": "Residual deviance", "Value": round(-2 * llf, 1)},
        {"Quantity": "Null deviance", "Value": round(-2 * ll0, 1)},
        {"Quantity": "max |coef − statsmodels|", "Value": f"{check['max |coef − statsmodels|']:.1e}"},
        {"Quantity": "max |SE − statsmodels|", "Value": f"{check['max |SE − statsmodels|']:.1e}"},
    ])
    table(stats_summary, "expH_summary")

    top = coef.iloc[:-1].assign(absz=np.abs(z[:-1])).sort_values("absz", ascending=False).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    yy_ = np.arange(len(top))
    colr = [AQUA if o < 1 else ORANGE for o in top["Odds ratio"]]
    ax.hlines(yy_, top["OR 95% low"], top["OR 95% high"], color=colr, lw=2)
    ax.scatter(top["Odds ratio"], yy_, color=colr, s=36, zorder=3)
    ax.axvline(1, color=MUTED, lw=1)
    ax.set_xscale("log")
    ax.set_xlim(0.04, 3.5)
    ax.set_yticks(yy_, top["Parameter"])
    ax.set_xlabel("Odds ratio of bad credit, 95% CI (log scale)")
    ax.set_title("Strongest effects: odds ratios with confidence intervals")
    ax.text(0.02, 0.02, "left of 1: lower risk   |   right of 1: higher risk", transform=ax.transAxes,
            fontsize=8.5, color=MUTED)
    ax.grid(axis="y", visible=False)
    save(fig, "expH_odds_ratios.png")

    # ============================================================ I: performance & uncertainty
    lams = [0, 0.1, 1, 3, 10, 30, 100, 300]
    skf = StratifiedKFold(10, shuffle=True, random_state=SEED)
    rows = []
    for lam in lams:
        a_, l_ = [], []
        for tr, va in skf.split(A, y):
            t_ = newton(A[tr], y[tr], lam=lam, tol=1e-10)
            q = sig(A[va] @ t_)
            a_.append(roc_auc_score(y[va], q))
            l_.append(-np.mean(y[va] * np.log(q) + (1 - y[va]) * np.log(1 - q)))
        rows.append({"λ": lam, "CV AUC": np.mean(a_), "AUC SE": np.std(a_) / np.sqrt(10),
                     "CV log-loss": np.mean(l_), "log-loss SE": np.std(l_) / np.sqrt(10)})
    cv = pd.DataFrame(rows)
    table(cv.round(4), "expI_cross_validation")
    best = cv.loc[cv["CV log-loss"].idxmin()]

    fig, ax = plt.subplots(figsize=(7, 4))
    xl = np.array([0.03] + lams[1:])
    ax.errorbar(xl, cv["CV log-loss"], yerr=cv["log-loss SE"], color=BLUE, marker="o", ms=5, capsize=3)
    ax.set_xscale("log")
    ax.set_xticks(xl, ["0"] + [str(v) for v in lams[1:]])
    ax.axvline(best["λ"], color=ORANGE, lw=1.2, ls="--")
    ax.text(best["λ"] * 1.1, cv["CV log-loss"].max() * 0.99, f"best λ = {best['λ']:g}", color=ORANGE, fontsize=9)
    ax.set_xlabel("Regularisation strength λ")
    ax.set_ylabel("10-fold CV log-loss (± 1 SE)")
    ax.set_title("Cross-validation on the training set chooses moderate λ")
    save(fig, "expI_cross_validation.png")

    p0 = sig(At @ th_star)
    rng = np.random.default_rng(1)
    boots = []
    for _ in range(2000):
        i = rng.integers(0, len(yt), len(yt))
        if yt[i].min() != yt[i].max():
            boots.append(roc_auc_score(yt[i], p0[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    split_auc = []
    for s in range(50):
        dd = prepare_data(df, random_state=s)
        t_ = newton(augment(dd["X_train"]), dd["y_train"], tol=1e-10)
        split_auc.append(roc_auc_score(dd["y_test"], sig(augment(dd["X_test"]) @ t_)))
    split_auc = np.array(split_auc)
    auc0 = roc_auc_score(yt, p0)
    unc = pd.DataFrame([
        {"Quantity": "Test AUC (seed 42, λ = 0)", "Value": round(auc0, 4)},
        {"Quantity": "Bootstrap 95% CI (2,000 resamples)", "Value": f"[{lo:.3f}, {hi:.3f}]"},
        {"Quantity": "Mean test AUC over 50 random splits", "Value": round(split_auc.mean(), 4)},
        {"Quantity": "SD over 50 splits", "Value": round(split_auc.std(), 4)},
        {"Quantity": "Range over 50 splits", "Value": f"[{split_auc.min():.3f}, {split_auc.max():.3f}]"},
        {"Quantity": "Rank of seed-42 split among 50", "Value": int((split_auc < auc0).sum()) + 1},
    ])
    table(unc, "expI_uncertainty")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ax = axes[0]
    ax.hist(boots, bins=40, color=BLUE, edgecolor=SURFACE)
    ax.axvline(auc0, color=INK, lw=1.5)
    ax.axvspan(lo, hi, color=BLUE, alpha=0.08)
    ax.set_title(f"Bootstrap: 95% CI [{lo:.3f}, {hi:.3f}]")
    ax.set_xlabel("Test AUC")
    ax.set_ylabel("Resamples")
    ax = axes[1]
    ax.hist(split_auc, bins=15, color=AQUA, edgecolor=SURFACE)
    ax.axvline(auc0, color=INK, lw=1.5)
    ax.text(auc0, ax.get_ylim()[1] * 0.92, " seed 42", fontsize=8.5)
    ax.axvline(split_auc.mean(), color=ORANGE, lw=1.5, ls="--")
    ax.text(split_auc.mean(), ax.get_ylim()[1] * 0.8, f" mean {split_auc.mean():.3f}", fontsize=8.5, color=ORANGE)
    ax.set_title("50 random train/test splits")
    ax.set_xlabel("Test AUC")
    save(fig, "expI_auc_uncertainty.png")

    # ============================================================ J: calibration & cost
    def hl(yv, pv_, g=10):
        q = pd.qcut(pv_, g, labels=False, duplicates="drop")
        D = pd.DataFrame({"y": yv, "p": pv_, "g": q}).groupby("g").agg(o=("y", "sum"), e=("p", "sum"), n=("y", "size"))
        H = (((D.o - D.e) ** 2) / (D.e * (1 - D.e / D.n))).sum()
        return H, stats.chi2.sf(H, g - 2)

    rows, curves = [], {}
    for lam in (0, 1, 3, 10, 100):
        t_ = newton(A, y, lam=lam, tol=1e-10)
        pt = sig(At @ t_)
        cs = sm.Logit(yt, sm.add_constant(At @ t_)).fit(disp=0).params
        H, hp = hl(yt, pt)
        rows.append({"λ": lam, "Test AUC": round(roc_auc_score(yt, pt), 4), "Brier score": round(brier_score_loss(yt, pt), 4),
                     "Calibration intercept": round(cs[0], 3), "Calibration slope": round(cs[1], 3),
                     "Hosmer–Lemeshow χ²": round(H, 2), "HL p-value": round(hp, 4)})
        curves[lam] = pt
    cal = pd.DataFrame(rows)
    table(cal, "expJ_calibration")

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="Perfect calibration")
    for lam, colr in ((0, ORANGE), (10, BLUE)):
        pt = curves[lam]
        q = pd.qcut(pt, 10, labels=False, duplicates="drop")
        D = pd.DataFrame({"y": yt, "p": pt, "g": q}).groupby("g").mean()
        r = cal.set_index("λ").loc[lam]
        ax.plot(D.p, D.y, color=colr, marker="o", ms=5,
                label=f"λ = {lam}: slope {r['Calibration slope']:.2f}, HL p " + ("< 0.001" if r['HL p-value'] < 0.001 else f"= {r['HL p-value']:.2f}"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability (test deciles)")
    ax.set_ylabel("Observed bad-credit rate")
    ax.set_title("Calibration: regularisation fixes overconfidence")
    ax.legend(loc="upper left", fontsize=8.5)
    save(fig, "expJ_calibration.png")

    ts_ = np.linspace(0.01, 0.99, 99)
    cost = np.array([5 * ((yt == 1) & (p0 < t)).sum() + ((yt == 0) & (p0 >= t)).sum() for t in ts_])
    def cost_at(t):
        return int(5 * ((yt == 1) & (p0 < t)).sum() + ((yt == 0) & (p0 >= t)).sum())
    tbest = ts_[np.argmin(cost)]
    cost_t = pd.DataFrame([
        {"Threshold": "1/6 (Bayes, 5:1 costs)", "Cost": cost_at(1 / 6)},
        {"Threshold": "0.5 (default)", "Cost": cost_at(0.5)},
        {"Threshold": f"{tbest:.2f} (lowest on test grid)", "Cost": int(cost.min())},
        {"Threshold": "Refuse everyone", "Cost": int((yt == 0).sum())},
        {"Threshold": "Approve everyone", "Cost": int(5 * (yt == 1).sum())},
    ])
    table(cost_t, "expJ_cost_thresholds")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(ts_, cost, color=BLUE)
    ax.axhline((yt == 0).sum(), color=MUTED, lw=1, ls=":")
    ax.text(0.7, (yt == 0).sum() + 4, "refuse everyone", fontsize=8.5, color=MUTED)
    for t, lab, colr in ((1 / 6, "1/6", ORANGE), (0.5, "0.5", INK)):
        ax.axvline(t, color=colr, lw=1.2, ls="--")
        ax.text(t + 0.01, cost.max() * 0.95, f"t = {lab}: cost {cost_at(t)}", color=colr, fontsize=8.5)
    ax.scatter([tbest], [cost.min()], color=AQUA, zorder=4)
    ax.annotate(f"lowest cost {cost.min()} at t = {tbest:.2f}", xy=(tbest, cost.min()), xytext=(0.22, 112),
                color=AQUA, fontsize=8.5, arrowprops=dict(arrowstyle="->", color=AQUA, lw=1))
    ax.set_ylim(100, cost.max() * 1.08)
    ax.set_xlabel("Decision threshold t (refuse if predicted PD ≥ t)")
    ax.set_ylabel("Cost on test set (5 × bad approved + good refused)")
    ax.set_title("Choosing the lending threshold from the costs of errors")
    save(fig, "expJ_cost_threshold.png")

    print("\nAll extended experiments done.")


if __name__ == "__main__":
    main()
