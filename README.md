# Convergence Analysis of Gradient Descent for Credit Risk Prediction Using Logistic Regression

**M.Sc. Mathematics Project — Vivekananda Global University (VGU), Jaipur**

## Overview

This project studies the convergence behaviour of batch Gradient Descent applied to
the Logistic Regression objective function, using the UCI South German Credit dataset
as a numerical testbed.

The central research question is:

> How do the mathematical properties of the Logistic Regression objective function —
> particularly its curvature and conditioning — influence the convergence and stability
> of Gradient Descent, and how does this affect credit-risk classification?

## Project Structure

```
credit-risk-gradient-descent/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                  # Original UCI dataset (not modified)
│   └── processed/            # Preprocessed arrays (generated)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_mathematical_experiments.ipynb
│   └── 03_credit_risk_model.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preparation.py   # Loading, encoding, scaling, splitting
│   ├── logistic_regression.py# Sigmoid, loss, gradient, Hessian
│   ├── gradient_descent.py   # Batch GD with full history tracking
│   ├── evaluation.py         # Classification & optimisation metrics
│   └── visualization.py      # All plotting functions
├── results/
│   ├── figures/              # Saved plots
│   └── tables/               # Saved summary tables
└── report/
    └── dissertation.md       # Dissertation chapter outline
```

## Dataset

**UCI South German Credit** (CC BY 4.0)
- 1,000 observations, 20 predictors, 1 binary target
- DOI: https://doi.org/10.24432/C5X89F
- Historical German credit data (1973–1975)

> **Placeholder data notice.** The file currently in `data/raw/` is a synthetic
> stand-in with the same shape and column names as the UCI file, not the real
> data. Everything in `results/` was produced from it. Run
> `python download_dataset.py` (or place the real `SouthGermanCredit.asc` in
> `data/raw/`) and then `python run_all_experiments.py` to produce real results.

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
#    Place SouthGermanCredit.asc in data/raw/

# 4. Regenerate every table and figure (about 15 seconds)
python run_all_experiments.py

# 5. Or explore step by step in the notebooks
jupyter notebook notebooks/
```

## Experiments

| Experiment | Description |
|-----------|-------------|
| A | Learning-rate sweep on both sides of the step-size ceiling 2/L (η = 0.001 to 4.0) |
| B | Scaled vs unscaled features, each run at its own step size 1/L |
| C | L2 regularisation (λ = 0, 0.1, 1, 10, 100) |
| D | Independent check against scikit-learn |

## Key Mathematical Concepts

- Binary cross-entropy (negative log-likelihood) objective
- Gradient and Hessian derivation for logistic regression
- Convexity and Lipschitz smoothness
- Hessian eigenvalues, condition number κ(H)
- Learning-rate bounds and convergence guarantees
- Strong convexity via L2 regularisation

## References

- Boyd & Vandenberghe (2004). *Convex Optimization*. Cambridge.
- Nocedal & Wright (2006). *Numerical Optimization*, 2nd ed. Springer.
- Bishop (2006). *Pattern Recognition and Machine Learning*. Springer.
- Hastie, Tibshirani & Friedman (2009). *Elements of Statistical Learning*, 2nd ed. Springer.

## License

Dataset: CC BY 4.0 (UCI Machine Learning Repository)
Code: MIT
