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

# 4. Run notebooks in order
jupyter notebook notebooks/
```

## Experiments

| Experiment | Description |
|-----------|-------------|
| A | Learning-rate sensitivity (η = 0.001 to 1.0) |
| B | Feature scaling vs no scaling |
| C | L2 regularisation (λ = 0 vs λ = 1.0) |
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
