# Convergence Analysis of Gradient Descent for Credit Risk Prediction Using Logistic Regression

## Dissertation Outline

---

### Chapter 1: Introduction and Problem Formulation

- Background: machine learning in financial decision-making
- Motivation: understanding optimisation as mathematics, not just a library call
- Research question and scope
- Contribution of the project
- Organisation of the report

### Chapter 2: Mathematical Preliminaries

- 2.1 Vectors, matrices and norms
- 2.2 Multivariable calculus: gradients, Hessians, Taylor expansion
- 2.3 Eigenvalues and condition number
- 2.4 Convex sets and convex functions
- 2.5 Strong convexity
- 2.6 Lipschitz continuity of gradients (L-smoothness)

### Chapter 3: Logistic Regression and Gradient Descent

- 3.1 The sigmoid function and its properties
- 3.2 Logistic Regression as a probabilistic model
- 3.3 Binary cross-entropy objective (negative log-likelihood)
- 3.4 Derivation of the gradient
- 3.5 Derivation of the Hessian
- 3.6 Convexity of the logistic loss
- 3.7 Batch Gradient Descent update rule
- 3.8 L2 regularisation and its effect on the objective

### Chapter 4: Convergence and Curvature Analysis

- 4.1 Convergence of GD for convex, L-smooth functions
- 4.2 Learning-rate conditions: η < 2/L
- 4.3 Convergence rate: O(1/k) for convex smooth objectives
- 4.4 Linear convergence under strong convexity
- 4.5 Role of eigenvalues and condition number
- 4.6 How regularisation can provide strong convexity
- 4.7 Why convergence ≠ classification performance

### Chapter 5: Dataset and Computational Methodology

- 5.1 UCI South German Credit dataset
  - Source, provenance, license
  - Variables and codebook
  - Target definition (y=1 for bad credit)
  - Data limitations (historical, stratified)
- 5.2 Preprocessing pipeline
  - Categorical encoding (one-hot, drop-first)
  - Numerical standardisation
  - Train/test split (stratified, 80/20)
  - Fit on training data only
- 5.3 Implementation
  - Python, NumPy, from-scratch GD
  - Stopping rule: gradient norm < tolerance
  - Recorded quantities per iteration
- 5.4 Experimental design
  - Experiment A: learning-rate sensitivity
  - Experiment B: feature scaling
  - Experiment C: L2 regularisation
  - Experiment D: sklearn comparison

### Chapter 6: Experimental Results and Discussion

- 6.1 Experiment A results and interpretation
- 6.2 Experiment B results and interpretation
- 6.3 Experiment C results and interpretation
- 6.4 Experiment D results and interpretation
- 6.5 Classification performance (confusion matrix, precision, recall, F1, ROC-AUC)
- 6.6 Theory vs observation: where do they agree? Where do they differ?

### Chapter 7: Conclusion, Limitations and Future Work

- Summary of findings
- Limitations
  - Historical data, not current
  - Stratified sample, not natural rate
  - Single dataset, single method
  - No hyperparameter optimisation
- Future extensions
  - Stochastic and mini-batch GD
  - Momentum, Adam
  - Other datasets
  - Stronger convergence proofs

### References

### Appendices

- A. Full mathematical derivations
- B. Dataset codebook
- C. Complete Python source code
- D. Additional figures and tables
- E. Experiment settings and random seeds
