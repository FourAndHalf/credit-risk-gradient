"""
Data Preparation Module
=======================
Handles loading, encoding, scaling and splitting the UCI South German Credit
dataset for the Gradient Descent convergence study.

Dataset: UCI South German Credit (CC BY 4.0)
    https://doi.org/10.24432/C5X89F
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from pathlib import Path


# ---------------------------------------------------------------------------
# Column metadata derived from the South German Credit codebook
# ---------------------------------------------------------------------------

# The 20 predictor columns in the original file order
PREDICTOR_COLS = [
    "laufkont",     # status of existing checking account
    "laufzeit",     # duration in months
    "moral",        # credit history
    "verw",         # purpose
    "hoehe",      # credit amount
    "sparkont",     # savings account / bonds
    "beszeit",   # present employment since
    "rate",         # installment rate (% of disposable income)
    "famges",       # personal status and sex
    "buerge",       # other debtors / guarantors
    "wohnzeit",    # present residence since
    "verm",       # property
    "alter",        # age in years
    "weitkred",     # other installment plans
    "wohn",       # housing
    "bishkred",      # number of existing credits at this bank
    "beruf",     # job
    "pers",     # number of people being liable for maintenance
    "telef",     # telephone
    "gastarb",      # foreign worker
]

TARGET_COL = "kredit"  # 0 = good credit, 1 = bad credit

# Columns treated as categorical / ordinal (will be one-hot encoded)
CATEGORICAL_COLS = [
    "laufkont", "moral", "verw", "sparkont", "beszeit",
    "famges", "buerge", "verm", "weitkred", "wohn",
    "beruf", "telef", "gastarb",
]

# Columns treated as numerical (will be standardised)
NUMERICAL_COLS = [
    "laufzeit", "hoehe", "rate", "wohnzeit", "alter", "bishkred", "pers",
]


def load_raw_data(data_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Load the South German Credit data file.

    The dataset ships as ``SouthGermanCredit.asc`` — a space-separated text
    file with a header row.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the extracted dataset files.

    Returns
    -------
    pd.DataFrame
    """
    data_dir = Path(data_dir)

    # Try the .asc file first (standard distribution)
    asc_path = data_dir / "SouthGermanCredit.asc"
    if asc_path.exists():
        df = pd.read_csv(asc_path, sep=r"\s+")
    else:
        # Fallback: look for a CSV export
        csv_path = data_dir / "SouthGermanCredit.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(
                f"Cannot find SouthGermanCredit.asc or .csv in {data_dir}"
            )

    # Rename target to a friendlier name and recode:
    #   original coding: 1 = good credit, 0 = bad credit  (in the .asc file)
    # We want: 1 = bad credit (positive class), 0 = good credit
    if TARGET_COL in df.columns:
        # The raw file uses kredit: 1 = good, 0 = bad
        df[TARGET_COL] = (df[TARGET_COL] == 0).astype(int)
    else:
        # Try the English-friendly column name variant
        for candidate in ["kredit", "credit_risk"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: TARGET_COL})
                df[TARGET_COL] = (df[TARGET_COL] == 0).astype(int)
                break

    return df


def describe_dataset(df: pd.DataFrame) -> dict:
    """Return a summary dict for quick inspection."""
    return {
        "n_rows": len(df),
        "n_features": df.shape[1] - 1,
        "class_counts": df[TARGET_COL].value_counts().to_dict(),
        "class_ratio": df[TARGET_COL].mean(),
        "missing_values": int(df.isnull().sum().sum()),
        "dtypes": df.dtypes.value_counts().to_dict(),
    }


def build_preprocessor(
    categorical_cols: list[str] | None = None,
    numerical_cols: list[str] | None = None,
) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for encoding + scaling.

    Categorical columns → OneHotEncoder (drop='first' to avoid collinearity).
    Numerical columns  → StandardScaler.
    """
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS
    if numerical_cols is None:
        numerical_cols = NUMERICAL_COLS

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    scale: bool = True,
) -> dict:
    """Full pipeline: split → fit preprocessor on train → transform both.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe with all columns present.
    test_size : float
        Fraction held out for testing.
    random_state : int
        Reproducibility seed.
    scale : bool
        If False, numerical columns are NOT standardised (for the
        unscaled-vs-scaled experiment).

    Returns
    -------
    dict with keys:
        X_train, X_test   — np.ndarray  (n, p)
        y_train, y_test   — np.ndarray  (n,)
        preprocessor      — fitted ColumnTransformer
        feature_names     — list[str]
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].values.astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if scale:
        preprocessor = build_preprocessor()
    else:
        # Only encode categoricals; leave numericals as-is
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", "passthrough", NUMERICAL_COLS),
                ("cat", OneHotEncoder(drop="first", sparse_output=False), CATEGORICAL_COLS),
            ],
            remainder="drop",
        )

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # Recover feature names
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except AttributeError:
        feature_names = [f"x{i}" for i in range(X_train_t.shape[1])]

    return {
        "X_train": X_train_t.astype(np.float64),
        "X_test": X_test_t.astype(np.float64),
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
    }
