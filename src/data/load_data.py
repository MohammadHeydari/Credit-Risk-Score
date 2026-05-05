# src/data/load_data.py

import pandas as pd
from pathlib import Path

# Column names for the German Credit dataset
# The dataset does not contain headers, so we assign them manually
COLUMN_NAMES = [
    "status_checking_account",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account",
    "employment_since",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "number_existing_credits",
    "job",
    "number_dependents",
    "telephone",
    "foreign_worker",
    "credit_risk"   # Target variable (loan risk classification)
]


def load_raw_data(file_path: str) -> pd.DataFrame:

    # Convert the file path to a Path object for OS-independent handling
    file_path = Path(file_path)

    df = pd.read_csv(
        file_path,
        sep=" ",
        header=None,
        names=COLUMN_NAMES
    )

    return df


def get_features_and_target(df: pd.DataFrame):
    """
    Split the loaded DataFrame into features (X) and target (y).
    """

    # Drop the target column to create the feature matrix
    X = df.drop("credit_risk", axis=1)

    # Extract the target column
    y = df["credit_risk"]

    return X, y
