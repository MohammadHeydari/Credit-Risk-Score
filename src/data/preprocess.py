# src/features/preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def build_preprocessing_pipeline(X: pd.DataFrame):
    """
    Create preprocessing pipeline for categorical and numerical features.
    """

    # Identify feature types
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(exclude=["object"]).columns

    # Numerical preprocessing: scaling
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    # Categorical preprocessing: one-hot encoding
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    return preprocessor


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split dataset into train and test sets.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y   # preserve class distribution
    )

    return X_train, X_test, y_train, y_test


def encode_target(y: pd.Series):
    """
    Convert target labels from (1,2) to (0,1).
    """

    y = y.replace({1: 0, 2: 1})

    return y
