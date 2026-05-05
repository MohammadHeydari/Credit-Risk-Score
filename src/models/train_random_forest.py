# src/models/train_random_forest.py

import mlflow

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def train_random_forest(preprocessor, X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 300)
    mlflow.log_param("max_depth", 8)
    mlflow.log_param("min_samples_split", 5)
    mlflow.log_param("min_samples_leaf", 2)

    return pipeline