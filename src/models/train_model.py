# src/models/train_model.py

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


def train_model(preprocessor, X_train, y_train):

    # Logistic Regression baseline model
    model = LogisticRegression(max_iter=1000, class_weight='balanced')

    # Build full pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Log model parameters
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)

    return pipeline
