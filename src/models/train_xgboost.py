import mlflow
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline


def train_xgboost(preprocessor, X_train, y_train):

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=2.33
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 4)
    mlflow.log_param("learning_rate", 0.1)

    return pipeline