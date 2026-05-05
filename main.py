# main.py
import joblib

import mlflow

from src.models.train_xgboost import train_xgboost
from src.models.train_model import train_model
from src.models.evaluate import evaluate_model
from src.data.load_data import load_raw_data, get_features_and_target
from src.data.preprocess import (
    encode_target,
    split_data,
    build_preprocessing_pipeline
)

from XAI.shap_explainer import SHAPExplainer
from XAI.lime_explainer import LIMEExplainer

#mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("credit_risk_experiment")

from src.models.train_random_forest import train_random_forest

print(f"Tracking URI: {mlflow.get_tracking_uri()}")


def main():

    # ---------------------------------
    # 1) Load dataset
    # ---------------------------------
    file_path = "data/german.data"

    df = load_raw_data(file_path)
    X, y = get_features_and_target(df)

    feature_names = X.columns
    joblib.dump(feature_names, "feature_names.pkl")

    # ---------------------------------
    # 2) Encode target variable
    # ---------------------------------
    y = encode_target(y)

    # ---------------------------------
    # 3) Train / Test split
    # ---------------------------------
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ---------------------------------
    # 4) Preprocessing pipeline
    # ---------------------------------
    preprocessor = build_preprocessing_pipeline(X_train)
    preprocessor.fit(X_train)

    print("\nTrain shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Pipeline ready for model training.\n")

    # ====================================================
    # Logistic Regression Run
    # ====================================================
    with mlflow.start_run(run_name="logistic_regression_run"):

        #mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("dataset", "german_credit_data")
        mlflow.log_param("num_samples", X.shape[0])
        mlflow.log_param("num_features", X.shape[1])
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        model = train_model(preprocessor, X_train, y_train)
        evaluate_model(model, X_test, y_test)

    # ====================================================
    # XGBoost Run
    # ====================================================
    with mlflow.start_run(run_name="xgboost_run"):

        mlflow.log_param("dataset", "german_credit_data")
        mlflow.log_param("num_samples", X.shape[0])
        mlflow.log_param("num_features", X.shape[1])
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        model = train_xgboost(preprocessor, X_train, y_train)
        evaluate_model(model, X_test, y_test)

        # ---------------------------------
        # SHAP
        # ---------------------------------
        shap_exp = SHAPExplainer(
            model=model.named_steps["model"],
            preprocessor=model.named_steps["preprocessor"]
        )

        shap_summary_path = shap_exp.summary_plot(X_test)
        shap_force_path = shap_exp.explain_instance(X_test, index=0)

        mlflow.log_artifact(shap_summary_path)
        mlflow.log_artifact(shap_force_path)

        # ---------------------------------
        # LIME
        # ---------------------------------
        lime_exp = LIMEExplainer(
            model=model.named_steps["model"],
            preprocessor=model.named_steps["preprocessor"]
        )

        lime_exp.build(X_train)

        lime_path = lime_exp.explain_instance(
            X_test,
            index=0
        )

        mlflow.log_artifact(lime_path)

    # ====================================================
    # Random Forest Run
    # ====================================================
    with mlflow.start_run(run_name="random_forest_run"):
        mlflow.log_param("dataset", "german_credit_data")
        mlflow.log_param("num_samples", X.shape[0])
        mlflow.log_param("num_features", X.shape[1])
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        model = train_random_forest(preprocessor, X_train, y_train)
        evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
