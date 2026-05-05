import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    roc_curve,
    confusion_matrix
)


def evaluate_model(pipeline, X_test, y_test):

    # preds = pipeline.predict(X_test)
    # probs = pipeline.predict_proba(X_test)[:, 1]
    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.40).astype(int)


    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc", auc)

    print(f"\nAccuracy: {acc:.3f}")
    print(f"ROC-AUC: {auc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    # -------- Confusion Matrix --------
    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(5,4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Good (0)", "Bad (1)"],
        yticklabels=["Good (0)", "Bad (1)"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    cm_path = "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    mlflow.log_artifact(cm_path)
    os.remove(cm_path)

    # -------- ROC Curve --------
    fpr, tpr, _ = roc_curve(y_test, probs)

    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0,1],[0,1],'--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    roc_path = "roc_curve.png"
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()

    mlflow.log_artifact(roc_path)
    os.remove(roc_path)

    # -------- Save model --------
    mlflow.sklearn.log_model(
        pipeline,
        name="credit_risk_model"
    )

    print("\nModel logged to MLflow successfully.")
