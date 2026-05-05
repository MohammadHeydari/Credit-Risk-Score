# Credit Risk Scoring System (ML + XAI + MLOps)

## Overview

This project implements an end-to-end machine learning system for credit risk scoring using the German Credit dataset. It covers data preprocessing, model training, evaluation, explainability, experiment tracking, and deployment via a REST API.

The system compares multiple machine learning models and provides interpretability using SHAP and LIME.

---

## Problem Definition

The goal is to predict credit risk:

- 0 → Bad credit risk  
- 1 → Good credit risk  

---

## Machine Learning Models

The following models are implemented:

- Logistic Regression (baseline model)
- Random Forest
- XGBoost

---

## Evaluation Metrics

Model performance is evaluated using:

- Accuracy
- ROC-AUC
- Confusion Matrix
- ROC Curve

---

## Explainability (XAI)

Model interpretability is provided using:

- SHAP (global and local explanations)
- LIME (local explanation for individual predictions)

---

## MLOps Components

- MLflow for experiment tracking:
  - parameters
  - metrics
  - artifacts
  - trained models
- Reproducible training pipeline
- Structured modular codebase

---

## API Service

A FastAPI-based inference service is implemented.

### Endpoint

`POST /predict`

### Request Body

```json
{
  "status_checking_account": "A11",
  "duration_months": 12,
  "credit_history": "A34",
  "credit_amount": 2500,
  "age": 35
}
```

### Response
```
{
  "risk_score": 0.73,
  "decision": "reject"
}
```
### Tech Stack

- Python
- Scikit-learn
- XGBoost
- MLflow
- SHAP
- LIME
- FastAPI
- Pandas
- NumPy
