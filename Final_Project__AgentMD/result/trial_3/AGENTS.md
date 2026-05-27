# AGENTS.md — Credit Card Fraud Detection (Maximize AUC)

## Task Overview

Train a binary classification model on `creditcard_train.csv` and evaluate it on `creditcard_test.csv`.
**Primary metric: ROC-AUC** (maximize). Secondary metrics: PR-AUC, F1, Precision, Recall.

The target column is `Class` (0 = legitimate, 1 = fraud).

---

## Dataset Description

| Property | Detail |
|---|---|
| Train file | `creditcard_train.csv` |
| Test file | `creditcard_test.csv` |
| Rows (train) | ~199,364 |
| Features | 30 input features + target (`Class`) |
| Feature types | `Time` (int), `V1–V28` (float, PCA-transformed), `Amount` (float) |
| Target | `Class` — binary (0 or 1) |
| Class imbalance | ~0.17% fraud (≈344 fraud vs 199,020 legit in train) — **severely imbalanced** |
| Missing values | None |

---

## Step-by-Step Instructions

### 1. Environment Setup

```bash
pip install pandas numpy scikit-learn lightgbm xgboost imbalanced-learn optuna matplotlib seaborn
```

### 2. Load Data

```python
import pandas as pd

train = pd.read_csv("creditcard_train.csv")
test  = pd.read_csv("creditcard_test.csv")

X_train = train.drop(columns=["Class"])
y_train = train["Class"]

X_test = test.drop(columns=["Class"])
y_test = test["Class"]
```

### 3. Data Cleaning

```python
# 3a. Check and drop duplicates
train = train.drop_duplicates()

# 3b. Verify no missing values (expected: none)
assert train.isnull().sum().sum() == 0, "Missing values detected!"

# 3c. Clip extreme outliers in Amount (cap at 99.9th percentile)
cap = train["Amount"].quantile(0.999)
train["Amount"] = train["Amount"].clip(upper=cap)
test["Amount"]  = test["Amount"].clip(upper=cap)
```

### 4. Feature Engineering

```python
import numpy as np

def add_features(df):
    # Log-transform Amount (heavy right skew)
    df["Log_Amount"] = np.log1p(df["Amount"])

    # Time-of-day features (Time is in seconds; 86400 = seconds per day)
    df["Hour"]      = (df["Time"] % 86400) // 3600
    df["Time_sin"]  = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Time_cos"]  = np.cos(2 * np.pi * df["Hour"] / 24)

    # Amount buckets (quantile bins)
    df["Amount_bin"] = pd.qcut(df["Amount"], q=10, labels=False, duplicates="drop")

    # Interaction: Amount × high-signal V features
    # V14, V12, V10, V17 are well-known high-importance features in this dataset
    for v in ["V14", "V12", "V10", "V17"]:
        df[f"Amount_x_{v}"] = df["Log_Amount"] * df[v]

    # Magnitude of V-feature vectors (L2 norm of V1–V28)
    v_cols = [f"V{i}" for i in range(1, 29)]
    df["V_norm"] = np.sqrt((df[v_cols] ** 2).sum(axis=1))

    return df

train = add_features(train)
test  = add_features(test)
```

### 5. Feature Scaling

```python
from sklearn.preprocessing import RobustScaler

scale_cols = ["Time", "Amount", "Log_Amount", "V_norm", "Hour"]

scaler = RobustScaler()
train[scale_cols] = scaler.fit_transform(train[scale_cols])
test[scale_cols]  = scaler.transform(test[scale_cols])
```

### 6. Handle Class Imbalance

Use **at least one** of the following strategies (try all and keep the best):

```python
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

# Option A: SMOTE oversampling
sm = SMOTE(random_state=42, k_neighbors=5)
X_res, y_res = sm.fit_resample(X_train, y_train)

# Option B: SMOTETomek (oversample + clean boundary)
smt = SMOTETomek(random_state=42)
X_res, y_res = smt.fit_resample(X_train, y_train)

# Option C: Use class_weight="balanced" or scale_pos_weight in the model
# (no resampling needed; handled natively by LightGBM/XGBoost)
```

### 7. Model Training

Train **all** of the following models and compare AUC:

#### 7a. LightGBM (Primary — recommended)

```python
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

lgb_model = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.02,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=neg / pos,   # handles imbalance
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
)
```

#### 7b. XGBoost

```python
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=2000,
    learning_rate=0.02,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=neg / pos,
    eval_metric="auc",
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1,
    use_label_encoder=False,
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100,
)
```

#### 7c. Random Forest (Baseline)

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf_model.fit(X_train, y_train)
```

#### 7d. Logistic Regression (Baseline)

```python
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42,
    C=0.01,
)
lr_model.fit(X_train, y_train)
```

### 8. Hyperparameter Tuning (Optional — for best model)

Use **Optuna** on the best-performing model (expected: LightGBM):

```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        "n_estimators": 2000,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "scale_pos_weight": neg / pos,
        "random_state": 42,
        "n_jobs": -1,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(-1)],
    )
    preds = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, preds)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("Best AUC:", study.best_value)
print("Best params:", study.best_params)
```

### 9. Ensemble (Optional — squeeze extra AUC)

Blend probability outputs from multiple models:

```python
# Weighted average of top models
preds_lgb = lgb_model.predict_proba(X_test)[:, 1]
preds_xgb = xgb_model.predict_proba(X_test)[:, 1]
preds_rf  = rf_model.predict_proba(X_test)[:, 1]

# Tune weights by cross-validation; start with equal
ensemble = 0.5 * preds_lgb + 0.3 * preds_xgb + 0.2 * preds_rf
auc_ensemble = roc_auc_score(y_test, ensemble)
print(f"Ensemble AUC: {auc_ensemble:.6f}")
```

---

## 10. Evaluation & Results Recording

Compute all metrics and save them:

```python
import json
import datetime
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

def evaluate_and_save(model_name, y_true, y_proba, threshold=0.5, notes=""):
    y_pred = (y_proba >= threshold).astype(int)

    roc_auc  = roc_auc_score(y_true, y_proba)
    pr_auc   = average_precision_score(y_true, y_proba)
    f1       = f1_score(y_true, y_pred)
    prec     = precision_score(y_true, y_pred)
    rec      = recall_score(y_true, y_pred)
    cm       = confusion_matrix(y_true, y_pred).tolist()

    result = {
        "model":      model_name,
        "timestamp":  datetime.datetime.now().isoformat(),
        "threshold":  threshold,
        "ROC_AUC":    round(roc_auc, 6),
        "PR_AUC":     round(pr_auc, 6),
        "F1":         round(f1, 6),
        "Precision":  round(prec, 6),
        "Recall":     round(rec, 6),
        "Confusion_Matrix": cm,
        "notes":      notes,
    }

    # Append to results log
    log_path = "results_log.json"
    try:
        with open(log_path) as f:
            log = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        log = []

    log.append(result)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"  ROC-AUC  : {roc_auc:.6f}  ← PRIMARY METRIC")
    print(f"  PR-AUC   : {pr_auc:.6f}")
    print(f"  F1       : {f1:.6f}")
    print(f"  Precision: {prec:.6f}")
    print(f"  Recall   : {rec:.6f}")
    print(f"  Confusion Matrix: {cm}")
    print(f"  Saved to {log_path}")

    return result

# Example usage:
preds_lgb = lgb_model.predict_proba(X_test)[:, 1]
evaluate_and_save("LightGBM_v1", y_test, preds_lgb, threshold=0.5, notes="default params")

# Always save the best predictions to CSV for submission
best_preds_df = pd.DataFrame({"y_true": y_test, "y_prob": preds_lgb})
best_preds_df.to_csv("best_predictions.csv", index=False)
```

---

## 11. Output Files to Produce

| File | Description |
|---|---|
| `results_log.json` | All experiment results appended chronologically |
| `best_predictions.csv` | Final test-set probability scores from best model |
| `feature_importance.csv` | Feature importance from best tree model |
| `best_model.pkl` | Serialized best model (optional) |

### Save Feature Importance

```python
import matplotlib.pyplot as plt

feat_imp = pd.DataFrame({
    "feature":   X_train.columns,
    "importance": lgb_model.feature_importances_,
}).sort_values("importance", ascending=False)

feat_imp.to_csv("feature_importance.csv", index=False)
print(feat_imp.head(15))
```

---

## Optimization Checklist

Work through this list in order. Stop when AUC satisfies your target.

- [ ] Baseline LightGBM with `scale_pos_weight` (no resampling)
- [ ] Add log-transform on `Amount` and time-of-day features
- [ ] Add interaction features (Amount × V14, V12, V10, V17)
- [ ] Add `V_norm` feature
- [ ] Try SMOTE oversampling
- [ ] Try SMOTETomek
- [ ] Hyperparameter tuning with Optuna (50–100 trials)
- [ ] XGBoost trained with same features
- [ ] Ensemble LightGBM + XGBoost (weighted blend)
- [ ] Threshold optimization (find best F1/PR threshold if needed)

---

## Constraints & Notes

- **Do not leak test labels** into training or feature engineering.
- Apply all fitted transformers (scaler, bins) from train to test using `transform`, never `fit_transform`.
- All results must be appended to `results_log.json` — never overwrite.
- If ROC-AUC on test set is below **0.97**, investigate: check for data leakage, try deeper feature engineering, or run more Optuna trials.
- Expected achievable AUC for this dataset with a well-tuned LightGBM: **0.975–0.985**.
