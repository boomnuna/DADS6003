# AGENTS.md — Credit Card Fraud Detection (Maximize AUC)

## Mission
Train a binary classification model on `creditcard_train.csv` and evaluate it on `creditcard_test.csv`.
**Primary metric: ROC-AUC** (maximize). Secondary: PR-AUC (report but do not optimize directly).

---

## Dataset Facts (do not re-derive, use these)
| Property | Value |
|---|---|
| Train rows | 199,364 |
| Features | `Time`, `V1`–`V28` (PCA-anonymized), `Amount` |
| Target | `Class` (0 = legitimate, 1 = fraud) |
| Class ratio | 344 fraud / 199,020 legit ≈ **0.17% positive rate** |
| Missing values | None |

---

## Environment Setup

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost optuna imbalanced-learn shap
```

---

## Step 1 — Data Loading & Feature Engineering

```python
import pandas as pd
import numpy as np

train = pd.read_csv("creditcard_train.csv")
test  = pd.read_csv("creditcard_test.csv")

def engineer_features(df):
    df = df.copy()
    # Log-transform Amount (heavy right skew)
    df["log_amount"] = np.log1p(df["Amount"])
    # Time-of-day cycle features (seconds in a day = 86400)
    df["time_sin"] = np.sin(2 * np.pi * df["Time"] / 86400)
    df["time_cos"] = np.cos(2 * np.pi * df["Time"] / 86400)
    # Hour of day bucket
    df["hour"] = (df["Time"] // 3600) % 24
    # Interaction: amount × top fraud-correlated PCA components
    for col in ["V1","V2","V3","V4","V10","V11","V12","V14","V17"]:
        df[f"{col}_x_logamt"] = df[col] * df["log_amount"]
    # Drop raw Time and Amount (replaced by engineered versions)
    df = df.drop(columns=["Time", "Amount"])
    return df

X_train = engineer_features(train.drop(columns=["Class"]))
y_train = train["Class"]
X_test  = engineer_features(test.drop(columns=["Class"]))
y_test  = test["Class"]
```

---

## Step 2 — Cross-Validation Strategy

Use **Stratified K-Fold** (k=5) throughout to preserve the rare-class ratio.
Use `roc_auc` as the scoring metric in all CV loops.

```python
from sklearn.model_selection import StratifiedKFold
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

---

## Step 3 — Model Training (Ensemble of Three)

Train three strong models independently. Do **not** upsample or downsample — instead rely on `scale_pos_weight` / `class_weight` parameters so probability calibration stays valid.

### 3a. LightGBM (primary)

```python
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

scale = (y_train == 0).sum() / (y_train == 1).sum()  # ≈ 579

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 3000,
    "learning_rate": 0.02,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "scale_pos_weight": scale,
    "lambda_l1": 0.1,
    "lambda_l2": 10.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

lgb_oof  = np.zeros(len(X_train))
lgb_test = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(CV.split(X_train, y_train)):
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_train.iloc[tr_idx], y_train.iloc[tr_idx],
        eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
        callbacks=[lgb.early_stopping(100, verbose=False),
                   lgb.log_evaluation(500)],
    )
    lgb_oof[val_idx]  = model.predict_proba(X_train.iloc[val_idx])[:, 1]
    lgb_test         += model.predict_proba(X_test)[:, 1] / 5

print(f"LGB OOF AUC: {roc_auc_score(y_train, lgb_oof):.6f}")
```

### 3b. XGBoost

```python
import xgboost as xgb

xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 3000,
    "learning_rate": 0.02,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": scale,
    "reg_alpha": 0.1,
    "reg_lambda": 10.0,
    "use_label_encoder": False,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}

xgb_oof  = np.zeros(len(X_train))
xgb_test = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(CV.split(X_train, y_train)):
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(
        X_train.iloc[tr_idx], y_train.iloc[tr_idx],
        eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
        early_stopping_rounds=100,
        verbose=False,
    )
    xgb_oof[val_idx]  = model.predict_proba(X_train.iloc[val_idx])[:, 1]
    xgb_test         += model.predict_proba(X_test)[:, 1] / 5

print(f"XGB OOF AUC: {roc_auc_score(y_train, xgb_oof):.6f}")
```

### 3c. CatBoost

```python
from catboost import CatBoostClassifier

cat_oof  = np.zeros(len(X_train))
cat_test = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(CV.split(X_train, y_train)):
    model = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.02,
        depth=6,
        scale_pos_weight=scale,
        eval_metric="AUC",
        early_stopping_rounds=100,
        random_seed=42,
        verbose=0,
    )
    model.fit(
        X_train.iloc[tr_idx], y_train.iloc[tr_idx],
        eval_set=(X_train.iloc[val_idx], y_train.iloc[val_idx]),
    )
    cat_oof[val_idx]  = model.predict_proba(X_train.iloc[val_idx])[:, 1]
    cat_test         += model.predict_proba(X_test)[:, 1] / 5

print(f"CAT OOF AUC: {roc_auc_score(y_train, cat_oof):.6f}")
```

---

## Step 4 — Stacking / Blending

First, find the optimal blending weights using OOF predictions (no data leakage):

```python
from scipy.optimize import minimize

def neg_auc(w):
    w = np.array(w)
    w = np.abs(w) / np.abs(w).sum()
    blend = w[0]*lgb_oof + w[1]*xgb_oof + w[2]*cat_oof
    return -roc_auc_score(y_train, blend)

result = minimize(neg_auc, [1/3, 1/3, 1/3], method="Nelder-Mead")
best_w = np.abs(result.x) / np.abs(result.x).sum()
print(f"Optimal weights → LGB:{best_w[0]:.3f}  XGB:{best_w[1]:.3f}  CAT:{best_w[2]:.3f}")

oof_blend  = best_w[0]*lgb_oof  + best_w[1]*xgb_oof  + best_w[2]*cat_oof
test_blend = best_w[0]*lgb_test + best_w[1]*xgb_test + best_w[2]*cat_test

print(f"Blended OOF AUC: {roc_auc_score(y_train, oof_blend):.6f}")
```

Optionally, add a **Level-2 logistic regression meta-learner** on the OOF stack:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

stack_train = np.column_stack([lgb_oof,  xgb_oof,  cat_oof])
stack_test  = np.column_stack([lgb_test, xgb_test, cat_test])

meta = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
meta.fit(stack_train, y_train)
meta_test_proba = meta.predict_proba(stack_test)[:, 1]
print(f"Meta-learner test AUC: {roc_auc_score(y_test, meta_test_proba):.6f}")
```

Use whichever gives better OOF AUC: simple blend or meta-learner.

---

## Step 5 — Optional: Optuna Hyperparameter Tuning

Run only if time allows (or if OOF AUC is below 0.980). Tune LightGBM first (usually yields the biggest gains).

```python
import optuna

def lgb_objective(trial):
    params = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 5,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "scale_pos_weight": scale,
        "random_state": 42,
        "verbose": -1,
    }
    aucs = []
    for tr_idx, val_idx in CV.split(X_train, y_train):
        m = lgb.LGBMClassifier(**params)
        m.fit(
            X_train.iloc[tr_idx], y_train.iloc[tr_idx],
            eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        aucs.append(roc_auc_score(y_train.iloc[val_idx],
                                  m.predict_proba(X_train.iloc[val_idx])[:, 1]))
    return np.mean(aucs)

study = optuna.create_study(direction="maximize")
study.optimize(lgb_objective, n_trials=50, show_progress_bar=True)
print("Best LGB AUC:", study.best_value)
print("Best params:", study.best_params)
```

---

## Step 6 — Final Evaluation on Test Set

```python
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)

final_proba = test_blend  # or meta_test_proba, whichever is better

test_auc   = roc_auc_score(y_test, final_proba)
test_prauc = average_precision_score(y_test, final_proba)

print("=" * 50)
print(f"  TEST ROC-AUC  : {test_auc:.6f}   ← primary metric")
print(f"  TEST PR-AUC   : {test_prauc:.6f}  ← secondary metric")
print("=" * 50)

# Threshold chosen to maximize F1 on OOF predictions
from sklearn.metrics import f1_score
thresholds = np.linspace(0.01, 0.99, 200)
f1s = [f1_score(y_train, oof_blend >= t) for t in thresholds]
best_thresh = thresholds[np.argmax(f1s)]
print(f"\nBest threshold (OOF F1): {best_thresh:.3f}")
y_pred = (final_proba >= best_thresh).astype(int)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))

# Save predictions
pd.DataFrame({"Class_proba": final_proba, "Class_pred": y_pred}).to_csv(
    "creditcard_test_predictions.csv", index=False
)
print("Predictions saved to creditcard_test_predictions.csv")
```

---

## Step 7 — SHAP Explainability (Optional but recommended)

```python
import shap

# Use the best-performing LGB model (last fold — approximate)
explainer = shap.TreeExplainer(model)  # model = last lgb fold model
shap_values = explainer.shap_values(X_test.iloc[:500])
shap.summary_plot(shap_values[1], X_test.iloc[:500], max_display=15)
```

---

## Expected Results

| Stage | ROC-AUC |
|---|---|
| Single LightGBM (baseline) | ≥ 0.975 |
| Blended ensemble (LGB+XGB+CAT) | ≥ 0.980 |
| After Optuna tuning | ≥ 0.983 |

---

## Key Rules for the Agent

1. **Never peek at test labels** before final evaluation — no leakage.
2. **Always use OOF predictions** for blending weights and threshold selection.
3. **Do not resample** (SMOTE/undersampling) — it distorts probability estimates and usually hurts AUC on this dataset.
4. **`scale_pos_weight`** must equal `n_negative / n_positive` for every model.
5. If total runtime exceeds 60 minutes, skip CatBoost and Optuna; LGB+XGB blend is sufficient.
6. Save `creditcard_test_predictions.csv` with columns `Class_proba` and `Class_pred`.
7. Print the final `TEST ROC-AUC` clearly at the end so it is easy to parse.

---

## File Structure Expected by Agent

```
./
├── creditcard_train.csv     # provided
├── creditcard_test.csv      # provided
├── AGENTS.md                # this file
├── train.py                 # generated by agent
└── creditcard_test_predictions.csv   # output
```
