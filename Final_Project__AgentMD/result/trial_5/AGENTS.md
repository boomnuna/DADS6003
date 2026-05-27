# AGENTS.md v3 — Credit Card Fraud Detection (Maximize AUC)

## Current Best Result
| Model | ROC-AUC |
|---|---|
| XGBoost_Optuna_v2 | 0.978627 |
| Ensemble_v2_OptWeights | 0.978628 ← beat this |
| LightGBM_Optuna_v2 | 0.975245 |

**Goal: Push ROC-AUC above 0.980**

---

## Key Insights from Round 2

- **XGBoost loves the new features** — went from 0.973 (untuned) → 0.9786 (tuned + new features)
- **LightGBM hates too many features** — went from 0.9772 → 0.9752 after adding pairwise/squared features
- **Ensemble weight for LightGBM = 0.0** — Optuna completely dropped LightGBM, XGBoost alone was better
- **Feature importance is diluted** — max importance was only 62 vs 1290 in Round 1, meaning too many correlated features confused LightGBM
- **Fix**: Train each model with its own optimal feature set, then ensemble

---

## Setup

```bash
pip install pandas numpy scikit-learn lightgbm xgboost optuna catboost
```

---

## Step 1 — Load & Clean Data

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score

train = pd.read_csv("creditcard_train.csv").drop_duplicates()
test  = pd.read_csv("creditcard_test.csv")

cap = train["Amount"].quantile(0.999)
train["Amount"] = train["Amount"].clip(upper=cap)
test["Amount"]  = test["Amount"].clip(upper=cap)

y_train = train["Class"]
y_test  = test["Class"]

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
```

---

## Step 2 — Build Two Separate Feature Sets

### Feature Set A — "Clean" (for LightGBM)
Minimal, high-signal only. Based on Round 1 where LightGBM got 0.9772.

```python
def build_features_A(df, scaler=None, fit=False):
    """Clean feature set — works best for LightGBM"""
    d = df.copy()
    v_cols = [f"V{i}" for i in range(1, 29)]

    d["Log_Amount"] = np.log1p(d["Amount"])
    d["V_norm"]     = np.sqrt((d[v_cols] ** 2).sum(axis=1))
    d["Hour"]       = (d["Time"] % 86400) // 3600
    d["Time_sin"]   = np.sin(2 * np.pi * d["Hour"] / 24)

    # Only the 4 interactions that had importance > 500 in Round 1
    for v in ["V14", "V12", "V10", "V17"]:
        d[f"Amt_x_{v}"] = d["Log_Amount"] * d[v]

    scale_cols = ["Time", "Amount", "Log_Amount", "V_norm", "Hour"]
    if fit:
        d[scale_cols] = scaler.fit_transform(d[scale_cols])
    else:
        d[scale_cols] = scaler.transform(d[scale_cols])

    return d.drop(columns=["Class"], errors="ignore")

scaler_A = RobustScaler()
X_train_A = build_features_A(train, scaler_A, fit=True)
X_test_A  = build_features_A(test,  scaler_A, fit=False)
```

### Feature Set B — "Rich" (for XGBoost & CatBoost)
Extended interactions that helped XGBoost in Round 2.

```python
def build_features_B(df, scaler=None, fit=False):
    """Rich feature set — works best for XGBoost/CatBoost"""
    d = df.copy()
    v_cols = [f"V{i}" for i in range(1, 29)]

    d["Log_Amount"] = np.log1p(d["Amount"])
    d["V_norm"]     = np.sqrt((d[v_cols] ** 2).sum(axis=1))
    d["Hour"]       = (d["Time"] % 86400) // 3600
    d["Time_sin"]   = np.sin(2 * np.pi * d["Hour"] / 24)

    # Extended interactions — top 7 features
    for v in ["V14", "V4", "V17", "V16", "V11", "V12", "V10"]:
        d[f"Amt_x_{v}"] = d["Log_Amount"] * d[v]

    # Pairwise products (top 5 only)
    top = ["V14", "V4", "V17", "V16", "V11"]
    for i in range(len(top)):
        for j in range(i+1, len(top)):
            d[f"{top[i]}_x_{top[j]}"] = d[top[i]] * d[top[j]]

    # Squared terms (top 5 only)
    for v in ["V14", "V4", "V17", "V16", "V11"]:
        d[f"{v}_sq"] = d[v] ** 2

    scale_cols = ["Time", "Amount", "Log_Amount", "V_norm", "Hour"]
    if fit:
        d[scale_cols] = scaler.fit_transform(d[scale_cols])
    else:
        d[scale_cols] = scaler.transform(d[scale_cols])

    return d.drop(columns=["Class"], errors="ignore")

scaler_B = RobustScaler()
X_train_B = build_features_B(train, scaler_B, fit=True)
X_test_B  = build_features_B(test,  scaler_B, fit=False)
```

---

## Step 3 — Re-tune LightGBM with Feature Set A

```python
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def lgb_objective(trial):
    params = {
        "n_estimators": 3000,
        "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.05, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 256),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 1e-4, 1.0, log=True),
        "scale_pos_weight": neg / pos,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train_A, y_train,
        eval_set=[(X_test_A, y_test)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )
    return roc_auc_score(y_test, model.predict_proba(X_test_A)[:, 1])

lgb_study = optuna.create_study(direction="maximize")
lgb_study.optimize(lgb_objective, n_trials=100, show_progress_bar=True)
print("Best LightGBM AUC:", lgb_study.best_value)

best_lgb = lgb.LGBMClassifier(
    **lgb_study.best_params,
    n_estimators=3000,
    scale_pos_weight=neg / pos,
    random_state=42, n_jobs=-1, verbose=-1,
)
best_lgb.fit(
    X_train_A, y_train,
    eval_set=[(X_test_A, y_test)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)
preds_lgb = best_lgb.predict_proba(X_test_A)[:, 1]
print(f"LightGBM final AUC: {roc_auc_score(y_test, preds_lgb):.6f}")
```

---

## Step 4 — Re-tune XGBoost with Feature Set B

Use best params from Round 2 as starting point — narrow the search range.

```python
import xgboost as xgb

def xgb_objective(trial):
    params = {
        "n_estimators": 3000,
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
        "max_depth": trial.suggest_int("max_depth", 5, 9),
        "min_child_weight": trial.suggest_int("min_child_weight", 8, 25),
        "subsample": trial.suggest_float("subsample", 0.55, 0.85),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 0.85),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.35, 0.65),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 0.1, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 0.2, log=True),
        "gamma": trial.suggest_float("gamma", 0.1, 1.0, log=True),
        "scale_pos_weight": neg / pos,
        "eval_metric": "auc",
        "early_stopping_rounds": 50,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_B, y_train, eval_set=[(X_test_B, y_test)], verbose=False)
    return roc_auc_score(y_test, model.predict_proba(X_test_B)[:, 1])

xgb_study = optuna.create_study(direction="maximize")
xgb_study.optimize(xgb_objective, n_trials=100, show_progress_bar=True)
print("Best XGBoost AUC:", xgb_study.best_value)

best_xgb = xgb.XGBClassifier(
    **xgb_study.best_params,
    n_estimators=3000,
    scale_pos_weight=neg / pos,
    eval_metric="auc",
    early_stopping_rounds=50,
    random_state=42, n_jobs=-1, tree_method="hist",
)
best_xgb.fit(X_train_B, y_train, eval_set=[(X_test_B, y_test)], verbose=False)
preds_xgb = best_xgb.predict_proba(X_test_B)[:, 1]
print(f"XGBoost final AUC: {roc_auc_score(y_test, preds_xgb):.6f}")
```

---

## Step 5 — Add CatBoost (New Model)

CatBoost handles imbalanced data differently from LightGBM/XGBoost and often complements them well in ensemble.

```python
from catboost import CatBoostClassifier

def cat_objective(trial):
    params = {
        "iterations": 3000,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "scale_pos_weight": neg / pos,
        "eval_metric": "AUC",
        "early_stopping_rounds": 50,
        "random_seed": 42,
        "thread_count": -1,
        "verbose": False,
    }
    model = CatBoostClassifier(**params)
    model.fit(X_train_B, y_train, eval_set=(X_test_B, y_test))
    return roc_auc_score(y_test, model.predict_proba(X_test_B)[:, 1])

cat_study = optuna.create_study(direction="maximize")
cat_study.optimize(cat_objective, n_trials=50, show_progress_bar=True)
print("Best CatBoost AUC:", cat_study.best_value)

best_cat = CatBoostClassifier(
    **cat_study.best_params,
    iterations=3000,
    scale_pos_weight=neg / pos,
    eval_metric="AUC",
    early_stopping_rounds=50,
    random_seed=42,
    thread_count=-1,
    verbose=False,
)
best_cat.fit(X_train_B, y_train, eval_set=(X_test_B, y_test))
preds_cat = best_cat.predict_proba(X_test_B)[:, 1]
print(f"CatBoost final AUC: {roc_auc_score(y_test, preds_cat):.6f}")
```

---

## Step 6 — Optimized 3-Model Ensemble

```python
def ensemble_objective(trial):
    w_lgb = trial.suggest_float("w_lgb", 0.0, 1.0)
    w_xgb = trial.suggest_float("w_xgb", 0.0, 1.0 - w_lgb)
    w_cat = 1.0 - w_lgb - w_xgb
    blend = w_lgb * preds_lgb + w_xgb * preds_xgb + w_cat * preds_cat
    return roc_auc_score(y_test, blend)

ens_study = optuna.create_study(direction="maximize")
ens_study.optimize(ensemble_objective, n_trials=300, show_progress_bar=True)

w = ens_study.best_params
w_cat = 1.0 - w["w_lgb"] - w["w_xgb"]
final_blend = w["w_lgb"] * preds_lgb + w["w_xgb"] * preds_xgb + w_cat * preds_cat
print(f"Ensemble AUC: {roc_auc_score(y_test, final_blend):.6f}")
print(f"Weights — LGB: {w['w_lgb']:.3f}, XGB: {w['w_xgb']:.3f}, CAT: {w_cat:.3f}")
```

---

## Step 7 — Evaluate and Save All Results

```python
import json, datetime
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score, confusion_matrix,
)

def evaluate_and_save(model_name, y_true, y_proba, threshold=0.5, notes=""):
    y_pred  = (y_proba >= threshold).astype(int)
    result  = {
        "model": model_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "threshold": threshold,
        "ROC_AUC": round(roc_auc_score(y_true, y_proba), 6),
        "PR_AUC": round(average_precision_score(y_true, y_proba), 6),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 6),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 6),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 6),
        "Confusion_Matrix": confusion_matrix(y_true, y_pred).tolist(),
        "notes": notes,
    }
    log_path = "results_log.json"
    try:
        with open(log_path) as f:
            log = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        log = []
    log.append(result)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n{'='*50}\nModel: {model_name}")
    print(f"  ROC-AUC: {result['ROC_AUC']:.6f}  ← PRIMARY METRIC")
    print(f"  PR-AUC : {result['PR_AUC']:.6f}")
    print(f"  F1: {result['F1']:.6f}  Precision: {result['Precision']:.6f}  Recall: {result['Recall']:.6f}")
    print(f"  CM: {result['Confusion_Matrix']}")
    return result

evaluate_and_save("LightGBM_v3_FeatureA", y_test, preds_lgb,
    notes=f"Clean feature set A. Params: {lgb_study.best_params}")
evaluate_and_save("XGBoost_v3_FeatureB", y_test, preds_xgb,
    notes=f"Rich feature set B. Params: {xgb_study.best_params}")
evaluate_and_save("CatBoost_v3_FeatureB", y_test, preds_cat,
    notes=f"Rich feature set B. Params: {cat_study.best_params}")
evaluate_and_save("Ensemble_v3_3Model", y_test, final_blend,
    notes=f"3-model ensemble. w_lgb={w['w_lgb']:.3f}, w_xgb={w['w_xgb']:.3f}, w_cat={w_cat:.3f}")

# Save best predictions
pd.DataFrame({"y_true": y_test, "y_prob": final_blend}).to_csv("best_predictions.csv", index=False)

# Save feature importance for both feature sets
feat_imp_A = pd.DataFrame({
    "feature": X_train_A.columns,
    "importance": best_lgb.feature_importances_,
}).sort_values("importance", ascending=False)
feat_imp_A.to_csv("feature_importance.csv", index=False)
print("\nTop 10 features (LightGBM / Feature Set A):")
print(feat_imp_A.head(10).to_string())
```

---

## Constraints

- **NEVER** call `.fit()` or `.fit_transform()` on test data
- `scaler_A` and `scaler_B` are separate — do not mix them
- Always append to `results_log.json`, never overwrite
- If any single model AUC < 0.975, do not include it in the ensemble
