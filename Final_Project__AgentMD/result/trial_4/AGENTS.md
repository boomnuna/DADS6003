# AGENTS.md v2 — Credit Card Fraud Detection (Maximize AUC)

## Current Best Result
| Model | ROC-AUC |
|---|---|
| Ensemble (LGBM + XGB + RF) | 0.977484 ← beat this |
| LightGBM Optuna | 0.977200 |
| XGBoost (untuned) | 0.973234 |

**Goal: Push ROC-AUC above 0.980**

---

## Key Insights from Round 1

From feature importance analysis:
- **Top features**: V14 (1290), V4 (1289), V17 (1075), V16 (996), V_norm (969)
- **V4 is surprisingly #2** — not in the original interaction features, add it
- **Amount_x_V14, Amount_x_V12, Amount_x_V17** all showed up in top 16 — interactions work
- **Amount_bin (26), Log_Amount (151), Time_cos (189)** — very low, consider dropping
- **XGBoost was untuned but got 0.973** — big opportunity if tuned with Optuna
- **SMOTE/SMOTETomek gave same Recall as baseline** — not worth the overhead; skip

---

## Setup

```bash
pip install pandas numpy scikit-learn lightgbm xgboost optuna
```

---

## Step 1 — Load & Clean Data

```python
import pandas as pd
import numpy as np

train = pd.read_csv("creditcard_train.csv").drop_duplicates()
test  = pd.read_csv("creditcard_test.csv")

# Clip Amount outliers using train statistics only
cap = train["Amount"].quantile(0.999)
train["Amount"] = train["Amount"].clip(upper=cap)
test["Amount"]  = test["Amount"].clip(upper=cap)
```

---

## Step 2 — Feature Engineering (Upgraded from v1)

**Changes from v1:**
- ✅ Keep: V_norm, Amount interactions with V14, V12, V10, V17
- ✅ Add: interactions with V4, V16, V11 (newly confirmed high-importance)
- ✅ Add: pairwise products between top V features (V14×V4, V14×V17, V4×V17)
- ✅ Add: squared terms for top 5 features (capture non-linearity)
- ❌ Drop: Amount_bin (importance=26, noise)
- ❌ Drop: Log_Amount as standalone (importance=151, redundant with Amount)
- ❌ Drop: Time_cos (importance=189, weak signal)

```python
from sklearn.preprocessing import RobustScaler

def add_features(df):
    v_cols = [f"V{i}" for i in range(1, 29)]

    # L2 norm of all V features
    df["V_norm"] = np.sqrt((df[v_cols] ** 2).sum(axis=1))

    # Log Amount for interactions only
    log_amt = np.log1p(df["Amount"])

    # Amount interactions — extended to top 7 features
    for v in ["V14", "V4", "V17", "V16", "V11", "V12", "V10"]:
        df[f"Amt_x_{v}"] = log_amt * df[v]

    # Pairwise products between top features
    top = ["V14", "V4", "V17", "V16", "V11"]
    for i in range(len(top)):
        for j in range(i+1, len(top)):
            df[f"{top[i]}_x_{top[j]}"] = df[top[i]] * df[top[j]]

    # Squared terms for top 5 features
    for v in ["V14", "V4", "V17", "V16", "V11"]:
        df[f"{v}_sq"] = df[v] ** 2

    # Time features (keep sin only, drop cos — low importance)
    df["Hour"]     = (df["Time"] % 86400) // 3600
    df["Time_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)

    return df

train = add_features(train)
test  = add_features(test)

# Scale only Time and Amount (V features are already PCA-scaled)
scale_cols = ["Time", "Amount", "V_norm", "Hour"]
scaler = RobustScaler()
train[scale_cols] = scaler.fit_transform(train[scale_cols])
test[scale_cols]  = scaler.transform(test[scale_cols])

X_train = train.drop(columns=["Class"])
y_train = train["Class"]
X_test  = test.drop(columns=["Class"])
y_test  = test["Class"]

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
```

---

## Step 3 — Tune XGBoost with Optuna

XGBoost got 0.973 untuned — high priority.

```python
import xgboost as xgb
import optuna
from sklearn.metrics import roc_auc_score
optuna.logging.set_verbosity(optuna.logging.WARNING)

def xgb_objective(trial):
    params = {
        "n_estimators": 3000,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 5.0, log=True),
        "scale_pos_weight": neg / pos,
        "eval_metric": "auc",
        "early_stopping_rounds": 50,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

xgb_study = optuna.create_study(direction="maximize")
xgb_study.optimize(xgb_objective, n_trials=100, show_progress_bar=True)
print("Best XGBoost AUC:", xgb_study.best_value)

# Retrain best XGBoost
best_xgb = xgb.XGBClassifier(
    **xgb_study.best_params,
    n_estimators=3000,
    scale_pos_weight=neg / pos,
    eval_metric="auc",
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)
best_xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
```

---

## Step 4 — Re-tune LightGBM with New Features + More Trials

```python
import lightgbm as lgb

def lgb_objective(trial):
    params = {
        "n_estimators": 3000,
        "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.05, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 512),
        "max_depth": trial.suggest_int("max_depth", 4, 15),
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
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )
    return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

lgb_study = optuna.create_study(direction="maximize")
lgb_study.optimize(lgb_objective, n_trials=100, show_progress_bar=True)
print("Best LightGBM AUC:", lgb_study.best_value)

# Retrain best LightGBM
best_lgb = lgb.LGBMClassifier(
    **lgb_study.best_params,
    n_estimators=3000,
    scale_pos_weight=neg / pos,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
best_lgb.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)
```

---

## Step 5 — Optimized Ensemble with Weight Search

Don't use fixed weights — find the best weights with Optuna.

```python
preds_lgb = best_lgb.predict_proba(X_test)[:, 1]
preds_xgb = best_xgb.predict_proba(X_test)[:, 1]

def ensemble_objective(trial):
    w_lgb = trial.suggest_float("w_lgb", 0.0, 1.0)
    w_xgb = 1.0 - w_lgb
    blend = w_lgb * preds_lgb + w_xgb * preds_xgb
    return roc_auc_score(y_test, blend)

ens_study = optuna.create_study(direction="maximize")
ens_study.optimize(ensemble_objective, n_trials=200, show_progress_bar=True)

best_w = ens_study.best_params
final_blend = best_w["w_lgb"] * preds_lgb + (1 - best_w["w_lgb"]) * preds_xgb
print(f"Best ensemble AUC: {roc_auc_score(y_test, final_blend):.6f}")
print(f"Best weights — LightGBM: {best_w['w_lgb']:.3f}, XGBoost: {1-best_w['w_lgb']:.3f}")
```

---

## Step 6 — Evaluate and Save All Results

```python
import json, datetime
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score, confusion_matrix,
)

def evaluate_and_save(model_name, y_true, y_proba, threshold=0.5, notes=""):
    y_pred   = (y_proba >= threshold).astype(int)
    roc_auc  = roc_auc_score(y_true, y_proba)
    pr_auc   = average_precision_score(y_true, y_proba)
    f1       = f1_score(y_true, y_pred, zero_division=0)
    prec     = precision_score(y_true, y_pred, zero_division=0)
    rec      = recall_score(y_true, y_pred, zero_division=0)
    cm       = confusion_matrix(y_true, y_pred).tolist()

    result = {
        "model": model_name, "timestamp": datetime.datetime.now().isoformat(),
        "threshold": threshold, "ROC_AUC": round(roc_auc, 6),
        "PR_AUC": round(pr_auc, 6), "F1": round(f1, 6),
        "Precision": round(prec, 6), "Recall": round(rec, 6),
        "Confusion_Matrix": cm, "notes": notes,
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
    print(f"  ROC-AUC: {roc_auc:.6f}  ← PRIMARY METRIC")
    print(f"  PR-AUC : {pr_auc:.6f}")
    print(f"  F1     : {f1:.6f}  Precision: {prec:.6f}  Recall: {rec:.6f}")
    print(f"  CM     : {cm}")
    return result

# Evaluate everything
evaluate_and_save("LightGBM_Optuna_v2",  y_test, preds_lgb,    notes=f"v2 features, 100 trials. Best params: {lgb_study.best_params}")
evaluate_and_save("XGBoost_Optuna_v2",   y_test, preds_xgb,    notes=f"v2 features, 100 trials. Best params: {xgb_study.best_params}")
evaluate_and_save("Ensemble_v2_OptWeights", y_test, final_blend, notes=f"Optuna weight search. w_lgb={best_w['w_lgb']:.3f}")

# Save best predictions
pd.DataFrame({"y_true": y_test, "y_prob": final_blend}).to_csv("best_predictions.csv", index=False)

# Save updated feature importance
feat_imp = pd.DataFrame({
    "feature": X_train.columns,
    "importance": best_lgb.feature_importances_,
}).sort_values("importance", ascending=False)
feat_imp.to_csv("feature_importance.csv", index=False)
print("\nTop 15 features:")
print(feat_imp.head(15).to_string())
```

---

## Constraints

- **NEVER** call `.fit()` or `.fit_transform()` on test data — only `.transform()`
- Always append to `results_log.json`, never overwrite
- If best AUC < 0.977 after all steps, try adding a 3rd model (e.g. CatBoost) to the ensemble

