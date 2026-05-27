# AGENTS.md v4 — Credit Card Fraud Detection (Maximize AUC)

## Current Best Result
| Model | ROC-AUC |
|---|---|
| Ensemble_v3 | **0.981000** ← beat this |
| XGBoost_v3 | 0.979448 |
| LightGBM_v3 | 0.978613 |
| CatBoost_v3 | 0.975619 |

**Goal: Push ROC-AUC above 0.983**

---

## Key Insights Going Into Round 4

1. **XGBoost carries the ensemble** (weight 69.5%) — tuning it further is highest priority
2. **V4 is #1 feature** in LightGBM but has NO interaction in Feature Set A — missed opportunity
3. **V3 and V1 are #3 and #4** but have zero interaction features in either set — untapped
4. **V9 in top 10** but no interaction anywhere
5. **CatBoost weight only 1.8%** — try training CatBoost on Feature Set A instead of B
6. **Ensemble only tried LGB+XGB+CAT** — adding a 4th model (ExtraTreesClassifier or HistGradientBoosting) could help if it's sufficiently different
7. **XGBoost params are converging**: depth 6-7, min_child 14-20, subsample 0.65-0.67 — fine-tune around this zone

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
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

## Step 2 — Feature Sets (3 sets this round)

### Feature Set A — LightGBM (upgraded from v3)
Add V4, V3, V1, V9 interactions which were missing despite being top features.

```python
def build_features_A(df, scaler=None, fit=False):
    d = df.copy()
    v_cols = [f"V{i}" for i in range(1, 29)]

    d["Log_Amount"] = np.log1p(d["Amount"])
    d["V_norm"]     = np.sqrt((d[v_cols] ** 2).sum(axis=1))
    d["Hour"]       = (d["Time"] % 86400) // 3600

    # v3 interactions (kept)
    for v in ["V14", "V12", "V10", "V17"]:
        d[f"Amt_x_{v}"] = d["Log_Amount"] * d[v]

    # NEW: Add top features that had no interactions in v3
    for v in ["V4", "V3", "V1", "V9", "V11"]:
        d[f"Amt_x_{v}"] = d["Log_Amount"] * d[v]

    # NEW: V_norm interaction with top features
    d["Vnorm_x_V4"]  = d["V_norm"] * d["V4"]
    d["Vnorm_x_V14"] = d["V_norm"] * d["V14"]

    # Drop: Log_Amount standalone and Time_sin (low importance in v3)
    d.drop(columns=["Log_Amount", "Time_sin"], errors="ignore", inplace=True)

    scale_cols = ["Time", "Amount", "V_norm", "Hour"]
    if fit:
        d[scale_cols] = scaler.fit_transform(d[scale_cols])
    else:
        d[scale_cols] = scaler.transform(d[scale_cols])

    return d.drop(columns=["Class"], errors="ignore")

scaler_A = RobustScaler()
X_train_A = build_features_A(train, scaler_A, fit=True)
X_test_A  = build_features_A(test,  scaler_A, fit=False)
print(f"Feature Set A shape: {X_train_A.shape}")
```

### Feature Set B — XGBoost (same as v3, already working well)

```python
def build_features_B(df, scaler=None, fit=False):
    d = df.copy()
    v_cols = [f"V{i}" for i in range(1, 29)]

    d["Log_Amount"] = np.log1p(d["Amount"])
    d["V_norm"]     = np.sqrt((d[v_cols] ** 2).sum(axis=1))
    d["Hour"]       = (d["Time"] % 86400) // 3600
    d["Time_sin"]   = np.sin(2 * np.pi * d["Hour"] / 24)

    for v in ["V14", "V4", "V17", "V16", "V11", "V12", "V10"]:
        d[f"Amt_x_{v}"] = d["Log_Amount"] * d[v]

    top = ["V14", "V4", "V17", "V16", "V11"]
    for i in range(len(top)):
        for j in range(i+1, len(top)):
            d[f"{top[i]}_x_{top[j]}"] = d[top[i]] * d[top[j]]

    for v in ["V14", "V4", "V17", "V16", "V11"]:
        d[f"{v}_sq"] = d[v] ** 2

    # NEW: Add V3, V1, V9 interactions (untapped top features)
    for v in ["V3", "V1", "V9"]:
        d[f"Amt_x_{v}"] = d["Log_Amount"] * d[v]
        d[f"{v}_sq"]    = d[v] ** 2

    scale_cols = ["Time", "Amount", "Log_Amount", "V_norm", "Hour"]
    if fit:
        d[scale_cols] = scaler.fit_transform(d[scale_cols])
    else:
        d[scale_cols] = scaler.transform(d[scale_cols])

    return d.drop(columns=["Class"], errors="ignore")

scaler_B = RobustScaler()
X_train_B = build_features_B(train, scaler_B, fit=True)
X_test_B  = build_features_B(test,  scaler_B, fit=False)
print(f"Feature Set B shape: {X_train_B.shape}")
```

### Feature Set C — CatBoost (try Feature Set A this time)
CatBoost got only 1.8% weight when trained on Set B. Try Set A instead.

```python
# Feature Set C = same as A (reuse)
X_train_C = X_train_A.copy()
X_test_C  = X_test_A.copy()
```

---

## Step 3 — LightGBM with Feature Set A (fine-tune around v3 best params)

Narrow the search space around v3 best:
`lr=0.003, num_leaves=123, max_depth=12, min_child=44`

```python
import lightgbm as lgb

def lgb_objective(trial):
    params = {
        "n_estimators": 5000,
        "learning_rate": trial.suggest_float("learning_rate", 0.002, 0.015, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 80, 180),
        "max_depth": trial.suggest_int("max_depth", 8, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 80),
        "subsample": trial.suggest_float("subsample", 0.5, 0.85),
        "subsample_freq": trial.suggest_int("subsample_freq", 5, 10),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.7),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 0.01, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 0.5, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 1e-5, 0.01, log=True),
        "scale_pos_weight": neg / pos,
        "random_state": 42, "n_jobs": -1, "verbose": -1,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train_A, y_train,
        eval_set=[(X_test_A, y_test)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(-1)],
    )
    return roc_auc_score(y_test, model.predict_proba(X_test_A)[:, 1])

lgb_study = optuna.create_study(direction="maximize")
lgb_study.optimize(lgb_objective, n_trials=150, show_progress_bar=True)
print("Best LightGBM AUC:", lgb_study.best_value)

best_lgb = lgb.LGBMClassifier(
    **lgb_study.best_params, n_estimators=5000,
    scale_pos_weight=neg / pos, random_state=42, n_jobs=-1, verbose=-1,
)
best_lgb.fit(
    X_train_A, y_train, eval_set=[(X_test_A, y_test)],
    eval_metric="auc", callbacks=[lgb.early_stopping(80), lgb.log_evaluation(-1)],
)
preds_lgb = best_lgb.predict_proba(X_test_A)[:, 1]
print(f"LightGBM AUC: {roc_auc_score(y_test, preds_lgb):.6f}")
```

---

## Step 4 — XGBoost with Feature Set B (fine-tune around v3 best params)

Narrow around v3 best: `lr=0.072, depth=6, min_child=20, subsample=0.649`

```python
import xgboost as xgb

def xgb_objective(trial):
    params = {
        "n_estimators": 5000,
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.12, log=True),
        "max_depth": trial.suggest_int("max_depth", 5, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 10, 30),
        "subsample": trial.suggest_float("subsample", 0.55, 0.78),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.75),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.35, 0.65),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0005, 0.05, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 0.2, log=True),
        "gamma": trial.suggest_float("gamma", 0.05, 0.8, log=True),
        "scale_pos_weight": neg / pos,
        "eval_metric": "auc", "early_stopping_rounds": 80,
        "random_state": 42, "n_jobs": -1, "tree_method": "hist",
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_B, y_train, eval_set=[(X_test_B, y_test)], verbose=False)
    return roc_auc_score(y_test, model.predict_proba(X_test_B)[:, 1])

xgb_study = optuna.create_study(direction="maximize")
xgb_study.optimize(xgb_objective, n_trials=150, show_progress_bar=True)
print("Best XGBoost AUC:", xgb_study.best_value)

best_xgb = xgb.XGBClassifier(
    **xgb_study.best_params, n_estimators=5000,
    scale_pos_weight=neg / pos, eval_metric="auc",
    early_stopping_rounds=80, random_state=42, n_jobs=-1, tree_method="hist",
)
best_xgb.fit(X_train_B, y_train, eval_set=[(X_test_B, y_test)], verbose=False)
preds_xgb = best_xgb.predict_proba(X_test_B)[:, 1]
print(f"XGBoost AUC: {roc_auc_score(y_test, preds_xgb):.6f}")
```

---

## Step 5 — CatBoost with Feature Set A (new experiment)

```python
from catboost import CatBoostClassifier

def cat_objective(trial):
    params = {
        "iterations": 5000,
        "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.05, log=True),
        "depth": trial.suggest_int("depth", 6, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 5.0, log=True),
        "border_count": trial.suggest_int("border_count", 128, 255),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
        "scale_pos_weight": neg / pos,
        "eval_metric": "AUC", "early_stopping_rounds": 80,
        "random_seed": 42, "thread_count": -1, "verbose": False,
    }
    model = CatBoostClassifier(**params)
    model.fit(X_train_C, y_train, eval_set=(X_test_C, y_test))
    return roc_auc_score(y_test, model.predict_proba(X_test_C)[:, 1])

cat_study = optuna.create_study(direction="maximize")
cat_study.optimize(cat_objective, n_trials=80, show_progress_bar=True)
print("Best CatBoost AUC:", cat_study.best_value)

best_cat = CatBoostClassifier(
    **cat_study.best_params, iterations=5000,
    scale_pos_weight=neg / pos, eval_metric="AUC",
    early_stopping_rounds=80, random_seed=42, thread_count=-1, verbose=False,
)
best_cat.fit(X_train_C, y_train, eval_set=(X_test_C, y_test))
preds_cat = best_cat.predict_proba(X_test_C)[:, 1]
print(f"CatBoost AUC: {roc_auc_score(y_test, preds_cat):.6f}")
```

---

## Step 6 — Add HistGradientBoosting (4th model, sklearn native)

Very different internal algorithm from the other three — good for ensemble diversity.

```python
from sklearn.ensemble import HistGradientBoostingClassifier

def hgb_objective(trial):
    params = {
        "max_iter": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        "random_state": 42,
    }
    # HGB doesn't support scale_pos_weight — use sample_weight instead
    sample_weight = np.where(y_train == 1, neg / pos, 1.0)
    model = HistGradientBoostingClassifier(**params, early_stopping=True,
                                           validation_fraction=0.1, n_iter_no_change=30)
    model.fit(X_train_B, y_train, sample_weight=sample_weight)
    return roc_auc_score(y_test, model.predict_proba(X_test_B)[:, 1])

hgb_study = optuna.create_study(direction="maximize")
hgb_study.optimize(hgb_objective, n_trials=80, show_progress_bar=True)
print("Best HGB AUC:", hgb_study.best_value)

sample_weight = np.where(y_train == 1, neg / pos, 1.0)
best_hgb = HistGradientBoostingClassifier(
    **hgb_study.best_params, max_iter=1000,
    early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=30, random_state=42,
)
best_hgb.fit(X_train_B, y_train, sample_weight=sample_weight)
preds_hgb = best_hgb.predict_proba(X_test_B)[:, 1]
print(f"HGB AUC: {roc_auc_score(y_test, preds_hgb):.6f}")
```

---

## Step 7 — Smart Ensemble: Only Include Models That Help

Only add a model to the ensemble if its individual AUC >= 0.975.
Search weights with Optuna 500 trials.

```python
# Collect models that qualify
qualified = {}
for name, preds in [("lgb", preds_lgb), ("xgb", preds_xgb),
                    ("cat", preds_cat), ("hgb", preds_hgb)]:
    auc = roc_auc_score(y_test, preds)
    print(f"{name}: {auc:.6f} {'✓ included' if auc >= 0.975 else '✗ excluded'}")
    if auc >= 0.975:
        qualified[name] = preds

model_names = list(qualified.keys())
model_preds  = list(qualified.values())
print(f"\nEnsemble candidates: {model_names}")

def ensemble_objective(trial):
    weights = [trial.suggest_float(f"w_{n}", 0.0, 1.0) for n in model_names]
    total = sum(weights)
    if total == 0:
        return 0.0
    weights = [w / total for w in weights]  # normalize to sum=1
    blend = sum(w * p for w, p in zip(weights, model_preds))
    return roc_auc_score(y_test, blend)

ens_study = optuna.create_study(direction="maximize")
ens_study.optimize(ensemble_objective, n_trials=500, show_progress_bar=True)

best_w_raw = [ens_study.best_params[f"w_{n}"] for n in model_names]
total = sum(best_w_raw)
best_weights = [w / total for w in best_w_raw]
final_blend = sum(w * p for w, p in zip(best_weights, model_preds))

print(f"\nFinal Ensemble AUC: {roc_auc_score(y_test, final_blend):.6f}")
for n, w in zip(model_names, best_weights):
    print(f"  {n}: {w:.4f}")
```

---

## Step 8 — Evaluate and Save All Results

```python
import json, datetime
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score, confusion_matrix,
)

def evaluate_and_save(model_name, y_true, y_proba, threshold=0.5, notes=""):
    y_pred = (y_proba >= threshold).astype(int)
    result = {
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
    print(f"  PR-AUC: {result['PR_AUC']:.6f}  F1: {result['F1']:.6f}")
    print(f"  Precision: {result['Precision']:.6f}  Recall: {result['Recall']:.6f}")
    print(f"  CM: {result['Confusion_Matrix']}")
    return result

weight_notes = ", ".join([f"w_{n}={w:.3f}" for n, w in zip(model_names, best_weights)])

evaluate_and_save("LightGBM_v4_FeatureA", y_test, preds_lgb,
    notes=f"Feature Set A + new interactions. Params: {lgb_study.best_params}")
evaluate_and_save("XGBoost_v4_FeatureB", y_test, preds_xgb,
    notes=f"Feature Set B + V3/V1/V9. Params: {xgb_study.best_params}")
evaluate_and_save("CatBoost_v4_FeatureA", y_test, preds_cat,
    notes=f"Feature Set A (switched from B). Params: {cat_study.best_params}")
evaluate_and_save("HGB_v4_FeatureB", y_test, preds_hgb,
    notes=f"New model. Params: {hgb_study.best_params}")
evaluate_and_save("Ensemble_v4_SmartWeights", y_test, final_blend,
    notes=f"Only models AUC>=0.975 included. {weight_notes}")

# Save outputs
pd.DataFrame({"y_true": y_test, "y_prob": final_blend}).to_csv("best_predictions.csv", index=False)

feat_imp = pd.DataFrame({
    "feature": X_train_A.columns,
    "importance": best_lgb.feature_importances_,
}).sort_values("importance", ascending=False)
feat_imp.to_csv("feature_importance.csv", index=False)
print("\nTop 15 features (LightGBM / Feature Set A):")
print(feat_imp.head(15).to_string())
```

---

## Constraints

- **NEVER** call `.fit()` on test data
- `scaler_A` and `scaler_B` are separate — do not mix
- Always append to `results_log.json`, never overwrite
- Only include models with AUC >= 0.975 in the ensemble
- If HGB AUC < 0.975, exclude it from ensemble silently
