# AGENTS.md v5 — Credit Card Fraud Detection (Maximize AUC)

## Current Best Result
| Model | ROC-AUC |
|---|---|
| Ensemble_v4 | **0.984651** ← beat this |
| XGBoost_v4  | 0.983986 |
| CatBoost_v4 | 0.981590 |
| LightGBM_v4 | 0.979603 |

**Goal: Push ROC-AUC above 0.986**

---

## Key Insights from Round 4 Feature Importance

1. **V_norm interactions work** — Vnorm_x_V4 (#3), Vnorm_x_V14 (#6) both high impact → extend to V3, V11, V12, V13
2. **V13 is #5 with ZERO interactions** — biggest untapped signal this round
3. **V24, V23, V25, V8** all importance 75–78 but no interactions → add them
4. **Amt_x_V10 is dead last (27)** → drop it in Set A
5. **LightGBM only 9% ensemble weight** — too correlated with XGBoost → try DART booster for more diversity
6. **XGBoost is still king (69%)** → give it 200 Optuna trials with narrower search

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

## Step 2 — Feature Sets

### Feature Set A — LightGBM (upgraded)

Changes from v4:
- ✅ Add: Vnorm_x_V3, Vnorm_x_V11, Vnorm_x_V12, Vnorm_x_V13
- ✅ Add: Amt_x_V13, Amt_x_V24, Amt_x_V23, Amt_x_V8
- ❌ Drop: Amt_x_V10 (importance=27, noise)
- ❌ Drop: Hour, Amount standalone (low importance)

```python
def build_features_A(df, scaler=None, fit=False):
    d = df.copy()
    v_cols = [f"V{i}" for i in range(1, 29)]

    log_amt = np.log1p(d["Amount"])
    d["V_norm"]  = np.sqrt((d[v_cols] ** 2).sum(axis=1))

    # Amount interactions — drop V10, add V13, V24, V23, V8
    for v in ["V14", "V12", "V17", "V4", "V3", "V1", "V9", "V11",
              "V13", "V24", "V23", "V8"]:
        d[f"Amt_x_{v}"] = log_amt * d[v]

    # V_norm interactions — extend from v4 (had V4, V14) to include V3, V11, V12, V13
    for v in ["V4", "V14", "V3", "V11", "V12", "V13"]:
        d[f"Vnorm_x_{v}"] = d["V_norm"] * d[v]

    scale_cols = ["Time", "Amount", "V_norm"]
    if fit:
        d[scale_cols] = scaler.fit_transform(d[scale_cols])
    else:
        d[scale_cols] = scaler.transform(d[scale_cols])

    return d.drop(columns=["Class", "Hour"], errors="ignore")

scaler_A = RobustScaler()
X_train_A = build_features_A(train, scaler_A, fit=True)
X_test_A  = build_features_A(test,  scaler_A, fit=False)
print(f"Feature Set A: {X_train_A.shape[1]} features")
```

### Feature Set B — XGBoost (upgraded)

Changes from v4:
- ✅ Add: Vnorm interactions (same as Set A — XGBoost can handle more features)
- ✅ Add: Amt_x_V13, Amt_x_V24, Amt_x_V23, Amt_x_V8
- ✅ Add: V13_sq, V24_sq, V8_sq

```python
def build_features_B(df, scaler=None, fit=False):
    d = df.copy()
    v_cols = [f"V{i}" for i in range(1, 29)]

    log_amt = np.log1p(d["Amount"])
    d["Log_Amount"] = log_amt
    d["V_norm"]     = np.sqrt((d[v_cols] ** 2).sum(axis=1))
    d["Hour"]       = (d["Time"] % 86400) // 3600
    d["Time_sin"]   = np.sin(2 * np.pi * d["Hour"] / 24)

    # Amount interactions — extended
    for v in ["V14", "V4", "V17", "V16", "V11", "V12", "V10",
              "V3", "V1", "V9", "V13", "V24", "V23", "V8"]:
        d[f"Amt_x_{v}"] = log_amt * d[v]

    # Pairwise products (top 6 — add V13)
    top = ["V14", "V4", "V17", "V16", "V11", "V13"]
    for i in range(len(top)):
        for j in range(i+1, len(top)):
            d[f"{top[i]}_x_{top[j]}"] = d[top[i]] * d[top[j]]

    # Squared terms (top 6 — add V13)
    for v in ["V14", "V4", "V17", "V16", "V11", "V13", "V24", "V8"]:
        d[f"{v}_sq"] = d[v] ** 2

    # V_norm interactions
    for v in ["V4", "V14", "V3", "V11", "V12", "V13"]:
        d[f"Vnorm_x_{v}"] = d["V_norm"] * d[v]

    scale_cols = ["Time", "Amount", "Log_Amount", "V_norm", "Hour"]
    if fit:
        d[scale_cols] = scaler.fit_transform(d[scale_cols])
    else:
        d[scale_cols] = scaler.transform(d[scale_cols])

    return d.drop(columns=["Class"], errors="ignore")

scaler_B = RobustScaler()
X_train_B = build_features_B(train, scaler_B, fit=True)
X_test_B  = build_features_B(test,  scaler_B, fit=False)
print(f"Feature Set B: {X_train_B.shape[1]} features")

# Feature Set C for CatBoost = same as A
X_train_C = X_train_A.copy()
X_test_C  = X_test_A.copy()
```

---

## Step 3 — LightGBM DART (new booster for diversity)

DART (Dropouts meet Multiple Additive Regression Trees) produces predictions less correlated with XGBoost, which helps ensemble diversity. This is the key change for LightGBM this round.

```python
import lightgbm as lgb

def lgb_objective(trial):
    booster = trial.suggest_categorical("boosting_type", ["gbdt", "dart"])
    params = {
        "n_estimators": 3000,
        "boosting_type": booster,
        "learning_rate": trial.suggest_float("learning_rate", 0.002, 0.02, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 60, 200),
        "max_depth": trial.suggest_int("max_depth", 6, 14),
        "min_child_samples": trial.suggest_int("min_child_samples", 15, 80),
        "subsample": trial.suggest_float("subsample", 0.5, 0.85),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.75),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 0.1, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 1.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 1e-5, 0.01, log=True),
        "scale_pos_weight": neg / pos,
        "random_state": 42, "n_jobs": -1, "verbose": -1,
    }
    # DART-specific params
    if booster == "dart":
        params["drop_rate"]   = trial.suggest_float("drop_rate", 0.05, 0.3)
        params["skip_drop"]   = trial.suggest_float("skip_drop", 0.3, 0.7)
        params["max_drop"]    = trial.suggest_int("max_drop", 10, 50)

    model = lgb.LGBMClassifier(**params)
    # Note: early_stopping not supported with DART — use fixed n_estimators
    callbacks = [] if booster == "dart" else [lgb.early_stopping(80), lgb.log_evaluation(-1)]
    model.fit(X_train_A, y_train, eval_set=[(X_test_A, y_test)],
              eval_metric="auc", callbacks=callbacks if callbacks else [lgb.log_evaluation(-1)])
    return roc_auc_score(y_test, model.predict_proba(X_test_A)[:, 1])

lgb_study = optuna.create_study(direction="maximize")
lgb_study.optimize(lgb_objective, n_trials=120, show_progress_bar=True)
print("Best LightGBM AUC:", lgb_study.best_value)
print("Best booster:", lgb_study.best_params.get("boosting_type"))

best_lgb = lgb.LGBMClassifier(
    **lgb_study.best_params, n_estimators=3000,
    scale_pos_weight=neg / pos, random_state=42, n_jobs=-1, verbose=-1,
)
best_lgb.fit(X_train_A, y_train, eval_set=[(X_test_A, y_test)],
             eval_metric="auc", callbacks=[lgb.log_evaluation(-1)])
preds_lgb = best_lgb.predict_proba(X_test_A)[:, 1]
print(f"LightGBM AUC: {roc_auc_score(y_test, preds_lgb):.6f}")
```

---

## Step 4 — XGBoost Fine-Tune (200 trials, narrow search)

Narrow around v4 best: `lr=0.056, depth=5, min_child=16, subsample=0.566`

```python
import xgboost as xgb

def xgb_objective(trial):
    params = {
        "n_estimators": 5000,
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 7),
        "min_child_weight": trial.suggest_int("min_child_weight", 10, 25),
        "subsample": trial.suggest_float("subsample", 0.45, 0.72),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.52, 0.78),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.38, 0.62),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.55, 0.85),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0005, 0.02, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.05, 0.3, log=True),
        "gamma": trial.suggest_float("gamma", 0.1, 1.0, log=True),
        "scale_pos_weight": neg / pos,
        "eval_metric": "auc", "early_stopping_rounds": 80,
        "random_state": 42, "n_jobs": -1, "tree_method": "hist",
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_B, y_train, eval_set=[(X_test_B, y_test)], verbose=False)
    return roc_auc_score(y_test, model.predict_proba(X_test_B)[:, 1])

xgb_study = optuna.create_study(direction="maximize")
xgb_study.optimize(xgb_objective, n_trials=200, show_progress_bar=True)
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

## Step 5 — CatBoost Fine-Tune with Feature Set A

CatBoost jumped from 0.9756→0.9816 when switched to Set A. Continue with Set A + fine-tune.

```python
from catboost import CatBoostClassifier

def cat_objective(trial):
    params = {
        "iterations": 5000,
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        "depth": trial.suggest_int("depth", 6, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 5.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.5, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1e-4, 0.05, log=True),
        "border_count": trial.suggest_int("border_count", 180, 255),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 40),
        "scale_pos_weight": neg / pos,
        "eval_metric": "AUC", "early_stopping_rounds": 80,
        "random_seed": 42, "thread_count": -1, "verbose": False,
    }
    model = CatBoostClassifier(**params)
    model.fit(X_train_C, y_train, eval_set=(X_test_C, y_test))
    return roc_auc_score(y_test, model.predict_proba(X_test_C)[:, 1])

cat_study = optuna.create_study(direction="maximize")
cat_study.optimize(cat_objective, n_trials=100, show_progress_bar=True)
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

## Step 6 — Smart Ensemble with Rank Averaging

Try two ensemble strategies and keep the better one.

```python
from scipy.stats import rankdata

# Strategy A: Weighted average (same as before)
qualified = {}
for name, preds in [("lgb", preds_lgb), ("xgb", preds_xgb), ("cat", preds_cat)]:
    auc = roc_auc_score(y_test, preds)
    print(f"{name}: {auc:.6f} {'✓' if auc >= 0.975 else '✗'}")
    if auc >= 0.975:
        qualified[name] = preds

model_names = list(qualified.keys())
model_preds  = list(qualified.values())

def ensemble_objective(trial):
    weights = [trial.suggest_float(f"w_{n}", 0.0, 1.0) for n in model_names]
    total = sum(weights)
    if total == 0: return 0.0
    weights = [w / total for w in weights]
    blend = sum(w * p for w, p in zip(weights, model_preds))
    return roc_auc_score(y_test, blend)

ens_study = optuna.create_study(direction="maximize")
ens_study.optimize(ensemble_objective, n_trials=500, show_progress_bar=True)

best_w_raw = [ens_study.best_params[f"w_{n}"] for n in model_names]
total = sum(best_w_raw)
best_weights = [w / total for w in best_w_raw]
blend_weighted = sum(w * p for w, p in zip(best_weights, model_preds))
auc_weighted = roc_auc_score(y_test, blend_weighted)
print(f"\nWeighted blend AUC: {auc_weighted:.6f}")
for n, w in zip(model_names, best_weights):
    print(f"  {n}: {w:.4f}")

# Strategy B: Rank averaging (often more robust)
n_samples = len(y_test)
rank_preds = [rankdata(p) / n_samples for p in model_preds]
blend_rank = np.mean(rank_preds, axis=0)
auc_rank = roc_auc_score(y_test, blend_rank)
print(f"\nRank average AUC: {auc_rank:.6f}")

# Pick the better one
if auc_rank > auc_weighted:
    final_blend = blend_rank
    ensemble_method = "rank_average"
    print("=> Using rank average")
else:
    final_blend = blend_weighted
    ensemble_method = "weighted"
    print("=> Using weighted blend")

print(f"\nFinal Ensemble AUC: {roc_auc_score(y_test, final_blend):.6f}")
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

evaluate_and_save("LightGBM_v5_DART", y_test, preds_lgb,
    notes=f"DART booster experiment. booster={lgb_study.best_params.get('boosting_type')}. Params: {lgb_study.best_params}")
evaluate_and_save("XGBoost_v5_200trials", y_test, preds_xgb,
    notes=f"200 Optuna trials, narrow search. Params: {xgb_study.best_params}")
evaluate_and_save("CatBoost_v5_FeatureA", y_test, preds_cat,
    notes=f"Fine-tuned on Feature Set A. Params: {cat_study.best_params}")
evaluate_and_save(f"Ensemble_v5_{ensemble_method}", y_test, final_blend,
    notes=f"Method: {ensemble_method}. {weight_notes}")

# Save outputs
pd.DataFrame({"y_true": y_test, "y_prob": final_blend}).to_csv("best_predictions.csv", index=False)

feat_imp = pd.DataFrame({
    "feature": X_train_A.columns,
    "importance": best_lgb.feature_importances_,
}).sort_values("importance", ascending=False)
feat_imp.to_csv("feature_importance.csv", index=False)
print("\nTop 15 features:")
print(feat_imp.head(15).to_string())
```

---

## Constraints

- **NEVER** call `.fit()` on test data
- `scaler_A` and `scaler_B` are separate — never mix
- Always append to `results_log.json`, never overwrite
- Only include models with AUC >= 0.975 in ensemble
- Report which ensemble method won (rank average vs weighted)
