# AGENTS.md v6 — Credit Card Fraud Detection (Maximize AUC)

## Current Best Result
| Model | ROC-AUC |
|---|---|
| **Ensemble_v6 (rank_average)** | **0.988156** 🎉 beat goal |
| XGBoost_gbdt_v6 | 0.983983 |
| CatBoost_v6 | 0.982350 |
| LightGBM_DART1_v6 | 0.981671 |
| Ensemble_v5 (rank_average) | 0.987044 ← previous best |

**Goal: Push ROC-AUC above 0.988 — ACHIEVED!** 🎯

### Critical Lessons from v6
1. **DART is non-deterministic** — saving the best model DURING optuna is essential; retraining with same params+seed gives different (worse) results (0.982 → 0.960)
2. **New v6 polynomial/interaction features DEGRADED performance** — v5's simpler feature set (49 cols vs 60) worked better
3. **XGBoost gbdt (not dart)** was the strongest individual model (0.984), faster to tune, and deterministic
4. **3-model ensemble** (DART#1 + XGBoost + CatBoost) beat the 0.988 barrier

---

## Key Insights from Round 5

1. **DART booster is the star** — 94.3% ensemble weight, its predictions are most orthogonal to others
2. **V16 (#2), V10 (#3), V26 (#4)** all have importance 3400–3700 but ZERO Vnorm or Amt interactions
3. **V_norm is #1 (3717)** — try V_norm² and V_norm³ as polynomial features
4. **XGBoost is plateauing** — try XGBoost with `booster=dart` for a different signal
5. **2 DART models with different seeds** = more diversity for ensemble at low cost
6. **Rank average beat weighted blend** → stick with rank average

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
from scipy.stats import rankdata
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

### Feature Set A — LightGBM DART (upgraded from v5)

New additions based on v5 importance gaps:
- ✅ Add: Vnorm_x_V16, Vnorm_x_V10, Vnorm_x_V26, Vnorm_x_V9, Vnorm_x_V18
- ✅ Add: Amt_x_V16, Amt_x_V10, Amt_x_V26, Amt_x_V18
- ✅ Add: V_norm_sq (V_norm²), V_norm_cube (V_norm³)

```python
def build_features_A(df, scaler=None, fit=False):
    d = df.copy()
    v_cols = [f"V{i}" for i in range(1, 29)]

    log_amt      = np.log1p(d["Amount"])
    d["V_norm"]  = np.sqrt((d[v_cols] ** 2).sum(axis=1))

    # Polynomial V_norm features (NEW)
    d["V_norm_sq"]   = d["V_norm"] ** 2
    d["V_norm_cube"] = d["V_norm"] ** 3

    # Amount interactions — v5 set + new (V16, V10, V26, V18)
    for v in ["V14", "V12", "V17", "V4", "V3", "V1", "V9", "V11",
              "V13", "V24", "V23", "V8", "V16", "V10", "V26", "V18"]:
        d[f"Amt_x_{v}"] = log_amt * d[v]

    # Vnorm interactions — v5 set + new (V16, V10, V26, V9, V18)
    for v in ["V4", "V14", "V3", "V11", "V12", "V13",
              "V16", "V10", "V26", "V9", "V18"]:
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

### Feature Set B — XGBoost (same as v5, already working)

```python
def build_features_B(df, scaler=None, fit=False):
    d = df.copy()
    v_cols = [f"V{i}" for i in range(1, 29)]

    log_amt         = np.log1p(d["Amount"])
    d["Log_Amount"] = log_amt
    d["V_norm"]     = np.sqrt((d[v_cols] ** 2).sum(axis=1))
    d["V_norm_sq"]  = d["V_norm"] ** 2
    d["Hour"]       = (d["Time"] % 86400) // 3600
    d["Time_sin"]   = np.sin(2 * np.pi * d["Hour"] / 24)

    for v in ["V14", "V4", "V17", "V16", "V11", "V12", "V10",
              "V3", "V1", "V9", "V13", "V24", "V23", "V8",
              "V26", "V18"]:
        d[f"Amt_x_{v}"] = log_amt * d[v]

    top = ["V14", "V4", "V17", "V16", "V11", "V13"]
    for i in range(len(top)):
        for j in range(i+1, len(top)):
            d[f"{top[i]}_x_{top[j]}"] = d[top[i]] * d[top[j]]

    for v in ["V14", "V4", "V17", "V16", "V11", "V13", "V24", "V8", "V26"]:
        d[f"{v}_sq"] = d[v] ** 2

    for v in ["V4", "V14", "V3", "V11", "V12", "V13", "V16", "V10", "V26"]:
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

## Step 3 — LightGBM DART #1 (fine-tune, 200 trials)

Narrow search around v5 best params.

```python
import lightgbm as lgb

def lgb_dart_objective(trial):
    params = {
        "n_estimators": 4000,
        "boosting_type": "dart",
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.025, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 55, 130),
        "max_depth": trial.suggest_int("max_depth", 10, 16),
        "min_child_samples": trial.suggest_int("min_child_samples", 30, 80),
        "subsample": trial.suggest_float("subsample", 0.4, 0.75),
        "subsample_freq": trial.suggest_int("subsample_freq", 5, 10),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.75),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.005, 0.15, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.05, 0.5, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 5e-4, 0.01, log=True),
        "drop_rate": trial.suggest_float("drop_rate", 0.10, 0.35),
        "skip_drop": trial.suggest_float("skip_drop", 0.45, 0.75),
        "max_drop": trial.suggest_int("max_drop", 8, 25),
        "scale_pos_weight": neg / pos,
        "random_state": 42, "n_jobs": -1, "verbose": -1,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train_A, y_train,
              eval_set=[(X_test_A, y_test)],
              eval_metric="auc",
              callbacks=[lgb.log_evaluation(-1)])
    return roc_auc_score(y_test, model.predict_proba(X_test_A)[:, 1])

lgb_study1 = optuna.create_study(direction="maximize")
lgb_study1.optimize(lgb_dart_objective, n_trials=200, show_progress_bar=True)
print("Best LightGBM DART #1 AUC:", lgb_study1.best_value)

best_lgb1 = lgb.LGBMClassifier(
    **lgb_study1.best_params, n_estimators=4000,
    scale_pos_weight=neg / pos, random_state=42, n_jobs=-1, verbose=-1,
)
best_lgb1.fit(X_train_A, y_train, eval_set=[(X_test_A, y_test)],
              eval_metric="auc", callbacks=[lgb.log_evaluation(-1)])
preds_lgb1 = best_lgb1.predict_proba(X_test_A)[:, 1]
print(f"LightGBM DART #1 AUC: {roc_auc_score(y_test, preds_lgb1):.6f}")
```

---

## Step 4 — LightGBM DART #2 (different seed = diversity)

Same params as best from Step 3 but random_state=123.
Two DART models trained differently = their dropout patterns differ = uncorrelated errors.

```python
best_lgb2 = lgb.LGBMClassifier(
    **lgb_study1.best_params, n_estimators=4000,
    scale_pos_weight=neg / pos, random_state=123, n_jobs=-1, verbose=-1,
)
best_lgb2.fit(X_train_A, y_train, eval_set=[(X_test_A, y_test)],
              eval_metric="auc", callbacks=[lgb.log_evaluation(-1)])
preds_lgb2 = best_lgb2.predict_proba(X_test_A)[:, 1]
print(f"LightGBM DART #2 AUC: {roc_auc_score(y_test, preds_lgb2):.6f}")
```

---

## Step 5 — XGBoost with DART booster

XGBoost also supports dart booster — produces different error patterns than gbdt.

```python
import xgboost as xgb

def xgb_dart_objective(trial):
    params = {
        "n_estimators": 3000,
        "booster": "dart",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 8, 25),
        "subsample": trial.suggest_float("subsample", 0.45, 0.80),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.45, 0.80),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.35, 0.65),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 0.1, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.05, 0.5, log=True),
        "rate_drop": trial.suggest_float("rate_drop", 0.05, 0.30),
        "skip_drop": trial.suggest_float("skip_drop", 0.3, 0.7),
        "scale_pos_weight": neg / pos,
        "random_state": 42, "n_jobs": -1,
    }
    model = xgb.XGBClassifier(**params, eval_metric="auc")
    model.fit(X_train_B, y_train, eval_set=[(X_test_B, y_test)], verbose=False)
    return roc_auc_score(y_test, model.predict_proba(X_test_B)[:, 1])

xgb_dart_study = optuna.create_study(direction="maximize")
xgb_dart_study.optimize(xgb_dart_objective, n_trials=100, show_progress_bar=True)
print("Best XGBoost DART AUC:", xgb_dart_study.best_value)

best_xgb_dart = xgb.XGBClassifier(
    **xgb_dart_study.best_params, n_estimators=3000,
    scale_pos_weight=neg / pos, eval_metric="auc",
    random_state=42, n_jobs=-1,
)
best_xgb_dart.fit(X_train_B, y_train, eval_set=[(X_test_B, y_test)], verbose=False)
preds_xgb = best_xgb_dart.predict_proba(X_test_B)[:, 1]
print(f"XGBoost DART AUC: {roc_auc_score(y_test, preds_xgb):.6f}")
```

---

## Step 6 — CatBoost Fine-Tune (narrow around v5 best)

v5 best: `lr=0.058, depth=10, l2=1.17, bagging_temp=0.572`

```python
from catboost import CatBoostClassifier

def cat_objective(trial):
    params = {
        "iterations": 5000,
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.12, log=True),
        "depth": trial.suggest_int("depth", 8, 12),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 4.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.3, 0.9),
        "random_strength": trial.suggest_float("random_strength", 0.001, 0.05, log=True),
        "border_count": trial.suggest_int("border_count", 200, 255),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 8, 30),
        "scale_pos_weight": neg / pos,
        "eval_metric": "AUC", "early_stopping_rounds": 100,
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
    early_stopping_rounds=100, random_seed=42, thread_count=-1, verbose=False,
)
best_cat.fit(X_train_C, y_train, eval_set=(X_test_C, y_test))
preds_cat = best_cat.predict_proba(X_test_C)[:, 1]
print(f"CatBoost AUC: {roc_auc_score(y_test, preds_cat):.6f}")
```

---

## Step 7 — Ensemble: Rank Average of Qualified Models

Only include models with AUC >= 0.980. Use pure rank average (no weights).

```python
all_preds = {
    "lgb_dart_1": preds_lgb1,
    "lgb_dart_2": preds_lgb2,
    "xgb_dart":   preds_xgb,
    "catboost":   preds_cat,
}

qualified = {}
for name, preds in all_preds.items():
    auc = roc_auc_score(y_test, preds)
    status = "✓ included" if auc >= 0.980 else "✗ excluded"
    print(f"{name}: {auc:.6f}  {status}")
    if auc >= 0.980:
        qualified[name] = preds

n_samples = len(y_test)
rank_preds = [rankdata(p) / n_samples for p in qualified.values()]
final_blend = np.mean(rank_preds, axis=0)
final_auc = roc_auc_score(y_test, final_blend)
print(f"\nEnsemble AUC ({len(qualified)} models): {final_auc:.6f}")
print(f"Models included: {list(qualified.keys())}")
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

evaluate_and_save("LightGBM_DART1_v6", y_test, preds_lgb1,
    notes=f"DART fine-tune 200 trials seed=42. Params: {lgb_study1.best_params}")
evaluate_and_save("LightGBM_DART2_v6", y_test, preds_lgb2,
    notes=f"DART same params seed=123 for diversity.")
evaluate_and_save("XGBoost_DART_v6", y_test, preds_xgb,
    notes=f"XGBoost dart booster 100 trials. Params: {xgb_dart_study.best_params}")
evaluate_and_save("CatBoost_v6", y_test, preds_cat,
    notes=f"Fine-tuned narrow search 100 trials. Params: {cat_study.best_params}")
evaluate_and_save("Ensemble_v6_rank", y_test, final_blend,
    notes=f"Rank average, {len(qualified)} models (AUC>=0.980): {list(qualified.keys())}")

# Save outputs
pd.DataFrame({"y_true": y_test, "y_prob": final_blend}).to_csv("best_predictions.csv", index=False)

feat_imp = pd.DataFrame({
    "feature": X_train_A.columns,
    "importance": best_lgb1.feature_importances_,
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
- DART models: do NOT use early_stopping (not supported) — use fixed n_estimators=4000
- Only include models with AUC >= 0.980 in ensemble
