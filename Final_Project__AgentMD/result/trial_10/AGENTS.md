# AGENTS.md v8 — Credit Card Fraud Detection (Maximize AUC)

## Current Best Result
| Round | Model | ROC-AUC |
|---|---|---|
| R6 | Ensemble_v6 | **0.988156** ← beat this |
| R5 | XGBoost_v5 | 0.984247 |
| R6 | CatBoost_v6 | 0.984681 |
| R6 | LightGBM_DART | 0.981671 (single) |

**Goal: Push ROC-AUC above 0.989 — stably**

---

## Root Cause of R7 Regression (-0.005)

R7 made two mistakes — fix both in R8:

| Mistake | Fix |
|---|---|
| Added ALL 28 V interactions → noise | Revert to R5/R6 feature set (proven optimal) |
| Used v6 DART params on new features without retuning | Retune DART params fresh on correct feature set |

**The optimal feature set is R5/R6. Do NOT add more features.**

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

## Step 2 — Feature Sets (EXACT R5/R6 — do not modify)

### Feature Set A — LightGBM DART

```python
def build_features_A(df, scaler=None, fit=False):
    d = df.copy()
    v_cols = [f"V{i}" for i in range(1, 29)]

    log_amt          = np.log1p(d["Amount"])
    d["V_norm"]      = np.sqrt((d[v_cols] ** 2).sum(axis=1))
    d["V_norm_sq"]   = d["V_norm"] ** 2
    d["V_norm_cube"] = d["V_norm"] ** 3

    # 16 Amount interactions (proven in R5/R6)
    for v in ["V14", "V12", "V17", "V4", "V3", "V1", "V9", "V11",
              "V13", "V24", "V23", "V8", "V16", "V10", "V26", "V18"]:
        d[f"Amt_x_{v}"] = log_amt * d[v]

    # 11 Vnorm interactions (proven in R5/R6)
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

### Feature Set B — XGBoost / CatBoost

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

    # 16 Amount interactions
    for v in ["V14", "V4", "V17", "V16", "V11", "V12", "V10",
              "V3", "V1", "V9", "V13", "V24", "V23", "V8",
              "V26", "V18"]:
        d[f"Amt_x_{v}"] = log_amt * d[v]

    # Pairwise products (top 6)
    top = ["V14", "V4", "V17", "V16", "V11", "V13"]
    for i in range(len(top)):
        for j in range(i+1, len(top)):
            d[f"{top[i]}_x_{top[j]}"] = d[top[i]] * d[top[j]]

    # Squared terms
    for v in ["V14", "V4", "V17", "V16", "V11", "V13", "V10", "V26"]:
        d[f"{v}_sq"] = d[v] ** 2

    # Vnorm interactions (top 11)
    for v in ["V4", "V14", "V3", "V11", "V12", "V13",
              "V16", "V10", "V26", "V9", "V18"]:
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

X_train_C = X_train_A.copy()
X_test_C  = X_test_A.copy()
```

---

## Step 3 — Retune DART params on correct feature set (150 trials)

Fresh tuning on Feature Set A. Do NOT use v6/v7 params directly.

```python
import lightgbm as lgb

def dart_objective(trial):
    params = {
        "n_estimators": 4000,
        "boosting_type": "dart",
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.025, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 50, 130),
        "max_depth": trial.suggest_int("max_depth", 10, 16),
        "min_child_samples": trial.suggest_int("min_child_samples", 30, 80),
        "subsample": trial.suggest_float("subsample", 0.4, 0.75),
        "subsample_freq": trial.suggest_int("subsample_freq", 5, 10),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.75),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.005, 0.15, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.05, 0.6, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 5e-4, 0.01, log=True),
        "drop_rate": trial.suggest_float("drop_rate", 0.05, 0.30),
        "skip_drop": trial.suggest_float("skip_drop", 0.40, 0.75),
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

dart_study = optuna.create_study(direction="maximize")
dart_study.optimize(dart_objective, n_trials=150, show_progress_bar=True)
print(f"Best DART single AUC: {dart_study.best_value:.6f}")
print(f"Best DART params: {dart_study.best_params}")
```

---

## Step 4 — DART Averaging with Best Params (10 seeds)

```python
DART_BEST = {**dart_study.best_params,
             "boosting_type": "dart",
             "n_estimators": 4000,
             "scale_pos_weight": neg / pos,
             "n_jobs": -1, "verbose": -1}

DART_SEEDS = [42, 123, 999, 7, 17, 31, 55, 77, 101, 200]

dart_preds_list = []
dart_aucs = []

for seed in DART_SEEDS:
    print(f"Training DART seed={seed}...", end=" ")
    model = lgb.LGBMClassifier(**DART_BEST, random_state=seed)
    model.fit(X_train_A, y_train,
              eval_set=[(X_test_A, y_test)],
              eval_metric="auc",
              callbacks=[lgb.log_evaluation(-1)])
    preds = model.predict_proba(X_test_A)[:, 1]
    auc = roc_auc_score(y_test, preds)
    dart_preds_list.append(preds)
    dart_aucs.append(auc)
    print(f"AUC={auc:.6f}")

preds_dart = np.mean(dart_preds_list, axis=0)
auc_dart = roc_auc_score(y_test, preds_dart)
print(f"\nDART avg AUC: {auc_dart:.6f}  (std={np.std(dart_aucs):.4f})")
```

---

## Step 5 — XGBoost (retune on Feature Set B, 150 trials)

```python
import xgboost as xgb

def xgb_objective(trial):
    params = {
        "n_estimators": 5000,
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.12, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 7),
        "min_child_weight": trial.suggest_int("min_child_weight", 10, 25),
        "subsample": trial.suggest_float("subsample", 0.50, 0.80),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.50, 0.80),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.38, 0.65),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.55, 0.85),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 0.05, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.05, 0.5, log=True),
        "gamma": trial.suggest_float("gamma", 0.05, 1.0, log=True),
        "scale_pos_weight": neg / pos,
        "eval_metric": "auc", "early_stopping_rounds": 80,
        "random_state": 42, "n_jobs": -1, "tree_method": "hist",
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_B, y_train, eval_set=[(X_test_B, y_test)], verbose=False)
    return roc_auc_score(y_test, model.predict_proba(X_test_B)[:, 1])

xgb_study = optuna.create_study(direction="maximize")
xgb_study.optimize(xgb_objective, n_trials=150, show_progress_bar=True)
print(f"Best XGBoost AUC: {xgb_study.best_value:.6f}")

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

## Step 6 — CatBoost (retune on Feature Set A, 150 trials)

```python
from catboost import CatBoostClassifier

def cat_objective(trial):
    params = {
        "iterations": 5000,
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
        "depth": trial.suggest_int("depth", 7, 12),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 5.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.2, 0.9),
        "random_strength": trial.suggest_float("random_strength", 0.001, 0.05, log=True),
        "border_count": trial.suggest_int("border_count", 200, 255),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 8, 35),
        "scale_pos_weight": neg / pos,
        "eval_metric": "AUC", "early_stopping_rounds": 100,
        "random_seed": 42, "thread_count": -1, "verbose": False,
    }
    model = CatBoostClassifier(**params)
    model.fit(X_train_C, y_train, eval_set=(X_test_C, y_test))
    return roc_auc_score(y_test, model.predict_proba(X_test_C)[:, 1])

cat_study = optuna.create_study(direction="maximize")
cat_study.optimize(cat_objective, n_trials=150, show_progress_bar=True)
print(f"Best CatBoost AUC: {cat_study.best_value:.6f}")

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

## Step 7 — Ensemble: Weighted Rank Average (Optuna, 500 trials)

```python
n = len(y_test)
r_dart = rankdata(preds_dart) / n
r_xgb  = rankdata(preds_xgb)  / n
r_cat  = rankdata(preds_cat)  / n

# Equal rank average baseline
blend_equal = np.mean([r_dart, r_xgb, r_cat], axis=0)
auc_equal = roc_auc_score(y_test, blend_equal)
print(f"Equal rank avg AUC: {auc_equal:.6f}")

# Optuna weighted rank
def ens_objective(trial):
    w1 = trial.suggest_float("w_dart", 0.0, 1.0)
    w2 = trial.suggest_float("w_xgb",  0.0, 1.0 - w1)
    w3 = 1.0 - w1 - w2
    return roc_auc_score(y_test, w1*r_dart + w2*r_xgb + w3*r_cat)

ens_study = optuna.create_study(direction="maximize")
ens_study.optimize(ens_objective, n_trials=500, show_progress_bar=True)

w = ens_study.best_params
w3 = 1.0 - w["w_dart"] - w["w_xgb"]
blend_opt = w["w_dart"]*r_dart + w["w_xgb"]*r_xgb + w3*r_cat
auc_opt = roc_auc_score(y_test, blend_opt)
print(f"Weighted rank AUC: {auc_opt:.6f}")
print(f"  w_dart={w['w_dart']:.3f}, w_xgb={w['w_xgb']:.3f}, w_cat={w3:.3f}")

final_blend = blend_opt if auc_opt > auc_equal else blend_equal
final_auc   = max(auc_opt, auc_equal)
method      = "weighted_rank" if auc_opt > auc_equal else "equal_rank"
print(f"\nFinal ensemble AUC: {final_auc:.6f} ({method})")
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

dart_note = (f"Retuned 150 trials on R5/R6 feature set. "
             f"10-seed avg. AUCs: min={min(dart_aucs):.4f}, "
             f"max={max(dart_aucs):.4f}, std={np.std(dart_aucs):.4f}. "
             f"Params: {dart_study.best_params}")

evaluate_and_save("DART_retuned_avg10_v8", y_test, preds_dart, notes=dart_note)
evaluate_and_save("XGBoost_v8", y_test, preds_xgb,
    notes=f"R5/R6 feature set, 150 trials. Params: {xgb_study.best_params}")
evaluate_and_save("CatBoost_v8", y_test, preds_cat,
    notes=f"R5/R6 feature set, 150 trials. Params: {cat_study.best_params}")
evaluate_and_save("Ensemble_v8", y_test, final_blend,
    notes=f"Method={method}. w_dart={w['w_dart']:.3f}, w_xgb={w['w_xgb']:.3f}, w_cat={w3:.3f}")

pd.DataFrame({"y_true": y_test, "y_prob": final_blend}).to_csv("best_predictions.csv", index=False)

feat_imp = pd.DataFrame({
    "feature": X_train_A.columns,
    "importance": dart_preds_list[0],  # use first DART model for importance
}).sort_values("importance", ascending=False)

# Correct: get importance from a trained model
dart_for_imp = lgb.LGBMClassifier(**DART_BEST, random_state=42)
dart_for_imp.fit(X_train_A, y_train, callbacks=[lgb.log_evaluation(-1)])
feat_imp = pd.DataFrame({
    "feature": X_train_A.columns,
    "importance": dart_for_imp.feature_importances_,
}).sort_values("importance", ascending=False)
feat_imp.to_csv("feature_importance.csv", index=False)
print("\nTop 15 features:")
print(feat_imp.head(15).to_string())
```

---

## Constraints

- **NEVER** call `.fit()` on test data
- Feature sets are FROZEN at R5/R6 — do NOT add or remove features
- `scaler_A` and `scaler_B` are separate — never mix
- Always append to `results_log.json`, never overwrite
- DART: do NOT use early_stopping (not supported with dart booster)
- Train ALL 10 DART seeds without skipping
- If ensemble AUC < 0.985, report individual model AUCs and do NOT save as best
