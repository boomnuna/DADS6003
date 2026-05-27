# AGENTS.md — Credit Card Fraud Detection (Maximize AUC)

## Mission
Train a binary classification model on `creditcard_train.csv` and evaluate it on `creditcard_test.csv`.
**Primary metric: ROC-AUC** (maximize). Secondary: PR-AUC (report but do not optimize directly).

---

## Dataset Facts
| Property | Value |
|---|---|
| Train rows | 199,364 |
| Features | `Time`, `V1`–`V28` (PCA-anonymized), `Amount` |
| Target | `Class` (0 = legitimate, 1 = fraud) |
| Class ratio | 344 fraud / 199,020 legit ≈ **0.17% positive rate** |
| Missing values | None |

---

## Step 0 — Environment Setup

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost optuna scipy
```

---

## Step 0.5 — GPU Detection (Auto, run before anything else)

```python
import subprocess

def detect_gpu():
    """Returns 'gpu' if NVIDIA GPU found, else 'cpu'."""
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if r.returncode == 0:
            print("✅ NVIDIA GPU detected — using GPU acceleration")
            return "gpu"
    except FileNotFoundError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA GPU: {torch.cuda.get_device_name(0)}")
            return "gpu"
    except ImportError:
        pass
    print("⚠️  No GPU — falling back to CPU")
    return "cpu"

DEVICE     = detect_gpu()
TREE_METHOD = "gpu_hist" if DEVICE == "gpu" else "hist"   # XGBoost
LGB_DEVICE  = "gpu"      if DEVICE == "gpu" else "cpu"    # LightGBM
CAT_TASK    = "GPU"      if DEVICE == "gpu" else "CPU"    # CatBoost
```

Pass `LGB_DEVICE`, `TREE_METHOD`, `CAT_TASK` into every model constructor.

---

## Step 1 — Feature Engineering

```python
import pandas as pd, numpy as np

def engineer_features(df):
    df = df.copy()
    df["log_amount"]  = np.log1p(df["Amount"])
    df["time_sin"]    = np.sin(2 * np.pi * df["Time"] / 86400)
    df["time_cos"]    = np.cos(2 * np.pi * df["Time"] / 86400)
    df["hour"]        = (df["Time"] // 3600) % 24
    df["amt_zscore"]  = (df["Amount"] - df["Amount"].mean()) / (df["Amount"].std() + 1e-8)
    for col in ["V1","V2","V3","V4","V10","V11","V12","V14","V17"]:
        df[f"{col}_x_logamt"] = df[col] * df["log_amount"]
    return df.drop(columns=["Time", "Amount"])

train = pd.read_csv("creditcard_train.csv")
test  = pd.read_csv("creditcard_test.csv")

X_train = engineer_features(train.drop(columns=["Class"]))
y_train = train["Class"]
X_test  = engineer_features(test.drop(columns=["Class"]) if "Class" in test.columns else test)
y_test  = test["Class"] if "Class" in test.columns else None

scale = (y_train == 0).sum() / (y_train == 1).sum()  # ≈ 578.5
```

---

## Step 2 — Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

---

## Step 3 — Train Three Models

### 3a. LightGBM

```python
import lightgbm as lgb

lgb_oof  = np.zeros(len(X_train))
lgb_test = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(CV.split(X_train, y_train)):
    m = lgb.LGBMClassifier(
        objective="binary", metric="auc", n_estimators=3000,
        learning_rate=0.02, num_leaves=127, min_child_samples=20,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        scale_pos_weight=scale, lambda_l1=0.1, lambda_l2=10.0,
        device=LGB_DEVICE, random_state=42, n_jobs=-1, verbose=-1,
    )
    m.fit(
        X_train.iloc[tr_idx], y_train.iloc[tr_idx],
        eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)],
    )
    lgb_oof[val_idx] = m.predict_proba(X_train.iloc[val_idx])[:, 1]
    lgb_test        += m.predict_proba(X_test)[:, 1] / 5
```

### 3b. XGBoost (v3+ API: early_stopping_rounds in constructor)

```python
import xgboost as xgb

xgb_oof  = np.zeros(len(X_train))
xgb_test = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(CV.split(X_train, y_train)):
    m = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="auc",
        n_estimators=3000, learning_rate=0.02,
        max_depth=6, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale, reg_alpha=0.1, reg_lambda=10.0,
        tree_method=TREE_METHOD, early_stopping_rounds=100,
        random_state=42, n_jobs=-1,
    )
    m.fit(
        X_train.iloc[tr_idx], y_train.iloc[tr_idx],
        eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
        verbose=False,
    )
    xgb_oof[val_idx] = m.predict_proba(X_train.iloc[val_idx])[:, 1]
    xgb_test        += m.predict_proba(X_test)[:, 1] / 5
```

### 3c. CatBoost

```python
from catboost import CatBoostClassifier

cat_oof  = np.zeros(len(X_train))
cat_test = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(CV.split(X_train, y_train)):
    m = CatBoostClassifier(
        iterations=3000, learning_rate=0.02, depth=6,
        scale_pos_weight=scale, eval_metric="AUC",
        early_stopping_rounds=100, task_type=CAT_TASK,
        random_seed=42, verbose=0,
    )
    m.fit(
        X_train.iloc[tr_idx], y_train.iloc[tr_idx],
        eval_set=(X_train.iloc[val_idx], y_train.iloc[val_idx]),
    )
    cat_oof[val_idx] = m.predict_proba(X_train.iloc[val_idx])[:, 1]
    cat_test        += m.predict_proba(X_test)[:, 1] / 5
```

---

## Step 4 — Optimal Blending

```python
from scipy.optimize import minimize

def neg_auc(w):
    w = np.abs(w) / np.abs(w).sum()
    return -roc_auc_score(y_train, w[0]*lgb_oof + w[1]*xgb_oof + w[2]*cat_oof)

result = minimize(neg_auc, [1/3, 1/3, 1/3], method="Nelder-Mead")
best_w = np.abs(result.x) / np.abs(result.x).sum()

test_blend = best_w[0]*lgb_test + best_w[1]*xgb_test + best_w[2]*cat_test
```

---

## Step 5 — Threshold & Output

```python
from sklearn.metrics import f1_score

thresholds = np.linspace(0.01, 0.99, 500)
oof_blend  = best_w[0]*lgb_oof + best_w[1]*xgb_oof + best_w[2]*cat_oof
f1s = [f1_score(y_train, oof_blend >= t) for t in thresholds]
best_thresh = thresholds[np.argmax(f1s)]

y_pred = (test_blend >= best_thresh).astype(int)

pd.DataFrame({"Class_proba": test_blend, "Class_pred": y_pred}) \
  .to_csv("creditcard_test_predictions.csv", index=False)
```

---

## Step 6 — Final Evaluation

```python
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix

if y_test is not None:
    print(f"TEST ROC-AUC : {roc_auc_score(y_test, test_blend):.6f}")
    print(f"TEST PR-AUC  : {average_precision_score(y_test, test_blend):.6f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))
```

---

## OOF Results (from last run)

| Model | OOF AUC |
|---|---|
| LightGBM | 0.960888 |
| XGBoost | 0.951164 |
| CatBoost | 0.959361 |
| **Blended** | **0.964811** |
| Optimal weights | LGB:0.473 XGB:0.002 CAT:0.526 |
| Best threshold | 0.8997 |

---

## Rules for the Agent

1. **Run GPU detection (Step 0.5) before any model training.**
2. Always use `scale_pos_weight = n_negative / n_positive` (≈ 578.5).
3. Never resample (no SMOTE/undersampling) — distorts AUC.
4. Use OOF predictions for blending weights and threshold — no leakage.
5. XGBoost v3+: put `early_stopping_rounds` in the **constructor**, not `.fit()`.
6. Save output to `creditcard_test_predictions.csv` with columns `Class_proba`, `Class_pred`.
7. Print `TEST ROC-AUC` clearly at the end.

---

## File Structure

```
./
├── AGENTS.md                        ← this file
├── train.py                         ← generated pipeline script
├── creditcard_train.csv
├── creditcard_test.csv
└── creditcard_test_predictions.csv  ← output
```
