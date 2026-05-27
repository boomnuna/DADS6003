import pandas as pd
import numpy as np
import subprocess, sys, os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, classification_report
from scipy.optimize import minimize

# ─────────────────────────────────────────
# 0. GPU DETECTION
# ─────────────────────────────────────────
def detect_gpu():
    """Auto-detect GPU. Returns device string for LGB/XGB/CAT."""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected — using GPU acceleration")
            return "gpu"
    except FileNotFoundError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA GPU detected: {torch.cuda.get_device_name(0)}")
            return "gpu"
    except ImportError:
        pass
    print("⚠️  No GPU found — falling back to CPU (still fast with histogram methods)")
    return "cpu"

DEVICE = detect_gpu()
TREE_METHOD = "gpu_hist" if DEVICE == "gpu" else "hist"   # XGBoost
LGB_DEVICE   = "gpu"     if DEVICE == "gpu" else "cpu"    # LightGBM
CAT_TASK     = "GPU"     if DEVICE == "gpu" else "CPU"    # CatBoost

print(f"\n{'='*55}")
print(f"  Device : {DEVICE.upper()}")
print(f"{'='*55}\n")

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
print("📂 Loading data...")
train = pd.read_csv("creditcard_train.csv")

# Support both with/without Class column in test
if os.path.exists("creditcard_test.csv"):
    test = pd.read_csv("creditcard_test.csv")
    has_test_labels = "Class" in test.columns
    print(f"   Test file found  — labels: {'YES' if has_test_labels else 'NO'}")
else:
    print("   ⚠️  creditcard_test.csv not found — skipping final eval")
    test = None
    has_test_labels = False

print(f"   Train shape : {train.shape}")
if test is not None:
    print(f"   Test  shape : {test.shape}")

# ─────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────
def engineer_features(df):
    df = df.copy()
    df["log_amount"]  = np.log1p(df["Amount"])
    df["time_sin"]    = np.sin(2 * np.pi * df["Time"] / 86400)
    df["time_cos"]    = np.cos(2 * np.pi * df["Time"] / 86400)
    df["hour"]        = (df["Time"] // 3600) % 24
    # Amount × top fraud-correlated PCA components
    for col in ["V1","V2","V3","V4","V10","V11","V12","V14","V17"]:
        df[f"{col}_x_logamt"] = df[col] * df["log_amount"]
    # Additional: rolling z-score of amount by hour
    df["amt_zscore"]  = (df["Amount"] - df["Amount"].mean()) / (df["Amount"].std() + 1e-8)
    df = df.drop(columns=["Time", "Amount"])
    return df

print("\n🔧 Engineering features...")
X_train = engineer_features(train.drop(columns=["Class"]))
y_train = train["Class"]

if test is not None:
    X_test = engineer_features(test.drop(columns=["Class"]) if has_test_labels else test)
    y_test = test["Class"] if has_test_labels else None

scale = (y_train == 0).sum() / (y_train == 1).sum()
print(f"   Features    : {X_train.shape[1]}")
print(f"   scale_pos_w : {scale:.1f}  (fraud ratio: {1/scale*100:.3f}%)")

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ─────────────────────────────────────────
# 3a. LIGHTGBM
# ─────────────────────────────────────────
import lightgbm as lgb

print("\n🌲 Training LightGBM...")
lgb_params = dict(
    objective="binary", metric="auc",
    n_estimators=3000, learning_rate=0.02,
    num_leaves=127, max_depth=-1,
    min_child_samples=20,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    scale_pos_weight=scale,
    lambda_l1=0.1, lambda_l2=10.0,
    device=LGB_DEVICE,
    random_state=42, n_jobs=-1, verbose=-1,
)

lgb_oof  = np.zeros(len(X_train))
lgb_test = np.zeros(len(X_test)) if test is not None else None

for fold, (tr_idx, val_idx) in enumerate(CV.split(X_train, y_train)):
    m = lgb.LGBMClassifier(**lgb_params)
    m.fit(
        X_train.iloc[tr_idx], y_train.iloc[tr_idx],
        eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)],
    )
    lgb_oof[val_idx] = m.predict_proba(X_train.iloc[val_idx])[:, 1]
    if test is not None:
        lgb_test += m.predict_proba(X_test)[:, 1] / 5
    print(f"   Fold {fold+1}  AUC: {roc_auc_score(y_train.iloc[val_idx], lgb_oof[val_idx]):.6f}")

lgb_auc = roc_auc_score(y_train, lgb_oof)
print(f"   ✅ LGB OOF AUC: {lgb_auc:.6f}")

# ─────────────────────────────────────────
# 3b. XGBOOST
# ─────────────────────────────────────────
import xgboost as xgb

print("\n🌳 Training XGBoost...")
xgb_params = dict(
    objective="binary:logistic", eval_metric="auc",
    n_estimators=3000, learning_rate=0.02,
    max_depth=6, min_child_weight=5,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale,
    reg_alpha=0.1, reg_lambda=10.0,
    tree_method=TREE_METHOD,
    early_stopping_rounds=100,
    random_state=42, n_jobs=-1,
)

xgb_oof  = np.zeros(len(X_train))
xgb_test = np.zeros(len(X_test)) if test is not None else None

for fold, (tr_idx, val_idx) in enumerate(CV.split(X_train, y_train)):
    m = xgb.XGBClassifier(**xgb_params)
    m.fit(
        X_train.iloc[tr_idx], y_train.iloc[tr_idx],
        eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
        verbose=False,
    )
    xgb_oof[val_idx] = m.predict_proba(X_train.iloc[val_idx])[:, 1]
    if test is not None:
        xgb_test += m.predict_proba(X_test)[:, 1] / 5
    print(f"   Fold {fold+1}  AUC: {roc_auc_score(y_train.iloc[val_idx], xgb_oof[val_idx]):.6f}")

xgb_auc = roc_auc_score(y_train, xgb_oof)
print(f"   ✅ XGB OOF AUC: {xgb_auc:.6f}")

# ─────────────────────────────────────────
# 3c. CATBOOST
# ─────────────────────────────────────────
from catboost import CatBoostClassifier

print("\n🐱 Training CatBoost...")
cat_oof  = np.zeros(len(X_train))
cat_test = np.zeros(len(X_test)) if test is not None else None

for fold, (tr_idx, val_idx) in enumerate(CV.split(X_train, y_train)):
    m = CatBoostClassifier(
        iterations=3000, learning_rate=0.02, depth=6,
        scale_pos_weight=scale, eval_metric="AUC",
        early_stopping_rounds=100,
        task_type=CAT_TASK,
        random_seed=42, verbose=0,
    )
    m.fit(
        X_train.iloc[tr_idx], y_train.iloc[tr_idx],
        eval_set=(X_train.iloc[val_idx], y_train.iloc[val_idx]),
    )
    cat_oof[val_idx] = m.predict_proba(X_train.iloc[val_idx])[:, 1]
    if test is not None:
        cat_test += m.predict_proba(X_test)[:, 1] / 5
    print(f"   Fold {fold+1}  AUC: {roc_auc_score(y_train.iloc[val_idx], cat_oof[val_idx]):.6f}")

cat_auc = roc_auc_score(y_train, cat_oof)
print(f"   ✅ CAT OOF AUC: {cat_auc:.6f}")

# ─────────────────────────────────────────
# 4. OPTIMAL BLENDING
# ─────────────────────────────────────────
print("\n⚖️  Finding optimal blend weights...")

def neg_auc(w):
    w = np.abs(w) / np.abs(w).sum()
    return -roc_auc_score(y_train, w[0]*lgb_oof + w[1]*xgb_oof + w[2]*cat_oof)

result = minimize(neg_auc, [1/3, 1/3, 1/3], method="Nelder-Mead")
best_w = np.abs(result.x) / np.abs(result.x).sum()
print(f"   Weights → LGB:{best_w[0]:.3f}  XGB:{best_w[1]:.3f}  CAT:{best_w[2]:.3f}")

oof_blend = best_w[0]*lgb_oof + best_w[1]*xgb_oof + best_w[2]*cat_oof
blend_auc = roc_auc_score(y_train, oof_blend)
print(f"   ✅ Blended OOF AUC: {blend_auc:.6f}")

# Best threshold from OOF
thresholds = np.linspace(0.01, 0.99, 500)
f1s = [f1_score(y_train, oof_blend >= t) for t in thresholds]
best_thresh = thresholds[np.argmax(f1s)]
print(f"   Best threshold (OOF F1): {best_thresh:.4f}")

# ─────────────────────────────────────────
# 5. FINAL TEST EVALUATION
# ─────────────────────────────────────────
if test is not None:
    test_blend = best_w[0]*lgb_test + best_w[1]*xgb_test + best_w[2]*cat_test
    y_pred = (test_blend >= best_thresh).astype(int)

    print(f"\n{'='*55}")
    if has_test_labels:
        test_auc   = roc_auc_score(y_test, test_blend)
        test_prauc = average_precision_score(y_test, test_blend)
        print(f"  TEST ROC-AUC  : {test_auc:.6f}   ← primary metric")
        print(f"  TEST PR-AUC   : {test_prauc:.6f}")
        print(f"{'='*55}")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, digits=4))
    else:
        print(f"  OOF ROC-AUC (proxy) : {blend_auc:.6f}")
        print(f"  Test labels not available — saved predictions only")
        print(f"{'='*55}")

    # Save predictions
    out = pd.DataFrame({"Class_proba": test_blend, "Class_pred": y_pred})
    out.to_csv("creditcard_test_predictions.csv", index=False)
    print(f"\n💾 Saved: creditcard_test_predictions.csv  ({len(out)} rows)")

print("\n✅ Pipeline complete!")
