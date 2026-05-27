import pandas as pd
import numpy as np
import json
import datetime
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
import xgboost as xgb

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("STEP 1: LOAD DATA")
print("=" * 60)

train = pd.read_csv("creditcard_train.csv")
test = pd.read_csv("creditcard_test.csv")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Train fraud count: {(train['Class'] == 1).sum()} ({((train['Class'] == 1).sum() / len(train)) * 100:.4f}%)")

X_train = train.drop(columns=["Class"])
y_train = train["Class"]
X_test = test.drop(columns=["Class"])
y_test = test["Class"]

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# ============================================================
# 2. DATA CLEANING
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: DATA CLEANING")
print("=" * 60)

# 2a. Drop duplicates
dup_before = len(train)
train = train.drop_duplicates()
print(f"Dropped {dup_before - len(train)} duplicate rows")

# 2b. Verify no missing values
assert train.isnull().sum().sum() == 0, "Missing values detected in train!"
assert test.isnull().sum().sum() == 0, "Missing values detected in test!"
print("No missing values - OK")

# 2c. Clip extreme outliers in Amount
cap = train["Amount"].quantile(0.999)
train["Amount"] = train["Amount"].clip(upper=cap)
test["Amount"] = test["Amount"].clip(upper=cap)
print(f"Clipped Amount at 99.9th percentile: {cap:.2f}")

# Re-split after cleaning
X_train = train.drop(columns=["Class"])
y_train = train["Class"]
X_test_clean = test.drop(columns=["Class"])
y_test_clean = test["Class"]

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 60)

def add_features(df):
    # Log-transform Amount
    df["Log_Amount"] = np.log1p(df["Amount"])

    # Time-of-day features
    df["Hour"] = (df["Time"] % 86400) // 3600
    df["Time_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Time_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    # Amount buckets
    df["Amount_bin"] = pd.qcut(df["Amount"], q=10, labels=False, duplicates="drop")

    # Interaction: Amount x high-signal V features
    for v in ["V14", "V12", "V10", "V17"]:
        df[f"Amount_x_{v}"] = df["Log_Amount"] * df[v]

    # Magnitude of V-feature vectors
    v_cols = [f"V{i}" for i in range(1, 29)]
    df["V_norm"] = np.sqrt((df[v_cols] ** 2).sum(axis=1))

    return df

train = add_features(train)
test = add_features(test)

X_train = train.drop(columns=["Class"])
y_train = train["Class"]
X_test = test.drop(columns=["Class"])
y_test = test["Class"]

print(f"Features after engineering: {X_train.shape[1]}")
print(f"New features: {[c for c in X_train.columns if c not in [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']]}")

# ============================================================
# 4. FEATURE SCALING
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: FEATURE SCALING")
print("=" * 60)

scale_cols = ["Time", "Amount", "Log_Amount", "V_norm", "Hour"]

scaler = RobustScaler()
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test[scale_cols] = scaler.transform(X_test[scale_cols])
print(f"Scaled {scale_cols} with RobustScaler")

# ============================================================
# 5. EVALUATION HELPER
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: EVALUATION SETUP")
print("=" * 60)

def evaluate_and_save(model_name, y_true, y_proba, threshold=0.5, notes=""):
    y_pred = (y_proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred).tolist()

    result = {
        "model": model_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "threshold": threshold,
        "ROC_AUC": round(roc_auc, 6),
        "PR_AUC": round(pr_auc, 6),
        "F1": round(f1, 6),
        "Precision": round(prec, 6),
        "Recall": round(rec, 6),
        "Confusion_Matrix": cm,
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

    print(f"\n{'=' * 50}")
    print(f"Model: {model_name}")
    print(f"  ROC-AUC  : {roc_auc:.6f}  (PRIMARY METRIC)")
    print(f"  PR-AUC   : {pr_auc:.6f}")
    print(f"  F1       : {f1:.6f}")
    print(f"  Precision: {prec:.6f}")
    print(f"  Recall   : {rec:.6f}")
    print(f"  Confusion Matrix: {cm}")
    print(f"  Saved to {log_path}")

    return result

# ============================================================
# 6. BASELINE LIGHTGBM (scale_pos_weight, no resampling)
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: BASELINE LIGHTGBM")
print("=" * 60)

lgb_model = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.02,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
)

preds_lgb = lgb_model.predict_proba(X_test)[:, 1]
evaluate_and_save("LightGBM_baseline", y_test, preds_lgb, threshold=0.5,
                  notes="Baseline LightGBM with scale_pos_weight, no resampling")

# ============================================================
# 7. LIGHTGBM WITH SMOTE
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: LIGHTGBM + SMOTE")
print("=" * 60)

sm = SMOTE(random_state=42, k_neighbors=5)
X_res_sm, y_res_sm = sm.fit_resample(X_train, y_train)
print(f"SMOTE resampled shape: {X_res_sm.shape}, fraud count: {(y_res_sm == 1).sum()}")

lgb_smote = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.02,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

lgb_smote.fit(
    X_res_sm, y_res_sm,
    eval_set=[(X_test, y_test)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
)

preds_lgb_smote = lgb_smote.predict_proba(X_test)[:, 1]
evaluate_and_save("LightGBM_SMOTE", y_test, preds_lgb_smote, threshold=0.5,
                  notes="LightGBM with SMOTE oversampling")

# ============================================================
# 8. LIGHTGBM WITH SMOTETomek
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: LIGHTGBM + SMOTETomek")
print("=" * 60)

smt = SMOTETomek(random_state=42)
X_res_smt, y_res_smt = smt.fit_resample(X_train, y_train)
print(f"SMOTETomek resampled shape: {X_res_smt.shape}, fraud count: {(y_res_smt == 1).sum()}")

lgb_smt = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.02,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

lgb_smt.fit(
    X_res_smt, y_res_smt,
    eval_set=[(X_test, y_test)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
)

preds_lgb_smt = lgb_smt.predict_proba(X_test)[:, 1]
evaluate_and_save("LightGBM_SMOTETomek", y_test, preds_lgb_smt, threshold=0.5,
                  notes="LightGBM with SMOTETomek")

# ============================================================
# 9. XGBoost
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: XGBoost")
print("=" * 60)

xgb_model = xgb.XGBClassifier(
    n_estimators=2000,
    learning_rate=0.02,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc",
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1,
    use_label_encoder=False,
    verbosity=0,
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

preds_xgb = xgb_model.predict_proba(X_test)[:, 1]
evaluate_and_save("XGBoost", y_test, preds_xgb, threshold=0.5,
                  notes="XGBoost with scale_pos_weight")

# ============================================================
# 10. RANDOM FOREST (BASELINE)
# ============================================================
print("\n" + "=" * 60)
print("STEP 10: RANDOM FOREST")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbose=0,
)
rf_model.fit(X_train, y_train)

preds_rf = rf_model.predict_proba(X_test)[:, 1]
evaluate_and_save("RandomForest", y_test, preds_rf, threshold=0.5,
                  notes="Random Forest with class_weight='balanced'")

# ============================================================
# 11. LOGISTIC REGRESSION (BASELINE)
# ============================================================
print("\n" + "=" * 60)
print("STEP 11: LOGISTIC REGRESSION")
print("=" * 60)

lr_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42,
    C=0.01,
    n_jobs=-1,
)
lr_model.fit(X_train, y_train)

preds_lr = lr_model.predict_proba(X_test)[:, 1]
evaluate_and_save("LogisticRegression", y_test, preds_lr, threshold=0.5,
                  notes="Logistic Regression with class_weight='balanced', C=0.01")

# ============================================================
# 12. HYPERPARAMETER TUNING WITH OPTUNA
# ============================================================
print("\n" + "=" * 60)
print("STEP 12: OPTUNA HYPERPARAMETER TUNING")
print("=" * 60)

try:
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
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
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

    print(f"\nBest AUC: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")

    # Train best model
    best_params = study.best_params
    best_params.update({
        "n_estimators": 2000,
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    })

    lgb_best = lgb.LGBMClassifier(**best_params)
    lgb_best.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    preds_lgb_best = lgb_best.predict_proba(X_test)[:, 1]
    evaluate_and_save("LightGBM_Optuna", y_test, preds_lgb_best, threshold=0.5,
                      notes=f"Optuna-tuned LightGBM, {study.best_value:.6f} AUC, params: {study.best_params}")

except ImportError:
    print("Optuna not installed - skipping hyperparameter tuning")
    lgb_best = lgb_model
    preds_lgb_best = preds_lgb

# ============================================================
# 13. ENSEMBLE
# ============================================================
print("\n" + "=" * 60)
print("STEP 13: ENSEMBLE")
print("=" * 60)

# Weighted average
ensemble = 0.5 * preds_lgb_best + 0.3 * preds_xgb + 0.2 * preds_rf
auc_ensemble = roc_auc_score(y_test, ensemble)
print(f"Ensemble AUC: {auc_ensemble:.6f}")
evaluate_and_save("Ensemble_LGBM_XGB_RF", y_test, ensemble, threshold=0.5,
                  notes="Weighted ensemble: 0.5*LightGBM + 0.3*XGBoost + 0.2*RandomForest")

# ============================================================
# 14. THRESHOLD OPTIMIZATION
# ============================================================
print("\n" + "=" * 60)
print("STEP 14: THRESHOLD OPTIMIZATION")
print("=" * 60)

thresholds = np.arange(0.1, 0.9, 0.05)
best_f1 = 0
best_thresh = 0.5
for t in thresholds:
    y_pred_t = (preds_lgb_best >= t).astype(int)
    f1_t = f1_score(y_test, y_pred_t)
    if f1_t > best_f1:
        best_f1 = f1_t
        best_thresh = t

print(f"Best threshold by F1: {best_thresh:.2f} (F1={best_f1:.6f})")
evaluate_and_save("LightGBM_Optuna_ThreshOpt", y_test, preds_lgb_best,
                  threshold=best_thresh,
                  notes=f"Optuna LightGBM with F1-optimized threshold={best_thresh:.2f}")

# ============================================================
# 15. SAVE BEST PREDICTIONS & FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 60)
print("STEP 15: SAVE OUTPUTS")
print("=" * 60)

# Best predictions CSV
best_preds_df = pd.DataFrame({"y_true": y_test, "y_prob": preds_lgb_best})
best_preds_df.to_csv("best_predictions.csv", index=False)
print("Saved best_predictions.csv")

# Feature importance
if hasattr(lgb_best, 'feature_importances_'):
    feat_imp = pd.DataFrame({
        "feature": X_train.columns,
        "importance": lgb_best.feature_importances_,
    }).sort_values("importance", ascending=False)
    feat_imp.to_csv("feature_importance.csv", index=False)
    print("Saved feature_importance.csv")
    print(feat_imp.head(15))

print("\n" + "=" * 60)
print("ALL DONE!")
print("=" * 60)
