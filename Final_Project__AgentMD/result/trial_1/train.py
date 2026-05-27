import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

train = pd.read_csv("creditcard_train.csv")
test  = pd.read_csv("creditcard_test.csv")

def engineer_features(df):
    df = df.copy()
    df["log_amount"] = np.log1p(df["Amount"])
    df["time_sin"] = np.sin(2 * np.pi * df["Time"] / 86400)
    df["time_cos"] = np.cos(2 * np.pi * df["Time"] / 86400)
    df["hour"] = (df["Time"] // 3600) % 24
    for col in ["V1","V2","V3","V4","V10","V11","V12","V14","V17"]:
        df[f"{col}_x_logamt"] = df[col] * df["log_amount"]
    df = df.drop(columns=["Time", "Amount"])
    return df

X_train_eng = engineer_features(train.drop(columns=["Class"]))
X_test_eng  = engineer_features(test.drop(columns=["Class"]))

raw_cols = [c for c in train.columns if c not in ("Class", "Time", "Amount")]
X_train_raw = train[raw_cols].copy()
X_test_raw  = test[raw_cols].copy()

y_train = train["Class"]
y_test  = test["Class"]

print(f"Train: {len(X_train_eng)} rows, Test: {len(X_test_eng)} rows")

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scale = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight = {scale:.0f}")

# --- LightGBM (engineered) ---
lgb_params = {
    "objective": "binary", "metric": "auc", "n_estimators": 3000,
    "learning_rate": 0.02, "num_leaves": 63, "max_depth": -1,
    "min_child_samples": 20, "feature_fraction": 0.8,
    "bagging_fraction": 0.8, "bagging_freq": 5,
    "scale_pos_weight": scale, "lambda_l1": 0.1, "lambda_l2": 10.0,
    "random_state": 42, "n_jobs": -1, "verbose": -1,
}
lgb_oof  = np.zeros(len(X_train_eng))
lgb_test = np.zeros(len(X_test_eng))
for fold, (tr_idx, val_idx) in enumerate(CV.split(X_train_eng, y_train)):
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_train_eng.iloc[tr_idx], y_train.iloc[tr_idx],
              eval_set=[(X_train_eng.iloc[val_idx], y_train.iloc[val_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    lgb_oof[val_idx]  = model.predict_proba(X_train_eng.iloc[val_idx])[:, 1]
    lgb_test         += model.predict_proba(X_test_eng)[:, 1] / 5
print(f"LGB OOF AUC: {roc_auc_score(y_train, lgb_oof):.6f}")

# --- XGBoost (engineered) ---
xgb_params = {
    "objective": "binary:logistic", "eval_metric": "auc", "n_estimators": 3000,
    "learning_rate": 0.02, "max_depth": 6, "min_child_weight": 5,
    "subsample": 0.8, "colsample_bytree": 0.8, "scale_pos_weight": scale,
    "reg_alpha": 0.1, "reg_lambda": 10.0, "random_state": 42, "n_jobs": -1,
    "tree_method": "hist", "early_stopping_rounds": 100,
}
xgb_oof  = np.zeros(len(X_train_eng))
xgb_test = np.zeros(len(X_test_eng))
for fold, (tr_idx, val_idx) in enumerate(CV.split(X_train_eng, y_train)):
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train_eng.iloc[tr_idx], y_train.iloc[tr_idx],
              eval_set=[(X_train_eng.iloc[val_idx], y_train.iloc[val_idx])], verbose=False)
    xgb_oof[val_idx]  = model.predict_proba(X_train_eng.iloc[val_idx])[:, 1]
    xgb_test         += model.predict_proba(X_test_eng)[:, 1] / 5
print(f"XGB OOF AUC: {roc_auc_score(y_train, xgb_oof):.6f}")

# --- CatBoost (raw features — better without engineering) ---
cat_oof  = np.zeros(len(X_train_raw))
cat_test = np.zeros(len(X_test_raw))
for fold, (tr_idx, val_idx) in enumerate(CV.split(X_train_raw, y_train)):
    model = CatBoostClassifier(
        iterations=3000, learning_rate=0.02, depth=6,
        scale_pos_weight=scale, eval_metric="AUC",
        early_stopping_rounds=100, random_seed=42, verbose=0,
    )
    model.fit(X_train_raw.iloc[tr_idx], y_train.iloc[tr_idx],
              eval_set=(X_train_raw.iloc[val_idx], y_train.iloc[val_idx]))
    cat_oof[val_idx]  = model.predict_proba(X_train_raw.iloc[val_idx])[:, 1]
    cat_test         += model.predict_proba(X_test_raw)[:, 1] / 5
print(f"CAT OOF AUC: {roc_auc_score(y_train, cat_oof):.6f}")

# --- Equal-weight blend ---
final_proba = (lgb_test + xgb_test + cat_test) / 3
final_oof   = (lgb_oof  + xgb_oof  + cat_oof)  / 3
print(f"Equal blend OOF AUC: {roc_auc_score(y_train, final_oof):.6f}")

# --- Final Evaluation ---
test_auc   = roc_auc_score(y_test, final_proba)
test_prauc = average_precision_score(y_test, final_proba)

print("=" * 50)
print(f"  TEST ROC-AUC  : {test_auc:.6f}   <- primary metric")
print(f"  TEST PR-AUC   : {test_prauc:.6f}  <- secondary metric")
print("=" * 50)

thresholds = np.linspace(0.01, 0.99, 200)
f1s = [f1_score(y_train, final_oof >= t) for t in thresholds]
best_thresh = thresholds[np.argmax(f1s)]
print(f"\nBest threshold (OOF F1): {best_thresh:.3f}")
y_pred = (final_proba >= best_thresh).astype(int)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))

pd.DataFrame({"Class_proba": final_proba, "Class_pred": y_pred}).to_csv(
    "creditcard_test_predictions.csv", index=False
)
print("Predictions saved to creditcard_test_predictions.csv")
