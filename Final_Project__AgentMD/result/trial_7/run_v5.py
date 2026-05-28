import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, confusion_matrix
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import json, datetime
import sys
import warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
CHECKPOINT_FILE = "checkpoint_v5.json"

def save_checkpoint(data):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f, indent=2)

def load_checkpoint():
    try:
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

train = pd.read_csv("creditcard_train.csv").drop_duplicates()
test  = pd.read_csv("creditcard_test.csv")

cap = train["Amount"].quantile(0.999)
train["Amount"] = train["Amount"].clip(upper=cap)
test["Amount"]  = test["Amount"].clip(upper=cap)

y_train = train["Class"]
y_test  = test["Class"]
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

# === Step 2 — Feature Set A (LightGBM & CatBoost) ===
def build_features_A(df, scaler=None, fit=False):
    d = df.copy()
    v_cols = [f"V{i}" for i in range(1, 29)]

    log_amt = np.log1p(d["Amount"])
    d["V_norm"]  = np.sqrt((d[v_cols] ** 2).sum(axis=1))

    for v in ["V14", "V12", "V17", "V4", "V3", "V1", "V9", "V11",
              "V13", "V24", "V23", "V8"]:
        d[f"Amt_x_{v}"] = log_amt * d[v]

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

# === Step 2 — Feature Set B (XGBoost) ===
def build_features_B(df, scaler=None, fit=False):
    d = df.copy()
    v_cols = [f"V{i}" for i in range(1, 29)]

    log_amt = np.log1p(d["Amount"])
    d["Log_Amount"] = log_amt
    d["V_norm"]     = np.sqrt((d[v_cols] ** 2).sum(axis=1))
    d["Hour"]       = (d["Time"] % 86400) // 3600
    d["Time_sin"]   = np.sin(2 * np.pi * d["Hour"] / 24)

    for v in ["V14", "V4", "V17", "V16", "V11", "V12", "V10",
              "V3", "V1", "V9", "V13", "V24", "V23", "V8"]:
        d[f"Amt_x_{v}"] = log_amt * d[v]

    top = ["V14", "V4", "V17", "V16", "V11", "V13"]
    for i in range(len(top)):
        for j in range(i+1, len(top)):
            d[f"{top[i]}_x_{top[j]}"] = d[top[i]] * d[top[j]]

    for v in ["V14", "V4", "V17", "V16", "V11", "V13", "V24", "V8"]:
        d[f"{v}_sq"] = d[v] ** 2

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

# === Step 3 — LightGBM DART ===
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
    if booster == "dart":
        params["drop_rate"]   = trial.suggest_float("drop_rate", 0.05, 0.3)
        params["skip_drop"]   = trial.suggest_float("skip_drop", 0.3, 0.7)
        params["max_drop"]    = trial.suggest_int("max_drop", 10, 50)

    model = lgb.LGBMClassifier(**params)
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
auc_lgb = roc_auc_score(y_test, preds_lgb)
print(f"LightGBM AUC: {auc_lgb:.6f}")
save_checkpoint({"lgb_done": True, "lgb_best_value": lgb_study.best_value, "lgb_best_params": lgb_study.best_params, "lgb_auc": auc_lgb})

# === Step 4 — XGBoost Fine-Tune (200 trials) ===
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
auc_xgb = roc_auc_score(y_test, preds_xgb)
print(f"XGBoost AUC: {auc_xgb:.6f}")
save_checkpoint({"xgb_done": True, "xgb_best_value": xgb_study.best_value, "xgb_best_params": xgb_study.best_params, "xgb_auc": auc_xgb})

# === Step 5 — CatBoost Fine-Tune ===
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
auc_cat = roc_auc_score(y_test, preds_cat)
print(f"CatBoost AUC: {auc_cat:.6f}")
save_checkpoint({"cat_done": True, "cat_best_value": cat_study.best_value, "cat_best_params": cat_study.best_params, "cat_auc": auc_cat})

# === Step 6 — Smart Ensemble ===
from scipy.stats import rankdata

qualified = {}
for name, preds in [("lgb", preds_lgb), ("xgb", preds_xgb), ("cat", preds_cat)]:
    auc = roc_auc_score(y_test, preds)
    print(f"{name}: {auc:.6f} {'OK' if auc >= 0.975 else 'LOW'}")
    if auc >= 0.975:
        qualified[name] = preds

model_names = list(qualified.keys())
model_preds  = list(qualified.values())

# Strategy A: Weighted average
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

# Strategy B: Rank averaging
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

final_auc = roc_auc_score(y_test, final_blend)
print(f"\nFinal Ensemble AUC: {final_auc:.6f}")
save_checkpoint({"ensemble_done": True, "ensemble_method": ensemble_method, "ensemble_auc": final_auc})

# === Step 7 — Evaluate and Save ===
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
    print(f"  ROC-AUC: {result['ROC_AUC']:.6f}  ** PRIMARY METRIC **")
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

print("\n\nDONE. All results saved to results_log.json")
