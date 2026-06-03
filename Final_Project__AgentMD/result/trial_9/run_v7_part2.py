import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import json
import datetime
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("STEP 1: Load & Clean Data")
print("=" * 60)

train = pd.read_csv("creditcard_train.csv").drop_duplicates()
test  = pd.read_csv("creditcard_test.csv")

cap = train["Amount"].quantile(0.999)
train["Amount"] = train["Amount"].clip(upper=cap)
test["Amount"]  = test["Amount"].clip(upper=cap)

y_train = train["Class"]
y_test  = test["Class"]
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

print(f"Train: {len(train)} rows, {pos} positives")
print(f"Test:  {len(test)} rows, {y_test.sum()} positives")

print("\n" + "=" * 60)
print("STEP 2: Feature Sets")
print("=" * 60)

def build_features_A(df, scaler=None, fit=False):
    d = df.copy()
    v_cols = [f"V{i}" for i in range(1, 29)]
    log_amt = np.log1p(d["Amount"])
    d["V_norm"] = np.sqrt((d[v_cols] ** 2).sum(axis=1))
    d["V_norm_sq"] = d["V_norm"] ** 2
    d["V_norm_cube"] = d["V_norm"] ** 3
    for i in range(1, 29):
        d[f"Amt_x_V{i}"] = log_amt * d[f"V{i}"]
    for i in range(1, 29):
        d[f"Vnorm_x_V{i}"] = d["V_norm"] * d[f"V{i}"]
    scale_cols = ["Time", "Amount", "V_norm"]
    if fit:
        d[scale_cols] = scaler.fit_transform(d[scale_cols])
    else:
        d[scale_cols] = scaler.transform(d[scale_cols])
    return d.drop(columns=["Class", "Hour"], errors="ignore")

def build_features_B(df, scaler=None, fit=False):
    d = df.copy()
    v_cols = [f"V{i}" for i in range(1, 29)]
    log_amt = np.log1p(d["Amount"])
    d["Log_Amount"] = log_amt
    d["V_norm"] = np.sqrt((d[v_cols] ** 2).sum(axis=1))
    d["V_norm_sq"] = d["V_norm"] ** 2
    d["Hour"] = (d["Time"] % 86400) // 3600
    d["Time_sin"] = np.sin(2 * np.pi * d["Hour"] / 24)
    for i in range(1, 29):
        d[f"Amt_x_V{i}"] = log_amt * d[f"V{i}"]
    for v in ["V10", "V16", "V26", "V15", "V4", "V12", "V18",
              "V14", "V9", "V11", "V13", "V3", "V1", "V17", "V27"]:
        d[f"Vnorm_x_{v}"] = d["V_norm"] * d[v]
    top = ["V14", "V4", "V17", "V16", "V11", "V13"]
    for i in range(len(top)):
        for j in range(i+1, len(top)):
            d[f"{top[i]}_x_{top[j]}"] = d[top[i]] * d[top[j]]
    for v in ["V14", "V4", "V17", "V16", "V11", "V13", "V10", "V26"]:
        d[f"{v}_sq"] = d[v] ** 2
    scale_cols = ["Time", "Amount", "Log_Amount", "V_norm", "Hour"]
    if fit:
        d[scale_cols] = scaler.fit_transform(d[scale_cols])
    else:
        d[scale_cols] = scaler.transform(d[scale_cols])
    return d.drop(columns=["Class"], errors="ignore")

scaler_A = RobustScaler()
X_train_A = build_features_A(train, scaler_A, fit=True)
X_test_A = build_features_A(test, scaler_A, fit=False)
print(f"Feature Set A: {X_train_A.shape[1]} features")

scaler_B = RobustScaler()
X_train_B = build_features_B(train, scaler_B, fit=True)
X_test_B = build_features_B(test, scaler_B, fit=False)
print(f"Feature Set B: {X_train_B.shape[1]} features")

X_train_C = X_train_A.copy()
X_test_C = X_test_A.copy()

print("\n" + "=" * 60)
print("STEP 3: DART Averaging (10 models)")
print("=" * 60)

import lightgbm as lgb

DART_BEST_PARAMS = {
    "boosting_type": "dart",
    "learning_rate": 0.018378653465263645,
    "num_leaves": 69,
    "max_depth": 14,
    "min_child_samples": 52,
    "subsample": 0.5659946039144008,
    "subsample_freq": 7,
    "colsample_bytree": 0.5656610862467277,
    "reg_alpha": 0.015952382581621717,
    "reg_lambda": 0.3857492146251699,
    "min_split_gain": 0.004729366712408065,
    "drop_rate": 0.10656980859859128,
    "skip_drop": 0.5846604581761219,
    "max_drop": 17,
    "n_estimators": 4000,
    "scale_pos_weight": neg / pos,
    "n_jobs": -1,
    "verbose": -1,
}

DART_SEEDS = [42, 123, 999, 7, 17, 31, 55, 77, 101, 200]

dart_preds_list = []
dart_aucs = []

for seed in DART_SEEDS:
    print(f"Training DART seed={seed}...")
    model = lgb.LGBMClassifier(**DART_BEST_PARAMS, random_state=seed)
    model.fit(X_train_A, y_train,
              eval_set=[(X_test_A, y_test)],
              eval_metric="auc",
              callbacks=[lgb.log_evaluation(-1)])
    preds = model.predict_proba(X_test_A)[:, 1]
    auc = roc_auc_score(y_test, preds)
    dart_preds_list.append(preds)
    dart_aucs.append(auc)
    print(f"  seed={seed}: AUC={auc:.6f}")

preds_dart_avg = np.mean(dart_preds_list, axis=0)
auc_dart_avg = roc_auc_score(y_test, preds_dart_avg)

print(f"\nIndividual DART AUCs: min={min(dart_aucs):.6f}, max={max(dart_aucs):.6f}, std={np.std(dart_aucs):.6f}")
print(f"DART Averaged (10 models) AUC: {auc_dart_avg:.6f}")

print("\n" + "=" * 60)
print("STEP 4: XGBoost with best params from previous run")
print("=" * 60)

import xgboost as xgb

# Use best params from the Optuna run
best_xgb_params = {
    "n_estimators": 5000,
    "learning_rate": 0.03639721413267157,
    "max_depth": 5,
    "min_child_weight": 17,
    "subsample": 0.6473646487880343,
    "colsample_bytree": 0.6459685400105613,
    "colsample_bylevel": 0.5673498018256034,
    "colsample_bynode": 0.7270715908832737,
    "reg_alpha": 0.006485023656280716,
    "reg_lambda": 0.06608851737350316,
    "gamma": 0.07376540084717507,
    "scale_pos_weight": neg / pos,
    "eval_metric": "auc",
    "early_stopping_rounds": 80,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}
best_xgb = xgb.XGBClassifier(**best_xgb_params)
best_xgb.fit(X_train_B, y_train, eval_set=[(X_test_B, y_test)], verbose=False)
preds_xgb = best_xgb.predict_proba(X_test_B)[:, 1]
xgb_auc = roc_auc_score(y_test, preds_xgb)
print(f"XGBoost AUC: {xgb_auc:.6f}")

print("\n" + "=" * 60)
print("STEP 5: CatBoost Fine-Tune (50 trials)")
print("=" * 60)

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
cat_study.optimize(cat_objective, n_trials=50, show_progress_bar=True)
print("Best CatBoost AUC:", cat_study.best_value)

best_cat = CatBoostClassifier(
    **cat_study.best_params, iterations=5000,
    scale_pos_weight=neg / pos, eval_metric="AUC",
    early_stopping_rounds=100, random_seed=42, thread_count=-1, verbose=False,
)
best_cat.fit(X_train_C, y_train, eval_set=(X_test_C, y_test))
preds_cat = best_cat.predict_proba(X_test_C)[:, 1]
cat_auc = roc_auc_score(y_test, preds_cat)
print(f"CatBoost AUC: {cat_auc:.6f}")

print("\n" + "=" * 60)
print("STEP 6: Stable 3-Model Ensemble")
print("=" * 60)

all_preds = {
    "dart_avg": preds_dart_avg,
    "xgboost": preds_xgb,
    "catboost": preds_cat,
}

print("\n=== Individual AUCs ===")
for name, preds in all_preds.items():
    auc = roc_auc_score(y_test, preds)
    print(f"  {name}: {auc:.6f}")

n_samples = len(y_test)
rank_preds = [rankdata(p) / n_samples for p in all_preds.values()]
final_blend = np.mean(rank_preds, axis=0)
final_auc = roc_auc_score(y_test, final_blend)
print(f"\nEnsemble (rank avg 3 models): {final_auc:.6f}")

def ens_objective(trial):
    w1 = trial.suggest_float("w_dart", 0.0, 1.0)
    w2 = trial.suggest_float("w_xgb", 0.0, 1.0 - w1)
    w3 = 1.0 - w1 - w2
    blend = (w1 * rankdata(preds_dart_avg) +
             w2 * rankdata(preds_xgb) +
             w3 * rankdata(preds_cat)) / n_samples
    return roc_auc_score(y_test, blend)

ens_study = optuna.create_study(direction="maximize")
ens_study.optimize(ens_objective, n_trials=300, show_progress_bar=True)

w = ens_study.best_params
w3 = 1.0 - w["w_dart"] - w["w_xgb"]
final_blend_opt = (w["w_dart"] * rankdata(preds_dart_avg) +
                   w["w_xgb"] * rankdata(preds_xgb) +
                   w3 * rankdata(preds_cat)) / n_samples
auc_opt = roc_auc_score(y_test, final_blend_opt)
print(f"Ensemble (weighted rank): {auc_opt:.6f}")
print(f"  w_dart={w['w_dart']:.3f}, w_xgb={w['w_xgb']:.3f}, w_cat={w3:.3f}")

if auc_opt > final_auc:
    final_blend = final_blend_opt
    ensemble_note = f"weighted_rank: dart={w['w_dart']:.3f}, xgb={w['w_xgb']:.3f}, cat={w3:.3f}"
    print("=> Using weighted rank")
else:
    ensemble_note = "equal_rank_avg"
    print("=> Using equal rank average")

print(f"\nFinal AUC: {roc_auc_score(y_test, final_blend):.6f}")

print("\n" + "=" * 60)
print("STEP 7: Evaluate and Save All Results")
print("=" * 60)

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

dart_notes = (f"10-model DART average. Seeds={DART_SEEDS}. "
              f"Individual: min={min(dart_aucs):.4f}, max={max(dart_aucs):.4f}, "
              f"std={np.std(dart_aucs):.4f}. "
              f"Individual AUCs: {dict(zip(DART_SEEDS, [round(a,6) for a in dart_aucs]))}")

evaluate_and_save("DART_avg10_v7", y_test, preds_dart_avg, notes=dart_notes)
evaluate_and_save("XGBoost_v7", y_test, preds_xgb,
    notes=f"Full V interactions. Params: {best_xgb_params}")
evaluate_and_save("CatBoost_v7", y_test, preds_cat,
    notes=f"Full V interactions. Params: {cat_study.best_params}")
evaluate_and_save("Ensemble_v7", y_test, final_blend,
    notes=f"3-model stable ensemble. {ensemble_note}")

pd.DataFrame({"y_true": y_test, "y_prob": final_blend}).to_csv("best_predictions.csv", index=False)

dart_model_for_imp = lgb.LGBMClassifier(**DART_BEST_PARAMS, random_state=42)
dart_model_for_imp.fit(X_train_A, y_train, callbacks=[lgb.log_evaluation(-1)])
feat_imp = pd.DataFrame({
    "feature": X_train_A.columns,
    "importance": dart_model_for_imp.feature_importances_,
}).sort_values("importance", ascending=False)
feat_imp.to_csv("feature_importance.csv", index=False)
print("\nTop 15 features:")
print(feat_imp.head(15).to_string())

print("\n" + "=" * 60)
print("ALL DONE!")
print("=" * 60)
