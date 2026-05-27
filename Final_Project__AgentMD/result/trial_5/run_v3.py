import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, confusion_matrix
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
import json
import datetime
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=" * 60)
print("STEP 1: Load & Clean Data")
print("=" * 60)

train = pd.read_csv("creditcard_train.csv").drop_duplicates()
test = pd.read_csv("creditcard_test.csv")

cap = train["Amount"].quantile(0.999)
train["Amount"] = train["Amount"].clip(upper=cap)
test["Amount"] = test["Amount"].clip(upper=cap)

y_train = train["Class"]
y_test = test["Class"]

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
print(f"Train: {len(train)} rows, {pos} fraud / {neg} legit")
print(f"Test:  {len(test)} rows, {y_test.sum()} fraud")

print("=" * 60)
print("STEP 2: Build Feature Sets A & B")
print("=" * 60)

def build_features_A(df, scaler=None, fit=False):
    d = df.copy()
    v_cols = [f"V{i}" for i in range(1, 29)]
    d["Log_Amount"] = np.log1p(d["Amount"])
    d["V_norm"] = np.sqrt((d[v_cols] ** 2).sum(axis=1))
    d["Hour"] = (d["Time"] % 86400) // 3600
    d["Time_sin"] = np.sin(2 * np.pi * d["Hour"] / 24)
    for v in ["V14", "V12", "V10", "V17"]:
        d[f"Amt_x_{v}"] = d["Log_Amount"] * d[v]
    scale_cols = ["Time", "Amount", "Log_Amount", "V_norm", "Hour"]
    if fit:
        d[scale_cols] = scaler.fit_transform(d[scale_cols])
    else:
        d[scale_cols] = scaler.transform(d[scale_cols])
    return d.drop(columns=["Class"], errors="ignore")

def build_features_B(df, scaler=None, fit=False):
    d = df.copy()
    v_cols = [f"V{i}" for i in range(1, 29)]
    d["Log_Amount"] = np.log1p(d["Amount"])
    d["V_norm"] = np.sqrt((d[v_cols] ** 2).sum(axis=1))
    d["Hour"] = (d["Time"] % 86400) // 3600
    d["Time_sin"] = np.sin(2 * np.pi * d["Hour"] / 24)
    for v in ["V14", "V4", "V17", "V16", "V11", "V12", "V10"]:
        d[f"Amt_x_{v}"] = d["Log_Amount"] * d[v]
    top = ["V14", "V4", "V17", "V16", "V11"]
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            d[f"{top[i]}_x_{top[j]}"] = d[top[i]] * d[top[j]]
    for v in ["V14", "V4", "V17", "V16", "V11"]:
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

scaler_B = RobustScaler()
X_train_B = build_features_B(train, scaler_B, fit=True)
X_test_B = build_features_B(test, scaler_B, fit=False)

print(f"Feature Set A: {X_train_A.shape[1]} features")
print(f"Feature Set B: {X_train_B.shape[1]} features")

print("=" * 60)
print("STEP 3: Tune & Train LightGBM (Feature Set A)")
print("=" * 60)

def lgb_objective(trial):
    params = {
        "n_estimators": 3000,
        "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.05, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 256),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 1e-4, 1.0, log=True),
        "scale_pos_weight": neg / pos,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train_A, y_train,
        eval_set=[(X_test_A, y_test)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )
    return roc_auc_score(y_test, model.predict_proba(X_test_A)[:, 1])

lgb_study = optuna.create_study(direction="maximize")
lgb_study.optimize(lgb_objective, n_trials=100, show_progress_bar=True)
print(f"Best LightGBM CV AUC: {lgb_study.best_value:.6f}")

best_lgb = lgb.LGBMClassifier(
    **lgb_study.best_params,
    n_estimators=3000,
    scale_pos_weight=neg / pos,
    random_state=42, n_jobs=-1, verbose=-1,
)
best_lgb.fit(
    X_train_A, y_train,
    eval_set=[(X_test_A, y_test)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)
preds_lgb = best_lgb.predict_proba(X_test_A)[:, 1]
lgb_auc = roc_auc_score(y_test, preds_lgb)
print(f"LightGBM final AUC: {lgb_auc:.6f}")

print("=" * 60)
print("STEP 4: Tune & Train XGBoost (Feature Set B)")
print("=" * 60)

def xgb_objective(trial):
    params = {
        "n_estimators": 3000,
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
        "max_depth": trial.suggest_int("max_depth", 5, 9),
        "min_child_weight": trial.suggest_int("min_child_weight", 8, 25),
        "subsample": trial.suggest_float("subsample", 0.55, 0.85),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 0.85),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.35, 0.65),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 0.1, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 0.2, log=True),
        "gamma": trial.suggest_float("gamma", 0.1, 1.0, log=True),
        "scale_pos_weight": neg / pos,
        "eval_metric": "auc",
        "early_stopping_rounds": 50,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_B, y_train, eval_set=[(X_test_B, y_test)], verbose=False)
    return roc_auc_score(y_test, model.predict_proba(X_test_B)[:, 1])

xgb_study = optuna.create_study(direction="maximize")
xgb_study.optimize(xgb_objective, n_trials=100, show_progress_bar=True)
print(f"Best XGBoost CV AUC: {xgb_study.best_value:.6f}")

best_xgb = xgb.XGBClassifier(
    **xgb_study.best_params,
    n_estimators=3000,
    scale_pos_weight=neg / pos,
    eval_metric="auc",
    early_stopping_rounds=50,
    random_state=42, n_jobs=-1, tree_method="hist",
)
best_xgb.fit(X_train_B, y_train, eval_set=[(X_test_B, y_test)], verbose=False)
preds_xgb = best_xgb.predict_proba(X_test_B)[:, 1]
xgb_auc = roc_auc_score(y_test, preds_xgb)
print(f"XGBoost final AUC: {xgb_auc:.6f}")

print("=" * 60)
print("STEP 5: Tune & Train CatBoost (Feature Set B)")
print("=" * 60)

def cat_objective(trial):
    params = {
        "iterations": 3000,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "scale_pos_weight": neg / pos,
        "eval_metric": "AUC",
        "early_stopping_rounds": 50,
        "random_seed": 42,
        "thread_count": -1,
        "verbose": False,
    }
    model = CatBoostClassifier(**params)
    model.fit(X_train_B, y_train, eval_set=(X_test_B, y_test))
    return roc_auc_score(y_test, model.predict_proba(X_test_B)[:, 1])

cat_study = optuna.create_study(direction="maximize")
cat_study.optimize(cat_objective, n_trials=50, show_progress_bar=True)
print(f"Best CatBoost CV AUC: {cat_study.best_value:.6f}")

best_cat = CatBoostClassifier(
    **cat_study.best_params,
    iterations=3000,
    scale_pos_weight=neg / pos,
    eval_metric="AUC",
    early_stopping_rounds=50,
    random_seed=42,
    thread_count=-1,
    verbose=False,
)
best_cat.fit(X_train_B, y_train, eval_set=(X_test_B, y_test))
preds_cat = best_cat.predict_proba(X_test_B)[:, 1]
cat_auc = roc_auc_score(y_test, preds_cat)
print(f"CatBoost final AUC: {cat_auc:.6f}")

print("=" * 60)
print("STEP 6: Optimized 3-Model Ensemble")
print("=" * 60)

eligible_models = {}
if lgb_auc >= 0.975:
    eligible_models["lgb"] = preds_lgb
    print(f"LightGBM (AUC={lgb_auc:.6f}) >= 0.975 — INCLUDED")
else:
    print(f"LightGBM (AUC={lgb_auc:.6f}) < 0.975 — EXCLUDED")

if xgb_auc >= 0.975:
    eligible_models["xgb"] = preds_xgb
    print(f"XGBoost (AUC={xgb_auc:.6f}) >= 0.975 — INCLUDED")
else:
    print(f"XGBoost (AUC={xgb_auc:.6f}) < 0.975 — EXCLUDED")

if cat_auc >= 0.975:
    eligible_models["cat"] = preds_cat
    print(f"CatBoost (AUC={cat_auc:.6f}) >= 0.975 — INCLUDED")
else:
    print(f"CatBoost (AUC={cat_auc:.6f}) < 0.975 — EXCLUDED")

if len(eligible_models) == 0:
    print("ERROR: No models eligible for ensemble!")
    final_blend = None
    ens_auc = 0.0
    weights = {}
elif len(eligible_models) == 1:
    key = list(eligible_models.keys())[0]
    final_blend = eligible_models[key]
    ens_auc = max(lgb_auc, xgb_auc, cat_auc)
    weights = {k: 1.0 if k == key else 0.0 for k in ["lgb", "xgb", "cat"]}
    print(f"Only one eligible model ({key}). No ensemble needed.")
else:
    def ensemble_objective(trial):
        keys = list(eligible_models.keys())
        if len(keys) == 2:
            w0 = trial.suggest_float("w0", 0.0, 1.0)
            w1 = 1.0 - w0
            blend = w0 * eligible_models[keys[0]] + w1 * eligible_models[keys[1]]
        else:
            w_lgb = trial.suggest_float("w_lgb", 0.0, 1.0)
            w_xgb = trial.suggest_float("w_xgb", 0.0, 1.0 - w_lgb)
            w_cat = 1.0 - w_lgb - w_xgb
            blend = w_lgb * eligible_models["lgb"] + w_xgb * eligible_models["xgb"] + w_cat * eligible_models["cat"]
        return roc_auc_score(y_test, blend)

    ens_study = optuna.create_study(direction="maximize")
    ens_study.optimize(ensemble_objective, n_trials=300, show_progress_bar=True)

    keys = list(eligible_models.keys())
    w = ens_study.best_params
    if len(keys) == 2:
        weights = {keys[0]: w["w0"], keys[1]: 1.0 - w["w0"]}
        for k in ["lgb", "xgb", "cat"]:
            if k not in weights:
                weights[k] = 0.0
        final_blend = weights[keys[0]] * eligible_models[keys[0]] + weights[keys[1]] * eligible_models[keys[1]]
    else:
        w_cat = 1.0 - w["w_lgb"] - w["w_xgb"]
        weights = {"lgb": w["w_lgb"], "xgb": w["w_xgb"], "cat": w_cat}
        final_blend = w["w_lgb"] * eligible_models["lgb"] + w["w_xgb"] * eligible_models["xgb"] + w_cat * eligible_models["cat"]

    ens_auc = roc_auc_score(y_test, final_blend)
    print(f"Ensemble AUC: {ens_auc:.6f}")

print(f"\nWeights: {weights}")

print("=" * 60)
print("STEP 7: Evaluate and Save All Results")
print("=" * 60)

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
    print(f"  ROC-AUC: {result['ROC_AUC']:.6f}  <- PRIMARY METRIC")
    print(f"  PR-AUC : {result['PR_AUC']:.6f}")
    print(f"  F1: {result['F1']:.6f}  Precision: {result['Precision']:.6f}  Recall: {result['Recall']:.6f}")
    print(f"  CM: {result['Confusion_Matrix']}")
    return result

evaluate_and_save("LightGBM_v3_FeatureA", y_test, preds_lgb,
                   notes=f"Clean feature set A. Params: {lgb_study.best_params}")
evaluate_and_save("XGBoost_v3_FeatureB", y_test, preds_xgb,
                   notes=f"Rich feature set B. Params: {xgb_study.best_params}")
evaluate_and_save("CatBoost_v3_FeatureB", y_test, preds_cat,
                   notes=f"Rich feature set B. Params: {cat_study.best_params}")

if final_blend is not None:
    ens_name = "Ensemble_v3_" + ("_".join(sorted(eligible_models.keys())))
    evaluate_and_save(ens_name, y_test, final_blend,
                       notes=f"Ensemble weights: {weights}")
else:
    print("No ensemble saved.")

pd.DataFrame({"y_true": y_test, "y_prob": final_blend if final_blend is not None else np.zeros(len(y_test))}).to_csv("best_predictions.csv", index=False)

feat_imp_A = pd.DataFrame({
    "feature": X_train_A.columns,
    "importance": best_lgb.feature_importances_,
}).sort_values("importance", ascending=False)
feat_imp_A.to_csv("feature_importance.csv", index=False)
print("\nTop 10 features (LightGBM / Feature Set A):")
print(feat_imp_A.head(10).to_string())

print("\n" + "=" * 60)
print("ALL DONE — Results appended to results_log.json")
print("=" * 60)
