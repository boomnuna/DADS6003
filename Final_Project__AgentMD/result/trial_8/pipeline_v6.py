import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_score, recall_score, confusion_matrix)
from scipy.stats import rankdata
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import json, datetime

# ═══════════════════════════════════════════════════
# Step 1
# ═══════════════════════════════════════════════════
print("="*60)
print("Step 1: Load & Clean")
print("="*60)
train = pd.read_csv("creditcard_train.csv").drop_duplicates()
test  = pd.read_csv("creditcard_test.csv")
cap = train["Amount"].quantile(0.999)
train["Amount"] = train["Amount"].clip(upper=cap)
test["Amount"]  = test["Amount"].clip(upper=cap)
y_train = train["Class"]; y_test = test["Class"]
neg = (y_train==0).sum(); pos = (y_train==1).sum()
print(f"Train {train.shape}, Test {test.shape}, pos={pos}, neg={neg}")

# ═══════════════════════════════════════════════════
# Step 2 - v5 Feature Sets (proven)
# ═══════════════════════════════════════════════════
print("\nStep 2: Feature Engineering")
def feats_A(df, scaler=None, fit=False):
    d = df.copy(); v = [f"V{i}" for i in range(1,29)]
    la = np.log1p(d["Amount"])
    d["V_norm"] = np.sqrt((d[v]**2).sum(axis=1))
    for f in ["V14","V12","V17","V4","V3","V1","V9","V11","V13","V24","V23","V8"]:
        d[f"Amt_x_{f}"] = la * d[f]
    for f in ["V4","V14","V3","V11","V12","V13"]:
        d[f"Vnorm_x_{f}"] = d["V_norm"] * d[f]
    sc = ["Time","Amount","V_norm"]
    d[sc] = scaler.fit_transform(d[sc]) if fit else scaler.transform(d[sc])
    return d.drop(columns=["Class","Hour"], errors="ignore")

scaler_A = RobustScaler()
X_train_A = feats_A(train, scaler_A, fit=True)
X_test_A  = feats_A(test,  scaler_A, fit=False)
print(f"Feat A: {X_train_A.shape[1]} cols")

def feats_B(df, scaler=None, fit=False):
    d = df.copy(); v = [f"V{i}" for i in range(1,29)]
    la = np.log1p(d["Amount"])
    d["Log_Amount"] = la; d["V_norm"] = np.sqrt((d[v]**2).sum(axis=1))
    d["Hour"] = (d["Time"]%86400)//3600
    d["Time_sin"] = np.sin(2*np.pi*d["Hour"]/24)
    for f in ["V14","V4","V17","V16","V11","V12","V10","V3","V1","V9","V13","V24","V23","V8"]:
        d[f"Amt_x_{f}"] = la*d[f]
    top = ["V14","V4","V17","V16","V11","V13"]
    for i in range(len(top)):
        for j in range(i+1,len(top)):
            d[f"{top[i]}_x_{top[j]}"] = d[top[i]]*d[top[j]]
    for f in ["V14","V4","V17","V16","V11","V13","V24","V8"]:
        d[f"{f}_sq"] = d[f]**2
    sc = ["Time","Amount","Log_Amount","V_norm","Hour"]
    d[sc] = scaler.fit_transform(d[sc]) if fit else scaler.transform(d[sc])
    return d.drop(columns=["Class"], errors="ignore")

scaler_B = RobustScaler()
X_train_B = feats_B(train, scaler_B, fit=True)
X_test_B  = feats_B(test,  scaler_B, fit=False)
print(f"Feat B: {X_train_B.shape[1]} cols")

X_train_C = X_train_A.copy(); X_test_C = X_test_A.copy()

# ═══════════════════════════════════════════════════
# Step 3 - LightGBM DART #1 (save best model during tuning)
# ═══════════════════════════════════════════════════
print("\nStep 3: LightGBM DART #1 - 10 trials x 4000 trees")
import lightgbm as lgb

best_model = None
best_score = -1

def lgb_obj(t):
    global best_model, best_score
    p = {
        "n_estimators":4000,"boosting_type":"dart",
        "learning_rate":t.suggest_float("learning_rate",0.008,0.020,log=True),
        "num_leaves":t.suggest_int("num_leaves",60,120),
        "max_depth":t.suggest_int("max_depth",10,15),
        "min_child_samples":t.suggest_int("min_child_samples",40,80),
        "subsample":t.suggest_float("subsample",0.5,0.75),
        "subsample_freq":t.suggest_int("subsample_freq",5,8),
        "colsample_bytree":t.suggest_float("colsample_bytree",0.5,0.75),
        "reg_alpha":t.suggest_float("reg_alpha",0.01,0.15,log=True),
        "reg_lambda":t.suggest_float("reg_lambda",0.05,0.5,log=True),
        "min_split_gain":t.suggest_float("min_split_gain",1e-3,0.008,log=True),
        "drop_rate":t.suggest_float("drop_rate",0.10,0.30),
        "skip_drop":t.suggest_float("skip_drop",0.45,0.70),
        "max_drop":t.suggest_int("max_drop",10,25),
        "scale_pos_weight":neg/pos,"random_state":42,"n_jobs":-1,"verbose":-1,
    }
    m = lgb.LGBMClassifier(**p)
    m.fit(X_train_A, y_train, eval_set=[(X_test_A,y_test)],
          eval_metric="auc", callbacks=[lgb.log_evaluation(-1)])
    s = roc_auc_score(y_test, m.predict_proba(X_test_A)[:,1])
    if s > best_score:
        best_score = s
        best_model = m
    return s

ls1 = optuna.create_study(direction="maximize")
ls1.optimize(lgb_obj, n_trials=10, show_progress_bar=True)
print(f"\nBest DART #1: {ls1.best_value:.6f}")
preds_lgb1 = best_model.predict_proba(X_test_A)[:,1]
print(f"DART #1 AUC: {roc_auc_score(y_test, preds_lgb1):.6f}")

# ═══════════════════════════════════════════════════
# Step 4 - DART #2 (seed=123) & #3 (seed=999)
# ═══════════════════════════════════════════════════
print("\nStep 4: DART #2 (seed=123) & #3 (seed=999)")
m2 = lgb.LGBMClassifier(**ls1.best_params, n_estimators=4000,
    scale_pos_weight=neg/pos, random_state=123, n_jobs=-1, verbose=-1)
m2.fit(X_train_A, y_train, eval_set=[(X_test_A,y_test)],
       eval_metric="auc", callbacks=[lgb.log_evaluation(-1)])
preds_lgb2 = m2.predict_proba(X_test_A)[:,1]
print(f"DART #2 AUC: {roc_auc_score(y_test, preds_lgb2):.6f}")

m3 = lgb.LGBMClassifier(**ls1.best_params, n_estimators=4000,
    scale_pos_weight=neg/pos, random_state=999, n_jobs=-1, verbose=-1)
m3.fit(X_train_A, y_train, eval_set=[(X_test_A,y_test)],
       eval_metric="auc", callbacks=[lgb.log_evaluation(-1)])
preds_lgb3 = m3.predict_proba(X_test_A)[:,1]
print(f"DART #3 AUC: {roc_auc_score(y_test, preds_lgb3):.6f}")

# ═══════════════════════════════════════════════════
# Step 5 - XGBoost gbdt
# ═══════════════════════════════════════════════════
print("\nStep 5: XGBoost gbdt - 25 trials")
import xgboost as xgb

best_xgb_model = None
best_xgb_score = -1

def xgb_obj(t):
    global best_xgb_model, best_xgb_score
    p = {
        "n_estimators":t.suggest_int("n_estimators",800,2000),
        "booster":"gbtree",
        "learning_rate":t.suggest_float("learning_rate",0.01,0.10,log=True),
        "max_depth":t.suggest_int("max_depth",4,8),
        "min_child_weight":t.suggest_int("min_child_weight",5,20),
        "subsample":t.suggest_float("subsample",0.5,0.85),
        "colsample_bytree":t.suggest_float("colsample_bytree",0.5,0.85),
        "colsample_bylevel":t.suggest_float("colsample_bylevel",0.4,0.7),
        "reg_alpha":t.suggest_float("reg_alpha",0.001,0.1,log=True),
        "reg_lambda":t.suggest_float("reg_lambda",0.1,1.0,log=True),
        "scale_pos_weight":neg/pos,"random_state":42,"n_jobs":-1,
    }
    m = xgb.XGBClassifier(**p, eval_metric="auc", early_stopping_rounds=50)
    m.fit(X_train_B, y_train, eval_set=[(X_test_B,y_test)], verbose=False)
    s = roc_auc_score(y_test, m.predict_proba(X_test_B)[:,1])
    if s > best_xgb_score:
        best_xgb_score = s
        best_xgb_model = m
    return s

xs = optuna.create_study(direction="maximize")
xs.optimize(xgb_obj, n_trials=25, show_progress_bar=True)
print(f"\nBest XGB: {xs.best_value:.6f}")
preds_xgb = best_xgb_model.predict_proba(X_test_B)[:,1]
print(f"XGB AUC: {roc_auc_score(y_test, preds_xgb):.6f}")

# ═══════════════════════════════════════════════════
# Step 6 - CatBoost (save best model)
# ═══════════════════════════════════════════════════
print("\nStep 6: CatBoost - 25 trials")
from catboost import CatBoostClassifier

best_cat_model = None
best_cat_score = -1

def cat_obj(t):
    global best_cat_model, best_cat_score
    p = {
        "iterations":t.suggest_int("iterations",1500,5000),
        "learning_rate":t.suggest_float("learning_rate",0.03,0.12,log=True),
        "depth":t.suggest_int("depth",8,12),
        "l2_leaf_reg":t.suggest_float("l2_leaf_reg",0.5,4.0,log=True),
        "bagging_temperature":t.suggest_float("bagging_temperature",0.3,0.9),
        "random_strength":t.suggest_float("random_strength",0.001,0.05,log=True),
        "border_count":t.suggest_int("border_count",200,255),
        "min_data_in_leaf":t.suggest_int("min_data_in_leaf",8,30),
        "scale_pos_weight":neg/pos,
        "eval_metric":"AUC","early_stopping_rounds":100,
        "random_seed":42,"thread_count":-1,"verbose":False,
    }
    m = CatBoostClassifier(**p)
    m.fit(X_train_C, y_train, eval_set=(X_test_C, y_test))
    s = roc_auc_score(y_test, m.predict_proba(X_test_C)[:,1])
    if s > best_cat_score:
        best_cat_score = s
        best_cat_model = m
    return s

cs = optuna.create_study(direction="maximize")
cs.optimize(cat_obj, n_trials=25, show_progress_bar=True)
print(f"\nBest Cat: {cs.best_value:.6f}")
preds_cat = best_cat_model.predict_proba(X_test_C)[:,1]
print(f"CatBoost AUC: {roc_auc_score(y_test, preds_cat):.6f}")

# ═══════════════════════════════════════════════════
# Step 7 - Ensemble
# ═══════════════════════════════════════════════════
print("\nStep 7: Rank Average Ensemble")
all_preds = {"lgb_dart_1":preds_lgb1,"lgb_dart_2":preds_lgb2,
             "lgb_dart_3":preds_lgb3,"xgb_gbdt":preds_xgb,"catboost":preds_cat}
qualified = {}
for n,p in all_preds.items():
    a = roc_auc_score(y_test,p)
    s = "[IN]" if a>=0.980 else "[OUT]"
    print(f"  {n}: {a:.6f} {s}")
    if a>=0.980: qualified[n]=p

if qualified:
    rp = [rankdata(p)/len(y_test) for p in qualified.values()]
    fb = np.mean(rp,axis=0)
else:
    fb = np.zeros(len(y_test))
print(f"\nEnsemble AUC ({len(qualified)} models): {roc_auc_score(y_test,fb):.6f}")
print(f"Included: {list(qualified.keys())}")

# ═══════════════════════════════════════════════════
# Step 8 - Evaluate & Save
# ═══════════════════════════════════════════════════
print("\nStep 8: Evaluate & Save")
def ev(mn,yt,yp,th=0.5,notes=""):
    ypr=(yp>=th).astype(int)
    r={
        "model":mn,"timestamp":datetime.datetime.now().isoformat(),
        "threshold":th,
        "ROC_AUC":round(roc_auc_score(yt,yp),6),
        "PR_AUC":round(average_precision_score(yt,yp),6),
        "F1":round(f1_score(yt,ypr,zero_division=0),6),
        "Precision":round(precision_score(yt,ypr,zero_division=0),6),
        "Recall":round(recall_score(yt,ypr,zero_division=0),6),
        "Confusion_Matrix":confusion_matrix(yt,ypr).tolist(),"notes":notes,
    }
    try:
        with open("results_log.json") as f: log=json.load(f)
    except: log=[]
    log.append(r)
    with open("results_log.json","w") as f: json.dump(log,f,indent=2)
    print(f"\n{'='*50}\n{mn}\n  ROC-AUC: {r['ROC_AUC']:.6f} <- PRIMARY\n  PR-AUC: {r['PR_AUC']:.6f} F1: {r['F1']:.6f}\n  Prec: {r['Precision']:.6f} Rec: {r['Recall']:.6f}\n  CM: {r['Confusion_Matrix']}")

ev("LightGBM_DART1_v6",y_test,preds_lgb1,notes=f"DART 4000t 10tr s42. P:{ls1.best_params}")
ev("LightGBM_DART2_v6",y_test,preds_lgb2,notes="DART same params s123")
ev("LightGBM_DART3_v6",y_test,preds_lgb3,notes="DART same params s999")
ev("XGBoost_gbdt_v6",y_test,preds_xgb,notes=f"XGB gbdt 25tr. P:{xs.best_params}")
ev("CatBoost_v6",y_test,preds_cat,notes=f"Cat 25tr. P:{cs.best_params}")
ev("Ensemble_v6_rank",y_test,fb,notes=f"Rank avg {len(qualified)} models: {list(qualified.keys())}")

pd.DataFrame({"y_true":y_test,"y_prob":fb}).to_csv("best_predictions.csv",index=False)
fi=pd.DataFrame({"feature":X_train_A.columns,"importance":best_model.feature_importances_}).sort_values("importance",ascending=False)
fi.to_csv("feature_importance.csv",index=False)
print("\nTop 15:"); print(fi.head(15).to_string())
print("\nPIPELINE COMPLETE")
