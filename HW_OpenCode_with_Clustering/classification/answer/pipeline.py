import warnings
import pandas as pd
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report

# Step 1 — Load Data
train = pd.read_csv("train_data.csv")
test = pd.read_csv("test_data.csv")

EXCLUDE = ["Unnamed: 0", "PassengerId", "Survived"]
FEATURES = [c for c in train.columns if c not in EXCLUDE]

X_train, y_train = train[FEATURES].values, train["Survived"].values
X_test, y_test = test[FEATURES].values, test["Survived"].values

PCA_COMPONENTS = [7, 10]

MODEL_CONFIGS = [
    {
        "name": "LogisticRegression",
        "estimator": LogisticRegression(max_iter=1000, random_state=42),
        "params": {
            "model__C": [0.01, 0.1, 1, 10],
            "model__penalty": ["l1", "l2"],
            "model__solver": ["liblinear"],
        },
    },
    {
        "name": "NaiveBayes",
        "estimator": GaussianNB(),
        "params": {
            "model__var_smoothing": [1e-9, 1e-7, 1e-5],
        },
    },
    {
        "name": "KNN",
        "estimator": KNeighborsClassifier(),
        "params": {
            "model__n_neighbors": [3, 5, 7, 11],
            "model__metric": ["euclidean", "manhattan"],
        },
    },
    {
        "name": "DecisionTree",
        "estimator": DecisionTreeClassifier(random_state=42),
        "params": {
            "model__max_depth": [3, 5, 10, None],
            "model__min_samples_split": [2, 5, 10],
        },
    },
    {
        "name": "RandomForest",
        "estimator": RandomForestClassifier(random_state=42),
        "params": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [5, 10, None],
            "model__min_samples_split": [2, 5],
        },
    },
    {
        "name": "SVM",
        "estimator": SVC(probability=True, random_state=42),
        "params": {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["rbf", "linear"],
        },
    },
    {
        "name": "NeuralNetwork",
        "estimator": MLPClassifier(max_iter=500, random_state=42, tol=1e-3, n_iter_no_change=10),
        "params": {
            "model__hidden_layer_sizes": [(64,), (128,), (64, 32)],
            "model__alpha": [0.0001, 0.001],
            "model__learning_rate_init": [0.001, 0.01],
        },
    },
]

# Step 3 — GridSearch with K-Fold CV (k=5 and k=10)
results = []

for k in [5, 10]:
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    for cfg in MODEL_CONFIGS:
        for use_pca in [False, True]:
            print(f"[k={k}] Starting {cfg['name']:20s} PCA={str(use_pca):5s} ...")
            steps = [("scaler", StandardScaler())]
            param_grid = dict(cfg["params"])

            if use_pca:
                steps.append(("pca", PCA()))
                param_grid["pca__n_components"] = PCA_COMPONENTS

            steps.append(("model", cfg["estimator"]))
            pipeline = Pipeline(steps)

            search = GridSearchCV(
                pipeline,
                param_grid,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_train, y_train)

            results.append({
                "model": cfg["name"],
                "pca": "Yes" if use_pca else "No",
                "k": k,
                "cv_auc": round(search.best_score_, 4),
                "best_params": search.best_params_,
                "fitted": search.best_estimator_,
            })

            print(
                f"[k={k}] {cfg['name']:20s} PCA={str(use_pca):5s} "
                f"=> CV AUC = {search.best_score_:.4f}"
            )

# Step 4 — Build Comparison Table
summary = pd.DataFrame([
    {
        "Model": r["model"],
        "PCA": r["pca"],
        "k-Fold": r["k"],
        "CV AUC": r["cv_auc"],
    }
    for r in results
])

pivot = summary.pivot_table(
    index=["Model", "PCA"],
    columns="k-Fold",
    values="CV AUC"
).rename(columns={5: "k=5 AUC", 10: "k=10 AUC"})

pivot["Best AUC"] = pivot.max(axis=1)
pivot = pivot.sort_values("Best AUC", ascending=False)

print("\n===== MODEL COMPARISON TABLE =====")
print(pivot.to_string())
pivot.to_csv("comparison_table.csv")
print("\nSaved → comparison_table.csv")

# Step 5 — Select Best Model and Evaluate on Test Set
best = max(results, key=lambda r: (r["cv_auc"], r["k"] == 10))
best_pipeline = best["fitted"]

print(f"\n===== BEST MODEL =====")
print(f"  Model      : {best['model']}")
print(f"  PCA        : {best['pca']}")
print(f"  k-Fold     : {best['k']}")
print(f"  CV AUC     : {best['cv_auc']}")
print(f"  Best Params: {best['best_params']}")

y_pred = best_pipeline.predict(X_test)
y_pred_prob = best_pipeline.predict_proba(X_test)[:, 1]

print("\n===== TEST SET RESULTS =====")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\nTest AUC Score : {roc_auc_score(y_test, y_pred_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Survived", "Survived"]))

# Step 6 — Export Predictions
output = test.copy()
output["predicted_label"] = y_pred
output.to_csv("predictions.csv", index=False)
print("Saved → predictions.csv")
