# AGENTS.md — Titanic Survival Classifier

## Project Goal
Train, tune, and compare multiple classifiers on Titanic survival data.
For each model, test both with and without PCA, and evaluate using k=5 and k=10 Stratified K-Fold cross-validation.
Select the best model based on CV AUC, then evaluate it once on the held-out test set and export predictions.

---

## Data Files
| File | Rows | Purpose |
|------|------|---------|
| `train_data.csv` | 792 | Training, GridSearch, and K-Fold CV |
| `test_data.csv` | 100 | Final evaluation only — touch once at the very end |

> ⚠️ Do NOT use `test_data.csv` during training, feature selection, GridSearch, or cross-validation. It simulates unseen real-world data.

---

## Data Schema

### Label
- `Survived` — binary target: `0` = did not survive, `1` = survived
- Class distribution: 486 negatives / 306 positives (imbalanced → use StratifiedKFold)

### Features (14 total)
| Feature | Description |
|---------|-------------|
| `Sex` | binary: 0 = female, 1 = male |
| `Age` | float [0–1], scaled |
| `Fare` | float [0–1], scaled |
| `Pclass_1`, `Pclass_2`, `Pclass_3` | one-hot passenger class |
| `Family_size` | float [0–1], scaled |
| `Title_1`, `Title_2`, `Title_3`, `Title_4` | one-hot passenger title |
| `Emb_1`, `Emb_2`, `Emb_3` | one-hot embarkation port |

### Columns to Exclude (never use as features)
- `Unnamed: 0` — row index
- `PassengerId` — identifier
- `Survived` — label (target)

---

## Step 1 — Load Data

```python
import pandas as pd

train = pd.read_csv("train_data.csv")
test  = pd.read_csv("test_data.csv")

EXCLUDE  = ["Unnamed: 0", "PassengerId", "Survived"]
FEATURES = [c for c in train.columns if c not in EXCLUDE]

X_train, y_train = train[FEATURES].values, train["Survived"].values
X_test,  y_test  = test[FEATURES].values,  test["Survived"].values
```

---

## Step 2 — GPU Detection and Model Definitions

```python
from sklearn.decomposition  import PCA
from sklearn.pipeline       import Pipeline
from sklearn.preprocessing  import StandardScaler

PCA_COMPONENTS = [7, 10]

# ── GPU detection ──────────────────────────────────────────────────────────────
# Try to import cuml (NVIDIA RAPIDS). If unavailable, fall back to scikit-learn.
# Nothing else in the pipeline changes — same GridSearch, same params, same steps.

try:
    from cuml.linear_model   import LogisticRegression
    from cuml.neighbors      import KNeighborsClassifier
    from cuml.svm            import SVC
    from cuml.ensemble       import RandomForestClassifier
    # cuml has no GaussianNB, DecisionTree, or MLP — fall back those to sklearn
    from sklearn.naive_bayes    import GaussianNB
    from sklearn.tree           import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier
    GPU = True
    print("✅ cuml found — using GPU for LogisticRegression, KNN, SVM, RandomForest")
except ImportError:
    from sklearn.linear_model   import LogisticRegression
    from sklearn.naive_bayes    import GaussianNB
    from sklearn.neighbors      import KNeighborsClassifier
    from sklearn.tree           import DecisionTreeClassifier
    from sklearn.ensemble       import RandomForestClassifier
    from sklearn.svm            import SVC
    from sklearn.neural_network import MLPClassifier
    GPU = False
    print("⚠️  cuml not found — using CPU (sklearn) for all models")

# ── Model configs (identical regardless of GPU/CPU) ───────────────────────────

MODEL_CONFIGS = [
    {
        "name": "LogisticRegression",
        "estimator": LogisticRegression(max_iter=1000, random_state=42),
        "params": {
            "model__C": [0.01, 0.1, 1, 10],
            "model__penalty": ["l1", "l2"],
            "model__solver": ["liblinear"] if not GPU else ["qn"],
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
        "estimator": MLPClassifier(max_iter=500, random_state=42),
        "params": {
            "model__hidden_layer_sizes": [(64,), (128,), (64, 32)],
            "model__alpha": [0.0001, 0.001],
            "model__learning_rate_init": [0.001, 0.01],
        },
    },
]
```

---

## Step 3 — GridSearch with K-Fold CV (k=5 and k=10)

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold

results = []  # one dict per (model, pca_variant, k) combination

for k in [5, 10]:
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    for cfg in MODEL_CONFIGS:
        for use_pca in [False, True]:

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
                "model":       cfg["name"],
                "pca":         "Yes" if use_pca else "No",
                "k":           k,
                "cv_auc":      round(search.best_score_, 4),
                "best_params": search.best_params_,
                "fitted":      search.best_estimator_,
            })

            print(
                f"[k={k}] {cfg['name']:20s} PCA={str(use_pca):5s} "
                f"→ CV AUC = {search.best_score_:.4f}"
            )
```

---

## Step 4 — Build Comparison Table

```python
import pandas as pd

summary = pd.DataFrame([
    {
        "Model":  r["model"],
        "PCA":    r["pca"],
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
```

---

## Step 5 — Select Best Model and Evaluate on Test Set

```python
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, classification_report
)

# Pick highest CV AUC; use k=10 as tiebreaker
best = max(results, key=lambda r: (r["cv_auc"], r["k"] == 10))
best_pipeline = best["fitted"]

print(f"\n===== BEST MODEL =====")
print(f"  Model      : {best['model']}")
print(f"  PCA        : {best['pca']}")
print(f"  k-Fold     : {best['k']}")
print(f"  CV AUC     : {best['cv_auc']}")
print(f"  Best Params: {best['best_params']}")

# Evaluate on test set — ONE TIME ONLY
y_pred      = best_pipeline.predict(X_test)
y_pred_prob = best_pipeline.predict_proba(X_test)[:, 1]

print("\n===== TEST SET RESULTS =====")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\nTest AUC Score : {roc_auc_score(y_test, y_pred_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Survived", "Survived"]))
```

---

## Step 6 — Export Predictions

```python
output = test.copy()
output["predicted_label"] = y_pred
output.to_csv("predictions.csv", index=False)
print("Saved → predictions.csv")
```

---

## Expected Output Files

| File | Description |
|------|-------------|
| `comparison_table.csv` | AUC of every model × PCA × k combination |
| `predictions.csv` | test_data.csv + `predicted_label` column from best model |

## Expected Console Output

1. Live progress lines during GridSearch (one per model/PCA/k combo → 28 lines total)
2. Full comparison table sorted by Best AUC
3. Best model summary (name, PCA used, best params, CV AUC)
4. Confusion matrix + Test AUC + Classification report

---

## GPU Setup (Optional)

To enable GPU acceleration on NVIDIA hardware, install RAPIDS cuml **before** running:
```bash
pip install cuml-cu11  # for CUDA 11.x
# or
pip install cuml-cu12  # for CUDA 12.x
```
Check your CUDA version with: `nvidia-smi`

If cuml is not installed, the script automatically falls back to scikit-learn (CPU). No code changes needed.

---

## Constraints

- `test_data.csv` must only be touched in Step 5 and Step 6 — never before
- Use `StratifiedKFold` (not plain `KFold`) because classes are imbalanced
- Every pipeline must start with `StandardScaler` as the first step
- Use `random_state=42` wherever the parameter is supported
- Run both k=5 and k=10 for every model × PCA variant combination
- Do not add, remove, or engineer features beyond those listed in the schema
