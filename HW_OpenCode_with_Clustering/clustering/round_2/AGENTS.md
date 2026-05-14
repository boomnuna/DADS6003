# AGENTS.md — Credit Card Clustering (Round 2: Enhanced)

## Goal
Improve on the Baseline results (K-Means Silhouette=0.2403, DBSCAN Silhouette=0.1409)
by adding: Outlier Removal, Feature Engineering, Elbow Method, and Cluster Profiling.
Produce a before/after comparison table at the end.

## Baseline Results (Round 1 — do NOT re-run, just use these numbers for comparison)
| Model   | Clusters | Silhouette |
|---------|----------|------------|
| K-Means | 3        | 0.2403     |
| DBSCAN  | 5        | 0.1409     |

---

## Data File
- `CC_GENERAL.csv` — 8,950 rows, 18 columns

## Columns to Exclude
- `CUST_ID` — identifier
- `KMeans_Cluster`, `DBSCAN_Cluster` — from round 1 (if present, drop them)

---

## Step 0 — Token Usage Tracker

```python
token_usage = {"input_tokens": 0, "output_tokens": 0}

def update_tokens(response_usage):
    token_usage["input_tokens"]  += response_usage.get("input_tokens", 0)
    token_usage["output_tokens"] += response_usage.get("output_tokens", 0)

def print_token_usage():
    total = token_usage["input_tokens"] + token_usage["output_tokens"]
    print(f"\n===== TOKEN USAGE =====")
    print(f"  Input tokens  : {token_usage['input_tokens']:,}")
    print(f"  Output tokens : {token_usage['output_tokens']:,}")
    print(f"  Total tokens  : {total:,}")
```

---

## Step 1 — Load and Basic Preprocessing

```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("CC_GENERAL.csv")

# Drop identifier and any leftover cluster columns from round 1
drop_cols = ["CUST_ID", "KMeans_Cluster", "DBSCAN_Cluster"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Fill missing values with median
df["CREDIT_LIMIT"]     = df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].median())
df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(df["MINIMUM_PAYMENTS"].median())

print(f"Shape after load     : {df.shape}")
print(f"Missing values       : {df.isnull().sum().sum()}")  # must be 0
```

---

## Step 2 — Feature Engineering (NEW in Round 2)

```python
# These ratios capture customer behavior better than raw amounts
# and reduce the effect of absolute scale differences

df["PURCHASE_TO_LIMIT_RATIO"]   = df["PURCHASES"]     / (df["CREDIT_LIMIT"] + 1)
df["CASH_ADVANCE_RATIO"]        = df["CASH_ADVANCE"]   / (df["CREDIT_LIMIT"] + 1)
df["PAYMENT_TO_BALANCE_RATIO"]  = df["PAYMENTS"]       / (df["BALANCE"] + 1)
df["INSTALLMENT_RATIO"]         = df["INSTALLMENTS_PURCHASES"] / (df["PURCHASES"] + 1)

print(f"Shape after feature engineering : {df.shape}")
print("New features added: PURCHASE_TO_LIMIT_RATIO, CASH_ADVANCE_RATIO,")
print("                    PAYMENT_TO_BALANCE_RATIO, INSTALLMENT_RATIO")
```

---

## Step 3 — Outlier Removal with IQR (NEW in Round 2)

```python
# Apply IQR outlier removal on key financial columns only
# (frequency/ratio columns 0-1 don't need it)

OUTLIER_COLS = [
    "BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "CASH_ADVANCE",
    "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS"
]

df_clean = df.copy()
outlier_mask = pd.Series(False, index=df_clean.index)

for col in OUTLIER_COLS:
    Q1  = df_clean[col].quantile(0.25)
    Q3  = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    col_outliers = (df_clean[col] < lower) | (df_clean[col] > upper)
    outlier_mask = outlier_mask | col_outliers
    print(f"  {col:30s}: {col_outliers.sum()} outliers removed")

df_clean = df_clean[~outlier_mask]
removed  = len(df) - len(df_clean)
print(f"\nTotal rows removed   : {removed} ({removed/len(df)*100:.1f}%)")
print(f"Rows remaining       : {len(df_clean)}")
```

---

## Step 4 — Scale Features

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

FEATURES = [c for c in df_clean.columns]  # all columns are now features

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)

# PCA for DBSCAN (95% variance) and visualization (2D)
pca_full = PCA(n_components=0.95, random_state=42)
X_pca_full = pca_full.fit_transform(X_scaled)

pca_2d   = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_scaled)

print(f"Scaled shape         : {X_scaled.shape}")
print(f"PCA full components  : {X_pca_full.shape[1]} (95% variance)")
print(f"PCA 2D variance      : {pca_2d.explained_variance_ratio_.sum():.2%}")
```

---

## Step 5 — K-Means with Elbow Method + Silhouette (ENHANCED)

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

K_RANGE       = range(2, 11)
kmeans_results = []
inertias       = []
silhouettes    = []

print("\n===== K-MEANS: ELBOW + SILHOUETTE =====")
for k in K_RANGE:
    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil    = silhouette_score(X_scaled, labels, sample_size=2000, random_state=42)

    inertias.append(km.inertia_)
    silhouettes.append(sil)
    kmeans_results.append({
        "k": k, "inertia": round(km.inertia_, 2),
        "silhouette": round(sil, 4), "model": km, "labels": labels
    })
    print(f"  k={k:2d}  Inertia={km.inertia_:>12.2f}  Silhouette={sil:.4f}")

# Plot Elbow + Silhouette side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(list(K_RANGE), inertias, "bo-", linewidth=2)
axes[0].set_title("Elbow Method — Inertia vs k")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia")
axes[0].grid(True)

axes[1].plot(list(K_RANGE), silhouettes, "rs-", linewidth=2)
axes[1].set_title("Silhouette Score vs k")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("elbow_silhouette.png", dpi=150)
plt.close()
print("\nSaved → elbow_silhouette.png")

# Best k = highest silhouette
best_kmeans = max(kmeans_results, key=lambda r: r["silhouette"])
print(f"\n✅ Best K-Means k = {best_kmeans['k']}  (Silhouette = {best_kmeans['silhouette']})")
```

---

## Step 6 — DBSCAN (same logic as Round 1, on cleaned data)

```python
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

MIN_SAMPLES  = 5
EPS_VALUES   = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
dbscan_results = []

print("\n===== DBSCAN: SEARCHING OPTIMAL eps =====")
for eps in EPS_VALUES:
    db     = DBSCAN(eps=eps, min_samples=MIN_SAMPLES)
    labels = db.fit_predict(X_pca_full)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    noise_pct  = n_noise / len(labels) * 100

    if n_clusters >= 2:
        mask = labels != -1
        sil  = silhouette_score(X_pca_full[mask], labels[mask],
                                sample_size=2000, random_state=42)
        sil_str = f"{sil:.4f}"
    else:
        sil = -1
        sil_str = "N/A"

    dbscan_results.append({
        "eps": eps, "n_clusters": n_clusters, "n_noise": n_noise,
        "noise_pct": round(noise_pct, 2), "silhouette": sil, "labels": labels
    })
    print(f"  eps={eps:.1f}  clusters={n_clusters:2d}  noise={n_noise:4d} ({noise_pct:.1f}%)  Silhouette={sil_str}")

valid_dbscan = [r for r in dbscan_results if r["n_clusters"] >= 2]
best_dbscan  = max(valid_dbscan, key=lambda r: r["silhouette"])
print(f"\n✅ Best DBSCAN eps={best_dbscan['eps']}  "
      f"clusters={best_dbscan['n_clusters']}  "
      f"Silhouette={best_dbscan['silhouette']:.4f}")
```

---

## Step 7 — Cluster Profiling (NEW in Round 2)

```python
# Add K-Means labels back to the cleaned dataframe for profiling
df_profile = df_clean.copy()
df_profile["KMeans_Cluster"] = best_kmeans["labels"]

print("\n===== CLUSTER PROFILING (K-Means) =====")

# Key features to profile
PROFILE_COLS = [
    "BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT",
    "PAYMENTS", "PRC_FULL_PAYMENT", "PURCHASES_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY", "PURCHASE_TO_LIMIT_RATIO",
    "CASH_ADVANCE_RATIO", "PAYMENT_TO_BALANCE_RATIO"
]

profile = df_profile.groupby("KMeans_Cluster")[PROFILE_COLS].mean().round(3)
print(profile.to_string())
profile.to_csv("cluster_profile.csv")
print("\nSaved → cluster_profile.csv")

# Cluster size
print("\n--- Cluster Sizes ---")
sizes = df_profile["KMeans_Cluster"].value_counts().sort_index()
for cluster, count in sizes.items():
    pct = count / len(df_profile) * 100
    print(f"  Cluster {cluster}: {count:5d} customers ({pct:.1f}%)")

# Auto-label clusters based on dominant behavior
print("\n--- Cluster Interpretation ---")
for cluster in profile.index:
    row = profile.loc[cluster]
    traits = []
    if row["PURCHASES"]            > profile["PURCHASES"].mean():            traits.append("High Spender")
    if row["CASH_ADVANCE"]         > profile["CASH_ADVANCE"].mean():         traits.append("Cash Advance User")
    if row["BALANCE"]              > profile["BALANCE"].mean():              traits.append("High Balance")
    if row["PRC_FULL_PAYMENT"]     > profile["PRC_FULL_PAYMENT"].mean():     traits.append("Pays in Full")
    if row["PURCHASES_FREQUENCY"]  < profile["PURCHASES_FREQUENCY"].mean():  traits.append("Infrequent Buyer")
    if not traits:
        traits = ["Average User"]
    print(f"  Cluster {cluster}: {', '.join(traits)}")
```

---

## Step 8 — Visualize Enhanced Clusters

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

scatter1 = axes[0].scatter(
    X_pca_2d[:, 0], X_pca_2d[:, 1],
    c=best_kmeans["labels"], cmap="tab10", s=5, alpha=0.5
)
axes[0].set_title(f"K-Means Enhanced (k={best_kmeans['k']}, Silhouette={best_kmeans['silhouette']:.4f})")
axes[0].set_xlabel("PCA Component 1")
axes[0].set_ylabel("PCA Component 2")
plt.colorbar(scatter1, ax=axes[0])

scatter2 = axes[1].scatter(
    X_pca_2d[:, 0], X_pca_2d[:, 1],
    c=best_dbscan["labels"], cmap="tab10", s=5, alpha=0.5
)
axes[1].set_title(f"DBSCAN Enhanced (eps={best_dbscan['eps']}, clusters={best_dbscan['n_clusters']}, Silhouette={best_dbscan['silhouette']:.4f})")
axes[1].set_xlabel("PCA Component 1")
axes[1].set_ylabel("PCA Component 2")
plt.colorbar(scatter2, ax=axes[1])

plt.tight_layout()
plt.savefig("cluster_visualization_enhanced.png", dpi=150)
plt.close()
print("Saved → cluster_visualization_enhanced.png")
```

---

## Step 9 — Before/After Comparison Table

```python
import pandas as pd

BASELINE = {
    "KMeans_Clusters":   3,
    "KMeans_Silhouette": 0.2403,
    "DBSCAN_Clusters":   5,
    "DBSCAN_Silhouette": 0.1409,
}

enhanced = {
    "KMeans_Clusters":   best_kmeans["k"],
    "KMeans_Silhouette": best_kmeans["silhouette"],
    "DBSCAN_Clusters":   best_dbscan["n_clusters"],
    "DBSCAN_Silhouette": round(best_dbscan["silhouette"], 4),
}

def delta(new, old):
    return f"+{((new-old)/abs(old)*100):.1f}%" if new > old else f"{((new-old)/abs(old)*100):.1f}%"

print("\n" + "=" * 60)
print("         BEFORE vs AFTER COMPARISON")
print("=" * 60)
print(f"{'Metric':<30} {'Baseline':>10} {'Enhanced':>10} {'Δ':>10}")
print("-" * 60)
print(f"{'K-Means Clusters':<30} {BASELINE['KMeans_Clusters']:>10} {enhanced['KMeans_Clusters']:>10}")
print(f"{'K-Means Silhouette':<30} {BASELINE['KMeans_Silhouette']:>10.4f} {enhanced['KMeans_Silhouette']:>10.4f} {delta(enhanced['KMeans_Silhouette'], BASELINE['KMeans_Silhouette']):>10}")
print(f"{'DBSCAN Clusters':<30} {BASELINE['DBSCAN_Clusters']:>10} {enhanced['DBSCAN_Clusters']:>10}")
print(f"{'DBSCAN Silhouette':<30} {BASELINE['DBSCAN_Silhouette']:>10.4f} {enhanced['DBSCAN_Silhouette']:>10.4f} {delta(enhanced['DBSCAN_Silhouette'], BASELINE['DBSCAN_Silhouette']):>10}")
print(f"{'Techniques Added':<30} {'None':>10} {'4':>10}")
print("=" * 60)

# Save comparison
comparison = pd.DataFrame([
    {"Metric": "K-Means Clusters",   "Baseline": BASELINE["KMeans_Clusters"],   "Enhanced": enhanced["KMeans_Clusters"]},
    {"Metric": "K-Means Silhouette", "Baseline": BASELINE["KMeans_Silhouette"], "Enhanced": enhanced["KMeans_Silhouette"]},
    {"Metric": "DBSCAN Clusters",    "Baseline": BASELINE["DBSCAN_Clusters"],   "Enhanced": enhanced["DBSCAN_Clusters"]},
    {"Metric": "DBSCAN Silhouette",  "Baseline": BASELINE["DBSCAN_Silhouette"], "Enhanced": enhanced["DBSCAN_Silhouette"]},
])
comparison.to_csv("before_after_comparison.csv", index=False)
print("Saved → before_after_comparison.csv")
```

---

## Step 10 — Export Results + Token Usage

```python
# Save final clustering results
df_out = pd.read_csv("CC_GENERAL.csv")
# Map enhanced labels back (drop outliers will have NaN — fill with -99)
df_out["KMeans_Cluster_Enhanced"] = -99
df_out["DBSCAN_Cluster_Enhanced"]  = -99
df_out.loc[df_clean.index, "KMeans_Cluster_Enhanced"] = best_kmeans["labels"]
df_out.loc[df_clean.index, "DBSCAN_Cluster_Enhanced"]  = best_dbscan["labels"]
df_out.to_csv("clustering_results_enhanced.csv", index=False)
print("Saved → clustering_results_enhanced.csv  (-99 = removed as outlier)")

print_token_usage()
```

---

## Expected Output Files

| File | Description |
|------|-------------|
| `elbow_silhouette.png` | Elbow + Silhouette plots for K-Means |
| `cluster_visualization_enhanced.png` | Side-by-side cluster plots (enhanced) |
| `cluster_profile.csv` | Mean feature values per cluster |
| `before_after_comparison.csv` | Baseline vs Enhanced silhouette scores |
| `clustering_results_enhanced.csv` | Original data + enhanced cluster labels |

## Expected Console Output

1. Preprocessing + outlier removal count per column
2. K-Means table: inertia + silhouette for k=2..10
3. DBSCAN table: clusters + noise % + silhouette for each eps
4. Cluster profiling: mean features per cluster + auto interpretation
5. Before/After comparison table with Δ %
6. Token usage

---

## Constraints

- Use the same `CC_GENERAL.csv` as Round 1 — do NOT use `clustering_results.csv`
- Baseline numbers (Round 1) are hardcoded in Step 9 — do not recompute them
- Outlier removal uses IQR × 1.5 on financial columns only
- Removed outlier rows get label `-99` in the final CSV (not dropped from export)
- Use `random_state=42` and `n_init=10` for K-Means
- Elbow method and Silhouette must both be plotted in one figure
- Cluster profiling must include auto-interpretation labels
