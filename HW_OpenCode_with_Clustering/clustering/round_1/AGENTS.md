# AGENTS.md — Credit Card Customer Clustering

## Project Goal
Train and compare two unsupervised clustering models — **K-Means** and **DBSCAN** — on credit card customer data.
Find the optimal number of clusters, evaluate using Silhouette Score, and export results.
Track and report total Anthropic API token usage at the end.

---

## Data File
| File | Rows | Features | Source |
|------|------|----------|--------|
| `CC_GENERAL.csv` | 8,950 | 17 (after dropping CUST_ID) | [Kaggle](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata) |

---

## Data Schema

### Exclude (never use as features)
- `CUST_ID` — customer identifier, not a feature

### Features (17 total, all numeric)
| Feature | Description |
|---------|-------------|
| `BALANCE` | Balance amount left in account |
| `BALANCE_FREQUENCY` | How frequently balance is updated (0–1) |
| `PURCHASES` | Amount of purchases made |
| `ONEOFF_PURCHASES` | Maximum purchase amount in one transaction |
| `INSTALLMENTS_PURCHASES` | Amount of purchases done in installments |
| `CASH_ADVANCE` | Cash advance amount given by user |
| `PURCHASES_FREQUENCY` | How frequently purchases are made (0–1) |
| `ONEOFF_PURCHASES_FREQUENCY` | How frequently one-off purchases are made (0–1) |
| `PURCHASES_INSTALLMENTS_FREQUENCY` | How frequently installment purchases are made (0–1) |
| `CASH_ADVANCE_FREQUENCY` | How frequently cash advances are made |
| `CASH_ADVANCE_TRX` | Number of cash advance transactions |
| `PURCHASES_TRX` | Number of purchase transactions |
| `CREDIT_LIMIT` | Credit card limit (1 missing value) |
| `PAYMENTS` | Amount of payment made by user |
| `MINIMUM_PAYMENTS` | Minimum amount of payments made (313 missing values) |
| `PRC_FULL_PAYMENT` | Percent of full payment made (0–1) |
| `TENURE` | Tenure of credit card service |

### Known Missing Values
- `CREDIT_LIMIT`: 1 missing → fill with median
- `MINIMUM_PAYMENTS`: 313 missing → fill with median

---

## Step 0 — Track Token Usage

```python
# Token counter — increment after every Anthropic API call if used
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

## Step 1 — Load and Preprocess Data

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load
df = pd.read_csv("CC_GENERAL.csv")

# Drop identifier
df = df.drop(columns=["CUST_ID"])

# Fill missing values with median
df["CREDIT_LIMIT"]    = df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].median())
df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(df["MINIMUM_PAYMENTS"].median())

print(f"Dataset shape    : {df.shape}")
print(f"Missing values   : {df.isnull().sum().sum()}")  # must be 0

# Scale features (required for both K-Means and DBSCAN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

print(f"Scaling complete : {X_scaled.shape}")
```

---

## Step 2 — Dimensionality Reduction with PCA (for DBSCAN and visualization)

```python
# Reduce to 2D for visualization and DBSCAN distance sensitivity
pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_scaled)
print(f"PCA 2D variance explained: {pca_2d.explained_variance_ratio_.sum():.2%}")

# Also keep full PCA (retain 95% variance) for cleaner DBSCAN clustering
pca_full = PCA(n_components=0.95, random_state=42)
X_pca_full = pca_full.fit_transform(X_scaled)
print(f"PCA full components kept : {X_pca_full.shape[1]} (95% variance)")
```

---

## Step 3 — K-Means: Find Optimal Number of Clusters

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

kmeans_results = []
K_RANGE = range(2, 11)  # test k = 2 to 10

print("\n===== K-MEANS: SEARCHING OPTIMAL K =====")
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertia = km.inertia_
    sil     = silhouette_score(X_scaled, labels, sample_size=2000, random_state=42)

    kmeans_results.append({
        "k":         k,
        "inertia":   round(inertia, 2),
        "silhouette": round(sil, 4),
        "model":     km,
        "labels":    labels,
    })
    print(f"  k={k:2d}  →  Inertia={inertia:>12.2f}  Silhouette={sil:.4f}")

# Select best k by highest silhouette score
best_kmeans = max(kmeans_results, key=lambda r: r["silhouette"])
print(f"\n✅ Best K-Means k = {best_kmeans['k']}  (Silhouette = {best_kmeans['silhouette']})")
```

---

## Step 4 — DBSCAN: Find Optimal eps with k-Distance Graph

```python
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Use k-distance graph to estimate eps (k = min_samples)
MIN_SAMPLES = 5

nbrs = NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(X_pca_full)
distances, _ = nbrs.kneighbors(X_pca_full)
k_distances   = np.sort(distances[:, -1])

# Plot k-distance to find the "elbow" (best eps)
plt.figure(figsize=(8, 4))
plt.plot(k_distances)
plt.title(f"k-Distance Graph (k={MIN_SAMPLES}) — look for the elbow")
plt.xlabel("Points sorted by distance")
plt.ylabel(f"{MIN_SAMPLES}-NN distance")
plt.grid(True)
plt.tight_layout()
plt.savefig("dbscan_kdistance.png", dpi=150)
plt.close()
print("Saved → dbscan_kdistance.png  (inspect the elbow to choose eps)")

# Search multiple eps values and pick best silhouette
EPS_VALUES = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
dbscan_results = []

print("\n===== DBSCAN: SEARCHING OPTIMAL eps =====")
for eps in EPS_VALUES:
    db = DBSCAN(eps=eps, min_samples=MIN_SAMPLES)
    labels = db.fit_predict(X_pca_full)

    n_clusters  = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise     = (labels == -1).sum()
    noise_pct   = n_noise / len(labels) * 100

    if n_clusters >= 2:
        # Silhouette excludes noise points (label == -1)
        mask = labels != -1
        sil  = silhouette_score(X_pca_full[mask], labels[mask],
                                sample_size=2000, random_state=42)
        sil_str = f"{sil:.4f}"
    else:
        sil     = -1
        sil_str = "N/A (too few clusters)"

    dbscan_results.append({
        "eps":        eps,
        "n_clusters": n_clusters,
        "n_noise":    n_noise,
        "noise_pct":  round(noise_pct, 2),
        "silhouette": sil,
        "labels":     labels,
    })
    print(f"  eps={eps:.1f}  →  clusters={n_clusters:2d}  noise={n_noise:4d} ({noise_pct:.1f}%)  Silhouette={sil_str}")

# Select best eps by highest silhouette (among runs with >= 2 clusters)
valid_dbscan = [r for r in dbscan_results if r["n_clusters"] >= 2]
best_dbscan  = max(valid_dbscan, key=lambda r: r["silhouette"])
print(f"\n✅ Best DBSCAN eps = {best_dbscan['eps']}  "
      f"clusters = {best_dbscan['n_clusters']}  "
      f"(Silhouette = {best_dbscan['silhouette']:.4f})")
```

---

## Step 5 — Compare K-Means vs DBSCAN

```python
print("\n===== MODEL COMPARISON =====")
print(f"{'Model':<20} {'Clusters':>10} {'Silhouette':>12}")
print("-" * 45)
print(f"{'K-Means':.<20} {best_kmeans['k']:>10} {best_kmeans['silhouette']:>12.4f}")
print(f"{'DBSCAN':.<20} {best_dbscan['n_clusters']:>10} {best_dbscan['silhouette']:>12.4f}")

# Full K-Means table
print("\n--- K-Means (all k) ---")
for r in kmeans_results:
    marker = " ← best" if r["k"] == best_kmeans["k"] else ""
    print(f"  k={r['k']:2d}  Silhouette={r['silhouette']:.4f}{marker}")

# Full DBSCAN table
print("\n--- DBSCAN (all eps) ---")
for r in dbscan_results:
    marker = " ← best" if r["eps"] == best_dbscan["eps"] else ""
    sil_str = f"{r['silhouette']:.4f}" if r["silhouette"] != -1 else "N/A"
    print(f"  eps={r['eps']:.1f}  clusters={r['n_clusters']:2d}  noise={r['noise_pct']:.1f}%  Silhouette={sil_str}{marker}")
```

---

## Step 6 — Visualize Clusters (2D PCA)

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# K-Means plot
scatter1 = axes[0].scatter(
    X_pca_2d[:, 0], X_pca_2d[:, 1],
    c=best_kmeans["labels"], cmap="tab10", s=5, alpha=0.5
)
axes[0].set_title(f"K-Means (k={best_kmeans['k']}, Silhouette={best_kmeans['silhouette']:.4f})")
axes[0].set_xlabel("PCA Component 1")
axes[0].set_ylabel("PCA Component 2")
plt.colorbar(scatter1, ax=axes[0])

# DBSCAN plot
scatter2 = axes[1].scatter(
    X_pca_2d[:, 0], X_pca_2d[:, 1],
    c=best_dbscan["labels"], cmap="tab10", s=5, alpha=0.5
)
axes[1].set_title(f"DBSCAN (eps={best_dbscan['eps']}, clusters={best_dbscan['n_clusters']}, Silhouette={best_dbscan['silhouette']:.4f})")
axes[1].set_xlabel("PCA Component 1")
axes[1].set_ylabel("PCA Component 2")
plt.colorbar(scatter2, ax=axes[1])

plt.tight_layout()
plt.savefig("cluster_visualization.png", dpi=150)
plt.close()
print("Saved → cluster_visualization.png")
```

---

## Step 7 — Export Results

```python
# Add best K-Means cluster labels to original data
df_out = pd.read_csv("CC_GENERAL.csv")
df_out["KMeans_Cluster"] = best_kmeans["labels"]
df_out["DBSCAN_Cluster"]  = best_dbscan["labels"]   # -1 = noise point
df_out.to_csv("clustering_results.csv", index=False)
print("Saved → clustering_results.csv")

# Save summary table
summary = pd.DataFrame([
    {
        "Model":      "K-Means",
        "N_Clusters": best_kmeans["k"],
        "Silhouette": best_kmeans["silhouette"],
        "Notes":      f"Best of k=2..10",
    },
    {
        "Model":      "DBSCAN",
        "N_Clusters": best_dbscan["n_clusters"],
        "Silhouette": round(best_dbscan["silhouette"], 4),
        "Notes":      f"eps={best_dbscan['eps']}, noise={best_dbscan['noise_pct']}%, min_samples={MIN_SAMPLES}",
    },
])
summary.to_csv("summary.csv", index=False)
print("Saved → summary.csv")
```

---

## Step 8 — Print Final Report + Token Usage

```python
print("\n" + "=" * 50)
print("       FINAL REPORT")
print("=" * 50)
print(f"\n📌 K-Means")
print(f"   Optimal clusters : {best_kmeans['k']}")
print(f"   Silhouette score : {best_kmeans['silhouette']}")

print(f"\n📌 DBSCAN")
print(f"   Optimal clusters : {best_dbscan['n_clusters']}")
print(f"   Silhouette score : {best_dbscan['silhouette']:.4f}")
print(f"   Noise points     : {best_dbscan['n_noise']} ({best_dbscan['noise_pct']}%)")
print(f"   Best eps         : {best_dbscan['eps']}")

print(f"\n📌 Output Files")
print(f"   clustering_results.csv   — original data + cluster labels")
print(f"   summary.csv              — model comparison table")
print(f"   cluster_visualization.png — 2D PCA cluster plots")
print(f"   dbscan_kdistance.png      — k-distance elbow graph")

# Token usage (update this if Anthropic API was called during the run)
print_token_usage()
```

---

## Expected Output Files

| File | Description |
|------|-------------|
| `clustering_results.csv` | Original data + `KMeans_Cluster` + `DBSCAN_Cluster` columns |
| `summary.csv` | Best model per algorithm with silhouette scores |
| `cluster_visualization.png` | Side-by-side 2D PCA scatter plots |
| `dbscan_kdistance.png` | k-distance graph to justify eps selection |

## Expected Console Output

1. Preprocessing confirmation (shape, zero missing values after fill)
2. K-Means table: inertia + silhouette for k=2..10
3. DBSCAN table: clusters + noise % + silhouette for each eps
4. Final comparison table (K-Means vs DBSCAN)
5. Token usage summary

---

## Submission Checklist

- [ ] `AGENTS.md` — this file
- [ ] **Number of clusters** — reported for both K-Means (best k) and DBSCAN (best eps)
- [ ] **Silhouette scores** — reported for all k values and all eps values, plus best of each
- [ ] **Token usage** — input tokens, output tokens, total (printed in Step 8)

---

## Constraints

- Drop `CUST_ID` — it is an identifier, not a feature
- Fill missing values with **median** (not mean, not drop)
- Always apply `StandardScaler` before clustering
- Use `PCA(n_components=0.95)` for DBSCAN input (reduces noise sensitivity)
- Use `PCA(n_components=2)` only for visualization
- K-Means: test k = 2 through 10 inclusive
- DBSCAN: test eps ∈ {0.3, 0.5, 0.7, 1.0, 1.5, 2.0} with min_samples=5
- Select best model by highest Silhouette score
- DBSCAN noise points (label = -1) must be excluded when computing silhouette
- Do not use ground-truth labels — this is unsupervised learning
