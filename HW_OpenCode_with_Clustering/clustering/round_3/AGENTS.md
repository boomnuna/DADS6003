# AGENTS.md — Credit Card Clustering (Round 3: Final)

## Goal
Build on Round 2 results by adding:
1. DBSCAN fine-tuning (fix 12 clusters → 4-6 meaningful clusters)
2. Hopkins Statistic (verify data is truly clusterable)
3. t-SNE Visualization (compare with PCA)
4. Davies-Bouldin Score (second validation metric)
5. Silhouette Plot per cluster (per-point quality)
6. Business Recommendation from Cluster Profiling

Produce a complete 3-round comparison table at the end.

---

## Baseline Numbers (do NOT recompute — hardcoded for comparison)

| Round | Model   | Clusters | Silhouette | Notes |
|-------|---------|----------|------------|-------|
| 1     | K-Means | 3        | 0.2403     | Raw data |
| 1     | DBSCAN  | 5        | 0.1409     | Raw data |
| 2     | K-Means | 2        | 0.1894     | + Outlier removal + Feature engineering |
| 2     | DBSCAN  | 12       | 0.2801     | + Outlier removal + Feature engineering |

---

## Data File
- `CC_GENERAL.csv` — 8,950 rows, 18 columns

## Columns to Exclude
- `CUST_ID` — identifier
- `KMeans_Cluster`, `DBSCAN_Cluster` — leftover from round 1/2, drop if present

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

## Step 1 — Load, Preprocess, Feature Engineering, Outlier Removal

```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("CC_GENERAL.csv")

# Drop identifier and leftover cluster columns
drop_cols = ["CUST_ID", "KMeans_Cluster", "DBSCAN_Cluster"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Fill missing with median
df["CREDIT_LIMIT"]     = df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].median())
df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(df["MINIMUM_PAYMENTS"].median())

# Feature Engineering
df["PURCHASE_TO_LIMIT_RATIO"]  = df["PURCHASES"]              / (df["CREDIT_LIMIT"] + 1)
df["CASH_ADVANCE_RATIO"]       = df["CASH_ADVANCE"]           / (df["CREDIT_LIMIT"] + 1)
df["PAYMENT_TO_BALANCE_RATIO"] = df["PAYMENTS"]               / (df["BALANCE"] + 1)
df["INSTALLMENT_RATIO"]        = df["INSTALLMENTS_PURCHASES"] / (df["PURCHASES"] + 1)

# Outlier Removal (IQR on financial columns)
OUTLIER_COLS = [
    "BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "CASH_ADVANCE",
    "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS"
]
outlier_mask = pd.Series(False, index=df.index)
for col in OUTLIER_COLS:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR    = Q3 - Q1
    outlier_mask |= (df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)

df_clean  = df[~outlier_mask].copy()
removed   = len(df) - len(df_clean)
print(f"Rows removed (outliers) : {removed} ({removed/len(df)*100:.1f}%)")
print(f"Rows remaining          : {len(df_clean)}")
print(f"Features                : {df_clean.shape[1]}")
```

---

## Step 2 — Scale + PCA + t-SNE

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

scaler     = StandardScaler()
X_scaled   = scaler.fit_transform(df_clean)

# PCA for DBSCAN (95% variance)
pca_full   = PCA(n_components=0.95, random_state=42)
X_pca_full = pca_full.fit_transform(X_scaled)

# PCA 2D for visualization
pca_2d     = PCA(n_components=2, random_state=42)
X_pca_2d   = pca_2d.fit_transform(X_scaled)

# t-SNE 2D for visualization (takes ~1-2 min)
print("Running t-SNE (this may take 1-2 minutes)...")
tsne       = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne     = tsne.fit_transform(X_scaled)

print(f"PCA full components : {X_pca_full.shape[1]} (95% variance)")
print(f"PCA 2D variance     : {pca_2d.explained_variance_ratio_.sum():.2%}")
print(f"t-SNE shape         : {X_tsne.shape}")
```

---

## Step 3 — Hopkins Statistic (is this data truly clusterable?)

```python
from sklearn.neighbors import NearestNeighbors

def hopkins_statistic(X, sample_size=150):
    """
    Hopkins statistic: H > 0.75 means data has meaningful cluster tendency.
    H ~ 0.5 means data is random (not clusterable).
    """
    np.random.seed(42)
    n, d   = X.shape
    sample = X[np.random.choice(n, sample_size, replace=False)]

    # Random uniform points in same bounding box
    mins   = X.min(axis=0)
    maxs   = X.max(axis=0)
    rand   = np.random.uniform(mins, maxs, (sample_size, d))

    nbrs   = NearestNeighbors(n_neighbors=2).fit(X)

    # Distances from sample points to nearest neighbor in X
    u_dist = nbrs.kneighbors(sample, n_neighbors=2)[0][:, 1]
    # Distances from random points to nearest neighbor in X
    w_dist = nbrs.kneighbors(rand,   n_neighbors=1)[0][:, 0]

    H = w_dist.sum() / (u_dist.sum() + w_dist.sum())
    return round(H, 4)

H = hopkins_statistic(X_scaled)
print(f"\n===== HOPKINS STATISTIC =====")
print(f"  H = {H}")
if H >= 0.75:
    print(f"  ✅ Data has strong cluster tendency (H >= 0.75) — safe to cluster")
elif H >= 0.5:
    print(f"  ⚠️  Data has moderate cluster tendency (0.5 <= H < 0.75)")
else:
    print(f"  ❌ Data appears random (H < 0.5) — clustering may not be meaningful")
```

---

## Step 4 — K-Means with Elbow + Silhouette + Davies-Bouldin

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

K_RANGE        = range(2, 11)
kmeans_results = []
inertias, silhouettes, db_scores = [], [], []

print("\n===== K-MEANS =====")
print(f"{'k':>3}  {'Inertia':>12}  {'Silhouette':>12}  {'Davies-Bouldin':>15}")
print("-" * 48)

for k in K_RANGE:
    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil    = silhouette_score(X_scaled, labels, sample_size=2000, random_state=42)
    db     = davies_bouldin_score(X_scaled, labels)

    inertias.append(km.inertia_)
    silhouettes.append(sil)
    db_scores.append(db)

    kmeans_results.append({
        "k": k, "inertia": round(km.inertia_, 2),
        "silhouette": round(sil, 4), "davies_bouldin": round(db, 4),
        "model": km, "labels": labels
    })
    print(f"{k:>3}  {km.inertia_:>12.2f}  {sil:>12.4f}  {db:>15.4f}")

# Best k = highest Silhouette (lower Davies-Bouldin is also better)
best_kmeans = max(kmeans_results, key=lambda r: r["silhouette"])
print(f"\n✅ Best K-Means k={best_kmeans['k']}  "
      f"Silhouette={best_kmeans['silhouette']}  "
      f"Davies-Bouldin={best_kmeans['davies_bouldin']}")

# Plot Elbow + Silhouette + Davies-Bouldin
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(list(K_RANGE), inertias, "bo-", linewidth=2)
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Inertia")
axes[0].grid(True)

axes[1].plot(list(K_RANGE), silhouettes, "rs-", linewidth=2)
axes[1].axvline(x=best_kmeans["k"], color="gray", linestyle="--", alpha=0.7)
axes[1].set_title("Silhouette Score (higher = better)")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Silhouette")
axes[1].grid(True)

axes[2].plot(list(K_RANGE), db_scores, "g^-", linewidth=2)
axes[2].axvline(x=best_kmeans["k"], color="gray", linestyle="--", alpha=0.7)
axes[2].set_title("Davies-Bouldin Score (lower = better)")
axes[2].set_xlabel("k")
axes[2].set_ylabel("Davies-Bouldin")
axes[2].grid(True)

plt.tight_layout()
plt.savefig("elbow_silhouette_db.png", dpi=150)
plt.close()
print("Saved → elbow_silhouette_db.png")
```

---

## Step 5 — DBSCAN Fine-Tuning (fix 12 clusters from Round 2)

```python
from sklearn.cluster import DBSCAN

MIN_SAMPLES    = 5
# Extend eps range beyond Round 2 to reduce cluster count
EPS_VALUES     = [0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
dbscan_results = []

print("\n===== DBSCAN FINE-TUNING =====")
print(f"{'eps':>5}  {'Clusters':>9}  {'Noise%':>7}  {'Silhouette':>12}  {'Davies-Bouldin':>15}")
print("-" * 55)

for eps in EPS_VALUES:
    db     = DBSCAN(eps=eps, min_samples=MIN_SAMPLES)
    labels = db.fit_predict(X_pca_full)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    noise_pct  = n_noise / len(labels) * 100

    if n_clusters >= 2:
        mask   = labels != -1
        sil    = silhouette_score(X_pca_full[mask], labels[mask],
                                  sample_size=2000, random_state=42)
        db_sco = davies_bouldin_score(X_pca_full[mask], labels[mask])
        sil_str  = f"{sil:.4f}"
        db_str   = f"{db_sco:.4f}"
    else:
        sil = db_sco = -1
        sil_str = db_str = "N/A"

    dbscan_results.append({
        "eps": eps, "n_clusters": n_clusters, "n_noise": n_noise,
        "noise_pct": round(noise_pct, 2), "silhouette": sil,
        "davies_bouldin": db_sco, "labels": labels
    })
    print(f"{eps:>5.1f}  {n_clusters:>9}  {noise_pct:>6.1f}%  {sil_str:>12}  {db_str:>15}")

# Best = highest silhouette among runs with 2-8 clusters (meaningful range)
valid = [r for r in dbscan_results if 2 <= r["n_clusters"] <= 8]
if not valid:
    valid = [r for r in dbscan_results if r["n_clusters"] >= 2]
best_dbscan = max(valid, key=lambda r: r["silhouette"])

print(f"\n✅ Best DBSCAN eps={best_dbscan['eps']}  "
      f"clusters={best_dbscan['n_clusters']}  "
      f"Silhouette={best_dbscan['silhouette']:.4f}  "
      f"Davies-Bouldin={best_dbscan['davies_bouldin']:.4f}")
```

---

## Step 6 — Silhouette Plot per Cluster (K-Means)

```python
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm

fig, ax = plt.subplots(figsize=(8, 6))
labels_km   = best_kmeans["labels"]
sil_vals    = silhouette_samples(X_scaled, labels_km)
n_clusters  = best_kmeans["k"]
y_lower     = 10

for i in range(n_clusters):
    ith_sil = np.sort(sil_vals[labels_km == i])
    size_i  = ith_sil.shape[0]
    y_upper = y_lower + size_i

    color = cm.tab10(i / n_clusters)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_sil,
                     facecolor=color, edgecolor=color, alpha=0.7)
    ax.text(-0.05, y_lower + 0.5 * size_i, str(i))
    y_lower = y_upper + 10

ax.axvline(x=best_kmeans["silhouette"], color="red", linestyle="--",
           label=f"Avg Silhouette = {best_kmeans['silhouette']:.4f}")
ax.set_title(f"Silhouette Plot — K-Means (k={n_clusters})")
ax.set_xlabel("Silhouette Coefficient")
ax.set_ylabel("Cluster")
ax.legend()
plt.tight_layout()
plt.savefig("silhouette_plot.png", dpi=150)
plt.close()
print("Saved → silhouette_plot.png")
```

---

## Step 7 — PCA vs t-SNE Visualization (4 plots)

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Row 1: PCA
sc1 = axes[0, 0].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],
    c=best_kmeans["labels"], cmap="tab10", s=5, alpha=0.5)
axes[0, 0].set_title(f"PCA — K-Means (k={best_kmeans['k']}, Sil={best_kmeans['silhouette']:.4f})")
axes[0, 0].set_xlabel("PCA 1"); axes[0, 0].set_ylabel("PCA 2")
plt.colorbar(sc1, ax=axes[0, 0])

sc2 = axes[0, 1].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],
    c=best_dbscan["labels"], cmap="tab10", s=5, alpha=0.5)
axes[0, 1].set_title(f"PCA — DBSCAN (eps={best_dbscan['eps']}, clusters={best_dbscan['n_clusters']}, Sil={best_dbscan['silhouette']:.4f})")
axes[0, 1].set_xlabel("PCA 1"); axes[0, 1].set_ylabel("PCA 2")
plt.colorbar(sc2, ax=axes[0, 1])

# Row 2: t-SNE
sc3 = axes[1, 0].scatter(X_tsne[:, 0], X_tsne[:, 1],
    c=best_kmeans["labels"], cmap="tab10", s=5, alpha=0.5)
axes[1, 0].set_title(f"t-SNE — K-Means (k={best_kmeans['k']})")
axes[1, 0].set_xlabel("t-SNE 1"); axes[1, 0].set_ylabel("t-SNE 2")
plt.colorbar(sc3, ax=axes[1, 0])

sc4 = axes[1, 1].scatter(X_tsne[:, 0], X_tsne[:, 1],
    c=best_dbscan["labels"], cmap="tab10", s=5, alpha=0.5)
axes[1, 1].set_title(f"t-SNE — DBSCAN (eps={best_dbscan['eps']}, clusters={best_dbscan['n_clusters']})")
axes[1, 1].set_xlabel("t-SNE 1"); axes[1, 1].set_ylabel("t-SNE 2")
plt.colorbar(sc4, ax=axes[1, 1])

plt.tight_layout()
plt.savefig("pca_vs_tsne.png", dpi=150)
plt.close()
print("Saved → pca_vs_tsne.png")
```

---

## Step 8 — Cluster Profiling + Business Recommendation

```python
df_profile = df_clean.copy()
df_profile["KMeans_Cluster"] = best_kmeans["labels"]

PROFILE_COLS = [
    "BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT",
    "PAYMENTS", "PRC_FULL_PAYMENT", "PURCHASES_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY", "PURCHASE_TO_LIMIT_RATIO",
    "CASH_ADVANCE_RATIO", "PAYMENT_TO_BALANCE_RATIO", "INSTALLMENT_RATIO"
]

profile = df_profile.groupby("KMeans_Cluster")[PROFILE_COLS].mean().round(3)
profile.to_csv("cluster_profile_final.csv")

print("\n===== CLUSTER PROFILE =====")
print(profile.to_string())

# Cluster sizes
print("\n--- Cluster Sizes ---")
sizes = df_profile["KMeans_Cluster"].value_counts().sort_index()
for cluster, count in sizes.items():
    print(f"  Cluster {cluster}: {count:5d} customers ({count/len(df_profile)*100:.1f}%)")

# Auto business label + recommendation
print("\n===== BUSINESS RECOMMENDATION =====")
BUSINESS_RULES = {
    "Transactor":        lambda r, m: r["PRC_FULL_PAYMENT"]      > m["PRC_FULL_PAYMENT"] * 1.2,
    "Revolver":          lambda r, m: r["BALANCE"]               > m["BALANCE"] * 1.2 and r["PRC_FULL_PAYMENT"] < m["PRC_FULL_PAYMENT"],
    "Cash Advance User": lambda r, m: r["CASH_ADVANCE_RATIO"]    > m["CASH_ADVANCE_RATIO"] * 1.2,
    "Big Spender":       lambda r, m: r["PURCHASES"]             > m["PURCHASES"] * 1.5,
    "Low Engager":       lambda r, m: r["PURCHASES_FREQUENCY"]   < m["PURCHASES_FREQUENCY"] * 0.8,
}

RECOMMENDATIONS = {
    "Transactor":        "Offer cashback rewards and premium cards — low risk, high value customers.",
    "Revolver":          "Offer balance transfer promotions and lower interest rate deals.",
    "Cash Advance User": "Flag for financial health check — high fee revenue but high risk of default.",
    "Big Spender":       "Offer travel rewards, lounge access, and high credit limit upgrades.",
    "Low Engager":       "Send activation campaigns and spending incentives to re-engage.",
}

means = profile.mean()
for cluster in profile.index:
    row    = profile.loc[cluster]
    labels_matched = [name for name, rule in BUSINESS_RULES.items() if rule(row, means)]
    label  = labels_matched[0] if labels_matched else "Average User"
    rec    = RECOMMENDATIONS.get(label, "Monitor and observe spending patterns.")
    size   = sizes[cluster]
    print(f"\n  Cluster {cluster} → [{label}] ({size} customers, {size/len(df_profile)*100:.1f}%)")
    print(f"  Recommendation: {rec}")
```

---

## Step 9 — 3-Round Comparison Table

```python
HISTORY = [
    {"Round": 1, "Model": "K-Means", "Clusters": 3,  "Silhouette": 0.2403, "Techniques": "Raw data"},
    {"Round": 1, "Model": "DBSCAN",  "Clusters": 5,  "Silhouette": 0.1409, "Techniques": "Raw data"},
    {"Round": 2, "Model": "K-Means", "Clusters": 2,  "Silhouette": 0.1894, "Techniques": "+ Outlier removal, Feature engineering"},
    {"Round": 2, "Model": "DBSCAN",  "Clusters": 12, "Silhouette": 0.2801, "Techniques": "+ Outlier removal, Feature engineering"},
    {"Round": 3, "Model": "K-Means", "Clusters": best_kmeans["k"],        "Silhouette": best_kmeans["silhouette"],               "Techniques": "+ Hopkins, t-SNE, Davies-Bouldin, Silhouette plot, Business rec"},
    {"Round": 3, "Model": "DBSCAN",  "Clusters": best_dbscan["n_clusters"], "Silhouette": round(best_dbscan["silhouette"], 4),   "Techniques": "+ Fine-tuned eps, Davies-Bouldin"},
]

df_history = pd.DataFrame(HISTORY)
df_history.to_csv("3round_comparison.csv", index=False)

print("\n" + "=" * 75)
print("              3-ROUND COMPARISON TABLE")
print("=" * 75)
print(f"{'Round':>6}  {'Model':>8}  {'Clusters':>9}  {'Silhouette':>12}  Techniques")
print("-" * 75)
for _, row in df_history.iterrows():
    print(f"{row['Round']:>6}  {row['Model']:>8}  {row['Clusters']:>9}  {row['Silhouette']:>12.4f}  {row['Techniques']}")
print("=" * 75)
print(f"\nHopkins Statistic : H = {H}  ({'✅ Clusterable' if H >= 0.75 else '⚠️ Moderate'})")
```

---

## Step 10 — Export + Token Usage

```python
# Save final results
df_out = pd.read_csv("CC_GENERAL.csv")
df_out["KMeans_Cluster_R3"] = -99
df_out["DBSCAN_Cluster_R3"] = -99
df_out.loc[df_clean.index, "KMeans_Cluster_R3"] = best_kmeans["labels"]
df_out.loc[df_clean.index, "DBSCAN_Cluster_R3"] = best_dbscan["labels"]
df_out.to_csv("clustering_results_final.csv", index=False)
print("Saved → clustering_results_final.csv")

print_token_usage()
```

---

## Expected Output Files

| File | Description |
|------|-------------|
| `elbow_silhouette_db.png` | Elbow + Silhouette + Davies-Bouldin (3 plots) |
| `silhouette_plot.png` | Per-cluster silhouette quality plot |
| `pca_vs_tsne.png` | 2×2 grid: PCA vs t-SNE × K-Means vs DBSCAN |
| `cluster_profile_final.csv` | Mean features per cluster |
| `3round_comparison.csv` | Full 3-round history table |
| `clustering_results_final.csv` | Original data + Round 3 cluster labels |

## Expected Console Output
1. Preprocessing summary (rows removed, features)
2. Hopkins Statistic + interpretation
3. K-Means table: Inertia + Silhouette + Davies-Bouldin for k=2..10
4. DBSCAN fine-tuning table: all eps values with both metrics
5. Business recommendation per cluster
6. 3-round comparison table
7. Token usage

---

## Constraints
- Do NOT recompute Round 1 or Round 2 numbers — they are hardcoded in Step 9
- DBSCAN best eps must come from runs with 2–8 clusters (meaningful range)
- t-SNE uses perplexity=30, n_iter=1000, random_state=42
- Hopkins uses sample_size=150, random_state=42
- Removed outlier rows get label -99 in final CSV
- Use random_state=42 and n_init=10 for K-Means throughout
