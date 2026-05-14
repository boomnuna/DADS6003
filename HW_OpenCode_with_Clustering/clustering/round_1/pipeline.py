import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ---- Step 0: Token counter ----
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

# ---- Step 1: Load and Preprocess Data ----
print("=" * 60)
print("STEP 1: LOAD AND PREPROCESS DATA")
print("=" * 60)

df = pd.read_csv("CC_GENERAL.csv")
df = df.drop(columns=["CUST_ID"])
df["CREDIT_LIMIT"]    = df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].median())
df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(df["MINIMUM_PAYMENTS"].median())

print(f"Dataset shape    : {df.shape}")
print(f"Missing values   : {df.isnull().sum().sum()}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
print(f"Scaling complete : {X_scaled.shape}")

# ---- Step 2: Dimensionality Reduction with PCA ----
print("\n" + "=" * 60)
print("STEP 2: DIMENSIONALITY REDUCTION WITH PCA")
print("=" * 60)

pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_scaled)
print(f"PCA 2D variance explained: {pca_2d.explained_variance_ratio_.sum():.2%}")

pca_full = PCA(n_components=0.95, random_state=42)
X_pca_full = pca_full.fit_transform(X_scaled)
print(f"PCA full components kept : {X_pca_full.shape[1]} (95% variance)")

# ---- Step 3: K-Means: Find Optimal Number of Clusters ----
print("\n" + "=" * 60)
print("STEP 3: K-MEANS — FIND OPTIMAL K")
print("=" * 60)

kmeans_results = []
K_RANGE = range(2, 11)

print(f"\n{'k':>3}  {'Inertia':>14}  {'Silhouette':>12}")
print("-" * 32)

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
    print(f"{k:3d}  {inertia:>14.2f}  {sil:>12.4f}")

best_kmeans = max(kmeans_results, key=lambda r: r["silhouette"])
print(f"\n>> Best K-Means k = {best_kmeans['k']}  (Silhouette = {best_kmeans['silhouette']})")

# ---- Step 4: DBSCAN: Find Optimal eps ----
print("\n" + "=" * 60)
print("STEP 4: DBSCAN — FIND OPTIMAL EPS")
print("=" * 60)

MIN_SAMPLES = 5

nbrs = NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(X_pca_full)
distances, _ = nbrs.kneighbors(X_pca_full)
k_distances   = np.sort(distances[:, -1])

plt.figure(figsize=(8, 4))
plt.plot(k_distances)
plt.title(f"k-Distance Graph (k={MIN_SAMPLES}) — look for the elbow")
plt.xlabel("Points sorted by distance")
plt.ylabel(f"{MIN_SAMPLES}-NN distance")
plt.grid(True)
plt.tight_layout()
plt.savefig("dbscan_kdistance.png", dpi=150)
plt.close()
print("Saved >> dbscan_kdistance.png")

EPS_VALUES = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
dbscan_results = []

print(f"\n{'eps':>5}  {'Clusters':>9}  {'Noise':>6}  {'Noise%':>7}  {'Silhouette':>12}")
print("-" * 45)

for eps in EPS_VALUES:
    db = DBSCAN(eps=eps, min_samples=MIN_SAMPLES)
    labels = db.fit_predict(X_pca_full)

    n_clusters  = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise     = (labels == -1).sum()
    noise_pct   = n_noise / len(labels) * 100

    if n_clusters >= 2:
        mask = labels != -1
        sil  = silhouette_score(X_pca_full[mask], labels[mask],
                                sample_size=2000, random_state=42)
        sil_str = f"{sil:.4f}"
    else:
        sil     = -1
        sil_str = "N/A"

    dbscan_results.append({
        "eps":        eps,
        "n_clusters": n_clusters,
        "n_noise":    n_noise,
        "noise_pct":  round(noise_pct, 2),
        "silhouette": sil,
        "labels":     labels,
    })
    print(f"{eps:5.1f}  {n_clusters:>9d}  {n_noise:>6d}  {noise_pct:>6.1f}%  {sil_str:>12}")

valid_dbscan = [r for r in dbscan_results if r["n_clusters"] >= 2]
best_dbscan  = max(valid_dbscan, key=lambda r: r["silhouette"])
print(f"\n>> Best DBSCAN eps = {best_dbscan['eps']}  "
      f"clusters = {best_dbscan['n_clusters']}  "
      f"(Silhouette = {best_dbscan['silhouette']:.4f})")

# ---- Step 5: Compare K-Means vs DBSCAN ----
print("\n" + "=" * 60)
print("STEP 5: MODEL COMPARISON")
print("=" * 60)

print(f"\n{'Model':<20} {'Clusters':>10} {'Silhouette':>12}")
print("-" * 45)
print(f"{'K-Means':.<20} {best_kmeans['k']:>10} {best_kmeans['silhouette']:>12.4f}")
print(f"{'DBSCAN':.<20} {best_dbscan['n_clusters']:>10} {best_dbscan['silhouette']:>12.4f}")

print("\n--- K-Means (all k) ---")
for r in kmeans_results:
    marker = " << best" if r["k"] == best_kmeans["k"] else ""
    print(f"  k={r['k']:2d}  Silhouette={r['silhouette']:.4f}{marker}")

print("\n--- DBSCAN (all eps) ---")
for r in dbscan_results:
    marker = " << best" if r["eps"] == best_dbscan["eps"] else ""
    sil_str = f"{r['silhouette']:.4f}" if r["silhouette"] != -1 else "N/A"
    print(f"  eps={r['eps']:.1f}  clusters={r['n_clusters']:2d}  noise={r['noise_pct']:.1f}%  Silhouette={sil_str}{marker}")

# ---- Step 6: Visualize Clusters ----
print("\n" + "=" * 60)
print("STEP 6: VISUALIZE CLUSTERS")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

scatter1 = axes[0].scatter(
    X_pca_2d[:, 0], X_pca_2d[:, 1],
    c=best_kmeans["labels"], cmap="tab10", s=5, alpha=0.5
)
axes[0].set_title(f"K-Means (k={best_kmeans['k']}, Silhouette={best_kmeans['silhouette']:.4f})")
axes[0].set_xlabel("PCA Component 1")
axes[0].set_ylabel("PCA Component 2")
plt.colorbar(scatter1, ax=axes[0])

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
print("Saved >> cluster_visualization.png")

# ---- Step 7: Export Results ----
print("\n" + "=" * 60)
print("STEP 7: EXPORT RESULTS")
print("=" * 60)

df_out = pd.read_csv("CC_GENERAL.csv")
df_out["KMeans_Cluster"] = best_kmeans["labels"]
df_out["DBSCAN_Cluster"]  = best_dbscan["labels"]
df_out.to_csv("clustering_results.csv", index=False)
print("Saved >> clustering_results.csv")

summary = pd.DataFrame([
    {
        "Model":      "K-Means",
        "N_Clusters": best_kmeans["k"],
        "Silhouette": best_kmeans["silhouette"],
        "Notes":      "Best of k=2..10",
    },
    {
        "Model":      "DBSCAN",
        "N_Clusters": best_dbscan["n_clusters"],
        "Silhouette": round(best_dbscan["silhouette"], 4),
        "Notes":      f"eps={best_dbscan['eps']}, noise={best_dbscan['noise_pct']}%, min_samples={MIN_SAMPLES}",
    },
])
summary.to_csv("summary.csv", index=False)
print("Saved >> summary.csv")

# ---- Step 8: Final Report + Token Usage ----
print("\n" + "=" * 50)
print("       FINAL REPORT")
print("=" * 50)

print(f"\nK-Means")
print(f"   Optimal clusters : {best_kmeans['k']}")
print(f"   Silhouette score : {best_kmeans['silhouette']}")

print(f"\nDBSCAN")
print(f"   Optimal clusters : {best_dbscan['n_clusters']}")
print(f"   Silhouette score : {best_dbscan['silhouette']:.4f}")
print(f"   Noise points     : {best_dbscan['n_noise']} ({best_dbscan['noise_pct']}%)")
print(f"   Best eps         : {best_dbscan['eps']}")

print(f"\nOutput Files")
print(f"   clustering_results.csv")
print(f"   summary.csv")
print(f"   cluster_visualization.png")
print(f"   dbscan_kdistance.png")

print_token_usage()
