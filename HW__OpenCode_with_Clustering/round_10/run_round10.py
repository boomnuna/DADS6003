# Round 10: Cap instead of Drop — full pipeline
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Token Usage Tracker ──
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

# ═══════════════════════════════════════════
# Step 1 — Load & Preprocess with CAPPING
# ═══════════════════════════════════════════
df = pd.read_csv("CC_GENERAL.csv")

drop_cols = ["CUST_ID", "KMeans_Cluster", "DBSCAN_Cluster",
             "KMeans_Cluster_Enhanced", "DBSCAN_Cluster_Enhanced",
             "KMeans_Cluster_R3", "DBSCAN_Cluster_R3",
             "KMeans_Cluster_R4", "DBSCAN_Cluster_R4",
             "KMeans_Cluster_R5", "DBSCAN_Cluster_R5",
             "KMeans_Cluster_R6", "DBSCAN_Cluster_R6", "HDBSCAN_Cluster_R6",
             "DBSCAN_Cluster_R7", "DBSCAN_Cluster_R8", "DBSCAN_Cluster_R9"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

df["CREDIT_LIMIT"]     = df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].median())
df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(df["MINIMUM_PAYMENTS"].median())

OUTLIER_COLS = [
    "BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "CASH_ADVANCE",
    "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS"
]

df_capped = df.copy()
cap_stats = []

for col in OUTLIER_COLS:
    Q1  = df_capped[col].quantile(0.25)
    Q3  = df_capped[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    n_capped_low  = (df_capped[col] < lower).sum()
    n_capped_high = (df_capped[col] > upper).sum()

    df_capped[col] = df_capped[col].clip(lower=lower, upper=upper)
    cap_stats.append({
        "feature": col,
        "capped_low": n_capped_low,
        "capped_high": n_capped_high,
        "total_capped": n_capped_low + n_capped_high
    })

print("✅ CAPPING applied (no rows removed)")
print(f"All rows kept  : {len(df_capped)} (100%)")
print()
print("Capping stats per feature:")
for s in cap_stats:
    print(f"  {s['feature']:30s}: {s['total_capped']:4d} values capped "
          f"(low={s['capped_low']}, high={s['capped_high']})")

# ═══════════════════════════════════════════
# Step 2 — Log Transform + Feature Engineering
# ═══════════════════════════════════════════
SKEWED_COLS = [
    "BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE", "PAYMENTS", "MINIMUM_PAYMENTS", "CREDIT_LIMIT"
]
df_log = df_capped.copy()
for col in SKEWED_COLS:
    df_log[col] = np.log1p(df_log[col])

df_log["PURCHASE_TO_LIMIT_RATIO"]  = df_log["PURCHASES"]              / (df_log["CREDIT_LIMIT"] + 1)
df_log["CASH_ADVANCE_RATIO"]       = df_log["CASH_ADVANCE"]           / (df_log["CREDIT_LIMIT"] + 1)
df_log["PAYMENT_TO_BALANCE_RATIO"] = df_log["PAYMENTS"]               / (df_log["BALANCE"] + 1)
df_log["INSTALLMENT_RATIO"]        = df_log["INSTALLMENTS_PURCHASES"] / (df_log["PURCHASES"] + 1)

DROP_CORRELATED = ["ONEOFF_PURCHASES", "PURCHASES_INSTALLMENTS_FREQUENCY", "CASH_ADVANCE_TRX"]
df_final = df_log.drop(columns=DROP_CORRELATED)

print(f"\nFeatures: {df_final.shape[1]}")

# ═══════════════════════════════════════════
# Step 3 — Scale
# ═══════════════════════════════════════════
from sklearn.preprocessing import RobustScaler

scaler   = RobustScaler()
X_scaled = scaler.fit_transform(df_final)
print(f"Scaled shape: {X_scaled.shape}")

# ═══════════════════════════════════════════
# Step 4 — UMAP + DBSCAN (fixed best params)
# ═══════════════════════════════════════════
import umap
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

N_NEIGHBORS  = 50
MIN_DIST     = 0.1
METRIC       = "euclidean"
N_COMPONENTS = 7
MIN_SAMPLES  = 3
EPS          = 2.0

print(f"\nFitting UMAP (n_components={N_COMPONENTS}, n_neighbors={N_NEIGHBORS}, "
      f"min_dist={MIN_DIST}, metric={METRIC})...")
umap_model = umap.UMAP(
    n_components=N_COMPONENTS,
    n_neighbors=N_NEIGHBORS,
    min_dist=MIN_DIST,
    metric=METRIC,
    random_state=42
)
X_umap = umap_model.fit_transform(X_scaled)

print(f"Running DBSCAN (eps={EPS}, min_samples={MIN_SAMPLES})...")
db     = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
labels = db.fit_predict(X_umap)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = (labels == -1).sum()
noise_pct  = n_noise / len(labels) * 100

mask   = labels != -1
sil    = silhouette_score(X_umap[mask], labels[mask],
                          sample_size=2000, random_state=42)
db_sco = davies_bouldin_score(X_umap[mask], labels[mask])

print(f"\n===== ROUND 10 RESULTS (CAPPING) =====")
print(f"  Rows used      : {len(df_final)} (100% — no data lost)")
print(f"  Clusters       : {n_clusters}")
print(f"  Noise points   : {n_noise} ({noise_pct:.1f}%)")
print(f"  Silhouette     : {sil:.4f}")
print(f"  Davies-Bouldin : {db_sco:.4f}")

# ═══════════════════════════════════════════
# Step 5 — Honest Comparison: Drop vs Cap
# ═══════════════════════════════════════════
DROP_RESULT = {
    "approach": "Drop (IQR×1.5)",
    "rows_used": 6157,
    "coverage": "69%",
    "silhouette": 0.7483,
    "note": "⚠️  31% of customers excluded from analysis"
}

CAP_RESULT = {
    "approach": "Cap (IQR×1.5)",
    "rows_used": len(df_final),
    "coverage": "100%",
    "silhouette": round(sil, 4),
    "note": "✅ All customers included"
}

print("\n" + "=" * 65)
print("         DROP vs CAP — HONEST COMPARISON")
print("=" * 65)
print(f"{'':25s}  {'DROP':>10}  {'CAP':>10}")
print("-" * 50)
print(f"{'Rows used':25s}  {'6,157':>10}  {len(df_final):>10,}")
print(f"{'Coverage':25s}  {'69%':>10}  {'100%':>10}")
print(f"{'Silhouette':25s}  {0.7483:>10.4f}  {sil:>10.4f}")
print(f"{'Davies-Bouldin':25s}  {'N/A':>10}  {db_sco:>10.4f}")
print(f"{'Clusters':25s}  {'3':>10}  {n_clusters:>10}")
print(f"{'Trustworthy?':25s}  {'⚠️  Partial':>10}  {'✅ Full':>10}")
print("=" * 65)

delta = (sil - 0.7483) / 0.7483 * 100
print(f"\nSilhouette change: {0.7483:.4f} → {sil:.4f} ({delta:+.1f}%)")
if sil >= 0.7483:
    print("🎉 Capping is BETTER than dropping — more data AND better score!")
elif sil >= 0.65:
    print("✅ Capping is slightly lower but FAR more trustworthy (100% coverage)")
else:
    print(f"⚠️  Score dropped {0.7483-sil:.4f} but coverage improved from 69% → 100%")
    print("   This is the honest trade-off between score and data integrity")

# ═══════════════════════════════════════════
# Step 6 — Cluster Profile (original scale)
# ═══════════════════════════════════════════
df_profile = df_capped.copy()
df_profile["DBSCAN_Cluster"] = labels

PROFILE_COLS = [
    "BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT",
    "PAYMENTS", "PRC_FULL_PAYMENT", "PURCHASES_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY", "MINIMUM_PAYMENTS"
]

valid    = df_profile[df_profile["DBSCAN_Cluster"] >= 0]
profile  = valid.groupby("DBSCAN_Cluster")[PROFILE_COLS].mean().round(2)
sizes    = valid["DBSCAN_Cluster"].value_counts().sort_index()

print("\n===== CLUSTER PROFILE (original scale, all 8,950 customers) =====")
print(profile.to_string())
print("\n--- Cluster Sizes ---")
for c, n in sizes.items():
    print(f"  Cluster {c}: {n:5,} customers ({n/len(df_final)*100:.1f}%)")
if n_noise > 0:
    print(f"  Noise (-1): {n_noise:5,} customers ({noise_pct:.1f}%)")

# ═══════════════════════════════════════════
# Step 7 — Visualization
# ═══════════════════════════════════════════
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

umap_2d   = umap.UMAP(n_components=2, n_neighbors=N_NEIGHBORS,
                       min_dist=MIN_DIST, metric=METRIC, random_state=42)
X_umap_2d = umap_2d.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sc1 = axes[0].scatter(X_umap_2d[:, 0], X_umap_2d[:, 1],
    c=labels, cmap="tab10", s=3, alpha=0.5)
axes[0].set_title(
    f"Round 10 — CAP approach\n"
    f"All 8,950 customers | Silhouette={sil:.4f}\n"
    f"clusters={n_clusters}, noise={noise_pct:.1f}%",
    fontsize=9
)
axes[0].set_xlabel("UMAP 1"); axes[0].set_ylabel("UMAP 2")
plt.colorbar(sc1, ax=axes[0])

methods   = ["Drop\n(Round 9)", "Cap\n(Round 10)"]
scores    = [0.7483, sil]
coverages = [69, 100]
colors    = ["#D85A30", "#1D9E75"]

ax2 = axes[1]
bars = ax2.bar(methods, scores, color=colors, alpha=0.8, width=0.4)
ax2.set_ylim(0, 0.85)
ax2.set_ylabel("Silhouette Score")
ax2.set_title("Drop vs Cap — Score & Coverage", fontsize=10)
ax2.grid(axis="y", alpha=0.3)

for bar, score, cov in zip(bars, scores, coverages):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.02,
             f"Sil={score:.4f}\nCov={cov}%",
             ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("r10_cap_vs_drop.png", dpi=150)
plt.close()
print("\nSaved → r10_cap_vs_drop.png")

# ═══════════════════════════════════════════
# Step 8 — Export + Token Usage
# ═══════════════════════════════════════════
df_out = pd.read_csv("CC_GENERAL.csv")
df_out["DBSCAN_Cluster_R10_Cap"] = labels
df_out.to_csv("clustering_results_r10.csv", index=False)
print("Saved → clustering_results_r10.csv  (no -99 values — all rows kept)")

print_token_usage()
