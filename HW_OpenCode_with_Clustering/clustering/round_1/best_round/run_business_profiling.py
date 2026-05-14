import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# Token counter
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

# === Step 1: Load Data ===
df = pd.read_csv("clustering_results.csv")

print(f"Total customers : {len(df):,}")
print(f"Columns         : {list(df.columns)}")
print()
print("Cluster distribution:")
sizes = df["KMeans_Cluster"].value_counts().sort_index()
for c, n in sizes.items():
    print(f"  Cluster {c}: {n:,} customers ({n/len(df)*100:.1f}%)")

# === Step 2: Compute Full Cluster Profile ===
PROFILE_COLS = [
    "BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS", "PRC_FULL_PAYMENT",
    "PURCHASES_FREQUENCY", "CASH_ADVANCE_FREQUENCY", "MINIMUM_PAYMENTS", "TENURE"
]

profile     = df.groupby("KMeans_Cluster")[PROFILE_COLS].mean().round(2)
profile_std = df.groupby("KMeans_Cluster")[PROFILE_COLS].std().round(2)

print("\n===== CLUSTER PROFILE (MEAN VALUES) =====")
print(profile.to_string())

profile.to_csv("cluster_profile_business.csv")
print("\nSaved -> cluster_profile_business.csv")

# === Step 3: Assign Business Labels ===
BUSINESS_LABELS = {
    0: "Active Spender",
    1: "Moderate User",
    2: "Cash Advance Reliant"
}

BUSINESS_DESCRIPTIONS = {
    0: (
        "Customers who purchase frequently and in large amounts. "
        "High credit limit with moderate full-payment rate. "
        "Low cash advance usage -- prefer to spend on goods/services."
    ),
    1: (
        "The majority segment. Moderate balance, purchases, and payments. "
        "No extreme behavior in any dimension. "
        "Reliable but not highly engaged customers."
    ),
    2: (
        "Customers heavily dependent on cash advances. "
        "Carry the highest balance and rarely pay in full (3%). "
        "High financial risk -- potential default candidates."
    )
}

MARKETING_STRATEGIES = {
    0: [
        "Offer premium rewards cards with cashback on purchases (travel, dining, shopping).",
        "Provide credit limit upgrades as loyalty incentives.",
        "Target with exclusive membership programs and concierge services.",
        "Cross-sell travel insurance and purchase protection plans.",
    ],
    1: [
        "Send engagement campaigns: spend X get Y bonus points.",
        "Offer installment payment plans to increase purchase volume.",
        "Educate on rewards programs -- many may not be using card benefits.",
        "Target with seasonal promotions and limited-time cashback offers.",
    ],
    2: [
        "Flag for financial risk monitoring -- high probability of default.",
        "Offer balance transfer deals with lower interest rates.",
        "Provide financial wellness programs and debt consolidation options.",
        "Reduce cash advance limits gradually to lower bank exposure.",
        "Send early payment reminders and minimum payment alerts.",
    ]
}

df["Business_Label"] = df["KMeans_Cluster"].map(BUSINESS_LABELS)

print("\n===== BUSINESS LABELS =====")
for c, label in BUSINESS_LABELS.items():
    size = sizes[c]
    print(f"\n  Cluster {c} -> [{label}] ({size:,} customers, {size/len(df)*100:.1f}%)")
    print(f"  Profile : {BUSINESS_DESCRIPTIONS[c]}")
    print(f"  Strategy:")
    for s in MARKETING_STRATEGIES[c]:
        print(f"    * {s}")

# === Step 4: Bar Chart Comparison ===
COMPARE_FEATURES = {
    "Balance ($)"            : "BALANCE",
    "Purchases ($)"          : "PURCHASES",
    "Cash Advance ($)"       : "CASH_ADVANCE",
    "Credit Limit ($)"       : "CREDIT_LIMIT",
    "Payments ($)"           : "PAYMENTS",
    "Full Payment Rate"      : "PRC_FULL_PAYMENT",
    "Purchase Frequency"     : "PURCHASES_FREQUENCY",
    "Cash Advance Frequency" : "CASH_ADVANCE_FREQUENCY",
}

COLORS = ["#378ADD", "#1D9E75", "#D85A30"]
LABELS = [f"Cluster {c}: {BUSINESS_LABELS[c]}" for c in [0, 1, 2]]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for idx, (feat_label, col) in enumerate(COMPARE_FEATURES.items()):
    ax  = axes[idx]
    vals = [profile.loc[c, col] for c in [0, 1, 2]]
    bars = ax.bar(LABELS, vals, color=COLORS, alpha=0.85, edgecolor="white")

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(vals)*0.01,
                f"{val:,.2f}" if val < 1 else f"{val:,.0f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title(feat_label, fontsize=11, fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"C{c}" for c in [0, 1, 2]], fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

patches = [mpatches.Patch(color=COLORS[i], label=LABELS[i]) for i in range(3)]
fig.legend(handles=patches, loc="lower center", ncol=3,
           fontsize=10, bbox_to_anchor=(0.5, -0.02))

plt.suptitle("Credit Card Customer Segments -- Feature Comparison",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("cluster_bar_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved -> cluster_bar_comparison.png")

# === Step 5: Radar Chart ===
RADAR_COLS = [
    "BALANCE", "PURCHASES", "CASH_ADVANCE",
    "PAYMENTS", "PRC_FULL_PAYMENT", "PURCHASES_FREQUENCY", "CASH_ADVANCE_FREQUENCY"
]
RADAR_LABELS = [
    "Balance", "Purchases", "Cash\nAdvance",
    "Payments", "Full\nPayment", "Purchase\nFreq", "Cash Adv\nFreq"
]

profile_norm = profile[RADAR_COLS].copy()
for col in RADAR_COLS:
    mn, mx = profile_norm[col].min(), profile_norm[col].max()
    profile_norm[col] = (profile_norm[col] - mn) / (mx - mn + 1e-9)

N      = len(RADAR_COLS)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for c, color in zip([0, 1, 2], COLORS):
    vals = profile_norm.loc[c, RADAR_COLS].tolist()
    vals += vals[:1]
    ax.plot(angles, vals, color=color, linewidth=2, label=BUSINESS_LABELS[c])
    ax.fill(angles, vals, color=color, alpha=0.15)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(RADAR_LABELS, fontsize=11)
ax.set_yticklabels([])
ax.set_title("Customer Segment Radar Chart\n(normalized features)",
             fontsize=13, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("cluster_radar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> cluster_radar.png")

# === Step 6: Export Final Business Report ===
df_out = df.copy()
df_out["Business_Label"] = df_out["KMeans_Cluster"].map(BUSINESS_LABELS)
df_out.to_csv("clustering_business_final.csv", index=False)
print("Saved -> clustering_business_final.csv")

rows = []
for c in [0, 1, 2]:
    size = sizes[c]
    rows.append({
        "Cluster"              : c,
        "Business_Label"       : BUSINESS_LABELS[c],
        "N_Customers"          : size,
        "Pct_Customers"        : round(size / len(df) * 100, 1),
        "Avg_Balance"          : profile.loc[c, "BALANCE"],
        "Avg_Purchases"        : profile.loc[c, "PURCHASES"],
        "Avg_Cash_Advance"     : profile.loc[c, "CASH_ADVANCE"],
        "Avg_Credit_Limit"     : profile.loc[c, "CREDIT_LIMIT"],
        "Avg_Full_Payment_Rate": profile.loc[c, "PRC_FULL_PAYMENT"],
        "Description"          : BUSINESS_DESCRIPTIONS[c],
        "Top_Strategy"         : MARKETING_STRATEGIES[c][0],
    })

pd.DataFrame(rows).to_csv("business_summary.csv", index=False)
print("Saved -> business_summary.csv")

# === Step 7: Final Report + Token Usage ===
print("\n" + "=" * 65)
print("        FINAL BUSINESS PROFILING REPORT")
print("=" * 65)
print(f"  Model     : K-Means (Round 1)")
print(f"  k         : 3 clusters")
print(f"  Silhouette: 0.2403")
print(f"  Coverage  : 100% of customers (8,950 rows)")
print()

line_sep = "=" * 55
for c in [0, 1, 2]:
    size = sizes[c]
    print(f"  {line_sep}")
    print(f"  Cluster {c} -> {BUSINESS_LABELS[c].upper()}")
    print(f"  Size      : {size:,} customers ({size/len(df)*100:.1f}%)")
    print(f"  Profile   : {BUSINESS_DESCRIPTIONS[c]}")
    print(f"  Strategy  :")
    for s in MARKETING_STRATEGIES[c]:
        print(f"    * {s}")

print(f"\n  {line_sep}")
print(f"\n  Output Files:")
print(f"    cluster_bar_comparison.png   -- 8-feature bar chart comparison")
print(f"    cluster_radar.png            -- radar chart (normalized)")
print(f"    cluster_profile_business.csv -- mean values per cluster")
print(f"    business_summary.csv         -- labels + strategies per cluster")
print(f"    clustering_business_final.csv -- full data + Business_Label column")

print_token_usage()
