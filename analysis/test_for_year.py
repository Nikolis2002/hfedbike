import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
from scipy.stats import wilcoxon

# ─── 1) Load each node's weekly results CSV ──────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
csv_files = sorted(glob.glob(str(REPO_ROOT / "results/current/*_results.csv")))
node_data = {}

for path in csv_files:
    node = path.split("/")[-1].split("_results")[0]
    df = pd.read_csv(path)

    if "week" not in df.columns:
        print(f"Skipping {path}: no 'week' column")
        continue

    # ensure numeric week + stable unique per week
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df = df.dropna(subset=["week"]).copy()
    df["week"] = df["week"].astype(int)

    df = (
        df.sort_values("week")
        .drop_duplicates(subset=["week"], keep="last")
        .reset_index(drop=True)
    )

    node_data[node] = df

if not node_data:
    raise SystemExit("No valid node CSVs loaded.")

# ─── 2) Master week list ─────────────────────────────────────
all_weeks = sorted({w for df in node_data.values() for w in df["week"].unique()})

# ─── 3) Per-week average MAE ± 95% CI ────────────────────────
avg_base_mae, ci_base = [], []
avg_fed_mae, ci_fed = [], []
avg_freeze_mae, ci_freeze = [], []

for wk in all_weeks:
    base_vals, fed_vals = [], []
    freeze_vals = []

    for df in node_data.values():
        row = df[df["week"] == wk]
        if not row.empty:
            base_vals.append(float(row["base_mae"].iloc[0]))
            fed_vals.append(float(row["fed_mae"].iloc[0]))
            freeze_vals.append(float(row["freeze_mae"].iloc[0]))

    # Baseline
    if base_vals:
        n = len(base_vals)
        mean = np.mean(base_vals)
        std = np.std(base_vals, ddof=1) if n > 1 else 0.0
        ci = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
        avg_base_mae.append(mean)
        ci_base.append(ci)
    else:
        avg_base_mae.append(np.nan)
        ci_base.append(0.0)

    # Federated
    if fed_vals:
        n = len(fed_vals)
        mean = np.mean(fed_vals)
        std = np.std(fed_vals, ddof=1) if n > 1 else 0.0
        ci = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
        avg_fed_mae.append(mean)
        ci_fed.append(ci)
    else:
        avg_fed_mae.append(np.nan)
        ci_fed.append(0.0)

    if freeze_vals:
        n = len(freeze_vals)
        mean = np.mean(freeze_vals)
        std = np.std(freeze_vals, ddof=1) if n > 1 else 0.0
        ci = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
        avg_freeze_mae.append(mean)
        ci_freeze.append(ci)
    else:
        avg_freeze_mae.append(np.nan)
        ci_freeze.append(0.0)


plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
    }
)

# Wider figure for better horizontal balance
fig, ax = plt.subplots(figsize=(8.4, 4.6))

# --- Convert to numpy ---
avg_freeze_mae = np.array(avg_freeze_mae)
avg_base_mae = np.array(avg_base_mae)
avg_fed_mae = np.array(avg_fed_mae)

ci_freeze = np.array(ci_freeze)
ci_base = np.array(ci_base)
ci_fed = np.array(ci_fed)

all_weeks = np.array(all_weeks)


def solid_marker(color, size):
    return dict(
        markersize=size,
        markerfacecolor=color,
        markeredgecolor="none",  # remove hollow effect
        fillstyle="full",
    )


# --- Baseline ---
ax.plot(
    all_weeks,
    avg_freeze_mae,
    linestyle="--",
    marker="o",
    linewidth=1.8,
    label="Baseline (Frozen)",
    **solid_marker("tab:blue", 5.2),
)
ax.fill_between(
    all_weeks, avg_freeze_mae - ci_freeze, avg_freeze_mae + ci_freeze, alpha=0.15
)

# --- Bias Addition ---
ax.plot(
    all_weeks,
    avg_base_mae,
    linestyle="--",
    marker="s",
    linewidth=1.8,
    label="Bias Addition",
    **solid_marker("tab:orange", 5.2),
)
ax.fill_between(all_weeks, avg_base_mae - ci_base, avg_base_mae + ci_base, alpha=0.15)

# --- Federated (dominant line) ---
ax.plot(
    all_weeks,
    avg_fed_mae,
    linestyle="-",
    marker="o",
    linewidth=2.4,
    label="Federated",
    **solid_marker("tab:green", 5.6),
)
ax.fill_between(all_weeks, avg_fed_mae - ci_fed, avg_fed_mae + ci_fed, alpha=0.18)

# Labels
ax.set_xlabel("Week")
ax.set_ylabel("MAE")

# X ticks
ax.set_xticks(range(0, 55, 4))

# Slight right padding
ax.set_xlim(all_weeks.min(), all_weeks.max() + 3)

# --- Controlled vertical padding ---
y_min = min(
    (avg_freeze_mae - ci_freeze).min(),
    (avg_base_mae - ci_base).min(),
    (avg_fed_mae - ci_fed).min(),
)
y_max = max(
    (avg_freeze_mae + ci_freeze).max(),
    (avg_base_mae + ci_base).max(),
    (avg_fed_mae + ci_fed).max(),
)

ax.set_ylim(y_min - 0.05, y_max + 0.05)

# Grid
ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.3)

# Compact legend above plot (no wasted right space)
ax.legend(
    loc="lower left",
    bbox_to_anchor=(0, 1.02),
    ncol=3,
    frameon=False,
    borderaxespad=0.0,
    columnspacing=1.2,
    handletextpad=0.6,
)

fig.tight_layout()

# Save
fig.savefig(REPO_ROOT / "figures/mae_comparison.pdf")
fig.savefig(REPO_ROOT / "figures/mae_comparison.png", dpi=450)

plt.show()
# ─── 5) Stats summary + Wilcoxon ─────────────────────────────
clean_idx = [
    i
    for i in range(len(all_weeks))
    if (
        not np.isnan(avg_freeze_mae[i])
        and not np.isnan(avg_base_mae[i])
        and not np.isnan(avg_fed_mae[i])
    )
]

freeze_b = [avg_freeze_mae[i] for i in clean_idx]
bias_b = [avg_base_mae[i] for i in clean_idx]
fed_b = [avg_fed_mae[i] for i in clean_idx]
n_weeks = len(clean_idx)


def mean_ci(vals):
    vals = np.array(vals, dtype=float)
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    ci = 1.96 * std / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return mean, ci


m_freeze, ci_freeze_all = mean_ci(freeze_b)
m_bias, ci_bias_all = mean_ci(bias_b)
m_fed, ci_fed_all = mean_ci(fed_b)

# Paired Wilcoxon tests
p_fed_vs_bias = wilcoxon(bias_b, fed_b).pvalue if n_weeks > 0 else np.nan
p_bias_vs_free = wilcoxon(freeze_b, bias_b).pvalue if n_weeks > 0 else np.nan
p_fed_vs_free = wilcoxon(freeze_b, fed_b).pvalue if n_weeks > 0 else np.nan

print(f"\nUsed {n_weeks} common weeks across all models.\n")

print(
    f"Freeze baseline: {m_freeze:.3f} (95% CI [{m_freeze - ci_freeze_all:.3f}, {m_freeze + ci_freeze_all:.3f}])\n"
    f"Bias Addition:   {m_bias:.3f} (95% CI [{m_bias - ci_bias_all:.3f}, {m_bias + ci_bias_all:.3f}])\n"
    f"Federated:       {m_fed:.3f} (95% CI [{m_fed - ci_fed_all:.3f}, {m_fed + ci_fed_all:.3f}])\n"
)


def sig(p):
    if np.isnan(p):
        return "n/a"
    return "significant" if p <= 0.05 else "not significant"


print(
    f"Wilcoxon p-values:\n"
    f"  Federated vs Bias Addition:   p = {p_fed_vs_bias:.4f} ({sig(p_fed_vs_bias)})\n"
    f"  Bias Addition vs Freeze:      p = {p_bias_vs_free:.4f} ({sig(p_bias_vs_free)})\n"
    f"  Federated vs Freeze:          p = {p_fed_vs_free:.4f} ({sig(p_fed_vs_free)})\n"
)
