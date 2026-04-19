import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
from scipy.stats import wilcoxon

# ─── 1) Load each node's weekly results CSV ──────────────────
csv_files = glob.glob("../results/current/*_results.csv")
node_data = {}
for path in csv_files:
    node = path.split('/')[-1].split('_results')[0]
    df = pd.read_csv(path)
    if 'Date' not in df.columns:
        print(f"Skipping {path}: no 'Date' column")
        continue
    df = (
        df.sort_values('Date')
          .drop_duplicates(subset=['Date'])
          .reset_index(drop=True)
    )
    node_data[node] = df

# ─── 2) Build a master list of all weeks ─────────────────────
all_weeks = sorted({ w for df in node_data.values() for w in df['Date'] })

# ─── 3) Compute per‐week average MAE ± 95% CI ────────────────
avg_base_mae, ci_base = [], []
avg_fed_mae,  ci_fed  = [], []

for week in all_weeks:
    base_vals = []
    fed_vals  = []
    for df in node_data.values():
        row = df[df['Date'] == week]
        if not row.empty:
            base_vals.append(row['base_mae'].iloc[0])
            fed_vals.append(row['fed_mae'].iloc[0])

    # Baseline
    if base_vals:
        n    = len(base_vals)
        mean = np.mean(base_vals)
        std  = np.std(base_vals, ddof=1)
        ci   = 1.96 * std / np.sqrt(n)
        avg_base_mae.append(mean)
        ci_base.append(ci)
    else:
        avg_base_mae.append(np.nan)
        ci_base.append(0.0)

    # Federated
    if fed_vals:
        n    = len(fed_vals)
        mean = np.mean(fed_vals)
        std  = np.std(fed_vals, ddof=1)
        ci   = 1.96 * std / np.sqrt(n)
        avg_fed_mae.append(mean)
        ci_fed.append(ci)
    else:
        avg_fed_mae.append(np.nan)
        ci_fed.append(0.0)

fig, ax = plt.subplots(figsize=(8,4))  # Adjust for column width

# --- Plot Baseline ---
ax.plot(all_weeks, avg_base_mae,
        linestyle='--', marker='o', color='tab:blue', label='Baseline')
ax.fill_between(all_weeks,
                np.array(avg_base_mae) - np.array(ci_base),
                np.array(avg_base_mae) + np.array(ci_base),
                color='tab:blue', alpha=0.2)

# --- Plot Federated ---
ax.plot(all_weeks, avg_fed_mae,
        linestyle='-', marker='x', color='tab:orange', label='Federated')
ax.fill_between(all_weeks,
                np.array(avg_fed_mae) - np.array(ci_fed),
                np.array(avg_fed_mae) + np.array(ci_fed),
                color='tab:orange', alpha=0.2)

# --- Labels and Title ---
ax.set_xlabel('Week', fontsize=11, fontweight='bold', labelpad=6)
ax.set_ylabel('MAE', fontsize=11, fontweight='bold', labelpad=6)
ax.set_title('Average MAE per Week with 95% Confidence Interval',
             fontsize=12, fontweight='bold', pad=10)

# --- Ticks ---
ax.xaxis.set_major_locator(mticker.MaxNLocator(8))
ax.tick_params(axis='x', rotation=45, labelsize=9, pad=2)
ax.tick_params(axis='y', labelsize=9, pad=2)
for tick in ax.get_xticklabels(): tick.set_fontweight('bold')
for tick in ax.get_yticklabels(): tick.set_fontweight('bold')

# --- Legend ---
ax.legend(loc='upper right', fontsize=9, frameon=False)

# --- Grid & Layout ---
ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)
plt.tight_layout(pad=0.5)

# --- Save for LaTeX ---
plt.savefig('mae_comparison.png', bbox_inches='tight')
plt.show()


n_weeks    = len([v for v in avg_base_mae if not np.isnan(v)])
mean_b_all = np.nanmean(avg_base_mae)
std_b_all  = np.nanstd(avg_base_mae, ddof=1)
ci_b_all   = 1.96 * std_b_all / np.sqrt(n_weeks)

mean_f_all = np.nanmean(avg_fed_mae)
std_f_all  = np.nanstd(avg_fed_mae, ddof=1)
ci_f_all   = 1.96 * std_f_all / np.sqrt(n_weeks)

# Paired Wilcoxon on the weekly series
clean_b = [v for v in avg_base_mae if not np.isnan(v)]
clean_f = [v for v in avg_fed_mae  if not np.isnan(v)]
stat, p_val = wilcoxon(clean_b, clean_f)

print(
    f"Across {n_weeks} weekly rounds, the federated model achieved an average MAE of "
    f"{mean_f_all:.2f}  (95% CI [{mean_f_all-ci_f_all:.2f}, {mean_f_all+ci_f_all:.2f}]), "
    f"compared to {mean_b_all:.2f}  (95% CI [{mean_b_all-ci_b_all:.2f}, {mean_b_all+ci_b_all:.2f}]) "
    f"for the centralized baseline. A paired Wilcoxon test yields p = {p_val:.3f}, "
    f"{'no significant difference' if p_val>0.05 else 'a significant difference'}."
)
