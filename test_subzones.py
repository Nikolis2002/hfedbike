import glob
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

records = []

for path in glob.glob("results/*_results.csv"):
    subzone = path.split('/')[-1].replace('_results.csv', '')
    df = pd.read_csv(path)

    base_vals = df['base_mae'].values
    fed_vals  = df['fed_mae'].values
    base_mean = base_vals.mean()
    fed_mean  = fed_vals.mean()
    rel_imp   = (base_mean - fed_mean) / base_mean * 100

    # Compute Wilcoxon W-statistic and p-value
    try:
        stat, p = wilcoxon(base_vals, fed_vals, zero_method="wilcox", correction=False)
        
        # Compute Z-score manually
        diffs = base_vals - fed_vals
        n = np.sum(diffs != 0)
        if n > 0:
            mean_T = n * (n + 1) / 4
            std_T  = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
            z_value = (stat - mean_T) / std_T
        else:
            z_value = np.nan
    except ValueError:
        stat = np.nan
        p = np.nan
        z_value = np.nan

    records.append({
        'Subzone':         subzone,
        'Base MAE':        base_mean,
        'Fed MAE':         fed_mean,
        'Rel. Imp (%)':    rel_imp,
        'Z-score':         z_value,
        'p-value':         p
    })

# Create DataFrame
df_sum = pd.DataFrame(records)
df_sum[['Base MAE', 'Fed MAE', 'Rel. Imp (%)', 'Z-score', 'p-value']] = df_sum[
    ['Base MAE', 'Fed MAE', 'Rel. Imp (%)', 'Z-score', 'p-value']
].round(3)

# Significance summary
sig = (df_sum['Rel. Imp (%)'] > 0) & (df_sum['p-value'] < 0.05)
num_sig = sig.sum()
total = len(df_sum)

print(f"{num_sig}/{total} subzones show a significant MAE reduction (p < 0.05),")
print(f"with mean relative improvement {df_sum['Rel. Imp (%)'].mean():.1f}% across all subzones.")
print(f"with mean Z-score {df_sum['Z-score'].mean():.2f} and mean p-value {df_sum['p-value'].mean():.4f}.\n")

# Print table
print(df_sum.to_string(index=False))

df_sum = df_sum.sort_values(by="Subzone")
# Export LaTeX
latex = df_sum.to_latex(index=False,
    caption="Per‐Subzone MAE, Relative Improvement, Z‐Score, and p‐value from Wilcoxon test",
    label="tab:subzone_stats",
    float_format="%.3f",
    column_format="lrrrrr")
print(latex)
