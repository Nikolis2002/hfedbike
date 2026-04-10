import glob
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def wilcoxon_with_z(x, y):
    """
    Paired Wilcoxon signed-rank test + normal-approx Z (same spirit as your code).
    Returns (W, p, z). Uses wilcoxon(x, y) where W is the test statistic returned by SciPy.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    try:
        stat, p = wilcoxon(x, y, zero_method="wilcox", correction=False)
        diffs = x - y
        n = np.sum(diffs != 0)
        if n > 0:
            mean_T = n * (n + 1) / 4
            std_T = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
            z = (stat - mean_T) / std_T
        else:
            z = np.nan
        return stat, p, z
    except ValueError:
        return np.nan, np.nan, np.nan


records = []

for path in glob.glob("results/*_results.csv"):
    subzone = path.split("/")[-1].replace("_results.csv", "")
    df = pd.read_csv(path)

    # --- rename for clarity (your file uses base_mae as the bias-addition variant) ---
    freeze = df["freeze_mae"].values
    bias = df["base_mae"].values
    fed = df["fed_mae"].values

    freeze_mean = float(np.mean(freeze))
    bias_mean = float(np.mean(bias))
    fed_mean = float(np.mean(fed))

    # relative improvements (positive = improvement, i.e., lower MAE)
    imp_bias_vs_freeze = (freeze_mean - bias_mean) / freeze_mean * 100
    imp_fed_vs_bias = (bias_mean - fed_mean) / bias_mean * 100
    imp_fed_vs_freeze = (freeze_mean - fed_mean) / freeze_mean * 100

    # Wilcoxon tests
    stat_fb, p_fb, z_fb = wilcoxon_with_z(freeze, bias)  # Freeze vs Bias
    stat_bf, p_bf, z_bf = wilcoxon_with_z(bias, fed)  # Bias vs Fed
    stat_ff, p_ff, z_ff = wilcoxon_with_z(freeze, fed)  # Freeze vs Fed

    records.append(
        {
            "Subzone": subzone,
            "Freeze MAE": freeze_mean,
            "Bias MAE": bias_mean,
            "Fed MAE": fed_mean,
            "Imp Bias vs Freeze (%)": imp_bias_vs_freeze,
            "Imp Fed vs Bias (%)": imp_fed_vs_bias,
            "Imp Fed vs Freeze (%)": imp_fed_vs_freeze,
            "Z (Freeze vs Bias)": z_fb,
            "p (Freeze vs Bias)": p_fb,
            "Z (Bias vs Fed)": z_bf,
            "p (Bias vs Fed)": p_bf,
            "Z (Freeze vs Fed)": z_ff,
            "p (Freeze vs Fed)": p_ff,
        }
    )

df_sum = pd.DataFrame(records).sort_values("Subzone")

# rounding for display/export
round_cols = [
    "Freeze MAE",
    "Bias MAE",
    "Fed MAE",
    "Imp Bias vs Freeze (%)",
    "Imp Fed vs Bias (%)",
    "Imp Fed vs Freeze (%)",
    "Z (Freeze vs Bias)",
    "p (Freeze vs Bias)",
    "Z (Bias vs Fed)",
    "p (Bias vs Fed)",
    "Z (Freeze vs Fed)",
    "p (Freeze vs Fed)",
]
df_sum[round_cols] = df_sum[round_cols].round(3)

# significance counts
alpha = 0.05
sig_fb = (df_sum["Imp Bias vs Freeze (%)"] > 0) & (df_sum["p (Freeze vs Bias)"] < alpha)
sig_bf = (df_sum["Imp Fed vs Bias (%)"] > 0) & (df_sum["p (Bias vs Fed)"] < alpha)
sig_ff = (df_sum["Imp Fed vs Freeze (%)"] > 0) & (df_sum["p (Freeze vs Fed)"] < alpha)

print(
    f"Bias vs Freeze: {sig_fb.sum()}/{len(df_sum)} subzones significantly improved (p<{alpha}). "
    f"Mean improvement: {df_sum['Imp Bias vs Freeze (%)'].mean():.1f}%"
)

print(
    f"Fed vs Bias:    {sig_bf.sum()}/{len(df_sum)} subzones significantly improved (p<{alpha}). "
    f"Mean improvement: {df_sum['Imp Fed vs Bias (%)'].mean():.1f}%"
)

print(
    f"Fed vs Freeze:  {sig_ff.sum()}/{len(df_sum)} subzones significantly improved (p<{alpha}). "
    f"Mean improvement: {df_sum['Imp Fed vs Freeze (%)'].mean():.1f}%\n"
)

print(df_sum.to_string(index=False))

# --- LaTeX export (choose a narrower subset of columns for IEEE width) ---
latex_cols = [
    "Subzone",
    "Freeze MAE",
    "Bias MAE",
    "Fed MAE",
    "Imp Bias vs Freeze (%)",
    "Imp Fed vs Bias (%)",
    "Imp Fed vs Freeze (%)",
    "p (Freeze vs Bias)",
    "p (Bias vs Fed)",
    "p (Freeze vs Fed)",
]
latex_table = df_sum[latex_cols].to_latex(
    index=False,
    caption="Per-subzone MAE and relative improvements across models, with paired Wilcoxon $p$-values.",
    label="tab:subzone_stats_3models",
    float_format="%.3f",
    column_format="lrrr|rrr|rrr",
)

print("\n" + latex_table)
