"""Per-week percentage improvement of HFedBike (Fed) over the online
bias-adapted baseline, for the demo paper's second figure.

Reads per-node weekly CSVs in results/current/ and plots one bar per
ISO week of 2024 showing 100 * (bias_mae - fed_mae) / bias_mae
averaged across the 16 clients. Unlike the absolute-MAE figure, this
chart is immediately legible to non-ML viewers at the poster.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "current_pre_v3"
OUT_DIR = REPO_ROOT / "figures"

frames = []
for path in sorted(RESULTS_DIR.glob("*_results.csv")):
    df = pd.read_csv(path)
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["week"])
    df["node"] = path.stem.split("_")[0]
    frames.append(df)

wide = pd.concat(frames, ignore_index=True)
per_week = wide.groupby("week").agg(
    bias=("base_mae", "mean"),
    fed=("fed_mae", "mean"),
).reset_index()
per_week["improvement_pct"] = 100 * (per_week.bias - per_week.fed) / per_week.bias

overall_mean = per_week.improvement_pct.mean()
print(f"Mean weekly improvement: {overall_mean:+.1f}%")
print(f"Min / max weekly:         {per_week.improvement_pct.min():+.1f}%"
      f" / {per_week.improvement_pct.max():+.1f}%")

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
})
fig, ax = plt.subplots(figsize=(8.4, 2.8))

colors = ["#2ea043" if v >= 0 else "#d1495b" for v in per_week.improvement_pct]
ax.bar(per_week.week, per_week.improvement_pct, color=colors,
       edgecolor="black", linewidth=0.3, width=0.85)

ax.axhline(0, color="black", linewidth=0.8)
ax.axhline(overall_mean, color="gray", linestyle="--", linewidth=0.9,
           label=f"52-week mean: {overall_mean:+.1f}%")

ax.set_xlabel("ISO week of 2024")
ax.set_ylabel("HFedBike vs. Bias (%)")
ax.set_xticks(range(0, 55, 4))
ax.set_xlim(per_week.week.min() - 0.5, per_week.week.max() + 0.5)
ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.35)
ax.legend(loc="lower right", frameon=True, framealpha=0.9)

fig.tight_layout()
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "weekly_improvement.pdf")
fig.savefig(OUT_DIR / "weekly_improvement.png", dpi=300)
print(f"\nSaved to {OUT_DIR / 'weekly_improvement.pdf'}")
