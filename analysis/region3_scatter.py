"""Scatter of per-subzone Fed/Bias improvement vs mean hourly demand.

Supports the paper's "regime" claim: federation helps when a subzone's
demand scale is near the federation mean; the four Region-3 subzones fall
in the high-demand tail and local adaptation dominates there.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "current"
USAGE_DIR = REPO_ROOT / "data" / "2024" / "entire_year"
OUT_DIR = REPO_ROOT / "figures"

NODE_TO_SUBZONE = {
    "X": ("region0", 0), "Y": ("region0", 1), "Z": ("region0", 2), "W": ("region0", 3),
    "P": ("region1", 0), "Q": ("region1", 1), "R": ("region1", 2), "L": ("region1", 3),
    "A": ("region2", 0), "B": ("region2", 1), "C": ("region2", 2), "D": ("region2", 3),
    "E": ("region3", 0), "F": ("region3", 1), "G": ("region3", 2), "H": ("region3", 3),
}

REGION_COLOR = {
    "region0": "tab:blue",
    "region1": "tab:orange",
    "region2": "tab:green",
    "region3": "tab:red",
}

rows = []
for node, (region, subzone) in NODE_TO_SUBZONE.items():
    res = pd.read_csv(RESULTS_DIR / f"{node}_results.csv")
    mean_bias = res["base_mae"].mean()
    mean_fed = res["fed_mae"].mean()
    improvement_pct = 100.0 * (mean_bias - mean_fed) / mean_bias

    usage = pd.read_csv(USAGE_DIR / f"{region}_subzone{subzone}_bike_usage.csv")
    mean_demand = usage["bike_usage"].mean()

    rows.append({
        "node": node,
        "region": region,
        "mean_demand": mean_demand,
        "fed_bias_improvement": improvement_pct,
    })

df = pd.DataFrame(rows)
federation_mean = df["mean_demand"].mean()
print(df.to_string(index=False))
print(f"\nFederation mean demand: {federation_mean:.1f} bikes/h")
print(f"Region-3 mean demand:   {df[df.region == 'region3']['mean_demand'].mean():.1f}")
print(f"Other regions mean:     {df[df.region != 'region3']['mean_demand'].mean():.1f}")

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
})

fig, ax = plt.subplots(figsize=(5.6, 3.6))

for region, group in df.groupby("region"):
    ax.scatter(
        group["mean_demand"],
        group["fed_bias_improvement"],
        color=REGION_COLOR[region],
        s=70,
        edgecolor="black",
        linewidth=0.6,
        label=region.replace("region", "Region "),
        zorder=3,
    )
    for _, r in group.iterrows():
        ax.annotate(
            r["node"],
            (r["mean_demand"], r["fed_bias_improvement"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )

ax.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.5, zorder=1)
ax.axvline(
    federation_mean,
    color="gray",
    linewidth=0.8,
    linestyle="--",
    alpha=0.6,
    zorder=1,
    label=f"fed. mean ({federation_mean:.0f} b/h)",
)

ax.set_xlabel("per-subzone mean hourly demand (bikes/h)")
ax.set_ylabel("Fed/Bias improvement (%)")
ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.35)
ax.legend(loc="lower left", frameon=False, ncol=2, columnspacing=1.0, handletextpad=0.4)

fig.tight_layout()
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "region3_scatter.pdf")
fig.savefig(OUT_DIR / "region3_scatter.png", dpi=300)
print(f"\nSaved to {OUT_DIR / 'region3_scatter.pdf'}")
