"""Produce the 16 per-subzone bike-usage CSVs consumed by the federation.

Pipeline (all in-memory, no MongoDB round-trip):

  1. Read every ``*citibike-tripdata*.csv`` under ``data/2024/raw/``,
     keeping only the five columns we actually need.
  2. Build one row per unique station (lat/lng) via a groupby.
  3. k-means (k=4) on station coordinates → 4 regions.
  4. k-means (k=4) within each region → 4 subzones → 16 subzones total.
  5. Vectorized: parse ``started_at``, floor to the hour, join trips with
     the station→(region, subzone) lookup, groupby
     (region, subzone, hour) and count → one hourly bike-usage row per
     cell.
  6. Write one CSV per (region, subzone) to ``data/2024/entire_year/``.
  7. Emit an optional Folium diagnostic map of the clusters.

The 16 output CSVs are the exact inputs read by
``federated/p2p_node/v2_node.py`` at runtime. Schema:
``region, subzone, hour, bike_usage``.

Why no MongoDB
--------------
The previous revision enriched every trip by looping over chunks in
Python (``pd.to_datetime`` per row, dict juggling, ``insert_many``),
then read the enriched collection back out via an aggregation pipeline
to produce the CSVs. On ~40M trips that was the bulk of the run time.
The rewrite vectorizes each stage -- single ``pd.to_datetime`` call on
a whole column, a hash-based merge, and one groupby -- producing the
same output files an order of magnitude faster. If you ever need the
enriched collection in Mongo for ad-hoc analysis, use ``pandas``'
``DataFrame.to_dict("records") + insert_many`` at the end; the
intermediate stored collection is not read by anything else in the
repo.

The subzone-to-node mapping under ``subzone_name_lookup`` below must
agree with ``federated/p2p_node/v2_node.py``'s ``subzone_node`` dict.
If you ever re-run k-means on a different station set, labels rotate
unpredictably; update both files together.
"""

from pathlib import Path
import pandas as pd
import folium
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans

# ─── Paths ─────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR   = REPO_ROOT / "data" / "2024" / "raw"
OUT_DIR   = REPO_ROOT / "data" / "2024" / "entire_year"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Columns we actually need out of the raw Citi Bike CSVs. Citi Bike's
# extra fields (ride_id, ended_at, end_*, member_casual, rideable_type)
# are ~70% of the row width and just slow us down.
USECOLS = [
    "started_at", "start_station_id",
    "start_station_name", "start_lat", "start_lng",
]

# ─── 1) Load all trip CSVs ────────────────────────────────────────
csv_files = sorted(RAW_DIR.rglob("*citibike-tripdata*.csv"))
print(f"Found {len(csv_files)} tripdata CSVs under {RAW_DIR}")


def _read_raw(path):
    """Read a single raw CSV with utf-8/latin1 fallback."""
    for enc in ("utf-8", "latin1"):
        try:
            return pd.read_csv(
                path,
                usecols=USECOLS,
                dtype={"start_station_id": "string",
                       "start_station_name": "string"},
                low_memory=False,
                encoding=enc,
            )
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode {path}")


dfs = []
for f in csv_files:
    print(f"  reading {f.name}")
    dfs.append(_read_raw(f))
df_all = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(df_all):,} raw trip rows")
# Free the per-file frames so we don't double memory when we start
# engineering features below.
del dfs

# ─── 2) Station table ─────────────────────────────────────────────
# One row per unique station. first() assumes the station's coordinates
# are stable year-over-year, which holds for the Citi Bike 2024 data.
station_df = (
    df_all
    .groupby("start_station_id", as_index=False)
    .agg({"start_station_name": "first",
          "start_lat": "first",
          "start_lng": "first"})
    .rename(columns={"start_station_id": "station_id",
                     "start_station_name": "station_name",
                     "start_lat": "lat",
                     "start_lng": "lng"})
)
print(f"{len(station_df):,} unique stations")

# ─── 3) k-means regions + subzones ────────────────────────────────
# random_state pinned so repeated runs on the same dataset produce
# stable labels. If you re-run on a different station set, labels can
# rotate; v2_node.py's subzone_node dict must stay aligned.
coords = station_df[["lat", "lng"]].to_numpy()
station_df["region_km"] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(coords)

station_df["subzone_km"] = -1
for region_label in sorted(station_df["region_km"].unique()):
    mask = station_df["region_km"] == region_label
    sub_coords = station_df.loc[mask, ["lat", "lng"]].to_numpy()
    if len(sub_coords) < 4:
        station_df.loc[mask, "subzone_km"] = 0
    else:
        station_df.loc[mask, "subzone_km"] = KMeans(
            n_clusters=4, random_state=42, n_init=10,
        ).fit_predict(sub_coords)

# ─── 4) Enrich trips with (region, subzone, hour) and aggregate ───
# Single pd.to_datetime call on the full column is ~100x faster than
# per-row parsing. `format="ISO8601"` is required: the Citi Bike dumps
# switched formats mid-year -- Jan-Apr 2024 use millisecond precision
# (``2024-04-27 13:56:13.940``) while May onward drops milliseconds
# (``2024-05-01 08:05:53``). Default format inference picks whichever
# the first rows use, so the other months silently become NaT. The
# ISO8601 parser accepts both variants and is also vectorized.
# errors="coerce" still protects against unparseable lines.
df_all["hour"] = pd.to_datetime(
    df_all["started_at"], format="ISO8601", errors="coerce"
).dt.floor("h")
df_all = df_all.dropna(subset=["hour"])

# Hash-join the station→cluster lookup onto the trip frame. Trips whose
# start_station_id isn't in station_df (extremely rare) get dropped.
df_all = df_all.merge(
    station_df[["station_id", "region_km", "subzone_km"]],
    left_on="start_station_id",
    right_on="station_id",
    how="inner",
)

hourly = (
    df_all.groupby(["region_km", "subzone_km", "hour"], sort=True)
    .size()
    .rename("bike_usage")
    .reset_index()
)
# Free the big trip frame ASAP.
del df_all
print(f"Aggregated to {len(hourly):,} (region, subzone, hour) rows")

# ─── 5) Per-subzone CSVs ──────────────────────────────────────────
written = 0
for (r, s), grp in hourly.groupby(["region_km", "subzone_km"], sort=True):
    fn = OUT_DIR / f"region{r}_subzone{s}_bike_usage.csv"
    (
        grp.rename(columns={"region_km": "region", "subzone_km": "subzone"})
        [["region", "subzone", "hour", "bike_usage"]]
        .to_csv(fn, index=False)
    )
    written += len(grp)
    print(f"  → {fn.name} ({len(grp):,} rows)")
print(f"Wrote {written:,} aggregated rows across 16 CSVs.")

# ─── 6) Folium diagnostic map ─────────────────────────────────────
# Not consumed by any downstream script; purely for eyeballing the
# clusters after a re-run.
region_colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}
subzone_name_lookup = {
    "region0": {"0": "X", "1": "Y", "2": "Z", "3": "W"},
    "region1": {"0": "P", "1": "Q", "2": "R", "3": "L"},
    "region2": {"0": "A", "1": "B", "2": "C", "3": "D"},
    "region3": {"0": "E", "1": "F", "2": "G", "3": "H"},
}

m = folium.Map(location=[40.75, -73.97], zoom_start=12)

for region_label in sorted(station_df["region_km"].unique()):
    pts = station_df.loc[station_df["region_km"] == region_label,
                         ["lat", "lng"]].to_numpy()
    if len(pts) < 3:
        continue
    hull = ConvexHull(pts)
    folium.Polygon(
        locations=[(pts[v, 0], pts[v, 1]) for v in hull.vertices],
        color=region_colors[region_label], weight=2,
        fill=True, fill_color=region_colors[region_label], fill_opacity=0.3,
        popup=f"Region {region_label}",
    ).add_to(m)

for region_label in sorted(station_df["region_km"].unique()):
    for sub_label in sorted(station_df.loc[station_df["region_km"] == region_label,
                                           "subzone_km"].unique()):
        pts = station_df.loc[
            (station_df["region_km"] == region_label) &
            (station_df["subzone_km"] == sub_label),
            ["lat", "lng"],
        ].to_numpy()
        if len(pts) < 3:
            continue
        hull = ConvexHull(pts)
        coords_ = [(pts[v, 0], pts[v, 1]) for v in hull.vertices]
        folium.PolyLine(
            locations=coords_ + [coords_[0]],
            color="black", weight=4, opacity=0.9,
        ).add_to(m)

        centroid = pts.mean(axis=0)
        name = subzone_name_lookup[f"region{region_label}"][str(sub_label)]
        folium.Marker(
            location=(centroid[0], centroid[1]),
            icon=folium.DivIcon(
                html=(
                    '<div style="font-size:16pt;font-weight:900;color:black;'
                    'text-shadow:1px 1px 2px white;">'
                    f"{name}</div>"
                )
            ),
        ).add_to(m)

for _, row in station_df.iterrows():
    folium.CircleMarker(
        location=(row["lat"], row["lng"]),
        radius=2.5, color="gray", fill=True,
        fill_color="gray", fill_opacity=0.6, weight=0.5,
    ).add_to(m)

map_path = OUT_DIR.parent / "region_subzone_map.html"
m.save(str(map_path))
print(f"Saved diagnostic map to {map_path}")

# ─── 7) Station-count summary (LaTeX-ready) ───────────────────────
summary = (
    station_df.groupby(["region_km", "subzone_km"])
    .size()
    .rename("# Stations")
    .reset_index()
    .sort_values(["region_km", "subzone_km"])
)
summary["Region"]    = summary["region_km"].apply(lambda x: f"Region {x}")
summary["Color"]     = summary["region_km"].map(
    {0: "blue", 1: "orange", 2: "green", 3: "red"}
)
summary["Subregion"] = summary["subzone_km"]
summary = summary[["Region", "Color", "Subregion", "# Stations"]]

print("\nStation counts per Region/Subregion:\n")
print(summary.to_string(index=False))
print("\nLaTeX:\n")
print(summary.to_latex(
    index=False,
    caption="Number of stations per region and subregion",
    label="tab:region_subzone_counts",
))
