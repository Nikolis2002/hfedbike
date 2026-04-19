"""Produce the 16 per-subzone bike-usage CSVs consumed by the federation.

Pipeline:
  1. Read the enriched 2024 trip records (cleaned by
     split_and_enrich.py) and the cleaned weather table
     (fixed_2024.csv).
  2. k-means on station coordinates, k=4, produces the four regions.
  3. k-means again within each region, k=4, produces the four
     subzones per region (16 subzones total).
  4. For each subzone, aggregate trips to hourly counts and join with
     weather.
  5. Write one CSV per subzone under data/2024/entire_year/, named
     region<G>_subzone<s>_bike_usage.csv.

These files are the exact inputs read by federated/p2p_node/v2_node.py
at runtime.
"""

import pandas as pd
from pymongo import MongoClient
from sklearn.cluster import KMeans
import folium
from scipy.spatial import ConvexHull
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path

# ─── CONFIGURATION ───────────────────────────────────────
STORE_TO_DB = False   # Set to True to drop & re-store enrichment into MongoDB
DB_NAME = "citibike"
COLLECTION_NAME = "test_setV2"
CSV_EXPORT = True    # Set to True to export per‐region/subzone CSVs at the end

# ─── MongoDB Connection & target collection ─────────────
client = MongoClient("mongodb://localhost:27017/")
db = client[DB_NAME]
enriched_collection = db[COLLECTION_NAME]

if STORE_TO_DB:
    enriched_collection.drop()  # clear out old data

# ─── 1) Read the January & February 2024 trip CSVs ───────
#csv_files = ["202401-citibike-tripdata.csv", "202402-citibike-tripdata.csv"]
#df_all = pd.concat(
#    [pd.read_csv(f, low_memory=False) for f in csv_files],
#    ignore_index=True
#)

BASE_DIR = Path(__file__).resolve().parent   # folder where the script is
csv_files = sorted(
    str(p) for p in BASE_DIR.rglob("*citibike-tripdata*.csv")
    if p.name != "fixed_2024.csv"  # optional: exclude other csvs like weather/etc
)

print(f"Found {len(csv_files)} tripdata CSVs:")
for f in csv_files[:10]:
    print(" -", f)
if len(csv_files) > 10:
    print(" ...")

def read_csv_safe(path):
    try:
        return pd.read_csv(path, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        print(f"⚠️ utf-8 failed → retrying latin1: {path}")
        return pd.read_csv(path, low_memory=False, encoding="latin1")

dfs = []
for f in csv_files:
    print("Reading:", f)
    dfs.append(read_csv_safe(f))

df_all = pd.concat(dfs, ignore_index=True)
# ─── 2) Extract Unique Stations ─────────────────────────
station_df = (
    df_all
    .groupby("start_station_id", as_index=False)
    .agg({
        "start_station_name": "first",
        "start_lat":          "first",
        "start_lng":          "first"
    })
    .rename(columns={
        "start_station_id":   "station_id",
        "start_station_name": "station_name",
        "start_lat":          "lat",
        "start_lng":          "lng"
    })
)

# ─── 3) Assign regions & subzones via K-means ───────────
coords = station_df[['lat', 'lng']].values
kmeans_regions = KMeans(n_clusters=4, random_state=42)
station_df['region_km'] = kmeans_regions.fit_predict(coords)

station_df['subzone_km'] = -1
for region_label in sorted(station_df['region_km'].unique()):
    mask = station_df['region_km'] == region_label
    sub_coords = station_df.loc[mask, ['lat', 'lng']].values
    if len(sub_coords) < 4:
        station_df.loc[mask, 'subzone_km'] = 0
    else:
        km_sub = KMeans(n_clusters=4, random_state=42)
        station_df.loc[mask, 'subzone_km'] = km_sub.fit_predict(sub_coords)

# Build lookup dicts for enrichment
region_lookup = station_df.set_index("station_id")["region_km"].to_dict()
sub_lookup = station_df.set_index("station_id")["subzone_km"].to_dict()

# ─── 4) Doc‐enrichment function using K-Means labels ─────
def process_doc(doc: dict) -> dict:
    try:
        sid = doc.get("start_station_id")
        if sid not in region_lookup:
            doc["start_region"] = "unknown"
            doc["subzone"] = "unknown"
        else:
            doc["start_region"] = int(region_lookup[sid])
            doc["subzone"] = int(sub_lookup[sid])

        ts = pd.to_datetime(doc.get("started_at"), errors="coerce")
        doc["hour"] = ts.floor("h") if pd.notnull(ts) else None
        doc["month"] = ts.strftime("%Y%m") if pd.notnull(ts) else None
        return doc
    except:
        return None

    
CHUNK = 50_000

def iter_csv_chunks(path):
    try:
        return pd.read_csv(path, chunksize=CHUNK, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, chunksize=CHUNK, low_memory=False, encoding="latin1")

if STORE_TO_DB:
    total = 0
    for f in csv_files:
        print("📄", f)
        for chunk in iter_csv_chunks(f):
            batch = chunk.to_dict("records")
            enriched = [d for d in (process_doc(d) for d in batch) if d]
            if enriched:
                enriched_collection.insert_many(enriched, ordered=False)
                total += len(enriched)
    print("✅ Inserted:", total)

# ─── 6) Create a Folium map showing regions & subzones ───
# Define distinct fill colors for each of the 4 regions
region_colors = {
    0: "#1f77b4",  # blue
    1: "#ff7f0e",  # orange
    2: "#2ca02c",  # green
    3: "#d62728"   # red
}

m = folium.Map(location=[40.75, -73.97], zoom_start=12)

# 1) Draw each region's convex hull with a less transparent fill
for region_label in sorted(station_df['region_km'].unique()):
    pts_region = station_df.loc[
        station_df['region_km'] == region_label,
        ['lat', 'lng']
    ].values
    if len(pts_region) < 3:
        continue
    hull_region = ConvexHull(pts_region)
    hull_coords_region = [(pts_region[v, 0], pts_region[v, 1]) for v in hull_region.vertices]

    folium.Polygon(
        locations=[(lat, lng) for lat, lng in hull_coords_region],
        color=region_colors[region_label],      # region border color
        weight=2,
        fill=True,
        fill_color=region_colors[region_label],  # same fill color
        fill_opacity=0.3,                        # more visible fill
        popup=f"Region {region_label}"
    ).add_to(m)

# 2) Draw subzones with thick black outline
for region_label in sorted(station_df['region_km'].unique()):
    for sub_label in sorted(station_df.loc[station_df['region_km'] == region_label, 'subzone_km'].unique()):
        pts_sub = station_df.loc[
            (station_df['region_km'] == region_label) &
            (station_df['subzone_km'] == sub_label),
            ['lat', 'lng']
        ].values
        if len(pts_sub) < 3:
            continue
        hull_sub = ConvexHull(pts_sub)
        hull_coords_sub = [(pts_sub[v, 0], pts_sub[v, 1]) for v in hull_sub.vertices]
        outline_coords = hull_coords_sub + [hull_coords_sub[0]]

        folium.PolyLine(
            locations=[(lat, lng) for lat, lng in outline_coords],
            color="black",
            weight=4,
            opacity=0.9
        ).add_to(m)

# Subzone label mapping
subzone_name_lookup = {
    'region0': {"0": "X", "1": "Y", "2": "Z", "3": "W"},
    'region1': {"0": "P", "1": "Q", "2": "R", "3": "L"},
    'region2': {"0": "A", "1": "B", "2": "C", "3": "D"},
    'region3': {"0": "E", "1": "F", "2": "G", "3": "H"},
}

# 3) Place bold and large subzone labels
for region_label in sorted(station_df['region_km'].unique()):
    region_key = f"region{region_label}"
    for sub_label in sorted(station_df.loc[station_df['region_km'] == region_label, 'subzone_km'].unique()):
        pts_sub = station_df.loc[
            (station_df['region_km'] == region_label) &
            (station_df['subzone_km'] == sub_label),
            ['lat', 'lng']
        ].values
        if len(pts_sub) < 3:
            continue

        centroid_lat = pts_sub[:, 0].mean()
        centroid_lng = pts_sub[:, 1].mean()
        subzone_name = subzone_name_lookup[region_key][str(sub_label)]

        folium.Marker(
            location=(centroid_lat, centroid_lng),
            icon=folium.DivIcon(
                html=f'''
                <div style="
                    font-size:16pt;
                    font-weight:900;
                    color:black;
                    text-shadow: 1px 1px 2px white;
                ">{subzone_name}</div>
                '''
            )
        ).add_to(m)
# 4) Lightly plot each station as a small grey circle
for _, row in station_df.iterrows():
    folium.CircleMarker(
        location=(row['lat'], row['lng']),
        radius=2.5,                    # small dot
        color='gray',
        fill=True,
        fill_color='gray',
        fill_opacity=0.6,
        weight=0.5                     # thin border
    ).add_to(m)


# Save output
m.save("region_subzone_map.html")
print("Saved map to region_subzone_map.html")

summary = (
station_df
.groupby(['region_km','subzone_km'])
.agg({'station_id':'count'})
.rename(columns={'station_id':'# Stations'})
.reset_index()
.sort_values(['region_km','subzone_km'])
)

# Map region index to readable color names
color_names = {
    0: "blue",
    1: "orange",
    2: "green",
    3: "red"
}
summary['Region'] = summary['region_km'].apply(lambda x: f"Region {x}")
summary['Color']  = summary['region_km'].map(color_names)
summary['Subregion'] = summary['subzone_km']

# Reorder columns for display
summary = summary[['Region','Color','Subregion','# Stations']]

# 1) Print the table to console
print("\nStation counts per Region/Subregion:\n")
print(summary.to_string(index=False))

# 2) Generate LaTeX code for the table
latex_table = summary.to_latex(
    index=False,
    caption="Number of stations per region and subregion",
    label="tab:region_subzone_counts"
)
print("\nLaTeX table code:\n")
print(latex_table)


# ─── 7) Optional CSV export for aggregated usage ───────
if CSV_EXPORT:
    import csv

    selected_regions = [0, 1, 2, 3]
    subzones = [0, 1, 2, 3]

    # (Optional but recommended) speed up match/group/sort
    enriched_collection.create_index([("start_region", 1), ("subzone", 1), ("hour", 1)])

    pipeline = [
        {"$match": {
            "start_region": {"$in": selected_regions},
            "hour": {"$ne": None},
            "subzone": {"$ne": "unknown"}
        }},
        {"$group": {
            "_id": {
                "hour":   "$hour",
                "region": "$start_region",
                "subzone":"$subzone"
            },
            "bike_usage": {"$sum": 1}
        }},
        {"$sort": {"_id.hour": 1}}
    ]

    # --- stream results, write per (region, subzone) CSV without pandas ---
    writers = {}
    files = {}

    def writer_for(region, subzone):
        key = (region, subzone)
        if key not in writers:
            fn = f"region{region}_subzone{subzone}_bike_usage.csv"
            f = open(fn, "w", newline="")
            files[key] = f
            w = csv.writer(f)
            w.writerow(["region", "subzone", "hour", "bike_usage"])
            writers[key] = w
            print(f"→ Opened {fn}")
        return writers[key]

    cursor = enriched_collection.aggregate(
        pipeline,
        allowDiskUse=True,
        batchSize=50_000
    )

    count_rows = 0
    for doc in cursor:
        _id = doc["_id"]
        r = int(_id["region"])
        sz = int(_id["subzone"])
        hour = _id["hour"]
        usage = int(doc["bike_usage"])

        # only keep your 0..3 subzones (in case something weird exists)
        if r in selected_regions and sz in subzones:
            writer_for(r, sz).writerow([r, sz, hour, usage])
            count_rows += 1

    for f in files.values():
        f.close()

    print(f"✅ Export done. Wrote {count_rows} aggregated rows into per-subzone CSVs.")
