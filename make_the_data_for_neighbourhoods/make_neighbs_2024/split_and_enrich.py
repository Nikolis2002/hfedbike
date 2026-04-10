import pandas as pd
from pymongo import MongoClient
from sklearn.cluster import KMeans
import folium
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ─── MongoDB Connection & target collection ──────────
client = MongoClient("mongodb://localhost:27017/")
db = client["citibike"]
enriched_collection = db["test_set"]
enriched_collection.drop()  # clear out old data

# ─── 1) Read your two CSVs into one DataFrame ───────
csv_files = ["202401-citibike-tripdata.csv", "202402-citibike-tripdata.csv"]
df_all = pd.concat(
    [pd.read_csv(f, low_memory=False) for f in csv_files],
    ignore_index=True
)

# ─── 2) Extract Unique Stations ──────────────────────
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

# ─── 3) Assign “regions” and “subzones” via K-means ───

# (A) Fit KMeans(k=4) on all (lat, lng) to get 4 region clusters
coords = station_df[['lat', 'lng']].values
kmeans_regions = KMeans(n_clusters=4, random_state=42)
station_df['region_km'] = kmeans_regions.fit_predict(coords)

# (B) (Optional) If you want human‐readable region names:
# Inspect centroids and map them in the order you like:
# region_centroids = kmeans_regions.cluster_centers_
# print("Region centroids (lat,lng):", region_centroids)
# Suppose you inspect and decide cluster→name mapping:
# region_name_map = {
#     0: "upper_manhattan",
#     1: "south_brooklyn",
#     2: "east_of_manhattan",
#     3: "lower_west_manhattan"
# }
# station_df['region_name'] = station_df['region_km'].map(region_name_map)

# (C) Within each region‐cluster, do KMeans(k=4) to get “subzone_km”
station_df['subzone_km'] = -1
for region_label in sorted(station_df['region_km'].unique()):
    mask = station_df['region_km'] == region_label
    sub_coords = station_df.loc[mask, ['lat', 'lng']].values
    
    if len(sub_coords) < 4:
        # If fewer than 4 stations, collapse to subzone 0
        station_df.loc[mask, 'subzone_km'] = 0
    else:
        km_sub = KMeans(n_clusters=4, random_state=42)
        station_df.loc[mask, 'subzone_km'] = km_sub.fit_predict(sub_coords)

# (D) Build lookups for enrichment
region_lookup  = station_df.set_index("station_id")["region_km"].to_dict()
sub_lookup     = station_df.set_index("station_id")["subzone_km"].to_dict()
# If you used region_name_map, then:
# region_lookup = station_df.set_index("station_id")["region_name"].to_dict()

# ─── 4) Visualize on Folium (color by region_km) ─────
m = folium.Map(location=[40.75, -73.97], zoom_start=12)

# Choose a color palette for 4 clusters:
km_colors = {
    0: "blue",
    1: "orange",
    2: "green",
    3: "red",
}

# Plot each station with its region‐color
for _, row in station_df.iterrows():
    reg = int(row['region_km'])
    popup = f"region_km={reg}, subzone_km={int(row['subzone_km'])}"
    
    folium.CircleMarker(
        location=[row.lat, row.lng],
        radius=4,
        color=km_colors.get(reg, "gray"),
        fill=True,
        fill_opacity=0.8,
        popup=popup
    ).add_to(m)

# Save interactive map
m.save("region_map.html")
print("🗺️  Saved map to region_map.html")


# ─── 5) Doc‐enrichment function using KMeans labels ───
def process_doc(doc: dict) -> dict:
    try:
        sid = doc.get("start_station_id")
        if sid not in region_lookup:
            doc["start_region"] = "unknown"
            doc["subzone"]      = "unknown"
        else:
            doc["start_region"] = int(region_lookup[sid])
            doc["subzone"]      = int(sub_lookup[sid])
        
        ts = pd.to_datetime(doc.get("started_at"), errors="coerce")
        doc["hour"]  = ts.floor("h") if pd.notnull(ts) else None
        doc["month"] = ts.strftime("%Y%m")   if pd.notnull(ts) else None
        
        return doc
    except:
        return None

def process_batch(batch: list[dict]) -> int:
    enriched = [d for d in (process_doc(d) for d in batch) if d]
    if enriched:
        enriched_collection.insert_many(enriched)
    return len(enriched)

# ─── 6) Batch‐process all rows into `test_set` ───────
BATCH_SIZE = 10_000
total_rows = len(df_all)
print(f"🚀 Enriching {total_rows} rows → citibike.test_set…")

with ThreadPoolExecutor(max_workers=8) as exe, \
     tqdm(total=total_rows, desc="Batches") as pbar:
    futures = []
    for start in range(0, total_rows, BATCH_SIZE):
        batch = df_all.iloc[start:start+BATCH_SIZE].to_dict("records")
        futures.append(exe.submit(process_batch, batch))
        pbar.update(len(batch))
    for f in futures:
        f.result()

print("✅ Enrichment complete.")
