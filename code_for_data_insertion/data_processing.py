import pandas as pd
from pymongo import MongoClient
from shapely.geometry import Point, LineString
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ─── MongoDB Connection ─────────────────────────────
client = MongoClient("mongodb://localhost:27017/")
db = client["citibike"]
raw_collection = db["bikes_raw"]
enriched_collection = db["bike_data_enriched"]

# ─── Extract Unique Stations ─────────────────────────
pipeline = [
    {
        "$group": {
            "_id": "$start_station_id",
            "station_name": {"$first": "$start_station_name"},
            "lat": {"$first": "$start_lat"},
            "lng": {"$first": "$start_lng"}
        }
    },
    {
        "$project": {
            "station_id": "$_id",
            "station_name": 1,
            "lat": 1,
            "lng": 1,
            "_id": 0
        }
    }
]
station_df = pd.DataFrame(list(raw_collection.aggregate(pipeline)))

# ─── Assign Regions to Stations ──────────────────────
line1 = LineString([(-74.0304, 40.7964), (-73.9181, 40.7543)])
line2 = LineString([(-74.0222, 40.7986), (-74.0055, 40.6635)])
line3 = LineString([(-73.9289, 40.8282), (-74.0143, 40.6420)])

def assign_region(lat, lng):
    point = Point(lng, lat)
    p1 = line1.interpolate(line1.project(point))
    p2 = line2.interpolate(line2.project(point))
    p3 = line3.interpolate(line3.project(point))
    if point.y > p1.y:
        return 'upper_manhattan'
    elif p2.y < point.y <= p1.y:
        if abs(point.x - p3.x) <= 0.01:
            return 'lower_manhattan'
        elif point.x > p3.x:
            return 'east_of_manhattan'
        elif point.x < p3.x:
            return 'west_of_manhattan'
    elif point.y <= p2.y:
        return 'south_brooklyn'
    return 'unknown'

station_df['region'] = station_df.apply(lambda row: assign_region(row['lat'], row['lng']), axis=1)
station_df['region'] = station_df['region'].replace({
    'lower_manhattan': 'lower_west_manhattan',
    'west_of_manhattan': 'lower_west_manhattan'
})

# ─── Calculate Subzone Boundaries ────────────────────
subzone_bounds = {}
for region, group in station_df.groupby('region'):
    lat_min, lat_max = group['lat'].min(), group['lat'].max()
    lng_min, lng_max = group['lng'].min(), group['lng'].max()
    subzone_bounds[region] = {
        'lat_mid': (lat_min + lat_max) / 2,
        'lng_mid': (lng_min + lng_max) / 2
    }

def assign_subzone(region, lat, lng):
    if region not in subzone_bounds:
        return 'unknown'
    bounds = subzone_bounds[region]
    if lat >= bounds['lat_mid'] and lng <= bounds['lng_mid']:
        return 'NW'
    elif lat >= bounds['lat_mid'] and lng > bounds['lng_mid']:
        return 'NE'
    elif lat < bounds['lat_mid'] and lng <= bounds['lng_mid']:
        return 'SW'
    else:
        return 'SE'

# ─── Pre-build Station → Region Map ─────────────────
station_region_map = station_df.set_index('station_id')['region'].to_dict()

# ─── Document Enrichment Function ───────────────────
def process_doc(doc):
    try:
        sid = doc.get('start_station_id')
        lat = doc.get('start_lat')
        lng = doc.get('start_lng')

        region = station_region_map.get(sid, 'unknown')
        subzone = assign_subzone(region, lat, lng)

        started_at = pd.to_datetime(doc.get('started_at'), errors='coerce')
        hour = started_at.floor('h') if pd.notnull(started_at) else None
        month = started_at.strftime("%Y%m") if pd.notnull(started_at) else None

        doc['start_region'] = region
        doc['subzone'] = subzone
        doc['hour'] = hour
        doc['month'] = month
        return doc
    except Exception as e:
        # Optionally log error 'e'
        return None

# ─── Batched Processing Function ───────────────────
def process_batch(batch):
    enriched = list(filter(None, map(process_doc, batch)))
    if enriched:
        enriched_collection.insert_many(enriched)
    return len(enriched)

# ─── Main Execution with Offset-Based Pagination ───
BATCH_SIZE = 10000
MAX_WORKERS = 8

total_docs = raw_collection.estimated_document_count()
print(f"🚀 Starting enrichment for ~{total_docs} documents (offset-based pagination)...")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor, tqdm(total=total_docs, desc="Processing batches") as pbar:
    futures = []
    for offset in range(0, total_docs, BATCH_SIZE):
        batch_cursor = raw_collection.find({}).skip(offset).limit(BATCH_SIZE)
        batch = list(batch_cursor)
        if batch:
            futures.append(executor.submit(process_batch, batch))
            pbar.update(len(batch))
    for future in futures:
        try:
            future.result()
        except Exception as e:
            print("Error processing batch:", e)

print("Enrichment completed!")
