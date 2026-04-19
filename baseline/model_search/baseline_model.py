"""
One-off data-extraction helper.

NOTE: filename is misleading -- this script does not define or train any
model. It reads the enriched Citi Bike trip records from the MongoDB
collection citibike.bike_data_enriched and aggregates them to
(station_id, hour, bike_usage) triples. It was used once during the
initial dataset build; the per-station CSVs in data/processed/ are the
actual downstream artifacts used by the rest of the pipeline.

Kept for reproducibility of that step. Not part of any current workflow.
"""

from pymongo import MongoClient
import pandas as pd

client = MongoClient("mongodb://localhost:27017/")
col    = client.citibike.bike_data_enriched

pipeline = [
    { "$group": {
        "_id": { "station_id": "$start_station_id", "hour": "$hour" },
        "bike_usage": { "$sum": 1 }
    }},
    { "$project": {
        "_id":        0,
        "station_id": "$_id.station_id",
        "hour":       "$_id.hour",
        "bike_usage": 1
    }},
    { "$sort": { "station_id": 1, "hour": 1 } }
]

results = list(col.aggregate(pipeline))
df = pd.DataFrame(results)
print(df.head())


