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


