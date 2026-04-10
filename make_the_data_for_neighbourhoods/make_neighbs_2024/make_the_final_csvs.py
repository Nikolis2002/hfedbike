import pandas as pd
from pymongo import MongoClient

# ─── Connect & point at your enriched set ────────────
client     = MongoClient("mongodb://localhost:27017/")
db         = client["citibike"]
collection = db["test_set"]

selected_regions = ["east_of_manhattan", "south_brooklyn"]
subzones         = ["NW", "NE", "SW", "SE"]

pipeline = [
    {"$match": {"start_region": {"$in": selected_regions}}},
    {"$group": {
        "_id": {
            "hour":    "$hour",
            "region":  "$start_region",
            "subzone": "$subzone"
        },
        "bike_usage": {"$sum": 1}
    }},
    {"$sort": {"_id.hour": 1}}
]

results = list(collection.aggregate(pipeline))
panda   = pd.DataFrame(results)
panda["hour"]    = panda["_id"].apply(lambda x: x["hour"])
panda["region"]  = panda["_id"].apply(lambda x: x["region"])
panda["subzone"] = panda["_id"].apply(lambda x: x["subzone"])
panda = panda.drop(columns=["_id"])

for region in selected_regions:
    for sz in subzones:
        chunk = panda[(panda.region == region) & (panda.subzone == sz)]
        fn    = f"{region}_{sz}_bike_usage.csv"
        chunk.to_csv(fn, index=False)
        print(f"→ Saved {fn} ({len(chunk)} rows)")
