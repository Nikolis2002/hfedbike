import pandas as pd
from pymongo import MongoClient


client = MongoClient("mongodb://localhost:27017/")
db = client["citibike"]
collection = db["bike_data_enriched"]

selected_regions = ["upper_manhattan", "east_of_manhattan"]
subzones = ["NW", "NE", "SW", "SE"]


pipeline = [
    {"$match": {"start_region": {"$in": selected_regions}}},
    {"$group":{
        "_id":{
            "hour":"$hour",
            "region":"$start_region",
            "subzone":"$subzone"
        },
        "bike_usage": {"$sum":1}
    }},
    {"$sort":{"_id.hour":1}}
]

results= list(collection.aggregate(pipeline))
panda=pd.DataFrame(results)

panda["hour"]= panda["_id"].apply(lambda x: x["hour"])
panda["region"] = panda["_id"].apply(lambda x: x["region"])
panda["subzone"] = panda["_id"].apply(lambda x: x["subzone"])
panda = panda.drop(columns=["_id"])

print(panda.head())

for region in selected_regions:
    for sub_region in subzones:
        data = panda[(panda["region"] == region) & (panda["subzone"] == sub_region)]
        print(data.head())
        file=f"{region}_{sub_region}_bike_usage.csv"
        data.to_csv(file,index=False)
        print(f"Save to {file} with {len(data)} ")