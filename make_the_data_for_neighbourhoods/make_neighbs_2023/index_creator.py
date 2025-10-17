from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["citibike"]
collection = db["bike_data_enriched"]

# Create compound index on start_region, hour, and month
compound_index_name = collection.create_index([("start_region", 1), ("hour", 1), ("month", 1)])
print(f"Compound index created: {compound_index_name}")

# Create individual index on start_region
start_region_index = collection.create_index([("start_region", 1)])
print(f"Individual index on start_region created: {start_region_index}")

# Create individual index on hour
hour_index = collection.create_index([("hour", 1)])
print(f"Individual index on hour created: {hour_index}")


