import pandas as pd
from glob import glob
import os
from pymongo import MongoClient

# ─── MongoDB Connection ─────────────────────
client = MongoClient("mongodb://localhost:27017/")
db = client["citibike"]
collection = db["bikes_raw"]

# Optional: clear the collection before inserting (only run once!)
# collection.delete_many({})

# ─── CSV Import Loop ─────────────────────────
csv_folder = 'all_csvs'  # Adjust if needed
csv_files = glob(os.path.join(csv_folder, '*.csv'))

for file in csv_files:
    try:
        print(f"Processing {os.path.basename(file)}")

        df = pd.read_csv(file)

        # Optional: preview
        print(f"   → {len(df)} records")

        # Convert to list of dicts for MongoDB
        records = df.to_dict(orient='records')

        # Insert into MongoDB
        if records:
            collection.insert_many(records)

        print(f"Inserted {len(records)} records")

    except Exception as e:
        print(f"Error in {file}: {e}")
