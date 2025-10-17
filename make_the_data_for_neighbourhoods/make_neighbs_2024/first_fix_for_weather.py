import pandas as pd
from pymongo import MongoClient
from glob import glob
# Custom parsing function to handle strings like "2023-01-01 00:00:00 +0000 UTC"
def parse_dt_iso(x):
    if pd.isna(x):
        return None
    # Remove trailing " UTC" so that the offset remains as e.g. "+0000"
    x = x.replace(" UTC", "")
    dt = pd.to_datetime(x, errors="coerce")
    # Remove timezone info to create a naive datetime
    if dt is not None and not pd.isna(dt):
        dt = dt.tz_localize(None)
    return dt



weather_df = pd.read_csv('2024.csv')

# Check if 'hour' column exists; if not, create it.
if "hour" not in weather_df.columns:
    # Parse the dt_iso column
    weather_df["dt_iso_parsed"] = weather_df["dt_iso"].apply(parse_dt_iso)
    # Create the 'hour' column by flooring to the hour
    weather_df["hour"] = weather_df["dt_iso_parsed"].dt.floor("h")

# Now that we have the 'hour' column, find duplicates
duplicate_mask = weather_df.duplicated(subset=["hour"], keep=False)
duplicates = weather_df[duplicate_mask]

print("Duplicate hours with multiple entries:")
print(duplicates["hour"].value_counts())

# Group by the 'hour' value and inspect each group.
grouped = duplicates.groupby("hour")

for hour, group in grouped:
    print(f"\nHour: {hour}, Count: {len(group)}")
    print("Unique counts per column in this group:")
    # Check key weather columns (adjust as needed)
    print(group[["temp", "visibility", "dew_point", "feels_like", "humidity", "wind_speed", "weather_main"]].nunique())

# Drop duplicate rows based only on the "hour" column.
clean_weather_df = weather_df.drop_duplicates(subset=["hour"])
print("Cleaned weather data shape:", clean_weather_df.shape)

clean_weather_df.to_csv(f"fixed_2024.csv", index=False)
print("Clean weather CSV saved.")
