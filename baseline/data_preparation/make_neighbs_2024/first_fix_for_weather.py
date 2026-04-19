"""Clean and normalize the 2024 OpenWeatherMap extract.

Reads the raw weather CSV (data/2024/raw/weather_2024.csv), parses the
irregular timestamp format used by OpenWeatherMap
('2024-01-01 00:00:00 +0000 UTC'), and writes a deduplicated hourly
table to data/2024/entire_year/fixed_2024.csv. That file is then
consumed by make_the_final_csvsV2.py (station-level join) and by
federated/p2p_node/v2_node.py at runtime.

Paths are resolved relative to the script's own location so it can be
invoked from any cwd.
"""

from pathlib import Path
import pandas as pd

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



REPO_ROOT = Path(__file__).resolve().parents[3]  # baseline/data_preparation/make_neighbs_2024/ -> repo root
RAW_PATH = REPO_ROOT / "data" / "2024" / "raw" / "weather_2024.csv"
OUT_PATH = REPO_ROOT / "data" / "2024" / "entire_year" / "fixed_2024.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

weather_df = pd.read_csv(RAW_PATH)

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

clean_weather_df.to_csv(OUT_PATH, index=False)
print(f"Clean weather CSV saved to {OUT_PATH}")
