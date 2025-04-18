import pandas as pd
from pymongo import MongoClient
from shapely.geometry import Point, LineString
import folium

# ─── MongoDB Connection (Read-Only) ─────────────────────────────
client = MongoClient("mongodb://localhost:27017/")
db = client["citibike"]
raw_collection = db["bikes_raw"]

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

# ─── Define Region Boundaries with Lines ──────────────────────
line1 = LineString([(-74.0304, 40.7964), (-73.9181, 40.7543)])
line2 = LineString([(-74.0222, 40.7986), (-74.0055, 40.6635)])
line3 = LineString([(-73.9289, 40.8282), (-74.0143, 40.6420)])

# ─── Assign Regions to Stations ──────────────────────
def assign_region(lat, lng):
    point = Point(lng, lat)  # Note: Shapely uses (x, y) which is (lng, lat)
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
# Normalize region naming as desired.
station_df['region'] = station_df['region'].replace({
    'lower_manhattan': 'lower_west_manhattan',
    'west_of_manhattan': 'lower_west_manhattan'
})

# ─── Calculate Subzone Boundaries and Assign Subzones ───────────────────
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

station_df['subzone'] = station_df.apply(lambda row: assign_subzone(row['region'], row['lat'], row['lng']), axis=1)

# ─── Define Color Mappings for Regions and Subregions ───────────────────
# Outline color for region
region_color_map = {
    'upper_manhattan': 'orange',
    'lower_west_manhattan': 'darkblue',
    'east_of_manhattan': 'darkgreen',
    'south_brooklyn': 'brown',
    'unknown': 'gray'
}

# Fill color for subregion
subzone_color_map = {
    'NW': 'red',
    'NE': 'blue',
    'SW': 'green',
    'SE': 'purple',
    'unknown': 'lightgray'
}

# ─── Create the Folium Map ─────────────────────────
avg_lat = station_df['lat'].mean()
avg_lng = station_df['lng'].mean()
m = folium.Map(location=[avg_lat, avg_lng], zoom_start=12)

# Add each station with dual-colored marker: outline (region), fill (subzone)
for idx, row in station_df.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['lng']],
        radius=7,
        color=region_color_map.get(row['region'], 'black'),  # Region as outline
        weight=3,
        fill=True,
        fill_color=subzone_color_map.get(row['subzone'], 'black'),  # Subregion as fill
        fill_opacity=0.7,
        popup=f"{row['station_name']}<br>Region: {row['region']}<br>Subzone: {row['subzone']}"
    ).add_to(m)

# ─── Optional: Add a Legend (HTML) ─────────────────────────
legend_html = '''
<div style="
    position: fixed;
    bottom: 50px;
    left: 50px;
    width: 220px;
    height: 180px;
    background-color: white;
    border: 2px solid grey;
    z-index: 9999;
    font-size: 14px;
    padding: 10px;">
<b>Legend</b><br>
<b>Region (Outline):</b><br>
&nbsp;<i style="color: orange;">■</i>&nbsp;Upper Manhattan<br>
&nbsp;<i style="color: darkblue;">■</i>&nbsp;Lower West Manhattan<br>
&nbsp;<i style="color: darkgreen;">■</i>&nbsp;East of Manhattan<br>
&nbsp;<i style="color: brown;">■</i>&nbsp;South Brooklyn<br>
&nbsp;<i style="color: gray;">■</i>&nbsp;Unknown<br>
<b>Subregion (Fill):</b><br>
&nbsp;<i style="color: red;">■</i>&nbsp;NW<br>
&nbsp;<i style="color: blue;">■</i>&nbsp;NE<br>
&nbsp;<i style="color: green;">■</i>&nbsp;SW<br>
&nbsp;<i style="color: purple;">■</i>&nbsp;SE<br>
&nbsp;<i style="color: lightgray;">■</i>&nbsp;Unknown
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# ─── Save and Display the Map ─────────────────────────
m.save('stations_map.html')
print("Map has been saved to stations_map.html")
