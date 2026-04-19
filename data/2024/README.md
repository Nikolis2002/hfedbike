# data/2024/

Everything 2024 — raw inputs and processed outputs. CSVs are gitignored.

## Layout

```
data/2024/
├── raw/
│   ├── weather_2024.csv              raw OpenWeatherMap hourly extract
│   ├── 202401-citibike-tripdata.csv  monthly Citi Bike trip histories
│   ├── 202402-citibike-tripdata.csv
│   ├── ...
│   ├── 202406-citibike-tripdata/     some months arrive pre-split
│   │   ├── 202406-citibike-tripdata_1.csv
│   │   └── ...
│   └── 202412-citibike-tripdata/
│
├── entire_year/                      processed, consumed by the federation
│   ├── fixed_2024.csv                deduplicated hourly weather
│   └── region<G>_subzone<s>_bike_usage.csv   16 per-subzone trip-count CSVs
│
├── first_2_months/                   Jan–Feb 2024 subset (early experiments)
│
└── _model_two_layers.keras           frozen baseline loaded by every node
```

## Where each piece comes from

| File | Produced by |
|---|---|
| `raw/weather_2024.csv`          | Hand-downloaded from OpenWeatherMap |
| `raw/*citibike-tripdata*.csv`   | Hand-downloaded from <https://citibikenyc.com/system-data> |
| `entire_year/fixed_2024.csv`    | `baseline/data_preparation/make_neighbs_2024/first_fix_for_weather.py` |
| `entire_year/region<G>_subzone<s>_bike_usage.csv` | `baseline/data_preparation/make_neighbs_2024/make_the_final_csvsV2.py` |
| `_model_two_layers.keras`       | `baseline/model_search/` grid search + retrain |

## How the federation mounts this

`docker-compose.yml` bind-mounts `data/2024/` to `/data/2024_csvs/`
inside every node container. `federated/p2p_node/v2_node.py` hard-codes
the in-container path `/data/2024_csvs/entire_year/...`, so if you ever
rename this directory, update the compose mounts (not the Python).
