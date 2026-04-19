# Baseline pipeline

Turns raw Citi Bike trip histories + OpenWeatherMap hourly observations
into a frozen neural-network baseline that every federated node loads at
start-up. Offline, one-time — distinct from the online federated runtime
under `federated/`.

## `data_preparation/`

Where the raw data lives (same convention for both years):
```
data/2023/     raw Citi Bike monthly trip CSVs
data/2024/raw/                monthly trip CSVs and sub-directories
data/2024/raw/weather_2024.csv  OpenWeatherMap hourly extract
data/2024/entire_year/        processed outputs consumed by the federation
```

### `make_neighbs_2023/`

One script kept:

| File | Role |
|---|---|
| `data_enrich.py` | Floors each trip's `started_at` to the hour and assigns an older, polygon-based region/subzone label. Reads from MongoDB (`citibike.bikes_raw`) and writes back to `citibike.bike_data_enriched`. |

The 2023 pipeline's other steps (raw ingestion, per-neighborhood
aggregation, station index build) were run once and their outputs live
under `data/processed/*_bike_usage.csv`. Those files are the input to
the baseline grid search (`baseline/model_search/pre_processing.py`); no
reason to re-run the 2023 pipeline unless you're rebuilding from
scratch.

### `make_neighbs_2024/`

Two scripts kept. Run in order:

1. **`first_fix_for_weather.py`**
   Reads `data/2024/raw/weather_2024.csv`, parses the irregular
   OpenWeatherMap timestamp, drops duplicate hours, writes
   `data/2024/entire_year/fixed_2024.csv`.

2. **`make_the_final_csvsV2.py`**
   Reads every `*citibike-tripdata*.csv` under `data/2024/raw/`, assigns
   each station to a (region, subzone) via k-means (k=4 then k=4 within
   each region), bulk-loads enriched records to MongoDB
   (`citibike.test_setV2`), aggregates to hourly bike-usage counts per
   subzone, and writes one CSV per subzone to
   `data/2024/entire_year/region<G>_subzone<s>_bike_usage.csv`.
   Also emits a diagnostic map (`region_subzone_map.html`).

   MongoDB is required; toggle `STORE_TO_DB = True` at the top of the
   file on a first run (the CSV-export step reads from the collection).

The 16 CSVs produced by step 2 are the exact inputs read by
`federated/p2p_node/v2_node.py` at runtime. The subzone-to-node mapping
(X→region0/subzone0, Y→region0/subzone1, …) is hard-coded in both
`make_the_final_csvsV2.py` and `v2_node.py` — if you ever re-cluster,
update both.

## `model_search/`

Picks the architecture and hyperparameters used as the frozen baseline.

| File | What it does |
|---|---|
| `pre_processing.py` | **The actual training worker** despite the name. Invoked by `fine_tuning.py` as a subprocess with CLI args. Loads `data/processed/clean_weather.csv` and the per-neighborhood bike-usage CSVs, builds the model, runs 5-fold CV, and logs the fold-average metrics to MongoDB (`citibike.resultsv2`). |
| `baseline_model.py` | One-off MongoDB data-extraction helper. Does **not** train a model despite the filename — pulls aggregated `(station_id, hour, bike_usage)` tuples. Kept for reproducibility. |
| `fine_tuning.py` | Grid-search driver. Enumerates 1,734 (optimizer, layer widths, L2, lr) combinations and invokes `pre_processing.py` once each. Resumable via `completed_runsv2.txt`. |
| `find_the_best_baseline.py` | Queries MongoDB for the minimum-MAE run per regularization regime (pure L1 vs pure L2). |
| `best_baseline_model.py` | Queries MongoDB for the top 10% of runs across MAE / RMSE / R² for manual review. |
| `best_baseline_parameters.json` | Winning config: 3 hidden layers (64, 128, 128), ReLU, L2=1e-4, Nadam, lr=1e-3, converges at epoch 69. |

### Rebuilding the baseline from scratch

```bash
cd baseline/model_search/
python3 fine_tuning.py             # several hours of CPU; requires MongoDB
python3 find_the_best_baseline.py  # query the winner
# Retrain the winner on full 2023 data (produces data/2024/_model_two_layers.keras)
```

The federated runtime expects the resulting keras file at
`data/2024/_model_two_layers.keras`.
