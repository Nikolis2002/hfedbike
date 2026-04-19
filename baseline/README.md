# Baseline pipeline

Turns raw Citi Bike trip histories + OpenWeatherMap hourly observations
into a frozen neural-network baseline that every federated node loads at
start-up. Not online — this is the offline pre-training side of the
project.

## `data_preparation/`

Produces the per-subzone, per-hour demand CSVs that the federated nodes
consume.

### `make_neighbs_2023/` (one-time, built the 2023 baseline training set)
| Script | Role |
|---|---|
| `store_raw.py`     | Downloads / unzips the monthly Citi Bike trip history CSVs |
| `csv_checker.py`   | Sanity-checks headers and row counts across months |
| `index_creator.py` | Builds a station-id → (lat, lon) map |
| `data_aggregator.py` | Hourly trip-count aggregation per station |
| `data_enrich.py`   | Joins the hourly trip counts with weather; outputs the 2023 training table |

### `make_neighbs_2024/` (built the 2024 evaluation set)
| Script | Role |
|---|---|
| `first_fix_for_weather.py`       | Cleans OpenWeatherMap 2024 extract |
| `first_two_months_weather_select.py` | Legacy subset for early Jan–Feb tests |
| `count.py`                       | Station → k-means cluster assignment (k=4 regions, k=4 subzones within each) |
| `make_the_final_csvsV2.py`       | Current: per-subzone hourly CSVs + `fixed_2024.csv` weather table, written to `data/2024/entire_year/` |
| `make_the_final_csvs.py`         | Older version, kept for reference |

Run order (2024): `first_fix_for_weather.py` → `count.py` →
`make_the_final_csvsV2.py`.

## `model_search/`

Picks the architecture and hyperparameters used as the frozen baseline.

| File | What it does |
|---|---|
| `pre_processing.py` | Offline feature engineering: cyclic hour features, Z-score for numeric weather, one-hot for day-of-week/month/weather_main. Matches the feature layout used by `federated/p2p_node/v2_node.py` at runtime. |
| `baseline_model.py` | Keras model factory (feedforward, 2–3 hidden layers) + `train_the_model()` helper |
| `find_the_best_baseline.py` | Grid search driver. 1,734 configurations × 5-fold month-stratified CV on the 2023 table. Writes progress to `completed_runsv2.txt` and the winning config to `best_baseline_parameters.json`. |
| `best_baseline_parameters.json` | The winning config: 3 hidden layers (64, 128, 128), ReLU, L2=1e-4, Nadam, converges at epoch 69. |
| `best_baseline_model.py` | Retrains the winning config on the full 2023 table and serializes the model. The output is copied to `data/2024/_model_two_layers.keras` for the federated runtime to load. |

### Rebuilding the baseline from scratch

```bash
cd baseline/model_search/
python3 find_the_best_baseline.py     # several hours of CPU
python3 best_baseline_model.py        # final training of the chosen config
```

The federated runtime expects the resulting keras file at
`data/2024/_model_two_layers.keras`.
