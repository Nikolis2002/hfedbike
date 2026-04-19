# data/2024/

CSVs here are gitignored. Expected contents:

- `fixed_2024.csv` — cleaned OpenWeatherMap hourly table, produced by
  `baseline/data_preparation/make_neighbs_2024/first_fix_for_weather.py`.
- `entire_year/fixed_2024.csv` — same table, mounted path for
  `federated/p2p_node/v2_node.py`.
- `entire_year/region<G>_subzone<s>_bike_usage.csv` — 16 per-subzone
  hourly bike-usage CSVs produced by
  `baseline/data_preparation/make_neighbs_2024/make_the_final_csvsV2.py`.
- `_model_two_layers.keras` — frozen baseline model produced by
  `baseline/model_search/best_baseline_model.py`.
