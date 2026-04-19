# Data

Raw and processed inputs for both the baseline-training pipeline
(`baseline/`) and the federated runtime (`federated/`).

## `2023/`
Raw Citi Bike trip history CSVs for 2023 plus the OpenWeatherMap hourly
weather join, used to train the frozen baseline model. This directory
is the input for `baseline/data_preparation/make_neighbs_2023/` and
`baseline/model_search/`. Gitignored — redownload from
<https://citibikenyc.com/system-data> if it's missing.

## `2024/`

Input for the federated evaluation.

| Path | What it is |
|---|---|
| `2024/fixed_2024.csv`                          | Hourly OpenWeatherMap observations for all of 2024, cleaned and gap-filled |
| `2024/entire_year/fixed_2024.csv`              | Same file, mounted path used by `v2_node.py` |
| `2024/entire_year/region<G>_subzone<s>_bike_usage.csv` | Hourly bike-usage counts per subzone. 16 files total (G∈{0,1,2,3}, s∈{0,1,2,3}). Each node reads exactly one of these inside its container. |
| `2024/first_2_months/`                         | Jan–Feb 2024 subset from early experiments. Kept for regression checks. |
| `2024/_model_two_layers.keras`                 | The frozen baseline model produced by `baseline/model_search/best_baseline_model.py`. Loaded by `federated/p2p_node/model.py` at node startup. Also the starting point for the `Bias` online-adapted baseline. |
| `2024/best_model.keras`, `best_model_fedora.keras` | Older baseline checkpoints, not used by the current runtime. |

## `processed/`

Per-neighborhood hourly bike-usage CSVs produced by older iterations of
the data-prep pipeline (`upper_manhattan_*`, `lower_west_manhattan_*`).
Retained for reproducibility of earlier experiments; the current
16-subzone setup under `2024/entire_year/` supersedes them.

## How the federated runtime finds this data

`docker-compose.yml` mounts the host `data/2024/` directory to the
in-container path `/data/2024_csvs`. The federated scripts hard-code
`/data/2024_csvs/...`, so if you move these directories, update the
compose mounts rather than the Python paths.
