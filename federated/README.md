# Federated runtime

The code that actually runs inside each of the 16 node containers and
the single coordinator container.

## Components

| File | Role |
|---|---|
| `coordinator.py` | Non-computational router. Pools Welford sufficient statistics `(n, μ, σ²)` from all 16 nodes into a global target mean/variance (Step 1), then routes neighborhood leaders' aggregated models between themselves (Step 4). Never averages models itself. Bound to ports 5560 (updates) and 5561 (stats). |
| `p2p_node/v2_node.py` | **The node script. This is what `docker-compose` runs on each node.** Implements the 5-step per-week protocol: local fit → Welford stats → ring reduce → leader↔router exchange → global FedAvg → broadcast. |
| `p2p_node/model.py` | Keras model loader. Reads the frozen baseline from `/data/2024_csvs/_model_two_layers.keras` (in-container path), re-compiles it with Nadam + MSE + MAE/RMSE/R² metrics. |
| `p2p_node/v3_node.py` | Experimental variant. Not used by the current docker-compose. |

## Per-week protocol (implemented in `v2_node.py::main`)

1. **Local fit** — `train_the_model(model, X_val, y_val, X_tr, y_tr, 20)` on that week's 168 hourly samples.
2. **Running statistics (Welford)** — each node sends `(n, μ, σ²)` to the coordinator via `send_z_score()`; receives back the pooled global `(μ, σ)` so everyone z-scores the target on the same scale.
3. **Ring reduce** — `TOTAL_ROUNDS = len(NODE_LIST) - 1` passes of parcel-forwarding with accumulator, producing the exact neighborhood mean.
4. **Inter-neighborhood exchange** — leaders upload their aggregated `w̄^(N)` to the coordinator (`send_update_and_wait_for_peers`), receive the bundle of the other three neighborhoods' models.
5. **Global FedAvg + broadcast** — leader averages across 4 neighborhood models, then uses its `ROUTER` socket to push the result to every follower, which overwrites its weights.

## Env vars read by `v2_node.py`

| Variable | Purpose |
|---|---|
| `NODE_ID` | Uppercase letter identifying this node (e.g. `X`, `P`, `A`, `E`). Leader of each region is the first letter in the `NODE_LIST` of its neighborhood. |
| `NEIGHBORHOOD` | `docker` / `docker2` / `docker3` / `docker4` → selects the 4-node group (region0/1/2/3) and the IP plan. |
| `LOG_DIR`  | Where to write the per-node log. Docker-compose mounts this to `./logs/node<ID>`. |

All three are set per-service in `docker-compose.yml`.

## In-container paths (don't change; they're hard-coded in `v2_node.py`)

| Path | Source volume (set in docker-compose.yml) |
|---|---|
| `/data/2024_csvs/`       | `data/2024/` |
| `/app/`                  | `federated/p2p_node/` |
| `/results/node_results/` | `results/current/` |
| `/logs/`                 | `logs/node<ID>/` |
| `/models/`               | `federated/p2p_node/models/` (only used by node X) |

## Launching

From repo root:

```bash
NODE_SCRIPT=v2_node.py docker compose up -d
```

`NODE_SCRIPT` is substituted into the per-service `command` block so you
can swap in `v3_node.py` without editing the compose file.
