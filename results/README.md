# Results

Per-node, per-week CSVs written by `federated/p2p_node/v2_node.py`. Each
CSV has one header row plus 52 data rows (one per ISO week of 2024).

## Schema

| Column        | Meaning |
|---------------|---------|
| `week`        | ISO week number (1–52) |
| `base_mae`    | MAE of the **bias-adapted baseline** on this week's 168 held-out hours (z-score units) |
| `base_mse`    | MSE of the bias-adapted baseline |
| `freeze_mae`  | MAE of the **frozen centralized baseline** (no online adaptation) |
| `freeze_mse`  | MSE of the frozen baseline |
| `fed_mae`     | MAE of the **federated model** on this node |
| `fed_mse`     | MSE of the federated model |

The MAE values are in **z-score units** (the target is normalized
per-week with Welford statistics pooled across all nodes at the
coordinator). Mean per-subzone σ ≈ 242 bikes/h, so a z-score MAE of
0.46 ≈ 111 bikes/h raw error.

## Layout

```
results/
├── current/      Most recent federated run — post-fix numbers used in the paper
│                 (avg: Freeze 0.648, Bias 0.571, Fed 0.459)
├── pre_fix/      Pre-fix run kept for before/after comparison
│                 (avg: Freeze 0.660, Bias 0.582, Fed 0.474)
└── archive/      Older experimental runs
    ├── baseline_frozen/  Frozen-baseline-only run
    ├── best/             Best-performing pre-fix run we kept
    ├── final/            Pre-presentation snapshot
    ├── one_hard_reset/   Variant: one weekly hard-reset of input stats
    ├── two_hard_reset/   Variant: two weekly hard-resets
    └── theoretical/      Synthetic / debugging run
```

## Regenerating `current/`

```bash
# wipe previous CSVs (written as root inside containers)
docker run --rm -v "$(pwd)/results/current:/r" alpine rm -f /r/*.csv

# launch the federation
NODE_SCRIPT=v2_node.py docker compose up -d

# ~20 min later, every node should be Exited(0) and the CSVs should be
# fresh. Plot:
python3 analysis/test_for_year.py
```

## Comparing runs

`analysis/compare_csvs.py` takes two directories and prints per-column
differences; useful for checking whether a code change actually moved
the numbers.
