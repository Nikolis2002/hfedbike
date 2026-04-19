"""
Prints the top 10% of grid-search runs sorted by Average MAE / RMSE / R2
from the MongoDB collection `citibike.resultsv2`. Used for manual review
of the grid-search output from pre_processing.py. The selected
configuration ends up written by hand into best_baseline_parameters.json
and then retrained in a separate step to produce
data/2024/_model_two_layers.keras.

Does not train anything itself -- pure query script.
"""

from pymongo import MongoClient
import math
import pprint

# 1) connect
client = MongoClient("mongodb://localhost:27017/")
col = client["citibike"]["resultsv2"]

# 2) how big is “top X %”?
PERCENTILE = 0.10
total = col.count_documents({})
top_n = max(1, math.floor(total * PERCENTILE))

print(
    f"Total runs = {total}, showing top {PERCENTILE * 100:.0f}% → top {top_n} entries\n"
)


# 3) helper to fetch & display
def show_top(field, descending=True):
    direction = -1 if descending else 1
    print(
        f"─── Top {top_n} by {'descending' if descending else 'ascending'} `{field}` ───"
    )
    cursor = col.find().sort(field, direction).limit(top_n)
    for doc in cursor:
        print(
            f"Run {doc['_id']:4d} | {field} = {doc[field]:.4f} | MAE = {doc['Average MAE']:.4f} | RMSE = {doc['Average RMSE']:.4f} | R2 = {doc['Average R2']:.4f} | arch: {doc['choosen architecture']}"
        )
    print()


# 4) show for each metric
show_top("Average R2", descending=True)  # highest R2
show_top("Average MAE", descending=False)  # lowest MAE
show_top("Average RMSE", descending=False)  # lowest RMSE
show_top("Average MSE", descending=False)  # lowest MSE
