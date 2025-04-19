from pymongo import MongoClient
import pprint

# 1) Connect
client     = MongoClient("mongodb://localhost:27017/")
db         = client["citibike"]
collection = db["results"]

pp = pprint.PrettyPrinter(indent=2)

def best_in_group(filter_query, metric_field):
    """Return the single doc in `collection` matching filter_query with lowest metric_field."""
    try:
        doc = collection.find(filter_query) \
                        .sort(metric_field, 1) \
                        .limit(1) \
                        .next()
    except StopIteration:
        return None

    return {
      "filter":       filter_query,
      "_id":          doc["_id"],
      "run_id":       doc.get("run_id"),
      "optimizer":    doc.get("optimizer", doc.get("params",{}).get("optimizer")),
      "layers":       doc.get("choosen architecture"),
      "batch_size":   doc.get("b_size", doc.get("params",{}).get("b_size")),
      "use_l1":       doc.get("use L1"),
      "use_l2":       doc.get("use L2"),
      "dropout":      doc.get("dropout_rate", doc.get("params",{}).get("dropout_rate")),
      "lr":           doc.get("lr", doc.get("params",{}).get("learning rate")),
      metric_field:   doc.get(metric_field)
    }

metrics = ["Average MAE", "Average MSE", "Average RMSE"]

# Define your two pure‐regularization filters
pure_l1_filter = {"use L1": True,  "use L2": False}
pure_l2_filter = {"use L1": False, "use L2": True}

print("\n=== Best pure‐L1 runs ===")
for m in metrics:
    best = best_in_group(pure_l1_filter, m)
    print(f"\nMetric: {m}")
    pp.pprint(best)

print("\n=== Best pure‐L2 runs ===")
for m in metrics:
    best = best_in_group(pure_l2_filter, m)
    print(f"\nMetric: {m}")
    pp.pprint(best)
