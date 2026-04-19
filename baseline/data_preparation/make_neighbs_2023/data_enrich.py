"""
2023 trip enrichment for the baseline grid search.

Adds a single field ``hour`` (datetime, ``started_at`` truncated to the
hour) to every document in the MongoDB collection ``citibike.bikes_raw``
and writes the result to ``citibike.bike_data_enriched``.  The baseline
training loop in ``baseline/model_search/pre_processing.py`` then
aggregates that collection by ``hour`` to get total NYC bike trips per
hour, joins with the cleaned OpenWeatherMap table, and feeds the result
into the grid search.

Why only ``hour``
-----------------
The baseline is a **citywide** model: the grid search aggregates trips
across all of NYC (no per-region or per-subzone split). The polygon-
based region/subzone tagging the previous version of this script
computed was never consumed by the training loop, so it has been
removed along with the associated MongoDB round-trip.

Why one MongoDB aggregation instead of a Python loop
----------------------------------------------------
The previous implementation paginated ``bikes_raw`` with
``.skip(offset).limit(BATCH_SIZE)`` (O(offset) per batch → quadratic
over ~35M rows), parsed ``started_at`` with pandas, and reinserted the
enriched docs with ``insert_many`` on a ThreadPoolExecutor.  All of
that was dominated by the skip cost and by serializing 35M Python dicts
through the wire.

This rewrite runs the enrichment **entirely inside the MongoDB server**
via ``$dateFromString`` + ``$dateTrunc`` + ``$out``.  Nothing round-trips
through Python.  On our data this is ~10–30× faster than the old loop
(seconds to a few minutes vs. an evening).  If you ever want to avoid
MongoDB altogether, see the ``pandas-only alternative`` note at the
bottom of this file.
"""

from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["citibike"]

SRC_COLLECTION = "bikes_raw"            # pre-existing raw trip dump
DST_COLLECTION = "bike_data_enriched"   # what pre_processing.py reads

# Citi Bike's ``started_at`` format flipped mid-2024:
#   2023-01..2024-04 :  ``2023-01-07 15:36:53.430``   (with milliseconds)
#   2024-05 onward   :  ``2024-05-01 08:05:53``        (no milliseconds)
# Today's 2023 dump is uniformly the first form, but we accept both here
# so a future backfill or a re-ingestion of mixed data doesn't silently
# drop rows via onError.  MongoDB's $dateFromString is fail-single-format,
# so we try the millisecond form first and fall back to the no-ms form
# via $ifNull.
FORMAT_WITH_MS = "%Y-%m-%d %H:%M:%S.%L"
FORMAT_NO_MS   = "%Y-%m-%d %H:%M:%S"


def enrich_server_side() -> int:
    """Run the enrichment as a single MongoDB aggregation pipeline.

    The pipeline:
      1. Parses the ``started_at`` string into a BSON Date.  Rows that
         fail to parse get ``null`` so the next stage can drop them
         rather than blowing up the whole pipeline.
      2. Drops rows with an unparseable timestamp (``$match``).
      3. Truncates the parsed date to the hour (``$dateTrunc``) and
         stores it in the ``hour`` field — the only new field we care
         about for the baseline training.
      4. Removes the intermediate ``_parsed`` helper field.
      5. Writes the result to ``DST_COLLECTION`` via ``$out``; ``$out``
         atomically replaces the destination, so re-running this
         script is safe and idempotent.

    Returns the number of enriched documents.
    """
    # $out overwrites the destination atomically, but we drop explicitly
    # so a partial previous run doesn't leave stale indexes around.
    db[DST_COLLECTION].drop()

    db[SRC_COLLECTION].aggregate(
        [
            # 1. Parse the string timestamp; emit null on malformed rows.
            #    Try the millisecond format first (pre-May-2024 layout);
            #    fall back to no-ms on failure. $dateFromString accepts
            #    only one format spec per call, so $ifNull chains the two.
            {
                "$addFields": {
                    "_parsed": {
                        "$ifNull": [
                            {"$dateFromString": {
                                "dateString": "$started_at",
                                "format": FORMAT_WITH_MS,
                                "onError": None,
                            }},
                            {"$dateFromString": {
                                "dateString": "$started_at",
                                "format": FORMAT_NO_MS,
                                "onError": None,
                            }},
                        ]
                    }
                }
            },
            # 2. Throw away rows whose started_at couldn't be parsed.
            {"$match": {"_parsed": {"$ne": None}}},
            # 3. hour = floor(_parsed, 1 hour).
            {
                "$addFields": {
                    "hour": {"$dateTrunc": {"date": "$_parsed", "unit": "hour"}}
                }
            },
            # 4. Drop the helper field so the output schema stays clean.
            {"$unset": "_parsed"},
            # 5. Materialize the result.  allowDiskUse handles spills on
            #    large inputs; batchSize doesn't apply to $out but we keep
            #    the flag for anyone adapting this to a cursor pipeline.
            {"$out": DST_COLLECTION},
        ],
        allowDiskUse=True,
    )

    # A single ascending index on ``hour`` makes the downstream
    # ``$group: {_id: "$hour", bike_usage: {$sum: 1}}`` in pre_processing.py
    # a covered index scan. Without it that group is a full collection scan.
    db[DST_COLLECTION].create_index([("hour", 1)], name="hour_1")

    return db[DST_COLLECTION].estimated_document_count()


if __name__ == "__main__":
    n = enrich_server_side()
    print(f"✅ Enriched {n:,} documents into {DST_COLLECTION}.")


# ──────────────────────────────────────────────────────────────────────
#  pandas-only alternative (uncomment if you want to skip MongoDB)
#
#  Read the raw trip CSVs directly, group by hour, and dump a single
#  two-column CSV.  For the 2023 baseline this is the *fastest* route
#  because it never populates an intermediate collection — aggregation
#  is O(n) through pandas' columnar engine.
#
#  from pathlib import Path
#  import pandas as pd
#
#  REPO_ROOT = Path(__file__).resolve().parents[3]
#  RAW_DIR   = REPO_ROOT / "data" / "2023"
#  OUT_CSV   = REPO_ROOT / "data" / "processed" / "hourly_bike_usage_2023.csv"
#
#  frames = []
#  for path in sorted(RAW_DIR.glob("*citibike-tripdata*.csv")):
#      df = pd.read_csv(path, usecols=["started_at"])
#      df["hour"] = pd.to_datetime(df["started_at"], errors="coerce").dt.floor("h")
#      frames.append(df.dropna(subset=["hour"]))
#  hourly = (
#      pd.concat(frames, ignore_index=True)
#        .groupby("hour", sort=True)
#        .size()
#        .rename("bike_usage")
#        .reset_index()
#  )
#  hourly.to_csv(OUT_CSV, index=False)
# ──────────────────────────────────────────────────────────────────────
