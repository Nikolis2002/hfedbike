#!/usr/bin/env python3
import pandas as pd
import argparse

def main(small_csv, big_csv):
    df_small = pd.read_csv(small_csv)
    df_big   = pd.read_csv(big_csv)
    
    categorical_columns = ['day_of_week', 'month', 'weather_main']
    
    for col in categorical_columns:
        print(f"\n=== Column: {col} ===")
        small_has = col in df_small.columns
        big_has   = col in df_big.columns

        if not small_has and not big_has:
            print("→ neither file has this column.")
            continue
        if not small_has:
            print("→ small CSV:    column not found")
        if not big_has:
            print("→ big CSV:      column not found")
        
        if small_has:
            small_vals = set(df_small[col].dropna().unique())
        else:
            small_vals = set()
        if big_has:
            big_vals   = set(df_big[col].dropna().unique())
        else:
            big_vals = set()

        # helper: normalize floats to ints if whole numbers
        def norm(vals):
            out = []
            for v in vals:
                if isinstance(v, (float,)) and float(v).is_integer():
                    out.append(int(v))
                else:
                    out.append(str(v))
            return sorted(out)

        print(f" in big file   ({len(big_vals)}): {norm(big_vals)}")
        print(f" in small file ({len(small_vals)}): {norm(small_vals)}")
        print(f" missing        ({len(big_vals - small_vals)}): {norm(big_vals - small_vals)}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Compare categorical uniques between two CSVs"
    )
    p.add_argument("small_csv", help="path to the small CSV (e.g. 2-month slice)")
    p.add_argument("big_csv",   help="path to the full CSV")
    args = p.parse_args()
    main(args.small_csv, args.big_csv)
