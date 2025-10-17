import pandas as pd

# Replace these with your actual CSV file paths
csv1_path = '202401-citibike-tripdata.csv'
csv2_path = '202402-citibike-tripdata.csv'

# Load and count rows
rows_csv1 = len(pd.read_csv(csv1_path))
rows_csv2 = len(pd.read_csv(csv2_path))

print(f"Rows in {csv1_path}: {rows_csv1}")
print(f"Rows in {csv2_path}: {rows_csv2}")
print(f"{rows_csv1+rows_csv2}")
