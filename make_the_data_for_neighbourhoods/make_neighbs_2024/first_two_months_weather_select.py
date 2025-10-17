import pandas as pd

df=pd.read_csv("fixed_2024.csv")

df['hour'] = pd.to_datetime(df['hour'])
# 2a) Filter by year/month
df_jan_feb = df[
    (df['hour'].dt.year  == 2024) &
    (df['hour'].dt.month <= 2)
]

# Now df_jan_feb contains only rows from Jan & Feb 2024
import pandas as pd

# 1) Make sure 'hour' is datetime
df['hour'] = pd.to_datetime(df['hour'])

# 2a) Filter by year/month
df_jan_feb = df[
    (df['hour'].dt.year  == 2024) &
    (df['hour'].dt.month <= 2)
]

# Now df_jan_feb contains only rows from Jan & Feb 2024
df_jan_feb.to_csv("fedJan_2024.csv",index=False)