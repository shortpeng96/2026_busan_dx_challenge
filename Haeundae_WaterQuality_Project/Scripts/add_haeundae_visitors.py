import pandas as pd
import numpy as np
import os

proj_dir = "C:\\Sandbox\\2026_busan_dx_challenge\\Haeundae_WaterQuality_Project"
visitor_path = os.path.join(proj_dir, "Data_Raw", "해운대해수욕장_방문객_2019_2025.csv")
master_path = os.path.join(proj_dir, "Data_Processed", "master_dataset_haeundae.csv")

# 1. Load visitor data
df_vis = pd.read_csv(visitor_path, encoding='utf-8-sig')

# Clean visitor count (remove commas and convert to float)
# The column name is '방문객수(명)', date is '방문일'
df_vis['visitor_count'] = df_vis['방문객수(명)'].astype(str).str.replace(',', '').astype(float)
df_vis['date'] = pd.to_datetime(df_vis['방문일'])
df_vis['year'] = df_vis['date'].dt.year

df_vis_daily = df_vis[['date', 'year', 'visitor_count']].groupby(['date', 'year']).sum().reset_index()

# Calculate cumulative visitor count per year
df_vis_daily['cumulative_visitor_count'] = df_vis_daily.groupby('year')['visitor_count'].cumsum()

# Drop the year column as we just needed it for grouping
df_vis_daily = df_vis_daily[['date', 'visitor_count', 'cumulative_visitor_count']]

# 2. Load master dataset
df_master = pd.read_csv(master_path, encoding='utf-8-sig')
df_master['date'] = pd.to_datetime(df_master['date'])

# 3. Merge
df_master = pd.merge(df_master, df_vis_daily, on='date', how='left')

# Save
df_master.to_csv(master_path, index=False, encoding='utf-8-sig')
print("Successfully merged Haeundae visitors into master_dataset_haeundae.csv")
