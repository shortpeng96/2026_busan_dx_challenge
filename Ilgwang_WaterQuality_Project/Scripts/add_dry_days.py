import pandas as pd
import numpy as np
import os

base_dir = "C:\\Sandbox\\2026_busan_dx_challenge"
proj_dir = os.path.join(base_dir, "Ilgwang_WaterQuality_Project")

print("1. Loading Continuous Hourly Weather Data...")
weather_path = os.path.join(proj_dir, "Data_Raw", "ilgwang_weather_2014_2026.csv")
df_weather = pd.read_csv(weather_path)
df_weather['time'] = pd.to_datetime(df_weather['time'])
df_weather['date'] = df_weather['time'].dt.date

# Calculate daily precipitation
df_daily_precip = df_weather.groupby('date')['precipitation_mm'].sum().reset_index()
df_daily_precip = df_daily_precip.sort_values('date')

print("2. Calculating Antecedent Dry Days (ADD)...")
# Calculate consecutive dry days
dry_days = []
current_dry_count = 0

for precip in df_daily_precip['precipitation_mm']:
    if precip == 0:
        current_dry_count += 1
    else:
        # It rained, reset dry count
        current_dry_count = 0
    dry_days.append(current_dry_count)

df_daily_precip['dry_days_count'] = dry_days
# The dry_days_count for TODAY is the number of dry days up to today. 
# But usually, if it rains today, dry days is 0. If we want "how many days before today were dry", 
# we can use shift(1). The user said: "해당 일자 기준으로 비가 오지 않은 연속 일수"
df_daily_precip['dry_days_count'] = df_daily_precip['dry_days_count']
df_daily_precip['date'] = pd.to_datetime(df_daily_precip['date'])

print("3. Merging with Master Dataset...")
master_path = os.path.join(proj_dir, "Data_Processed", "master_dataset_ilgwang_v2.csv")
df_master = pd.read_csv(master_path)
df_master['date'] = pd.to_datetime(df_master['date'])

df_master = pd.merge(df_master, df_daily_precip[['date', 'dry_days_count']], on='date', how='left')

# Save updated master dataset
df_master.to_csv(master_path, index=False, encoding='utf-8-sig')
print("Complete! Added 'dry_days_count' to master_dataset_ilgwang_v2.csv")
