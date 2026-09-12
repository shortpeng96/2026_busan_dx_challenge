import pandas as pd
import numpy as np
import glob
import os

base_dir = "C:\\Sandbox\\2026_busan_dx_challenge"
proj_dir = os.path.join(base_dir, "Ilgwang_WaterQuality_Project")
master_path = os.path.join(proj_dir, "Data_Processed", "master_dataset_ilgwang.csv")

print("1. Loading Master Dataset to determine date range...")
df_master = pd.read_csv(master_path)
df_master['date'] = pd.to_datetime(df_master['date'])
min_date = df_master['date'].min()
max_date = df_master['date'].max()
print(f"Target Date Range: {min_date.date()} ~ {max_date.date()}")

print("2. Parsing Water Temperature Data (2014~2026)...")
temp_files = glob.glob('C:\\Sandbox\\temp\\기장수온*.xlsx')
df_temps = []
for f in temp_files:
    try:
        tmp = pd.read_excel(f)
        df_temps.append(tmp)
    except: pass
df_temp = pd.concat(df_temps, ignore_index=True)
df_temp['관측일'] = pd.to_datetime(df_temp['관측일'], errors='coerce')
# Filter target dates and '표층' (Surface)
mask_surface = df_temp['수층'] == '표층'
mask_date = (df_temp['관측일'] >= min_date) & (df_temp['관측일'] <= max_date)
df_temp = df_temp[mask_surface & mask_date].copy()

# Keep date and avg temp
df_temp_daily = df_temp[['관측일', '평균수온(°C)']].rename(columns={'관측일': 'date', '평균수온(°C)': 'avg_water_temp'})
df_temp_daily = df_temp_daily.groupby('date').mean().reset_index()
# Generate time-lag
df_temp_daily['avg_water_temp_1d_lag'] = df_temp_daily['avg_water_temp'].shift(1)

print("3. Parsing Tide Data (Busan 1hr)...")
tide_files = glob.glob('C:\\Sandbox\\Tide\\조석성과1시간조위\\부산_*_1시간 조위.txt')
tide_records = []
for f in tide_files:
    with open(f, 'r', encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()
        # Skip header lines, actual data usually starts around line 6
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 3 and parts[0].count('/') == 2:
                date_str, time_str, val_str = parts
                try:
                    tide_records.append({'date': date_str, 'tide_val': float(val_str)})
                except: pass

df_tide = pd.DataFrame(tide_records)
df_tide['date'] = pd.to_datetime(df_tide['date'], format='%Y/%m/%d')
# Calculate daily min, max, and tide_range
df_tide_daily = df_tide.groupby('date')['tide_val'].agg(['max', 'min']).reset_index()
df_tide_daily['tide_range'] = df_tide_daily['max'] - df_tide_daily['min']
df_tide_daily = df_tide_daily[['date', 'tide_range']]
# Generate time-lag
df_tide_daily['tide_range_1d_lag'] = df_tide_daily['tide_range'].shift(1)

print("4. Parsing Wind Data (Gijang)...")
wind_files = glob.glob('C:\\Sandbox\\wind\\OBS_AWS_DD_*.csv')
df_winds = []
for f in wind_files:
    try:
        tmp = pd.read_csv(f, encoding='cp949')
        df_winds.append(tmp)
    except: pass
df_wind = pd.concat(df_winds, ignore_index=True)
df_wind['일시'] = pd.to_datetime(df_wind['일시'])
df_wind = df_wind.rename(columns={'일시': 'date', '최대 순간 풍속 풍향(deg)': 'wind_dir'})

# Vectorize Wind Direction using sin and cos
df_wind['wind_dir_rad'] = np.deg2rad(df_wind['wind_dir'])
df_wind['wind_sin'] = np.sin(df_wind['wind_dir_rad'])
df_wind['wind_cos'] = np.cos(df_wind['wind_dir_rad'])
df_wind_daily = df_wind[['date', 'wind_sin', 'wind_cos']].groupby('date').mean().reset_index()

# Generate time-lag
df_wind_daily['wind_sin_1d_lag'] = df_wind_daily['wind_sin'].shift(1)
df_wind_daily['wind_cos_1d_lag'] = df_wind_daily['wind_cos'].shift(1)

print("5. Merging External Data into Master Dataset...")
# Left join onto master
df_master = pd.merge(df_master, df_temp_daily, on='date', how='left')
df_master = pd.merge(df_master, df_tide_daily, on='date', how='left')
df_master = pd.merge(df_master, df_wind_daily, on='date', how='left')

out_path = os.path.join(proj_dir, "Data_Processed", "master_dataset_ilgwang_v2.csv")
df_master.to_csv(out_path, index=False, encoding='utf-8-sig')

print(f"Merge Complete! Total features: {df_master.shape[1]}")
print(f"Saved to: {out_path}")
