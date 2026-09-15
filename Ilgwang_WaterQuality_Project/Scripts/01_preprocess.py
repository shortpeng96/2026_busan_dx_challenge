import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, 'Data_Raw')
PROC = os.path.join(BASE, 'Data_Processed')
os.makedirs(PROC, exist_ok=True)

print("[Ilgwang] Preprocessing data from 10 distinct category CSVs...")

def read_raw_csv(filename):
    path = os.path.join(RAW, filename)
    if not os.path.exists(path):
        print(f"  Warning: {filename} not found. Returning empty dataframe.")
        return pd.DataFrame(columns=['date'])
    try:
        df = pd.read_csv(path, encoding='utf-8-sig')
    except:
        df = pd.read_csv(path, encoding='cp949', errors='replace')
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
    return df

# 1. Read all 10 files
df_precip = read_raw_csv('일광_강수.csv')
df_temp = read_raw_csv('일광_기온.csv')
df_wind = read_raw_csv('일광_바람.csv')
df_uv = read_raw_csv('일광_자외선.csv')
df_sewage = read_raw_csv('일광_인근_하수_방류량.csv')
df_river = read_raw_csv('일광_인근_하천_방류량.csv')
df_visitor = read_raw_csv('일광_방문객.csv')
df_tide = read_raw_csv('일광_조수.csv')
df_water = read_raw_csv('일광_수질.csv')
df_temp_water = read_raw_csv('일광_부이데이터.csv')

# 2. Base Date Frame (Use Water Quality as the base target)
if df_water.empty:
    print("  Error: 일광_수질.csv is empty or missing. Cannot build target dataset.")
    exit(1)

master_df = df_water.copy()

if 'ecoli_max' in master_df.columns:
    master_df['ecoli_exceed'] = master_df['ecoli_max'] > 500
    master_df['enterococcus_exceed'] = master_df.get('enterococcus_max', 0) > 100
    master_df['any_exceed'] = (master_df['ecoli_exceed'] | master_df['enterococcus_exceed']).astype(int)

# Helper function to merge
def safe_merge(left, right):
    if right.empty or len(right.columns) <= 1:
        return left
    return pd.merge(left, right, on='date', how='left')

# 3. Create a continuous daily timeline to compute lags
min_date = master_df['date'].min()
max_date = master_df['date'].max()
all_dates = pd.DataFrame({'date': pd.date_range(start=min_date, end=max_date)})

# Merge everything onto the continuous timeline
feature_table = all_dates.copy()
feature_table = safe_merge(feature_table, df_precip)
feature_table = safe_merge(feature_table, df_temp)
feature_table = safe_merge(feature_table, df_wind)
feature_table = safe_merge(feature_table, df_uv)
feature_table = safe_merge(feature_table, df_sewage)
feature_table = safe_merge(feature_table, df_river)
feature_table = safe_merge(feature_table, df_visitor)
feature_table = safe_merge(feature_table, df_tide)
feature_table = safe_merge(feature_table, df_temp_water)

feature_table = feature_table.sort_values('date').reset_index(drop=True)
feature_table['year'] = feature_table['date'].dt.year
feature_table['month'] = feature_table['date'].dt.month
feature_table['is_weekend'] = feature_table['date'].dt.dayofweek.isin([5, 6]).astype(int)

# 4. Feature Engineering (Lags and Derived)
if 'precip_daily' in feature_table.columns:
    feature_table['precip_1d_lag'] = feature_table['precip_daily'].shift(1)
    feature_table['precip_2d_sum_lag'] = feature_table['precip_daily'].rolling(2).sum().shift(1)
    feature_table['precip_3d_sum_lag'] = feature_table['precip_daily'].rolling(3).sum().shift(1)
    feature_table['precip_5d_sum_lag'] = feature_table['precip_daily'].rolling(5).sum().shift(1)
    
    feature_table['CSO_Flag_Rain'] = (feature_table['precip_daily'] >= 3.0).astype(int)
    feature_table['dry_days_count'] = (feature_table['precip_daily'] == 0).astype(int).groupby((feature_table['precip_daily'] > 0).cumsum()).cumsum()

if 'temp_daily' in feature_table.columns:
    feature_table['temp_1d_lag'] = feature_table['temp_daily'].shift(1)

if 'wind_max' in feature_table.columns:
    feature_table['wind_max_1d_lag'] = feature_table['wind_max'].shift(1)

if 'solar_radiation_sum' in feature_table.columns:
    feature_table['solar_radiation_1d_lag'] = feature_table['solar_radiation_sum'].shift(1)

if 'gijang_discharge_m3_day' in feature_table.columns:
    feature_table['gijang_discharge_1d_lag'] = feature_table['gijang_discharge_m3_day'].shift(1)
    feature_table['gijang_discharge_3d_mean'] = feature_table['gijang_discharge_m3_day'].rolling(3).mean().shift(1)

if 'jeonggwan_discharge_m3_day' in feature_table.columns:
    feature_table['jeonggwan_discharge_1d_lag'] = feature_table['jeonggwan_discharge_m3_day'].shift(1)
    feature_table['jeonggwan_discharge_3d_mean'] = feature_table['jeonggwan_discharge_m3_day'].rolling(3).mean().shift(1)

if 'tide_max' in feature_table.columns and 'tide_min' in feature_table.columns:
    feature_table['tide_range'] = feature_table['tide_max'] - feature_table['tide_min']
    feature_table['tide_range_1d_lag'] = feature_table['tide_range'].shift(1)

if 'avg_water_temp' in feature_table.columns:
    feature_table['avg_water_temp_1d_lag'] = feature_table['avg_water_temp'].shift(1)

# Shift all buoy metrics by 1 day
for c in feature_table.columns:
    if 'buoy_' in c and '_1d_lag' not in c:
        feature_table[f"{c}_1d_lag"] = feature_table[c].shift(1)

# 5. Final Join
master_df = pd.merge(master_df, feature_table, on='date', how='left')

if 'ecoli_max' in master_df.columns:
    master_df = master_df.dropna(subset=['ecoli_max'])

out_path = os.path.join(PROC, "master_dataset.csv")
master_df.to_csv(out_path, index=False, encoding='utf-8-sig')

print(f"Master dataset created with {len(master_df)} rows and {len(master_df.columns)} columns.")
print(f"Saved to {out_path}")
print("[Ilgwang] Preprocessing complete!")
