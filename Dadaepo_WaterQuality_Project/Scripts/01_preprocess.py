import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, 'Data_Raw')
PROC = os.path.join(BASE, 'Data_Processed')
os.makedirs(PROC, exist_ok=True)

print("[Dadaepo] Preprocessing data from 10 distinct category CSVs...")

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
df_precip = read_raw_csv('다대포_강수.csv')
df_temp = read_raw_csv('다대포_기온.csv')
df_wind = read_raw_csv('다대포_바람.csv')
df_uv = read_raw_csv('다대포_자외선.csv')
df_sewage = read_raw_csv('다대포_인근_하수_방류량.csv')
df_river = read_raw_csv('다대포_인근_하천_방류량.csv')
df_visitor = read_raw_csv('다대포_방문객.csv')
df_tide = read_raw_csv('다대포_조수.csv')
df_water = read_raw_csv('다대포_수질.csv')
df_temp_water = read_raw_csv('다대포_수온.csv')

# 2. Base Date Frame (Use Water Quality as the base target)
if df_water.empty:
    print("  Error: 수질.csv is empty or missing. Cannot build target dataset.")
    exit(1)

master_df = df_water.copy()

def assign_distance(loc):
    loc_str = str(loc).upper().replace(' ', '')
    if '좌측' in loc_str: return 0.2
    elif '우측' in loc_str: return 0.8
    else: return 0.5

# Dadaepo-specific Water Quality columns
if 'ecoli_max' in master_df.columns:
    master_df.rename(columns={'ecoli_max': 'ecoli'}, inplace=True)
    master_df['any_exceed'] = (master_df['ecoli'] >= 500).astype(int)
    master_df['distance_to_outfall'] = master_df['examinLcDetail'].apply(assign_distance)
    master_df['log_ecoli'] = np.log1p(master_df['ecoli'])

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

# 4. Feature Engineering (Lags and Derived)
if 'precip_daily' in feature_table.columns:
    feature_table['precip_1d_lag'] = feature_table['precip_daily'].shift(1)
    feature_table['precip_2d_sum_lag'] = feature_table['precip_daily'].rolling(2).sum().shift(1)
    feature_table['precip_3d_sum_lag'] = feature_table['precip_daily'].rolling(3).sum().shift(1)

if 'temp_daily' in feature_table.columns:
    feature_table['temp_1d_lag'] = feature_table['temp_daily'].shift(1)

if 'wind_max' in feature_table.columns:
    feature_table['wind_max_1d_lag'] = feature_table['wind_max'].shift(1)

if 'solar_radiation_sum' in feature_table.columns:
    feature_table['solar_radiation_1d_lag'] = feature_table['solar_radiation_sum'].shift(1)

if 'sewage_discharge' in feature_table.columns:
    feature_table['sewage_discharge_1d_lag'] = feature_table['sewage_discharge'].shift(1)
    feature_table['sewage_discharge_3d_sum_lag'] = feature_table['sewage_discharge'].rolling(3).sum().shift(1)
    if 'precip_3d_sum_lag' in feature_table.columns:
        cso_cond = (feature_table['sewage_discharge_1d_lag'] >= 450000) & (feature_table['precip_3d_sum_lag'] >= 5.0)
        feature_table['CSO_Flag'] = cso_cond.astype(int)

if 'river_discharge' in feature_table.columns:
    feature_table['discharge_1d_lag'] = feature_table['river_discharge'].shift(1)
    feature_table['discharge_3d_sum_lag'] = feature_table['river_discharge'].rolling(3).sum().shift(1)

if 'tide_max' in feature_table.columns and 'tide_min' in feature_table.columns:
    feature_table['tide_range'] = feature_table['tide_max'] - feature_table['tide_min']
    feature_table['tide_range_1d_lag'] = feature_table['tide_range'].shift(1)

# Sensor and Visitor lags
lag_cols = [c for c in df_temp_water.columns if c != 'date'] + ['visitor_count']
for col in lag_cols:
    if col in feature_table.columns:
        feature_table[f"{col}_1d_lag"] = feature_table[col].shift(1)

# 5. Final Join
master_df = pd.merge(master_df, feature_table, on='date', how='left')

# Drop rows where target is missing
if 'ecoli' in master_df.columns:
    master_df = master_df.dropna(subset=['ecoli'])

out_path = os.path.join(PROC, "master_dataset_v2.csv")
master_df.to_csv(out_path, index=False, encoding='utf-8-sig')

print(f"Master dataset created with {len(master_df)} rows and {len(master_df.columns)} columns.")
if 'CSO_Flag' in master_df.columns:
    print(f"Total CSO events observed: {master_df['CSO_Flag'].sum()}")
print(f"Saved to {out_path}")
print("[Dadaepo] Preprocessing complete!")
