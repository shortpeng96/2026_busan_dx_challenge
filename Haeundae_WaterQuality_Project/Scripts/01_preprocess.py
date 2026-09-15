import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, 'Data_Raw')
PROC = os.path.join(BASE, 'Data_Processed')
os.makedirs(PROC, exist_ok=True)

print("[Haeundae] Preprocessing data from 10 distinct category CSVs...")

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
df_precip = read_raw_csv('해운대_강수.csv')
df_temp = read_raw_csv('해운대_기온.csv')
df_wind = read_raw_csv('해운대_바람.csv')
df_uv = read_raw_csv('해운대_자외선.csv')
df_sewage = read_raw_csv('해운대_인근_하수_방류량.csv')
df_river = read_raw_csv('해운대_인근_하천_방류량.csv')
df_visitor = read_raw_csv('해운대_방문객.csv')
df_tide = read_raw_csv('해운대_조수.csv')
df_water = read_raw_csv('해운대_수질.csv')
df_temp_water = read_raw_csv('해운대_수온.csv')

# 2. Base Date Frame (Use Water Quality as the base target)
if df_water.empty:
    print("  Error: 수질.csv is empty or missing. Cannot build target dataset.")
    exit(1)

master_df = df_water.copy()

def map_location_to_distance(detail):
    d = str(detail).upper()
    if 'A' in d or '조선비치' in d: return 0.0
    elif 'B' in d or '그랜드호텔' in d or '아쿠아리움' in d: return 0.25
    elif 'C' in d: return 0.5
    elif 'D' in d or '파라다이스' in d or '노보텔' in d: return 0.75
    elif 'E' in d or '미포' in d: return 1.0
    return np.nan

# Haeundae-specific Water Quality processing
master_df['distance_from_estuary_km'] = master_df['examinLcDetail'].apply(map_location_to_distance)
master_df = master_df.dropna(subset=['distance_from_estuary_km'])

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
if 'sewage_discharge' in df_sewage.columns:
    df_sewage = df_sewage.rename(columns={'sewage_discharge': 'suyeong_vol'})
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
    feature_table['precip_5d_sum_lag'] = feature_table['precip_daily'].rolling(5).sum().shift(1)

    cso_rain = (feature_table['precip_3d_sum_lag'] >= 20.0)
    feature_table['CSO_Flag_Rain'] = cso_rain.astype(int)

if 'temp_daily' in feature_table.columns:
    feature_table['temp_1d_lag'] = feature_table['temp_daily'].shift(1)

if 'wind_max' in feature_table.columns:
    feature_table['wind_max_1d_lag'] = feature_table['wind_max'].shift(1)

if 'solar_radiation_sum' in feature_table.columns:
    feature_table['solar_radiation_1d_lag'] = feature_table['solar_radiation_sum'].shift(1)

if 'suyeong_vol' in feature_table.columns:
    feature_table['suyeong_vol_1d_lag'] = feature_table['suyeong_vol'].shift(1)
    suyeong_cap = feature_table.groupby('year')['suyeong_vol'].transform(lambda x: np.percentile(x.dropna(), 95) if len(x.dropna())>0 else np.nan)
    feature_table['suyeong_cap'] = suyeong_cap
    if 'precip_3d_sum_lag' in feature_table.columns:
        cso_suyeong = (feature_table['suyeong_vol_1d_lag'] >= feature_table['suyeong_cap']) & (feature_table['precip_3d_sum_lag'] >= 5.0)
        feature_table['CSO_Flag_Suyeong'] = cso_suyeong.astype(int)

if 'visitor_count' in feature_table.columns:
    feature_table['cumulative_visitor_count'] = feature_table.groupby('year')['visitor_count'].cumsum()

# 5. Final Join
master_df = pd.merge(master_df, feature_table, on='date', how='left')

# Drop rows where target is missing
if 'ecoli_max' in master_df.columns:
    master_df = master_df.dropna(subset=['ecoli_max'])

out_path = os.path.join(PROC, "master_dataset.csv")
master_df.to_csv(out_path, index=False, encoding='utf-8-sig')

print(f"Master dataset created with {len(master_df)} rows and {len(master_df.columns)} columns.")
if 'CSO_Flag_Suyeong' in master_df.columns:
    print(f"Total CSO_Suyeong events observed: {master_df['CSO_Flag_Suyeong'].sum()}")
print(f"Saved to {out_path}")
print("[Haeundae] Preprocessing complete!")
