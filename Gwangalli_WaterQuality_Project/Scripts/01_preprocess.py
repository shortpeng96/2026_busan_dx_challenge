import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, 'Data_Raw')
PROC = os.path.join(BASE, 'Data_Processed')
os.makedirs(PROC, exist_ok=True)

def clean_microbial_value(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    if not val_str:
        return np.nan
    if '<' in val_str:
        try:
            return float(val_str.replace('<', '')) * 0.5
        except:
            return 0.5
    if '>' in val_str:
        try:
            return float(val_str.replace('>', ''))
        except:
            return np.nan
    val_str = val_str.replace(',', '').replace(' ', '')
    try:
        return float(val_str)
    except:
        return np.nan

def get_distance(row):
    detail = str(row['examinLcDetail']).upper()
    if 'A' in detail: return 0.0
    elif 'B' in detail: return 0.25
    elif 'C' in detail: return 0.5
    elif 'D' in detail: return 0.75
    elif 'E' in detail: return 1.0
    return 0.5 

print("[Gwangalli] Preprocessing data...")

# 1. Water Quality
f_water = os.path.join(RAW, "water_quality_raw.csv")
try:
    df_water = pd.read_csv(f_water, encoding='utf-8-sig')
except:
    df_water = pd.read_csv(f_water, encoding='cp949', errors='replace')

df_water = df_water.dropna(subset=['examinDe']).copy()
df_water = df_water[df_water['examinDe'].str.strip() != '']
df_water['examinDe'] = pd.to_datetime(df_water['examinDe'], errors='coerce')
df_water = df_water.dropna(subset=['examinDe'])
df_water['date'] = pd.to_datetime(df_water['examinDe'].dt.date)

df_water['ecoli_max'] = df_water['coliDetectCn'].apply(clean_microbial_value)
df_water['enterococcus_max'] = df_water['entrcccsDetectCn'].apply(clean_microbial_value)
df_water['ecoli_exceed'] = df_water['ecoli_max'] > 500
df_water['enterococcus_exceed'] = df_water['enterococcus_max'] > 100
df_water['any_exceed'] = (df_water['ecoli_exceed'] | df_water['enterococcus_exceed']).astype(int)
df_water['distance_from_estuary_km'] = df_water.apply(get_distance, axis=1)

cols_to_keep = ['examinDe', 'beachKoreanNm', 'examinLcDetail', 'distance_from_estuary_km', 
                'ecoli_max', 'enterococcus_max', 'any_exceed', 'date']
df_water = df_water[cols_to_keep].copy()

# 2. Weather
f_weather = os.path.join(RAW, "gwangalli_weather_2014_2026.csv")
df_weather = pd.read_csv(f_weather)
df_weather['time'] = pd.to_datetime(df_weather['time'])
df_weather['date'] = df_weather['time'].dt.date

df_weather_daily = df_weather.groupby('date').agg({
    'precipitation_mm': 'sum',
    'temperature_2m': 'mean',
    'wind_speed_m_s': 'max'
}).reset_index()
df_weather_daily['date'] = pd.to_datetime(df_weather_daily['date'])
df_weather_daily.rename(columns={'precipitation_mm': 'precip_daily', 
                                 'temperature_2m': 'temp_daily', 
                                 'wind_speed_m_s': 'wind_max'}, inplace=True)

# 3. Sewage (Suyeong & Nambu)
f_suyeong = os.path.join(RAW, "Discharge", "수영사업단 일일방류량.csv")
df_suyeong = pd.read_csv(f_suyeong, encoding='cp949')
df_suyeong['date'] = pd.to_datetime(df_suyeong.iloc[:, 0].astype(str), format='%Y%m%d')
df_suyeong['suyeong_vol'] = df_suyeong.iloc[:, 1]
df_suyeong = df_suyeong[['date', 'suyeong_vol']]

f_nambu = os.path.join(RAW, "Discharge", "남부사업단 일일방류량.csv")
df_nambu = pd.read_csv(f_nambu, encoding='cp949')
df_nambu['date'] = pd.to_datetime(df_nambu.iloc[:, 0].astype(str), format='%Y%m%d')
df_nambu['nambu_vol'] = df_nambu.iloc[:, 1]
df_nambu = df_nambu[['date', 'nambu_vol']]

df_suyeong['year'] = df_suyeong['date'].dt.year
df_nambu['year'] = df_nambu['date'].dt.year

suyeong_cap = df_suyeong.groupby('year')['suyeong_vol'].apply(lambda x: np.percentile(x.dropna(), 95)).reset_index()
suyeong_cap.rename(columns={'suyeong_vol': 'suyeong_cap'}, inplace=True)

nambu_cap = df_nambu.groupby('year')['nambu_vol'].apply(lambda x: np.percentile(x.dropna(), 95)).reset_index()
nambu_cap.rename(columns={'nambu_vol': 'nambu_cap'}, inplace=True)

# 4. Merge Features Table
feature_table = df_weather_daily.copy()
feature_table = pd.merge(feature_table, df_suyeong[['date', 'suyeong_vol']], on='date', how='left')
feature_table = pd.merge(feature_table, df_nambu[['date', 'nambu_vol']], on='date', how='left')

feature_table['year'] = feature_table['date'].dt.year
feature_table = pd.merge(feature_table, suyeong_cap, on='year', how='left')
feature_table = pd.merge(feature_table, nambu_cap, on='year', how='left')

feature_table = feature_table.sort_values('date').reset_index(drop=True)

# 5. Lagged Features
feature_table['precip_1d_lag'] = feature_table['precip_daily'].shift(1)
feature_table['precip_2d_sum_lag'] = feature_table['precip_daily'].rolling(2).sum().shift(1)
feature_table['precip_3d_sum_lag'] = feature_table['precip_daily'].rolling(3).sum().shift(1)
feature_table['temp_1d_lag'] = feature_table['temp_daily'].shift(1)
feature_table['wind_max_1d_lag'] = feature_table['wind_max'].shift(1)
feature_table['suyeong_vol_1d_lag'] = feature_table['suyeong_vol'].shift(1)
feature_table['nambu_vol_1d_lag'] = feature_table['nambu_vol'].shift(1)

cso_east_cond = (feature_table['suyeong_vol_1d_lag'] >= feature_table['suyeong_cap']) & (feature_table['precip_3d_sum_lag'] >= 5.0)
feature_table['CSO_Flag_East'] = cso_east_cond.astype(int)

cso_west_cond = (feature_table['nambu_vol_1d_lag'] >= feature_table['nambu_cap']) & (feature_table['precip_3d_sum_lag'] >= 5.0)
feature_table['CSO_Flag_West'] = cso_west_cond.astype(int)

# 6. Final Merge
master_df = pd.merge(df_water, feature_table, on='date', how='left')
master_df = master_df.dropna(subset=['ecoli_max'])

out_path = os.path.join(PROC, "master_dataset.csv")
master_df.to_csv(out_path, index=False, encoding='utf-8-sig')

print(f"Master dataset created with {len(master_df)} rows.")
print(f"Total CSO_East events observed: {master_df['CSO_Flag_East'].sum()}")
print(f"Total CSO_West events observed: {master_df['CSO_Flag_West'].sum()}")
print(f"Saved to {out_path}")
print("[Gwangalli] Preprocessing complete!")
