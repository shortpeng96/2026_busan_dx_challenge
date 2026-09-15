# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import glob

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, 'Data_Raw')
PROC = os.path.join(BASE, 'Data_Processed')
os.makedirs(PROC, exist_ok=True)

def clean_val(v):
    if pd.isna(v): return np.nan
    v_str = str(v).replace(',', '').replace('<', '').replace('>', '').replace(' ', '')
    try: return float(v_str)
    except: return np.nan

def map_location_to_distance(detail):
    d = str(detail).upper()
    if 'A' in d or '조선비치' in d: return 0.0
    elif 'B' in d or '그랜드호텔' in d or '아쿠아리움' in d: return 0.25
    elif 'C' in d: return 0.5
    elif 'D' in d or '파라다이스' in d or '노보텔' in d: return 0.75
    elif 'E' in d or '미포' in d: return 1.0
    return np.nan

print("[Haeundae] Preprocessing data...")

# 1. Water Quality Samples
f_path = os.path.join(RAW, "water_quality_raw.csv")
try:
    df_water = pd.read_csv(f_path, encoding='utf-8-sig')
except:
    df_water = pd.read_csv(f_path, encoding='cp949', errors='replace')
    
col_map = {
    '조사일자': 'examinDe',
    '해수욕장명': 'beachKoreanNm',
    '조사위치상세': 'examinLcDetail',
    '대장균': 'coliDetectCn',
    '장구균': 'entrcccsDetectCn'
}
if 'examinDe' not in df_water.columns:
    df_water.rename(columns=col_map, inplace=True)
    
df_water = df_water[['examinDe', 'beachKoreanNm', 'examinLcDetail', 'coliDetectCn', 'entrcccsDetectCn']].copy()
df_water['examinDe'] = pd.to_datetime(df_water['examinDe'].astype(str).str.replace('-', '').str.replace('.', '').str[:8], format='%Y%m%d', errors='coerce')
df_water = df_water.dropna(subset=['examinDe'])
df_water['date'] = pd.to_datetime(df_water['examinDe'].dt.date)

df_water['coliDetectCn'] = df_water['coliDetectCn'].apply(clean_val)
df_water['entrcccsDetectCn'] = df_water['entrcccsDetectCn'].apply(clean_val)
df_water['distance_from_estuary_km'] = df_water['examinLcDetail'].apply(map_location_to_distance)
df_water = df_water.dropna(subset=['distance_from_estuary_km'])

df_grouped = df_water.groupby(['date', 'examinDe', 'beachKoreanNm', 'examinLcDetail', 'distance_from_estuary_km']).agg({
    'coliDetectCn': 'max',
    'entrcccsDetectCn': 'max'
}).reset_index()

df_grouped.rename(columns={'coliDetectCn': 'ecoli_max', 'entrcccsDetectCn': 'enterococcus_max'}, inplace=True)
df_grouped['ecoli_exceed'] = df_grouped['ecoli_max'] > 500
df_grouped['enterococcus_exceed'] = df_grouped['enterococcus_max'] > 100
df_grouped['any_exceed'] = (df_grouped['ecoli_exceed'] | df_grouped['enterococcus_exceed']).astype(int)

# 2. Weather
f_weather = os.path.join(RAW, "haeundae_weather_2014_2026.csv")
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

# 3. Sewage (Suyeong)
f_suyeong = os.path.join(RAW, "Discharge", "수영사업단 일일방류량.csv")
df_suyeong = pd.read_csv(f_suyeong, encoding='cp949')
df_suyeong['date'] = pd.to_datetime(df_suyeong.iloc[:, 0].astype(str), format='%Y%m%d')
df_suyeong['suyeong_vol'] = df_suyeong.iloc[:, 1]
df_suyeong = df_suyeong[['date', 'suyeong_vol']]

df_suyeong['year'] = df_suyeong['date'].dt.year
suyeong_cap = df_suyeong.groupby('year')['suyeong_vol'].apply(lambda x: np.percentile(x.dropna(), 95)).reset_index()
suyeong_cap.rename(columns={'suyeong_vol': 'suyeong_cap'}, inplace=True)

# 4. Visitors
visitor_files = [os.path.join(RAW, "Visitors", "visitors_20240521.csv")]
df_visitors_list = []
for f in visitor_files:
    v = pd.read_csv(f, encoding='utf-8-sig')
    v = v[v['해수욕장이름'].astype(str).str.contains('해운대', na=False)]
    v['date'] = pd.to_datetime(v['해수욕장일일일자']).dt.strftime('%Y-%m-%d')
    v = v.rename(columns={'해수욕장일일사용자수': 'visitor_count'})
    v = v[['date', 'visitor_count']]
    df_visitors_list.append(v)

if df_visitors_list:
    df_visitor = pd.concat(df_visitors_list, ignore_index=True)
    df_visitor['date'] = pd.to_datetime(df_visitor['date'])
    df_visitor = df_visitor.sort_values('date').drop_duplicates(subset=['date'])
    df_visitor['year'] = df_visitor['date'].dt.year
    df_visitor['cumulative_visitor_count'] = df_visitor.groupby('year')['visitor_count'].cumsum()
    df_visitor = df_visitor.drop(columns=['year'])
else:
    df_visitor = pd.DataFrame(columns=['date', 'visitor_count', 'cumulative_visitor_count'])

# 5. Merge Features Table
feature_table = df_weather_daily.copy()
feature_table = pd.merge(feature_table, df_suyeong[['date', 'suyeong_vol']], on='date', how='left')
feature_table = pd.merge(feature_table, df_visitor, on='date', how='left')

feature_table['year'] = feature_table['date'].dt.year
feature_table = pd.merge(feature_table, suyeong_cap, on='year', how='left')
feature_table = feature_table.sort_values('date').reset_index(drop=True)

# Generate Lagged Features
feature_table['precip_1d_lag'] = feature_table['precip_daily'].shift(1)
feature_table['precip_2d_sum_lag'] = feature_table['precip_daily'].rolling(2).sum().shift(1)
feature_table['precip_3d_sum_lag'] = feature_table['precip_daily'].rolling(3).sum().shift(1)
feature_table['precip_5d_sum_lag'] = feature_table['precip_daily'].rolling(5).sum().shift(1)
feature_table['temp_1d_lag'] = feature_table['temp_daily'].shift(1)
feature_table['wind_max_1d_lag'] = feature_table['wind_max'].shift(1)
feature_table['suyeong_vol_1d_lag'] = feature_table['suyeong_vol'].shift(1)

# Dual CSO Flags
cso_suyeong = (feature_table['suyeong_vol_1d_lag'] >= feature_table['suyeong_cap']) & (feature_table['precip_3d_sum_lag'] >= 5.0)
feature_table['CSO_Flag_Suyeong'] = cso_suyeong.astype(int)

cso_rain = (feature_table['precip_3d_sum_lag'] >= 20.0)
feature_table['CSO_Flag_Rain'] = cso_rain.astype(int)

# 6. Final Merge
master_df = pd.merge(df_grouped, feature_table, on='date', how='left')
master_df = master_df.dropna(subset=['ecoli_max'])

out_path = os.path.join(PROC, "master_dataset.csv")
master_df.to_csv(out_path, index=False, encoding='utf-8-sig')

print(f"Master dataset created with {len(master_df)} rows.")
print(f"Total CSO_Suyeong events observed: {master_df['CSO_Flag_Suyeong'].sum()}")
print(f"Saved to {out_path}")
print("[Haeundae] Preprocessing complete!")
