# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, 'Data_Raw')
PROC = os.path.join(BASE, 'Data_Processed')
os.makedirs(PROC, exist_ok=True)

BEACH = 'Songjeong'

def clean_val(v):
    if pd.isna(v): return np.nan
    v_str = str(v).strip()
    if '>' in v_str: v_str = v_str.replace('>', '')
    if '<' in v_str: v_str = v_str.replace('<', '')
    v_str = v_str.replace(',', '')
    try: return float(v_str)
    except: return np.nan

def assign_distance(loc):
    loc_str = str(loc).upper().replace(' ', '')
    # Arbitrary spatial assignment for Songjeong
    if 'A' in loc_str or 'B' in loc_str or '369' in loc_str:
        return 0.0 # South/West side (Closer to Suyeong)
    elif 'D' in loc_str or 'E' in loc_str or '라온' in loc_str or 'LAON' in loc_str:
        return 1.0 # North/East side (Closer to Gijang)
    else:
        return 0.5 # Middle

print(f"[{BEACH}] Preprocessing data...")

# 1. Processing Water Quality Data
print("  1. Loading water quality data...")
f_water = os.path.join(RAW, 'water_quality_raw.csv')
try:
    df_w = pd.read_csv(f_water, encoding='utf-8')
except:
    df_w = pd.read_csv(f_water, encoding='cp949')

if 'examinDe' in df_w.columns:
    df_w = df_w.dropna(subset=['examinDe'])
    df_w['date'] = pd.to_datetime(df_w['examinDe'], errors='coerce')
    df_w = df_w.dropna(subset=['date'])
    df_w['ecoli'] = df_w['coliDetectCn'].apply(clean_val)
    df_w = df_w.dropna(subset=['ecoli'])
    df_w['any_exceed'] = (df_w['ecoli'] >= 500).astype(int)
    df_w['distance_to_outfall'] = df_w['examinLcDetail'].apply(assign_distance)
else:
    raise ValueError("Unexpected water quality format")

df_w = df_w[['date', 'examinLcDetail', 'distance_to_outfall', 'ecoli', 'any_exceed']]
print(f"     Records: {len(df_w)}, Exceed count: {df_w['any_exceed'].sum()}")

# 2. Processing Weather Data
print("  2. Loading weather data...")
df_wea_raw = pd.read_csv(os.path.join(RAW, 'weather_2014_2026.csv'), encoding='utf-8-sig')
df_wea_raw['time'] = pd.to_datetime(df_wea_raw['time'])
df_wea_raw['date'] = df_wea_raw['time'].dt.date

df_wea = df_wea_raw.groupby('date').agg({
    'precipitation_mm': 'sum',
    'temperature_2m': 'mean',
    'wind_speed_m_s': 'mean',
    'wind_direction_deg': 'mean'
}).reset_index()
df_wea['date'] = pd.to_datetime(df_wea['date'])

df_wea['precip_1d_lag'] = df_wea['precipitation_mm'].shift(1)
df_wea['precip_3d_sum_lag'] = df_wea['precipitation_mm'].rolling(3).sum().shift(1)
df_wea['precip_5d_sum_lag'] = df_wea['precipitation_mm'].rolling(5).sum().shift(1)
df_wea['temp_1d_lag'] = df_wea['temperature_2m'].shift(1)
df_wea['temp_daily'] = df_wea['temperature_2m']
df_wea['wind_speed_1d_lag'] = df_wea['wind_speed_m_s'].shift(1)
df_wea['wind_dir_1d_lag'] = df_wea['wind_direction_deg'].shift(1)
df_wea['month'] = df_wea['date'].dt.month
df_wea['is_summer'] = df_wea['month'].isin([7, 8]).astype(int)

# 3. Processing CSO Discharge Data
print("  3. Processing CSO Discharge Data...")
discharge_dir = os.path.join(RAW, 'Discharge')
cso_files = ['수영사업단 일일방류량.csv', '기장사업소 (기장) 일일방류량.csv']
df_cso_list = []
for cso_file in cso_files:
    cso_path = os.path.join(discharge_dir, cso_file)
    if not os.path.exists(cso_path):
        continue
    df_c = pd.read_csv(cso_path, encoding='cp949', skiprows=1, header=None)
    df_c = df_c.iloc[:, :2]
    df_c.columns = ['date', 'discharge']
    df_c = df_c.dropna(subset=['date'])
    df_c['date'] = df_c['date'].astype(str).str.split(' ').str[0]
    df_c = df_c[df_c['date'].str.strip() != '']
    df_c['date'] = pd.to_datetime(df_c['date'], errors='coerce')
    df_c = df_c.dropna(subset=['date'])
    df_c['discharge'] = df_c['discharge'].apply(clean_val)
    df_c = df_c.groupby('date')['discharge'].sum().reset_index()
    cso_name = cso_file.split(' ')[0]
    df_c = df_c.rename(columns={'discharge': f'{cso_name}_vol'})
    df_c[f'{cso_name}_vol_1d_lag'] = df_c[f'{cso_name}_vol'].shift(1)
    df_c[f'{cso_name}_vol_3d_sum_lag'] = df_c[f'{cso_name}_vol'].rolling(3).sum().shift(1)
    df_cso_list.append(df_c)

# 4. Merging Master Dataset
print("  4. Merging datasets...")
df_master = pd.merge(df_w, df_wea, on='date', how='left')
for df_c in df_cso_list:
    df_master = pd.merge(df_master, df_c, on='date', how='left')

df_master = df_master.dropna(subset=['ecoli'])
df_master['log_ecoli'] = np.log1p(df_master['ecoli'])

out_path = os.path.join(PROC, "master_dataset.csv")
df_master.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f"  Final master dataset: {df_master.shape}")
print(f"  Saved: {out_path}")
print(f"[{BEACH}] Preprocessing complete!")
