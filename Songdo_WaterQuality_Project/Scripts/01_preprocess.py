"""
01_preprocess.py - Songdo Beach Water Quality
Reads raw data from Data_Raw/ and produces Data_Processed/master_dataset.csv
"""
import pandas as pd
import numpy as np
import os, glob, warnings
warnings.filterwarnings('ignore')

# ── Path configuration ──────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW  = os.path.join(BASE, 'Data_Raw')
PROC = os.path.join(BASE, 'Data_Processed')
os.makedirs(PROC, exist_ok=True)

BEACH = 'Songdo'
ECOLI_THRESHOLD = 500  # CFU/100ml - Korean Beach Safety Standard

def clean_value(val):
    if pd.isna(val): return np.nan
    v = str(val).strip().replace('>', '').replace('<', '').replace(',', '')
    try: return float(v)
    except: return np.nan

def assign_distance(loc):
    loc_str = str(loc).upper().replace(' ', '')
    if any(k in loc_str for k in ['호텔', 'A', '관광']): return 0.0
    elif any(k in loc_str for k in ['스포츠', 'C']): return 1.0
    else: return 0.5

print(f"[{BEACH}] 01_preprocess.py starting...")

# 1. Water Quality
print("  1. Loading water quality data...")
wq_files = glob.glob(os.path.join(RAW, 'water_quality_raw.csv'))
if not wq_files:
    raise FileNotFoundError("No water_quality_raw.csv found in Data_Raw/")
try:
    df_w = pd.read_csv(wq_files[0], encoding='utf-8')
except:
    df_w = pd.read_csv(wq_files[0], encoding='cp949')

if 'examinDe' in df_w.columns:
    df_w['date'] = pd.to_datetime(df_w['examinDe'], errors='coerce')
    df_w['ecoli'] = pd.to_numeric(df_w['coliDetectCn'].astype(str).str.replace('>','').str.replace('<','').str.replace(',',''), errors='coerce')
elif 'sample_date' in df_w.columns:
    df_w['date'] = pd.to_datetime(df_w['sample_date'])
    df_w['ecoli'] = pd.to_numeric(df_w['water01'], errors='coerce')
elif 'date' in df_w.columns:
    df_w['date'] = pd.to_datetime(df_w['date'])
    if 'ecoli' in df_w.columns:
        df_w['ecoli'] = pd.to_numeric(df_w['ecoli'], errors='coerce')
    elif 'ecoli_max' in df_w.columns:
        df_w['ecoli'] = pd.to_numeric(df_w['ecoli_max'], errors='coerce')

df_w = df_w.dropna(subset=['date', 'ecoli'])
if 'distance_to_outfall' not in df_w.columns:
    df_w['distance_to_outfall'] = 0.5

df_w['any_exceed'] = (df_w['ecoli'] >= ECOLI_THRESHOLD).astype(int)
print(f"     Records: {len(df_w)}, Exceed count: {df_w['any_exceed'].sum()}")

# Aggregate to daily max per sampling point
daily_w = df_w.groupby(['date', 'distance_to_outfall']).agg(
    ecoli_max=('ecoli', 'max'),
    any_exceed=('any_exceed', 'max')
).reset_index()

# 2. Weather (Gwangalli station - nearest to Songdo)
print("  2. Loading weather data...")
wea_files = glob.glob(os.path.join(RAW, 'weather_2014_2026.csv'))
if not wea_files:
    wea_files = glob.glob(os.path.join(RAW, '*weather*.csv'))
df_wea_raw = pd.read_csv(wea_files[0], encoding='utf-8-sig')
df_wea_raw['time'] = pd.to_datetime(df_wea_raw['time'])
df_wea_raw['date'] = df_wea_raw['time'].dt.date

df_wea = df_wea_raw.groupby('date').agg(
    precip_daily=('precipitation_mm', 'sum'),
    temp_daily=('temperature_2m', 'mean'),
    wind_speed=('wind_speed_m_s', 'mean'),
    wind_dir=('wind_direction_deg', 'mean'),
).reset_index()
df_wea['date'] = pd.to_datetime(df_wea['date'])

# Lag features
for d in [1, 2, 3, 5, 7]:
    df_wea[f'precip_{d}d_lag'] = df_wea['precip_daily'].shift(d)
df_wea['precip_2d_sum_lag'] = df_wea['precip_daily'].shift(1) + df_wea['precip_daily'].shift(2)
df_wea['precip_3d_sum_lag'] = sum(df_wea['precip_daily'].shift(i) for i in range(1, 4))
df_wea['precip_5d_sum_lag'] = sum(df_wea['precip_daily'].shift(i) for i in range(1, 6))
df_wea['temp_1d_lag'] = df_wea['temp_daily'].shift(1)
df_wea['wind_speed_1d_lag'] = df_wea['wind_speed'].shift(1)
df_wea['wind_dir_1d_lag'] = df_wea['wind_dir'].shift(1)
df_wea['month'] = df_wea['date'].dt.month
df_wea['is_weekend'] = df_wea['date'].dt.dayofweek.isin([5, 6]).astype(int)
df_wea['year'] = df_wea['date'].dt.year

# 3. MEIS Buoy (Geoje - nearest marine station to Songdo)
print("  3. Loading Geoje buoy data...")
buoy_file = os.path.join(RAW, 'meis_buoy_geoje.csv')
if os.path.exists(buoy_file):
    df_buoy = pd.read_csv(buoy_file)
    # Standardize date column
    date_col = [c for c in df_buoy.columns if 'date' in c.lower() or 'time' in c.lower() or '일시' in c]
    if date_col:
        df_buoy['date'] = pd.to_datetime(df_buoy[date_col[0]])
    else:
        df_buoy['date'] = pd.to_datetime(df_buoy.iloc[:, 2])
    df_buoy = df_buoy.rename(columns={c: f'meis_{c.lower().replace(" ", "_")}' for c in df_buoy.columns if c != 'date'})

    # Rolling buoy features
    numeric_buoy = [c for c in df_buoy.columns if c.startswith('meis_') and df_buoy[c].dtype in ['float64', 'int64']]
    for col in numeric_buoy[:5]:  # Top 5 buoy variables
        df_buoy[f'{col}_3d_mean'] = df_buoy[col].shift(1).rolling(3).mean()
        df_buoy[f'{col}_7d_mean'] = df_buoy[col].shift(1).rolling(7).mean()
        df_buoy[f'{col}_5d_sum'] = df_buoy[col].shift(1).rolling(5).sum()

    df_wea = df_wea.merge(df_buoy, on='date', how='left')
    print(f"     Buoy features added: {len([c for c in df_wea.columns if c.startswith('meis_')])}")

# 4. Merge all
print("  4. Merging datasets...")
daily_w['date'] = pd.to_datetime(daily_w['date'])
df_master = daily_w.merge(df_wea, on='date', how='left')

# 5. Feature engineering
df_master['wind_x_distance'] = df_master.get('wind_speed_1d_lag', 0) * df_master['distance_to_outfall']
df_master['cso_x_distance'] = df_master['precip_3d_sum_lag'] * df_master['distance_to_outfall']
df_master['storm_intensity'] = df_master['precip_3d_sum_lag'] * df_master.get('wind_speed', 0)
df_master['CSO_Flag_Rain'] = (df_master['precip_daily'] >= 3.0).astype(int)
df_master['log_ecoli'] = np.log1p(df_master['ecoli_max'])

print(f"  5. Final master dataset: {df_master.shape}")
out_path = os.path.join(PROC, 'master_dataset.csv')
df_master.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f"  Saved: {out_path}")
print(f"[{BEACH}] Preprocessing complete!")
