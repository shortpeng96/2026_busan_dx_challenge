"""
01_preprocess.py - Imrang Beach Water Quality
Final verified AUC: 0.873
"""
import pandas as pd
import numpy as np
import os, glob, warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW  = os.path.join(BASE, 'Data_Raw')
PROC = os.path.join(BASE, 'Data_Processed')
os.makedirs(PROC, exist_ok=True)

BEACH = 'Imrang'
ECOLI_THRESHOLD = 500
ENTERO_THRESHOLD = 200  # 장구균 기준 (임랑은 장구균 초과가 주요 지표)

print(f"[{BEACH}] 01_preprocess.py starting...")

# 1. Water Quality
print("  1. Loading water quality data...")
wq_files = glob.glob(os.path.join(RAW, 'water_quality_raw.csv'))
if not wq_files:
    raise FileNotFoundError("No water_quality_raw.csv in Data_Raw/")
try:
    df_w = pd.read_csv(wq_files[0], encoding='utf-8')
except:
    df_w = pd.read_csv(wq_files[0], encoding='cp949')

# Handle possible different column structures
if 'examinDe' in df_w.columns:
    df_w['date'] = pd.to_datetime(df_w['examinDe'], errors='coerce')
    df_w['ecoli'] = pd.to_numeric(df_w['coliDetectCn'].astype(str).str.replace('>','').str.replace('<','').str.replace(',',''), errors='coerce')
    df_w['enterococcus'] = np.nan
elif 'sample_date' in df_w.columns:
    df_w['date'] = pd.to_datetime(df_w['sample_date'])
    df_w['ecoli'] = pd.to_numeric(df_w['water01'], errors='coerce')
    # water02 = 장구균(enterococcus)
    df_w['enterococcus'] = pd.to_numeric(df_w['water02'], errors='coerce') if 'water02' in df_w.columns else np.nan
elif 'date' in df_w.columns:
    df_w['date'] = pd.to_datetime(df_w['date'])
    if 'ecoli' in df_w.columns:
        df_w['ecoli'] = pd.to_numeric(df_w['ecoli'], errors='coerce')
    elif 'ecoli_max' in df_w.columns:
        df_w['ecoli'] = pd.to_numeric(df_w['ecoli_max'], errors='coerce')
    df_w['enterococcus'] = pd.to_numeric(df_w.get('enterococcus', np.nan), errors='coerce')

df_w = df_w.dropna(subset=['date', 'ecoli'])
if 'distance_to_outfall' not in df_w.columns:
    df_w['distance_to_outfall'] = 0.5

# any_exceed: 대장균 >=500 OR 장구균 >=200 (임랑은 장구균 초과가 지배적)
df_w['ecoli_exceed'] = (df_w['ecoli'] >= ECOLI_THRESHOLD).astype(int)
df_w['entero_exceed'] = (df_w['enterococcus'] >= ENTERO_THRESHOLD).fillna(0).astype(int)
df_w['any_exceed'] = ((df_w['ecoli_exceed'] == 1) | (df_w['entero_exceed'] == 1)).astype(int)

# Distance feature (if available)
if 'examinLcDetail' in df_w.columns:
    loc_map = {'A': 0.0, 'B': 0.5, 'C': 1.0}
    df_w['distance_to_outfall'] = df_w['examinLcDetail'].astype(str).str[:1].map(loc_map).fillna(0.5)
else:
    df_w['distance_to_outfall'] = 0.5

daily_w = df_w.groupby(['date']).agg(
    ecoli_max=('ecoli', 'max'),
    enterococcus_max=('enterococcus', 'max'),
    any_exceed=('any_exceed', 'max'),
    distance_to_outfall=('distance_to_outfall', 'mean'),
).reset_index()
daily_w['date'] = pd.to_datetime(daily_w['date'])
print(f"     Records: {len(daily_w)}, Exceed count: {daily_w['any_exceed'].sum()} ({daily_w['any_exceed'].mean():.1%})")

# 2. Weather
print("  2. Loading weather data...")
wea_files = glob.glob(os.path.join(RAW, 'weather_2014_2026.csv')) or glob.glob(os.path.join(RAW, '*weather*.csv'))
df_wea_raw = pd.read_csv(wea_files[0], encoding='utf-8-sig')
if 'time' in df_wea_raw.columns:
    df_wea_raw['date'] = pd.to_datetime(df_wea_raw['time']).dt.date
elif 'date' in df_wea_raw.columns:
    df_wea_raw['date'] = pd.to_datetime(df_wea_raw['date']).dt.date

prec_col = [c for c in df_wea_raw.columns if 'prec' in c.lower() or 'rain' in c.lower()][0]
temp_col = [c for c in df_wea_raw.columns if 'temp' in c.lower()][0]
wind_col = [c for c in df_wea_raw.columns if 'wind' in c.lower() and 'speed' in c.lower()]

agg_dict = {prec_col: 'sum', temp_col: 'mean'}
if wind_col:
    agg_dict[wind_col[0]] = 'mean'

df_wea = df_wea_raw.groupby('date').agg(agg_dict).reset_index()
df_wea = df_wea.rename(columns={prec_col: 'precip_daily', temp_col: 'temp_daily'})
if wind_col:
    df_wea = df_wea.rename(columns={wind_col[0]: 'wind_speed'})
df_wea['date'] = pd.to_datetime(df_wea['date'])

# Lag features
df_wea['precip_1d_lag'] = df_wea['precip_daily'].shift(1)
df_wea['precip_2d_sum_lag'] = df_wea['precip_daily'].shift(1) + df_wea['precip_daily'].shift(2)
df_wea['precip_3d_sum_lag'] = sum(df_wea['precip_daily'].shift(i) for i in range(1, 4))
df_wea['precip_5d_sum_lag'] = sum(df_wea['precip_daily'].shift(i) for i in range(1, 6))
df_wea['temp_1d_lag'] = df_wea['temp_daily'].shift(1)
df_wea['month'] = df_wea['date'].dt.month
df_wea['year'] = df_wea['date'].dt.year
df_wea['is_weekend'] = df_wea['date'].dt.dayofweek.isin([5, 6]).astype(int)
df_wea['CSO_Flag_Rain'] = (df_wea['precip_daily'] >= 3.0).astype(int)
df_wea['dry_days_count'] = (df_wea['precip_daily'] == 0).astype(int).groupby(
    (df_wea['precip_daily'] > 0).cumsum()).cumsum()

# 3. MEIS Buoy (Ulsan - nearest to Imrang/Northeast coast)
# 3. MEIS Buoy (Ulsan & Gijang)
print("  3. Loading Ulsan & Gijang buoy data...")
import glob
for b_file in glob.glob(os.path.join(RAW, 'meis_buoy_*.csv')):
    fname = os.path.basename(b_file)
    try: df_buoy = pd.read_csv(b_file, encoding='utf-8-sig')
    except: df_buoy = pd.read_csv(b_file, encoding='cp949')
    
    date_col = [c for c in df_buoy.columns if '일시' in c or 'date' in c.lower() or 'time' in c.lower()]
    df_buoy['date'] = pd.to_datetime(df_buoy[date_col[0]] if date_col else df_buoy.iloc[:, 2])
    df_buoy['date'] = df_buoy['date'].dt.normalize()
    
    prefix = 'gijang_' if 'gijang' in fname else 'meis_'
    for c in df_buoy.columns:
        if c not in ['관측소명', '관측일시', '풍향', 'date']:
            df_buoy[c] = pd.to_numeric(df_buoy[c], errors='coerce')
            
    numeric_cols = df_buoy.select_dtypes(include='number').columns.tolist()
    df_buoy_daily = df_buoy.groupby('date')[numeric_cols].mean().reset_index()
    for col in numeric_cols:
        df_buoy_daily[f'{prefix}{col}_1d_lag'] = df_buoy_daily[col].shift(1)
        df_buoy_daily[f'{prefix}{col}_3d_mean'] = df_buoy_daily[col].shift(1).rolling(3).mean()
        df_buoy_daily[f'{prefix}{col}_7d_mean'] = df_buoy_daily[col].shift(1).rolling(7).mean()
        
    df_wea = df_wea.merge(df_buoy_daily, on='date', how='left')
    print(f"     {fname} features merged")
for station in ['기장사업소', '정관사업소']:
    prefix = 'gijang' if '기장' in station else 'jeonggwan'
    discharge_files = [f for f in os.listdir(os.path.join(RAW, 'Discharge')) if station in f]
    if discharge_files:
        f_discharge = os.path.join(RAW, 'Discharge', discharge_files[0])
        try:
            df_discharge = pd.read_csv(f_discharge, encoding='cp949', on_bad_lines='skip')
        except:
            df_discharge = pd.read_csv(f_discharge, encoding='utf-8', on_bad_lines='skip')
        
        df_discharge.columns.values[0] = 'date'
        df_discharge.columns.values[1] = f'{prefix}_discharge_m3_day'
        df_discharge['date'] = pd.to_datetime(df_discharge['date'].astype(str), format='%Y%m%d', errors='coerce')
        df_discharge[f'{prefix}_discharge_m3_day'] = pd.to_numeric(df_discharge[f'{prefix}_discharge_m3_day'].astype(str).str.replace(',', ''), errors='coerce')
        df_discharge = df_discharge.dropna(subset=['date', f'{prefix}_discharge_m3_day'])
        df_discharge_daily = df_discharge.groupby('date')[f'{prefix}_discharge_m3_day'].sum().reset_index()
        
        df_discharge_daily[f'{prefix}_discharge_1d_lag'] = df_discharge_daily[f'{prefix}_discharge_m3_day'].shift(1)
        df_discharge_daily[f'{prefix}_discharge_3d_mean'] = df_discharge_daily[f'{prefix}_discharge_m3_day'].shift(1).rolling(3).mean()
        
        df_wea = df_wea.merge(df_discharge_daily, on='date', how='left')

# 4.5 External Data (Tide, Temp, Wind)
print("  4.5 Loading External Tide, Temp, Wind data...")
import glob
# Tide
tide_files = glob.glob(os.path.join(RAW, 'Tide', '부산_*_1시간 조위.txt'))
tide_records = []
for f in tide_files:
    with open(f, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3 and parts[0].count('/') == 2:
                try: tide_records.append({'date': parts[0], 'tide_val': float(parts[2])})
                except: pass
if tide_records:
    df_tide = pd.DataFrame(tide_records)
    df_tide['date'] = pd.to_datetime(df_tide['date'], format='%Y/%m/%d')
    df_tide_daily = df_tide.groupby('date')['tide_val'].agg(['max', 'min']).reset_index()
    df_tide_daily['tide_range'] = df_tide_daily['max'] - df_tide_daily['min']
    df_tide_daily = df_tide_daily[['date', 'tide_range']]
    df_tide_daily['tide_range_1d_lag'] = df_tide_daily['tide_range'].shift(1)
    df_wea = df_wea.merge(df_tide_daily, on='date', how='left')

# Temp
temp_files = glob.glob(os.path.join(RAW, 'Temp', '기장수온*.xlsx'))
df_temps = []
for f in temp_files:
    try: df_temps.append(pd.read_excel(f))
    except: pass
if df_temps:
    df_temp = pd.concat(df_temps, ignore_index=True)
    df_temp['관측일'] = pd.to_datetime(df_temp['관측일'], errors='coerce').dt.normalize()
    df_temp = df_temp[df_temp['수층'] == '표층'].copy()
    df_temp_daily = df_temp[['관측일', '평균수온(°C)']].rename(columns={'관측일': 'date', '평균수온(°C)': 'avg_water_temp'})
    df_temp_daily = df_temp_daily.groupby('date').mean().reset_index()
    df_temp_daily['avg_water_temp_1d_lag'] = df_temp_daily['avg_water_temp'].shift(1)
    df_wea = df_wea.merge(df_temp_daily, on='date', how='left')

# Wind
wind_files = glob.glob(os.path.join(RAW, 'Wind', 'OBS_AWS_DD_*.csv'))
df_winds = []
for f in wind_files:
    try: df_winds.append(pd.read_csv(f, encoding='cp949'))
    except: pass
if df_winds:
    df_wind = pd.concat(df_winds, ignore_index=True)
    df_wind['일시'] = pd.to_datetime(df_wind['일시']).dt.normalize()
    df_wind = df_wind.rename(columns={'일시': 'date', '최대 순간 풍속 풍향(deg)': 'wind_dir'})
    df_wind['wind_dir_rad'] = np.deg2rad(df_wind['wind_dir'])
    df_wind['wind_sin'] = np.sin(df_wind['wind_dir_rad'])
    df_wind['wind_cos'] = np.cos(df_wind['wind_dir_rad'])
    df_wind_daily = df_wind[['date', 'wind_sin', 'wind_cos']].groupby('date').mean().reset_index()
    df_wind_daily['wind_sin_1d_lag'] = df_wind_daily['wind_sin'].shift(1)
    df_wind_daily['wind_cos_1d_lag'] = df_wind_daily['wind_cos'].shift(1)
    df_wea = df_wea.merge(df_wind_daily, on='date', how='left')

# 5. Merge
df_master = daily_w.merge(df_wea, on='date', how='left')
df_master['log_ecoli'] = np.log1p(df_master['ecoli_max'])

print(f"  Final master: {df_master.shape}")
out_path = os.path.join(PROC, 'master_dataset.csv')
df_master.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f"  Saved: {out_path}")
print(f"[{BEACH}] Preprocessing complete!")
