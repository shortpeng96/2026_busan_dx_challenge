import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

base_dir = "C:\\Sandbox\\2026_busan_dx_challenge"
raw_water_dir = "C:\\Sandbox\\Water_Quality"
discharge_dir = "C:\\Sandbox\\Discharge"
proj_dir = os.path.join(base_dir, "Imrang_WaterQuality_Project")

os.makedirs(os.path.join(proj_dir, "Data_Processed"), exist_ok=True)
os.makedirs(os.path.join(proj_dir, "Scripts"), exist_ok=True)

def clean_value(val):
    if pd.isna(val): return np.nan
    v_str = str(val).strip()
    if '>' in v_str: v_str = v_str.replace('>', '')
    if '<' in v_str: v_str = v_str.replace('<', '')
    v_str = v_str.replace(',', '')
    try: return float(v_str)
    except: return np.nan

def assign_distance(loc):
    loc_str = str(loc).upper().replace(' ', '')
    if 'C' in loc_str or '카페' in loc_str or '좌광천' in loc_str or '꽃밭' in loc_str:
        return 0.0 # Closest to Jeonggwan CSO (Jwagwang Stream)
    elif 'A' in loc_str or '서편' in loc_str:
        return 1.0 # Furthest
    else:
        return 0.5 # Middle

print("1. Processing Water Quality Data...")
try:
    df_w = pd.read_csv(os.path.join(raw_water_dir, 'busan_beach_임랑.csv'), encoding='utf-8')
except:
    df_w = pd.read_csv(os.path.join(raw_water_dir, 'busan_beach_임랑.csv'), encoding='cp949')

df_w = df_w.dropna(subset=['examinDe'])
df_w['date'] = pd.to_datetime(df_w['examinDe'], errors='coerce')
df_w = df_w.dropna(subset=['date'])
df_w['ecoli'] = df_w['coliDetectCn'].apply(clean_value)
df_w = df_w.dropna(subset=['ecoli'])
df_w['any_exceed'] = (df_w['ecoli'] >= 500).astype(int)
df_w['distance_to_outfall'] = df_w['examinLcDetail'].apply(assign_distance)
print(f"Total valid spot records: {len(df_w)}")

print("2. Processing Weather Data...")
df_wea_raw = pd.read_csv(os.path.join(base_dir, 'Ilgwang_WaterQuality_Project', 'Data_Raw', 'ilgwang_weather_2014_2026.csv'), encoding='utf-8-sig')
df_wea_raw['time'] = pd.to_datetime(df_wea_raw['time'])
df_wea_raw['date'] = df_wea_raw['time'].dt.date

df_wea = df_wea_raw.groupby('date').agg({
    'precipitation_mm': 'sum',
    'temperature_2m': 'mean',
    'wind_speed_m_s': 'mean'
}).reset_index()
df_wea['date'] = pd.to_datetime(df_wea['date'])

df_wea['precip_1d_lag'] = df_wea['precipitation_mm'].shift(1)
df_wea['precip_3d_sum_lag'] = df_wea['precipitation_mm'].rolling(3).sum().shift(1)
df_wea['precip_5d_sum_lag'] = df_wea['precipitation_mm'].rolling(5).sum().shift(1)
df_wea['temp_1d_lag'] = df_wea['temperature_2m'].shift(1)
df_wea['temp_daily'] = df_wea['temperature_2m']
df_wea['wind_speed_1d_lag'] = df_wea['wind_speed_m_s'].shift(1)
df_wea['month'] = df_wea['date'].dt.month
df_wea['is_summer'] = df_wea['month'].isin([7, 8]).astype(int)

print("3. Processing CSO Discharge Data...")
cso_path = os.path.join(discharge_dir, '정관사업단 일일방류량.csv')
df_c = pd.read_csv(cso_path, encoding='cp949', skiprows=1, header=None)
df_c = df_c.iloc[:, :2]
df_c.columns = ['date', 'discharge']
df_c = df_c.dropna(subset=['date'])
df_c['date'] = df_c['date'].astype(str).str.split(' ').str[0]
df_c = df_c[df_c['date'].str.strip() != '']
df_c['date'] = pd.to_datetime(df_c['date'], errors='coerce')
df_c = df_c.dropna(subset=['date'])
df_c['discharge'] = df_c['discharge'].apply(clean_value)
df_c = df_c.groupby('date')['discharge'].sum().reset_index()
df_c = df_c.rename(columns={'discharge': '정관사업단_vol'})
df_c['정관사업단_vol_1d_lag'] = df_c['정관사업단_vol'].shift(1)
df_c['정관사업단_vol_3d_sum_lag'] = df_c['정관사업단_vol'].rolling(3).sum().shift(1)

print("4. Merging Master Dataset...")
df_master = pd.merge(df_w[['date', 'examinLcDetail', 'distance_to_outfall', 'ecoli', 'any_exceed']], df_wea, on='date', how='left')
df_master = pd.merge(df_master, df_c, on='date', how='left')
df_master = df_master.dropna(subset=['ecoli'])
df_master['log_ecoli'] = np.log1p(df_master['ecoli'])

out_path = os.path.join(proj_dir, "Data_Processed", "imrang_spatial_master.csv")
df_master.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f"Done! Created spatial dataset with {len(df_master)} rows")
print(f"Total Exceedances: {df_master['any_exceed'].sum()}")
