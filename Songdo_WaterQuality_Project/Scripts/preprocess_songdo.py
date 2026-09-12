import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

base_dir = "C:\\Sandbox\\2026_busan_dx_challenge"
raw_water_dir = "C:\\Sandbox\\Water_Quality"
discharge_dir = "C:\\Sandbox\\Discharge"
proj_dir = os.path.join(base_dir, "Songdo_WaterQuality_Project")

os.makedirs(os.path.join(proj_dir, "Data_Processed"), exist_ok=True)

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
    # East side (Closest to Jungang/Yeongdo)
    if '호텔' in loc_str or 'A' in loc_str or '관광' in loc_str:
        return 0.0
    # West side (Furthest)
    elif '스포츠' in loc_str or 'C' in loc_str:
        return 1.0
    # Middle
    elif '행정' in loc_str or '봉사' in loc_str or 'B' in loc_str:
        return 0.5
    else:
        return 0.5 # Default to middle if unknown

print("1. Processing Water Quality Data (Preserving all spots)...")
try:
    df_w = pd.read_csv(os.path.join(raw_water_dir, 'busan_beach_송도.csv'), encoding='utf-8')
except:
    df_w = pd.read_csv(os.path.join(raw_water_dir, 'busan_beach_송도.csv'), encoding='cp949')

df_w = df_w.dropna(subset=['examinDe'])
df_w['date'] = pd.to_datetime(df_w['examinDe'], errors='coerce')
df_w = df_w.dropna(subset=['date'])
df_w['ecoli'] = df_w['coliDetectCn'].apply(clean_value)
df_w = df_w.dropna(subset=['ecoli'])
df_w['any_exceed'] = (df_w['ecoli'] >= 500).astype(int)
df_w['distance_to_outfall'] = df_w['examinLcDetail'].apply(assign_distance)

print(f"Total valid spot records: {len(df_w)}")

print("2. Processing Weather Data (Including Wind)...")
df_wea_raw = pd.read_csv(os.path.join(base_dir, 'Gwangalli_WaterQuality_Project', 'Data_Raw', 'gwangalli_weather_2014_2026.csv'), encoding='utf-8-sig')
df_wea_raw['time'] = pd.to_datetime(df_wea_raw['time'])
df_wea_raw['date'] = df_wea_raw['time'].dt.date

# Calculate daily aggregates including wind
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

print("3. Processing CSO Discharge Data...")
cso_files = ['중앙사업단 일일방류량.csv', '영도사업단 일일방류량.csv']
df_cso_list = []
for cso_file in cso_files:
    cso_path = os.path.join(discharge_dir, cso_file)
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
    cso_name = cso_file.split(' ')[0]
    df_c = df_c.rename(columns={'discharge': f'{cso_name}_vol'})
    df_c[f'{cso_name}_vol_1d_lag'] = df_c[f'{cso_name}_vol'].shift(1)
    df_c[f'{cso_name}_vol_3d_sum_lag'] = df_c[f'{cso_name}_vol'].rolling(3).sum().shift(1)
    df_cso_list.append(df_c)

print("4. Merging Master Dataset...")
df_master = pd.merge(df_w[['date', 'examinLcDetail', 'distance_to_outfall', 'ecoli', 'any_exceed']], df_wea, on='date', how='left')
for df_c in df_cso_list:
    df_master = pd.merge(df_master, df_c, on='date', how='left')

df_master = df_master.dropna(subset=['ecoli'])
df_master['log_ecoli'] = np.log1p(df_master['ecoli'])

out_path = os.path.join(proj_dir, "Data_Processed", "songdo_spatial_master.csv")
df_master.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f"Done! Created spatial dataset with {len(df_master)} rows at {out_path}")
print(f"Total Exceedances in spatial dataset: {df_master['any_exceed'].sum()}")
