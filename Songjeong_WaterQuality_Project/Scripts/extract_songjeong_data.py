# -*- coding: utf-8 -*-
import pandas as pd, os, glob, requests, numpy as np

BASE = 'c:/Sandbox/2026_busan_dx_challenge/Songjeong_WaterQuality_Project'
RAW = os.path.join(BASE, 'Data_Raw')
NEW_RAW = os.path.join(BASE, 'Data_Raw_New')
os.makedirs(NEW_RAW, exist_ok=True)

# 1-4. Open-Meteo
print('Fetching Open-Meteo...')
lat, lon = 35.1781, 129.1982
url = (f'https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}'
       f'&start_date=2014-01-01&end_date=2026-09-01'
       f'&daily=precipitation_sum,temperature_2m_mean,wind_speed_10m_max,'
       f'wind_direction_10m_dominant,shortwave_radiation_sum&timezone=Asia%2FSeoul')
res = requests.get(url)
df_om = pd.DataFrame(res.json()['daily'])
df_om[['time', 'precipitation_sum']].rename(
    columns={'time': 'date', 'precipitation_sum': 'precip_daily'}
).to_csv(os.path.join(NEW_RAW, '송정_강수.csv'), index=False)
df_om[['time', 'temperature_2m_mean']].rename(
    columns={'time': 'date', 'temperature_2m_mean': 'temp_daily'}
).to_csv(os.path.join(NEW_RAW, '송정_기온.csv'), index=False)
df_om[['time', 'wind_speed_10m_max', 'wind_direction_10m_dominant']].rename(
    columns={'time': 'date', 'wind_speed_10m_max': 'wind_max', 'wind_direction_10m_dominant': 'wind_dir'}
).to_csv(os.path.join(NEW_RAW, '송정_바람.csv'), index=False)
df_om[['time', 'shortwave_radiation_sum']].rename(
    columns={'time': 'date', 'shortwave_radiation_sum': 'solar_radiation_sum'}
).to_csv(os.path.join(NEW_RAW, '송정_자외선.csv'), index=False)

def clean_val(v):
    if pd.isna(v): return np.nan
    v_str = str(v).replace(',', '').replace('<', '').replace('>', '').replace(' ', '')
    try: return float(v_str)
    except: return np.nan

# 5. Water Quality
print('Extracting Water Quality...')
f_water = 'C:/Sandbox/Water_Quality/busan_beach_송정.csv'
try: df_wq = pd.read_csv(f_water, encoding='utf-8-sig')
except: df_wq = pd.read_csv(f_water, encoding='cp949', errors='replace')

if 'examinDe' in df_wq.columns:
    df_wq['date'] = pd.to_datetime(df_wq['examinDe'], errors='coerce')
else:
    df_wq['date'] = pd.NaT

df_wq = df_wq.dropna(subset=['date'])
df_wq['date'] = pd.to_datetime(df_wq['date'].dt.date)
df_wq['coliDetectCn'] = df_wq['coliDetectCn'].apply(clean_val)
df_wq['entrcccsDetectCn'] = df_wq['entrcccsDetectCn'].apply(clean_val)
df_grouped = df_wq.groupby(['date', 'examinLcSeCode']).agg(
    {'coliDetectCn': 'max', 'entrcccsDetectCn': 'max'}
).reset_index()
df_grouped.rename(columns={
    'examinLcSeCode': 'examinLcDetail',
    'coliDetectCn': 'ecoli_max',
    'entrcccsDetectCn': 'enterococcus_max'
}, inplace=True)
df_grouped[['date', 'examinLcDetail', 'ecoli_max', 'enterococcus_max']].to_csv(
    os.path.join(NEW_RAW, '송정_수질.csv'), index=False
)

# 6. Sewage (동부사업소 + 해운대사업소)
print('Extracting Sewage...')
sewage_dfs = []
for fname, prefix in [
    ('동부사업소.csv', 'dongbu'),
    ('해운대사업소 일일방류량.csv', 'haeundae_sewage'),
]:
    f = os.path.join('C:/Sandbox/Discharge', fname)
    if not os.path.exists(f):
        print(f'  Warning: {fname} not found, skipping.')
        continue
    try: df_d = pd.read_csv(f, encoding='cp949', on_bad_lines='skip')
    except: df_d = pd.read_csv(f, encoding='utf-8', on_bad_lines='skip')
    df_d.columns.values[0] = 'date'
    df_d.columns.values[1] = f'{prefix}_discharge_m3_day'
    df_d['date'] = pd.to_datetime(df_d['date'].astype(str), format='%Y%m%d', errors='coerce')
    df_d[f'{prefix}_discharge_m3_day'] = pd.to_numeric(
        df_d[f'{prefix}_discharge_m3_day'].astype(str).str.replace(',', ''), errors='coerce'
    )
    df_d = df_d.dropna(subset=['date', f'{prefix}_discharge_m3_day'])
    sewage_dfs.append(df_d.groupby('date')[f'{prefix}_discharge_m3_day'].sum().reset_index())

if sewage_dfs:
    sewage_df = sewage_dfs[0]
    for df in sewage_dfs[1:]:
        sewage_df = pd.merge(sewage_df, df, on='date', how='outer')
    sewage_df.to_csv(os.path.join(NEW_RAW, '송정_인근_하수_방류량.csv'), index=False)
else:
    pd.DataFrame(columns=['date']).to_csv(os.path.join(NEW_RAW, '송정_인근_하수_방류량.csv'), index=False)

# 7. Tide
print('Extracting Tide...')
# Buoy-based tide if no raw tide file
pd.DataFrame(columns=['date']).to_csv(os.path.join(NEW_RAW, '송정_조수.csv'), index=False)

# 8. Water Temp
print('Extracting Water Temp...')
pd.DataFrame(columns=['date']).to_csv(os.path.join(NEW_RAW, '송정_수온.csv'), index=False)

# 9. Empties
print('Creating empties...')
pd.DataFrame(columns=['date']).to_csv(os.path.join(NEW_RAW, '송정_방문객.csv'), index=False)
pd.DataFrame(columns=['date']).to_csv(os.path.join(NEW_RAW, '송정_인근_하천_방류량.csv'), index=False)

print('Done!')
