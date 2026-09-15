# -*- coding: utf-8 -*-
import pandas as pd, os, glob, requests, numpy as np

BASE = 'c:/Sandbox/2026_busan_dx_challenge/Ilgwang_WaterQuality_Project'
RAW = os.path.join(BASE, 'Data_Raw')
NEW_RAW = os.path.join(BASE, 'Data_Raw_New')
os.makedirs(NEW_RAW, exist_ok=True)

# 1-4. Open-Meteo
print('Fetching Open-Meteo...')
lat, lon = 35.2599, 129.2328
url = f'https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2014-01-01&end_date=2026-09-01&daily=precipitation_sum,temperature_2m_mean,wind_speed_10m_max,wind_direction_10m_dominant,shortwave_radiation_sum&timezone=Asia%2FSeoul'
res = requests.get(url)
df_om = pd.DataFrame(res.json()['daily'])
df_om[['time', 'precipitation_sum']].rename(columns={'time':'date', 'precipitation_sum':'precip_daily'}).to_csv(os.path.join(NEW_RAW, '일광_강수.csv'), index=False)
df_om[['time', 'temperature_2m_mean']].rename(columns={'time':'date', 'temperature_2m_mean':'temp_daily'}).to_csv(os.path.join(NEW_RAW, '일광_기온.csv'), index=False)
df_om[['time', 'wind_speed_10m_max', 'wind_direction_10m_dominant']].rename(columns={'time':'date', 'wind_speed_10m_max':'wind_max', 'wind_direction_10m_dominant':'wind_dir'}).to_csv(os.path.join(NEW_RAW, '일광_바람.csv'), index=False)
df_om[['time', 'shortwave_radiation_sum']].rename(columns={'time':'date', 'shortwave_radiation_sum':'solar_radiation_sum'}).to_csv(os.path.join(NEW_RAW, '일광_자외선.csv'), index=False)

def clean_val(v):
    if pd.isna(v): return np.nan
    v_str = str(v).replace(',', '').replace('<', '').replace('>', '').replace(' ', '')
    try: return float(v_str)
    except: return np.nan

# 5. Water Quality
print('Extracting Water Quality...')
f_water = os.path.join(RAW, 'water_quality_raw.csv')
try: df_wq = pd.read_csv(f_water, encoding='utf-8-sig')
except: df_wq = pd.read_csv(f_water, encoding='cp949', errors='replace')
col_map = {'조사일자': 'examinDe', '해수욕장명': 'beachKoreanNm', '조사위치상세': 'examinLcDetail', '대장균': 'coliDetectCn', '장구균': 'entrcccsDetectCn'}
if 'examinDe' not in df_wq.columns:
    df_wq.rename(columns=col_map, inplace=True)
df_wq['examinDe'] = pd.to_datetime(df_wq['examinDe'].astype(str).str.replace('-', '').str.replace('.', '').str[:8], format='%Y%m%d', errors='coerce')
df_wq = df_wq.dropna(subset=['examinDe'])
df_wq['date'] = pd.to_datetime(df_wq['examinDe'].dt.date)
df_wq['coliDetectCn'] = df_wq['coliDetectCn'].apply(clean_val)
df_wq['entrcccsDetectCn'] = df_wq['entrcccsDetectCn'].apply(clean_val)
df_grouped = df_wq.groupby(['date', 'examinDe', 'beachKoreanNm', 'examinLcDetail']).agg({'coliDetectCn': 'max', 'entrcccsDetectCn': 'max'}).reset_index()
df_grouped.rename(columns={'coliDetectCn': 'ecoli_max', 'entrcccsDetectCn': 'enterococcus_max'}, inplace=True)
df_grouped[['date', 'examinLcDetail', 'ecoli_max', 'enterococcus_max']].to_csv(os.path.join(NEW_RAW, '일광_수질.csv'), index=False)

# 6. Sewage
print('Extracting Sewage...')
sewage_dfs = []
for station in ['기장', '정관']:
    prefix = 'gijang' if station == '기장' else 'jeonggwan'
    discharge_files = [f for f in os.listdir(os.path.join(RAW, 'Discharge')) if station in f]
    if discharge_files:
        f_discharge = os.path.join(RAW, 'Discharge', discharge_files[0])
        try: df_discharge = pd.read_csv(f_discharge, encoding='cp949', on_bad_lines='skip')
        except: df_discharge = pd.read_csv(f_discharge, encoding='utf-8', on_bad_lines='skip')
        df_discharge.columns.values[0] = 'date'
        df_discharge.columns.values[1] = f'{prefix}_discharge_m3_day'
        df_discharge['date'] = pd.to_datetime(df_discharge['date'].astype(str), format='%Y%m%d', errors='coerce')
        df_discharge[f'{prefix}_discharge_m3_day'] = pd.to_numeric(df_discharge[f'{prefix}_discharge_m3_day'].astype(str).str.replace(',', ''), errors='coerce')
        df_discharge = df_discharge.dropna(subset=['date', f'{prefix}_discharge_m3_day'])
        df_discharge_daily = df_discharge.groupby('date')[f'{prefix}_discharge_m3_day'].sum().reset_index()
        sewage_dfs.append(df_discharge_daily)
if sewage_dfs:
    sewage_df = sewage_dfs[0]
    if len(sewage_dfs) > 1:
        sewage_df = pd.merge(sewage_df, sewage_dfs[1], on='date', how='outer')
    sewage_df.to_csv(os.path.join(NEW_RAW, '일광_인근_하수_방류량.csv'), index=False)

# 7. Tide
print('Extracting Tide...')
tide_files = glob.glob(os.path.join(RAW, 'Tide', '부산*_1시간 조위.txt'))
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
    df_tide_daily.rename(columns={'max': 'tide_max', 'min': 'tide_min'}, inplace=True)
    df_tide_daily.to_csv(os.path.join(NEW_RAW, '일광_조수.csv'), index=False)
else:
    pd.DataFrame(columns=['date']).to_csv(os.path.join(NEW_RAW, '일광_조수.csv'), index=False)

# 8. Water Temp
print('Extracting Water Temp...')
df_temps = []
temp_files = glob.glob(os.path.join(RAW, 'Temp', '기장수온*.xlsx'))
for f in temp_files:
    try: df_temps.append(pd.read_excel(f))
    except: pass
if df_temps:
    df_temp = pd.concat(df_temps, ignore_index=True)
    df_temp['관측일'] = pd.to_datetime(df_temp['관측일'], errors='coerce').dt.normalize()
    # Handle string literal comparison correctly
    s_col = [c for c in df_temp.columns if '수층' in c][0]
    s_val = '표층'.encode('utf-8').decode('utf-8')
    df_temp = df_temp[df_temp[s_col] == s_val].copy()
    val_col = [c for c in df_temp.columns if '수온' in c][0]
    df_temp_daily = df_temp[['관측일', val_col]].rename(columns={'관측일': 'date', val_col: 'avg_water_temp'})
    df_temp_daily = df_temp_daily.groupby('date').mean().reset_index()
else:
    df_temp_daily = pd.DataFrame(columns=['date'])

# Merge Buoys into Water Temp
for b_file in glob.glob(os.path.join(RAW, 'meis_buoy_*.csv')):
    fname = os.path.basename(b_file)
    try: df_buoy = pd.read_csv(b_file, encoding='utf-8-sig')
    except: df_buoy = pd.read_csv(b_file, encoding='cp949')
    date_col = [c for c in df_buoy.columns if '일시' in c or 'date' in c.lower() or 'time' in c.lower()]
    df_buoy['date'] = pd.to_datetime(df_buoy[date_col[0]] if date_col else df_buoy.iloc[:, 2]).dt.normalize()
    prefix = 'gijang_buoy_' if 'gijang' in fname else 'meis_buoy_'
    for c in df_buoy.columns:
        if c not in ['관측소명', '관측일시', '풍향', 'date']:
            df_buoy[c] = pd.to_numeric(df_buoy[c], errors='coerce')
    numeric_cols = df_buoy.select_dtypes(include='number').columns.tolist()
    df_buoy_daily = df_buoy.groupby('date')[numeric_cols].mean().reset_index()
    # Rename numeric cols with prefix
    df_buoy_daily.rename(columns={c: f"{prefix}{c}" for c in numeric_cols}, inplace=True)
    
    if not df_temp_daily.empty and not df_buoy_daily.empty:
        df_temp_daily = pd.merge(df_temp_daily, df_buoy_daily, on='date', how='outer')
    elif df_temp_daily.empty:
        df_temp_daily = df_buoy_daily

df_temp_daily.to_csv(os.path.join(NEW_RAW, '일광_수온.csv'), index=False)

# 9. Empties
print('Creating empties...')
pd.DataFrame(columns=['date']).to_csv(os.path.join(NEW_RAW, '일광_방문객.csv'), index=False)
pd.DataFrame(columns=['date']).to_csv(os.path.join(NEW_RAW, '일광_인근_하천_방류량.csv'), index=False)

print('Done!')
