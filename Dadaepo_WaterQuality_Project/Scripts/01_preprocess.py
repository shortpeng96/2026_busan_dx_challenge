# -*- coding: utf-8 -*-
"""
01_preprocess.py - Dadaepo Beach Water Quality
Generates master_dataset_v2.csv with Advanced Features locally for Dadaepo.
"""
import pandas as pd
import numpy as np
import os, glob
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, 'Data_Raw')
PROC = os.path.join(BASE, 'Data_Processed')
os.makedirs(PROC, exist_ok=True)

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
    if '다대포해수욕장(좌)' in loc_str or '서측' in loc_str:
        return 0.2
    elif '다대포해수욕장(우)' in loc_str or '동측' in loc_str:
        return 0.8
    else:
        return 0.5

def build_local_master_dataset():
    print("[Dadaepo] Loading datasets for V2 Advanced Features from Data_Raw...")
    
    # 1. Water Quality
    f_water = os.path.join(RAW, 'water_quality_raw.csv')
    try:
        df_water = pd.read_csv(f_water, encoding='utf-8')
    except:
        df_water = pd.read_csv(f_water, encoding='cp949')
    
    df_water = df_water.dropna(subset=['examinDe'])
    df_water['date'] = pd.to_datetime(df_water['examinDe'], errors='coerce')
    df_water = df_water.dropna(subset=['date'])
    df_water['ecoli'] = df_water['coliDetectCn'].apply(clean_val)
    df_water = df_water.dropna(subset=['ecoli'])
    df_water['any_exceed'] = (df_water['ecoli'] >= 500).astype(int)
    df_water['distance_to_outfall'] = df_water['examinLcDetail'].apply(assign_distance)
    df_water['log_ecoli'] = np.log1p(df_water['ecoli'])
    
    # 2. Weather
    weather_path = os.path.join(RAW, 'dadaepo_weather_2014_2026.csv')
    df_weather = pd.read_csv(weather_path, encoding='utf-8-sig')
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    df_weather['date'] = pd.to_datetime(df_weather['time'].dt.date)
    daily_weather = df_weather.groupby('date').agg(
        precip_daily=('precipitation_mm', 'sum'),
        temp_daily_mean=('temperature_2m', 'mean'),
        wind_speed_daily_max=('wind_speed_m_s', 'max')
    ).reset_index()

    # 3. Sensor & Visitors
    df_sensor = pd.read_csv(os.path.join(RAW, 'nakdong_sensor_daily.csv'))
    df_sensor['date'] = pd.to_datetime(df_sensor['date'])
    df_visitor = pd.read_csv(os.path.join(RAW, 'dadaepo_visitors_daily.csv'))
    df_visitor['date'] = pd.to_datetime(df_visitor['date'])
    
    # 4. Discharge (Nakdong & Sewage)
    df_discharge = pd.DataFrame(columns=['date', 'discharge_volume'])
    df_sewage = pd.DataFrame(columns=['date', 'sewage_volume'])
    
    discharge_dir = os.path.join(RAW, 'Discharge')
    if os.path.exists(discharge_dir):
        # Nakdong Total Discharge
        nd_files = glob.glob(os.path.join(discharge_dir, '*최종통합*.csv'))
        if nd_files:
            try: df_nd = pd.read_csv(nd_files[0], encoding='utf-8')
            except: df_nd = pd.read_csv(nd_files[0], encoding='cp949')
            cols = df_nd.columns.tolist()
            d_col = next((c for c in cols if '날짜' in c or '관측' in c), cols[0])
            v_col = next((c for c in cols if '총방류량' in c), cols[6])
            df_nd = df_nd[[d_col, v_col]].copy()
            df_nd = df_nd.rename(columns={d_col: 'date', v_col: 'discharge_volume'})
            df_nd['date'] = pd.to_datetime(df_nd['date'], errors='coerce')
            df_nd['discharge_volume'] = pd.to_numeric(df_nd['discharge_volume'], errors='coerce')
            df_discharge = df_nd.dropna(subset=['date']).groupby('date')['discharge_volume'].sum().reset_index()
        
        # Gangbyeon Sewage
        sw_files = glob.glob(os.path.join(discharge_dir, '*강변*.csv'))
        if sw_files:
            df_sw = pd.read_csv(sw_files[0], encoding='cp949', skiprows=1, header=None)
            df_sw = df_sw.iloc[:, :2]
            df_sw.columns = ['date', 'sewage_volume']
            df_sw['date'] = df_sw['date'].astype(str).str.split(' ').str[0]
            df_sw['date'] = pd.to_datetime(df_sw['date'], errors='coerce')
            df_sw['sewage_volume'] = df_sw['sewage_volume'].apply(clean_val)
            df_sewage = df_sw.dropna(subset=['date']).groupby('date')['sewage_volume'].sum().reset_index()
            
    # 5. Tide
    df_tide = pd.DataFrame(columns=['date', 'tide_max', 'tide_min', 'tide_range'])
    tide_dir = os.path.join(RAW, 'Tide')
    if os.path.exists(tide_dir):
        txt_files = glob.glob(os.path.join(tide_dir, '*.txt'))
        all_tide = []
        for f in txt_files:
            try:
                df = pd.read_csv(f, sep=r'\s+', skiprows=5, header=None, names=['date', 'time', 'tide_level'], encoding='utf-8', on_bad_lines='skip')
                all_tide.append(df)
            except: pass
        if all_tide:
            tide_df = pd.concat(all_tide, ignore_index=True).dropna()
            tide_df['date'] = tide_df['date'].str.replace('/', '-')
            df_tide = tide_df.groupby('date').agg(tide_max=('tide_level', 'max'), tide_min=('tide_level', 'min')).reset_index()
            df_tide['tide_range'] = df_tide['tide_max'] - df_tide['tide_min']
            df_tide['date'] = pd.to_datetime(df_tide['date'])

    # 6. Merge Feature Table
    feature_table = pd.merge(daily_weather, df_sensor, on='date', how='outer')
    feature_table = pd.merge(feature_table, df_visitor, on='date', how='outer')
    feature_table = pd.merge(feature_table, df_discharge, on='date', how='outer')
    feature_table = pd.merge(feature_table, df_tide, on='date', how='outer')
    feature_table = pd.merge(feature_table, df_sewage, on='date', how='outer')
    feature_table = feature_table.sort_values('date').reset_index(drop=True)
    
    # 7. Lags and CSO Flag
    feature_table['precip_1d_lag'] = feature_table['precip_daily'].shift(1)
    feature_table['precip_2d_sum_lag'] = feature_table['precip_daily'].rolling(2).sum().shift(1)
    feature_table['precip_3d_sum_lag'] = feature_table['precip_daily'].rolling(3).sum().shift(1)
    feature_table['discharge_1d_lag'] = feature_table['discharge_volume'].shift(1)
    feature_table['discharge_3d_sum_lag'] = feature_table['discharge_volume'].rolling(3).sum().shift(1)
    feature_table['temp_1d_lag'] = feature_table['temp_daily_mean'].shift(1)
    feature_table['wind_max_1d_lag'] = feature_table['wind_speed_daily_max'].shift(1)
    feature_table['sewage_discharge_1d_lag'] = feature_table['sewage_volume'].shift(1)
    feature_table['sewage_discharge_3d_sum_lag'] = feature_table['sewage_volume'].rolling(3).sum().shift(1)
    
    cso_cond = (feature_table['sewage_discharge_1d_lag'] >= 450000) & (feature_table['precip_3d_sum_lag'] >= 5.0)
    feature_table['CSO_Flag'] = cso_cond.astype(int)
    
    lag_cols = [c for c in df_sensor.columns if c != 'date'] + ['visitor_count']
    for col in lag_cols:
        if col in feature_table.columns:
            feature_table[f"{col}_1d_lag"] = feature_table[col].shift(1)
            
    # 8. Merge with Water Quality target
    master_df = pd.merge(df_water[['date', 'examinLcDetail', 'distance_to_outfall', 'ecoli', 'log_ecoli', 'any_exceed']], feature_table, on='date', how='left')
    cols_to_drop = [c for c in feature_table.columns if not c.endswith('_lag') and c not in ['date', 'tide_max', 'tide_min', 'tide_range', 'CSO_Flag']]
    master_df = master_df.drop(columns=[c for c in cols_to_drop if c in master_df.columns], errors='ignore')
    
    master_df = master_df.dropna(subset=['ecoli'])
    
    out_path = os.path.join(PROC, "master_dataset_v2.csv")
    master_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"[Dadaepo] Preprocessing complete! Master dataset V2 generated locally at: {out_path}")
    print(f"Final shape: {master_df.shape}")

if __name__ == "__main__":
    build_local_master_dataset()
