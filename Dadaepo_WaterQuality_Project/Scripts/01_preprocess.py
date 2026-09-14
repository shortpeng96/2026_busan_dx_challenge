"""
01_preprocess.py - Dadaepo Beach Water Quality
Generates master_dataset_v2.csv with Advanced Features locally for Dadaepo.
"""
import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, 'Data_Processed')
os.makedirs(PROC, exist_ok=True)

def build_local_master_dataset():
    water_path = "C:\\Sandbox\\Preprocessed\\dadaepo_water_samples.csv"
    weather_path = "C:\\Sandbox\\Weather\\dadaepo_weather_2014_2026.csv"
    sensor_path = "C:\\Sandbox\\Preprocessed\\nakdong_sensor_daily.csv"
    visitor_path = "C:\\Sandbox\\Preprocessed\\dadaepo_visitors_daily.csv"
    discharge_path = "C:\\Sandbox\\Preprocessed\\dadaepo_discharge_daily.csv"
    sewage_path = "C:\\Sandbox\\Preprocessed\\sewage_daily.csv"
    tide_path = "C:\\Sandbox\\Preprocessed\\dadaepo_tide_daily.csv"
    
    print("[Dadaepo] Loading datasets for V2 Advanced Features...")
    df_water = pd.read_csv(water_path)
    df_weather = pd.read_csv(weather_path)
    df_sensor = pd.read_csv(sensor_path)
    
    has_visitors = os.path.exists(visitor_path)
    df_visitor = pd.read_csv(visitor_path) if has_visitors else pd.DataFrame(columns=['date', 'visitor_count'])
    has_discharge = os.path.exists(discharge_path)
    df_discharge = pd.read_csv(discharge_path) if has_discharge else pd.DataFrame(columns=['date', 'discharge_volume'])
    has_tide = os.path.exists(tide_path)
    df_tide = pd.read_csv(tide_path) if has_tide else pd.DataFrame(columns=['date', 'tide_max', 'tide_min', 'tide_range'])
    has_sewage = os.path.exists(sewage_path)
    df_sewage = pd.read_csv(sewage_path) if has_sewage else pd.DataFrame(columns=['date', 'sewage_volume'])
        
    df_water['date_obj'] = pd.to_datetime(df_water['examinDe'])
    df_sensor['date_obj'] = pd.to_datetime(df_sensor['date'])
    df_visitor['date_obj'] = pd.to_datetime(df_visitor['date'])
    df_discharge['date_obj'] = pd.to_datetime(df_discharge['date'])
    df_tide['date_obj'] = pd.to_datetime(df_tide['date'])
    df_sewage['date_obj'] = pd.to_datetime(df_sewage['date'])
    
    for d in [df_discharge, df_tide, df_sewage]:
        if 'date' in d.columns:
            d.drop(columns=['date'], inplace=True)
    
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    df_weather['date'] = df_weather['time'].dt.date
    daily_weather = df_weather.groupby('date').agg(
        precip_daily=('precipitation_mm', 'sum'),
        temp_daily_mean=('temperature_2m', 'mean'),
        wind_speed_daily_max=('wind_speed_m_s', 'max')
    ).reset_index()
    daily_weather['date_obj'] = pd.to_datetime(daily_weather['date'])
    
    feature_table = pd.merge(daily_weather, df_sensor, on='date_obj', how='outer')
    feature_table = pd.merge(feature_table, df_visitor, on='date_obj', how='outer')
    feature_table = pd.merge(feature_table, df_discharge, on='date_obj', how='outer')
    feature_table = pd.merge(feature_table, df_tide, on='date_obj', how='outer')
    feature_table = pd.merge(feature_table, df_sewage, on='date_obj', how='outer')
    feature_table = feature_table.sort_values('date_obj').reset_index(drop=True)
    
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
    
    lag_cols = [c for c in df_sensor.columns if c not in ['date', 'date_obj']] + ['visitor_count']
    for col in lag_cols:
        if col in feature_table.columns:
            feature_table[f"{col}_1d_lag"] = feature_table[col].shift(1)
            
    master_df = pd.merge(df_water, feature_table, on='date_obj', how='left')
    cols_to_drop = ['date_y', 'date_obj', 'date_x'] + [c for c in feature_table.columns if not c.endswith('_lag') and c not in ['date', 'tide_max', 'tide_min', 'tide_range', 'CSO_Flag']]
    
    master_df = master_df.rename(columns={'date_x': 'date'})
    master_df = master_df.drop(columns=[c for c in cols_to_drop if c in master_df.columns], errors='ignore')
    
    out_path = os.path.join(PROC, "master_dataset_v2.csv")
    master_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    
    print(f"[Dadaepo] Preprocessing complete! Master dataset V2 generated locally at: {out_path}")

if __name__ == "__main__":
    build_local_master_dataset()
