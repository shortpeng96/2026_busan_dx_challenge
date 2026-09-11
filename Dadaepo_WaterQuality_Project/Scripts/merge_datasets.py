import pandas as pd
import numpy as np
import os

def build_master_dataset():
    water_path = "C:\\Sandbox\\Preprocessed\\dadaepo_water_samples.csv"
    weather_path = "C:\\Sandbox\\Weather\\dadaepo_weather_2014_2026.csv"
    sensor_path = "C:\\Sandbox\\Preprocessed\\nakdong_sensor_daily.csv"
    visitor_path = "C:\\Sandbox\\Preprocessed\\dadaepo_visitors_daily.csv"
    discharge_path = "C:\\Sandbox\\Preprocessed\\dadaepo_discharge_daily.csv"
    sewage_path = "C:\\Sandbox\\Preprocessed\\sewage_daily.csv"
    tide_path = "C:\\Sandbox\\Preprocessed\\dadaepo_tide_daily.csv"
    
    print("Loading datasets...")
    df_water = pd.read_csv(water_path)
    df_weather = pd.read_csv(weather_path)
    df_sensor = pd.read_csv(sensor_path)
    
    # Load Visitors
    has_visitors = False
    if os.path.exists(visitor_path):
        df_visitor = pd.read_csv(visitor_path)
        has_visitors = True
    else:
        df_visitor = pd.DataFrame(columns=['date', 'visitor_count'])
        
    # Load Discharge (Nakdong River)
    has_discharge = False
    if os.path.exists(discharge_path):
        df_discharge = pd.read_csv(discharge_path)
        has_discharge = True
    else:
        df_discharge = pd.DataFrame(columns=['date', 'discharge_volume'])
        
    # Load Tide
    has_tide = False
    if os.path.exists(tide_path):
        df_tide = pd.read_csv(tide_path)
        has_tide = True
    else:
        df_tide = pd.DataFrame(columns=['date', 'tide_max', 'tide_min', 'tide_range'])
        
    # Load Sewage
    has_sewage = False
    if os.path.exists(sewage_path):
        df_sewage = pd.read_csv(sewage_path)
        has_sewage = True
    else:
        df_sewage = pd.DataFrame(columns=['date', 'sewage_volume'])
        
    df_water['date_obj'] = pd.to_datetime(df_water['examinDe'])
    df_sensor['date_obj'] = pd.to_datetime(df_sensor['date'])
    df_visitor['date_obj'] = pd.to_datetime(df_visitor['date'])
    df_discharge['date_obj'] = pd.to_datetime(df_discharge['date'])
    df_tide['date_obj'] = pd.to_datetime(df_tide['date'])
    df_sewage['date_obj'] = pd.to_datetime(df_sewage['date'])
    
    # Drop string 'date' column so it doesn't cause merge conflicts
    for d in [df_discharge, df_tide, df_sewage]:
        if 'date' in d.columns:
            d.drop(columns=['date'], inplace=True)
    
    # Process Weather Data
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    df_weather['date'] = df_weather['time'].dt.date
    daily_weather = df_weather.groupby('date').agg(
        precip_daily=('precipitation_mm', 'sum'),
        temp_daily_mean=('temperature_2m', 'mean'),
        wind_speed_daily_max=('wind_speed_m_s', 'max')
    ).reset_index()
    daily_weather['date_obj'] = pd.to_datetime(daily_weather['date'])
    
    # Merge all daily features into one big Feature Table
    feature_table = pd.merge(daily_weather, df_sensor, on='date_obj', how='outer')
    feature_table = pd.merge(feature_table, df_visitor, on='date_obj', how='outer')
    feature_table = pd.merge(feature_table, df_discharge, on='date_obj', how='outer')
    feature_table = pd.merge(feature_table, df_tide, on='date_obj', how='outer')
    feature_table = pd.merge(feature_table, df_sewage, on='date_obj', how='outer')
    
    feature_table = feature_table.sort_values('date_obj').reset_index(drop=True)
    
    # Create Lag features
    feature_table['precip_1d_lag'] = feature_table['precip_daily'].shift(1)
    feature_table['precip_2d_sum_lag'] = feature_table['precip_daily'].rolling(2).sum().shift(1)
    feature_table['precip_3d_sum_lag'] = feature_table['precip_daily'].rolling(3).sum().shift(1)
    
    feature_table['discharge_1d_lag'] = feature_table['discharge_volume'].shift(1)
    feature_table['discharge_3d_sum_lag'] = feature_table['discharge_volume'].rolling(3).sum().shift(1)
    
    feature_table['temp_1d_lag'] = feature_table['temp_daily_mean'].shift(1)
    feature_table['wind_max_1d_lag'] = feature_table['wind_speed_daily_max'].shift(1)
    
    feature_table['sewage_discharge_1d_lag'] = feature_table['sewage_volume'].shift(1)
    feature_table['sewage_discharge_3d_sum_lag'] = feature_table['sewage_volume'].rolling(3).sum().shift(1)
    
    # Calculate CSO Flag
    # CSO condition: Sewage Volume on the day prior to testing >= 450,000 AND 3-day accumulated rain >= 5.0mm
    cso_cond = (feature_table['sewage_discharge_1d_lag'] >= 450000) & (feature_table['precip_3d_sum_lag'] >= 5.0)
    feature_table['CSO_Flag'] = cso_cond.astype(int)
    
    # Sensor and visitor lag features (only 1 day prior)
    lag_cols = [c for c in df_sensor.columns if c not in ['date', 'date_obj']] + ['visitor_count']
    for col in lag_cols:
        if col in feature_table.columns:
            feature_table[f"{col}_1d_lag"] = feature_table[col].shift(1)
            
    # Merge onto water quality target table (dadaepo_water_samples.csv has 305 rows)
    master_df = pd.merge(df_water, feature_table, on='date_obj', how='left')
    
    # Clean up redundant columns
    cols_to_drop = ['date_y', 'date_obj', 'date_x'] + [c for c in feature_table.columns if not c.endswith('_lag') and c not in ['date', 'tide_max', 'tide_min', 'tide_range', 'CSO_Flag']]
    
    # Rename date_x back to date
    master_df = master_df.rename(columns={'date_x': 'date'})
    master_df = master_df.drop(columns=[c for c in cols_to_drop if c in master_df.columns], errors='ignore')
    
    out_dir = "C:\\Sandbox\\Preprocessed"
    out_path = os.path.join(out_dir, "master_dataset_v2.csv")
    master_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    
    print(f"Master dataset V2 created with {len(master_df)} rows and {len(master_df.columns)} features.")
    print(f"Total CSO events observed in test samples: {master_df['CSO_Flag'].sum()}")
    print(f"Saved to {out_path}")
    
if __name__ == "__main__":
    build_master_dataset()
