import pandas as pd
import numpy as np
import os

def build_master_dataset():
    proj_dir = "C:\\Sandbox\\Ilgwang_WaterQuality_Project"
    
    # 1. Water Quality Samples
    f_water = os.path.join(proj_dir, "Data_Processed", "ilgwang_water_samples.csv")
    df_water = pd.read_csv(f_water, encoding='utf-8-sig')
    df_water['date'] = pd.to_datetime(df_water['date'])
    
    # 2. Weather
    f_weather = os.path.join(proj_dir, "Data_Raw", "ilgwang_weather_2014_2026.csv")
    df_weather = pd.read_csv(f_weather)
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    df_weather['date'] = df_weather['time'].dt.date
    
    df_weather_daily = df_weather.groupby('date').agg({
        'precipitation_mm': 'sum',
        'temperature_2m': 'mean',
        'wind_speed_m_s': 'max'
    }).reset_index()
    df_weather_daily['date'] = pd.to_datetime(df_weather_daily['date'])
    df_weather_daily.rename(columns={'precipitation_mm': 'precip_daily', 
                                     'temperature_2m': 'temp_daily', 
                                     'wind_speed_m_s': 'wind_max'}, inplace=True)
                                     
    # Merge Features Table
    feature_table = df_weather_daily.copy()
    feature_table = feature_table.sort_values('date').reset_index(drop=True)
    
    # Generate Lagged Features
    feature_table['precip_1d_lag'] = feature_table['precip_daily'].shift(1)
    feature_table['precip_2d_sum_lag'] = feature_table['precip_daily'].rolling(2).sum().shift(1)
    feature_table['precip_3d_sum_lag'] = feature_table['precip_daily'].rolling(3).sum().shift(1)
    feature_table['precip_5d_sum_lag'] = feature_table['precip_daily'].rolling(5).sum().shift(1)
    
    feature_table['temp_1d_lag'] = feature_table['temp_daily'].shift(1)
    feature_table['wind_max_1d_lag'] = feature_table['wind_max'].shift(1)
    
    # Urban Runoff Rain Flag (Heavy rain > 20mm in 3 days)
    cso_rain = (feature_table['precip_3d_sum_lag'] >= 20.0)
    feature_table['CSO_Flag_Rain'] = cso_rain.astype(int)
    
    # Merge to Water Quality samples
    master_df = pd.merge(df_water, feature_table, on='date', how='left')
    master_df = master_df.dropna(subset=['ecoli_max'])
    
    out_path = os.path.join(proj_dir, "Data_Processed", "master_dataset_ilgwang.csv")
    master_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    
    print(f"Master dataset created with {len(master_df)} rows.")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    build_master_dataset()
