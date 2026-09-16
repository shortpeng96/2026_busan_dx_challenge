import requests
import pandas as pd
import os

def fetch_and_split(beach_name, lat, lon, out_dir):
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2014-01-01",
        "end_date": "2026-09-10",
        "hourly": "temperature_2m,precipitation,wind_speed_10m,wind_direction_10m,shortwave_radiation",
        "timezone": "Asia/Seoul"
    }
    
    print(f"Fetching data for {beach_name} ({lat}, {lon})...")
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        hourly = data['hourly']
        
        df = pd.DataFrame({
            "date": pd.to_datetime(hourly['time']),
            "temp": hourly['temperature_2m'],
            "precip": hourly['precipitation'],
            "wind_speed": [v * 0.277778 if v is not None else None for v in hourly['wind_speed_10m']],
            "wind_dir": hourly['wind_direction_10m'],
            "solar": hourly['shortwave_radiation']
        })
        
        # Aggregate to daily
        df['day'] = df['date'].dt.date
        
        # Temp (Mean)
        df_temp = df.groupby('day')['temp'].mean().reset_index()
        df_temp.columns = ['date', 'temperature_2m_mean']
        
        # Precip (Sum)
        df_precip = df.groupby('day')['precip'].sum().reset_index()
        df_precip.columns = ['date', 'precipitation_sum']
        
        # Wind (Max speed, dominant direction - simplified to mean direction for now or just max speed)
        # To match existing columns, usually wind_speed_10m_max, wind_direction_10m_dominant
        # Actually, let's just get daily max wind speed and mean direction
        df_wind_speed = df.groupby('day')['wind_speed'].max().reset_index()
        df_wind_dir = df.groupby('day')['wind_dir'].mean().reset_index()
        df_wind = pd.merge(df_wind_speed, df_wind_dir, on='day')
        df_wind.columns = ['date', 'wind_speed_10m_max', 'wind_direction_10m_dominant']
        
        # Solar (Sum) - Convert W/m2 to MJ/m2 roughly by multiplying by 0.0864 (for daily sum of hourly average)
        # Actually, if we just sum the hourly W/m2, we multiply by 3600 J/h / 1,000,000 = 0.0036 MJ/m2 per hour
        df['solar_mj'] = df['solar'] * 0.0036
        df_solar = df.groupby('day')['solar_mj'].sum().reset_index()
        df_solar.columns = ['date', 'solar_radiation_sum']
        
        os.makedirs(out_dir, exist_ok=True)
        
        df_temp.to_csv(os.path.join(out_dir, f"{beach_name}_기온.csv"), index=False)
        df_precip.to_csv(os.path.join(out_dir, f"{beach_name}_강수.csv"), index=False)
        df_wind.to_csv(os.path.join(out_dir, f"{beach_name}_바람.csv"), index=False)
        df_solar.to_csv(os.path.join(out_dir, f"{beach_name}_자외선.csv"), index=False)
        
        print(f"Successfully saved 4 files for {beach_name}")
    else:
        print(f"Error fetching data: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    songdo_dir = r"C:\Sandbox\2026_busan_dx_challenge\Songdo_WaterQuality_Project\Data_Raw"
    songjeong_dir = r"C:\Sandbox\2026_busan_dx_challenge\Songjeong_WaterQuality_Project\Data_Raw"
    
    fetch_and_split("송도", 35.0760, 129.0173, songdo_dir)
    fetch_and_split("송정", 35.1786, 129.1997, songjeong_dir)
