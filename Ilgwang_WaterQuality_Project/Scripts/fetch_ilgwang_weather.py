import requests
import pandas as pd
import os

def fetch_weather():
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 35.259,
        "longitude": 129.234,
        "start_date": "2014-01-01",
        "end_date": "2026-09-01",
        "hourly": ["temperature_2m", "precipitation", "wind_speed_10m"],
        "timezone": "Asia/Seoul"
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    hourly = data['hourly']
    df = pd.DataFrame({
        'time': pd.to_datetime(hourly['time']),
        'temperature_2m': hourly['temperature_2m'],
        'precipitation_mm': hourly['precipitation'],
        'wind_speed_m_s': hourly['wind_speed_10m']
    })
    
    out_dir = "C:\\Sandbox\\Ilgwang_WaterQuality_Project\\Data_Raw"
    out_path = os.path.join(out_dir, "ilgwang_weather_2014_2026.csv")
    df.to_csv(out_path, index=False)
    print(f"Weather data saved to {out_path} ({len(df)} records)")

if __name__ == "__main__":
    fetch_weather()
