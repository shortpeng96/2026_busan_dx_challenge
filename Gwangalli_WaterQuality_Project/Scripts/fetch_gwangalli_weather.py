import requests
import pandas as pd
import os

def fetch_open_meteo_historical(lat, lon, start_date, end_date, output_path):
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,precipitation,wind_speed_10m,wind_direction_10m",
        "timezone": "Asia/Seoul"
    }
    
    print(f"Fetching historical weather data from {start_date} to {end_date}...")
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        
        # Open-Meteo returns hourly data as a dictionary of lists
        hourly = data['hourly']
        
        df = pd.DataFrame({
            "time": hourly['time'],
            "temperature_2m": hourly['temperature_2m'],
            "precipitation_mm": hourly['precipitation'],
            "wind_speed_m_s": [v * 0.277778 if v is not None else None for v in hourly['wind_speed_10m']], # Convert km/h to m/s
            "wind_direction_deg": hourly['wind_direction_10m']
        })
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Successfully saved {len(df)} hourly records to {output_path}")
        return df
    else:
        print(f"Error fetching data: {response.status_code}")
        print(response.text)
        return None

if __name__ == "__main__":
    # Gwangalli Beach exact coordinates
    lat = 35.1531
    lon = 129.1186
    
    # We want data from 2014 to 2026 to match the water quality data
    start_date = "2014-01-01"
    end_date = "2026-09-10" 
    
    output_path = "C:\\Sandbox\\Gwangalli_WaterQuality_Project\\Data_Raw\\gwangalli_weather_2014_2026.csv"
    
    fetch_open_meteo_historical(lat, lon, start_date, end_date, output_path)
