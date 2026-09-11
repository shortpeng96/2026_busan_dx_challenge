import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import os

def fetch_weather():
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 35.1587,
        "longitude": 129.1604,
        "start_date": "2014-01-01",
        "end_date": "2026-12-31",
        "hourly": ["temperature_2m", "precipitation", "wind_speed_10m"],
        "timezone": "Asia/Seoul"
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(2).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )}
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["precipitation_mm"] = hourly_precipitation
    hourly_data["wind_speed_m_s"] = hourly_wind_speed_10m

    df = pd.DataFrame(data=hourly_data)
    df['time'] = df['date'].dt.tz_convert('Asia/Seoul')
    df.drop('date', axis=1, inplace=True)
    
    out_dir = "C:\\Sandbox\\Haeundae_WaterQuality_Project\\Data_Raw"
    out_path = os.path.join(out_dir, "haeundae_weather_2014_2026.csv")
    df.to_csv(out_path, index=False)
    print(f"Weather data saved to {out_path} ({len(df)} records)")

if __name__ == "__main__":
    fetch_weather()
