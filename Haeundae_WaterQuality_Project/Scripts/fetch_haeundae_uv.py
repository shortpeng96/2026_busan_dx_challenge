import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import os

def fetch_uv_data():
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 35.1587,
        "longitude": 129.1604,
        "start_date": "2014-01-01",
        "end_date": "2026-08-31",
        "daily": ["uv_index_max", "shortwave_radiation_sum"],
        "timezone": "Asia/Seoul"
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    daily = response.Daily()
    daily_uv_index_max = daily.Variables(0).ValuesAsNumpy()
    daily_shortwave_radiation_sum = daily.Variables(1).ValuesAsNumpy()

    daily_data = {"date": pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left"
    )}
    daily_data["uv_index_max"] = daily_uv_index_max
    daily_data["shortwave_radiation_sum"] = daily_shortwave_radiation_sum

    df = pd.DataFrame(data=daily_data)
    df['date'] = df['date'].dt.tz_convert('Asia/Seoul').dt.strftime('%Y-%m-%d')
    
    # Merge into Master dataset
    master_path = "C:\\Sandbox\\2026_busan_dx_challenge\\Haeundae_WaterQuality_Project\\Data_Processed\\master_dataset_haeundae.csv"
    df_master = pd.read_csv(master_path)
    
    df_master = pd.merge(df_master, df[['date', 'uv_index_max', 'shortwave_radiation_sum']], on='date', how='left')
    df_master.to_csv(master_path, index=False, encoding='utf-8-sig')
    print("Merged UV Index and Shortwave Radiation into Haeundae Master Dataset!")

if __name__ == "__main__":
    fetch_uv_data()
