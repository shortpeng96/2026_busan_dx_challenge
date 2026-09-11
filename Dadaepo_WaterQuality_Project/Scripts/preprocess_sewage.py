import pandas as pd
import numpy as np
import os

def preprocess_sewage():
    file_path = "C:\\Sandbox\\Discharge\\강변사업단 일일방류량.csv"
    
    if not os.path.exists(file_path):
        print(f"Sewage data not found at {file_path}")
        return
        
    # Read sewage data
    df = pd.read_csv(file_path, encoding='cp949')
    
    # Rename columns
    df.columns = ['date_str', 'sewage_volume', 'ph', 'bod', 'cod', 'ss', 'tn', 'tp']
    
    # Convert date
    df['date_str'] = df['date_str'].astype(str)
    df['date'] = pd.to_datetime(df['date_str'], format='%Y%m%d')
    df = df.sort_values('date').reset_index(drop=True)
    
    # We will let merge_datasets.py handle the lag and CSO Flag computation
    # because it has the weather data already processed.
    
    cols_to_keep = ['date', 'sewage_volume']
    
    out_dir = "C:\\Sandbox\\Preprocessed"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sewage_daily.csv")
    df[cols_to_keep].to_csv(out_path, index=False)
    print(f"Saved sewage data to {out_path}")

if __name__ == "__main__":
    preprocess_sewage()
