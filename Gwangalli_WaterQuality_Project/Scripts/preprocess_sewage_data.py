import pandas as pd
import numpy as np
import os

def clean_vol(v):
    if pd.isna(v): return np.nan
    try: return float(str(v).replace(',', ''))
    except: return np.nan

def process_file(f_path, out_name):
    try:
        df = pd.read_csv(f_path, encoding='utf-8-sig')
    except:
        df = pd.read_csv(f_path, encoding='cp949')
        
    date_col = df.columns[0]
    vol_col = df.columns[1]
    
    df_clean = df[[date_col, vol_col]].copy()
    df_clean.columns = ['date', 'sewage_volume']
    
    # Handle int/str dates like 20260910
    df_clean['date'] = df_clean['date'].astype(str).str.replace('-', '').str.replace('.', '').str[:8]
    df_clean['date'] = pd.to_datetime(df_clean['date'], format='%Y%m%d', errors='coerce')
    df_clean = df_clean.dropna(subset=['date'])
    
    df_clean['sewage_volume'] = df_clean['sewage_volume'].apply(clean_vol)
    
    out_dir = "C:\\Sandbox\\Gwangalli_WaterQuality_Project\\Data_Processed"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    df_clean.to_csv(out_path, index=False, encoding='utf-8-sig')
    
    print(f"Processed {out_name}: {len(df_clean)} records")
    return df_clean

def preprocess_sewage():
    f_suyeong = "C:\\Sandbox\\Discharge\\수영사업단 일일방류량.csv"
    f_nambu = "C:\\Sandbox\\Discharge\\남부사업단 일일방류량.csv"
    
    process_file(f_suyeong, "suyeong_sewage_daily.csv")
    process_file(f_nambu, "nambu_sewage_daily.csv")

if __name__ == "__main__":
    preprocess_sewage()
