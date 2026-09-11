import pandas as pd
import numpy as np
import os

def preprocess_sewage():
    f = "C:\\Sandbox\\Discharge\\수영사업단 일일방류량.csv"
    try:
        df = pd.read_csv(f, encoding='utf-8-sig')
    except:
        df = pd.read_csv(f, encoding='cp949')
        
    date_col = df.columns[0]
    vol_col = df.columns[1]
    
    df_clean = df[[date_col, vol_col]].copy()
    df_clean.columns = ['date', 'sewage_volume']
    
    df_clean['date'] = df_clean['date'].astype(str).str.replace('-', '').str.replace('.', '').str[:8]
    df_clean['date'] = pd.to_datetime(df_clean['date'], format='%Y%m%d', errors='coerce')
    df_clean = df_clean.dropna(subset=['date'])
    
    # Convert string volumes with commas to float
    def clean_vol(v):
        if pd.isna(v): return np.nan
        try: return float(str(v).replace(',', ''))
        except: return np.nan
        
    df_clean['sewage_volume'] = df_clean['sewage_volume'].apply(clean_vol)
    
    out_dir = "C:\\Sandbox\\Gwangalli_WaterQuality_Project\\Data_Processed"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "suyeong_sewage_daily.csv")
    df_clean.to_csv(out_path, index=False, encoding='utf-8-sig')
    
    print(f"Processed Suyeong sewage data: {len(df_clean)} records")
    print(df_clean.head())

if __name__ == "__main__":
    preprocess_sewage()
