import pandas as pd
import numpy as np
import os

def clean_val(v):
    if pd.isna(v): return np.nan
    v_str = str(v).replace(',', '').replace('<', '').replace('>', '').replace(' ', '')
    try: return float(v_str)
    except: return np.nan

def map_location_to_distance(loc_str):
    s = str(loc_str).upper()
    if '1' in s or 'A' in s:
        return 0.0 # West
    elif '2' in s or 'B' in s:
        return 0.5 # Center (Ilgwang stream)
    elif '3' in s or 'C' in s:
        return 1.0 # East
    # Discard anything else like 4, 5, 6 as they might be offshore points
    return np.nan

def preprocess_water():
    # Use the true master file that contains all data and cleaner strings
    f_path = "C:\\Sandbox\\Water_Quality\\busan_beach_water_quality_all.csv"
    df = pd.read_csv(f_path, encoding='utf-8')
    df.columns = df.columns.str.replace('\ufeff', '')
    
    # Filter for Ilgwang
    df = df[df['inspecArea'].str.contains('일광')].copy()
    
    df['date'] = pd.to_datetime(df['sample_date'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    # water01 is Enterococcus, water02 is E.coli (discovered during cross-validation)
    df['ecoli'] = df['water02'].apply(clean_val)
    df['entero'] = df['water01'].apply(clean_val)
    
    df['distance_from_estuary_km'] = df['inspecArea'].apply(map_location_to_distance)
    df = df.dropna(subset=['distance_from_estuary_km'])
    
    # Aggregate by Date and Location
    df_grouped = df.groupby(['date', 'distance_from_estuary_km']).agg({
        'ecoli': 'max',
        'entero': 'max'
    }).reset_index()
    
    df_grouped.rename(columns={'ecoli': 'ecoli_max', 'entero': 'enterococcus_max'}, inplace=True)
    
    # Define Exceedance (Official MoF Standard: E.coli 500, Entero 100)
    df_grouped['ecoli_exceed'] = df_grouped['ecoli_max'] > 500
    df_grouped['enterococcus_exceed'] = df_grouped['enterococcus_max'] > 100
    df_grouped['any_exceed'] = df_grouped['ecoli_exceed'] | df_grouped['enterococcus_exceed']
    
    print(f"Total Ilgwang samples mapped: {len(df_grouped)}")
    print(f"Total Exceedances: {df_grouped['any_exceed'].sum()} out of {len(df_grouped)} samples")
    
    out_dir = "C:\\Sandbox\\Ilgwang_WaterQuality_Project\\Data_Processed"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ilgwang_water_samples.csv")
    df_grouped.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    preprocess_water()
