import pandas as pd
import numpy as np
import os
import glob

def clean_microbial_value(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    if not val_str:
        return np.nan
    if '<' in val_str:
        try:
            return float(val_str.replace('<', '')) * 0.5
        except:
            return 0.5
    if '>' in val_str:
        try:
            return float(val_str.replace('>', ''))
        except:
            return np.nan
    val_str = val_str.replace(',', '').replace(' ', '')
    try:
        return float(val_str)
    except:
        return np.nan

def get_distance(row):
    detail = str(row['examinLcDetail']).upper()
    
    if 'A' in detail:
        return 0.0
    elif 'B' in detail:
        return 0.25
    elif 'C' in detail:
        return 0.5
    elif 'D' in detail:
        return 0.75
    elif 'E' in detail:
        return 1.0
        
    return 0.5 # Default to center if unknown

def preprocess_water_quality():
    f = "C:\\Sandbox\\Water_Quality\\busan_beach_광안리.csv"
    
    try:
        df_all = pd.read_csv(f, encoding='utf-8-sig')
    except:
        df_all = pd.read_csv(f, encoding='cp949', errors='replace')
            
    # Drop rows without date
    df_clean = df_all.dropna(subset=['examinDe']).copy()
    df_clean = df_clean[df_clean['examinDe'].str.strip() != '']
    df_clean['examinDe'] = pd.to_datetime(df_clean['examinDe'], errors='coerce')
    df_clean = df_clean.dropna(subset=['examinDe'])
    
    # Clean microbial values
    df_clean['ecoli_max'] = df_clean['coliDetectCn'].apply(clean_microbial_value)
    df_clean['enterococcus_max'] = df_clean['entrcccsDetectCn'].apply(clean_microbial_value)
    
    # Exceedance Labels
    df_clean['ecoli_exceed'] = df_clean['ecoli_max'] > 500
    df_clean['enterococcus_exceed'] = df_clean['enterococcus_max'] > 100
    df_clean['any_exceed'] = df_clean['ecoli_exceed'] | df_clean['enterococcus_exceed']
    
    # Add Spatial Feature
    df_clean['distance_from_estuary_km'] = df_clean.apply(get_distance, axis=1)
    
    # Keep essential columns for modeling
    cols_to_keep = ['examinDe', 'beachKoreanNm', 'examinLcDetail', 'distance_from_estuary_km', 
                    'ecoli_max', 'enterococcus_max', 'any_exceed']
    
    df_samples = df_clean[cols_to_keep].copy()
    
    print(f"Total Exceedances (Sample level): {df_samples['any_exceed'].sum()} out of {len(df_samples)} samples")
    
    # Save
    out_dir = "C:\\Sandbox\\Gwangalli_WaterQuality_Project\\Data_Processed"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "gwangalli_water_samples.csv")
    df_samples.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"Saved {len(df_samples)} spatial samples to {out_path}")

if __name__ == "__main__":
    preprocess_water_quality()
