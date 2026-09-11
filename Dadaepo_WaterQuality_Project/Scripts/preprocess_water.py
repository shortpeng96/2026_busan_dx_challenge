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
    val_str = val_str.replace(',', '')
    try:
        return float(val_str)
    except:
        return np.nan

def get_distance(row):
    detail = str(row['examinLcDetail']).upper()
    beach = str(row['beachKoreanNm']).upper()
    
    # 다대포 서측(강이랑 가까운 해변)은 기본 거리 0.0km 시작
    if '서측' in beach:
        base_dist = 0.0
    # 다대포 동측(강이랑 먼 해변)은 기본 거리 0.5km 시작
    else: 
        base_dist = 0.5
        
    # 각 해변 내에서의 세부 지점(A,B,C) 더하기
    if 'A' in detail or '우측' in detail:
        return base_dist + 0.0
    elif 'B' in detail or '중앙' in detail:
        return base_dist + 0.25
    elif 'C' in detail or '좌측' in detail:
        return base_dist + 0.5
        
    return base_dist + 0.25 # Default to center if unknown

def preprocess_water_quality():
    # Use glob to avoid encoding issues with korean filenames in different environments
    files = glob.glob("C:\\Sandbox\\Water_Quality\\*다대포*.csv")
    
    df_list = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding='utf-8-sig')
        except:
            df = pd.read_csv(f, encoding='cp949')
        df_list.append(df)
            
    if not df_list:
        print("No Dadaepo water quality files found.")
        return
        
    df_all = pd.concat(df_list, ignore_index=True)
    
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
    
    # We DO NOT groupby here anymore! We keep all 305 sample rows.
    df_samples = df_clean[cols_to_keep].copy()
    
    print(f"Total Exceedances (Sample level): {df_samples['any_exceed'].sum()} out of {len(df_samples)} samples")
    
    # Save
    out_dir = "C:\\Sandbox\\Preprocessed"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dadaepo_water_samples.csv")
    df_samples.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"Saved {len(df_samples)} spatial samples to {out_path}")

if __name__ == "__main__":
    preprocess_water_quality()
