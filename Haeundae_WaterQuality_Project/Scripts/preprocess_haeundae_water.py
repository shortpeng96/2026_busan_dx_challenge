import pandas as pd
import numpy as np
import os

def clean_val(v):
    if pd.isna(v): return np.nan
    v_str = str(v).replace(',', '').replace('<', '').replace('>', '').replace(' ', '')
    try: return float(v_str)
    except: return np.nan

def map_location_to_distance(detail):
    d = str(detail).upper()
    
    # West edge (0.0)
    if 'A' in d or '조선비치' in d:
        return 0.0
    # West-mid (0.25)
    elif 'B' in d or '그랜드호텔' in d or '아쿠아리움' in d:
        return 0.25
    # Center (0.5)
    elif 'C' in d:
        return 0.5
    # East-mid (0.75)
    elif 'D' in d or '파라다이스' in d or '노보텔' in d:
        return 0.75
    # East edge (1.0)
    elif 'E' in d or '미포' in d:
        return 1.0
        
    return np.nan # Unknown locations will be dropped

def preprocess_water():
    f_path = "C:\\Sandbox\\Water_Quality\\busan_beach_해운대.csv"
    try:
        df = pd.read_csv(f_path, encoding='utf-8-sig')
    except:
        df = pd.read_csv(f_path, encoding='cp949', errors='replace')
        
    # Standardize column names
    col_map = {
        '조사일자': 'examinDe',
        '해수욕장명': 'beachKoreanNm',
        '조사위치상세': 'examinLcDetail',
        '대장균': 'coliDetectCn',
        '장구균': 'entrcccsDetectCn'
    }
    
    # Check if English names exist, else map Korean
    if 'examinDe' not in df.columns:
        df.rename(columns=col_map, inplace=True)
        
    df = df[['examinDe', 'beachKoreanNm', 'examinLcDetail', 'coliDetectCn', 'entrcccsDetectCn']].copy()
    
    df['examinDe'] = pd.to_datetime(df['examinDe'].astype(str).str.replace('-', '').str.replace('.', '').str[:8], format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['examinDe'])
    
    df['coliDetectCn'] = df['coliDetectCn'].apply(clean_val)
    df['entrcccsDetectCn'] = df['entrcccsDetectCn'].apply(clean_val)
    
    df['distance_from_estuary_km'] = df['examinLcDetail'].apply(map_location_to_distance)
    df = df.dropna(subset=['distance_from_estuary_km'])
    
    # Aggregate by Date and Location
    df_grouped = df.groupby(['examinDe', 'beachKoreanNm', 'examinLcDetail', 'distance_from_estuary_km']).agg({
        'coliDetectCn': 'max',
        'entrcccsDetectCn': 'max'
    }).reset_index()
    
    df_grouped.rename(columns={'coliDetectCn': 'ecoli_max', 'entrcccsDetectCn': 'enterococcus_max'}, inplace=True)
    
    # Define Exceedance (Official MoF Standard)
    df_grouped['ecoli_exceed'] = df_grouped['ecoli_max'] > 500
    df_grouped['enterococcus_exceed'] = df_grouped['enterococcus_max'] > 100
    df_grouped['any_exceed'] = df_grouped['ecoli_exceed'] | df_grouped['enterococcus_exceed']
    
    df_samples = df_grouped[['examinDe', 'beachKoreanNm', 'examinLcDetail', 'distance_from_estuary_km',
                    'ecoli_max', 'enterococcus_max', 'any_exceed']]
                    
    print(f"Total Haeundae samples mapped: {len(df_samples)}")
    print(f"Total Exceedances: {df_samples['any_exceed'].sum()} out of {len(df_samples)} samples")
    
    out_dir = "C:\\Sandbox\\Haeundae_WaterQuality_Project\\Data_Processed"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "haeundae_water_samples.csv")
    df_samples.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    preprocess_water()
