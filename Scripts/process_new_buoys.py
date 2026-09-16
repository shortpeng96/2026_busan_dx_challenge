import pandas as pd
import os

BASE = 'c:/Sandbox/2026_busan_dx_challenge'
RAW_DIR = os.path.join(BASE, '부이데이터')

def process_buoy(filename, prefix):
    filepath = os.path.join(RAW_DIR, filename)
    if not os.path.exists(filepath):
        print(f"File not found: {filename}")
        return None
    try:
        df = pd.read_csv(filepath, encoding='cp949')
    except:
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
        except:
            print(f"Could not read {filename}")
            return None
    
    date_cols = [c for c in df.columns if '일시' in c]
    if not date_cols:
        return None
    date_col = date_cols[0]
    
    df['date'] = pd.to_datetime(df[date_col]).dt.normalize()
    
    for c in df.columns:
        if c not in [date_col, 'date', '지점명', '지점코드', '풍향(16points)', '유향(16points)']:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace('-', ''), errors='coerce')
    
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    agg_funcs = {c: 'mean' for c in numeric_cols}
    
    # Custom aggregations
    for c in ['최대파고(m)', '유의파고(m)', '풍속(m/s)']:
        if c in agg_funcs:
            agg_funcs[c] = 'max'
            
    df_daily = df.groupby('date').agg(agg_funcs).reset_index()
    df_daily.rename(columns={c: f"{prefix}_{c}" for c in numeric_cols}, inplace=True)
    return df_daily

dfs = {}
for buoy in ['송정_부이데이터.csv', '감천항_부이데이터.csv', '부산신항_부이데이터.csv', '부산항_부이데이터.csv', '임랑해수욕장_부이데이터.csv', '해운대해수욕장_부이데이터.csv']:
    name = buoy.split('_')[0]
    print(f"Processing {name}...")
    dfs[name] = process_buoy(buoy, name)

# 1. Dadaepo: Gamcheon + Busan New Port
if dfs.get('감천항') is not None and dfs.get('부산신항') is not None:
    df_dadaepo = pd.merge(dfs['감천항'], dfs['부산신항'], on='date', how='outer')
    # 분리 적용: 평균(Mean) 파생 변수 생성을 제거하고, 각 항구별 원본 컬럼을 그대로 유지합니다.
    out_dir = os.path.join(BASE, 'Dadaepo_WaterQuality_Project', 'Data_Raw')
    os.makedirs(out_dir, exist_ok=True)
    df_dadaepo.to_csv(os.path.join(out_dir, '다대포_부이데이터.csv'), index=False)
    print("Saved Dadaepo")

# 2. Songdo: Gamcheon + Busan Port
if dfs.get('감천항') is not None and dfs.get('부산항') is not None:
    df_songdo = pd.merge(dfs['감천항'], dfs['부산항'], on='date', how='outer')
    # 분리 적용: 평균(Mean) 파생 변수 생성을 제거하고, 각 항구별 원본 컬럼을 그대로 유지합니다.
    out_dir = os.path.join(BASE, 'Songdo_WaterQuality_Project', 'Data_Raw')
    os.makedirs(out_dir, exist_ok=True)
    df_songdo.to_csv(os.path.join(out_dir, '송도_부이데이터.csv'), index=False)
    print("Saved Songdo")

# 3. Songjeong
if dfs.get('송정') is not None:
    out_dir = os.path.join(BASE, 'Songjeong_WaterQuality_Project', 'Data_Raw')
    os.makedirs(out_dir, exist_ok=True)
    dfs['송정'].to_csv(os.path.join(out_dir, '송정_부이데이터.csv'), index=False)
    print("Saved Songjeong")

# 4. Gwangalli
# Uses Haeundae + Busan Port
if dfs.get('부산항') is not None and dfs.get('해운대해수욕장') is not None:
    df_gwangalli = pd.merge(dfs['해운대해수욕장'], dfs['부산항'], on='date', how='outer')
    out_dir = os.path.join(BASE, 'Gwangalli_WaterQuality_Project', 'Data_Raw')
    os.makedirs(out_dir, exist_ok=True)
    df_gwangalli.to_csv(os.path.join(out_dir, '광안리_부이데이터.csv'), index=False)
    print("Saved Gwangalli")
elif dfs.get('부산항') is not None:
    out_dir = os.path.join(BASE, 'Gwangalli_WaterQuality_Project', 'Data_Raw')
    os.makedirs(out_dir, exist_ok=True)
    dfs['부산항'].to_csv(os.path.join(out_dir, '광안리_부이데이터.csv'), index=False)
    print("Saved Gwangalli (Busan Port only)")

# 5. Imrang
if dfs.get('임랑해수욕장') is not None:
    out_dir = os.path.join(BASE, 'Imrang_WaterQuality_Project', 'Data_Raw')
    os.makedirs(out_dir, exist_ok=True)
    dfs['임랑해수욕장'].to_csv(os.path.join(out_dir, '임랑_부이데이터.csv'), index=False)
    print("Saved Imrang")

# 6. Haeundae
if dfs.get('해운대해수욕장') is not None:
    out_dir = os.path.join(BASE, 'Haeundae_WaterQuality_Project', 'Data_Raw')
    os.makedirs(out_dir, exist_ok=True)
    dfs['해운대해수욕장'].to_csv(os.path.join(out_dir, '해운대_부이데이터.csv'), index=False)
    print("Saved Haeundae")

# 7. Ilgwang (Imrang + Songjeong)
if dfs.get('임랑해수욕장') is not None and dfs.get('송정') is not None:
    df_ilgwang = pd.merge(dfs['임랑해수욕장'], dfs['송정'], on='date', how='outer')
    out_dir = os.path.join(BASE, 'Ilgwang_WaterQuality_Project', 'Data_Raw')
    os.makedirs(out_dir, exist_ok=True)
    df_ilgwang.to_csv(os.path.join(out_dir, '일광_부이데이터.csv'), index=False)
    print("Saved Ilgwang")

print("Done")
