import pandas as pd
import os

raw_path = r'C:\Sandbox\Visitors\해수욕장_일일정보.csv'
out_path = r'C:\Sandbox\2026_busan_dx_challenge\Gwangalli_WaterQuality_Project\Data_Raw\광안리_방문객.csv'

try:
    df = pd.read_csv(raw_path, encoding='cp949')
except:
    df = pd.read_csv(raw_path, encoding='utf-8')

print("Original Columns:", df.columns.tolist())

beach_col = df.columns[0]
date_col = df.columns[2]
visitor_col = df.columns[3]

gwangalli_df = df[df[beach_col].astype(str).str.contains('광안리')].copy()

if gwangalli_df.empty:
    print("Warning: No Gwangalli data found!")
else:
    gwangalli_df = gwangalli_df[[date_col, visitor_col]]
    gwangalli_df.columns = ['date', 'visitor_count']
    
    gwangalli_df['date'] = pd.to_datetime(gwangalli_df['date'])
    gwangalli_df = gwangalli_df.sort_values('date').dropna(subset=['date'])
    gwangalli_df['visitor_count'] = pd.to_numeric(gwangalli_df['visitor_count'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    
    gwangalli_df = gwangalli_df.groupby('date', as_index=False)['visitor_count'].sum()
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    gwangalli_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"Saved {len(gwangalli_df)} records to {out_path}")
