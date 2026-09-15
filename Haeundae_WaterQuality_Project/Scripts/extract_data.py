import pandas as pd, os, glob

BASE = 'c:/Sandbox/2026_busan_dx_challenge/Haeundae_WaterQuality_Project'
RAW = os.path.join(BASE, 'Data_Raw')
NEW_RAW = os.path.join(BASE, 'Data_Raw_New')

print('Extracting Visitors...')
v_dir = os.path.join(RAW, 'Visitors')
visitor_files = glob.glob(os.path.join(v_dir, '*.csv'))
df_visitors_list = []
for f in visitor_files:
    v = pd.read_csv(f, encoding='utf-8-sig')
    name_col = v.columns[6]
    count_col = v.columns[7]
    date_col = v.columns[8]
    
    # Filter by Haeundae
    # We can check if '해운대'.encode('utf-8') matches, or just use substring
    # To avoid encoding issues in string literals, we can use bytes:
    haeundae_str = b'\xed\x95\xb4\xec\x9a\xb4\xeb\x8c\x80'.decode('utf-8')
    v = v[v[name_col].astype(str).str.contains(haeundae_str, na=False)]
    v['date'] = pd.to_datetime(v[date_col]).dt.strftime('%Y-%m-%d')
    v = v.rename(columns={count_col: 'visitor_count'})
    v = v[['date', 'visitor_count']]
    df_visitors_list.append(v)
if df_visitors_list:
    df_visitor = pd.concat(df_visitors_list, ignore_index=True)
    df_visitor['date'] = pd.to_datetime(df_visitor['date'])
    df_visitor = df_visitor.sort_values('date').drop_duplicates(subset=['date'])
    df_visitor.to_csv(os.path.join(NEW_RAW, '해운대_방문객.csv'), index=False)

print('Extracting Sewage...')
f_suyeong = glob.glob(os.path.join(RAW, 'Discharge', '*수영*.csv'))
if f_suyeong:
    df_sy = pd.read_csv(f_suyeong[0], encoding='cp949')
    df_sy['date'] = pd.to_datetime(df_sy.iloc[:, 0].astype(str), format='%Y%m%d')
    df_sy['sewage_discharge'] = df_sy.iloc[:, 1]
    df_sy[['date', 'sewage_discharge']].to_csv(os.path.join(NEW_RAW, '해운대_인근_하수_방류량.csv'), index=False)

print('Creating empties...')
pd.DataFrame(columns=['date']).to_csv(os.path.join(NEW_RAW, '해운대_인근_하천_방류량.csv'), index=False)
pd.DataFrame(columns=['date']).to_csv(os.path.join(NEW_RAW, '해운대_조수.csv'), index=False)
pd.DataFrame(columns=['date']).to_csv(os.path.join(NEW_RAW, '해운대_수온.csv'), index=False)

print('Done!')
