import os
import glob

new_block = """cat_importances = {
    '기상 및 강우 요인': 0, 
    '지역 하천 및 하수 유입': 0, 
    '해양 파랑 및 수온(부이)': 0, 
    '조위(조석) 요인': 0, 
    '시간 및 기타 특성': 0
}

for feat, imp in zip(features, importances):
    kor_name = feat_kor_map.get(feat, feat)
    
    if any(k in kor_name for k in ['부이', '파고', '파주기', '수온', '염분', '유속', '유향', '해수']):
        cat_importances['해양 파랑 및 수온(부이)'] += imp
    elif any(k in kor_name for k in ['강수', '비', '기온', '풍속', '풍향', '바람', '일사량', '건조', '기압']):
        cat_importances['기상 및 강우 요인'] += imp
    elif any(k in kor_name for k in ['방류', '하천', '하수', '월류', 'CSO']):
        cat_importances['지역 하천 및 하수 유입'] += imp
    elif any(k in kor_name for k in ['조위', '만조', '간조']):
        cat_importances['조위(조석) 요인'] += imp
    else:
        cat_importances['시간 및 기타 특성'] += imp

cat_series = pd.Series(cat_importances)
cat_series = cat_series[cat_series > 0].sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(9, 7))
colors_donut = [C_NAVY, C_BLUE, C_ORANGE, '#6c757d', '#adb5bd'][:len(cat_series)]

# Use pctdistance to push numbers into the donut ring, hide labels outside, set text color and size
wedges, texts, autotexts = ax.pie(cat_series, labels=None, autopct='%1.1f%%', pctdistance=0.75,
                                  startangle=90, colors=colors_donut, wedgeprops=dict(width=0.4, edgecolor='w'))"""

def replace_in_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    start_marker = 'category_map = {\n'
    # we want to replace from start_marker to the end of the pie chart call
    end_marker = "wedgeprops=dict(width=0.4, edgecolor='w'))"
    
    if start_marker not in content or end_marker not in content:
        print(f"Markers not found in {filepath}")
        return

    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker) + len(end_marker)
    
    new_content = content[:start_idx] + new_block + content[end_idx:]
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"Updated {filepath}")

for path in glob.glob("*_WaterQuality_Project/Scripts/03_generate_results.py"):
    replace_in_file(path)
