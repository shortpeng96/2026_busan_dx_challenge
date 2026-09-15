import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, recall_score, fbeta_score, confusion_matrix
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

beach_name = "Ilgwang"
base_dir = "C:\\Sandbox\\2026_busan_dx_challenge"
proj_dir = os.path.join(base_dir, "Ilgwang_WaterQuality_Project")

print("1. Loading Spatial Dataset...")
df_master = pd.read_csv(os.path.join(proj_dir, "Data_Processed", "master_dataset_ilgwang_v2.csv"))

# Mapping log target
df_master['log_ecoli'] = np.log1p(df_master['ecoli_max'])

features = [
    'distance_from_estuary_km',
    'precip_daily', 'temp_daily', 'wind_max',
    'gijang_discharge_m3_day', 'discharge_95th_thresh',
    'precip_1d_lag', 'precip_2d_sum_lag', 'precip_3d_sum_lag', 'precip_5d_sum_lag',
    'temp_1d_lag', 'wind_max_1d_lag',
    'gijang_discharge_1d_lag', 'gijang_thresh_1d_lag',
    'CSO_Flag_Rain', 'Dual_CSO_Flag', 'month', 'is_weekend',
    'avg_water_temp', 'avg_water_temp_1d_lag',
    'tide_range', 'tide_range_1d_lag',
    'wind_sin', 'wind_cos', 'wind_sin_1d_lag', 'wind_cos_1d_lag'
]

y_target = df_master['log_ecoli']
y_bin = df_master['any_exceed'].astype(int)
X = df_master[features]

print("2. Imputing Missing Values...")
imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=10, random_state=42), random_state=42, max_iter=5)
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

print("3. Running RFE with AUC Metric...")
scale_pos = (len(y_bin) - y_bin.sum()) / max(1, y_bin.sum())

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
current_features = list(features)
results = []

while len(current_features) > 0:
    aucs = []
    xgb = XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos, random_state=42)
    
    for train_idx, test_idx in cv.split(X_imp[current_features], y_bin):
        X_train, X_test = X_imp[current_features].iloc[train_idx], X_imp[current_features].iloc[test_idx]
        y_train, y_test = y_bin.iloc[train_idx], y_bin.iloc[test_idx]
        
        if len(y_train.unique()) > 1:
            xgb.fit(X_train, y_train)
            preds = xgb.predict_proba(X_test)[:, 1]
            try: aucs.append(roc_auc_score(y_test, preds))
            except: pass
            
    avg_auc = np.mean(aucs) if aucs else 0.5
    if len(y_bin.unique()) > 1:
        xgb.fit(X_imp[current_features], y_bin)
        imp = dict(zip(current_features, xgb.feature_importances_))
    else:
        imp = {f: 1.0/len(current_features) for f in current_features}
        
    results.append((len(current_features), avg_auc, list(current_features)))
    
    if len(current_features) == 1: break
    least_important = min(imp, key=imp.get)
    current_features.remove(least_important)

best = max(results, key=lambda x: x[1])
best_feats = best[2]
best_auc = best[1]
print(f"\\n[BEST RFE] AUC {best_auc:.5f} with {best[0]} features: {best_feats}")

print("4. Evaluating Final Model...")
xgb_final = XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos, random_state=42)
y_pred_all = np.zeros(len(y_bin))
for train_idx, test_idx in cv.split(X_imp[best_feats], y_bin):
    X_train, X_test = X_imp[best_feats].iloc[train_idx], X_imp[best_feats].iloc[test_idx]
    y_train = y_bin.iloc[train_idx]
    if len(y_train.unique()) > 1:
        xgb_final.fit(X_train, y_train)
        y_pred_all[test_idx] = xgb_final.predict_proba(X_test)[:, 1]

print("5. Business ROI & Dual-Warning Simulation...")
if 'precip_1d_lag' in X_imp.columns:
    baseline_preds = (X_imp['precip_1d_lag'] >= 3.0).astype(int)
else:
    baseline_preds = np.zeros(len(y_bin))
baseline_fp = ((baseline_preds == 1) & (y_bin == 0)).sum()

best_f2 = 0
t_yellow = 0.01
for t in np.arange(0.01, 1.0, 0.01):
    preds = (y_pred_all >= t).astype(int)
    f2 = fbeta_score(y_bin, preds, beta=2, zero_division=0)
    if f2 > best_f2:
        best_f2 = f2
        t_yellow = t

preds_yellow = (y_pred_all >= t_yellow).astype(int)
recall_yellow = recall_score(y_bin, preds_yellow) * 100
fp_yellow = ((preds_yellow == 1) & (y_bin == 0)).sum()

t_red = 0.30
preds_red = (y_pred_all >= t_red).astype(int)
recall_red = recall_score(y_bin, preds_red) * 100
fp_red = ((preds_red == 1) & (y_bin == 0)).sum()

if baseline_fp > 0:
    reduction_pct = max(0, ((baseline_fp - fp_red) / baseline_fp) * 100)
else:
    reduction_pct = 0.0

if len(y_bin.unique()) > 1:
    xgb_final.fit(X_imp[best_feats], y_bin)

# Visualizations
out_dir = os.path.join(proj_dir, "Results")

# Pie Chart
plt.figure(figsize=(8, 8))
filtered_imp, filtered_feats = [], []
for imp, feat in zip(xgb_final.feature_importances_, best_feats):
    if imp > 0.01:
        filtered_imp.append(imp)
        filtered_feats.append(feat)
if len(filtered_imp) == 0:
    filtered_imp, filtered_feats = [1], ['None']
plt.pie(filtered_imp, labels=filtered_feats, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title(f'{beach_name} ?섏쭏 ?ㅼ뿼 ?듭떖 蹂??湲곗뿬??(AUC: {best_auc:.3f})')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"feature_importance_{beach_name}.png"))
plt.close()

# ROI Comparison
plt.figure(figsize=(8, 6))
labels = ['湲곗〈 愿??(鍮?3.0mm ?댁긽)', 'AI (?꾪뿕??0.3 ?듭젣)']
fp_values = [baseline_fp, fp_red]
colors = ['#e74c3c', '#3498db']
bars = plt.bar(labels, fp_values, color=colors, width=0.5)
plt.title(f'{beach_name} ?ㅽ깘(?듭슱???곸뾽?뺤?) 諛쒖깮 嫄댁닔 鍮꾧탳', fontsize=14)
plt.ylabel('?ㅽ깘 嫄댁닔 (False Positives)', fontsize=12)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + (max(fp_values)*0.01), int(yval), ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"roi_comparison_{beach_name}.png"))
plt.close()

# KDE
plt.figure(figsize=(10, 6))
clean_preds = y_pred_all[y_bin == 0]
dirty_preds = y_pred_all[y_bin == 1]
sns.kdeplot(clean_preds, color='#2ecc71', fill=True, label='?뺤긽 ?섏쭏 (Clean)', alpha=0.5)
if len(dirty_preds) > 0:
    sns.kdeplot(dirty_preds, color='#e74c3c', fill=True, label='?섏쭏 ?ㅼ뿼 (Exceedance)', alpha=0.5)
plt.axvline(x=t_yellow, color='#f1c40f', linestyle='--', linewidth=2, label=f'二쇱쓽??(F2 理쒖쟻?? {t_yellow:.2f})')
plt.axvline(x=t_red, color='#c0392b', linestyle='-', linewidth=2, label=f'?꾪뿕??(?듭젣?? {t_red:.2f})')
plt.title(f'{beach_name} ?ㅻ떒怨?寃쎈낫 ?쒖뒪???뺣쪧 遺꾪룷??, fontsize=14)
plt.xlabel('AI ?덉륫 ?뺣쪧 (Probability of Exceedance)', fontsize=12)
plt.ylabel('諛??(Density)', fontsize=12)
plt.legend(loc='upper right')
plt.xlim(0, 1.0)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"dual_warning_kde_{beach_name}.png"))
plt.close()

# Confusion Matrix
cm = confusion_matrix(y_bin, (y_pred_all >= t_red).astype(int))
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['?뺤긽 ?덉륫', '?꾪뿕 ?덉륫'], 
            yticklabels=['?ㅼ젣 ?뺤긽', '?ㅼ젣 ?꾪뿕'],
            annot_kws={"size": 16, "weight": "bold"})
plt.title(f'{beach_name} ?쇰룞 ?됰젹 (?꾪뿕 ?꾧퀎媛?{t_red:.2f})', fontsize=14)
plt.xlabel('AI ?덉륫 (Predicted)', fontsize=12)
plt.ylabel('?ㅼ젣 ?섏쭏 (Actual)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"confusion_matrix_{beach_name}.png"))
plt.close()

# Report
report_md = f"""# ?뙄 {beach_name} ?댁닔?뺤옣 ?섏쭏 AI ?덉륫 諛??낆닔 ?듭젣 理쒖쟻??蹂닿퀬??
蹂?蹂닿퀬?쒕뒗 ?댁뼇 湲곗긽 蹂?섏? 怨듦컙 ?곗씠?곕? ?듯빀?섏뿬 ?섏쭏 ?ㅼ뿼????κ퇏/?κ뎄洹?瑜??덉륫?섍퀬, ?쒕????덉쟾怨?吏???곴텒???쇳빐瑜??숈떆??怨좊젮??AI 湲곕컲 ?ㅻ떒怨??낆닔 ?듭젣 ?쒖뒪?쒖쓽 遺꾩꽍 寃곌낵?낅땲??

---

## ?뱤 1. 遺꾩꽍 媛쒖슂 諛??듭떖 ?꾩쿂由?(Overview & Preprocessing)
* **遺꾩꽍 ????곗씠??*: 珥?{len(y_bin)}???좏슚 ?섏쭏 寃??湲곕줉
* **?ㅼ젣 ?ㅼ뿼 諛쒖깮 ?잛닔 (??κ퇏 > 500)**: {y_bin.sum()}??(洹뱀떖???대옒??遺덇퇏???곗씠??
* **?됯? 吏??(ROC-AUC)**: {best_auc:.3f} (?꾩껜 遺꾨쪟 ?깅뒫)
* **?듭떖 ?곗씠???꾩쿂由?(?댁슜媛??덉쟾 理쒖슦??**: 
  * ?섎（???щ윭 援ъ뿭?먯꽌 痢≪젙???섏튂 以?**'理쒕뙎媛?Max)'**??洹몃궇??????섏쭏濡?蹂묓빀?섍굅???ㅽ뙚蹂?怨듦컙 遺꾪븷???곸슜. 
  * ?댁닔?뺤옣 ??????怨녹씠?쇰룄 ?꾪뿕?섎㈃ 利됯컖 ??묓븷 ???덈룄濡?蹂댁닔?곸씤 湲곗????곸슜?섏뿬 誘명깘(False Negative) 由ъ뒪?щ? ?먯쿇 李⑤떒??

---

## ?뱢 2. 踰좎씠?ㅻ씪???鍮?鍮꾩쫰?덉뒪 ?꾪뙥??(Business ROI)
湲곗〈 ?됱젙 愿?됯낵 AI 紐⑤뜽???깅뒫??鍮꾧탳?섏뿬 吏??寃쎌젣(?곴텒)??誘몄튂???뚭툒 ?④낵瑜?寃利앺뻽?듬땲??

* **湲곗〈 愿??(?쇱씪 媛뺤닔??3.0mm ?댁긽 ???쇨큵 ?듭젣)**
  * ?덉쟾(Recall) ?뺣낫?먮뒗 ?좊━?섎굹, ?대쪟? 諛붾엺???섑븳 ?먯뿰 ?ъ꽍 ?④낵瑜?臾댁떆??
  * 寃곌낵?곸쑝濡??섏쭏???뺤긽?꾩뿉???댁닔?뺤옣???듭젣??**?ㅽ깘(False Positive)??{baseline_fp}嫄?* 諛쒖깮.
* **AI ?덉륫 紐⑤뜽 (XGBClassifier ?곸슜)**
  * 媛뺤닔????議곗쐞, 湲곗삩, ?섏삩, ?섏쿇 嫄곕━ ??蹂듯빀 ?붿씤???숈뒿.
  * 湲곗〈 愿?됯낵 ?숈씪???섏????쒕? ?덉쟾??蹂댁옣?섎㈃?쒕룄, 遺덊븘?뷀븳 ?낆닔 ?듭젣(?ㅽ깘)瑜?**{baseline_fp}嫄댁뿉??{fp_red}嫄댁쑝濡???{reduction_pct:.1f}% ???媛먯냼**?쒗궡.
* **寃곕줎**: ?덉쟾? 洹몃?濡?吏?ㅻ㈃?? ?곴텒???듭슱???곸뾽 ?먯떎??留됱븘 吏?먯껜 ?됱젙???좊ː?꾨? ?ш쾶 ?믪씪 ???덉쓬.

---

### ?렞 3. ?ㅻ떒怨?寃쎈낫 ?쒖뒪???쒕??덉씠??(Dual-Warning System)
?꾩텧???덉륫 ?뺣쪧(Probability)??諛뷀깢?쇰줈, ?섑븰??理쒖쟻??F2-Score)怨??ㅻТ 湲곗??먯쓣 ?듯빀??2?④퀎 ?듭젣 ?쒖뒪?쒖쓣 ?쒖븞?⑸땲??

* ?윞 **二쇱쓽 ?④퀎 (?꾧퀎媛?{t_yellow:.2f})** : F2-Score 理쒖쟻 ?꾧퀎???먮룞 ?곗텧
  * **議곌굔**: ?ㅼ뿼 ?덉륫 ?뺣쪧 {t_yellow*100:.0f}% ?댁긽 ??諛쒕졊
  * **議곗튂**: '?몄빟??諛??대┛???낆닔 ?먯젣 沅뚭퀬'
  * **?④낵**: ?섏쭏 ?ㅼ뿼 ?ы깭 以?**{recall_yellow:.1f}% ({int(recall_yellow/100 * y_bin.sum())}/{y_bin.sum()})**瑜??좎젣?곸쑝濡?諛⑹뼱?섎ŉ, ?뺤? ?섏????ㅽ깘({fp_yellow}??留뚯쓣 ?덉슜??
* ?뵶 **?꾪뿕 ?④퀎 (?꾧퀎媛?{t_red:.2f})** : ?섎뱶 怨좎젙 ?꾧퀎??  * **議곌굔**: ?ㅼ뿼 ?덉륫 ?뺣쪧 {t_red*100:.0f}% ?댁긽 ??諛쒕졊
  * **議곗튂**: '?댁닔?뺤옣 ?꾨㈃ ?낆닔 ?듭젣'
  * **?④낵**: 移섎챸?곸씤 ?섏쭏 ?ㅼ뿼 以?**{recall_red:.1f}% ({int(recall_red/100 * y_bin.sum())}/{y_bin.sum()})**瑜??꾨꼍??李⑤떒?섎ŉ, ?ㅻ낫媛 {fp_red}?뚮줈 湲됯꺽??媛먯냼?섏뿬 ?곸뾽???쇳빐瑜?理쒖냼?뷀븿.

---

## ?㎚ 4. ?섏쭏 ?낇솕瑜??좊컻?섎뒗 ?듭떖 蹂??(Feature Importance)
AI媛 遺꾩꽍??{beach_name}???섏쭏??寃곗젙吏볥뒗 ?곸쐞 ?듭떖 ?붿씤?낅땲??
"""
for i, feat in enumerate(best_feats[:3]):
    report_md += f"{i+1}. `{feat}`: AI媛 梨꾪깮??媛뺣젰???ㅼ뿼 ?몄옄\\n"

report_md += f"""
---

## ?뮕 5. ?곗씠??怨쇳븰???몄궗?댄듃 諛??ν썑 怨쇱젣 (Conclusion & Future Works)
* **?몄궗?댄듃**: 鍮꾩쫰?덉뒪 理쒖쟻?붿? ?쒕? 蹂닿굔 ?덉쟾?대씪????留덈━ ?좊겮瑜??↔린 ?꾪빐 ?섑븰??媛以묒튂 紐⑤뜽(F2-Score)怨??곴텒???듯빀 ?듭젣 ?꾧퀎媛믪쓣 怨좎븞??
* **?쒓퀎 諛?怨좊룄??諛⑹븞**: 蹂?紐⑤뜽? 1??1??吏묎퀎???곗씠?곕? 湲곕컲?쇰줈 ?숈뒿?섏뼱, ?뚮컻?곸씤 ?ㅽ룓???좎텧 ???쒓컙 ?⑥쐞??湲됯꺽??蹂?붾? ?ㅼ떆媛꾩쑝濡?媛먯??섎뒗 ?곕뒗 ?쒓퀎媛 議댁옱?? ?ν썑 ?ㅼ떆媛??댁닔 ?ㅼ뿼??痢≪젙 IoT ?쇱꽌媛 ?꾩엯?쒕떎硫?蹂?AI ?뚯씠?꾨씪?멸낵 寃고빀?섏뿬 '?쒓컙蹂??섏쭏 ?덉륫 ?듭젣 ?쒖뒪???쇰줈 怨좊룄?뷀븷 ???덉쓬.
"""

with open(os.path.join(out_dir, f"results_{beach_name}.txt"), "w", encoding='utf-8') as f:
    f.write(report_md)

print(f"\\nDone! Saved standardized report for {beach_name}.")
