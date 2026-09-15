"""
03_generate_results.py - Songdo Beach Water Quality (Advanced)
"""
import pandas as pd
import numpy as np
import os, pickle, warnings
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score, fbeta_score
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, 'Results')
os.makedirs(RES, exist_ok=True)
BEACH = 'Songdo'
BEACH_KOR = '송도'
print(f"[{BEACH}] 03_generate_results.py starting...")

pred_df = pd.read_csv(os.path.join(RES, 'predictions.csv'))
y_true = pred_df['y_true']
y_pred = pred_df['y_pred']

with open(os.path.join(RES, 'model.pkl'), 'rb') as f:
    model_data = pickle.load(f)

final_auc = model_data['auc']
t_yellow = model_data['t_yellow']
t_red = model_data['t_red']
features = model_data['features']
model = model_data['model']

preds_y = (y_pred >= t_yellow).astype(int)
preds_r = (y_pred >= t_red).astype(int)
recall_y = recall_score(y_true, preds_y, zero_division=0) * 100
recall_r = recall_score(y_true, preds_r, zero_division=0) * 100
fp_yellow = ((preds_y == 1) & (y_true == 0)).sum()
fp_red = ((preds_r == 1) & (y_true == 0)).sum()

PROC = os.path.join(BASE, 'Data_Processed')
df = pd.read_csv(os.path.join(PROC, 'master_dataset.csv'))
baseline_fp = ((df.get('precip_daily', pd.Series([0]*len(df)))) >= 30.0).sum()
if baseline_fp == 0: baseline_fp = max(1, int(len(df) * 0.05))

# Theme Colors
C_NAVY = '#0F2537'
C_BLUE = '#1D70B8'
C_ORANGE = '#FF6B35'
C_BG = '#F4F7F6'
C_TEXT = '#2D3748'
C_WARN = '#FFC107'

plt.rcParams['text.color'] = C_TEXT
plt.rcParams['axes.labelcolor'] = C_TEXT
plt.rcParams['xtick.color'] = C_TEXT
plt.rcParams['ytick.color'] = C_TEXT
plt.rcParams['figure.facecolor'] = C_BG
plt.rcParams['axes.facecolor'] = C_BG

# Feature Translation Dictionary (Korean)
feat_kor_map = {
    'precip_1d_lag': '1일 전 강수량',
    'precip_2d_sum_lag': '2일 누적 강수량',
    'precip_3d_sum_lag': '3일 누적 강수량',
    'precip_5d_sum_lag': '5일 누적 강수량',
    'temp_1d_lag': '1일 전 기온',
    'temp_daily': '일평균 기온',
    'wind_speed_1d_lag': '1일 전 평균 풍속',
    'wind_x_distance': '바람 확산 스코어(거리x풍속)',
    'cso_x_distance': 'CSO 확산 스코어(거리x강수)',
    'storm_intensity': '폭풍 강도(강수x바람)',
    'distance_to_outfall': '측정소까지의 거리',
    'month': '월(Month)',
    'meis_max_wave_3d_mean': '최대 파고 3일 평균',
    'meis_max_wave_7d_mean': '최대 파고 7일 평균',
    'meis_wind_speed_7d_mean': '부이 풍속 7일 평균',
    'meis_water_temp_7d_mean': '수온 7일 평균',
}

print("  Generating feature importance...")
fig, ax = plt.subplots(figsize=(10, 6))
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False).head(15).sort_values(ascending=True)

feat_imp.index = feat_imp.index.map(lambda x: feat_kor_map.get(x, x))

import matplotlib
cmap = matplotlib.colormaps['Blues']
colors_bar = [cmap(0.4 + 0.55 * (i / max(1, len(feat_imp)-1))) for i in range(len(feat_imp))]
feat_imp.plot(kind='barh', ax=ax, color=colors_bar, width=0.75, edgecolor='none')
ax.set_title(f'{BEACH_KOR} 해수욕장 - 수질 예측 핵심 변수 중요도', color=C_NAVY, weight='bold', fontsize=16)

for i, v in enumerate(feat_imp.values):
    ax.text(v + (max(feat_imp.values) * 0.01), i, f'{v:.3f}', va='center', color=C_NAVY, fontsize=11, weight='bold')

ax.xaxis.grid(True, linestyle='--', alpha=0.5, color='#adb5bd')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(RES, f'feature_importance_{BEACH}.png'), dpi=150, facecolor=fig.get_facecolor())
plt.close()

print("  Generating feature importance donut chart...")
category_map = {
    '기상 요인': ['precip_1d_lag', 'precip_2d_sum_lag', 'precip_3d_sum_lag', 'precip_5d_sum_lag', 'temp_1d_lag', 'temp_daily', 'wind_speed_1d_lag', 'storm_intensity', 'wind_x_distance', 'cso_x_distance', 'month'],
    '해양기상(부이) 요인': ['meis_max_wave_3d_mean', 'meis_max_wave_7d_mean', 'meis_wind_speed_7d_mean', 'meis_water_temp_7d_mean'],
    '지리적 요인': ['distance_to_outfall']
}
cat_importances = {'기상 요인': 0, '해양기상(부이) 요인': 0, '지리적 요인': 0}
for feat, imp in zip(features, importances):
    found = False
    for cat, feats in category_map.items():
        if feat in feats:
            cat_importances[cat] += imp
            found = True
            break
    if not found:
        cat_importances['기상 요인'] += imp

cat_series = pd.Series(cat_importances)
cat_series = cat_series[cat_series > 0].sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(9, 7))
colors_donut = [C_NAVY, C_BLUE, C_ORANGE, '#6c757d', '#adb5bd']

wedges, texts, autotexts = ax.pie(cat_series, labels=None, autopct='%1.1f%%', pctdistance=0.75,
                                  startangle=90, colors=colors_donut, wedgeprops=dict(width=0.4, edgecolor='w'))

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(14)
    autotext.set_weight('bold')

ax.text(0, 0, f'AUC\n{final_auc:.3f}', ha='center', va='center', fontsize=22, weight='bold', color=C_NAVY)
ax.legend(wedges, cat_series.index, title="원인 카테고리", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
ax.set_title(f'{BEACH_KOR} 해수욕장 - 카테고리별 오염 기여도', color=C_NAVY, weight='bold', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(RES, f'feature_importance_donut_{BEACH}.png'), dpi=150, facecolor=fig.get_facecolor())
plt.close()

print("  Generating confusion matrix...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (threshold, label, color) in zip(axes, [
    (t_yellow, f'주의 단계 (기준 점수: {t_yellow:.3f})', C_WARN),
    (t_red, f'위험 단계 (기준 점수: {t_red:.3f})', C_ORANGE),
]):
    preds = (y_pred >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=sns.light_palette(color, as_cmap=True), xticklabels=['정상', '오염'], yticklabels=['정상', '오염'])
    ax.set_title(label, fontsize=13, color=C_NAVY, weight='bold')
    ax.set_xlabel('AI 모델이 내린 판정 (예측값)', weight='bold')
    ax.set_ylabel('바다의 실제 상태 (실제값)', weight='bold')

plt.suptitle(f'{BEACH_KOR} 해수욕장 이중 기준선 혼동행렬 (Confusion Matrix)', color=C_NAVY, weight='bold', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(RES, f'confusion_matrix_{BEACH}.png'), dpi=150, facecolor=fig.get_facecolor())
plt.close()

print("  Generating KDE plot...")
fig, ax = plt.subplots(figsize=(10, 6))
y_pred_arr = np.array(y_pred)
y_true_arr = np.array(y_true)

epsilon = 1e-5
if len(y_pred_arr[y_true_arr == 0]) > 0:
    sns.kdeplot(y_pred_arr[y_true_arr == 0] + epsilon, ax=ax, color=C_BLUE, fill=True, alpha=0.5, label='정상 수질을 기록한 날의 AI 예측 점수', linewidth=2, log_scale=True)
if len(y_pred_arr[y_true_arr == 1]) > 0:
    sns.kdeplot(y_pred_arr[y_true_arr == 1] + epsilon, ax=ax, color=C_ORANGE, fill=True, alpha=0.5, label='실제 오염이 발생했던 날의 AI 예측 점수', linewidth=2, log_scale=True)

ax.axvline(t_yellow + epsilon, color=C_WARN, linestyle='--', lw=2.5, label=f'1단계: 경고 알림 발송 기준선 (점수: {t_yellow:.3f})')
ax.axvline(t_red + epsilon, color=C_ORANGE, linestyle=':', lw=2.5, label=f'2단계: 해수욕장 입수 통제 기준선 (점수: {t_red:.3f})')

ax.set_title(f'{BEACH_KOR} 해수욕장 - 정상/오염 데이터별 예측 점수 분포도 (Log Scale)', color=C_NAVY, weight='bold', fontsize=16)
ax.set_xlabel('AI가 예측한 오염 위험도 점수 (로그 스케일)', fontsize=12, color=C_TEXT, weight='bold', labelpad=10)
ax.set_ylabel('데이터 밀도', fontsize=12, color=C_TEXT, weight='bold', labelpad=10)

from matplotlib.ticker import ScalarFormatter
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.set_xlim(0.01, max(y_pred_arr.max() * 1.5, 10))

ax.legend(fontsize=10, loc='upper right', frameon=True, shadow=True)
plt.tight_layout()
plt.savefig(os.path.join(RES, f'dual_warning_kde_{BEACH}.png'), dpi=150, facecolor=fig.get_facecolor())
plt.close()

print("  Generating ROI comparison...")
from sklearn.metrics import precision_score
precision_y = precision_score(y_true, preds_y, zero_division=0) * 100
precision_r = precision_score(y_true, preds_r, zero_division=0) * 100

# Dynamic Baseline Calculation (Precipitation >= 30mm)
PROC = os.path.join(BASE, 'Data_Processed')
try:
    df = pd.read_csv(os.path.join(PROC, 'master_dataset.csv'))
except:
    df = pd.read_csv(os.path.join(PROC, 'master_dataset_v2.csv'))

baseline_preds = (df.get('precip_daily', pd.Series([0]*len(df))) >= 30.0).astype(int)
baseline_tp = ((baseline_preds == 1) & (y_true == 1)).sum()
baseline_fp_actual = ((baseline_preds == 1) & (y_true == 0)).sum()

baseline_recall = (baseline_tp / max(1, y_true.sum())) * 100
baseline_precision = (baseline_tp / max(1, (baseline_tp + baseline_fp_actual))) * 100

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

scenarios_recall = ['기존 강수량 기준', 'AI 1단계\n(주의 알림)']
recalls = [baseline_recall, recall_y]
colors_recall = ['#6c757d', C_WARN]

bars1 = axes[0].bar(scenarios_recall, recalls, color=colors_recall, alpha=0.9, edgecolor='none', width=0.6)
axes[0].set_ylim(0, 100)
axes[0].set_title('1단계 목표: 재현율 (사고 예방력) 비교', color=C_NAVY, weight='bold', fontsize=13)
axes[0].set_ylabel('재현율 (%)', weight='bold')

text_colors_recall = ['white', C_TEXT]
for bar, r, tc in zip(bars1, recalls, text_colors_recall):
    h = bar.get_height()
    if h < 5:
        axes[0].text(bar.get_x() + bar.get_width()/2, h + 2, f'{r:.1f}%', ha='center', va='bottom', fontsize=14, color=C_TEXT, weight='bold')
    else:
        axes[0].text(bar.get_x() + bar.get_width()/2, h - 3, f'{r:.1f}%', ha='center', va='top', fontsize=14, color=tc, weight='bold')

scenarios_precision = ['기존 강수량 기준', 'AI 2단계\n(입수 통제)']
precisions = [baseline_precision, precision_r]
colors_precision = ['#6c757d', C_ORANGE]

bars2 = axes[1].bar(scenarios_precision, precisions, color=colors_precision, alpha=0.85, edgecolor='none', width=0.6)
axes[1].set_ylim(0, 110)
axes[1].set_title('2단계 목표: 정밀도 (오탐 방지력) 비교', color=C_NAVY, weight='bold', fontsize=13)
axes[1].set_ylabel('정밀도 (%)', weight='bold')

text_colors_precision = ['white', C_TEXT]
for bar, p, tc in zip(bars2, precisions, text_colors_precision):
    h = bar.get_height()
    if h < 5:
        axes[1].text(bar.get_x() + bar.get_width()/2, h + 2, f'{p:.1f}%', ha='center', va='bottom', fontsize=14, color=C_TEXT, weight='bold')
    else:
        axes[1].text(bar.get_x() + bar.get_width()/2, h - 3, f'{p:.1f}%', ha='center', va='top', fontsize=14, color=tc, weight='bold')

plt.suptitle(f'{BEACH_KOR} 해수욕장 - AI 도입 효과 (ROI) 시각화', color=C_NAVY, weight='bold', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(RES, f'roi_comparison_{BEACH}.png'), dpi=150, facecolor=fig.get_facecolor())
plt.close()

# Time Series Line Plot
print("  Generating Time Series Line Plot...")
df_plot = pred_df.copy()
# Note: Since pred_df might not have 'date' yet, we pull it from master_dataset
master_df = pd.read_csv(os.path.join(PROC, 'master_dataset.csv'))
master_df['date'] = pd.to_datetime(master_df['date'])
df_plot['date'] = master_df['date'].values

# Reconstruct predictions. 
# y_target in Songdo is log_ecoli, so we invert it with expm1
df_plot['pred_entero'] = np.expm1(df_plot['y_pred'])
df_plot['true_entero'] = master_df['ecoli_max'].values

year_counts = df_plot['date'].dt.year.value_counts()
best_year = year_counts.idxmax()
df_year = df_plot[df_plot['date'].dt.year == best_year]

df_melt = df_year.melt(id_vars=['date'], value_vars=['true_entero', 'pred_entero'], 
                       var_name='Type', value_name='Concentration')
df_melt['Type'] = df_melt['Type'].map({'true_entero': '실제 수치 (Actual)', 'pred_entero': 'AI 예측 (Predicted)'})

fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=df_melt, x='date', y='Concentration', hue='Type', 
             linewidth=2.5, palette=[C_TEXT, C_ORANGE], ax=ax)

ax.set_title(f'{BEACH_KOR} 해수욕장 - 실제 수질 vs AI 예측 트렌드 ({best_year}년)', color=C_NAVY, weight='bold', fontsize=16)
ax.set_xlabel('측정 일자 (Date)', weight='bold', fontsize=12)
ax.set_ylabel('대장균 농도 (E.coli)', weight='bold', fontsize=12)

import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m월 %d일'))
plt.xticks(rotation=45)
ax.xaxis.grid(True, linestyle='--', alpha=0.5, color='#adb5bd')
ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#adb5bd')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper right', frameon=True)

plt.tight_layout()
plt.savefig(os.path.join(RES, f'timeseries_lineplot_{BEACH}.png'), dpi=150, facecolor=fig.get_facecolor())
plt.close()

# Enterprise-level Markdown Report
print("  Generating enterprise markdown report...")
md_report = f"""# 🌊 {BEACH_KOR} 해수욕장 수질 AI 예측 입수 통제 보고서

> [!TIP]
> **Executive Summary**
> 본 보고서는 {BEACH_KOR} 해수욕장의 수질 오염(대장균/장구균 초과)을 예측하기 위한 AI 모델의 최종 성능 및 운영 기준을 요약한 기업용 엔터프라이즈 리포트입니다.

## 📌 1. 최종 모델 성능 (Model Performance)

| 지표 (Metrics) | 결과 (Result) | 비고 (Note) |
| :--- | :--- | :--- |
| **ROC-AUC** | **{final_auc:.3f}** | 5-fold CV, XGBoost Regressor |
| **학습 데이터** | **{len(y_true)}건** | 수질 검사 기록 총량 |
| **오염 발생 빈도** | **{int(y_true.sum())}건** | 대장균 기준치 초과 사례 |

---

## 🎯 2. 이중 기준선 운영 시스템 (Dual-Threshold System)

AI 모델은 오염 피해를 선제적으로 차단하기 위해 2단계의 경보 시스템을 가동합니다.

> [!WARNING]
> ### 🟡 1단계: 주의 알림 (기준 점수: {t_yellow:.3f})
> * **목적**: 관리자에게 선제적 경고 알림 발송
> * **재현율 (Recall)**: **{recall_y:.1f}%** (오염 사태 {int(recall_y/100 * y_true.sum())}/{int(y_true.sum())}건 선제 탐지)
> * **오탐 (False Positive)**: {fp_yellow}건

> [!CAUTION]
> ### 🔴 2단계: 입수 통제 (기준 점수: {t_red:.3f})
> * **목적**: 해수욕장 입수 전면 통제 및 안내 방송
> * **재현율 (Recall)**: **{recall_r:.1f}%** (치명적 오염 사태 {int(recall_r/100 * y_true.sum())}/{int(y_true.sum())}건 탐지)
> * **오탐 (False Positive)**: {fp_red}건 (과잉 통제 최소화)

---

## 📊 3. 시각화 분석 (Data Visualization)

### 3.1 카테고리별 오염 기여도 분석
![카테고리별 오염 기여도](./feature_importance_donut_{BEACH}.png)

### 3.2 핵심 변수 세부 중요도
![세부 중요도](./feature_importance_{BEACH}.png)

### 3.3 정상/오염 예측 점수 분포도 (Log Scale)
![점수 분포도](./dual_warning_kde_{BEACH}.png)

### 3.4 이중 기준선 혼동 행렬 (Confusion Matrix)
![혼동 행렬](./confusion_matrix_{BEACH}.png)

### 3.5 오탐(FP) 방어 비교 분석
![오탐 비교](./roi_comparison_{BEACH}.png)

### 3.6 실제 수질 vs AI 예측 트렌드 (시계열)
![시계열 트렌드](./timeseries_lineplot_{BEACH}.png)

---
*보고서 생성일: 시스템 자동 생성*
"""
with open(os.path.join(RES, f'analysis_report_{BEACH}.md'), 'w', encoding='utf-8') as f:
    f.write(md_report)

print(f"[{BEACH}] All results generated!")
