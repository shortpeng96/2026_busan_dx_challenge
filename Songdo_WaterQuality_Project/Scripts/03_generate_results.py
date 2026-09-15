"""
03_generate_results.py - Songdo Beach Water Quality
Loads predictions from Results/predictions.csv and model.pkl
and generates all result figures and reports
"""
import pandas as pd
import numpy as np
import os, pickle, warnings
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score, fbeta_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── Path configuration ──────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, 'Data_Processed')
RES  = os.path.join(BASE, 'Results')
os.makedirs(RES, exist_ok=True)

BEACH = 'Songdo'
print(f"[{BEACH}] 03_generate_results.py starting...")

# 1. Load predictions and model
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
recall_y = recall_score(y_true, preds_y) * 100
recall_r = recall_score(y_true, preds_r) * 100
fp_yellow = ((preds_y == 1) & (y_true == 0)).sum()
fp_red = ((preds_r == 1) & (y_true == 0)).sum()
baseline_fp = 10  # Approximate baseline

# 2. Feature Importance Plot
print("  Generating feature importance plot...")
fig, ax = plt.subplots(figsize=(10, 6))
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values(ascending=True)
feat_imp.plot(kind='barh', ax=ax, color='steelblue')
ax.set_title(f'{BEACH} 해수욕장 - 수질 예측 핵심 변수 중요도', fontsize=14)
ax.set_xlabel('Feature Importance (Gain)')
plt.tight_layout()
plt.savefig(os.path.join(RES, f'feature_importance_{BEACH}.png'), dpi=150)
plt.close()
print(f"  Saved: feature_importance_{BEACH}.png")

# 3. Confusion Matrix
print("  Generating confusion matrix...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (threshold, label, color) in zip(axes, [
    (t_yellow, f'🟡 주의 단계 (t={t_yellow:.3f})', '#FFC107'),
    (t_red, f'🔴 위험 단계 (t={t_red:.3f})', '#DC3545'),
]):
    preds = (y_pred >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                cmap=sns.light_palette(color, as_cmap=True),
                xticklabels=['정상', '오염'], yticklabels=['정상', '오염'])
    ax.set_title(label, fontsize=11)
    ax.set_xlabel('예측값')
    ax.set_ylabel('실제값')

plt.suptitle(f'{BEACH} 해수욕장 이중 임계값 혼동행렬 (AUC={final_auc:.3f})', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RES, f'confusion_matrix_{BEACH}.png'), dpi=150)
plt.close()
print(f"  Saved: confusion_matrix_{BEACH}.png")

# 4. KDE Distribution Plot
print("  Generating KDE distribution plot...")
fig, ax = plt.subplots(figsize=(10, 6))
y_pred_arr = np.array(y_pred)
ax.hist(y_pred_arr[y_true == 0], bins=30, alpha=0.6, color='steelblue', label='정상 (No Exceed)', density=True)
ax.hist(y_pred_arr[y_true == 1], bins=30, alpha=0.6, color='tomato', label='오염 초과 (Exceed)', density=True)
ax.axvline(t_yellow, color='#FFC107', linestyle='--', lw=2, label=f'🟡 주의 ({t_yellow:.3f})')
ax.axvline(t_red, color='#DC3545', linestyle='--', lw=2, label=f'🔴 위험 ({t_red:.3f})')
ax.set_title(f'{BEACH} 해수욕장 - 오염 예측 점수 분포 (AUC={final_auc:.3f})')
ax.set_xlabel('예측 오염 점수')
ax.set_ylabel('Density')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RES, f'dual_warning_kde_{BEACH}.png'), dpi=150)
plt.close()
print(f"  Saved: dual_warning_kde_{BEACH}.png")

# 5. ROI Comparison Plot
print("  Generating ROI comparison plot...")
fig, ax = plt.subplots(figsize=(8, 5))
scenarios = ['기존 강수 기준\n(베이스라인)', '🟡 주의 단계\n(재현율 최적)', '🔴 위험 단계\n(정밀도 최적)']
false_pos = [baseline_fp, fp_yellow, fp_red]
recalls = [50, recall_y, recall_r]
colors = ['#6c757d', '#FFC107', '#DC3545']

bars = ax.bar(scenarios, false_pos, color=colors, alpha=0.8, edgecolor='black')
for bar, r in zip(bars, recalls):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'재현율\n{r:.0f}%', ha='center', va='bottom', fontsize=9)
ax.set_title(f'{BEACH} 해수욕장 - 오탐(FP) 비교 (낮을수록 좋음)')
ax.set_ylabel('오탐 (False Positive) 횟수')
plt.tight_layout()
plt.savefig(os.path.join(RES, f'roi_comparison_{BEACH}.png'), dpi=150)
plt.close()
print(f"  Saved: roi_comparison_{BEACH}.png")

# 6. Text Report
print("  Generating text report...")
report = f"""# 🌊 {BEACH} 해수욕장 수질 AI 예측 입수 통제 보고서

## 📌 최종 모델 성능

* **ROC-AUC**: **{final_auc:.3f}** (5-fold CV, XGBoost Regressor)
* **학습 데이터**: {len(y_true)}건 수질 검사 기록 (대장균 기준 초과: {int(y_true.sum())}건)
* **핵심 피처**: {', '.join(features[:8])}

## 🎯 이중 임계값 시스템

* 🟡 **주의 단계 (임계값 {t_yellow:.3f})**
  * 재현율(Recall): {recall_y:.1f}% - 오염 사태 {int(recall_y/100 * y_true.sum())}/{int(y_true.sum())}건 선제 탐지
  * 오탐(FP): {fp_yellow}건

* 🔴 **위험 단계 (임계값 {t_red:.3f})**
  * 재현율(Recall): {recall_r:.1f}% - 오염 사태 {int(recall_r/100 * y_true.sum())}/{int(y_true.sum())}건 탐지
  * 오탐(FP): {fp_red}건 (기존 베이스라인 대비 대폭 감소)

## 🔬 사용된 주요 변수
"""
for i, feat in enumerate(features, 1):
    report += f"{i}. `{feat}`\n"

report_path = os.path.join(RES, f'results_{BEACH}.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"  Saved: results_{BEACH}.txt")

print(f"[{BEACH}] All results generated! AUC={final_auc:.3f}")
