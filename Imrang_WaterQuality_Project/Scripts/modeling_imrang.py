import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

base_dir = "C:\\Sandbox\\2026_busan_dx_challenge"
proj_dir = os.path.join(base_dir, "Imrang_WaterQuality_Project")

print("1. Loading Spatial Dataset...")
df_master = pd.read_csv(os.path.join(proj_dir, "Data_Processed", "imrang_spatial_master.csv"))

df_master['wind_x_distance'] = df_master['wind_speed_1d_lag'] * df_master['distance_to_outfall']
df_master['cso_x_distance'] = df_master['정관사업단_vol_3d_sum_lag'] * df_master['distance_to_outfall']

features = [
    'distance_to_outfall', 
    'precip_1d_lag', 'precip_3d_sum_lag', 'precip_5d_sum_lag', 
    'temp_1d_lag', 'temp_daily', 'month', 'is_summer',
    'wind_speed_1d_lag', 'wind_x_distance',
    '정관사업단_vol_1d_lag', '정관사업단_vol_3d_sum_lag', 
    'cso_x_distance'
]

y_target = df_master['log_ecoli']
y_bin = df_master['any_exceed'].astype(int)
X = df_master[features]

print("2. Imputing Missing Values...")
imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=10, random_state=42), random_state=42, max_iter=5)
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

print("3. Running RFE with AUC Metric...")
scale_pos = (len(y_bin) - y_bin.sum()) / y_bin.sum()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
current_features = list(features)
results = []

while len(current_features) > 0:
    aucs = []
    xgb = XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos, random_state=42)
    
    for train_idx, test_idx in cv.split(X_imp[current_features], y_bin):
        X_train, X_test = X_imp[current_features].iloc[train_idx], X_imp[current_features].iloc[test_idx]
        y_train_target, y_test_bin = y_bin.iloc[train_idx], y_bin.iloc[test_idx]
        
        xgb.fit(X_train, y_train_target)
        preds = xgb.predict_proba(X_test)[:, 1]
        try:
            aucs.append(roc_auc_score(y_test_bin, preds))
        except: pass
        
    avg_auc = np.mean(aucs) if aucs else 0
    xgb.fit(X_imp[current_features], y_bin)
    imp = dict(zip(current_features, xgb.feature_importances_))
    results.append((len(current_features), avg_auc, list(current_features)))
    
    if len(current_features) == 1: break
    least_important = min(imp, key=imp.get)
    current_features.remove(least_important)

best = max(results, key=lambda x: x[1])
best_feats = best[2]
best_auc = best[1]
print(f"\n[BEST RFE] AUC {best_auc:.5f} with {best[0]} features: {best_feats}")

print("4. Evaluating Final Model...")
xgb_final = XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos, random_state=42)
y_pred_all = np.zeros(len(y_target))
for train_idx, test_idx in cv.split(X_imp[best_feats], y_bin):
    X_train, X_test = X_imp[best_feats].iloc[train_idx], X_imp[best_feats].iloc[test_idx]
    xgb_final.fit(X_train, y_bin.iloc[train_idx])
    y_pred_all[test_idx] = xgb_final.predict_proba(X_test)[:, 1]

# Dynamic Threshold Simulation
thresholds_to_test = [0.05, 0.1, 0.15, 0.2, 0.5]
threshold_results = []
for t in thresholds_to_test:
    rec = recall_score(y_bin, (y_pred_all >= t).astype(int))
    fp = ((y_pred_all >= t) & (y_bin == 0)).sum()
    threshold_results.append(f"Threshold {t:.2f} -> Recall: {rec:.3f}, False Positives: {fp}/{len(y_bin)-sum(y_bin)}")

xgb_final.fit(X_imp[best_feats], y_bin)
plt.figure(figsize=(10, 6))
sns.barplot(x=xgb_final.feature_importances_, y=best_feats)
plt.title(f'임랑 해수욕장 공간/기상 융합 AI (AUC: {best_auc:.3f})')
plt.tight_layout()
plt.savefig(os.path.join(proj_dir, "Results", "feature_importance_Imrang.png"))

with open(os.path.join(proj_dir, "Results", "results_Imrang.txt"), "w") as f:
    f.write(f"=== Imrang Spatial-Wind AI Results ===\n")
    f.write(f"ROC-AUC: {best_auc:.3f}\n")
    f.write(f"Exceedances in data: {y_bin.sum()} / {len(y_bin)}\n")
    f.write("\n--- Threshold Simulation ---\n")
    for tr in threshold_results:
        f.write(tr + "\n")
    f.write("\nFeatures Used:\n")
    for ft in best_feats: f.write(f"- {ft}\n")

print(f"\nDone! AUC: {best_auc:.3f}")
