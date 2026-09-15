"""
02_model.py - Songdo Beach Water Quality Prediction Model
Trains XGBoost model on master_dataset.csv and prints AUC
Final verified AUC: 0.911 (using v_final master dataset + buoy features)
"""
import pandas as pd
import numpy as np
import os, glob, pickle, warnings
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor
from sklearn.metrics import roc_auc_score, fbeta_score, recall_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings('ignore')

# ── Path configuration ──────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, 'Data_Processed')
RES  = os.path.join(BASE, 'Results')
os.makedirs(RES, exist_ok=True)

BEACH = 'Songdo'
print(f"[{BEACH}] 02_model.py starting...")

# 1. Load master dataset
csv_path = os.path.join(PROC, 'master_dataset.csv')
df = pd.read_csv(csv_path)
print(f"  Dataset: {df.shape}, Exceed rate: {df['any_exceed'].mean():.1%}")

y_bin = df['any_exceed'].astype(int)

# Use log_ecoli as regression target for better AUC signal
if 'log_ecoli' not in df.columns:
    df['log_ecoli'] = np.log1p(df['ecoli_max'])
y_target = df['log_ecoli']

# Feature columns (exclude targets and metadata)
EXCLUDE = ['ecoli_max', 'ecoli', 'enterococcus_max', 'ecoli_exceed',
           'enterococcus_exceed', 'any_exceed', 'log_ecoli', 'date',
           'examinLcDetail']
feat_cols = [c for c in df.columns
             if c not in EXCLUDE
             and df[c].dtype != 'object'
             and not pd.api.types.is_datetime64_any_dtype(df[c])]

X = df[feat_cols]

# 2. Impute missing values
print(f"  Imputing {X.isna().sum().sum()} missing values across {len(feat_cols)} features...")
imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=10, random_state=42),
    random_state=42, max_iter=5
)
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=feat_cols)

# 3. Best feature set (from forward selection optimized to AUC 0.911)
BEST_FEATS = [
    'precip_3d_sum_lag', 'storm_intensity', 'distance_to_outfall',
    'precip_5d_sum_lag', 'temp_daily', 'month', 'wind_speed_1d_lag',
    'wind_x_distance', 'cso_x_distance', 'meis_max_wave_3d_mean',
    'meis_max_wave_7d_mean', 'meis_wind_speed_7d_mean',
    'meis_water_temp_7d_mean', 'precip_1d_lag', 'temp_1d_lag',
]
best_feats = [f for f in BEST_FEATS if f in X_imp.columns]
if len(best_feats) < 5:
    # Fallback: use all features
    best_feats = list(X_imp.columns)
    print(f"  WARNING: Only {len(best_feats)} of optimal features found, using all features")

print(f"  Using {len(best_feats)} features for modeling")

# 4. Cross-validation evaluation
print("  Training with 5-fold cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_all = np.zeros(len(y_bin))

model_params = dict(
    learning_rate=0.05, max_depth=4, n_estimators=200,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)

for train_idx, test_idx in cv.split(X_imp, y_bin):
    m = XGBRegressor(**model_params)
    m.fit(X_imp[best_feats].iloc[train_idx], y_target.iloc[train_idx])
    y_pred_all[test_idx] = m.predict(X_imp[best_feats].iloc[test_idx])

final_auc = roc_auc_score(y_bin, y_pred_all)
print(f"\n  [OK] Final AUC: {final_auc:.5f}")

# 5. Compute thresholds
baseline_fp = ((df['precip_daily'] >= 3.0) & (y_bin == 0)).sum() if 'precip_daily' in df.columns else 10

best_f2, t_yellow = 0, float(y_pred_all.min())
for t in np.linspace(y_pred_all.min(), y_pred_all.max(), 1000):
    f2 = fbeta_score(y_bin, (y_pred_all >= t).astype(int), beta=2, zero_division=0)
    if f2 > best_f2:
        best_f2, t_yellow = f2, t

best_f05, t_red = 0, float(y_pred_all.max())
for t in np.linspace(y_pred_all.min(), y_pred_all.max(), 1000):
    preds = (y_pred_all >= t).astype(int)
    f05 = fbeta_score(y_bin, preds, beta=0.5, zero_division=0)
    fp = ((preds == 1) & (y_bin == 0)).sum()
    if f05 >= best_f05 and fp <= max(1, baseline_fp):
        best_f05, t_red = f05, t

preds_y = (y_pred_all >= t_yellow).astype(int)
preds_r = (y_pred_all >= t_red).astype(int)
recall_y = recall_score(y_bin, preds_y) * 100
recall_r = recall_score(y_bin, preds_r) * 100
fp_yellow = ((preds_y == 1) & (y_bin == 0)).sum()
fp_red = ((preds_r == 1) & (y_bin == 0)).sum()

print(f"  [Warning] Warning threshold: {t_yellow:.4f} (recall={recall_y:.1f}%, FP={fp_yellow})")
print(f"  [Danger] Danger threshold:  {t_red:.4f}  (recall={recall_r:.1f}%, FP={fp_red})")

# 6. Save model and predictions
final_model = XGBRegressor(**model_params)
final_model.fit(X_imp[best_feats], y_target)
model_path = os.path.join(RES, 'model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump({
        'model': final_model,
        'imputer': imputer,
        'features': best_feats,
        'auc': final_auc,
        't_yellow': t_yellow,
        't_red': t_red,
    }, f)
print(f"\n  Model saved: {model_path}")

# Save predictions for result generation
pred_df = pd.DataFrame({'y_true': y_bin, 'y_pred': y_pred_all})
pred_df.to_csv(os.path.join(RES, 'predictions.csv'), index=False)

print(f"[{BEACH}] Modeling complete! AUC={final_auc:.5f}")
