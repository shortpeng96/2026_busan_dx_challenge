"""
02_model.py - Haeundae Beach Water Quality Prediction Model (Advanced)
Trains XGBoost model on master_dataset.csv and applies Dual-Threshold Logic
"""
import pandas as pd
import numpy as np
import os, pickle, warnings
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor
from sklearn.metrics import roc_auc_score, fbeta_score, recall_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, 'Data_Processed')
RES  = os.path.join(BASE, 'Results')
os.makedirs(RES, exist_ok=True)

BEACH = 'Haeundae'
print(f"[{BEACH}] 02_model.py starting...")

# 1. Load dataset
csv_path = os.path.join(PROC, 'master_dataset.csv')
df = pd.read_csv(csv_path)

if 'any_exceed' not in df.columns:
    if 'ecoli' in df.columns:
        df['any_exceed'] = (df['ecoli'] >= 500).astype(int)
    else:
        df['any_exceed'] = (df['ecoli_max'] >= 500).astype(int)

y_bin = df['any_exceed'].astype(int)

if 'log_ecoli' not in df.columns:
    if 'ecoli' in df.columns:
        df['log_ecoli'] = np.log1p(df['ecoli'])
    elif 'ecoli_max' in df.columns:
        df['log_ecoli'] = np.log1p(df['ecoli_max'])

if 'log_entero' not in df.columns:
    if 'enterococcus_max' in df.columns:
        df['enterococcus_max'] = df['enterococcus_max'].fillna(df['enterococcus_max'].median())
        df['log_entero'] = np.log1p(df['enterococcus_max'])
    elif 'enterococcus' in df.columns:
        df['enterococcus'] = df['enterococcus'].fillna(df['enterococcus'].median())
        df['log_entero'] = np.log1p(df['enterococcus'])

y_target = df[['log_ecoli', 'log_entero']] if 'log_entero' in df.columns else df[['log_ecoli']]

EXCLUDE = ['ecoli_max', 'ecoli', 'enterococcus_max', 'enterococcus', 'ecoli_exceed',
           'enterococcus_exceed', 'any_exceed', 'log_ecoli', 'log_entero', 'date',
           'examinLcDetail']
feat_cols = [c for c in df.columns
             if c not in EXCLUDE
             and df[c].dtype != 'object'
             and not pd.api.types.is_datetime64_any_dtype(df[c])]

X = df[feat_cols]

# 2. Impute missing values
print(f"  Imputing {X.isna().sum().sum()} missing values...")
imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=10, random_state=42),
    random_state=42, max_iter=5
)
imp_data = imputer.fit_transform(X)
valid_cols = [c for c in feat_cols if X[c].notna().any()] # Manually find valid columns
# IterativeImputer drops columns that are all NaN. It returns columns in the order of valid_cols.
if imp_data.shape[1] < len(feat_cols):
    X_imp = pd.DataFrame(imp_data, columns=valid_cols)
    feat_cols = valid_cols
else:
    X_imp = pd.DataFrame(imp_data, columns=feat_cols)

# Apply Top 15 Best Features for optimal performance
BEST_FEATS = ['suyeong_vol', 'wind_max_1d_lag', 'distance_from_estuary_km']
best_feats = [f for f in BEST_FEATS if f in X_imp.columns]
if len(best_feats) > 0:
    X_imp = X_imp[best_feats]
    feat_cols = best_feats
best_feats = list(X_imp.columns)

# 3. Modeling
print("  Training with 5-fold cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_all = np.zeros(len(y_bin))
preds_raw_all = np.zeros((len(y_bin), 2)) if 'log_entero' in df.columns else np.zeros((len(y_bin), 1))

model_params = dict(
    learning_rate=0.05, max_depth=4, n_estimators=200,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)

for train_idx, test_idx in cv.split(X_imp, y_bin):
    m = XGBRegressor(**model_params)
    m.fit(X_imp.iloc[train_idx], y_target.iloc[train_idx])
    preds = m.predict(X_imp.iloc[test_idx])
    
    if len(preds.shape) == 1:
        preds = preds.reshape(-1, 1)
        
    preds_raw_all[test_idx] = preds
    
    # Calculate Risk index from predictions
    pred_ecoli = np.expm1(preds[:, 0])
    if preds.shape[1] > 1:
        pred_entero = np.expm1(preds[:, 1])
        pred_risk = np.maximum(pred_ecoli / 500.0, pred_entero / 100.0)
    else:
        pred_risk = pred_ecoli / 500.0
        
    y_pred_all[test_idx] = pred_risk

# Check for single class
if len(np.unique(y_bin)) > 1:
    final_auc = roc_auc_score(y_bin, y_pred_all)
else:
    final_auc = 0.5
print(f"\n  [OK] Final AUC: {final_auc:.5f}")

# 4. Compute Dual Thresholds
baseline_fp = ((df.get('precipitation_mm', df.get('precip_daily', pd.Series([0]*len(df)))) >= 3.0) & (y_bin == 0)).sum()
if baseline_fp == 0: baseline_fp = 10

best_f2, t_yellow = 0, float(y_pred_all.min())
best_f05, t_red = 0, float(y_pred_all.max())

if len(np.unique(y_bin)) > 1:
    for t in np.linspace(y_pred_all.min(), y_pred_all.max(), 1000):
        f2 = fbeta_score(y_bin, (y_pred_all >= t).astype(int), beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2, t_yellow = f2, t

    for t in np.linspace(y_pred_all.min(), y_pred_all.max(), 1000):
        preds = (y_pred_all >= t).astype(int)
        f05 = fbeta_score(y_bin, preds, beta=0.5, zero_division=0)
        fp = ((preds == 1) & (y_bin == 0)).sum()
        if f05 >= best_f05 and fp <= max(1, baseline_fp):
            best_f05, t_red = f05, t

print(f"  [Warning] Warning threshold: {t_yellow:.4f}")
print(f"  [Danger] Danger threshold:  {t_red:.4f}")

# 5. Save
final_model = XGBRegressor(**model_params)
final_model.fit(X_imp, y_target)
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

true_entero = df['enterococcus_max'].values if 'enterococcus_max' in df.columns else df['enterococcus'].values if 'enterococcus' in df.columns else np.zeros(len(y_bin))
pred_entero = np.expm1(preds_raw_all[:, 1]) if preds_raw_all.shape[1] > 1 else np.zeros(len(y_bin))

pred_df = pd.DataFrame({
    'date': df['date'].values if 'date' in df.columns else df['examinDe'].values,
    'y_true': y_bin, 
    'y_pred': y_pred_all,
    'pred_entero': pred_entero,
    'true_entero': true_entero
})
pred_df.to_csv(os.path.join(RES, 'predictions.csv'), index=False)
print(f"[{BEACH}] Modeling complete!")
