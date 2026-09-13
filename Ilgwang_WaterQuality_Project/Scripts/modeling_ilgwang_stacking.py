import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

proj_dir = "C:\\Sandbox\\2026_busan_dx_challenge\\Ilgwang_WaterQuality_Project"
df_master = pd.read_csv(os.path.join(proj_dir, "Data_Processed", "master_dataset_ilgwang_v2.csv"))

all_candidates = [
    'distance_from_estuary_km', 'precip_daily', 'temp_daily', 'wind_max',
    'gijang_discharge_m3_day', 'discharge_95th_thresh',
    'precip_1d_lag', 'precip_2d_sum_lag', 'precip_3d_sum_lag', 'precip_5d_sum_lag',
    'temp_1d_lag', 'wind_max_1d_lag',
    'gijang_discharge_1d_lag', 'gijang_thresh_1d_lag',
    'CSO_Flag_Rain', 'Dual_CSO_Flag', 'month', 'is_weekend',
    'avg_water_temp', 'avg_water_temp_1d_lag',
    'tide_range', 'tide_range_1d_lag',
    'wind_sin', 'wind_cos', 'wind_sin_1d_lag', 'wind_cos_1d_lag',
    'dry_days_count'
]

y_bin = df_master['any_exceed'].astype(int)
X_full = df_master[all_candidates]

imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=10, random_state=42), random_state=42, max_iter=5)
X_imp_full = pd.DataFrame(imputer.fit_transform(X_full), columns=X_full.columns)

optimal_features = ['discharge_95th_thresh', 'wind_cos', 'precip_3d_sum_lag', 'tide_range']
X_imp = X_imp_full[optimal_features]

scale_pos = (len(y_bin) - y_bin.sum()) / max(1, y_bin.sum())

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs_xgb, aucs_lgb, aucs_cb, aucs_ensemble = [], [], [], []

for train_idx, test_idx in cv.split(X_imp, y_bin):
    X_train, X_test = X_imp.iloc[train_idx], X_imp.iloc[test_idx]
    y_train, y_test = y_bin.iloc[train_idx], y_bin.iloc[test_idx]
    
    if len(y_train.unique()) > 1:
        # Scale pos_weight inside fold just to be perfectly identical to stepwise
        pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
        
        xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, scale_pos_weight=pos_weight, random_state=42, eval_metric='logloss')
        lgb = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, scale_pos_weight=pos_weight, random_state=42, verbose=-1)
        cb = CatBoostClassifier(iterations=100, learning_rate=0.05, depth=3, scale_pos_weight=pos_weight, verbose=False, random_state=42)
        
        xgb.fit(X_train, y_train)
        lgb.fit(X_train, y_train)
        cb.fit(X_train, y_train)
        
        p_xgb = xgb.predict_proba(X_test)[:, 1]
        p_lgb = lgb.predict_proba(X_test)[:, 1]
        p_cb = cb.predict_proba(X_test)[:, 1]
        
        p_ensemble = (p_xgb + p_lgb + p_cb) / 3.0
        
        try: 
            aucs_xgb.append(roc_auc_score(y_test, p_xgb))
            aucs_lgb.append(roc_auc_score(y_test, p_lgb))
            aucs_cb.append(roc_auc_score(y_test, p_cb))
            aucs_ensemble.append(roc_auc_score(y_test, p_ensemble))
        except: pass

print(f"XGBoost AUC: {np.mean(aucs_xgb):.5f}")
print(f"LightGBM AUC: {np.mean(aucs_lgb):.5f}")
print(f"CatBoost AUC: {np.mean(aucs_cb):.5f}")
print(f"Soft Voting Ensemble AUC: {np.mean(aucs_ensemble):.5f}")
