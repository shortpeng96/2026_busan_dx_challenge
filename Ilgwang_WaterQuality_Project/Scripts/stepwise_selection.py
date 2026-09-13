import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

proj_dir = "C:\\Sandbox\\2026_busan_dx_challenge\\Ilgwang_WaterQuality_Project"
df_master = pd.read_csv(os.path.join(proj_dir, "Data_Processed", "master_dataset_ilgwang_v2.csv"))

# Define all possible candidate features (excluding raw dates, categories, target)
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
    'dry_days_count', 'antecedent_dry_days', 'nps_first_flush'
]

y_target = df_master['any_exceed'].astype(int)
X_full = df_master[all_candidates]

# Imputation
imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=10, random_state=42), random_state=42, max_iter=5)
X_imputed = pd.DataFrame(imputer.fit_transform(X_full), columns=X_full.columns)

def evaluate_features(features):
    if len(features) == 0: return 0.5
    X_subset = X_imputed[features]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for train_idx, test_idx in cv.split(X_subset, y_target):
        X_train, X_test = X_subset.iloc[train_idx], X_subset.iloc[test_idx]
        y_train, y_test = y_target.iloc[train_idx], y_target.iloc[test_idx]
        
        # Scale pos_weight
        pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
        
        model = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            scale_pos_weight=pos_weight, random_state=42, eval_metric='auc'
        )
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y_test, preds))
    return np.mean(aucs)

# Stepwise Selection (Forward & Backward)
selected = []
best_auc = 0.5

while True:
    changed = False
    
    # 1. Forward step: Try adding a feature
    best_forward_feature = None
    best_forward_auc = best_auc
    
    for feature in all_candidates:
        if feature not in selected:
            current_test = selected + [feature]
            auc = evaluate_features(current_test)
            if auc > best_forward_auc:
                best_forward_auc = auc
                best_forward_feature = feature
                
    if best_forward_feature is not None:
        selected.append(best_forward_feature)
        best_auc = best_forward_auc
        changed = True
        print(f"[+] Added {best_forward_feature}, New AUC: {best_auc:.5f}")
        
    # 2. Backward step: Try removing a feature
    best_backward_feature = None
    best_backward_auc = best_auc
    
    if len(selected) > 1:
        for feature in selected:
            current_test = [f for f in selected if f != feature]
            auc = evaluate_features(current_test)
            if auc > best_backward_auc:
                best_backward_auc = auc
                best_backward_feature = feature
                
        if best_backward_feature is not None:
            selected.remove(best_backward_feature)
            best_auc = best_backward_auc
            changed = True
            print(f"[-] Removed {best_backward_feature}, New AUC: {best_auc:.5f}")
            
    if not changed:
        break

print("\n=== FINAL RESULT ===")
print(f"Best AUC: {best_auc:.5f}")
print(f"Best Features ({len(selected)}): {selected}")
