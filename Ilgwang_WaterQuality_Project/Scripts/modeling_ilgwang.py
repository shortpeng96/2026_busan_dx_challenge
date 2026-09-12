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

# Stepwise Selection Optimal Features
features = [
    'wind_sin', 
    'avg_water_temp', 
    'wind_cos_1d_lag', 
    'gijang_discharge_m3_day'
]

y_target = df_master['log_ecoli']
y_bin = df_master['any_exceed'].astype(int)
X = df_master[features]

print("2. Imputing Missing Values...")
imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=10, random_state=42), random_state=42, max_iter=5)
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

print("3. Training Final Model with Optimal Features...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs = []

for train_idx, test_idx in cv.split(X_imp, y_bin):
    X_train, X_test = X_imp.iloc[train_idx], X_imp.iloc[test_idx]
    y_train, y_test = y_bin.iloc[train_idx], y_bin.iloc[test_idx]
    
    if len(y_train.unique()) > 1:
        # Scale pos_weight
        pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
        xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, scale_pos_weight=pos_weight, random_state=42, eval_metric='auc')
        xgb.fit(X_train, y_train)
        preds = xgb.predict_proba(X_test)[:, 1]
        try: aucs.append(roc_auc_score(y_test, preds))
        except: pass

final_auc = np.mean(aucs) if aucs else 0
print(f"Final Model AUC: {final_auc:.5f}")

# Train full model for feature importance & KDE
xgb.fit(X_imp, y_bin)

# --- SAVE REPORT ---
report_path = os.path.join(proj_dir, "Results", f"results_{beach_name}.txt")
with open(report_path, "w", encoding='utf-8') as f:
    f.write(f"=== {beach_name} Water Quality Prediction Report (v2 Optimal) ===\n")
    f.write(f"ROC-AUC: {final_auc:.3f}\n")
    f.write("Features: " + ", ".join(features) + "\n")

print(f"\nDone! Saved standardized report for {beach_name}.")
