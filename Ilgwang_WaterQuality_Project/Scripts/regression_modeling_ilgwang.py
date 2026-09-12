import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score, confusion_matrix
from xgboost import XGBClassifier

def model_ilgwang():
    proj_dir = "C:\\Sandbox\\Ilgwang_WaterQuality_Project"
    f = os.path.join(proj_dir, "Data_Processed", "master_dataset_ilgwang.csv")
    df = pd.read_csv(f, encoding='utf-8-sig')
    
    # Target
    df['log_ecoli'] = np.log1p(df['ecoli_max'])
    
    # Features (Meteorology + Spatial + Temporal Proxies + Gijang Sewage)
    features = [
        'precip_1d_lag',
        'precip_3d_sum_lag',
        'precip_5d_sum_lag',
        'distance_from_estuary_km',
        'temp_daily',
        'month',
        'is_weekend',
        'gijang_discharge_1d_lag'
    ]
    
    X = df[features]
    y_reg = df['log_ecoli']
    y_bin = df['any_exceed'].astype(int)
    
    X_train, X_test, y_train_reg, y_test_reg, y_train_bin, y_test_bin = train_test_split(
        X, y_reg, y_bin, test_size=0.2, random_state=42
    )
    
    # Impute
    imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=50, random_state=42), random_state=42, max_iter=10)
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X.columns)
    
    # Switch to XGBClassifier to maximize AUC
    xgb_model = XGBClassifier(
        n_estimators=200, 
        learning_rate=0.01, 
        max_depth=3, 
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(len(y_train_bin) - sum(y_train_bin)) / sum(y_train_bin),
        random_state=42
    )
    xgb_model.fit(X_train_imp, y_train_bin)
    
    y_pred_xgb = xgb_model.predict_proba(X_test_imp)[:, 1]
    
    # Evaluate with standard (0.5) and conservative (0.15) thresholds
    y_pred_50 = (y_pred_xgb >= 0.5).astype(int)
    y_pred_15 = (y_pred_xgb >= 0.15).astype(int)
    
    res_dir = os.path.join(proj_dir, "Results")
    os.makedirs(res_dir, exist_ok=True)
    
    try:
        xgb_auc = roc_auc_score(y_test_bin, y_pred_xgb)
        recall_50 = recall_score(y_test_bin, y_pred_50)
        recall_15 = recall_score(y_test_bin, y_pred_15)
        
        with open(os.path.join(res_dir, "classification_results.txt"), "w") as f:
            f.write("Ilgwang Public Health Classification Results (XGBoost)\n")
            f.write("="*50 + "\n")
            f.write(f"ROC-AUC: {xgb_auc:.3f}\n\n")
            
            f.write("--- Standard Threshold (0.5) ---\n")
            f.write(f"Recall (재현율): {recall_50:.3f}\n")
            f.write(f"Confusion Matrix:\n{confusion_matrix(y_test_bin, y_pred_50)}\n\n")
            
            f.write("--- Public Health Threshold (0.15) ---\n")
            f.write(f"Recall (재현율): {recall_15:.3f}\n")
            f.write(f"Confusion Matrix:\n{confusion_matrix(y_test_bin, y_pred_15)}\n")
            
        print(f"Modeling complete. AUC: {xgb_auc:.3f}")
        print(f"Recall @ 0.5: {recall_50:.3f}")
        print(f"Recall @ 0.15: {recall_15:.3f} (Public Health Optimized)")
    except:
        xgb_auc = np.nan
        
    # Feature Importance
    importance = xgb_model.feature_importances_
    df_imp = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values('Importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(df_imp['Feature'], df_imp['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.title('XGBoost Feature Importance - Ilgwang Water Quality (Temporal/Temp Proxies)')
    
    res_dir = os.path.join(proj_dir, "Results")
    os.makedirs(res_dir, exist_ok=True)
    plt.savefig(os.path.join(res_dir, "feature_importance_ilgwang.png"), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    model_ilgwang()
