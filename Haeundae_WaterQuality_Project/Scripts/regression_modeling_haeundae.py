import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from xgboost import XGBRegressor

def model_haeundae():
    proj_dir = "C:\\Sandbox\\Haeundae_WaterQuality_Project"
    f = os.path.join(proj_dir, "Data_Processed", "master_dataset_haeundae.csv")
    df = pd.read_csv(f, encoding='utf-8-sig')
    
    # Target
    df['log_ecoli'] = np.log1p(df['ecoli_max'])
    
    # Features (Ultra-Minimalist RFE Optimized)
    # Kept only the 3 core features identified by RFE (ROC-AUC 0.856)
    features = [
        'precip_3d_sum_lag',
        'precip_5d_sum_lag',
        'suyeong_vol_1d_lag'
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
    
    # XGBoost
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    xgb_model.fit(X_train_imp, y_train_reg)
    
    y_pred_xgb = xgb_model.predict(X_test_imp)
    
    try:
        xgb_auc = roc_auc_score(y_test_bin, y_pred_xgb)
    except:
        xgb_auc = np.nan
        
    print(f"Modeling complete. AUC: {xgb_auc:.3f}")
    
    # Feature Importance
    importance = xgb_model.feature_importances_
    df_imp = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values('Importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(df_imp['Feature'], df_imp['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.title('XGBoost Feature Importance - Haeundae Water Quality')
    
    res_dir = os.path.join(proj_dir, "Results")
    os.makedirs(res_dir, exist_ok=True)
    plt.savefig(os.path.join(res_dir, "feature_importance_haeundae.png"), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    model_haeundae()
