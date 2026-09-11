import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def run_modeling():
    proj_dir = "C:\\Sandbox\\Gwangalli_WaterQuality_Project"
    f = os.path.join(proj_dir, "Data_Processed", "master_dataset_gwangalli.csv")
    df = pd.read_csv(f, encoding='utf-8-sig')
    
    # Calculate log target
    df['log_ecoli'] = np.log1p(df['ecoli_max'])
    
    # Features
    features = [
        'precip_1d_lag', 'precip_2d_sum_lag', 'precip_3d_sum_lag',
        'temp_1d_lag', 'wind_max_1d_lag',
        'distance_from_estuary_km',
        'suyeong_vol_1d_lag', 'nambu_vol_1d_lag', 
        'CSO_Flag_East', 'CSO_Flag_West'
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
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_train_imp, y_train_reg)
    
    y_pred = model.predict(X_test_imp)
    
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred))
    r2 = r2_score(y_test_reg, y_pred)
    auc = roc_auc_score(y_test_bin, y_pred)
    
    # Save Results
    out_dir = os.path.join(proj_dir, "Results")
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "regression_results.txt"), "w", encoding='utf-8-sig') as txt_file:
        txt_file.write("=== XGBoost (Gwangalli Spatio-Temporal & CSO) ===\n\n")
        txt_file.write(f"RMSE: {rmse:.3f} | R^2: {r2:.3f} | ROC-AUC: {auc:.3f}\n")
    
    # Feature Importance Plot
    importance = model.feature_importances_
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance, y=X.columns)
    plt.title('Gwangalli XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_importance_gwangalli.png"))
    plt.close()
    
    print(f"Modeling complete. AUC: {auc:.3f}")
    print("Results saved.")

if __name__ == "__main__":
    run_modeling()
