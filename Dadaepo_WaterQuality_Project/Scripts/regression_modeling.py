import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
import os

# Set font for Korean text in plots
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def run_regression_modeling():
    data_path = "C:\\Sandbox\\Preprocessed\\master_dataset_v2.csv"
    out_dir = "C:\\Sandbox\\Preprocessed"
    
    df = pd.read_csv(data_path)
    
    # Calculate Risk Index
    df['ecoli_risk'] = df['ecoli_max'] / 500.0
    df['entero_risk'] = df['enterococcus_max'] / 100.0
    df['risk_index'] = df[['ecoli_risk', 'entero_risk']].max(axis=1)
    
    # Predict log(1+x) to handle scale
    df['log_ecoli'] = np.log1p(df['ecoli_max'])
    df['log_entero'] = np.log1p(df['enterococcus_max'])
    
    # Base Features (Original 90-row model baseline features for fair comparison, minus discharge as we use sewage now)
    features = ['precip_1d_lag', 'precip_2d_sum_lag', 'precip_3d_sum_lag', 'temp_1d_lag', 'wind_max_1d_lag', 'discharge_1d_lag', 'discharge_3d_sum_lag']
    
    # V2 Advanced Features (Spatial + Sewage CSO)
    # We include all base features, plus spatial (distance), sensors, visitors, tide, and sewage.
    adv_features = features + [
        'sensor_turbidity_max_1d_lag', 'sensor_salinity_min_1d_lag', 'sensor_temp_mean_1d_lag', 
        'visitor_count_1d_lag', 
        'distance_from_estuary_km', 'sewage_discharge_1d_lag', 'sewage_discharge_3d_sum_lag', 'CSO_Flag'
    ]
    
    X_base = df[features]
    X_adv = df[adv_features]
    y_reg = df[['log_ecoli', 'log_entero']]
    y_bin = df['any_exceed'].astype(int)
    
    # MICE Imputer
    imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=50, random_state=42), random_state=42, max_iter=10)
    
    # Ridge Pipeline
    ridge_pipe = Pipeline([
        ('imputer', imputer),
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0, random_state=42))
    ])
    
    # XGBoost Pipeline
    # Data is now 305 rows, we can afford slightly deeper trees, but still keep it robust.
    xgb_pipe = Pipeline([
        ('imputer', imputer),
        ('model', XGBRegressor(
            n_estimators=100, 
            learning_rate=0.05, 
            max_depth=4, 
            subsample=0.8, 
            colsample_bytree=0.8, 
            random_state=42
        ))
    ])
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    def evaluate_model(pipe, X, y_reg, y_bin):
        rmses = []
        r2s = []
        aucs = []
        
        preds_all = np.zeros((len(y_reg), 2))
        
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train_reg, y_test_reg = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
            y_test_bin = y_bin.iloc[test_idx]
            
            pipe.fit(X_train, y_train_reg)
            preds = pipe.predict(X_test)
            preds_all[test_idx] = preds
            
            rmses.append(np.sqrt(mean_squared_error(y_test_reg, preds)))
            r2s.append(r2_score(y_test_reg, preds))
            
            # Reconstruct binary prediction for AUC
            pred_ecoli = np.expm1(preds[:, 0])
            pred_entero = np.expm1(preds[:, 1])
            pred_risk = np.maximum(pred_ecoli / 500.0, pred_entero / 100.0)
            
            try:
                aucs.append(roc_auc_score(y_test_bin, pred_risk))
            except ValueError:
                pass
                
        return np.mean(rmses), np.mean(r2s), np.mean(aucs), preds_all

    # Evaluate Ridge
    ridge_base_rmse, ridge_base_r2, ridge_base_auc, ridge_base_preds = evaluate_model(ridge_pipe, X_base, y_reg, y_bin)
    ridge_adv_rmse, ridge_adv_r2, ridge_adv_auc, ridge_adv_preds = evaluate_model(ridge_pipe, X_adv, y_reg, y_bin)
    
    # Evaluate XGBoost
    xgb_base_rmse, xgb_base_r2, xgb_base_auc, xgb_base_preds = evaluate_model(xgb_pipe, X_base, y_reg, y_bin)
    xgb_adv_rmse, xgb_adv_r2, xgb_adv_auc, xgb_adv_preds = evaluate_model(xgb_pipe, X_adv, y_reg, y_bin)
    
    # Save Results
    with open(os.path.join(out_dir, "regression_results.txt"), 'w', encoding='utf-8') as f:
        f.write("=== XGBoost V2 (공간 확장 + 하수처리장 CSO 결합) 결과 ===\n\n")
        f.write("1. Ridge (Baseline):\n")
        f.write(f"   RMSE: {ridge_base_rmse:.3f} | R^2: {ridge_base_r2:.3f} | ROC-AUC: {ridge_base_auc:.3f}\n\n")
        
        f.write("2. Ridge (Advanced V2):\n")
        f.write(f"   RMSE: {ridge_adv_rmse:.3f} | R^2: {ridge_adv_r2:.3f} | ROC-AUC: {ridge_adv_auc:.3f}\n\n")
        
        f.write("3. XGBoost (Baseline):\n")
        f.write(f"   RMSE: {xgb_base_rmse:.3f} | R^2: {xgb_base_r2:.3f} | ROC-AUC: {xgb_base_auc:.3f}\n\n")
        
        f.write("4. XGBoost (Advanced V2):\n")
        f.write(f"   RMSE: {xgb_adv_rmse:.3f} | R^2: {xgb_adv_r2:.3f} | ROC-AUC: {xgb_adv_auc:.3f}\n\n")
        
    # Scatter plot Actual vs Predicted (using Advanced XGBoost V2)
    plt.figure(figsize=(10, 5))
    
    actual_ecoli = np.expm1(y_reg['log_ecoli'])
    actual_entero = np.expm1(y_reg['log_entero'])
    
    pred_ecoli = np.expm1(xgb_adv_preds[:, 0])
    pred_entero = np.expm1(xgb_adv_preds[:, 1])
    
    plt.subplot(1, 2, 1)
    plt.scatter(actual_ecoli, pred_ecoli, alpha=0.6, edgecolors='k')
    plt.plot([0, max(actual_ecoli)], [0, max(actual_ecoli)], 'r--')
    plt.xscale('symlog')
    plt.yscale('symlog')
    plt.xlabel('Actual E.coli')
    plt.ylabel('Predicted E.coli')
    plt.title('XGBoost V2 E.coli Prediction')
    
    plt.subplot(1, 2, 2)
    plt.scatter(actual_entero, pred_entero, alpha=0.6, edgecolors='k')
    plt.plot([0, max(actual_entero)], [0, max(actual_entero)], 'r--')
    plt.xscale('symlog')
    plt.yscale('symlog')
    plt.xlabel('Actual Enterococcus')
    plt.ylabel('Predicted Enterococcus')
    plt.title('XGBoost V2 Enterococcus Prediction')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "regression_scatter.png"))

    # Feature Importance for XGBoost V2
    xgb_pipe.fit(X_adv, y_reg)
    
    avg_importance = xgb_pipe.named_steps['model'].feature_importances_
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=avg_importance, y=X_adv.columns)
    plt.title('XGBoost V2 Feature Importances (Spatial + CSO)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_importance_xgboost.png"))

if __name__ == "__main__":
    run_regression_modeling()
