"""
02_model.py - Imrang Beach Water Quality Prediction Model
Trains XGBoost model on master_dataset.csv and prints AUC
Architecture: Dual (XGBClassifier for AUC/warnings + XGBRegressor for time series UI)
Feature Selection: RFE (Recursive Feature Elimination) auto-optimized
"""
import pandas as pd
import numpy as np
import os, pickle, warnings
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score, fbeta_score, recall_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTE
import optuna
warnings.filterwarnings('ignore')

# ── Path configuration ──────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, 'Data_Processed')
RES  = os.path.join(BASE, 'Results')
os.makedirs(RES, exist_ok=True)

BEACH = 'Imrang'
print(f"[{BEACH}] 02_model.py starting...")

# 1. Load master dataset
csv_path = os.path.join(PROC, 'master_dataset.csv')
df = pd.read_csv(csv_path)
print(f"  Dataset: {df.shape}, Exceed rate: {df['any_exceed'].mean():.1%}")

y_bin = df['any_exceed'].astype(int)

if 'log_ecoli' not in df.columns:
    if 'ecoli_max' in df.columns:
        df['log_ecoli'] = np.log1p(df['ecoli_max'])
    elif 'ecoli' in df.columns:
        df['log_ecoli'] = np.log1p(df['ecoli'])

y_target = df['log_ecoli'] if 'log_ecoli' in df.columns else y_bin.astype(float)

# Feature columns (exclude targets and metadata)
EXCLUDE = ['ecoli_max', 'ecoli', 'enterococcus_max', 'ecoli_exceed',
           'enterococcus_exceed', 'any_exceed', 'log_ecoli', 'date',
           'examinLcDetail']
feat_cols = [c for c in df.columns
             if c not in EXCLUDE
             and df[c].dtype != 'object'
             and not pd.api.types.is_datetime64_any_dtype(df[c])]

X = df[feat_cols]

# 2. Split raw data first; fit a separate imputer on each training fold.
def make_imputer():
    return IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=10, random_state=42),
        random_state=42, max_iter=5, keep_empty_features=True
    )


def prepare_cv_folds(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = []
    for train_idx, test_idx in cv.split(X, y):
        fold_imputer = make_imputer()
        X_train = pd.DataFrame(
            fold_imputer.fit_transform(X.iloc[train_idx].to_numpy(copy=True)),
            columns=X.columns, index=X.index[train_idx]
        )
        X_test = pd.DataFrame(
            fold_imputer.transform(X.iloc[test_idx].to_numpy(copy=True)),
            columns=X.columns, index=X.index[test_idx]
        )
        folds.append((train_idx, test_idx, X_train, X_test))
    return folds


print(f"  Imputing {X.isna().sum().sum()} missing values within training folds...")
# These stages use identical splits and imputation settings, so reuse the
# fold-specific transforms without refitting during every RFE step/trial.
cv_folds = prepare_cv_folds(X, y_bin)

# Preserve the existing full-data RFE ranking procedure for this comparison.
# This matrix is ONLY for ranking, never for CV training or predictions.
# Feature-selection/tuning bias still requires a separate nested-CV change.
ranking_imputer = make_imputer()
X_rank = pd.DataFrame(ranking_imputer.fit_transform(X), columns=feat_cols)

# 3. Dynamic Feature Selection (RFE based on AUC)
scale_pos = (len(y_bin) - y_bin.sum()) / max(1, y_bin.sum())
print(f"  Class imbalance: scale_pos_weight={scale_pos:.1f}")
print("  Running Recursive Feature Elimination (RFE) to find best features...")
current_features = list(X.columns)
best_auc_rfe = 0
best_feats = list(current_features)

while len(current_features) > 3:
    y_pred_proba_rfe = np.zeros(len(y_bin))
    xgb_rfe = XGBClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=3,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos, random_state=42, n_jobs=-1
    )
    smote_rfe = SMOTE(random_state=42, k_neighbors=min(4, y_bin.sum()-1))

    for train_idx, test_idx, X_train, X_test in cv_folds:
        X_tr, y_tr = X_train[current_features], y_bin.iloc[train_idx]
        if y_tr.sum() >= 2:
            try:
                X_tr_sm, y_tr_sm = smote_rfe.fit_resample(X_tr, y_tr)
            except:
                X_tr_sm, y_tr_sm = X_tr, y_tr
        else:
            X_tr_sm, y_tr_sm = X_tr, y_tr
        xgb_rfe.fit(X_tr_sm, y_tr_sm)
        y_pred_proba_rfe[test_idx] = xgb_rfe.predict_proba(X_test[current_features])[:, 1]

    auc = roc_auc_score(y_bin, y_pred_proba_rfe)
    if auc > best_auc_rfe:
        best_auc_rfe = auc
        best_feats = list(current_features)

    xgb_rfe.fit(X_rank[current_features], y_bin)
    least_imp_idx = np.argmin(xgb_rfe.feature_importances_)
    current_features.pop(least_imp_idx)

print(f"  Using {len(best_feats)} features for modeling (RFE AUC: {best_auc_rfe:.5f})")

# 4. Optuna Hyperparameter Tuning
print("  Tuning hyperparameters with Optuna (100 trials)...")
optuna.logging.set_verbosity(optuna.logging.WARNING)

pos_weight = (len(y_bin) - y_bin.sum()) / max(1, y_bin.sum())
smote = SMOTE(random_state=42, k_neighbors=min(4, int(y_bin.sum())-1))

def objective(trial):
    xgb_params = dict(
        learning_rate=trial.suggest_float('xgb_lr', 0.01, 0.3, log=True),
        max_depth=trial.suggest_int('xgb_depth', 2, 8),
        n_estimators=trial.suggest_int('xgb_n', 50, 800),
        subsample=trial.suggest_float('xgb_sub', 0.5, 1.0),
        colsample_bytree=trial.suggest_float('xgb_col', 0.5, 1.0),
        reg_alpha=trial.suggest_float('xgb_alpha', 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float('xgb_lambda', 1e-3, 10.0, log=True),
        scale_pos_weight=pos_weight, random_state=42, n_jobs=-1
    )
    rf_params = dict(
        n_estimators=trial.suggest_int('rf_n', 50, 500),
        max_depth=trial.suggest_int('rf_depth', 3, 12),
        min_samples_leaf=trial.suggest_int('rf_leaf', 1, 10),
        class_weight='balanced', random_state=42
    )
    xgb_w = trial.suggest_float('xgb_weight', 0.3, 0.8)

    y_pred = np.zeros(len(y_bin))

    for tr_idx, te_idx, X_train, X_test in cv_folds:
        X_tr, y_tr = X_train[best_feats], y_bin.iloc[tr_idx]
        if y_tr.sum() >= 2:
            try: X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)
            except: X_tr_sm, y_tr_sm = X_tr, y_tr
        else: X_tr_sm, y_tr_sm = X_tr, y_tr

        c_xgb = XGBClassifier(**xgb_params)
        c_xgb.fit(X_tr_sm, y_tr_sm)
        c_rf = RandomForestClassifier(**rf_params)
        c_rf.fit(X_tr_sm, y_tr_sm)

        p_xgb = c_xgb.predict_proba(X_test[best_feats])[:, 1]
        p_rf = c_rf.predict_proba(X_test[best_feats])[:, 1]
        y_pred[te_idx] = xgb_w * p_xgb + (1 - xgb_w) * p_rf

    return roc_auc_score(y_bin, y_pred)

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=100, show_progress_bar=False)

best = study.best_params
print(f"  Best Optuna AUC: {study.best_value:.5f}")

# 5. Final CV with best params
print("  Training with 5-fold cross-validation (Optimized Dual Ensemble)...")
y_pred_proba = np.zeros(len(y_bin))
preds_raw_all = np.zeros(len(y_bin))

clf_xgb_params = dict(
    learning_rate=best['xgb_lr'], max_depth=best['xgb_depth'],
    n_estimators=best['xgb_n'], subsample=best['xgb_sub'],
    colsample_bytree=best['xgb_col'], reg_alpha=best['xgb_alpha'],
    reg_lambda=best['xgb_lambda'], scale_pos_weight=pos_weight, random_state=42
)
clf_rf_params = dict(
    n_estimators=best['rf_n'], max_depth=best['rf_depth'],
    min_samples_leaf=best['rf_leaf'], class_weight='balanced', random_state=42
)
xgb_w = best['xgb_weight']

reg_xgb_params = dict(
    learning_rate=0.05, max_depth=3, n_estimators=100,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
reg_rf_params = dict(n_estimators=100, max_depth=4, random_state=42)

for train_idx, test_idx, X_train, X_test in cv_folds:
    X_tr = X_train[best_feats]
    y_tr = y_bin.iloc[train_idx]
    y_tgt_tr = y_target.iloc[train_idx]

    if y_tr.sum() >= 2:
        try:
            X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)
            X_tr_reg, y_tr_reg = X_tr, y_tgt_tr
        except:
            X_tr_sm, y_tr_sm = X_tr, y_tr
            X_tr_reg, y_tr_reg = X_tr, y_tgt_tr
    else:
        X_tr_sm, y_tr_sm = X_tr, y_tr
        X_tr_reg, y_tr_reg = X_tr, y_tgt_tr

    c_xgb = XGBClassifier(**clf_xgb_params)
    c_xgb.fit(X_tr_sm, y_tr_sm)
    c_rf = RandomForestClassifier(**clf_rf_params)
    c_rf.fit(X_tr_sm, y_tr_sm)

    prob_xgb = c_xgb.predict_proba(X_test[best_feats])[:, 1]
    prob_rf = c_rf.predict_proba(X_test[best_feats])[:, 1]
    y_pred_proba[test_idx] = xgb_w * prob_xgb + (1 - xgb_w) * prob_rf


    # Regressor ensemble (for smooth UI trend lines) - no SMOTE
    m_xgb = XGBRegressor(**reg_xgb_params)
    m_rf = RandomForestRegressor(**reg_rf_params)
    m_xgb.fit(X_tr_reg, y_tr_reg)
    m_rf.fit(X_tr_reg, y_tr_reg)
    preds_raw_all[test_idx] = (m_xgb.predict(X_test[best_feats]) +
                                m_rf.predict(X_test[best_feats])) / 2.0

final_auc = roc_auc_score(y_bin, y_pred_proba)
print(f"\n  [OK] Final Ensemble AUC (Classifier): {final_auc:.5f}")

# 5. Compute thresholds based on Classifier Probabilities
baseline_fp = 10

best_f2 = 0
t_yellow = 0.01
for t in np.arange(0.01, 1.0, 0.01):
    preds = (y_pred_proba >= t).astype(int)
    f2 = fbeta_score(y_bin, preds, beta=2, zero_division=0)
    if f2 > best_f2:
        best_f2 = f2
        t_yellow = t

preds_yellow = (y_pred_proba >= t_yellow).astype(int)
recall_y = recall_score(y_bin, preds_yellow, zero_division=0) * 100
fp_yellow = ((preds_yellow == 1) & (y_bin == 0)).sum()
print(f"  [Warning] Warning threshold: {t_yellow:.4f} (recall={recall_y:.1f}%, FP={fp_yellow})")

t_red = t_yellow
for t in np.arange(t_yellow, 1.0, 0.01):
    preds = (y_pred_proba >= t).astype(int)
    fp = ((preds == 1) & (y_bin == 0)).sum()
    if fp <= baseline_fp:
        t_red = t
        break

preds_red = (y_pred_proba >= t_red).astype(int)
recall_r = recall_score(y_bin, preds_red, zero_division=0) * 100
fp_red = ((preds_red == 1) & (y_bin == 0)).sum()
print(f"  [Danger] Danger threshold:  {t_red:.4f}  (recall={recall_r:.1f}%, FP={fp_red})")

# 6. Save model and predictions
# 6. Save models and predictions
# Fit on all data only for the final saved model, after CV scoring is finished.
imputer = make_imputer()
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=feat_cols)
final_xgb = XGBClassifier(**clf_xgb_params)
final_xgb.fit(X_imp[best_feats], y_bin)

final_rf = RandomForestClassifier(**clf_rf_params)
final_rf.fit(X_imp[best_feats], y_bin)

model_path = os.path.join(RES, 'model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump({
        'model_xgb': final_xgb,
        'model_rf': final_rf,
        'imputer': imputer,
        'features': best_feats,
        'auc': final_auc,
        't_yellow': t_yellow,
        't_red': t_red,
        'xgb_w': xgb_w,
    }, f)

true_ecoli = df['ecoli_max'].values if 'ecoli_max' in df.columns else (
    df['ecoli'].values if 'ecoli' in df.columns else np.zeros(len(y_bin)))
pred_ecoli = np.expm1(preds_raw_all)

pred_df = pd.DataFrame({
    'date': df['date'].values if 'date' in df.columns else np.arange(len(y_bin)),
    'y_true': y_bin,
    'y_pred': y_pred_proba,
    'pred_entero': pred_ecoli,
    'pred_ecoli': pred_ecoli,
    'true_entero': true_ecoli,
    'true_ecoli': true_ecoli
})
pred_df.to_csv(os.path.join(RES, 'predictions.csv'), index=False)

print(f"  Model saved: {model_path}")
print(f"[{BEACH}] Modeling complete! AUC={final_auc:.5f}")
