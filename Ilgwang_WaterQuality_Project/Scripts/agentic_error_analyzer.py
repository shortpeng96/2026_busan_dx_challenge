import pandas as pd
import numpy as np
import os

proj_dir = "C:\\Sandbox\\2026_busan_dx_challenge\\Ilgwang_WaterQuality_Project"
dump_path = os.path.join(proj_dir, "Results", "error_analysis_dump.csv")
df = pd.read_csv(dump_path)

# Calculate means by error type
# FP: actual=0, predicted=1
# FN: actual=1, predicted=0
# TP: actual=1, predicted=1
# TN: actual=0, predicted=0

df['type'] = ''
df.loc[(df['actual']==0) & (df['predicted_class']==1), 'type'] = 'FP'
df.loc[(df['actual']==1) & (df['predicted_class']==0), 'type'] = 'FN'
df.loc[(df['actual']==1) & (df['predicted_class']==1), 'type'] = 'TP'
df.loc[(df['actual']==0) & (df['predicted_class']==0), 'type'] = 'TN'

features = df.columns.drop(['actual', 'predicted_prob', 'predicted_class', 'is_error', 'type'])

means = df.groupby('type')[features].mean().T

print("=== AGENTIC ERROR ANALYSIS REPORT ===")
print(f"Total FP: {len(df[df['type']=='FP'])}")
print(f"Total FN: {len(df[df['type']=='FN'])}")
print(f"Total TP: {len(df[df['type']=='TP'])}")
print(f"Total TN: {len(df[df['type']=='TN'])}")

print("\\n[1] False Positives (AI predicted Danger, but it was Safe) vs True Negatives (Correctly Safe)")
print("Looking for features that were deceptively high/low in FP compared to TN:")
for f in features:
    fp_val = means.loc[f, 'FP'] if 'FP' in means.columns else 0
    tn_val = means.loc[f, 'TN'] if 'TN' in means.columns else 0
    if abs(fp_val - tn_val) > (abs(tn_val) * 0.5) and abs(fp_val - tn_val) > 0.1:  # 50% difference
        print(f"  - {f}: FP mean={fp_val:.2f}, TN mean={tn_val:.2f}")

print("\\n[2] False Negatives (AI predicted Safe, but it was Danger) vs True Positives (Correctly Danger)")
print("Looking for features that were unexpectedly high/low in FN compared to TP:")
for f in features:
    fn_val = means.loc[f, 'FN'] if 'FN' in means.columns else 0
    tp_val = means.loc[f, 'TP'] if 'TP' in means.columns else 0
    if abs(fn_val - tp_val) > (abs(tp_val) * 0.5) and abs(fn_val - tp_val) > 0.1:  # 50% difference
        print(f"  - {f}: FN mean={fn_val:.2f}, TP mean={tp_val:.2f}")

