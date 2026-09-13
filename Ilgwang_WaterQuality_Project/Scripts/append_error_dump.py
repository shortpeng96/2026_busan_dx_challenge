import os
filepath = r'C:\Sandbox\2026_busan_dx_challenge\Ilgwang_WaterQuality_Project\Scripts\modeling_ilgwang.py'
with open(filepath, 'a', encoding='utf-8') as f:
    f.write('\n\n# --- ERROR DUMP FOR AGENTIC ANALYSIS ---\n')
    f.write('error_df = X_imp_full.copy()\n')
    f.write('error_df["actual"] = y_bin\n')
    f.write('error_df["predicted_prob"] = y_pred_all\n')
    f.write('error_df["predicted_class"] = (y_pred_all >= 0.30).astype(int)\n')
    f.write('error_df["is_error"] = error_df["actual"] != error_df["predicted_class"]\n')
    f.write('errors_only = error_df[error_df["is_error"] == True]\n')
    f.write('errors_only.to_csv(os.path.join(out_dir, "error_analysis_dump.csv"), index=False)\n')
    f.write('print(f"Dumped {len(errors_only)} error cases to error_analysis_dump.csv")\n')
