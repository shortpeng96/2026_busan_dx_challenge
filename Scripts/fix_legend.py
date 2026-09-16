import os
import glob

def fix_legend_cutoff(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    target_str = "plt.savefig(os.path.join(RES, f'feature_importance_donut_{BEACH}.png'), dpi=150, facecolor=fig.get_facecolor())"
    replace_str = "plt.savefig(os.path.join(RES, f'feature_importance_donut_{BEACH}.png'), dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')"
    
    if target_str in content:
        content = content.replace(target_str, replace_str)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed {filepath}")
    else:
        print(f"Target string not found in {filepath}")

for path in glob.glob("*_WaterQuality_Project/Scripts/03_generate_results.py"):
    fix_legend_cutoff(path)
