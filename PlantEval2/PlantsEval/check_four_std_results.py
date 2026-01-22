import pickle as pkl
from pathlib import Path
import numpy as np

# Load results from all four_std experiments
results = []
folders = [
    'results/catboost_plant_10ints_10reals_four_std_uniform',
    'results/dofm2_plant_four_std_all_unknown',
    'results/dofm2_plant_four_std_full_graph'
]

for folder in folders:
    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"Folder not found: {folder}")
        continue
    
    for file_path in folder_path.glob('*'):
        if file_path.is_file():
            try:
                with open(file_path, 'rb') as f:
                    res = pkl.load(f)
                    results.append({
                        'exp': folder.split('/')[-1],
                        'model': res['model'],
                        'dataset': res['dataset'],
                        'realization': res['realization'],
                        'mise': res['mise']
                    })
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

# Group by experiment
from collections import defaultdict
by_exp = defaultdict(list)
for r in results:
    by_exp[r['exp']].append(r['mise'])

print('Results for CID_10_ints_10_reals_four_std_uniform_std:')
print('='*70)
print()
for exp, mises in sorted(by_exp.items()):
    mean = np.mean(mises)
    stderr = np.std(mises, ddof=1) / np.sqrt(len(mises))
    print(f'{exp}:')
    print(f'  MISE: {mean:.4f} ± {stderr:.4f} (n={len(mises)})')
    print()

print('='*70)
print('SUMMARY:')
print()
print(f"catboost:     {np.mean(by_exp['catboost_plant_10ints_10reals_four_std_uniform']):.4f} ± {np.std(by_exp['catboost_plant_10ints_10reals_four_std_uniform'], ddof=1) / np.sqrt(len(by_exp['catboost_plant_10ints_10reals_four_std_uniform'])):.4f}")
print(f"all_unknown:  {np.mean(by_exp['dofm2_plant_four_std_all_unknown']):.4f} ± {np.std(by_exp['dofm2_plant_four_std_all_unknown'], ddof=1) / np.sqrt(len(by_exp['dofm2_plant_four_std_all_unknown'])):.4f}")
print(f"full_graph:   {np.mean(by_exp['dofm2_plant_four_std_full_graph']):.4f} ± {np.std(by_exp['dofm2_plant_four_std_full_graph'], ddof=1) / np.sqrt(len(by_exp['dofm2_plant_four_std_full_graph'])):.4f}")
