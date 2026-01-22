#!/usr/bin/env python3
"""
Convert ComplexMechIDK configs to ComplexMechNTest configs.

Changes:
1. Rename hide_X.X.yaml -> ntest_X.yaml (based on mapping)
2. Set hide_fraction_matrix to 0.0 (fixed)
3. Set n_test_samples_per_dataset to the new value
4. Update experiment names and descriptions
"""

import os
import yaml
import shutil
from pathlib import Path

# Mapping: hide_fraction -> n_test_samples
HIDE_TO_NTEST = {
    '0.0': 100,
    '0.25': 250,
    '0.5': 500,
    '0.75': 750,
    '1.0': 900,
}

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data, path):
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

def convert_config(config, node_count, variant, n_test):
    """Update config for n_test variant."""
    # Update experiment name
    old_name = config.get('experiment_name', '')
    new_name = old_name.replace('complexmech_idk', 'complexmech_ntest')
    for hide_val in ['0.0', '0.25', '0.5', '0.75', '1.0']:
        new_name = new_name.replace(f'_hide{hide_val}', f'_ntest{n_test}')
    config['experiment_name'] = new_name
    
    # Update description
    old_desc = config.get('description', '')
    new_desc = old_desc.replace('hide_fraction=', 'n_test_samples=')
    for hide_val, ntest_val in HIDE_TO_NTEST.items():
        new_desc = new_desc.replace(f'={hide_val}', f'={n_test}')
    config['description'] = new_desc
    
    # Set hide_fraction to 0.0 (always full graph knowledge)
    config['dataset_config']['hide_fraction_matrix'] = {'value': 0.0}
    
    # Set n_test_samples_per_dataset to new value
    config['dataset_config']['n_test_samples_per_dataset'] = {'value': n_test}
    
    # Update training_config
    if 'training_config' in config:
        tc = config['training_config']
        
        for key in ['experiment_name', 'save_dir', 'checkpoint_dir', 'wandb_run_name', 'wandb_notes']:
            if key in tc and isinstance(tc[key], dict) and 'value' in tc[key]:
                old_val = tc[key]['value']
                if isinstance(old_val, str):
                    new_val = old_val.replace('ComplexMechIDK', 'ComplexMechNTest')
                    new_val = new_val.replace('complexmech_idk', 'complexmech_ntest')
                    for hide_val in ['0.0', '0.25', '0.5', '0.75', '1.0']:
                        new_val = new_val.replace(f'_hide{hide_val}', f'_ntest{n_test}')
                        new_val = new_val.replace(f'hide{hide_val}', f'ntest{n_test}')
                    tc[key] = {'value': new_val}
        
        if 'wandb_project' in tc:
            tc['wandb_project'] = {'value': 'DoPFN_v2_ComplexMechNTest'}
        
        if 'wandb_tags' in tc and isinstance(tc['wandb_tags'], dict):
            old_tags = tc['wandb_tags'].get('value', [])
            new_tags = []
            for tag in old_tags:
                new_tag = tag.replace('complexmech_idk', 'complexmech_ntest')
                for hide_val in ['0.0', '0.25', '0.5', '0.75', '1.0']:
                    new_tag = new_tag.replace(f'hide{hide_val}', f'ntest{n_test}')
                new_tags.append(new_tag)
            tc['wandb_tags'] = {'value': new_tags}
    
    return config


def main():
    configs_dir = Path(__file__).parent / 'configs'
    
    print("Converting ComplexMechIDK configs to ComplexMechNTest...")
    print(f"Mapping: hide_fraction -> n_test_samples")
    for hide, ntest in HIDE_TO_NTEST.items():
        print(f"  hide_{hide} -> ntest_{ntest}")
    print()
    
    total_converted = 0
    
    for node_dir in sorted(configs_dir.iterdir()):
        if not node_dir.is_dir():
            continue
        
        node_count = node_dir.name
        print(f"\nProcessing {node_count}...")
        
        # Update base.yaml
        base_yaml = node_dir / 'base.yaml'
        if base_yaml.exists():
            config = load_yaml(base_yaml)
            config['experiment_name'] = config.get('experiment_name', '').replace('complexmech_idk', 'complexmech_ntest')
            if 'training_config' in config and 'wandb_project' in config['training_config']:
                config['training_config']['wandb_project'] = {'value': 'DoPFN_v2_ComplexMechNTest'}
            save_yaml(config, base_yaml)
            print(f"  Updated {base_yaml.name}")
        
        for variant_dir in sorted(node_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            
            variant = variant_dir.name
            print(f"  Processing {variant}...")
            
            for hide_yaml in sorted(variant_dir.glob('hide_*.yaml')):
                hide_str = hide_yaml.stem.replace('hide_', '')
                
                if hide_str not in HIDE_TO_NTEST:
                    print(f"    Warning: Unknown hide fraction {hide_str}, skipping")
                    continue
                
                n_test = HIDE_TO_NTEST[hide_str]
                
                config = load_yaml(hide_yaml)
                config = convert_config(config, node_count, variant, n_test)
                
                new_yaml = variant_dir / f'ntest_{n_test}.yaml'
                save_yaml(config, new_yaml)
                hide_yaml.unlink()
                
                print(f"    {hide_yaml.name} -> {new_yaml.name}")
                total_converted += 1
    
    print(f"\n{'='*60}")
    print(f"Conversion complete! Total: {total_converted}")


if __name__ == '__main__':
    main()
