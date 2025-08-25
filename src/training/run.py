#!/usr/bin/env python3
"""
Simple Training Runner - Load YAML configs and run training

This script reads YAML configuration files and converts them to the format
expected by SetupTraining, then runs the training pipeline.

Usage:
    python run.py path/to/config.yaml
    python run.py ../experiments/FirstTests/configs/early_test1.yaml
"""

import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from training.SetupTraining import SetupTraining


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary containing the loaded configuration
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"[OK] Loaded config: {config_path.name}")
    return config


def convert_yaml_to_training_format(yaml_config: Dict[str, Any]) -> tuple:
    """
    Convert YAML config to format expected by SetupTraining.
    
    Args:
        yaml_config: Raw YAML configuration dictionary
        
    Returns:
        Tuple of (model_config, data_config, training_config)
    """
    # Extract main sections
    model_config = yaml_config.get('model_config', {})
    training_config = yaml_config.get('training_config', {})
    
    # Bundle data configs together
    data_config = {
        'scm_config': yaml_config.get('scm_config', {}),
        'dataset_config': yaml_config.get('dataset_config', {}),
        'preprocessing_config': yaml_config.get('preprocessing_config', {})
    }
    
    # Validate critical model parameter
    max_features = data_config['dataset_config'].get('max_number_features', {}).get('value', 10)
    expected_features = max_features - 1  # One feature becomes target
    
    if 'num_features' not in model_config:
        model_config['num_features'] = expected_features
        print(f"[OK] Auto-set num_features = {expected_features} (max_features - 1)")
    elif model_config['num_features'] != expected_features:
        print(f"[WARN] num_features={model_config['num_features']} but expected {expected_features}")
    
    return model_config, data_config, training_config


def main():
    """Main training runner."""
    parser = argparse.ArgumentParser(description='Run SimplePFN training from YAML config')
    parser.add_argument('config_path', type=str, help='Path to YAML configuration file')
    parser.add_argument('--dry-run', action='store_true', help='Load config but do not run training')
    
    args = parser.parse_args()
    
    try:
        # Load and convert configuration
        print("=== SimplePFN Training Runner ===")
        yaml_config = load_yaml_config(args.config_path)
        model_config, data_config, training_config = convert_yaml_to_training_format(yaml_config)

        # Print experiment info
        exp_name = yaml_config.get('experiment_name', 'unnamed_experiment')
        description = yaml_config.get('description', 'No description provided')
        print(f"[OK] Experiment: {exp_name}")
        print(f"[OK] Description: {description}")

        # Print key settings
        print(f"[OK] Model features: {model_config.get('num_features', 'unknown')}")
        print(f"[OK] Training steps: {training_config.get('max_steps', {}).get('value', 'unknown')}")
        print(f"[OK] Batch size: {training_config.get('batch_size', {}).get('value', 'unknown')}")
        print(f"[OK] Device: {training_config.get('device', {}).get('value', 'unknown')}")

        if args.dry_run:
            print("[OK] Dry run completed - config loaded successfully")
            return

        # Initialize and run training
        print("\n=== Starting Training ===")
        setup = SetupTraining(
            model_config=model_config,
            data_config=data_config,
            training_config=training_config
        )

        print(f"[OK] Model: {type(setup.model).__name__}")
        print(f"[OK] Dataset size: {len(setup.train_dataset)}")

        # Run training
        trained_model = setup.run_training()

        # Save model if training completed
        save_dir = training_config.get('save_dir', {}).get('value', './models')
        save_path = Path(save_dir) / f"{exp_name}_final.pth"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        setup.save_model(str(save_path))

        print(f"[OK] Training completed successfully!")
        print(f"[OK] Model saved: {save_path}")

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
