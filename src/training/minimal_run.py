
"""
Minimal Training Runner - Bare minimum functionality for training.
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

# AFTER the preamble, do project imports
from training.MinimalSetupTraining import MinimalSetupTraining, minimal_quick_train


def create_minimal_config():
    """Create a minimal default configuration."""
    return {
        'model': {
            'model_type': 'small',
            'model_params': {}
        },
        'data': {
            'n_samples_train': 500,
            'n_samples_val': 100,
            'causal_config': {
                'n_vars': 5,
                'n_samples_per_graph': 100,
                'noise_type': 'gaussian'
            }
        },
        'training': {
            'batch_size': 16,
            'learning_rate': 1e-3,
            'max_steps': 10,
            'num_workers': 2,
            'device': 'cpu'
        }
    }


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print(f"Loaded config from: {config_path}")
        return config
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        print("Using minimal default configuration...")
        return create_minimal_config()


def main():
    """Main entry point for minimal training."""
    parser = argparse.ArgumentParser(description='Minimal Training Runner')
    parser.add_argument('config', nargs='?', help='Path to YAML configuration file')
    parser.add_argument('--default', action='store_true', 
                       help='Use default minimal configuration')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.default or args.config is None:
        print("Using minimal default configuration...")
        config = create_minimal_config()
    else:
        config = load_yaml_config(args.config)
    
    # Print configuration
    print("=== Minimal Training Configuration ===")
    print(f"Model: {config.get('model', {}).get('model_type', 'small')}")
    print(f"Training samples: {config.get('data', {}).get('n_samples_train', 500)}")
    print(f"Max steps: {config.get('training', {}).get('max_steps', 500)}")
    print(f"Batch size: {config.get('training', {}).get('batch_size', 16)}")
    print(f"Workers: {config.get('training', {}).get('num_workers', 1)}")
    print("=" * 40)
    
    try:
        # Run training using the minimal quick train function
        trained_model = minimal_quick_train(config)
        print("Minimal training completed successfully!")
        return trained_model
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
