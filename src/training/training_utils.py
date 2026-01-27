"""
Training Utilities Module
Contains utility functions for configuration loading, device handling, and other common tasks.
"""

import yaml
import torch
from pathlib import Path


# Default config path
DEFAULT_CONFIG_PATH = "<REPO_ROOT>/experiments/FirstTests/configs/early_test.yaml"


def load_yaml_config(config_path: str = None):
    """Load YAML configuration file.
    
    Args:
        config_path: Path to config file. Can be:
            - Absolute path: /path/to/config.yaml
            - Relative path: resolved relative to current working directory
            - None: uses DEFAULT_CONFIG_PATH
    
    Returns:
        dict: Loaded configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    config_path = Path(config_path)
    
    # Try to resolve the path
    if not config_path.is_absolute():
        # First try relative to current working directory
        resolved_path = config_path.resolve()
        if not resolved_path.exists():
            # If that doesn't work, try relative to the script location
            # (useful when running from different directories)
            script_dir = Path(__file__).parent.parent.parent  # Go up to repo root
            alt_path = (script_dir / config_path).resolve()
            if alt_path.exists():
                config_path = alt_path
            else:
                config_path = resolved_path  # Use original for error message
        else:
            config_path = resolved_path
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"  Tried resolving relative to: {Path.cwd()}"
        )
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded config from: {config_path}")
    return config


def extract_config_values(config_dict):
    """Extract values from YAML config format (handles both 'value' and direct values)."""
    extracted = {}
    for key, value in config_dict.items():
        if isinstance(value, dict) and 'value' in value:
            extracted[key] = value['value']
        else:
            extracted[key] = value
    return extracted


def get_device(device_config):
    """Get device based on config and availability. Prints available GPU type and memory if CUDA is used."""
    if device_config == "cuda":
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
            print(f"Using CUDA device: {gpu_name} ({gpu_mem} MB memory available)")
            return "cuda"
        else:
            raise RuntimeError("CUDA device requested but not available")
    elif device_config == "cpu":
        print("Using CPU device")
        return "cpu"
    else:
        raise ValueError(f"Unknown device: {device_config}")


def determine_input_size(first_batch):
    """
    Determine input and output sizes from the first batch.
    
    Args:
        first_batch: First batch from the dataloader
        
    Returns:
        tuple: (input_size, output_size)
    """
    print(f"First batch type: {type(first_batch)}")
    print(f"First batch structure: {type(first_batch[0]) if isinstance(first_batch, (list, tuple)) else 'single item'}")
    
    if isinstance(first_batch, (tuple, list)) and len(first_batch) == 4:
        # SimplePFN format: [X_train, y_train, X_test, y_test]
        X_sample = first_batch[0]
        y_sample = first_batch[1]
        input_size = X_sample.shape[-1]
        output_size = y_sample.shape[-1] if len(y_sample.shape) > 1 else 1
    elif isinstance(first_batch, (tuple, list)) and len(first_batch) == 2:
        # Standard format: [X, y]
        X_sample, y_sample = first_batch
        if isinstance(X_sample, list):
            X_sample = X_sample[0] if len(X_sample) > 0 else torch.zeros(1, 10)
        if isinstance(y_sample, list):
            y_sample = y_sample[0] if len(y_sample) > 0 else torch.zeros(1, 1)
        input_size = X_sample.shape[-1]
        output_size = y_sample.shape[-1] if len(y_sample.shape) > 1 else 1
    else:
        # Single tensor
        X_sample = first_batch
        if isinstance(X_sample, list):
            X_sample = X_sample[0] if len(X_sample) > 0 else torch.zeros(1, 10)
        input_size = X_sample.shape[-1]
        output_size = input_size

    return input_size, output_size
