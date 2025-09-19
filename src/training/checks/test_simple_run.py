#!/usr/bin/env python3
"""
Simple test script for testing simple_run.py

This script creates a minimal test config and runs the simple_run.py script
to verify it works correctly without Unicode encoding issues.
"""

import sys
import os
import tempfile
import subprocess
from pathlib import Path
import yaml

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def load_and_modify_existing_config():
    """Load the existing FirstTests config and modify it for quick testing."""
    # Path to the existing config (from checks/ directory, need to go up more levels)
    config_path = Path(__file__).parent.parent.parent.parent / "experiments" / "FirstTests" / "configs" / "early_test2.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load the existing config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify for quick testing
    config['experiment_name'] = 'test_run'
    config['description'] = 'Quick test run based on early_test1 config'
    
    # Reduce training iterations for quick test

    config['training_config']['max_steps'] = {'value': 3}  # Very few steps
    config['training_config']['batch_size'] = {'value': 2}  # Small batch
    config['training_config']['device'] = {'value': 'cpu'}  # Force CPU usage
    config['training_config']['num_workers'] = {'value': 0}  # Avoid multiprocessing issues
    
    # Reduce dataset size for speed
    #if 'dataset_config' in config:
    config['dataset_config']['dataset_size'] = {'value': 20}  # Much smaller
    
    return config


def run_simple_run_test(config_path):
    """Run the simple_run.py script with the test config."""
    script_path = Path(__file__).parent.parent / 'simple_run.py'  # Go up one level to training directory
    
    if not script_path.exists():
        print(f"[ERROR] simple_run.py not found at: {script_path}")
        return False
    
    # Build command
    cmd = [
        sys.executable,
        str(script_path),
        '--config', str(config_path)
    ]
    
    print(f"[INFO] Running command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        # Run the script and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        print(f"\nReturn code: {result.returncode}")
        
        if result.returncode == 0:
            print("[OK] simple_run.py executed successfully!")
            return True
        else:
            print("[ERROR] simple_run.py failed!")
            return False
            
    except subprocess.TimeoutExpired:
        print("[ERROR] Test timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to run test: {e}")
        return False


def main():
    """Main test function."""
    print("=" * 60)
    print("         Testing simple_run.py Script")
    print("=" * 60)
    
    # Create temporary config file
    print("[INFO] Loading and modifying existing config...")
    config = load_and_modify_existing_config()
    
    # Write config to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f, default_flow_style=False)
        temp_config_path = f.name
    
    print(f"[INFO] Test config created: {temp_config_path}")
    
    try:
        # Run the test
        print("[INFO] Starting test run...")
        success = run_simple_run_test(temp_config_path)
        
        print("-" * 50)
        if success:
            print("[OK] TEST PASSED - simple_run.py works correctly!")
            return 0
        else:
            print("[ERROR] TEST FAILED - simple_run.py has issues!")
            return 1
            
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_config_path)
            print(f"[INFO] Cleaned up temporary config file")
        except Exception as e:
            print(f"[WARN] Could not clean up temp file: {e}")


if __name__ == "__main__":
    sys.exit(main())
