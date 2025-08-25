#!/usr/bin/env python3
"""
Example usage of the training runner

Shows how to run experiments using YAML configurations.
"""

import subprocess
import sys
from pathlib import Path

# Get paths
src_dir = Path(__file__).parent.parent
training_dir = src_dir / "training"
experiments_dir = src_dir.parent / "experiments" / "FirstTests"

def run_experiment(config_name: str, dry_run: bool = False):
    """Run an experiment with the given config."""
    config_path = experiments_dir / "configs" / f"{config_name}.yaml"
    run_script = training_dir / "run.py"
    
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        return False

    if not run_script.exists():
        print(f"[ERROR] Run script not found: {run_script}")
        return False

    # Build command
    cmd = [sys.executable, str(run_script), str(config_path)]
    if dry_run:
        cmd.append("--dry-run")

    print(f"[INFO] Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[OK] Success!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("[ERROR] Failed!")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    """Run example experiments."""
    print("=== Training Runner Examples ===\n")
    
    # Example 1: Dry run to test config loading
    print("1. Testing config loading (dry run):")
    success = run_experiment("early_test1", dry_run=True)
    
    if not success:
        print("Config test failed - check your setup")
        return
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Actual training (uncomment to run)
    print("2. Actual training run:")
    print("Uncomment the line below to run actual training:")
    print("# run_experiment('early_test1', dry_run=False)")
    
    # Uncomment this line to run actual training:
    run_experiment("early_test1", dry_run=False)

if __name__ == "__main__":
    main()
