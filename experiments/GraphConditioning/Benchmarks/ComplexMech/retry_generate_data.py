#!/usr/bin/env python3
"""
Hacky auto-retry wrapper for generate_all_variants_data.py with resume capability.
Reruns the script up to 25 times when it crashes, and can resume from where it left off.
"""

import subprocess
import sys
import time
import os
from pathlib import Path


def run_with_retry(script_args, max_attempts=25):
    """Run the data generation script with automatic retries."""
    
    script_name = "generate_all_variants_data.py"
    python_executable = "python"
    
    print("=" * 60)
    print("Auto-Retry Wrapper for ComplexMech Data Generation")
    print("=" * 60)
    print(f"Script: {script_name}")
    print(f"Python: {python_executable}")
    print(f"Max attempts: {max_attempts}")
    print(f"Script args: {' '.join(script_args)}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    attempt = 1
    total_successful_runs = 0
    
    while attempt <= max_attempts:
        print(f"\n{'='*20} ATTEMPT {attempt}/{max_attempts} {'='*20}")
        print(f"Starting attempt {attempt} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run the script
        try:
            cmd = [python_executable, script_name] + script_args
            print(f"Command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=Path.cwd(),
                capture_output=False,  # Let output go to console
                text=True,
                timeout=3600  # 1 hour timeout per attempt
            )
            
            exit_code = result.returncode
            print(f"\nExit code: {exit_code}")
            
            if exit_code == 0:
                print(f"✓ SUCCESS! Script completed successfully on attempt {attempt}")
                total_successful_runs += 1
                break
            else:
                print(f"✗ FAILED! Script crashed with exit code {exit_code}")
                
                # Analyze error type
                if exit_code == 134:
                    error_type = "SIGABRT (C extension refcount error)"
                elif exit_code == 139:
                    error_type = "SIGSEGV (segmentation fault)" 
                elif exit_code == 6:
                    error_type = "SIGABRT (abort trap)"
                elif exit_code == 130:
                    error_type = "SIGINT (interrupted by user)"
                    print("  User interrupted. Exiting.")
                    break
                else:
                    error_type = f"Unknown error (exit code {exit_code})"
                
                print(f"  Error type: {error_type}")
                
                if attempt < max_attempts:
                    wait_time = min(5 + (attempt - 1) * 2, 30)  # Increasing wait time, max 30s
                    print(f"  Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"  Max attempts ({max_attempts}) reached.")
                
        except subprocess.TimeoutExpired:
            print(f"✗ TIMEOUT! Script exceeded 1 hour limit on attempt {attempt}")
            if attempt < max_attempts:
                print("  Killing and retrying...")
            else:
                print(f"  Max attempts ({max_attempts}) reached.")
        
        except KeyboardInterrupt:
            print(f"\n✗ INTERRUPTED! User stopped execution on attempt {attempt}")
            break
        
        except Exception as e:
            print(f"✗ UNEXPECTED ERROR on attempt {attempt}: {e}")
        
        attempt += 1
    
    # Final summary
    print(f"\n{'='*60}")
    print("Final Summary:")
    print("=" * 60)
    
    if total_successful_runs > 0:
        print(f"✓ SUCCESS after {attempt} attempt(s)")
        print(f"Total successful runs: {total_successful_runs}")
        
        # Check what files were generated
        data_cache = Path("data_cache")
        if data_cache.exists():
            pkl_files = list(data_cache.glob("*.pkl"))
            print(f"Generated files: {len(pkl_files)} .pkl files in data_cache/")
            if len(pkl_files) <= 10:  # Only show if not too many
                for pkl_file in sorted(pkl_files):
                    size_mb = pkl_file.stat().st_size / 1024 / 1024
                    print(f"  - {pkl_file.name} ({size_mb:.1f} MB)")
        
        exit_code = 0
    else:
        print(f"✗ FAILED after {attempt-1} attempts")
        print("The script consistently crashes - there may be a fundamental issue.")
        print("\nPossible solutions:")
        print("- Reduce --num-samples to a smaller number (e.g., 100)")
        print("- Try fewer node counts (e.g., --node-counts 5)")
        print("- Use fewer variants (e.g., --variants base)")
        print("- Check system memory usage")
        exit_code = 1
    
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    return exit_code


if __name__ == "__main__":
    # Pass through all command line arguments except the script name
    script_args = sys.argv[1:]
    
    # Default args if none provided
    if not script_args:
        script_args = ["--num-samples", "1000"]
        print("No arguments provided, using default: --num-samples 1000")
    
    exit_code = run_with_retry(script_args)
    sys.exit(exit_code)