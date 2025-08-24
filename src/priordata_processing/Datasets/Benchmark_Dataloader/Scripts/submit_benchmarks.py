"""
HTCondor job submission script generator for benchmark runs.
Creates job files for different configurations and submits them.
"""

import os
import sys
from pathlib import Path


def load_config(config_name):
    """Load configuration from config file."""
    config_file = f"Configs/config_{config_name}.py"
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file {config_file} not found")
    
    # Load the config module
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"config_{config_name}", config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    return config_module


def create_run_script(config_name):
    """Create a run script for the specific configuration."""
    run_script_content = f"""#!/bin/bash
# Run script for benchmark config: {config_name}

# Extract environment
unzip venv.zip
source venv/bin/activate

# Set up Python path
cd CausalPriorFitting
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# Run the benchmark
cd src/priordata_processing/Datasets/Benchmark_Dataloader
python3 Scripts/run_benchmark.py {config_name}
"""
    
    script_file = f"Scripts/run_{config_name}.sh"
    with open(script_file, 'w') as f:
        f.write(run_script_content)
    
    # Make executable
    os.chmod(script_file, 0o755)
    return script_file


def create_job_file(config_name, config):
    """Create HTCondor job submission file."""
    
    job_content = f"""universe   = vanilla
executable = Scripts/run_{config_name}.sh
arguments  = 
output     = Logs/{config_name}_job.out
error      = Logs/{config_name}_job.err
log        = Logs/{config_name}_job.log

transfer_input_files = ../../../../venv.zip, ../../../../CausalPriorFitting
should_transfer_files = Yes
when_to_transfer_output = ON_EXIT

# Resource requests based on config
request_cpus = {config.REQUEST_CPUS}
request_memory = {config.REQUEST_MEMORY}
request_disk = {config.REQUEST_DISK}
request_gpus = 0

# Job metadata
+JobDescription = "DataLoader Benchmark - {config.DESCRIPTION}"
+ConfigName = "{config.CONFIG_NAME}"

queue
"""
    
    job_file = f"Scripts/{config_name}_job.sub"
    with open(job_file, 'w') as f:
        f.write(job_content)
    
    return job_file


def submit_job(job_file, bid_amount=1000):
    """Submit job to HTCondor with optional bid."""
    import subprocess
    
    try:
        # Submit with bid
        cmd = f"condor_submit_bid {bid_amount} {job_file}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Successfully submitted job: {job_file}")
            print(result.stdout)
            return True
        else:
            print(f"Failed to submit job: {job_file}")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"Error submitting job {job_file}: {e}")
        return False


def main():
    # Available configurations
    configs = ["small", "medium", "large", "high_memory"]
    
    if len(sys.argv) < 2:
        print("Usage: python submit_benchmarks.py <config_name|all>")
        print(f"Available configs: {', '.join(configs)}")
        sys.exit(1)
    
    requested_config = sys.argv[1]
    
    if requested_config == "all":
        configs_to_run = configs
    elif requested_config in configs:
        configs_to_run = [requested_config]
    else:
        print(f"Unknown config: {requested_config}")
        print(f"Available configs: {', '.join(configs)}")
        sys.exit(1)
    
    # Create Logs directory if it doesn't exist
    Path("Logs").mkdir(exist_ok=True)
    
    print(f"Preparing to submit {len(configs_to_run)} benchmark job(s)...")
    
    for config_name in configs_to_run:
        print(f"\nPreparing job for config: {config_name}")
        
        try:
            # Load config
            config = load_config(config_name)
            
            # Create run script
            run_script = create_run_script(config_name)
            print(f"Created run script: {run_script}")
            
            # Create job file  
            job_file = create_job_file(config_name, config)
            print(f"Created job file: {job_file}")
            
            # Submit job
            if submit_job(job_file):
                print(f"✓ Successfully submitted {config_name} benchmark")
            else:
                print(f"✗ Failed to submit {config_name} benchmark")
                
        except Exception as e:
            print(f"Error preparing job for {config_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nJob submission complete!")
    print("Use 'condor_q $USER' to check job status")


if __name__ == "__main__":
    main()
