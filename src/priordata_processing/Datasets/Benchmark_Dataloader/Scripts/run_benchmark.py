"""
Main benchmark runner script that loads configurations and runs benchmarks.
This script should be executed on the cluster.
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

# Add the parent directories to the path to import our modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from torch.utils.data import DataLoader
from Benchmark_Dataloader import DataloaderBenchmark
from priordata_processing.Datasets.MakePurelyObservationalDataset import MakePurelyObservationalDataset
from priors.causal_prior.ExampleConfigs.Basic_Configs import default_sampling_config as prior_config
from priordata_processing.ExampleConfigs.BasicConfigs import default_dataset_config as dataset_config
from priordata_processing.ExampleConfigs.BasicConfigs import default_preprocessing_config as preprocessing_config


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


def save_results(results, config_name, timestamp):
    """Save benchmark results to Results folder."""
    results_dir = Path("Results")
    results_dir.mkdir(exist_ok=True)
    
    # Create detailed results with metadata
    detailed_results = {
        "config_name": config_name,
        "timestamp": timestamp,
        "results": results,
        "environment": {
            "hostname": os.uname().nodename if hasattr(os, 'uname') else "unknown",
            "python_version": sys.version,
        }
    }
    
    # Save as JSON
    result_file = results_dir / f"benchmark_results_{config_name}_{timestamp}.json"
    with open(result_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"Results saved to: {result_file}")
    return result_file


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_benchmark.py <config_name>")
        print("Available configs: small, medium, large, high_memory")
        sys.exit(1)
    
    config_name = sys.argv[1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Starting benchmark with config: {config_name}")
    print(f"Timestamp: {timestamp}")
    
    try:
        # Load configuration
        config = load_config(config_name)
        print(f"Loaded config: {config.DESCRIPTION}")
        
        # Create dataset
        print("Creating dataset...")
        dataset_factory = MakePurelyObservationalDataset(
            scm_config=prior_config,
            preprocessing_config=preprocessing_config,
            dataset_config=dataset_config,
        )
        dataset = dataset_factory.create_dataset(seed=42)
        print(f"Dataset created with {len(dataset)} items")
        
        # Create dataloader
        print("Creating DataLoader with:")
        print(f"  Batch size: {config.BATCH_SIZE}")
        print(f"  Num workers: {config.NUM_WORKERS}")
        print(f"  Prefetch factor: {config.PREFETCH_FACTOR}")
        print(f"  Shuffle: {config.SHUFFLE}")
        
        dataloader = DataLoader(
            dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=config.SHUFFLE, 
            num_workers=config.NUM_WORKERS, 
            prefetch_factor=config.PREFETCH_FACTOR
        )
        
        # Run benchmark
        benchmark = DataloaderBenchmark(dataloader, verbose=True)
        
        print("\n" + "="*60)
        print(f"DATALOADER BENCHMARK: {config.CONFIG_NAME.upper()}")
        print("="*60)
        
        results = benchmark.benchmark_batch_loading(
            num_batches=config.NUM_BATCHES, 
            warmup_batches=config.WARMUP
        )
        
        # Save results
        result_file = save_results(results, config_name, timestamp)
        
        print("\n" + "="*60)
        print("BENCHMARK COMPLETE!")
        print(f"Config: {config.CONFIG_NAME}")
        print(f"Results saved to: {result_file}")
        print("="*60)
        
    except Exception as e:
        print(f"Error running benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
