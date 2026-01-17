"""Simple script to run ComplexMechBenchmark with a specific model."""

import sys
import os
import json
import shutil
from pathlib import Path
from datetime import datetime

# Add repo paths
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, 'src'))

from ComplexMechBenchmark import ComplexMechBenchmark

# Model configuration
CONFIG_PATH = "/Users/arikreuter/Documents/PhD/CausalPriorFitting/experiments/FirstTests/checkpoints/final_earlytest_16773250.0/final_model_with_bardist_config.yaml"
CHECKPOINT_PATH = "/Users/arikreuter/Documents/PhD/CausalPriorFitting/experiments/FirstTests/checkpoints/final_earlytest_16773250.0/final_model_with_bardist.pt"

# Extract model ID from checkpoint path (e.g., "final_earlytest_16773250.0")
MODEL_ID = Path(CHECKPOINT_PATH).parent.name

# Initialize benchmark
benchmark = ComplexMechBenchmark()

# Load model
benchmark.load_model(
    config_path=CONFIG_PATH,
    checkpoint_path=CHECKPOINT_PATH,
)

# Run evaluation on 100 samples
results = benchmark.run_evaluation(max_samples=10, save_results=True)

# Compute aggregate metrics
import numpy as np
mse_values = [r['mse'] for r in results if 'mse' in r]
r2_values = [r['r2'] for r in results if 'r2' in r]
nll_values = [r['nll'] for r in results if 'nll' in r]

aggregate_metrics = {
    'mse_mean': float(np.mean(mse_values)),
    'mse_std': float(np.std(mse_values)),
    'r2_mean': float(np.mean(r2_values)),
    'r2_std': float(np.std(r2_values)),
    'nll_mean': float(np.mean(nll_values)),
    'nll_std': float(np.std(nll_values)),
    'n_samples': len(results)
}

print("\n=== Results ===")
print(f"MSE: {aggregate_metrics['mse_mean']:.4f} ± {aggregate_metrics['mse_std']:.4f}")
print(f"R²: {aggregate_metrics['r2_mean']:.4f} ± {aggregate_metrics['r2_std']:.4f}")
print(f"NLL: {aggregate_metrics['nll_mean']:.4f} ± {aggregate_metrics['nll_std']:.4f}")
print(f"Samples evaluated: {aggregate_metrics['n_samples']}")

# Create results directory with model ID and timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = Path(__file__).parent / "results" / f"{MODEL_ID}_{timestamp}"
results_dir.mkdir(parents=True, exist_ok=True)

print(f"\nSaving results to: {results_dir}")

# Save detailed results
with open(results_dir / "detailed_results.json", 'w') as f:
    json.dump(results, f, indent=2)

# Save aggregate metrics
with open(results_dir / "aggregate_metrics.json", 'w') as f:
    json.dump(aggregate_metrics, f, indent=2)

# Save model information
model_info = {
    'model_id': MODEL_ID,
    'config_path': CONFIG_PATH,
    'checkpoint_path': CHECKPOINT_PATH,
    'timestamp': timestamp,
}
with open(results_dir / "model_info.json", 'w') as f:
    json.dump(model_info, f, indent=2)

# Copy model config and checkpoint to results directory
shutil.copy2(CONFIG_PATH, results_dir / "model_config.yaml")
print(f"  Copied model config to: {results_dir / 'model_config.yaml'}")

# Note: Checkpoint file might be large, so we just save the path
# If you want to copy it too, uncomment the next line:
# shutil.copy2(CHECKPOINT_PATH, results_dir / "model_checkpoint.pt")

print(f"\nResults saved successfully!")
print(f"  Detailed results: {results_dir / 'detailed_results.json'}")
print(f"  Aggregate metrics: {results_dir / 'aggregate_metrics.json'}")
print(f"  Model info: {results_dir / 'model_info.json'}")
