"""
Results analysis script for benchmark results.
Loads and analyzes benchmark results from the Results folder.
"""

import json
import os
from pathlib import Path
import pandas as pd
from typing import List, Dict


def load_all_results():
    """Load all benchmark results from Results folder."""
    results_dir = Path("Results")
    if not results_dir.exists():
        print("No Results folder found")
        return []
    
    results = []
    for result_file in results_dir.glob("benchmark_results_*.json"):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Error loading {result_file}: {e}")
    
    return results


def analyze_results(results: List[Dict]):
    """Analyze and summarize benchmark results."""
    if not results:
        print("No results to analyze")
        return
    
    print("="*80)
    print("BENCHMARK RESULTS ANALYSIS")
    print("="*80)
    
    # Create summary table
    summary_data = []
    
    for result in results:
        config_name = result['config_name']
        timestamp = result['timestamp']
        bench_results = result['results']
        
        summary_data.append({
            'Config': config_name,
            'Timestamp': timestamp,
            'Batches': bench_results['num_batches'],
            'Avg Time (ms)': f"{bench_results['average_time']*1000:.2f}",
            'Min Time (ms)': f"{bench_results['min_time']*1000:.2f}",
            'Max Time (ms)': f"{bench_results['max_time']*1000:.2f}",
            'Median Time (ms)': f"{bench_results['median_time']*1000:.2f}",
            'Std Dev (ms)': f"{bench_results['std_dev']*1000:.2f}",
            'Batches/sec': f"{bench_results['batches_per_second']:.2f}",
            'Total Time (s)': f"{bench_results['total_time']:.2f}",
            'CI Lower (ms)': f"{bench_results['ci_lower']*1000:.2f}",
            'CI Upper (ms)': f"{bench_results['ci_upper']*1000:.2f}",
        })
    
    # Convert to DataFrame for nice formatting
    try:
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))
    except ImportError:
        # Fallback without pandas
        print("\\nSummary (install pandas for better formatting):")
        for summary in summary_data:
            print(f"\\nConfig: {summary['Config']}")
            for key, value in summary.items():
                if key != 'Config':
                    print(f"  {key}: {value}")
    
    print("\\n" + "="*80)
    
    # Find best performing configuration
    best_throughput = max(results, key=lambda x: x['results']['batches_per_second'])
    best_latency = min(results, key=lambda x: x['results']['average_time'])
    
    print("PERFORMANCE HIGHLIGHTS:")
    print(f"Best Throughput: {best_throughput['config_name']} "
          f"({best_throughput['results']['batches_per_second']:.2f} batches/sec)")
    print(f"Best Latency: {best_latency['config_name']} "
          f"({best_latency['results']['average_time']*1000:.2f}ms avg)")
    
    print("="*80)


def compare_configs(results: List[Dict], config1: str, config2: str):
    """Compare two specific configurations."""
    config1_results = [r for r in results if r['config_name'] == config1]
    config2_results = [r for r in results if r['config_name'] == config2]
    
    if not config1_results:
        print(f"No results found for config: {config1}")
        return
    if not config2_results:
        print(f"No results found for config: {config2}")
        return
    
    # Use most recent results
    c1 = sorted(config1_results, key=lambda x: x['timestamp'])[-1]
    c2 = sorted(config2_results, key=lambda x: x['timestamp'])[-1]
    
    print(f"\\nCOMPARISON: {config1} vs {config2}")
    print("-" * 50)
    
    metrics = [
        ('Average Time', 'average_time', 'ms', 1000),
        ('Throughput', 'batches_per_second', 'batches/sec', 1),
        ('Total Time', 'total_time', 's', 1),
    ]
    
    for metric_name, key, unit, multiplier in metrics:
        val1 = c1['results'][key] * multiplier
        val2 = c2['results'][key] * multiplier
        
        if 'time' in key.lower():
            # Lower is better for time metrics
            better = config1 if val1 < val2 else config2
            improvement = abs((val2 - val1) / val2 * 100)
        else:
            # Higher is better for throughput
            better = config1 if val1 > val2 else config2
            improvement = abs((val1 - val2) / min(val1, val2) * 100)
        
        print(f"{metric_name:15}: {config1}={val1:.2f}{unit}, "
              f"{config2}={val2:.2f}{unit} (Winner: {better}, +{improvement:.1f}%)")


def main():
    import sys
    
    # Load all results
    results = load_all_results()
    
    if not results:
        print("No benchmark results found in Results folder")
        return
    
    # Analyze all results
    analyze_results(results)
    
    # If specific configs provided for comparison
    if len(sys.argv) >= 3:
        config1, config2 = sys.argv[1], sys.argv[2]
        compare_configs(results, config1, config2)


if __name__ == "__main__":
    main()
