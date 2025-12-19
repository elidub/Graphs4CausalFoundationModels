#!/usr/bin/env python3
# --- HPC/cluster preamble: MUST be first ---
# -------------------------------------------

"""
Simple Training Runner - Create dataloader in main and train SimplePFN models.
Uses YAML configuration files like the original approach but with clean dataloader-in-main structure.
"""

import sys
import os
import yaml
import json
import argparse
import time
import traceback
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from typing import Callable

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# AFTER the preamble, do project imports
from priordata_processing.Datasets.ObservationalDataset import ObservationalDataset
from priordata_processing.Datasets.InterventionalDataset import InterventionalDataset
# Import interpolated curriculum dataset (for t0/t1 curriculum configs)
from priordata_processing.Datasets.InterpolatedObservationalDataset import InterpolatedObservationalDataset
from models.SimplePFN import SimplePFNRegressor
from models.InterventionalPFN import InterventionalPFN
from models.GraphConditionedInterventionalPFN import GraphConditionedInterventionalPFN
from models.FlatGraphConditionedInterventionalPFN import FlatGraphConditionedInterventionalPFN
from models.UltimateGraphConditionedInterventionalPFN import UltimateGraphConditionedInterventionalPFN

# Import our training modules
from trainer import Trainer
from training_utils import load_yaml_config, extract_config_values, get_device, determine_input_size

# Import BarDistribution for probabilistic output
from Losses.BarDistribution import BarDistribution
from priordata_processing.Datasets.Collator import BatchSplitCollator

# Weights & Biases for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    raise ImportError("wandb library not found. Please install it or disable wandb logging in the config.")

# Robust import for Benchmark class
try:
    from benchmarking.Benchmark import Benchmark
except ImportError:
    try:
        from src.benchmarking.Benchmark import Benchmark
    except ImportError:
        # Fallback: add repo root to path
        repo_root = Path(__file__).parent.parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from src.benchmarking.Benchmark import Benchmark

# Import LinGausBenchmark for graph-conditioned model evaluation
LinGausBenchmark = None
try:
    # Try relative path from experiments/GraphConditioning/Benchmarks/LinGaus/
    lingaus_bench_path = Path(__file__).parent.parent.parent / "experiments" / "GraphConditioning" / "Benchmarks" / "LinGaus"
    if str(lingaus_bench_path) not in sys.path:
        sys.path.insert(0, str(lingaus_bench_path))
    from LingausBenchmark import LinGausBenchmark
except ImportError:
    print("Warning: LinGausBenchmark not available. LinGaus benchmark integration will be disabled.")


def _normalize_interp_name(s: str) -> str:
    if not isinstance(s, str) or not s:
        return 'linear'
    s2 = s.strip()
    if s2.endswith('.'):
        s2 = s2[:-1]
    return s2


def main():
    """Main entry point for SimplePFN training with YAML config."""
    parser = argparse.ArgumentParser(description="Run SimplePFN training with YAML config.")
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    args = parser.parse_args()

    try:
        print("=" * 60)
        print("           SimplePFN Training Session Started")
        print("=" * 60)
        
        config = load_yaml_config(args.config)
        
        # Print full config at the beginning
        print("\nFULL CONFIGURATION:")
        print("-" * 40)
        print(json.dumps(config, indent=2, default=str))
        print("-" * 40)

        # Extract mode flag (predictive or interventional)
        mode = config.get('mode', 'interventional').lower()
        if mode not in ['predictive', 'interventional']:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'predictive' or 'interventional'")
        
        # Check if graph conditioning is enabled (only for interventional mode)
        use_graph_conditioning = False
        graph_conditioning_mode = None
        if mode == 'interventional':
            model_config = config.get('model_config', {})
            # Extract the actual boolean/string values from the config format
            use_graph_cond_cfg = model_config.get('use_graph_conditioning', False)
            if isinstance(use_graph_cond_cfg, dict) and 'value' in use_graph_cond_cfg:
                use_graph_conditioning = use_graph_cond_cfg['value']
            else:
                use_graph_conditioning = use_graph_cond_cfg
            
            graph_cond_mode_cfg = model_config.get('graph_conditioning_mode', 'hard_attention_only')
            if isinstance(graph_cond_mode_cfg, dict) and 'value' in graph_cond_mode_cfg:
                graph_conditioning_mode = graph_cond_mode_cfg['value']
            else:
                graph_conditioning_mode = graph_cond_mode_cfg
        
        print(f"\nMODE:")
        print(f"   Training mode: {mode.upper()}")
        if mode == 'predictive':
            print(f"   Using: ObservationalDataset + SimplePFN")
        else:
            model_type_str = 'InterventionalPFN'
            if use_graph_conditioning:
                if graph_conditioning_mode == 'flat_append':
                    model_type_str = 'FlatGraphConditionedInterventionalPFN (flat adjacency append)'
                elif graph_conditioning_mode == 'ultimate_hard_attention_only':
                    model_type_str = 'UltimateGraphConditionedInterventionalPFN (hard attention only)'
                elif graph_conditioning_mode == 'ultimate_gcn_only':
                    model_type_str = 'UltimateGraphConditionedInterventionalPFN (GCN+AdaLN only)'
                elif graph_conditioning_mode == 'ultimate_gcn_and_hard_attention':
                    model_type_str = 'UltimateGraphConditionedInterventionalPFN (GCN+AdaLN+hard attention)'
                else:
                    model_type_str = 'GraphConditionedInterventionalPFN (hard masking)'
            print(f"   Using: InterventionalDataset + {model_type_str}")
            if use_graph_conditioning:
                print(f"   Graph conditioning mode: {graph_conditioning_mode}")
            else:
                print(f"   Graph conditioning: DISABLED")

        # Detect curriculum setup (t0/t1) vs classic single-config
        has_curriculum = ('scm_config_t0' in config and 'scm_config_t1' in config and
                          'dataset_config_t0' in config and 'dataset_config_t1' in config)

        # Extract config sections (classic)
        scm_config = config.get('scm_config', {})
        dataset_config = config.get('dataset_config', {})
        preprocessing_config = config.get('preprocessing_config', {})
        model_config = extract_config_values(config.get('model_config', {}))
        training_config = extract_config_values(config.get('training_config', {}))
        
        # Extract wandb config directly from training_config
        wandb_config = extract_config_values({
            k: v for k, v in config.get('training_config', {}).items() 
            if k.startswith('wandb_')
        })
        
        # Debug: Print wandb config
        print(f"\nDEBUG WANDB CONFIG:")
        for key, value in wandb_config.items():
            print(f"   {key}: {value}")
        
        # Initialize Weights & Biases if enabled
        wandb_run = None
        if WANDB_AVAILABLE and wandb_config.get('wandb_project'):
            print(f"\nINITIALIZING WEIGHTS & BIASES:")
            print(f"   Project: {wandb_config.get('wandb_project')}")
            print(f"   Offline mode: {wandb_config.get('wandb_offline', False)}")
            
            # Check for API key
            if 'WANDB_API_KEY' in os.environ:
                print(f"   API key: Found ({'***' + os.environ['WANDB_API_KEY'][-4:]})")
            else:
                print(f"   API key: Not found in environment")
            
            try:
                # Determine the run name based on HTCondor job ID (same format as log files)
                cluster_id = os.environ.get('CLUSTER_ID', '')  # HTCondor sets this
                proc_id = os.environ.get('PROCESS_ID', '0')    # HTCondor sets this
                
                # If HTCondor variables aren't available, try to extract from _CONDOR_JOB_AD
                if not cluster_id and '_CONDOR_JOB_AD' in os.environ:
                    try:
                        with open(os.environ['_CONDOR_JOB_AD'], 'r') as f:
                            for line in f:
                                if line.startswith('ClusterId = '):
                                    cluster_id = line.split('=')[1].strip()
                                elif line.startswith('ProcId = '):
                                    proc_id = line.split('=')[1].strip()
                    except Exception as e:
                        print(f"Warning: Could not read job ID from Condor job ad: {e}")
                
                # Use job ID for run name if available, otherwise use config name
                if cluster_id:
                    run_name = f"simple_pfn_{cluster_id}.{proc_id}"
                    print(f"   Using log file name as run name: {run_name}")
                else:
                    run_name = wandb_config.get('wandb_run_name', config.get('experiment_name'))
                    print(f"   Using config name as run name: {run_name}")
                
                wandb_run = wandb.init(
                    project=wandb_config.get('wandb_project'),
                    name=run_name,
                    tags=wandb_config.get('wandb_tags', []),
                    notes=wandb_config.get('wandb_notes', ''),
                    mode='offline' if wandb_config.get('wandb_offline', False) else 'online',
                    config={
                        'experiment_name': config.get('experiment_name'),
                        'description': config.get('description'),
                        **model_config,
                        **training_config,
                        'scm_config': scm_config,
                        'dataset_config': dataset_config,
                        'preprocessing_config': preprocessing_config
                    }
                )
                print(f"   Wandb initialized successfully!")
                print(f"   Run URL: {wandb_run.get_url()}")
                
                # Log the full raw config to wandb
                print(f"\n   Logging full configuration to wandb...")
                print(f"   " + "=" * 56)
                print(f"   FULL CONFIG DICT (being logged to wandb):")
                print(f"   " + "-" * 56)
                config_str = json.dumps(config, indent=2, default=str)
                for line in config_str.split('\n'):
                    print(f"   {line}")
                print(f"   " + "=" * 56)
                wandb_run.config.update({"full_config": config}, allow_val_change=True)
                print(f"   Full configuration logged to wandb successfully!\n")
            except Exception as e:
                print(f"   Wandb initialization failed: {e}")
                wandb_run = None
        else:
            print(f"\nWANDB DISABLED:")
            if not WANDB_AVAILABLE:
                print("   Reason: wandb not installed")
            else:
                print("   Reason: wandb_project not set in config")

        print(f"\nEXPERIMENT INFO:")
        print(f"   Name: {config.get('experiment_name', 'unnamed')}")
        print(f"   Description: {config.get('description', 'no description')}")
        
        print(f"\nMODEL CONFIG:")
        for key, value in model_config.items():
            print(f"   {key}: {value}")
            
        print(f"\nTRAINING CONFIG:")
        for key, value in training_config.items():
            print(f"   {key}: {value}")

        print(f"\nDATASET CREATION:")
        # Branch: curriculum (t0/t1) vs classic
        if has_curriculum:
            print("   Curriculum mode detected (t0/t1 configs)")
            interp_raw = config.get('interpolation_function', 'linear')
            interp_name = _normalize_interp_name(interp_raw)

            scm_t0 = config['scm_config_t0']
            scm_t1 = config['scm_config_t1']
            ds_t0 = config['dataset_config_t0']
            ds_t1 = config['dataset_config_t1']
            # preprocessing: accept either explicit t0/t1 or a single shared block
            pre_t0 = config.get('preprocessing_config_t0') or config.get('preprocessing_config') or {}
            pre_t1 = config.get('preprocessing_config_t1') or config.get('preprocessing_config') or {}

            # Auto-compute dataset_size when configured as None for t0/t1
            try:
                max_steps = int(training_config.get('max_steps', 10))
                batch_size_cfg = int(training_config.get('batch_size', 1))
                accum = int(training_config.get('accumulate_grad_batches', 1))
                computed_ds_size = max(1, max_steps * batch_size_cfg * accum)
            except Exception:
                computed_ds_size = None

            def _need_auto_size(dcfg: dict) -> bool:
                if not isinstance(dcfg, dict):
                    return False
                ds_field = dcfg.get('dataset_size')
                if ds_field is None:
                    return True
                if isinstance(ds_field, dict):
                    val = ds_field.get('value')
                else:
                    val = ds_field
                if val is None:
                    return True
                if isinstance(val, str) and val.strip().lower() == 'none':
                    return True
                return False

            if computed_ds_size is not None and (_need_auto_size(ds_t0) or _need_auto_size(ds_t1)):
                ds_t0.setdefault('dataset_size', {})
                ds_t1.setdefault('dataset_size', {})
                ds_t0['dataset_size']['value'] = computed_ds_size
                ds_t1['dataset_size']['value'] = computed_ds_size
                print(f"   Auto-set dataset_size for t0/t1 to {computed_ds_size} (max_steps={max_steps} * batch_size={batch_size_cfg} * accumulate_grad_batches={accum})")
                if wandb_run:
                    wandb_run.log({
                        'config/dataset_size_auto': computed_ds_size,
                        'config/accumulate_grad_batches': accum,
                        'config/max_steps': max_steps,
                        'config/batch_size': batch_size_cfg,
                    })

            print(f"   Building interpolated curriculum dataset (schedule={interp_name})...")
            dataset = InterpolatedObservationalDataset(
                scm_config_t0=scm_t0,
                scm_config_t1=scm_t1,
                preprocessing_config_t0=pre_t0,
                preprocessing_config_t1=pre_t1,
                dataset_config_t0=ds_t0,
                dataset_config_t1=ds_t1,
                interpolation_function=interp_name,
                seed=42,
            )
            print(f"   Interpolated curriculum dataset created with {len(dataset)} samples")
        else:
            # Classic (non-curriculum) mode: use mode flag to select dataset type
            if mode == 'predictive':
                print("   Creating observational dataset (predictive mode)...")
            else:
                print("   Creating interventional dataset (interventional mode)...")
            
            # Auto-compute dataset_size when configured as None for classic config too
            try:
                max_steps = int(training_config.get('max_steps', 10))
                batch_size_cfg = int(training_config.get('batch_size', 1))
                accum = int(training_config.get('accumulate_grad_batches', 1))
                computed_ds_size = max(1, max_steps * batch_size_cfg * accum)
            except Exception:
                computed_ds_size = None
            def _need_auto_size_single(dcfg: dict) -> bool:
                if not isinstance(dcfg, dict):
                    return False
                ds_field = dcfg.get('dataset_size')
                if ds_field is None:
                    return True
                if isinstance(ds_field, dict):
                    val = ds_field.get('value')
                else:
                    val = ds_field
                if val is None:
                    return True
                if isinstance(val, str) and val.strip().lower() == 'none':
                    return True
                return False
            if computed_ds_size is not None and _need_auto_size_single(dataset_config):
                dataset_config.setdefault('dataset_size', {})
                dataset_config['dataset_size']['value'] = computed_ds_size
                print(f"   Auto-set dataset_size to {computed_ds_size} (max_steps={max_steps} * batch_size={batch_size_cfg} * accumulate_grad_batches={accum})")
                if wandb_run:
                    wandb_run.log({
                        'config/dataset_size_auto': computed_ds_size,
                        'config/accumulate_grad_batches': accum,
                        'config/max_steps': max_steps,
                        'config/batch_size': batch_size_cfg,
                    })
            
            # Create dataset based on mode
            if mode == 'predictive':
                print("   Creating ObservationalDataset...")
                dataset = ObservationalDataset(
                    scm_config=scm_config,
                    preprocessing_config=preprocessing_config,
                    dataset_config=dataset_config,
                    seed=42,
                )
            else:
                print("   Creating InterventionalDataset...")
                # Enable adjacency matrix output if graph conditioning is enabled
                if use_graph_conditioning:
                    dataset_config['return_adjacency_matrix'] = {'value': True}
                    print("   Adjacency matrix output: ENABLED (for graph conditioning)")
                
                dataset = InterventionalDataset(
                    scm_config=scm_config,
                    preprocessing_config=preprocessing_config,
                    dataset_config=dataset_config,
                    seed=42,
                )
        print(f"   Dataset created with {len(dataset)} samples")

        # Create DataLoader in main method
        batch_size = training_config['batch_size']
        num_workers = training_config.get("num_workers", 4)
        print(f"\nDATALOADER SETUP:")
        print(f"   Batch size: {batch_size}")
        print(f"   Number of workers: {num_workers}")
        print(f"   Shuffle: False")
        
        # Build per-batch collator using dataset_config
        # Expected keys (new scheme):
        # - max_number_samples_per_dataset
        # - max_number_train_samples_per_dataset
        # - max_number_test_samples_per_dataset
        # - n_test_samples_per_dataset (can be fixed value or distribution spec)
        import torch.distributions as dist

        def _get_int(cfg: dict, key: str, default_val: int) -> int:
            raw = cfg.get(key, {"value": default_val})
            val = extract_config_values(raw)
            try:
                return int(val)
            except Exception:
                return int(default_val)

        max_total = _get_int(dataset_config, "max_number_samples_per_dataset", 1000)
        max_train_cap = _get_int(dataset_config, "max_number_train_samples_per_dataset", max_total)
        max_test_cap = _get_int(dataset_config, "max_number_test_samples_per_dataset", max_total)

        # Build distribution for n_test per dataset
        n_test_cfg = dataset_config.get("n_test_samples_per_dataset")
        if isinstance(n_test_cfg, dict) and n_test_cfg.get("distribution"):
            dist_type = n_test_cfg.get("distribution")
            params = n_test_cfg.get("distribution_parameters", {})
            if dist_type == "discrete_uniform":
                low = int(params.get("low", 0))
                high = int(params.get("high", max_total))
                # Use Uniform and cast to int inside collator
                n_test_dist = dist.Uniform(low=low, high=high)
            elif dist_type == "uniform":
                n_test_dist = dist.Uniform(low=float(params.get("low", 0)), high=float(params.get("high", max_total)))
            elif dist_type == "normal":
                n_test_dist = dist.Normal(loc=float(params.get("mean", max_total // 2)), scale=float(params.get("std", max(1, max_total/10))))
            else:
                # Fallback: fixed half split
                n_test_dist = max_total // 2
        else:
            # Fixed integer
            n_test_dist = extract_config_values(n_test_cfg) if n_test_cfg is not None else (max_total // 2)

        collator = BatchSplitCollator(
            max_number_samples_per_dataset=max_total,
            max_number_train_samples_per_dataset=max_train_cap,
            max_number_test_samples_per_dataset=max_test_cap,
            n_test_samples_distribution=n_test_dist,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
            persistent_workers=num_workers > 0  # Only use persistent workers when num_workers > 0
        )
        print(f"   DataLoader created successfully")

        # Test first batch to get input size
        print(f"\nANALYZING BATCH STRUCTURE:")
        first_batch = next(iter(dataloader))
        input_size, output_size = determine_input_size(first_batch)
        print(f"   Input features: {input_size}")
        print(f"   Output size: {output_size}")

        # Print shapes of the first few batches to verify collator behavior
        print(f"\nBATCH SHAPE DIAGNOSTICS (first 3 batches):")
        def _fmt_shape(x):
            try:
                return tuple(x.shape)
            except Exception:
                return 'n/a'
        try:
            for i, b in enumerate(dataloader):
                if i >= 3:
                    break
                if isinstance(b, (list, tuple)):
                    if len(b) == 4:
                        X_tr, Y_tr, X_te, Y_te = b
                        print(f"   [Batch {i}] observational -> X_tr{_fmt_shape(X_tr)} Y_tr{_fmt_shape(Y_tr)} | X_te{_fmt_shape(X_te)} Y_te{_fmt_shape(Y_te)}")
                    elif len(b) == 6:
                        X_tr, T_tr, Y_tr, X_te, T_te, Y_te = b
                        print(f"   [Batch {i}] interventional -> X_tr{_fmt_shape(X_tr)} T_tr{_fmt_shape(T_tr)} Y_tr{_fmt_shape(Y_tr)} | X_te{_fmt_shape(X_te)} T_te{_fmt_shape(T_te)} Y_te{_fmt_shape(Y_te)}")
                    elif len(b) == 7:
                        X_tr, T_tr, Y_tr, X_te, T_te, Y_te, adj = b
                        print(f"   [Batch {i}] interventional+graph -> X_tr{_fmt_shape(X_tr)} T_tr{_fmt_shape(T_tr)} Y_tr{_fmt_shape(Y_tr)} | X_te{_fmt_shape(X_te)} T_te{_fmt_shape(T_te)} Y_te{_fmt_shape(Y_te)} | adj{_fmt_shape(adj)}")
                    else:
                        print(f"   [Batch {i}] unknown batch format len={len(b)}: shapes={[ _fmt_shape(x) for x in b ]}")
                else:
                    print(f"   [Batch {i}] unexpected batch type: {type(b)}")
        except Exception as e:
            print(f"   Warning: failed to iterate diagnostic batches: {e}")
        
        # Use num_features from config if available, otherwise use detected input_size
        num_features = model_config.get("num_features", input_size)
        if num_features != input_size:
            print(f"   WARNING: Using config num_features ({num_features}) instead of detected ({input_size})")
        else:
            print(f"   Using detected num_features: {num_features}")
        
        # Create evaluation dataloaders (head & tail) if enabled
        eval_dataloaders = None
        eval_enabled = training_config.get("eval_enabled", False)
        eval_every = training_config.get("eval_every", 0)
        eval_setup_start = time.time()
        
        if eval_enabled and eval_every > 0:
            print(f"\nEVALUATION SETUP:")
            eval_batches = training_config.get("eval_batches", 10)
            n_eval_requested = eval_batches * batch_size
            total_len = len(dataset)
            if n_eval_requested <= 0:
                print("   Warning: eval_batches * batch_size <= 0; disabling evaluation")
            else:
                max_per_subset = max(1, min(n_eval_requested, total_len))
                if max_per_subset * 2 > total_len:
                    print("   Note: Requested eval slice size causes overlap (dataset too small). Proceeding with overlap.")
                head_end = min(max_per_subset, total_len)
                tail_start = max(0, total_len - max_per_subset)
                head_indices = list(range(0, head_end))
                tail_indices = list(range(tail_start, total_len))
                t0 = time.time()
                head_subset = Subset(dataset, head_indices)
                tail_subset = Subset(dataset, tail_indices)
                head_loader = DataLoader(head_subset, batch_size=batch_size, shuffle=False, num_workers=0)
                tail_loader = DataLoader(tail_subset, batch_size=batch_size, shuffle=False, num_workers=0)
                build_time = time.time() - t0
                eval_dataloaders = [head_loader, tail_loader]
                print(f"   Created 2 evaluation dataloaders (head & tail)")
                print(f"   Head subset size: {len(head_subset)} (indices 0..{head_end-1})")
                print(f"   Tail subset size: {len(tail_subset)} (indices {tail_start}..{total_len-1})")
                print(f"   Batches per subset (requested): {eval_batches} (will cap by subset length)")
                print(f"   Eval build time: {build_time:.4f}s")
                print(f"   Using dedicated evaluation dataloaders (training dataloader will NOT be used for evaluation)")
                if wandb_run:
                    wandb_run.log({'eval/build_time': build_time, 'eval/head_size': len(head_subset), 'eval/tail_size': len(tail_subset)})
                print(f"   Evaluation setup complete")
        else:
            print(f"\nEVALUATION: Disabled (eval_enabled={eval_enabled}, eval_every={eval_every})")
        if eval_dataloaders:
            print(f"Evaluation setup total time: {time.time() - eval_setup_start:.4f}s")
            
        # Get device first (needed for BarDistribution)
        device = get_device(training_config.get("device", "cpu"))
        print(f"\nCOMPUTE SETUP:")
        print(f"   Device: {device}")

        # Check if BarDistribution should be used
        use_bar_distribution = model_config.get("use_bar_distribution", False)
        bar_distribution = None
        output_dim = 1  # Default single output
        
        if use_bar_distribution:
            print(f"\nBAR DISTRIBUTION SETUP:")
            num_bars = model_config.get("num_bars", 11)
            min_width = model_config.get("min_width", 1e-6)
            scale_floor = model_config.get("scale_floor", 1e-6)
            max_fit_items = model_config.get("max_fit_items", None)
            max_fit_batches = model_config.get("max_fit_batches", None)
            log_prob_clip_min = model_config.get("log_prob_clip_min", -50.0)
            log_prob_clip_max = model_config.get("log_prob_clip_max", 50.0)
            
            print(f"   Creating BarDistribution with {num_bars} bars...")
            bar_distribution = BarDistribution(
                num_bars=num_bars,
                min_width=min_width,
                scale_floor=scale_floor,
                device=device,
                max_fit_items=max_fit_items,
                log_prob_clip_min=log_prob_clip_min,
                log_prob_clip_max=log_prob_clip_max
            )
            
            print(f"   Fitting BarDistribution to data...")
            if max_fit_batches is not None:
                print(f"   Using max {max_fit_batches} batches for fitting")
            
            bar_fit_dataloader = dataloader
            bar_distribution.fit(bar_fit_dataloader, max_batches=max_fit_batches)
            output_dim = bar_distribution.num_params  # K + 4 parameters
            print(f"   BarDistribution fitted with output dimension: {output_dim}")
            print(f"   Parameters needed: {num_bars} bars + 4 tail parameters = {output_dim}")
            
            # Store BarDistribution parameters for later saving with the trained model
            print("   BarDistribution fitted successfully and will be saved with model checkpoints")
        else:
            print(f"\n   Using standard MSE loss (no BarDistribution)")
            output_dim = model_config.get("output_dim", 1)
        
        # Create model based on mode
        print(f"\nMODEL CREATION:")
        if mode == 'predictive':
            print(f"   Creating SimplePFN model (predictive mode)...")
            model = SimplePFNRegressor(
                num_features=num_features,
                d_model=model_config.get("d_model", 8),
                depth=model_config.get("depth", 1), 
                heads_feat=model_config.get("heads_feat", 2),
                heads_samp=model_config.get("heads_samp", 2),
                dropout=model_config.get("dropout", 0.1),
                hidden_mult=model_config.get("hidden_mult", 4),
                output_dim=output_dim,  # Use calculated output dimension
                normalize_features=model_config.get("normalize_features", True),  # Apply normalization (default: True)
                n_sample_attention_sink_rows=model_config.get("n_sample_attention_sink_rows", 0),  # Attention sink rows
                n_feature_attention_sink_cols=model_config.get("n_feature_attention_sink_cols", 0),  # Attention sink columns
            )
        elif use_graph_conditioning:
            if graph_conditioning_mode == 'flat_append':
                print(f"   Creating FlatGraphConditionedInterventionalPFN model (interventional mode with flat graph conditioning)...")
                print(f"   Graph conditioning mode: {graph_conditioning_mode} (adjacency matrix flattened and appended to inputs)")
                model = FlatGraphConditionedInterventionalPFN(
                    num_features=num_features,
                    d_model=model_config.get("d_model", 8),
                    depth=model_config.get("depth", 1), 
                    heads_feat=model_config.get("heads_feat", 2),
                    heads_samp=model_config.get("heads_samp", 2),
                    dropout=model_config.get("dropout", 0.1),
                    hidden_mult=model_config.get("hidden_mult", 4),
                    output_dim=output_dim,  # Use calculated output dimension
                    normalize_features=model_config.get("normalize_features", True),  # Apply normalization (default: True)
                    n_sample_attention_sink_rows=model_config.get("n_sample_attention_sink_rows", 0),  # Attention sink rows
                    n_feature_attention_sink_cols=model_config.get("n_feature_attention_sink_cols", 0),  # Attention sink columns
                )
            elif graph_conditioning_mode == 'ultimate_hard_attention_only':
                print(f"   Creating UltimateGraphConditionedInterventionalPFN model (hard attention only)...")
                print(f"   Graph conditioning mode: {graph_conditioning_mode} (attention masking only, no GCN/AdaLN)")
                # Get soft attention bias parameters
                use_soft_attention_bias = model_config.get("use_soft_attention_bias", False)
                soft_bias_init = model_config.get("soft_bias_init", 5.0)
                if use_soft_attention_bias:
                    print(f"   Using soft attention bias (init: {soft_bias_init})")
                model = UltimateGraphConditionedInterventionalPFN(
                    num_features=num_features,
                    d_model=model_config.get("d_model", 8),
                    depth=model_config.get("depth", 1), 
                    heads_feat=model_config.get("heads_feat", 2),
                    heads_samp=model_config.get("heads_samp", 2),
                    dropout=model_config.get("dropout", 0.1),
                    hidden_mult=model_config.get("hidden_mult", 4),
                    output_dim=output_dim,  # Use calculated output dimension
                    normalize_features=model_config.get("normalize_features", True),  # Apply normalization (default: True)
                    n_sample_attention_sink_rows=model_config.get("n_sample_attention_sink_rows", 0),  # Attention sink rows
                    n_feature_attention_sink_cols=model_config.get("n_feature_attention_sink_cols", 0),  # Attention sink columns
                    use_attention_masking=True,  # Enable attention masking
                    use_gcn=False,  # Disable GCN
                    use_adaln=False,  # Disable AdaLN
                    use_soft_attention_bias=use_soft_attention_bias,
                    soft_bias_init=soft_bias_init,
                )
            elif graph_conditioning_mode == 'ultimate_gcn_only':
                print(f"   Creating UltimateGraphConditionedInterventionalPFN model (GCN+AdaLN only)...")
                print(f"   Graph conditioning mode: {graph_conditioning_mode} (GCN+AdaLN, no attention masking)")
                model = UltimateGraphConditionedInterventionalPFN(
                    num_features=num_features,
                    d_model=model_config.get("d_model", 8),
                    depth=model_config.get("depth", 1), 
                    heads_feat=model_config.get("heads_feat", 2),
                    heads_samp=model_config.get("heads_samp", 2),
                    dropout=model_config.get("dropout", 0.1),
                    hidden_mult=model_config.get("hidden_mult", 4),
                    output_dim=output_dim,  # Use calculated output dimension
                    normalize_features=model_config.get("normalize_features", True),  # Apply normalization (default: True)
                    n_sample_attention_sink_rows=model_config.get("n_sample_attention_sink_rows", 0),  # Attention sink rows
                    n_feature_attention_sink_cols=model_config.get("n_feature_attention_sink_cols", 0),  # Attention sink columns
                    use_attention_masking=False,  # Disable attention masking
                    use_gcn=True,  # Enable GCN
                    use_adaln=True,  # Enable AdaLN
                    use_soft_attention_bias=False,  # No soft bias without masking
                )
            elif graph_conditioning_mode == 'ultimate_gcn_and_hard_attention':
                print(f"   Creating UltimateGraphConditionedInterventionalPFN model (full conditioning)...")
                print(f"   Graph conditioning mode: {graph_conditioning_mode} (GCN+AdaLN+hard/soft attention masking)")
                # Get soft attention bias parameters
                use_soft_attention_bias = model_config.get("use_soft_attention_bias", False)
                soft_bias_init = model_config.get("soft_bias_init", 5.0)
                if use_soft_attention_bias:
                    print(f"   Using soft attention bias (init: {soft_bias_init})")
                model = UltimateGraphConditionedInterventionalPFN(
                    num_features=num_features,
                    d_model=model_config.get("d_model", 8),
                    depth=model_config.get("depth", 1), 
                    heads_feat=model_config.get("heads_feat", 2),
                    heads_samp=model_config.get("heads_samp", 2),
                    dropout=model_config.get("dropout", 0.1),
                    hidden_mult=model_config.get("hidden_mult", 4),
                    output_dim=output_dim,  # Use calculated output dimension
                    normalize_features=model_config.get("normalize_features", True),  # Apply normalization (default: True)
                    n_sample_attention_sink_rows=model_config.get("n_sample_attention_sink_rows", 0),  # Attention sink rows
                    n_feature_attention_sink_cols=model_config.get("n_feature_attention_sink_cols", 0),  # Attention sink columns
                    use_attention_masking=True,  # Enable attention masking
                    use_gcn=True,  # Enable GCN
                    use_adaln=True,  # Enable AdaLN
                    use_soft_attention_bias=use_soft_attention_bias,
                    soft_bias_init=soft_bias_init,
                )
            elif graph_conditioning_mode == 'ultimate_soft_attention':
                print(f"   Creating UltimateGraphConditionedInterventionalPFN model (soft attention only)...")
                print(f"   Graph conditioning mode: {graph_conditioning_mode} (soft attention bias only, no GCN/AdaLN)")
                print(f"   Using soft attention bias with default initialization")
                model = UltimateGraphConditionedInterventionalPFN(
                    num_features=num_features,
                    d_model=model_config.get("d_model", 8),
                    depth=model_config.get("depth", 1), 
                    heads_feat=model_config.get("heads_feat", 2),
                    heads_samp=model_config.get("heads_samp", 2),
                    dropout=model_config.get("dropout", 0.1),
                    hidden_mult=model_config.get("hidden_mult", 4),
                    output_dim=output_dim,  # Use calculated output dimension
                    normalize_features=model_config.get("normalize_features", True),  # Apply normalization (default: True)
                    n_sample_attention_sink_rows=model_config.get("n_sample_attention_sink_rows", 0),  # Attention sink rows
                    n_feature_attention_sink_cols=model_config.get("n_feature_attention_sink_cols", 0),  # Attention sink columns
                    use_attention_masking=True,  # Enable attention masking
                    use_gcn=False,  # Disable GCN
                    use_adaln=False,  # Disable AdaLN
                    use_soft_attention_bias=True,  # Enable soft attention bias
                    soft_bias_init=5.0,  # Use default initialization
                )
            elif graph_conditioning_mode == 'ultimate_gcn_and_soft_attention':
                print(f"   Creating UltimateGraphConditionedInterventionalPFN model (full soft conditioning)...")
                print(f"   Graph conditioning mode: {graph_conditioning_mode} (GCN+AdaLN+soft attention bias)")
                print(f"   Using soft attention bias with default initialization")
                model = UltimateGraphConditionedInterventionalPFN(
                    num_features=num_features,
                    d_model=model_config.get("d_model", 8),
                    depth=model_config.get("depth", 1), 
                    heads_feat=model_config.get("heads_feat", 2),
                    heads_samp=model_config.get("heads_samp", 2),
                    dropout=model_config.get("dropout", 0.1),
                    hidden_mult=model_config.get("hidden_mult", 4),
                    output_dim=output_dim,  # Use calculated output dimension
                    normalize_features=model_config.get("normalize_features", True),  # Apply normalization (default: True)
                    n_sample_attention_sink_rows=model_config.get("n_sample_attention_sink_rows", 0),  # Attention sink rows
                    n_feature_attention_sink_cols=model_config.get("n_feature_attention_sink_cols", 0),  # Attention sink columns
                    use_attention_masking=True,  # Enable attention masking
                    use_gcn=True,  # Enable GCN
                    use_adaln=True,  # Enable AdaLN
                    use_soft_attention_bias=True,  # Enable soft attention bias
                    soft_bias_init=5.0,  # Use default initialization
                )
            else:
                print(f"   Creating GraphConditionedInterventionalPFN model (interventional mode with hard graph conditioning)...")
                print(f"   Graph conditioning mode: {graph_conditioning_mode} (hard attention masking)")
                model = GraphConditionedInterventionalPFN(
                    num_features=num_features,
                    d_model=model_config.get("d_model", 8),
                    depth=model_config.get("depth", 1), 
                    heads_feat=model_config.get("heads_feat", 2),
                    heads_samp=model_config.get("heads_samp", 2),
                    dropout=model_config.get("dropout", 0.1),
                    hidden_mult=model_config.get("hidden_mult", 4),
                    output_dim=output_dim,  # Use calculated output dimension
                    normalize_features=model_config.get("normalize_features", True),  # Apply normalization (default: True)
                    n_sample_attention_sink_rows=model_config.get("n_sample_attention_sink_rows", 0),  # Attention sink rows
                    n_feature_attention_sink_cols=model_config.get("n_feature_attention_sink_cols", 0),  # Attention sink columns
                )
        else:
            print(f"   Creating InterventionalPFN model (interventional mode)...")
            model = InterventionalPFN(
                num_features=num_features,
                d_model=model_config.get("d_model", 8),
                depth=model_config.get("depth", 1), 
                heads_feat=model_config.get("heads_feat", 2),
                heads_samp=model_config.get("heads_samp", 2),
                dropout=model_config.get("dropout", 0.1),
                hidden_mult=model_config.get("hidden_mult", 4),
                output_dim=output_dim,  # Use calculated output dimension
                normalize_features=model_config.get("normalize_features", True),  # Apply normalization (default: True)
                n_sample_attention_sink_rows=model_config.get("n_sample_attention_sink_rows", 0),  # Attention sink rows
                n_feature_attention_sink_cols=model_config.get("n_feature_attention_sink_cols", 0),  # Attention sink columns
            )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        if mode == 'predictive':
            model_type = "SimplePFN"
        elif use_graph_conditioning:
            if graph_conditioning_mode == 'flat_append':
                model_type = "FlatGraphConditionedInterventionalPFN"
            elif graph_conditioning_mode == 'ultimate_hard_attention_only':
                model_type = "UltimateGraphConditionedInterventionalPFN (hard attention only)"
            elif graph_conditioning_mode == 'ultimate_gcn_only':
                model_type = "UltimateGraphConditionedInterventionalPFN (GCN+AdaLN only)"
            elif graph_conditioning_mode == 'ultimate_gcn_and_hard_attention':
                model_type = "UltimateGraphConditionedInterventionalPFN (full)"
            else:
                model_type = "GraphConditionedInterventionalPFN"
        else:
            model_type = "InterventionalPFN"
        print(f"   {model_type} model created")
        
        # Print attention sink configuration
        n_sink_rows = model_config.get("n_sample_attention_sink_rows", 0)
        n_sink_cols = model_config.get("n_feature_attention_sink_cols", 0)
        print(f"\nATTENTION SINK CONFIGURATION:")
        print(f"   Sample attention sink rows: {n_sink_rows} {'(disabled)' if n_sink_rows == 0 else '(enabled)'}")
        print(f"   Feature attention sink cols: {n_sink_cols} {'(disabled)' if n_sink_cols == 0 else '(enabled)'}")
        if n_sink_rows > 0 or n_sink_cols > 0:
            print(f"   Purpose: Provide stable attention targets for improved training stability")
        
        # Print normalization configuration
        normalize_enabled = model_config.get("normalize_features", True)
        print(f"\nFEATURE NORMALIZATION:")
        print(f"   Internal normalization: {'ENABLED' if normalize_enabled else 'DISABLED'}")
        if normalize_enabled:
            print(f"   Method: Quantile transform + Standard normalization")
            print(f"   - Step 1: Uniform quantile transform (support set)")
            print(f"   - Step 2: Mean/std normalization (support set)")
        else:
            print(f"   Features will be passed through as-is (ensure external preprocessing!)")
        
        # Extract training configuration parameters
        # Build scheduler config from individual keys in training_config
        scheduler_config = {}
        if training_config.get("scheduler_type"):
            scheduler_config["type"] = training_config.get("scheduler_type")
            scheduler_config["warmup_ratio"] = training_config.get("warmup_ratio", 0.03)
            scheduler_config["min_lr_ratio"] = training_config.get("min_lr_ratio", 0.1)
            print(f"\nSCHEDULER CONFIG:")
            print(f"   Type: {scheduler_config['type']}")
            print(f"   Warmup ratio: {scheduler_config['warmup_ratio']}")
            print(f"   Min LR ratio: {scheduler_config['min_lr_ratio']}")
        
        save_dir = training_config.get("checkpoint_dir")
        save_every = training_config.get("save_every", 0)
        # Respect both 'use_amp' and a 'precision' flag ("16" -> enable AMP)
        use_amp = training_config.get("use_amp", False)
        precision_flag = training_config.get("precision", None)
        if precision_flag in ("16", 16):
            use_amp = True
            # Ensure CUDA device when 16-bit requested
            if str(device).lower() == "cpu":
                print("WARNING: 16-bit precision requested but device is CPU. Falling back to float32. Set device: 'cuda' for AMP.")
        # Allow user to request bf16 via precision flag string "bf16" or explicit amp_dtype
        amp_dtype = None
        if isinstance(precision_flag, str) and precision_flag.strip().lower() == 'bf16':
            amp_dtype = 'bf16'
            use_amp = True
        elif training_config.get('amp_dtype', None) in ['bf16', 'fp16']:
            amp_dtype = training_config.get('amp_dtype')
        gradient_clip_val = training_config.get("gradient_clip_val", 0.0)
        
        # Print mixed precision training status
        print(f"\nMIXED PRECISION TRAINING:")
        if use_amp:
            dtype_name = 'bfloat16' if amp_dtype == 'bf16' else 'float16'
            print(f"   Status: ENABLED ({dtype_name})")
            print(f"   Expected speedup: 1.5-2x on modern GPUs (requires CUDA)")
            print(f"   Memory usage: ~50% reduction")
        else:
            print(f"   Status: DISABLED (float32)")
            print(f"   To enable: set use_amp: true in training_config")
        
        # Extract ensemble parameters from training config (shared with benchmark)
        norm_methods = training_config.get("norm_methods", None)
        outlier_strategies = training_config.get("outlier_strategies", None)
        
        # Extract preprocessing parameters from preprocessing_config (CRITICAL: must match training!)
        # These MUST be the same for training and benchmark to ensure consistency
        preproc_negative_one_one_scaling = extract_config_values(preprocessing_config.get("negative_one_one_scaling", {"value": True}))
        if isinstance(preproc_negative_one_one_scaling, dict):
            preproc_negative_one_one_scaling = preproc_negative_one_one_scaling.get("value", True)
        
        preproc_standardize = extract_config_values(preprocessing_config.get("standardize", {"value": True}))
        if isinstance(preproc_standardize, dict):
            preproc_standardize = preproc_standardize.get("value", True)
        
        preproc_yeo_johnson = extract_config_values(preprocessing_config.get("yeo_johnson", {"value": False}))
        if isinstance(preproc_yeo_johnson, dict):
            preproc_yeo_johnson = preproc_yeo_johnson.get("value", False)
        
        preproc_remove_outliers = extract_config_values(preprocessing_config.get("remove_outliers", {"value": True}))
        if isinstance(preproc_remove_outliers, dict):
            preproc_remove_outliers = preproc_remove_outliers.get("value", True)
        
        preproc_outlier_quantile = extract_config_values(preprocessing_config.get("outlier_quantile", {"value": 0.95}))
        if isinstance(preproc_outlier_quantile, dict):
            preproc_outlier_quantile = preproc_outlier_quantile.get("value", 0.95)
        
        # Shuffle parameters for benchmark (always true for reproducibility)
        preproc_shuffle_samples = True
        preproc_shuffle_features = True
        
        # PRINT TRAINING PREPROCESSING PARAMETERS
        print(f"\n{'='*60}")
        print(f"TRAINING PREPROCESSING PARAMETERS (from preprocessing_config)")
        print(f"{'='*60}")
        print(f"  negative_one_one_scaling: {preproc_negative_one_one_scaling}")
        print(f"  standardize: {preproc_standardize}")
        print(f"  yeo_johnson: {preproc_yeo_johnson}")
        print(f"  remove_outliers: {preproc_remove_outliers}")
        print(f"  outlier_quantile: {preproc_outlier_quantile}")
        print(f"  shuffle_samples: {preproc_shuffle_samples}")
        print(f"  shuffle_features: {preproc_shuffle_features}")
        print(f"{'='*60}\n")
        
        # Optional: build a Benchmark instance for periodic and final evaluation
        benchmark = None
        benchmark_enabled = training_config.get("benchmark_enabled", True)  # Default to True for backward compatibility
        benchmark_eval_fidelity = training_config.get("benchmark_eval_fidelity")
        benchmark_final_fidelity = training_config.get("benchmark_final_fidelity")
        
        if not benchmark_enabled:
            print(f"\nOPENML BENCHMARK DISABLED (benchmark_enabled=false in config)")
        
        if benchmark_enabled and (benchmark_eval_fidelity or benchmark_final_fidelity):
            print(f"\nBENCHMARK SETUP:")
            print(f"   Eval fidelity: {benchmark_eval_fidelity or 'disabled'}")
            print(f"   Final fidelity: {benchmark_final_fidelity or 'disabled'}")
            print(f"   CRITICAL: Using preprocessing_config values to match training!")
            print(f"   - negative_one_one_scaling: {preproc_negative_one_one_scaling}")
            print(f"   - standardize: {preproc_standardize}")
            print(f"   - yeo_johnson: {preproc_yeo_johnson}")
            print(f"   - remove_outliers: {preproc_remove_outliers}")
            print(f"   - outlier_quantile: {preproc_outlier_quantile}")
            
            # Configure benchmark subsampling to match training data limits from dataset_config
            # Extract from dataset_config (these define the actual training data size limits)
            max_n_features_from_dataset = extract_config_values(dataset_config.get("max_number_features", {"value": num_features}))
            if isinstance(max_n_features_from_dataset, dict):
                max_n_features_from_dataset = max_n_features_from_dataset.get("value", num_features)
            
            max_n_train_from_dataset = extract_config_values(dataset_config.get("max_number_train_samples", {"value": 100}))
            if isinstance(max_n_train_from_dataset, dict):
                max_n_train_from_dataset = max_n_train_from_dataset.get("value", 100)
            
            max_n_test_from_dataset = extract_config_values(dataset_config.get("max_number_test_samples", {"value": 100}))
            if isinstance(max_n_test_from_dataset, dict):
                max_n_test_from_dataset = max_n_test_from_dataset.get("value", 100)
            
            # For benchmark, use the same max limits as training to ensure consistency
            n_features_bm = int(max_n_features_from_dataset)
            max_n_features_bm = int(max_n_features_from_dataset)
            n_train_bm = int(max_n_train_from_dataset)
            max_n_train_bm = int(max_n_train_from_dataset)
            n_test_bm = int(max_n_test_from_dataset)
            max_n_test_bm = int(max_n_test_from_dataset)
            
            print(f"   Benchmark data limits (from dataset_config):")
            print(f"   - n_features / max_n_features: {n_features_bm}")
            print(f"   - n_train / max_n_train: {n_train_bm}")
            print(f"   - n_test / max_n_test: {n_test_bm}")

            benchmark = Benchmark(
                data_dir="data_cache",
                device=device,
                verbose=True,
                # benchmark configuration
                tasks=None,
                max_tasks=training_config.get("benchmark_max_tasks", 20),
                # subsampling (align with training config where possible)
                n_features=int(n_features_bm) if n_features_bm else 0,
                max_n_features=int(max_n_features_bm) if max_n_features_bm else 0,
                n_train=int(n_train_bm) if n_train_bm else 0,
                max_n_train=int(max_n_train_bm) if max_n_train_bm else 0,
                n_test=int(n_test_bm) if n_test_bm else 0,
                max_n_test=int(max_n_test_bm) if max_n_test_bm else 0,
                prefer_numeric=True,
                only_numeric=False,
                # model / output config
                config_path=args.config,
                output_csv="training_benchmark_results.csv",
                bootstrap_samples=int(training_config.get("benchmark_bootstrap_samples", 1000)),
                # SimplePFN ensemble parameters (n_estimators is benchmark-specific, but norm/outlier use training config)
                n_estimators=training_config.get("benchmark_n_estimators", 1),
                norm_methods=norm_methods,  # Use same as training
                outlier_strategies=outlier_strategies,  # Use same as training
                # preprocessing configuration (CRITICAL: use preprocessing_config values!)
                negative_one_one_scaling=preproc_negative_one_one_scaling,
                standardize=preproc_standardize,
                yeo_johnson=preproc_yeo_johnson,
                remove_outliers=preproc_remove_outliers,
                outlier_quantile=float(preproc_outlier_quantile),
                shuffle_samples=preproc_shuffle_samples,
                shuffle_features=preproc_shuffle_features,
                # logging
                quiet=False,
            )
            print(f"   Benchmark instance created with preprocessing matching training data!")
            print(f"   Ensemble: n_estimators={training_config.get('benchmark_n_estimators', 1)}")
            print(f"   Norm methods: {norm_methods}")
            print(f"   Outlier strategies: {outlier_strategies}")

        # Optional: build a LinGausBenchmark instance for periodic and final evaluation
        lingaus_benchmark = None
        lingaus_benchmark_eval_fidelity = training_config.get("lingaus_benchmark_eval_fidelity")
        lingaus_benchmark_final_fidelity = training_config.get("lingaus_benchmark_final_fidelity")
        if (lingaus_benchmark_eval_fidelity or lingaus_benchmark_final_fidelity) and LinGausBenchmark is not None:
            print(f"\nLINGAUS BENCHMARK SETUP:")
            print(f"   Eval fidelity: {lingaus_benchmark_eval_fidelity or 'disabled'}")
            print(f"   Final fidelity: {lingaus_benchmark_final_fidelity or 'disabled'}")
            
            # Get benchmark directory from config
            lingaus_benchmark_dir = training_config.get(
                "lingaus_benchmark_dir",
                "/Users/arikreuter/Documents/PhD/CausalPriorFitting/experiments/GraphConditioning/Benchmarks/LinGaus"
            )
            
            try:
                lingaus_benchmark = LinGausBenchmark(
                    benchmark_dir=lingaus_benchmark_dir,
                    cache_dir=None,  # Will use default: benchmark_dir/data_cache
                    verbose=True,
                    max_samples=None,  # Will be controlled by fidelity level
                )
                print(f"   LinGausBenchmark instance created!")
                print(f"   Benchmark dir: {lingaus_benchmark_dir}")
                print(f"   Data cache: {lingaus_benchmark.cache_dir}")
            except Exception as e:
                print(f"   WARNING: Failed to create LinGausBenchmark: {e}")
                lingaus_benchmark = None
        elif lingaus_benchmark_eval_fidelity or lingaus_benchmark_final_fidelity:
            print(f"\nLINGAUS BENCHMARK SETUP:")
            print(f"   WARNING: LinGaus benchmark requested but LinGausBenchmark class not available")
            print(f"   Eval fidelity: {lingaus_benchmark_eval_fidelity or 'disabled'}")
            print(f"   Final fidelity: {lingaus_benchmark_final_fidelity or 'disabled'}")

        # Initialize trainer
        trainer = Trainer(
            model=model,
            dataloader=dataloader,
            learning_rate=training_config.get("learning_rate", 1e-3),
            max_steps=training_config.get("max_steps", 10),
            device=device,
            wandb_run=wandb_run,  # Pass wandb run for logging
            scheduler_config=scheduler_config,
            save_dir=save_dir,
            save_every=save_every,
            run_name=run_name,
            bar_distribution=bar_distribution,  # Pass BarDistribution for probabilistic training
            eval_dataloaders=eval_dataloaders,  # Pass evaluation dataloaders list
            eval_every=training_config.get("eval_every", 0),  # Evaluation frequency
            eval_batches=training_config.get("eval_batches", 10),  # Number of eval batches
            accumulate_grad_batches=training_config.get("accumulate_grad_batches", 1),  # Gradient accumulation factor
            # Model selection parameters
            enable_model_selection=training_config.get("model_selection_enabled", False),
            model_selection_metric=training_config.get("model_selection_metric", "eval/mse_median"),
            model_selection_mode=training_config.get("model_selection_mode", "min"),
            # Benchmark integration
            config_path=args.config,
            benchmark_eval_fidelity=benchmark_eval_fidelity,
            benchmark_final_fidelity=benchmark_final_fidelity,
            benchmark=benchmark,
            # LinGaus Benchmark integration
            lingaus_benchmark_eval_fidelity=lingaus_benchmark_eval_fidelity,
            lingaus_benchmark_final_fidelity=lingaus_benchmark_final_fidelity,
            lingaus_benchmark=lingaus_benchmark,
            # Mixed precision training
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            gradient_clip_val=gradient_clip_val,
            schedule_name=interp_name if has_curriculum else "none",
        )
        
        print(f"   Learning rate: {training_config.get('learning_rate', 1e-3)}")
        print(f"   Max steps: {training_config.get('max_steps', 10)}")
        print(f"   Gradient clipping: {'enabled' if gradient_clip_val > 0 else 'disabled'}")
        if gradient_clip_val > 0:
            print(f"   Gradient clip value: {gradient_clip_val}")
        print(f"   Wandb logging: {'enabled' if wandb_run else 'disabled'}")
        print(f"   Model checkpoints: {'enabled' if save_dir else 'disabled'}")
        print(f"   Evaluation: {'enabled' if eval_dataloaders else 'disabled'}")
        if eval_dataloaders:
            print(f"   Eval frequency: every {training_config.get('eval_every', 0)} steps")
            print(f"   Eval batches: {training_config.get('eval_batches', 10)}")
        print(f"   Model selection: {'enabled' if training_config.get('model_selection_enabled', False) else 'disabled'}")
        if training_config.get('model_selection_enabled', False):
            print(f"   Selection metric: {training_config.get('model_selection_metric', 'eval/mse_median')} ({training_config.get('model_selection_mode', 'min')})")
        if save_dir:
            print(f"   Save directory: {save_dir}")
            print(f"   Save frequency: {'never' if save_every <= 0 else f'every {save_every} steps'}")
            print(f"   Run name: {run_name}")
        print(f"   Trainer initialized")
        
        # Start training
        print(f"\nSTARTING TRAINING:")
        print("=" * 60)
        trained_model = trainer.fit()
        
        # Save final model with BarDistribution parameters
        if training_config.get("checkpoint_dir") and bar_distribution is not None:
            print("\nSaving final model with BarDistribution...")
            final_model_path = trainer.save_model(
                filename="final_model_with_bardist.pt",
                metadata={'stage': 'final', 'training_complete': True}
            )
            print(f"Final model with BarDistribution saved to: {final_model_path}")

        print("=" * 60)
        print("SimplePFN Training Complete!")
        print("   Training completed successfully!")
        print("=" * 60)
        
        # Finish wandb run
        if wandb_run:
            wandb.finish()
            print("   Wandb run finished successfully!")
            
        return trained_model
        
    except Exception as e:
        print("=" * 60)
        print("TRAINING FAILED!")
        print("=" * 60)
        print(f"Error: {e}")
        print(f"Error type: {type(e).__name__}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("=" * 60)
        
        # Clean up wandb on failure
        if 'wandb_run' in locals() and wandb_run:
            wandb.finish(exit_code=1)
            print("   Wandb run terminated due to error")
            
        return None


if __name__ == "__main__":
    main()
