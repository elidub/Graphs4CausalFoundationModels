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
# Import interpolated curriculum dataset (for t0/t1 curriculum configs)
from priordata_processing.Datasets.InterpolatedObservationalDataset import InterpolatedObservationalDataset
from models.SimplePFN import SimplePFNRegressor

# Import our training modules
from trainer import Trainer
from training_utils import load_yaml_config, extract_config_values, get_device, determine_input_size

# Import BarDistribution for probabilistic output
from Losses.BarDistribution import BarDistribution

# Weights & Biases for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    raise ImportError("wandb library not found. Please install it or disable wandb logging in the config.")


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
            print("   Creating observational dataset (classic)...")
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
            dataset = ObservationalDataset(
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
        print(f"   Shuffle: True")
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0  # Only use persistent workers when num_workers > 0
        )
        print(f"   DataLoader created successfully")

        # Test first batch to get input size
        print(f"\nANALYZING BATCH STRUCTURE:")
        first_batch = next(iter(dataloader))
        input_size, output_size = determine_input_size(first_batch)
        print(f"   Input features: {input_size}")
        print(f"   Output size: {output_size}")
        
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
        
        # Create SimplePFN model
        print(f"\nMODEL CREATION:")
        model = SimplePFNRegressor(
            num_features=num_features,
            d_model=model_config.get("d_model", 8),
            depth=model_config.get("depth", 1), 
            heads_feat=model_config.get("heads_feat", 2),
            heads_samp=model_config.get("heads_samp", 2),
            dropout=model_config.get("dropout", 0.1),
            hidden_mult=model_config.get("hidden_mult", 4),
            output_dim=output_dim  # Use calculated output dimension
        )
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   SimplePFN model created")
        
        # Create trainer with the dataloader
        print(f"\nTRAINER SETUP:")
        
        # Configure scheduler if enabled
        scheduler_config = None
        if "scheduler_type" in training_config:
            scheduler_type = training_config.get("scheduler_type")
            if scheduler_type == "linear_warmup_cosine_decay":
                # Get scheduler parameters from config
                warmup_ratio = training_config.get("warmup_ratio", 0.03)
                
                # Use explicit min_lr_ratio if provided, otherwise calculate from eta_min
                min_lr_ratio = training_config.get("min_lr_ratio")
                if min_lr_ratio is None and "eta_min" in training_config:
                    eta_min = training_config.get("eta_min", 0.0)
                    base_lr = training_config.get("learning_rate", 1e-3)
                    if base_lr > 0:  # Avoid division by zero
                        min_lr_ratio = eta_min / base_lr
                
                # Default if still not save
                if min_lr_ratio is None:
                    min_lr_ratio = 0.1
                
                scheduler_config = {
                    "type": scheduler_type,
                    "warmup_ratio": warmup_ratio,
                    "min_lr_ratio": min_lr_ratio
                }
                
                # Calculate actual warmup steps for display
                max_steps = training_config.get("max_steps", 10)
                warmup_steps = int(round(warmup_ratio * max_steps))
                
                print(f"   Scheduler: Linear warmup + cosine decay")
                print(f"   Warmup ratio: {warmup_ratio:.2f} ({warmup_steps} steps)")
                print(f"   Min LR ratio: {min_lr_ratio:.6f}")
            else:
                print(f"   Scheduler: None (constant learning rate)")
        
        # Get model saving configuration
        save_dir = training_config.get("checkpoint_dir")
        save_every = training_config.get("save_every", 0)  # 0 means no periodic saving
        
        # Get precision configuration
        precision = training_config.get("precision", "32")  # "32", "16", or "bf16"
        use_amp = precision in ["16", "16-mixed"]  # Enable mixed precision for float16
        
        # Get gradient clipping configuration
        gradient_clip_val = training_config.get("gradient_clip_val", 0.0)
        
        # Use the run name (derived from job ID or config) for model naming
        run_name = None
        if cluster_id:
            run_name = f"simple_pfn_{cluster_id}"
        else:
            run_name = config.get('experiment_name', 'simplepfn')
        
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
            benchmark_eval_fidelity=training_config.get("benchmark_eval_fidelity"),
            benchmark_final_fidelity=training_config.get("benchmark_final_fidelity"),
            # Mixed precision training
            use_amp=use_amp,
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
