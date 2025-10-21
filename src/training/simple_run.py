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
import warnings
import traceback
from pathlib import Path
from torch.utils.data import DataLoader

# Capture the initial working directory BEFORE any path manipulation
# This is where HTCondor unzips files (including data_cache)
INITIAL_WORKING_DIR = os.getcwd()

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# AFTER the preamble, do project imports
from priordata_processing.Datasets.MakePurelyObservationalDataset import MakePurelyObservationalDataset
from models.SimplePFN import SimplePFNRegressor

# Import our training modules
from simplepfn_trainer import SimplePFNTrainer
from training_utils import load_yaml_config, extract_config_values, get_device, determine_input_size

# Import BarDistribution for probabilistic output
from Losses.BarDistribution import BarDistribution


# Suppress duplicate warnings globally after first occurrence
def _suppress_warnings_after_first():
    seen_keys = set()
    _orig_showwarning = warnings.showwarning

    def _custom_showwarning(message, category, filename, lineno, file=None, line=None):
        key = (str(message), category.__name__)
        if key in seen_keys:
            return
        seen_keys.add(key)
        _orig_showwarning(message, category, filename, lineno, file, line)

    warnings.showwarning = _custom_showwarning
    # Ensure warnings aren't globally ignored by environment
    warnings.filterwarnings("default")


def _dl_worker_init_fn(worker_id: int):
    """Initialize DataLoader workers to also suppress duplicate warnings."""
    try:
        seen_keys = set()
        _orig_showwarning = warnings.showwarning

        def _custom_showwarning(message, category, filename, lineno, file=None, line=None):
            key = (str(message), category.__name__)
            if key in seen_keys:
                return
            seen_keys.add(key)
            _orig_showwarning(message, category, filename, lineno, file, line)

        warnings.showwarning = _custom_showwarning
        warnings.filterwarnings("default")
    except Exception:
        warnings.simplefilter("once")

# Weights & Biases for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    raise ImportError("wandb library not found. Please install it or disable wandb logging in the config.")


def main():
    """Main entry point for SimplePFN training with YAML config."""
    parser = argparse.ArgumentParser(description="Run SimplePFN training with YAML config.")
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    args = parser.parse_args()

    try:
        # Install global duplicate-warning suppression
        _suppress_warnings_after_first()
        print("=" * 60)
        print("           SimplePFN Training Session Started")
        print("=" * 60)
        
        config = load_yaml_config(args.config)
        
        # Print full config at the beginning
        print("\nFULL CONFIGURATION:")
        print("-" * 40)
        print(json.dumps(config, indent=2, default=str))
        print("-" * 40)

        # Extract config sections
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
                    },
                    group="DDP"
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
        print("   Creating dataset maker...")
        dataset_maker = MakePurelyObservationalDataset(
            scm_config=scm_config,
            preprocessing_config=preprocessing_config,
            dataset_config=dataset_config
        )

        print("   Creating dataset...")
        dataset = dataset_maker.create_dataset(seed=42)
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
            persistent_workers=num_workers > 0,  # Only use persistent workers when num_workers > 0
            worker_init_fn=_dl_worker_init_fn if num_workers and num_workers > 0 else None,
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
        
        # Create evaluation dataloader if enabled
        eval_dataloader = None
        eval_enabled = training_config.get("eval_enabled", False)
        eval_every = training_config.get("eval_every", 0)
        
        if eval_enabled and eval_every > 0:
            print(f"\nEVALUATION SETUP:")
            eval_batches = training_config.get("eval_batches", 10)
            print(f"   Using training dataloader for evaluation")
            print(f"   Evaluation batches: {eval_batches}")
            eval_dataloader = dataloader  # Use the same training dataloader
            print(f"   Evaluation setup complete")
        else:
            print(f"\nEVALUATION: Disabled (eval_enabled={eval_enabled}, eval_every={eval_every})")

            
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
            output_dim=output_dim,  # Use calculated output dimension
            use_flash_attention=model_config.get("use_flash_attention", False),
            use_feature_positional=model_config.get("use_feature_positional", True),
            feature_pos_rank=model_config.get("feature_pos_rank", 16),
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
        
        # Use the run name (derived from job ID or config) for model naming
        run_name = None
        if cluster_id:
            run_name = f"simple_pfn_{cluster_id}"
        else:
            run_name = config.get('experiment_name', 'simplepfn')
        
        # Initialize trainer
        trainer = SimplePFNTrainer(
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
            eval_dataloader=eval_dataloader,  # Pass evaluation dataloader
            eval_every=training_config.get("eval_every", 0),  # Evaluation frequency
            eval_batches=training_config.get("eval_batches", 10),  # Number of eval batches
            # Model selection parameters
            enable_model_selection=training_config.get("model_selection_enabled", False),
            model_selection_metric=training_config.get("model_selection_metric", "eval/mse_median"),
            model_selection_mode=training_config.get("model_selection_mode", "min"),
            # Benchmark integration
            config_path=args.config,
            benchmark_eval_fidelity=training_config.get("benchmark_eval_fidelity"),
            benchmark_final_fidelity=training_config.get("benchmark_final_fidelity"),
            benchmark_data_dir=training_config.get("benchmark_data_dir", os.path.join(INITIAL_WORKING_DIR, "data_cache")),
            benchmark_offline=training_config.get("benchmark_offline", True),
        )
        
        print(f"   Learning rate: {training_config.get('learning_rate', 1e-3)}")
        print(f"   Max steps: {training_config.get('max_steps', 10)}")
        print(f"   Wandb logging: {'enabled' if wandb_run else 'disabled'}")
        print(f"   Model checkpoints: {'enabled' if save_dir else 'disabled'}")
        print(f"   Evaluation: {'enabled' if eval_dataloader else 'disabled'}")
        if eval_dataloader:
            print(f"   Eval frequency: every {training_config.get('eval_every', 0)} steps")
            print(f"   Eval batches: {training_config.get('eval_batches', 10)}")
        print(f"   Model selection: {'enabled' if training_config.get('model_selection_enabled', False) else 'disabled'}")
        if training_config.get('model_selection_enabled', False):
            print(f"   Selection metric: {training_config.get('model_selection_metric', 'eval/mse_median')} ({training_config.get('model_selection_mode', 'min')})")
        
        # Print benchmark configuration
        if training_config.get("benchmark_eval_fidelity") or training_config.get("benchmark_final_fidelity"):
            benchmark_data_dir = training_config.get("benchmark_data_dir", os.path.join(INITIAL_WORKING_DIR, "data_cache"))
            print(f"   Benchmarking: enabled")
            print(f"   Benchmark data_cache path: {benchmark_data_dir}")
            print(f"   Benchmark data_cache exists: {os.path.exists(benchmark_data_dir)}")
            if training_config.get("benchmark_eval_fidelity"):
                print(f"   Benchmark eval fidelity: {training_config.get('benchmark_eval_fidelity')}")
            if training_config.get("benchmark_final_fidelity"):
                print(f"   Benchmark final fidelity: {training_config.get('benchmark_final_fidelity')}")
        if save_dir:
            print(f"   Save directory: {save_dir}")
            print(f"   Save frequency: {'never' if save_every <= 0 else f'every {save_every} steps'}")
            print(f"   Run name: {run_name}")
        print(f"   Trainer initialized")
        
        # Echo full config right before training starts
        print("\nCONFIG BEFORE TRAINING (full):")
        print("-" * 40)
        print(json.dumps(config, indent=2, default=str))
        print("-" * 40)

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
