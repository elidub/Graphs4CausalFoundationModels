#!/usr/bin/env python3
# --- HPC/cluster preamble: MUST be first ---
# -------------------------------------------

"""
Simple Training Runner - Create dataloader in main and train SimplePFN models.
Uses YAML configuration files like the original approach but with clean dataloader-in-main structure.
"""

import sys
import yaml
from pathlib import Path
from torch.utils.data import DataLoader

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


def main():
    """Main entry point for SimplePFN training with YAML config."""
    import argparse
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
        import json
        print(json.dumps(config, indent=2, default=str))
        print("-" * 40)

        # Extract config sections
        scm_config = config.get('scm_config', {})
        dataset_config = config.get('dataset_config', {})
        preprocessing_config = config.get('preprocessing_config', {})
        model_config = extract_config_values(config.get('model_config', {}))
        training_config = extract_config_values(config.get('training_config', {}))

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
            persistent_workers=True
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

            
        # Get device
        device = get_device(training_config.get("device", "cpu"))
        print(f"\nCOMPUTE SETUP:")
        print(f"   Device: {device}")
        
        # Create SimplePFN model
        print(f"\nMODEL CREATION:")
        model = SimplePFNRegressor(
            num_features=num_features,
            d_model=model_config.get("d_model", 8),
            depth=model_config.get("depth", 1), 
            heads_feat=model_config.get("heads_feat", 2),
            heads_samp=model_config.get("heads_samp", 2),
            dropout=model_config.get("dropout", 0.1)
        )
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   SimplePFN model created")
        
        # Create trainer with the dataloader
        print(f"\nTRAINER SETUP:")
        trainer = SimplePFNTrainer(
            model=model,
            dataloader=dataloader,
            learning_rate=training_config.get("learning_rate", 1e-3),
            max_steps=training_config.get("max_steps", 10),
            device=device
        )
        print(f"   Learning rate: {training_config.get('learning_rate', 1e-3)}")
        print(f"   Max steps: {training_config.get('max_steps', 10)}")
        print(f"   Trainer initialized")
        
        # Start training
        print(f"\nSTARTING TRAINING:")
        print("=" * 60)
        trained_model = trainer.fit()
        
        print("=" * 60)
        print("SimplePFN Training Complete!")
        print("   Training completed successfully!")
        print("=" * 60)
        return trained_model
        
    except Exception as e:
        print("=" * 60)
        print("TRAINING FAILED!")
        print("=" * 60)
        print(f"Error: {e}")
        print(f"Error type: {type(e).__name__}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        return None


if __name__ == "__main__":
    main()
