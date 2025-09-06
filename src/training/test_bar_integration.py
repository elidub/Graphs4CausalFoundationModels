#!/usr/bin/env python3
"""
Test script to verify BarDistribution integration with SimplePFN
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

# Import our modules
from models.SimplePFN import SimplePFNRegressor
from Losses.BarDistribution import BarDistribution
from training.simplepfn_trainer import SimplePFNTrainer
from torch.utils.data import DataLoader, TensorDataset


def create_synthetic_data(num_samples=100, batch_size=16, n_train=20, n_test=10, num_features=5):
    """Create synthetic data for testing"""
    print(f"Creating synthetic data:")
    print(f"  - {num_samples} samples, batch_size={batch_size}")
    print(f"  - {n_train} train points, {n_test} test points")
    print(f"  - {num_features} features")
    
    # Generate random data
    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []
    
    for _ in range(num_samples):
        # Random training data
        X_train = torch.randn(n_train, num_features)
        y_train = torch.randn(n_train)  # 1D instead of (n_train, 1)
        
        # Random test data
        X_test = torch.randn(n_test, num_features)
        y_test = torch.randn(n_test)    # 1D instead of (n_test, 1)
        
        X_train_list.append(X_train)
        y_train_list.append(y_train)
        X_test_list.append(X_test)
        y_test_list.append(y_test)
    
    # Stack into tensors
    X_train_tensor = torch.stack(X_train_list)  # (num_samples, n_train, num_features)
    y_train_tensor = torch.stack(y_train_list)  # (num_samples, n_train)
    X_test_tensor = torch.stack(X_test_list)    # (num_samples, n_test, num_features)
    y_test_tensor = torch.stack(y_test_list)    # (num_samples, n_test)
    
    print(f"  - X_train shape: {X_train_tensor.shape}")
    print(f"  - y_train shape: {y_train_tensor.shape}")
    print(f"  - X_test shape: {X_test_tensor.shape}")
    print(f"  - y_test shape: {y_test_tensor.shape}")
    
    # Create a custom dataset that returns the 4-tuple format expected by SimplePFNTrainer
    class SimplePFNDataset(torch.utils.data.Dataset):
        def __init__(self, X_train, y_train, X_test, y_test):
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test
            
        def __len__(self):
            return len(self.X_train)
            
        def __getitem__(self, idx):
            # Return in the format expected by SimplePFNTrainer (with extra dimension for y)
            return [
                self.X_train[idx],
                self.y_train[idx].unsqueeze(-1),  # Add dimension: (n_train,) -> (n_train, 1)
                self.X_test[idx],
                self.y_test[idx].unsqueeze(-1)    # Add dimension: (n_test,) -> (n_test, 1)
            ]
    
    dataset = SimplePFNDataset(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create a special dataloader for BarDistribution fitting (expects 2D y tensors)
    class BarDistributionDataset(torch.utils.data.Dataset):
        def __init__(self, X_train, y_train, X_test, y_test):
            self.X_train = X_train
            self.y_train = y_train  # Keep as 2D (B, N)
            self.X_test = X_test
            self.y_test = y_test    # Keep as 2D (B, M)
            
        def __len__(self):
            return len(self.X_train)
            
        def __getitem__(self, idx):
            # Return in the format expected by BarDistribution.fit() (2D y tensors)
            return [
                self.X_train[idx],
                self.y_train[idx],  # (n_train,)
                self.X_test[idx],
                self.y_test[idx]    # (n_test,)
            ]
    
    bar_fit_dataset = BarDistributionDataset(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)
    bar_fit_dataloader = DataLoader(bar_fit_dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader, bar_fit_dataloader


def test_mse_training():
    """Test standard MSE training (baseline)"""
    print("\n" + "="*60)
    print("TESTING MSE TRAINING (BASELINE)")
    print("="*60)
    
    # Create synthetic data
    dataloader, _ = create_synthetic_data(num_samples=64, batch_size=8, n_train=15, n_test=8, num_features=5)
    
    # Create model with single output
    model = SimplePFNRegressor(
        num_features=5,
        d_model=64,
        depth=2,
        heads_feat=2,
        heads_samp=2,
        dropout=0.1,
        output_dim=1  # Single output for MSE
    )
    
    print(f"\nModel created with output_dim=1")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer without BarDistribution
    trainer = SimplePFNTrainer(
        model=model,
        dataloader=dataloader,
        learning_rate=1e-3,
        max_steps=50,
        device="cpu",
        bar_distribution=None  # No BarDistribution = MSE loss
    )
    
    print(f"\nStarting MSE training...")
    trained_model = trainer.fit()
    print(f"MSE training completed successfully!")
    
    return trained_model


def test_bar_distribution_training():
    """Test BarDistribution training"""
    print("\n" + "="*60)
    print("TESTING BAR DISTRIBUTION TRAINING")
    print("="*60)
    
    # Create synthetic data
    dataloader, bar_fit_dataloader = create_synthetic_data(num_samples=64, batch_size=8, n_train=15, n_test=8, num_features=5)
    
    # Create BarDistribution
    print(f"\nCreating BarDistribution...")
    num_bars = 5
    bar_distribution = BarDistribution(
        num_bars=num_bars,
        min_width=1e-6,
        scale_floor=1e-6,
        device=torch.device("cpu"),
        max_fit_items=1000,  # Limit for faster fitting
        log_prob_clip_min=-50.0,
        log_prob_clip_max=50.0
    )
    
    print(f"Fitting BarDistribution to data...")
    bar_distribution.fit(bar_fit_dataloader)
    output_dim = bar_distribution.num_params  # Should be num_bars + 4
    print(f"BarDistribution fitted: {num_bars} bars + 4 tail params = {output_dim} output dimensions")
    
    # Create model with BarDistribution output dimension
    model = SimplePFNRegressor(
        num_features=5,
        d_model=64,
        depth=2,
        heads_feat=2,
        heads_samp=2,
        dropout=0.1,
        output_dim=output_dim  # High-dimensional output for BarDistribution
    )
    
    print(f"\nModel created with output_dim={output_dim}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer with BarDistribution
    trainer = SimplePFNTrainer(
        model=model,
        dataloader=dataloader,
        learning_rate=1e-3,
        max_steps=50,
        device="cpu",
        bar_distribution=bar_distribution  # Use BarDistribution loss
    )
    
    print(f"\nStarting BarDistribution training...")
    trained_model = trainer.fit()
    print(f"BarDistribution training completed successfully!")
    
    return trained_model, bar_distribution


def test_model_predictions():
    """Test that the trained model can generate predictions in the right format"""
    print("\n" + "="*60)
    print("TESTING MODEL PREDICTIONS")
    print("="*60)
    
    # Train a BarDistribution model
    trained_model, bar_distribution = test_bar_distribution_training()
    
    # Create a small test batch
    batch_size = 4
    n_train, n_test = 10, 5
    num_features = 5
    
    X_train = torch.randn(batch_size, n_train, num_features)
    y_train = torch.randn(batch_size, n_train)  # Note: 2D for SimplePFN input
    X_test = torch.randn(batch_size, n_test, num_features)
    y_test = torch.randn(batch_size, n_test)    # Target values
    
    print(f"\nTesting model predictions:")
    print(f"  Input shapes: X_train={X_train.shape}, y_train={y_train.shape}, X_test={X_test.shape}")
    
    # Get model predictions
    with torch.no_grad():
        output = trained_model(X_train, y_train, X_test)
        predictions = output['predictions'] if isinstance(output, dict) else output
    
    print(f"  Output predictions shape: {predictions.shape}")
    print(f"  Expected shape for BarDistribution: ({batch_size}, {n_test}, {bar_distribution.num_params})")
    
    # Test BarDistribution methods
    print(f"\nTesting BarDistribution methods:")
    try:
        # Test log probability computation
        log_prob = bar_distribution.average_log_prob(predictions, y_test)
        print(f"  average_log_prob: {log_prob.shape} -> {log_prob.mean().item():.4f}")
        
        # Test mode computation
        mode = bar_distribution.mode(predictions)
        print(f"  mode: {mode.shape} -> mean={mode.mean().item():.4f}")
        
        # Test mean computation
        mean = bar_distribution.mean(predictions)
        print(f"  mean: {mean.shape} -> mean={mean.mean().item():.4f}")
        
        # Test sampling
        samples = bar_distribution.sample(predictions, num_samples=10)
        print(f"  sample: {samples.shape} -> mean={samples.mean().item():.4f}")
        
        print(f"All BarDistribution methods work correctly!")
        
    except Exception as e:
        print(f"  ERROR in BarDistribution methods: {e}")
        raise


def main():
    """Main test function"""
    print("="*60)
    print("SIMPLEPFN + BARDISTRIBUTION INTEGRATION TEST")
    print("="*60)
    
    try:
        # Test 1: MSE training (baseline)
        test_mse_training()
        
        # Test 2: BarDistribution training
        test_bar_distribution_training()
        
        # Test 3: Model predictions and BarDistribution methods
        test_model_predictions()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("SimplePFN + BarDistribution integration is working correctly.")
        print("="*60)
        
    except Exception as e:
        print("\n" + "="*60)
        print("TEST FAILED!")
        print("="*60)
        print(f"Error: {e}")
        print(f"Error type: {type(e).__name__}")
        
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        print("="*60)


if __name__ == "__main__":
    main()
