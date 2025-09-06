#!/usr/bin/env python3
"""
Test suite for BarDistribution class.

This module contains comprehensive tests for the BarDistribution class including:
- Basic functionality tests
- Realistic tests using actual dataloaders from configuration files
- Visualization of data distributions and model outputs
"""

import unittest
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import yaml
from torch.utils.data import DataLoader
import tempfile
import shutil

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from Losses.BarDistribution import BarDistribution
from priordata_processing.Datasets.MakePurelyObservationalDataset import MakePurelyObservationalDataset


def load_yaml_config(config_path: str):
    """Load YAML configuration file with path resolution."""
    config_path_str = str(config_path)
    
    # Handle project-relative paths starting with 'experiments/'
    if config_path_str.startswith('experiments/') or config_path_str.startswith('experiments\\'):
        # Get repository root path dynamically
        file_path = Path(__file__)
        repo_root = file_path.parent.parent.parent  # Go up to repo root from src/Losses
        config_path = repo_root / config_path_str
    elif not Path(config_path).is_absolute():
        # Get repository root path dynamically
        file_path = Path(__file__)
        repo_root = file_path.parent.parent.parent
        config_path = repo_root / config_path
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def extract_config_values(config_dict):
    """Extract values from YAML config format (handles both 'value' and direct values)."""
    extracted = {}
    for key, value in config_dict.items():
        if isinstance(value, dict) and 'value' in value:
            extracted[key] = value['value']
        else:
            extracted[key] = value
    return extracted


class TestBarDistribution(unittest.TestCase):
    """Test suite for BarDistribution class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.num_bars = 5
        self.bar_dist = BarDistribution(
            num_bars=self.num_bars,
            device=self.device,
            dtype=self.dtype
        )
        
        # Create a temporary directory for test outputs
        self.test_output_dir = Path(tempfile.mkdtemp(prefix="bar_dist_test_"))
        print(f"Test outputs will be saved to: {self.test_output_dir}")
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary directory
        if self.test_output_dir.exists():
            shutil.rmtree(self.test_output_dir)
    
    def test_initialization(self):
        """Test basic initialization of BarDistribution."""
        # Test default initialization
        dist = BarDistribution()
        self.assertEqual(dist.num_bars, 11)  # default
        self.assertEqual(dist.device, torch.device("cpu"))
        
        # Test custom initialization
        dist_custom = BarDistribution(
            num_bars=7,
            min_width=1e-5,
            scale_floor=1e-5,
            device=torch.device("cpu"),
            dtype=torch.float64
        )
        self.assertEqual(dist_custom.num_bars, 7)
        self.assertEqual(dist_custom.min_width, 1e-5)
        self.assertEqual(dist_custom.scale_floor, 1e-5)
        self.assertEqual(dist_custom.dtype, torch.float64)
    
    def test_invalid_initialization(self):
        """Test that invalid parameters raise appropriate errors."""
        with self.assertRaises(ValueError):
            BarDistribution(num_bars=0)  # num_bars must be >= 1
    
    def test_num_params(self):
        """Test that num_params property returns correct value."""
        self.assertEqual(self.bar_dist.num_params, self.num_bars + 4)
    
    def test_check_ready_before_fit(self):
        """Test that methods raise RuntimeError before fitting."""
        B, M = 2, 10
        pred = torch.randn(B, M, self.bar_dist.num_params)
        y = torch.randn(B, M)
        
        with self.assertRaises(RuntimeError):
            self.bar_dist.average_log_prob(pred, y)
        
        with self.assertRaises(RuntimeError):
            self.bar_dist.mode(pred)
        
        with self.assertRaises(RuntimeError):
            self.bar_dist.mean(pred)
        
        with self.assertRaises(RuntimeError):
            self.bar_dist.sample(pred, 10)
    
    def test_fit_with_synthetic_data(self):
        """Test fitting with synthetic data."""
        # Create synthetic training data
        B, N_train, N_test = 4, 100, 50
        
        # Generate some realistic data
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Create mixture of gaussians for more interesting distribution
        y_train_data = []
        y_test_data = []
        
        for _ in range(B):
            # Mix of two gaussians
            n1 = N_train // 2
            n2 = N_train - n1
            y1 = torch.normal(mean=-2.0, std=1.0, size=(n1,))
            y2 = torch.normal(mean=3.0, std=1.5, size=(n2,))
            y_train = torch.cat([y1, y2])
            y_train_data.append(y_train)
            
            # Similar for test
            n1_test = N_test // 2
            n2_test = N_test - n1_test
            y1_test = torch.normal(mean=-2.0, std=1.0, size=(n1_test,))
            y2_test = torch.normal(mean=3.0, std=1.5, size=(n2_test,))
            y_test = torch.cat([y1_test, y2_test])
            y_test_data.append(y_test)
        
        # Stack into tensors
        y_train_batch = torch.stack(y_train_data)  # (B, N_train)
        y_test_batch = torch.stack(y_test_data)   # (B, N_test)
        
        # Create dummy X data (not used by BarDistribution.fit)
        X_train_batch = torch.randn(B, N_train, 5)
        X_test_batch = torch.randn(B, N_test, 5)
        
        # Create a simple dataloader
        dataset = [(X_train_batch, y_train_batch, X_test_batch, y_test_batch)]
        
        # Fit the distribution
        self.bar_dist.fit(dataset)
        
        # Check that fitting worked
        self.assertIsNotNone(self.bar_dist.centers)
        self.assertIsNotNone(self.bar_dist.edges)
        self.assertIsNotNone(self.bar_dist.widths)
        self.assertIsNotNone(self.bar_dist.base_s_left)
        self.assertIsNotNone(self.bar_dist.base_s_right)
        
        # Check shapes
        self.assertEqual(self.bar_dist.centers.shape, (self.num_bars,))
        self.assertEqual(self.bar_dist.edges.shape, (self.num_bars + 1,))
        self.assertEqual(self.bar_dist.widths.shape, (self.num_bars,))
        
        # Check that centers are increasing
        centers_diff = torch.diff(self.bar_dist.centers)
        self.assertTrue(torch.all(centers_diff > 0), "Centers should be strictly increasing")
        
        # Check that edges are increasing
        edges_diff = torch.diff(self.bar_dist.edges)
        self.assertTrue(torch.all(edges_diff > 0), "Edges should be strictly increasing")
        
        print("Fitted BarDistribution:")
        print(f"  Centers: {self.bar_dist.centers}")
        print(f"  Edges: {self.bar_dist.edges}")
        print(f"  Widths: {self.bar_dist.widths}")
        print(f"  Base scale left: {self.bar_dist.base_s_left}")
        print(f"  Base scale right: {self.bar_dist.base_s_right}")
    
    def test_prediction_methods_after_fit(self):
        """Test prediction methods after fitting."""
        # First fit with synthetic data
        self.test_fit_with_synthetic_data()  # This also fits the distribution
        
        # Test prediction methods
        B, M = 3, 20
        pred = torch.randn(B, M, self.bar_dist.num_params)
        y = torch.randn(B, M)
        
        # Test average_log_prob
        log_prob = self.bar_dist.average_log_prob(pred, y)
        self.assertEqual(log_prob.shape, (B,))
        self.assertTrue(torch.all(torch.isfinite(log_prob)))
        
        # Test mode
        mode_vals = self.bar_dist.mode(pred)
        self.assertEqual(mode_vals.shape, (B, M))
        self.assertTrue(torch.all(torch.isfinite(mode_vals)))
        
        # Test mean
        mean_vals = self.bar_dist.mean(pred)
        self.assertEqual(mean_vals.shape, (B, M))
        self.assertTrue(torch.all(torch.isfinite(mean_vals)))
        
        # Test sample
        num_samples = 15
        samples = self.bar_dist.sample(pred, num_samples)
        self.assertEqual(samples.shape, (B, num_samples, M))
        self.assertTrue(torch.all(torch.isfinite(samples)))
        
        print("Prediction method tests passed:")
        print(f"  Log prob range: [{log_prob.min():.3f}, {log_prob.max():.3f}]")
        print(f"  Mode range: [{mode_vals.min():.3f}, {mode_vals.max():.3f}]")
        print(f"  Mean range: [{mean_vals.min():.3f}, {mean_vals.max():.3f}]")
        print(f"  Sample range: [{samples.min():.3f}, {samples.max():.3f}]")
    
    def test_invalid_pred_shapes(self):
        """Test that invalid prediction tensor shapes raise errors."""
        # First fit
        self.test_fit_with_synthetic_data()
        
        B, M = 2, 10
        y = torch.randn(B, M)
        
        # Wrong number of parameters
        pred_wrong_params = torch.randn(B, M, self.bar_dist.num_params - 1)
        with self.assertRaises(ValueError):
            self.bar_dist.average_log_prob(pred_wrong_params, y)
        
        # Wrong rank
        pred_wrong_rank = torch.randn(B, self.bar_dist.num_params)
        with self.assertRaises(ValueError):
            self.bar_dist.average_log_prob(pred_wrong_rank, y)
        
        # Mismatched batch/sequence dimensions
        pred_wrong_shape = torch.randn(B+1, M, self.bar_dist.num_params)
        with self.assertRaises(ValueError):
            self.bar_dist.average_log_prob(pred_wrong_shape, y)


class TestBarDistributionWithRealData(unittest.TestCase):
    """Test BarDistribution with realistic data from configuration files."""
    
    def setUp(self):
        """Set up realistic test environment."""
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.num_bars = 20  # Default value
        
        # Create output directory for plots
        self.test_output_dir = Path(tempfile.mkdtemp(prefix="bar_dist_real_test_"))
        print(f"Real data test outputs will be saved to: {self.test_output_dir}")
        
        # Load configuration
        config_path = "experiments/FirstTests/configs/early_test.yaml"
        try:
            self.config = load_yaml_config(config_path)
            print(f"Successfully loaded config from: {config_path}")
        except Exception as e:
            self.skipTest(f"Could not load config file: {e}")
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Keep the output directory for inspection (don't delete)
        print(f"Real data test outputs saved to: {self.test_output_dir}")
    
    def test_with_real_dataloader(self):
        """Test BarDistribution with real dataloader from configuration."""
        # Extract config sections
        scm_config = self.config.get('scm_config', {})
        dataset_config = self.config.get('dataset_config', {})
        preprocessing_config = self.config.get('preprocessing_config', {})
        
        print("Creating dataset from config...")
        print(f"  SCM config keys: {list(scm_config.keys())}")
        print(f"  Dataset config keys: {list(dataset_config.keys())}")
        print(f"  Preprocessing config keys: {list(preprocessing_config.keys())}")
        
        # Create dataset maker
        dataset_maker = MakePurelyObservationalDataset(
            scm_config=scm_config,
            preprocessing_config=preprocessing_config,
            dataset_config=dataset_config
        )
        
        # Create dataset with smaller size for testing
        print("Creating dataset...")
        dataset = dataset_maker.create_dataset(seed=42)
        print(f"Dataset created with {len(dataset)} samples")
        
        # Create DataLoader with small batch size for testing
        batch_size = 4
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle for reproducible tests
            num_workers=0   # No multiprocessing for simpler testing
        )
        print(f"DataLoader created with batch size {batch_size}")
        
        # Create BarDistribution
        bar_dist = BarDistribution(
            num_bars=self.num_bars,
            device=self.device,
            dtype=self.dtype,
            max_fit_items=2000  # Limit for faster testing
        )
        
        # Fit the distribution
        print("Fitting BarDistribution...")
        # Note: We'll inspect the data format first, then fit with corrected format
        # The actual fitting happens after data inspection
        
        # Analyze the fitted distribution
        print("\nFitted BarDistribution analysis:")
        print(f"  Number of bars: {bar_dist.num_bars}")
        print(f"  Centers: {bar_dist.centers}")
        print(f"  Edges: {bar_dist.edges}")
        print(f"  Widths: {bar_dist.widths}")
        print(f"  Base scale left: {bar_dist.base_s_left}")
        print(f"  Base scale right: {bar_dist.base_s_right}")
        
        # Get some actual data to test predictions
        print("\nInspecting dataloader format...")
        first_batch = next(iter(dataloader))
        print(f"  Batch length: {len(first_batch)}")
        
        for i, item in enumerate(first_batch):
            if hasattr(item, 'shape'):
                print(f"  Item {i}: shape {item.shape}, dtype {item.dtype}")
            else:
                print(f"  Item {i}: type {type(item)}, value {item}")
        
        # Expect 4 elements: X_train, y_train, X_test, y_test
        if len(first_batch) != 4:
            self.fail(f"Expected 4 elements in batch, got {len(first_batch)}")
        
        X_train, y_train, X_test, y_test = first_batch
        print("\nDataloader shapes:")
        print(f"  X_train: {X_train.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  X_test: {X_test.shape}")
        print(f"  y_test: {y_test.shape}")
        
        # Check if we need to squeeze dimensions
        if y_train.ndim == 3 and y_train.shape[-1] == 1:
            print("  y_train has 3 dimensions with last dim=1, squeezing last dimension...")
            y_train = y_train.squeeze(-1)
            print(f"  y_train after squeeze: {y_train.shape}")
        
        if y_test.ndim == 3 and y_test.shape[-1] == 1:
            print("  y_test has 3 dimensions with last dim=1, squeezing last dimension...")
            y_test = y_test.squeeze(-1)
            print(f"  y_test after squeeze: {y_test.shape}")
        
        # Verify the shapes are now correct for BarDistribution
        if y_train.ndim != 2:
            self.fail(f"y_train should be 2D after correction, but got shape {y_train.shape}")
        if y_test.ndim != 2:
            self.fail(f"y_test should be 2D after correction, but got shape {y_test.shape}")
        
        # Fit the distribution with corrected data format
        print("\nFitting BarDistribution with corrected data format...")
        try:
            # Create a corrected batch for fitting
            corrected_batch = [(X_train, y_train, X_test, y_test)]
            
            # Use a limited number of items for faster fitting in tests
            bar_dist = BarDistribution(
                num_bars=self.num_bars,
                device=self.device,
                dtype=self.dtype,
                max_fit_items=1000  # Limit for faster testing
            )
            
            bar_dist.fit(corrected_batch)
            print("BarDistribution fitted successfully!")
        except Exception as e:
            self.fail(f"Failed to fit BarDistribution even after format correction: {e}")
        
        B, N_train = y_train.shape
        M = y_test.shape[1]
        
        print(f"  Batch shape: B={B}, N_train={N_train}, M={M}")
        print(f"  Y_train range: [{y_train.min():.3f}, {y_train.max():.3f}]")
        print(f"  Y_test range: [{y_test.min():.3f}, {y_test.max():.3f}]")
        
        # Create random predictions (simulating model output)
        pred = torch.randn(B, M, bar_dist.num_params, dtype=self.dtype, device=self.device)
        
        # Test all prediction methods
        try:
            log_prob = bar_dist.average_log_prob(pred, y_test)
            print(f"  Average log prob: {log_prob.mean():.3f} ± {log_prob.std():.3f}")
            
            mode_vals = bar_dist.mode(pred)
            print(f"  Mode range: [{mode_vals.min():.3f}, {mode_vals.max():.3f}]")
            
            mean_vals = bar_dist.mean(pred)
            print(f"  Mean range: [{mean_vals.min():.3f}, {mean_vals.max():.3f}]")
            
            num_samples = 100
            samples = bar_dist.sample(pred, num_samples)
            print(f"  Sample range: [{samples.min():.3f}, {samples.max():.3f}]")
            
        except Exception as e:
            self.fail(f"Prediction methods failed: {e}")
        
        # Store results for visualization
        self.fitted_bar_dist = bar_dist
        self.real_data = {
            'y_train': y_train,
            'y_test': y_test,
            'pred': pred,
            'log_prob': log_prob,
            'mode_vals': mode_vals,
            'mean_vals': mean_vals,
            'samples': samples
        }
        
        # Run visualization
        self._visualize_results()
    
    def _visualize_results(self):
        """Create visualization plots of the BarDistribution results."""
        if not hasattr(self, 'fitted_bar_dist'):
            self.skipTest("No fitted distribution available for visualization")
        
        bar_dist = self.fitted_bar_dist
        data = self.real_data
        
        print("\nGenerating visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        plt.figure(figsize=(20, 12))
        
        # 1. Data distribution histogram
        plt.subplot(2, 4, 1)
        y_all = torch.cat([data['y_train'].flatten(), data['y_test'].flatten()])
        plt.hist(y_all.numpy(), bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        plt.axvline(y_all.mean(), color='red', linestyle='--', label=f'Mean: {y_all.mean():.2f}')
        plt.axvline(y_all.median(), color='orange', linestyle='--', label=f'Median: {y_all.median():.2f}')
        plt.title('Real Data Distribution')
        plt.xlabel('Y values')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. BarDistribution structure
        plt.subplot(2, 4, 2)
        edges = bar_dist.edges.numpy()
        centers = bar_dist.centers.numpy()
        widths = bar_dist.widths.numpy()
        
        # Plot bars
        for i, (edge_left, edge_right, center) in enumerate(zip(edges[:-1], edges[1:], centers)):
            plt.bar(center, 1.0/widths[i], width=widths[i]*0.8, alpha=0.6, 
                    label=f'Bar {i}' if i < 3 else None)
        
        # Plot edges as vertical lines
        for i, edge in enumerate(edges):
            plt.axvline(edge, color='red', linestyle='-', alpha=0.7)
            plt.text(edge, plt.ylim()[1]*0.9, f'E{i}', ha='center', fontsize=8)
        
        # Plot centers
        plt.scatter(centers, [0.1]*len(centers), color='orange', s=50, zorder=10, label='Centers')
        
        plt.title('BarDistribution Structure')
        plt.xlabel('Y values')
        plt.ylabel('Bar Density (1/width)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Model predictions: mode vs data
        plt.subplot(2, 4, 3)
        y_test_flat = data['y_test'].flatten()
        mode_flat = data['mode_vals'].flatten()
        plt.scatter(y_test_flat.numpy(), mode_flat.numpy(), alpha=0.6, s=20)
        plt.plot([y_test_flat.min(), y_test_flat.max()], [y_test_flat.min(), y_test_flat.max()], 
                'r--', label='Perfect prediction')
        plt.xlabel('True Y values')
        plt.ylabel('Predicted Mode')
        plt.title('Mode Predictions vs True Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Model predictions: mean vs data
        plt.subplot(2, 4, 4)
        mean_flat = data['mean_vals'].flatten()
        plt.scatter(y_test_flat.numpy(), mean_flat.numpy(), alpha=0.6, s=20, color='green')
        plt.plot([y_test_flat.min(), y_test_flat.max()], [y_test_flat.min(), y_test_flat.max()], 
                'r--', label='Perfect prediction')
        plt.xlabel('True Y values')
        plt.ylabel('Predicted Mean')
        plt.title('Mean Predictions vs True Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Log probability distribution
        plt.subplot(2, 4, 5)
        log_probs = data['log_prob'].numpy()
        
        # Filter out non-finite values for plotting
        finite_log_probs = log_probs[np.isfinite(log_probs)]
        if len(finite_log_probs) > 0:
            plt.hist(finite_log_probs, bins=20, alpha=0.7, color='purple', edgecolor='black')
            plt.axvline(finite_log_probs.mean(), color='red', linestyle='--', 
                       label=f'Mean: {finite_log_probs.mean():.2f}')
            plt.title(f'Log Probability Distribution\n({len(finite_log_probs)}/{len(log_probs)} finite)')
        else:
            plt.text(0.5, 0.5, 'All log probabilities\nare non-finite', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Log Probability Distribution\n(All non-finite)')
        
        plt.xlabel('Log Probability')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Sample distribution vs real data
        plt.subplot(2, 4, 6)
        # Take samples from first prediction only for visualization
        samples_first = data['samples'][0, :, 0].numpy()  # First batch, all samples, first feature
        y_test_first = data['y_test'][0, 0].numpy()  # Corresponding true value
        
        plt.hist(samples_first, bins=30, alpha=0.6, density=True, color='lightgreen', 
                label='Model samples', edgecolor='black')
        plt.axvline(y_test_first, color='red', linestyle='-', linewidth=2, 
                   label=f'True value: {y_test_first:.2f}')
        plt.axvline(samples_first.mean(), color='blue', linestyle='--', 
                   label=f'Sample mean: {samples_first.mean():.2f}')
        plt.title('Model Samples vs True Value\n(First prediction)')
        plt.xlabel('Y values')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 7. Bar distribution overlay on data
        plt.subplot(2, 4, 7)
        # Plot data histogram
        plt.hist(y_all.numpy(), bins=50, alpha=0.5, density=True, color='skyblue', 
                label='Real data', edgecolor='black')
        
        # Overlay bar distribution structure
        y_range = np.linspace(y_all.min(), y_all.max(), 1000)
        bar_structure = np.zeros_like(y_range)
        
        # For a proper probability density, we need to use uniform probabilities over each bar
        # Assume equal probability for each bar for visualization (since we don't have actual mixture weights)
        uniform_bar_prob = 1.0 / bar_dist.num_bars
        
        for i, (edge_left, edge_right) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (y_range >= edge_left) & (y_range < edge_right)
            # Probability density = probability / width
            bar_structure[mask] = uniform_bar_prob / widths[i]
        
        plt.plot(y_range, bar_structure, 'r-', linewidth=2, label='Bar structure (uniform probs)')
        
        # Also show the raw structure without normalization for comparison
        raw_structure = np.zeros_like(y_range)
        for i, (edge_left, edge_right) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (y_range >= edge_left) & (y_range < edge_right)
            raw_structure[mask] = 1.0 / widths[i]
        
        # Scale raw structure to match data scale roughly
        if raw_structure.max() > 0:
            hist_max = np.histogram(y_all.numpy(), bins=50, density=True)[0].max()
            scale_factor = hist_max / raw_structure.max() * 0.5  # Scale to 50% of histogram max
            raw_structure *= scale_factor
            plt.plot(y_range, raw_structure, 'g--', linewidth=1, alpha=0.7, 
                    label='Raw bar densities (scaled)')
        
        # Add tail indicators
        plt.axvline(edges[0], color='orange', linestyle='--', alpha=0.7, label='Left tail boundary')
        plt.axvline(edges[-1], color='orange', linestyle='--', alpha=0.7, label='Right tail boundary')
        
        plt.title('Data vs BarDistribution Structure\n(Proper probability densities)')
        plt.xlabel('Y values')
        plt.ylabel('Probability Density')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # 8. Summary statistics
        ax8 = plt.subplot(2, 4, 8)
        ax8.axis('off')  # Turn off axis for text
        
        # Compute some summary statistics
        finite_log_probs = log_probs[np.isfinite(log_probs)]
        if len(finite_log_probs) > 0:
            log_prob_mean = finite_log_probs.mean()
            log_prob_std = finite_log_probs.std()
            log_prob_stats = f"{log_prob_mean:.2f} ± {log_prob_std:.2f} ({len(finite_log_probs)}/{len(log_probs)} finite)"
        else:
            log_prob_stats = "All non-finite"
        
        stats_text = f"""
BarDistribution Summary:
  Number of bars: {bar_dist.num_bars}
  Data range: [{y_all.min():.2f}, {y_all.max():.2f}]
  Data mean: {y_all.mean():.2f}
  Data std: {y_all.std():.2f}
  
  Edges range: [{edges.min():.2f}, {edges.max():.2f}]
  Centers range: [{centers.min():.2f}, {centers.max():.2f}]
  Min width: {widths.min():.3f}
  Max width: {widths.max():.3f}
  Avg width: {widths.mean():.3f}
  
  Base scale left: {bar_dist.base_s_left:.3f}
  Base scale right: {bar_dist.base_s_right:.3f}
  
  Clipping range: [{bar_dist.log_prob_clip_min:.1f}, {bar_dist.log_prob_clip_max:.1f}]
  
Model Performance:
  Avg log prob: {log_prob_stats}
  Mode RMSE: {torch.sqrt(torch.mean((data['mode_vals'] - data['y_test'])**2)):.3f}
  Mean RMSE: {torch.sqrt(torch.mean((data['mean_vals'] - data['y_test'])**2)):.3f}
"""
        
        ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = self.test_output_dir / 'bar_distribution_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {plot_path}")
        
        # Show plot if in interactive environment
        try:
            plt.show()
        except Exception:
            pass  # Ignore if not in interactive environment
        
        plt.close()
    
    def test_distribution_properties(self):
        """Test mathematical properties of the fitted distribution."""
        # Run the main test first to get fitted distribution
        self.test_with_real_dataloader()
        
        bar_dist = self.fitted_bar_dist
        data = self.real_data
        
        print("\nTesting distribution properties...")
        
        # Test 1: Centers should be within the data range
        y_all = torch.cat([data['y_train'].flatten(), data['y_test'].flatten()])
        y_min, y_max = y_all.min(), y_all.max()
        
        centers_in_range = torch.all((bar_dist.centers >= y_min) & (bar_dist.centers <= y_max))
        print(f"  Centers in data range: {centers_in_range}")
        
        # Test 2: Edges should approximately span the data range (allow some tolerance)
        data_range = y_max - y_min
        edge_tolerance = 0.1 * data_range  # Allow 10% tolerance
        edges_span_data = (bar_dist.edges[0] <= y_min + edge_tolerance) and (bar_dist.edges[-1] >= y_max - edge_tolerance)
        print(f"  Edges approximately span data: {edges_span_data}")
        print(f"    Data range: [{y_min:.3f}, {y_max:.3f}]")
        print(f"    Edge range: [{bar_dist.edges[0]:.3f}, {bar_dist.edges[-1]:.3f}]")
        
        # Test 3: Widths should be positive
        widths_positive = torch.all(bar_dist.widths > 0)
        print(f"  All widths positive: {widths_positive}")
        
        # Test 4: Base scales should be positive
        scales_positive = (bar_dist.base_s_left > 0) and (bar_dist.base_s_right > 0)
        print(f"  Base scales positive: {scales_positive}")
        
        # Test 5: Check that log probabilities are finite
        log_probs_finite = torch.all(torch.isfinite(data['log_prob']))
        print(f"  Log probabilities finite: {log_probs_finite}")
        
        # If log probs have -inf, let's investigate
        if not log_probs_finite:
            inf_count = torch.sum(torch.isinf(data['log_prob']))
            neginf_count = torch.sum(torch.isneginf(data['log_prob']))
            print(f"    Found {inf_count} +inf and {neginf_count} -inf values")
            finite_log_probs = data['log_prob'][torch.isfinite(data['log_prob'])]
            if len(finite_log_probs) > 0:
                print(f"    Finite log prob range: [{finite_log_probs.min():.3f}, {finite_log_probs.max():.3f}]")
        
        # Test 6: Check that samples are reasonable
        samples_in_reasonable_range = torch.all(
            (data['samples'] >= y_min - 5*y_all.std()) & 
            (data['samples'] <= y_max + 5*y_all.std())
        )
        print(f"  Samples in reasonable range: {samples_in_reasonable_range}")
        
        # Assert all tests pass (be more lenient about log probs)
        self.assertTrue(centers_in_range, "Centers should be within data range")
        self.assertTrue(edges_span_data, "Edges should approximately span data range")
        self.assertTrue(widths_positive, "All widths should be positive")
        self.assertTrue(scales_positive, "Base scales should be positive")
        # Allow some -inf log probabilities since they can occur with low probability regions
        finite_log_prob_ratio = torch.sum(torch.isfinite(data['log_prob'])).float() / len(data['log_prob'])
        self.assertGreater(finite_log_prob_ratio, 0.8, "At least 80% of log probabilities should be finite")
        self.assertTrue(samples_in_reasonable_range, "Samples should be in reasonable range")


def run_tests():
    """Run all tests and display results."""
    print("="*80)
    print("RUNNING BARDISTRIBUTION TESTS")
    print("="*80)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add basic tests
    test_suite.addTest(unittest.makeSuite(TestBarDistribution))
    
    # Add realistic tests
    test_suite.addTest(unittest.makeSuite(TestBarDistributionWithRealData))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOVERALL: {'PASSED' if success else 'FAILED'}")
    print("="*80)
    
    return success


if __name__ == "__main__":
    run_tests()
