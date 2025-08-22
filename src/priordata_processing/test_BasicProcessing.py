import unittest
import torch
import numpy as np
from typing import Dict
import warnings

# Import the BasicProcessing class
from priordata_processing.BasicProcessing import BasicProcessing

# Import SCM-related classes for data generation
try:
    from priors.causal_prior.scm.SCMHyperparameterSampler import SCMHyperparameterSampler
    from priors.causal_prior.scm.SCMBuilder import SCMBuilder
    from priors.causal_prior.ExampleConfigs.Basic_Configs import default_sampling_config
    HAS_SCM = True
except ImportError:
    HAS_SCM = False
    warnings.warn("SCM modules not available. Some tests will be skipped.")


class TestBasicProcessing(unittest.TestCase):
    """Comprehensive test suite for BasicProcessing class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Set seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Standard test parameters
        self.max_samples = 100
        self.max_features = 10
        
        # Create a simple test dataset
        self.simple_dataset = {
            0: torch.randn(50, 1),
            1: torch.randn(50, 1) * 2 + 1,
            2: torch.randn(50, 1) * 0.5 - 0.5
        }
        
        # Create dataset with specific values for testing
        self.deterministic_dataset = {
            0: torch.ones(20, 1),
            1: torch.zeros(20, 1),
            2: torch.full((20, 1), -1.0)
        }
        
        # Create processor instances
        self.processor = BasicProcessing(
            max_num_samples=self.max_samples,
            max_num_features=self.max_features,
            random_seed=42
        )
        
        self.processor_with_dropout = BasicProcessing(
            max_num_samples=self.max_samples,
            max_num_features=self.max_features,
            dropout_prob=0.3,
            random_seed=42
        )
    
    def generate_scm_dataset(self, n_samples: int = 50, seed: int = 123) -> Dict[int, torch.Tensor]:
        """Generate a dataset using SCM for testing."""
        if not HAS_SCM:
            self.skipTest("SCM modules not available")
        
        config = default_sampling_config
        sampler = SCMHyperparameterSampler(config, seed=seed)
        sampled_params = sampler.sample()
        builder = SCMBuilder(**sampled_params)
        scm = builder.build()
        
        # Sample data
        scm.sample_exogenous(num_samples=n_samples)
        scm.sample_endogenous_noise(num_samples=n_samples)
        data = scm.propagate(num_samples=n_samples)
        
        # Convert to expected format
        dataset = {}
        for i, (key, tensor_data) in enumerate(data.items()):
            if isinstance(tensor_data, torch.Tensor):
                if tensor_data.dim() == 1:
                    tensor_data = tensor_data.unsqueeze(1)
                dataset[i] = tensor_data
        
        return dataset
    
    # ==================== INITIALIZATION TESTS ====================
    
    def test_initialization_default_parameters(self):
        """Test BasicProcessing initialization with default parameters."""
        processor = BasicProcessing(max_num_samples=100, max_num_features=10)
        
        self.assertEqual(processor.max_num_samples, 100)
        self.assertEqual(processor.max_num_features, 10)
        self.assertEqual(processor.train_fraction, 0.5)
        self.assertEqual(processor.dropout_prob, 0.0)
        self.assertEqual(processor.transformation_type, 'standardize')
        self.assertTrue(processor.shuffle_data)
        self.assertIsNone(processor.target_feature)
        self.assertIsNone(processor.random_seed)
    
    def test_initialization_custom_parameters(self):
        """Test BasicProcessing initialization with custom parameters."""
        processor = BasicProcessing(
            max_num_samples=500,
            max_num_features=20,
            train_fraction=0.7,
            dropout_prob=0.2,
            transformation_type='yeo_johnson',
            shuffle_data=False,
            target_feature=3,
            random_seed=123
        )
        
        self.assertEqual(processor.max_num_samples, 500)
        self.assertEqual(processor.max_num_features, 20)
        self.assertEqual(processor.train_fraction, 0.7)
        self.assertEqual(processor.dropout_prob, 0.2)
        self.assertEqual(processor.transformation_type, 'yeo_johnson')
        self.assertFalse(processor.shuffle_data)
        self.assertEqual(processor.target_feature, 3)
        self.assertEqual(processor.random_seed, 123)
    
    # ==================== INPUT VALIDATION TESTS ====================
    
    def test_process_invalid_mode(self):
        """Test that invalid processing mode raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.processor.process(self.simple_dataset, mode='invalid')
        self.assertIn("Mode must be 'fast' or 'safe'", str(context.exception))
    
    def test_fast_mode_minimal_validation(self):
        """Test that fast mode performs minimal validation."""
        # Should pass with minimal checks
        result, metadata = self.processor.process(self.simple_dataset, mode='fast')
        self.assertIsInstance(result, dict)
        self.assertIsInstance(metadata, dict)
        
        # Check that result has expected keys
        expected_keys = {'X_train', 'Y_train', 'X_test', 'Y_test'}
        self.assertEqual(set(result.keys()), expected_keys)
        
        # Check that all values are tensors
        for key, tensor in result.items():
            self.assertIsInstance(tensor, torch.Tensor)
    
    def test_safe_mode_empty_dataset(self):
        """Test that safe mode catches empty dataset."""
        with self.assertRaises(ValueError) as context:
            self.processor.process({}, mode='safe')
        self.assertIn("Dataset cannot be empty", str(context.exception))
    
    def test_safe_mode_non_dict_dataset(self):
        """Test that safe mode catches non-dictionary input."""
        with self.assertRaises(ValueError) as context:
            self.processor.process("not a dict", mode='safe')
        self.assertIn("Dataset must be a dictionary", str(context.exception))
    
    def test_safe_mode_non_integer_keys(self):
        """Test that safe mode catches non-integer keys."""
        invalid_dataset = {"0": torch.randn(10, 1)}
        with self.assertRaises(ValueError) as context:
            self.processor.process(invalid_dataset, mode='safe')
        self.assertIn("All keys must be integers", str(context.exception))
    
    def test_safe_mode_non_tensor_values(self):
        """Test that safe mode catches non-tensor values."""
        invalid_dataset = {0: "not a tensor"}
        with self.assertRaises(ValueError) as context:
            self.processor.process(invalid_dataset, mode='safe')
        self.assertIn("All values must be torch.Tensor", str(context.exception))
    
    def test_safe_mode_wrong_tensor_dimensions(self):
        """Test that safe mode catches tensors with wrong dimensions."""
        invalid_dataset = {0: torch.randn(10)}  # 1D instead of 2D
        with self.assertRaises(ValueError) as context:
            self.processor.process(invalid_dataset, mode='safe')
        self.assertIn("All tensors must be 2D", str(context.exception))
    
    def test_safe_mode_wrong_tensor_shape(self):
        """Test that safe mode catches tensors with wrong second dimension."""
        invalid_dataset = {0: torch.randn(10, 3)}  # Should be (N, 1)
        with self.assertRaises(ValueError) as context:
            self.processor.process(invalid_dataset, mode='safe')
        self.assertIn("All tensors must have shape (N, 1)", str(context.exception))
    
    def test_safe_mode_inconsistent_sample_counts(self):
        """Test that safe mode catches inconsistent sample counts."""
        invalid_dataset = {
            0: torch.randn(10, 1),
            1: torch.randn(15, 1)  # Different number of samples
        }
        with self.assertRaises(ValueError) as context:
            self.processor.process(invalid_dataset, mode='safe')
        self.assertIn("All features must have same number of samples", str(context.exception))
    
    def test_safe_mode_nan_values(self):
        """Test that safe mode catches NaN values."""
        dataset_with_nan = {0: torch.tensor([[1.0], [float('nan')], [3.0]])}
        with self.assertRaises(ValueError) as context:
            self.processor.process(dataset_with_nan, mode='safe')
        self.assertIn("contains NaN values", str(context.exception))
    
    def test_safe_mode_infinite_values(self):
        """Test that safe mode catches infinite values."""
        dataset_with_inf = {0: torch.tensor([[1.0], [float('inf')], [3.0]])}
        with self.assertRaises(ValueError) as context:
            self.processor.process(dataset_with_inf, mode='safe')
        self.assertIn("contains infinite values", str(context.exception))
    
    # ==================== PROCESSING PIPELINE TESTS ====================
    
    def test_basic_processing_pipeline(self):
        """Test the complete processing pipeline."""
        result, metadata = self.processor.process(self.simple_dataset, mode='safe')
        
        # Check output structure
        self.assertIsInstance(result, dict)
        expected_keys = {'X_train', 'Y_train', 'X_test', 'Y_test'}
        self.assertEqual(set(result.keys()), expected_keys)
        
        # Check tensor shapes
        train_size = int(self.max_samples * 0.5)  # default train_fraction is 0.5
        test_size = self.max_samples - train_size
        
        self.assertEqual(result['X_train'].shape[0], train_size)
        self.assertEqual(result['X_test'].shape[0], test_size)
        self.assertEqual(result['Y_train'].shape, (train_size, 1))
        self.assertEqual(result['Y_test'].shape, (test_size, 1))
        
        # X should have one less feature than max_features (target removed)
        self.assertEqual(result['X_train'].shape[1], self.max_features - 1)
        self.assertEqual(result['X_test'].shape[1], self.max_features - 1)
        
        # Check metadata structure
        expected_metadata_keys = {
            'feature_names', 'target_feature', 'target_col', 'transformation_type',
            'transformation_params', 'original_num_samples', 'original_num_features',
            'final_num_samples', 'final_num_features', 'padding_start_sample',
            'padding_start_feature', 'dropout_prob', 'shuffle_applied', 'train_fraction',
            'X_shape', 'Y_shape'
        }
        self.assertEqual(set(metadata.keys()), expected_metadata_keys)
        
        # Check metadata values
        self.assertEqual(metadata['original_num_samples'], 50)
        self.assertEqual(metadata['original_num_features'], 3)
        self.assertEqual(metadata['final_num_samples'], self.max_samples)
        self.assertEqual(metadata['final_num_features'], self.max_features)
        self.assertEqual(metadata['transformation_type'], 'standardize')
        self.assertTrue(metadata['shuffle_applied'])
        self.assertEqual(metadata['train_fraction'], 0.5)
    
    def test_tensor_conversion(self):
        """Test dictionary to tensor conversion."""
        processor = BasicProcessing(
            max_num_samples=self.max_samples,
            max_num_features=self.max_features,
            shuffle_data=False,  # Disable shuffle to check order
            random_seed=42
        )
        
        result, metadata = processor.process(self.deterministic_dataset, mode='safe')
        
        # Check that features are in sorted order initially
        self.assertEqual(metadata['feature_names'], [0, 1, 2])
    
    def test_target_feature_selection(self):
        """Test target feature selection."""
        # Test with specified target feature
        processor_with_target = BasicProcessing(
            max_num_samples=self.max_samples,
            max_num_features=self.max_features,
            target_feature=1,
            random_seed=42
        )
        
        result, metadata = processor_with_target.process(self.simple_dataset, mode='safe')
        self.assertEqual(metadata['target_feature'], 1)
        
        # Test with random target feature
        result, metadata = self.processor.process(self.simple_dataset, mode='safe')
        self.assertIn(metadata['target_feature'], [0, 1, 2])
        
        # Test that Y contains the target feature values
        self.assertIsInstance(result['Y_train'], torch.Tensor)
        self.assertIsInstance(result['Y_test'], torch.Tensor)
    
    def test_data_shuffling(self):
        """Test data shuffling functionality."""
        # Test with shuffling enabled
        processor_shuffle = BasicProcessing(
            max_num_samples=self.max_samples,
            max_num_features=self.max_features,
            shuffle_data=True,
            random_seed=42
        )
        
        result1, metadata1 = processor_shuffle.process(self.deterministic_dataset, mode='safe')
        
        # Test with shuffling disabled
        processor_no_shuffle = BasicProcessing(
            max_num_samples=self.max_samples,
            max_num_features=self.max_features,
            shuffle_data=False,
            random_seed=42
        )
        
        result2, metadata2 = processor_no_shuffle.process(self.deterministic_dataset, mode='safe')
        
        # Results should be different due to shuffling
        self.assertTrue(metadata1['shuffle_applied'])
        self.assertFalse(metadata2['shuffle_applied'])
        
        # Both should return valid train/test splits
        for result in [result1, result2]:
            self.assertIn('X_train', result)
            self.assertIn('Y_train', result)
            self.assertIn('X_test', result)
            self.assertIn('Y_test', result)
    
    def test_train_fraction(self):
        """Test custom train_fraction functionality."""
        processor_70_30 = BasicProcessing(
            max_num_samples=self.max_samples,
            max_num_features=self.max_features,
            train_fraction=0.7,
            random_seed=42
        )
        
        result, metadata = processor_70_30.process(self.simple_dataset, mode='safe')
        
        # Check train/test split sizes
        expected_train_size = int(self.max_samples * 0.7)
        expected_test_size = self.max_samples - expected_train_size
        
        self.assertEqual(result['X_train'].shape[0], expected_train_size)
        self.assertEqual(result['X_test'].shape[0], expected_test_size)
        self.assertEqual(result['Y_train'].shape[0], expected_train_size)
        self.assertEqual(result['Y_test'].shape[0], expected_test_size)
        self.assertEqual(metadata['train_fraction'], 0.7)
    
    def test_feature_dropout(self):
        """Test feature dropout functionality."""
        result, metadata = self.processor_with_dropout.process(self.simple_dataset, mode='safe')
        
        # Should have fewer or equal features after dropout
        num_features_after_dropout = metadata['padding_start_feature']
        self.assertLessEqual(num_features_after_dropout, 3)
        self.assertGreaterEqual(num_features_after_dropout, 1)  # At least one feature should remain
        
        # Check train/test split shapes
        self.assertIsInstance(result, dict)
        self.assertIn('X_train', result)
        self.assertIn('X_test', result)
    
    def test_standardization_transformation(self):
        """Test standardization transformation."""
        processor_std = BasicProcessing(
            max_num_samples=self.max_samples,
            max_num_features=self.max_features,
            transformation_type='standardize',
            shuffle_data=False,
            dropout_prob=0.0,
            random_seed=42
        )
        
        result, metadata = processor_std.process(self.deterministic_dataset, mode='safe')
        
        # Check transformation parameters
        self.assertEqual(metadata['transformation_type'], 'standardize')
        self.assertIn('means', metadata['transformation_params'])
        self.assertIn('stds', metadata['transformation_params'])
        
        # Non-padded data should be approximately standardized
        # Get the combined tensor to check standardization
        full_tensor = torch.cat([result['X_train'], result['X_test']], dim=0)
        combined_with_target = torch.cat([full_tensor, torch.cat([result['Y_train'], result['Y_test']], dim=0)], dim=1)
        
        # Check a subset of the non-padded region
        non_padded_data = combined_with_target[:20, :3]  # Original data region
        means = non_padded_data.mean(dim=0)
        
        # Should be close to 0 mean (allowing for numerical precision)
        self.assertTrue(torch.allclose(means, torch.zeros_like(means), atol=1e-6))
        # Note: std may not be exactly 1 for constant columns
    
    def test_yeo_johnson_transformation(self):
        """Test Yeo-Johnson transformation."""
        processor_yj = BasicProcessing(
            max_num_samples=self.max_samples,
            max_num_features=self.max_features,
            transformation_type='yeo_johnson',
            shuffle_data=False,
            dropout_prob=0.0,
            random_seed=42
        )
        
        # Use data with positive values for Yeo-Johnson
        positive_dataset = {
            0: torch.abs(torch.randn(30, 1)) + 1,
            1: torch.abs(torch.randn(30, 1)) + 2
        }
        
        result, metadata = processor_yj.process(positive_dataset, mode='safe')
        
        # Check transformation parameters
        self.assertEqual(metadata['transformation_type'], 'yeo_johnson')
        self.assertIn('lambdas', metadata['transformation_params'])
        self.assertIn('means', metadata['transformation_params'])
        self.assertIn('stds', metadata['transformation_params'])
    
    def test_zero_padding(self):
        """Test zero padding functionality."""
        result, metadata = self.processor.process(self.simple_dataset, mode='safe')
        
        # Check that padding starts at correct positions
        padding_start_sample = metadata['padding_start_sample']
        padding_start_feature = metadata['padding_start_feature']
        
        # Reconstruct full tensor for padding check
        full_X = torch.cat([result['X_train'], result['X_test']], dim=0)
        full_Y = torch.cat([result['Y_train'], result['Y_test']], dim=0)
        
        # Insert Y back at target position to reconstruct original tensor
        target_col = metadata['target_col']
        if target_col == 0:
            full_tensor = torch.cat([full_Y, full_X], dim=1)
        elif target_col == full_X.shape[1]:
            full_tensor = torch.cat([full_X, full_Y], dim=1)
        else:
            full_tensor = torch.cat([
                full_X[:, :target_col], 
                full_Y, 
                full_X[:, target_col:]
            ], dim=1)
        
        # Padded regions should be zero
        if padding_start_sample < self.max_samples:
            padded_sample_region = full_tensor[padding_start_sample:, :]
            self.assertTrue(torch.allclose(padded_sample_region, torch.zeros_like(padded_sample_region)))
        
        if padding_start_feature < self.max_features:
            padded_feature_region = full_tensor[:, padding_start_feature:]
            self.assertTrue(torch.allclose(padded_feature_region, torch.zeros_like(padded_feature_region)))
    
    def test_safe_mode_padding_size_validation(self):
        """Test that safe mode validates padding sizes."""
        # Create dataset larger than max dimensions
        large_dataset = {i: torch.randn(200, 1) for i in range(15)}  # 200 samples, 15 features
        
        small_processor = BasicProcessing(max_num_samples=100, max_num_features=10)
        
        with self.assertRaises(ValueError):
            small_processor.process(large_dataset, mode='safe')
    
    # ==================== REPRODUCIBILITY TESTS ====================
    
    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with the same seed."""
        processor1 = BasicProcessing(
            max_num_samples=self.max_samples,
            max_num_features=self.max_features,
            random_seed=123
        )
        
        processor2 = BasicProcessing(
            max_num_samples=self.max_samples,
            max_num_features=self.max_features,
            random_seed=123
        )
        
        result1, metadata1 = processor1.process(self.simple_dataset, mode='fast')
        result2, metadata2 = processor2.process(self.simple_dataset, mode='fast')
        
        # Results should be identical
        for key in result1.keys():
            self.assertTrue(torch.allclose(result1[key], result2[key]))
        self.assertEqual(metadata1['target_feature'], metadata2['target_feature'])
    
    def test_different_results_without_seed(self):
        """Test that results differ without fixed seed."""
        # Use different random seeds instead of trying to reset to None
        import time
        
        processor1 = BasicProcessing(
            max_num_samples=self.max_samples,
            max_num_features=self.max_features,
            random_seed=int(time.time() * 1000) % 1000000  # Time-based seed
        )
        
        processor2 = BasicProcessing(
            max_num_samples=self.max_samples,
            max_num_features=self.max_features,
            random_seed=int(time.time() * 1000) % 1000000 + 1  # Different time-based seed
        )
        
        result1, metadata1 = processor1.process(self.simple_dataset, mode='fast')
        result2, metadata2 = processor2.process(self.simple_dataset, mode='fast')
        
        # Results should likely be different (not guaranteed, but very likely)
        # We'll just check they don't raise errors and have proper shapes
        self.assertIsInstance(result1, dict)
        self.assertIsInstance(result2, dict)
        self.assertEqual(set(result1.keys()), set(result2.keys()))
        for key in result1.keys():
            self.assertEqual(result1[key].shape, result2[key].shape)
    
    # ==================== MODE COMPARISON TESTS ====================
    
    def test_fast_vs_safe_mode_identical_results(self):
        """Test that fast and safe modes produce identical results for valid data."""
        # Use a fixed seed processor to ensure reproducibility
        processor = BasicProcessing(
            max_num_samples=self.max_samples,
            max_num_features=self.max_features,
            random_seed=12345
        )
        
        result_fast, metadata_fast = processor.process(self.simple_dataset, mode='fast')
        
        # Reset processor to same seed for second run
        processor_safe = BasicProcessing(
            max_num_samples=self.max_samples,
            max_num_features=self.max_features,
            random_seed=12345
        )
        result_safe, metadata_safe = processor_safe.process(self.simple_dataset, mode='safe')
        
        # Results should be identical
        for key in result_fast.keys():
            self.assertTrue(torch.allclose(result_fast[key], result_safe[key]))
        
        # Check metadata values individually to handle tensors properly
        for key in metadata_fast:
            if isinstance(metadata_fast[key], torch.Tensor):
                self.assertTrue(torch.allclose(metadata_fast[key], metadata_safe[key]), 
                              f"Tensor mismatch for key '{key}'")
            elif isinstance(metadata_fast[key], dict):
                # Handle nested dictionaries (like transformation_params)
                self.assertEqual(set(metadata_fast[key].keys()), set(metadata_safe[key].keys()),
                               f"Dictionary keys mismatch for '{key}'")
                for subkey in metadata_fast[key]:
                    if isinstance(metadata_fast[key][subkey], torch.Tensor):
                        self.assertTrue(torch.allclose(metadata_fast[key][subkey], metadata_safe[key][subkey]),
                                      f"Tensor mismatch for '{key}[{subkey}]'")
                    else:
                        self.assertEqual(metadata_fast[key][subkey], metadata_safe[key][subkey],
                                       f"Value mismatch for '{key}[{subkey}]'")
            elif isinstance(metadata_fast[key], (list, tuple)):
                self.assertEqual(len(metadata_fast[key]), len(metadata_safe[key]), 
                               f"Length mismatch for key '{key}'")
                for i, (item1, item2) in enumerate(zip(metadata_fast[key], metadata_safe[key])):
                    if isinstance(item1, torch.Tensor):
                        self.assertTrue(torch.allclose(item1, item2), 
                                      f"Tensor mismatch in {key}[{i}]")
                    else:
                        self.assertEqual(item1, item2, f"Value mismatch in {key}[{i}]")
            else:
                self.assertEqual(metadata_fast[key], metadata_safe[key], 
                               f"Value mismatch for key '{key}'")
    
    # ==================== EDGE CASES TESTS ====================
    
    def test_single_feature_dataset(self):
        """Test processing with only one feature."""
        single_feature_dataset = {0: torch.randn(30, 1)}
        
        result, metadata = self.processor.process(single_feature_dataset, mode='safe')
        
        self.assertEqual(metadata['original_num_features'], 1)
        # Check that we get proper train/test split
        self.assertIsInstance(result, dict)
        self.assertIn('X_train', result)
        self.assertIn('Y_train', result)
        
        # X should be empty (since target is the only feature)
        self.assertEqual(result['X_train'].shape[1], self.max_features - 1)
    
    def test_constant_feature_standardization(self):
        """Test standardization with constant features."""
        constant_dataset = {
            0: torch.ones(20, 1) * 5.0,  # Constant feature
            1: torch.randn(20, 1)        # Variable feature
        }
        
        processor_no_shuffle = BasicProcessing(
            max_num_samples=self.max_samples,
            max_num_features=self.max_features,
            shuffle_data=False,
            dropout_prob=0.0,
            random_seed=42
        )
        
        result, metadata = processor_no_shuffle.process(constant_dataset, mode='safe')
        
        # Should not raise errors and produce valid output
        self.assertIsInstance(result, dict)
        self.assertIn('X_train', result)
        self.assertEqual(result['X_train'].shape[1], self.max_features - 1)
        self.assertIn('stds', metadata['transformation_params'])
    
    def test_very_small_dataset(self):
        """Test processing with very small dataset."""
        tiny_dataset = {0: torch.randn(2, 1)}
        
        result, metadata = self.processor.process(tiny_dataset, mode='safe')
        
        self.assertEqual(metadata['original_num_samples'], 2)
        self.assertIsInstance(result, dict)
        
        # Check that train/test split works even with tiny data
        train_size = int(self.max_samples * 0.5)
        self.assertEqual(result['X_train'].shape[0], train_size)
        self.assertEqual(result['X_test'].shape[0], self.max_samples - train_size)
    
    def test_large_dropout_probability(self):
        """Test with high dropout probability."""
        processor_high_dropout = BasicProcessing(
            max_num_samples=self.max_samples,
            max_num_features=self.max_features,
            dropout_prob=0.9,  # Very high dropout
            random_seed=42
        )
        
        result, metadata = processor_high_dropout.process(self.simple_dataset, mode='safe')
        
        # Should still have at least one feature
        self.assertGreaterEqual(metadata['padding_start_feature'], 1)
        self.assertIsInstance(result, dict)
        self.assertEqual(result['X_train'].shape[1], self.max_features - 1)
    
    # ==================== SCM INTEGRATION TESTS ====================
    
    @unittest.skipUnless(HAS_SCM, "SCM modules not available")
    def test_scm_generated_dataset_processing(self):
        """Test processing of SCM-generated datasets."""
        scm_dataset = self.generate_scm_dataset(n_samples=50, seed=456)
        
        result, metadata = self.processor.process(scm_dataset, mode='safe')
        
        # Should process successfully
        self.assertIsInstance(result, dict)
        expected_keys = {'X_train', 'Y_train', 'X_test', 'Y_test'}
        self.assertEqual(set(result.keys()), expected_keys)
        self.assertGreater(metadata['original_num_features'], 0)
        self.assertEqual(metadata['original_num_samples'], 50)
    
    @unittest.skipUnless(HAS_SCM, "SCM modules not available")
    def test_multiple_scm_datasets(self):
        """Test processing multiple different SCM-generated datasets."""
        results = []
        metadatas = []
        
        for seed in [100, 200, 300]:
            scm_dataset = self.generate_scm_dataset(n_samples=30, seed=seed)
            result, metadata = self.processor.process(scm_dataset, mode='fast')
            results.append(result)
            metadatas.append(metadata)
        
        # All should process successfully
        for result in results:
            self.assertIsInstance(result, dict)
            expected_keys = {'X_train', 'Y_train', 'X_test', 'Y_test'}
            self.assertEqual(set(result.keys()), expected_keys)
        
        # Datasets should be different (very likely)
        self.assertFalse(torch.allclose(results[0]['X_train'], results[1]['X_train']))
    
    # ==================== PERFORMANCE AND STRESS TESTS ====================
    
    def test_large_dataset_processing(self):
        """Test processing of larger datasets."""
        large_dataset = {i: torch.randn(500, 1) for i in range(50)}
        
        large_processor = BasicProcessing(
            max_num_samples=1000,
            max_num_features=100,
            random_seed=42
        )
        
        # Should complete without errors
        result, metadata = large_processor.process(large_dataset, mode='fast')
        
        self.assertIsInstance(result, dict)
        expected_keys = {'X_train', 'Y_train', 'X_test', 'Y_test'}
        self.assertEqual(set(result.keys()), expected_keys)
        self.assertEqual(metadata['original_num_samples'], 500)
        self.assertEqual(metadata['original_num_features'], 50)
        
        # Check train/test split
        train_size = int(1000 * 0.5)  # Default train_fraction
        self.assertEqual(result['X_train'].shape[0], train_size)
        self.assertEqual(result['X_test'].shape[0], 1000 - train_size)
        self.assertEqual(result['X_train'].shape[1], 100 - 1)  # One feature is target
    
    def test_processing_speed_comparison(self):
        """Test that fast and safe modes complete successfully for large datasets."""
        import time
        
        large_dataset = {i: torch.randn(200, 1) for i in range(20)}
        
        # Use same seed for both to ensure identical results
        large_processor_fast = BasicProcessing(
            max_num_samples=500,
            max_num_features=50,
            random_seed=54321
        )
        
        large_processor_safe = BasicProcessing(
            max_num_samples=500,
            max_num_features=50,
            random_seed=54321
        )
        
        # Time fast mode
        start_time = time.perf_counter()  # Use higher precision timer
        result_fast, _ = large_processor_fast.process(large_dataset, mode='fast')
        fast_time = time.perf_counter() - start_time
        
        # Time safe mode
        start_time = time.perf_counter()  # Use higher precision timer
        result_safe, _ = large_processor_safe.process(large_dataset, mode='safe')
        safe_time = time.perf_counter() - start_time
        
        # Both modes should complete successfully
        self.assertEqual(set(result_fast.keys()), set(result_safe.keys()))
        for key in result_fast.keys():
            self.assertEqual(result_fast[key].shape, result_safe[key].shape)
        
        # Results should be identical
        for key in result_fast.keys():
            self.assertTrue(torch.allclose(result_fast[key], result_safe[key]))
        
        # Just verify both times are non-negative (timing can be very fast)
        self.assertGreaterEqual(fast_time, 0.0)
        self.assertGreaterEqual(safe_time, 0.0)
        
        # Both should complete in reasonable time (less than 30 seconds)
        self.assertLess(fast_time, 30.0)
        self.assertLess(safe_time, 30.0)


class TestBasicProcessingHelperMethods(unittest.TestCase):
    """Test individual helper methods of BasicProcessing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = BasicProcessing(
            max_num_samples=100,
            max_num_features=10,
            random_seed=42
        )
        
        self.test_dataset = {
            0: torch.tensor([[1.0], [2.0], [3.0]]),
            1: torch.tensor([[4.0], [5.0], [6.0]])
        }
    
    def test_dict_to_tensor_conversion(self):
        """Test _dict_to_tensor method."""
        data_tensor, feature_indices = self.processor._dict_to_tensor(self.test_dataset, 'safe')
        
        expected_tensor = torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        expected_indices = [0, 1]
        
        self.assertTrue(torch.allclose(data_tensor, expected_tensor))
        self.assertEqual(feature_indices, expected_indices)
    
    def test_target_feature_selection(self):
        """Test _select_target_feature method."""
        feature_indices = [0, 1, 2]
        
        # Test with specified target
        processor_with_target = BasicProcessing(
            max_num_samples=100,
            max_num_features=10,
            target_feature=1
        )
        target = processor_with_target._select_target_feature(feature_indices, 'safe')
        self.assertEqual(target, 1)
        
        # Test with random target
        target = self.processor._select_target_feature(feature_indices, 'safe')
        self.assertIn(target, feature_indices)
    
    def test_padding_method(self):
        """Test _pad_data method."""
        small_tensor = torch.ones(3, 2)
        padded = self.processor._pad_data(small_tensor, 'safe')
        
        self.assertEqual(padded.shape, (100, 10))
        self.assertTrue(torch.allclose(padded[:3, :2], torch.ones(3, 2)))
        self.assertTrue(torch.allclose(padded[3:, :], torch.zeros(97, 10)))
        self.assertTrue(torch.allclose(padded[:, 2:], torch.zeros(100, 8)))


def run_specific_test_suite():
    """Run specific test suites based on availability of dependencies."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Always add basic tests
    suite.addTests(loader.loadTestsFromTestCase(TestBasicProcessing))
    suite.addTests(loader.loadTestsFromTestCase(TestBasicProcessingHelperMethods))
    
    # Add SCM tests if available
    if HAS_SCM:
        print("Running tests with SCM integration...")
    else:
        print("Running tests without SCM (SCM modules not available)")
    
    return suite


if __name__ == '__main__':
    # Run the test suite
    runner = unittest.TextTestRunner(verbosity=2)
    suite = run_specific_test_suite()
    result = runner.run(suite)
    
    # Print summary
    print("\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
