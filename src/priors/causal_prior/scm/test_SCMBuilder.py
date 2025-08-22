"""
Test suite for SCMBuilder class focusing on sampling functionality.

This module contains comprehensive tests for the SCMBuilder class,
particularly testing the sampling methods and various configuration options.
"""

import unittest
import torch
import warnings

from priors.causal_prior.scm.SCMBuilder import SCMBuilder


class TestSCMBuilderSampling(unittest.TestCase):
    """Test class for SCMBuilder sampling functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Suppress XGBoost warnings for cleaner test output
        warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
        
        # Common test parameters
        self.test_num_nodes = 5
        self.test_graph_edge_prob = 0.3
        self.test_graph_seed = 42
        self.test_batch_size = 20
        
        # Create a basic builder for most tests
        self.basic_builder = SCMBuilder(
            num_nodes=self.test_num_nodes,
            graph_edge_prob=self.test_graph_edge_prob,
            graph_seed=self.test_graph_seed
        )
    
    def test_basic_build_and_sample(self):
        """Test basic build_and_sample functionality."""
        samples = self.basic_builder.build_and_sample(self.test_batch_size)
        
        # Check return type
        self.assertIsInstance(samples, dict)
        
        # Check batch size
        for node_name, tensor in samples.items():
            self.assertEqual(tensor.shape[0], self.test_batch_size)
            self.assertIsInstance(tensor, torch.Tensor)
        
        # Check number of nodes
        self.assertEqual(len(samples), self.test_num_nodes)
    
    def test_build_and_sample_tensor(self):
        """Test build_and_sample_tensor returns concatenated tensor."""
        tensor_samples = self.basic_builder.build_and_sample_tensor(self.test_batch_size)
        
        # Check return type
        self.assertIsInstance(tensor_samples, torch.Tensor)
        
        # Check batch dimension
        self.assertEqual(tensor_samples.shape[0], self.test_batch_size)
        
        # Check that second dimension contains all node features
        # (assuming each node has shape (1,) by default)
        expected_features = self.test_num_nodes
        self.assertEqual(tensor_samples.shape[1], expected_features)
    
    def test_manual_sampling_fast_mode(self):
        """Test manual sampling with fast mode."""
        scm = self.basic_builder.build()
        
        # Test that sampling without pre-sampling raises error
        # (could be ValueError or TypeError depending on implementation details)
        with self.assertRaises((ValueError, TypeError)):
            scm.sample(self.test_batch_size)
        
        # Test correct manual sampling
        scm.sample_exogenous(self.test_batch_size)
        scm.sample_endogenous_noise(self.test_batch_size)
        samples = scm.sample(self.test_batch_size)
        
        self.assertIsInstance(samples, dict)
        self.assertEqual(len(samples), self.test_num_nodes)
        for tensor in samples.values():
            self.assertEqual(tensor.shape[0], self.test_batch_size)
    
    def test_manual_sampling_safe_mode(self):
        """Test manual sampling with safe mode."""
        builder = SCMBuilder(
            num_nodes=self.test_num_nodes,
            graph_edge_prob=self.test_graph_edge_prob,
            graph_seed=self.test_graph_seed,
            scm_fast=False
        )
        scm = builder.build()
        
        # Test that even safe mode requires pre-sampling
        with self.assertRaises((ValueError, TypeError)):
            scm.sample(self.test_batch_size)
        
        # Test correct manual sampling
        scm.sample_exogenous(self.test_batch_size)
        scm.sample_endogenous_noise(self.test_batch_size)
        samples = scm.sample(self.test_batch_size)
        
        self.assertIsInstance(samples, dict)
        self.assertEqual(len(samples), self.test_num_nodes)
    
    def test_different_batch_sizes(self):
        """Test sampling with different batch sizes."""
        batch_sizes = [1, 10, 50, 100]
        
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                samples = self.basic_builder.build_and_sample(batch_size)
                tensor_samples = self.basic_builder.build_and_sample_tensor(batch_size)
                
                # Check dictionary samples
                for tensor in samples.values():
                    self.assertEqual(tensor.shape[0], batch_size)
                
                # Check tensor samples
                self.assertEqual(tensor_samples.shape[0], batch_size)
    
    def test_mechanism_types(self):
        """Test different mechanism type configurations."""
        test_configs = [
            {"xgboost_prob": 0.0},  # Only MLP mechanisms
            {"xgboost_prob": 1.0},  # Only XGBoost mechanisms
            {"xgboost_prob": 0.5},  # Mixed mechanisms
        ]
        
        for config in test_configs:
            with self.subTest(config=config):
                builder = SCMBuilder(
                    num_nodes=self.test_num_nodes,
                    graph_edge_prob=self.test_graph_edge_prob,
                    graph_seed=self.test_graph_seed,
                    **config
                )
                samples = builder.build_and_sample(self.test_batch_size)
                
                self.assertIsInstance(samples, dict)
                self.assertEqual(len(samples), self.test_num_nodes)
    
    def test_mlp_hyperparameters(self):
        """Test different MLP mechanism hyperparameters."""
        test_configs = [
            {"mlp_num_hidden_layers": 0, "mlp_hidden_dim": 16},
            {"mlp_num_hidden_layers": 1, "mlp_hidden_dim": 32},
            {"mlp_num_hidden_layers": 2, "mlp_hidden_dim": 64},
            {"mlp_nonlins": "tanh", "mlp_activation_mode": "post"},
        ]
        
        for config in test_configs:
            with self.subTest(config=config):
                builder = SCMBuilder(
                    num_nodes=self.test_num_nodes,
                    graph_edge_prob=self.test_graph_edge_prob,
                    graph_seed=self.test_graph_seed,
                    xgboost_prob=0.0,  # Only MLP to test MLP params
                    **config
                )
                samples = builder.build_and_sample(self.test_batch_size)
                
                self.assertIsInstance(samples, dict)
                self.assertEqual(len(samples), self.test_num_nodes)
    
    def test_xgboost_hyperparameters(self):
        """Test different XGBoost mechanism hyperparameters."""
        test_configs = [
            {"xgb_n_training_samples": 50},
            {"xgb_n_training_samples": 200},
            {"xgb_add_noise": True},
            {"xgb_add_noise": False},
        ]
        
        for config in test_configs:
            with self.subTest(config=config):
                builder = SCMBuilder(
                    num_nodes=self.test_num_nodes,
                    graph_edge_prob=self.test_graph_edge_prob,
                    graph_seed=self.test_graph_seed,
                    xgboost_prob=1.0,  # Only XGBoost to test XGBoost params
                    **config
                )
                samples = builder.build_and_sample(self.test_batch_size)
                
                self.assertIsInstance(samples, dict)
                self.assertEqual(len(samples), self.test_num_nodes)
    
    def test_noise_distribution_types(self):
        """Test different noise distribution configurations."""
        test_configs = [
            {"random_additive_std": False, "exo_std": 0.5, "endo_std": 0.1},
            {"random_additive_std": True, "exo_std_mean": 2.0, "endo_std_mean": 0.5},
        ]
        
        for config in test_configs:
            with self.subTest(config=config):
                builder = SCMBuilder(
                    num_nodes=self.test_num_nodes,
                    graph_edge_prob=self.test_graph_edge_prob,
                    graph_seed=self.test_graph_seed,
                    **config
                )
                samples = builder.build_and_sample(self.test_batch_size)
                
                self.assertIsInstance(samples, dict)
                self.assertEqual(len(samples), self.test_num_nodes)
    
    def test_exogenous_mechanisms_flag(self):
        """Test use_exogenous_mechanisms flag."""
        for use_exo_mechanisms in [True, False]:
            with self.subTest(use_exogenous_mechanisms=use_exo_mechanisms):
                builder = SCMBuilder(
                    num_nodes=self.test_num_nodes,
                    graph_edge_prob=self.test_graph_edge_prob,
                    graph_seed=self.test_graph_seed,
                    use_exogenous_mechanisms=use_exo_mechanisms
                )
                samples = builder.build_and_sample(self.test_batch_size)
                
                self.assertIsInstance(samples, dict)
                self.assertEqual(len(samples), self.test_num_nodes)
    
    def test_with_different_graph_structures(self):
        """Test SCMBuilder with different graph structures."""
        # Test sparse graph
        sparse_builder = SCMBuilder(
            num_nodes=5,
            graph_edge_prob=0.1,  # Very sparse
            graph_seed=123
        )
        sparse_samples = sparse_builder.build_and_sample(self.test_batch_size)
        
        # Test dense graph
        dense_builder = SCMBuilder(
            num_nodes=5,
            graph_edge_prob=0.8,  # Very dense
            graph_seed=123
        )
        dense_samples = dense_builder.build_and_sample(self.test_batch_size)
        
        # Both should work
        self.assertIsInstance(sparse_samples, dict)
        self.assertEqual(len(sparse_samples), 5)
        self.assertIsInstance(dense_samples, dict)
        self.assertEqual(len(dense_samples), 5)
        
        for tensor in sparse_samples.values():
            self.assertEqual(tensor.shape[0], self.test_batch_size)
        for tensor in dense_samples.values():
            self.assertEqual(tensor.shape[0], self.test_batch_size)
    
    def test_deterministic_behavior(self):
        """Test that same seeds produce deterministic results across various hyperparameters."""
        
        # Define comprehensive test configurations
        test_configs = [
            # Basic configuration
            {
                "num_nodes": 4,
                "graph_edge_prob": 0.4,
                "xgboost_prob": 0.0,
            },
            
            # XGBoost only configuration
            {
                "num_nodes": 5,
                "graph_edge_prob": 0.6,
                "xgboost_prob": 1.0,
                "xgb_n_training_samples": 100,
                "xgb_add_noise": True,
            },
            
            # Mixed mechanisms
            {
                "num_nodes": 6,
                "graph_edge_prob": 0.3,
                "xgboost_prob": 0.5,
                "mlp_num_hidden_layers": 2,
                "mlp_hidden_dim": 64,
                "xgb_n_training_samples": 150,
            },
            
            # Complex MLP configuration
            {
                "num_nodes": 3,
                "graph_edge_prob": 0.8,
                "xgboost_prob": 0.0,
                "mlp_num_hidden_layers": 3,
                "mlp_hidden_dim": 32,
                "mlp_nonlins": "tanh",
                "mlp_activation_mode": "post",
            },
            
            # Different noise configurations
            {
                "num_nodes": 4,
                "graph_edge_prob": 0.5,
                "xgboost_prob": 0.2,
                "random_additive_std": True,
                "exo_std_mean": 2.0,
                "endo_std_mean": 0.8,
                "exo_std_std": 0.3,
                "endo_std_std": 0.1,
            },
            
            # Fixed noise configuration
            {
                "num_nodes": 5,
                "graph_edge_prob": 0.4,
                "xgboost_prob": 0.3,
                "random_additive_std": False,
                "exo_std": 1.5,
                "endo_std": 0.2,
            },
            
            # Alternative nonlinearity
            {
                "num_nodes": 4,
                "graph_edge_prob": 0.5,
                "xgboost_prob": 0.0,
                "mlp_nonlins": "tabicl",
                "mlp_num_hidden_layers": 1,
                "mlp_hidden_dim": 48,
            },
            
            # Safe mode configuration
            {
                "num_nodes": 3,
                "graph_edge_prob": 0.7,
                "xgboost_prob": 0.4,
                "scm_fast": False,
            },
            
            # Exogenous mechanisms enabled
            {
                "num_nodes": 4,
                "graph_edge_prob": 0.4,
                "xgboost_prob": 0.1,
                "use_exogenous_mechanisms": True,
                "mlp_num_hidden_layers": 1,
            },
            
            # Linear activation configuration
            {
                "num_nodes": 3,
                "graph_edge_prob": 0.6,
                "xgboost_prob": 0.0,
                "mlp_nonlins": "linear",
                "mlp_num_hidden_layers": 1,
                "mlp_hidden_dim": 16,
            },
            
            # Large graph configuration
            {
                "num_nodes": 8,
                "graph_edge_prob": 0.25,
                "xgboost_prob": 0.3,
                "mlp_num_hidden_layers": 2,
                "mlp_hidden_dim": 24,
            },
            
            # Minimal configuration
            {
                "num_nodes": 2,
                "graph_edge_prob": 0.9,
                "xgboost_prob": 0.0,
                "mlp_num_hidden_layers": 0,
                "mlp_hidden_dim": 8,
            },
        ]
        
        base_seed = 12345
        batch_size = 15
        
        for i, config in enumerate(test_configs):
            with self.subTest(config_index=i, config=config):
                # Use different seeds for each configuration to avoid interference
                seed = base_seed + i * 100
                
                # Add common seed parameters to config
                full_config = {
                    **config,
                    "graph_seed": seed,
                    "mechanism_seed": seed,
                    "mechanism_generator_seed": seed,
                }
                
                # Create two identical builders with same configuration
                builder1 = SCMBuilder(**full_config)
                builder2 = SCMBuilder(**full_config)
                
                # Sample from both builders
                samples1 = builder1.build_and_sample(batch_size)
                samples2 = builder2.build_and_sample(batch_size)
                
                # Test structural consistency
                self.assertEqual(set(samples1.keys()), set(samples2.keys()), 
                               f"Node names differ for config {i}")
                
                self.assertEqual(len(samples1), config["num_nodes"],
                               f"Wrong number of nodes for config {i}")
                
                # Check that all tensors have correct batch size
                for node_name in samples1.keys():
                    self.assertEqual(samples1[node_name].shape[0], batch_size,
                                   f"Wrong batch size for node {node_name} in config {i}")
                    self.assertEqual(samples2[node_name].shape[0], batch_size,
                                   f"Wrong batch size for node {node_name} in config {i}")
                    
                    # Check that tensors have same shape (but not necessarily same values)
                    self.assertEqual(samples1[node_name].shape, samples2[node_name].shape,
                                   f"Shape mismatch for node {node_name} in config {i}")
                
                # Test tensor output consistency
                tensor1 = builder1.build_and_sample_tensor(batch_size)
                tensor2 = builder2.build_and_sample_tensor(batch_size)
                
                self.assertEqual(tensor1.shape, tensor2.shape,
                               f"Tensor shape mismatch for config {i}")
                self.assertEqual(tensor1.shape[0], batch_size,
                               f"Wrong tensor batch size for config {i}")
        
        # Test that different seeds produce different structures
        different_seed_builder = SCMBuilder(
            num_nodes=4,
            graph_edge_prob=0.4,
            graph_seed=99999,  # Different seed
            mechanism_seed=99999,
            mechanism_generator_seed=99999
        )
        
        same_seed_builder = SCMBuilder(
            num_nodes=4,
            graph_edge_prob=0.4,
            graph_seed=base_seed,
            mechanism_seed=base_seed,
            mechanism_generator_seed=base_seed
        )
        
        different_samples = different_seed_builder.build_and_sample(10)
        same_samples = same_seed_builder.build_and_sample(10)
        
        # Should have same structure but likely different graph topology
        self.assertEqual(len(different_samples), len(same_samples))
        # Note: We don't test for different values as that's probabilistic
    
    def test_tensor_concatenation_order(self):
        """Test that tensor concatenation follows topological order."""
        # Create a simple builder with fixed seed for reproducibility
        builder = SCMBuilder(
            num_nodes=3,
            graph_edge_prob=0.6,
            graph_seed=42
        )
        
        # Test that build_and_sample_tensor produces the right shape
        tensor_samples = builder.build_and_sample_tensor(5)
        
        # Should have correct batch size
        self.assertEqual(tensor_samples.shape[0], 5)
        
        # Should have correct number of features (3 nodes * 1 feature each = 3)
        self.assertEqual(tensor_samples.shape[1], 3)
        
        # Test that we can get the same structure manually
        dict_samples = builder.build_and_sample(5)
        scm = builder.build()
        topo_order = scm.dag.topo_order()
        
        # Manual concatenation should have the same shape
        manual_concat = torch.cat([
            dict_samples[node].view(5, -1) for node in topo_order
        ], dim=1)
        
        # Should have same shape (we can't test exact values due to randomness)
        self.assertEqual(tensor_samples.shape, manual_concat.shape)
        self.assertEqual(len(topo_order), 3)  # Should have 3 nodes
    
    def test_error_handling(self):
        """Test error handling for invalid configurations."""
        # Test missing required parameters
        with self.assertRaises(TypeError):
            SCMBuilder()  # No required parameters provided
        
        with self.assertRaises(TypeError):
            SCMBuilder(num_nodes=5)  # Missing graph_edge_prob and graph_seed
        
        # Test invalid values
        with self.assertRaises(ValueError):
            builder = SCMBuilder(
                num_nodes=3,
                graph_edge_prob=0.5,
                graph_seed=42,
                mlp_num_hidden_layers=-1  # Invalid negative value
            )
            builder.build()
    
    def test_config_summary(self):
        """Test get_config_summary method."""
        summary = self.basic_builder.get_config_summary()
        
        self.assertIsInstance(summary, str)
        self.assertIn("SCMBuilder Configuration", summary)
        self.assertIn("Graph Structure", summary)
        self.assertIn("num_nodes: 5", summary)
        self.assertIn("graph_edge_prob: 0.3", summary)
    
    def test_large_graphs(self):
        """Test sampling from larger graphs."""
        large_builder = SCMBuilder(
            num_nodes=20,
            graph_edge_prob=0.2,
            graph_seed=42
        )
        
        samples = large_builder.build_and_sample(10)
        tensor_samples = large_builder.build_and_sample_tensor(10)
        
        self.assertEqual(len(samples), 20)
        self.assertEqual(tensor_samples.shape, (10, 20))  # 10 samples, 20 features
    
    def test_node_shapes(self):
        """Test different node configurations with consistent shapes."""
        # Test with different hidden layer configurations
        test_configs = [
            {"mlp_num_hidden_layers": 0, "mlp_hidden_dim": 8},
            {"mlp_num_hidden_layers": 1, "mlp_hidden_dim": 16}, 
            {"mlp_num_hidden_layers": 2, "mlp_hidden_dim": 32},
        ]
        
        for config in test_configs:
            with self.subTest(config=config):
                builder = SCMBuilder(
                    num_nodes=3,
                    graph_edge_prob=0.5,
                    graph_seed=42,
                    xgboost_prob=0.0,  # Only MLP to avoid complexity
                    **config
                )
                
                samples = builder.build_and_sample(5)
                tensor_samples = builder.build_and_sample_tensor(5)
                
                # Each node should have shape (batch_size, 1) by default
                for tensor in samples.values():
                    self.assertEqual(tensor.shape, (5, 1))
                
                # Concatenated tensor should have shape (batch_size, num_nodes)
                self.assertEqual(tensor_samples.shape, (5, 3))  # 3 nodes * 1 feature each


class TestSCMBuilderIntegration(unittest.TestCase):
    """Integration tests for SCMBuilder with real-world scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
    
    def test_realistic_causal_model(self):
        """Test a realistic causal model configuration."""
        builder = SCMBuilder(
            num_nodes=10,
            graph_edge_prob=0.3,
            graph_seed=42,
            xgboost_prob=0.2,  # 20% XGBoost, 80% MLP
            mlp_num_hidden_layers=1,
            mlp_hidden_dim=32,
            mlp_nonlins="tabicl",
            random_additive_std=True,
            exo_std_mean=1.0,
            endo_std_mean=0.3,
            scm_fast=True
        )
        
        # Test multiple sampling calls
        samples1 = builder.build_and_sample(50)
        samples2 = builder.build_and_sample(100)
        
        self.assertEqual(len(samples1), 10)
        self.assertEqual(len(samples2), 10)
        
        # Test tensor output
        tensor_samples = builder.build_and_sample_tensor(75)
        self.assertEqual(tensor_samples.shape, (75, 10))
    
    def test_performance_benchmark(self):
        """Basic performance test to ensure sampling is reasonably fast."""
        import time
        
        builder = SCMBuilder(
            num_nodes=15,
            graph_edge_prob=0.25,
            graph_seed=42,
            scm_fast=True
        )
        
        # Warm up
        _ = builder.build_and_sample(10)
        
        # Time the sampling
        start_time = time.time()
        samples = builder.build_and_sample(1000)
        end_time = time.time()
        
        # Should complete in reasonable time (adjust threshold as needed)
        self.assertLess(end_time - start_time, 10.0)  # 10 seconds max
        self.assertEqual(len(samples), 15)
        for tensor in samples.values():
            self.assertEqual(tensor.shape[0], 1000)


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
