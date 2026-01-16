#!/usr/bin/env python3
"""
Visualize Attention Maps from Trained Graph-Conditioned Models

This script loads a trained PartialGraphConditionedInterventionalPFN model and visualizes
its attention patterns for different graph configurations and edge states.

Usage:
    python visualize_trained_model_attention.py

The model paths are hardcoded in the main() function - modify them to visualize different models.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add the necessary directories to Python path
script_dir = Path(__file__).parent
causal_prior_dir = script_dir  # This is the CausalPriorFitting directory
src_dir = causal_prior_dir / 'src'
src_models_dir = causal_prior_dir / 'src' / 'models'

sys.path.insert(0, str(causal_prior_dir))
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(src_models_dir))

from models.GraphConditionedInterventionalPFN_sklearn import GraphConditionedInterventionalPFNSklearn
from models.InterventionalPFN_sklearn_batched import InterventionalPFNSklearn

# Import dataset class for data generation
from priordata_processing.Datasets.InterventionalDataset import InterventionalDataset


def load_trained_model(config_path: str, checkpoint_path: str, verbose: bool = True):
    """
    Load a trained model from config and checkpoint files.
    
    Args:
        config_path: Path to the model config YAML file
        checkpoint_path: Path to the model checkpoint (.pt file)
        verbose: Whether to print loading messages
        
    Returns:
        Loaded model (either GraphConditionedInterventionalPFNSklearn or InterventionalPFNSklearn)
    """
    if verbose:
        print(f"Loading config from: {config_path}")
        print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load config to check model type
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Helper to get value from wandb-style or flat config
    def _get_cfg_value(cfg_dict, key, default=None):
        if isinstance(cfg_dict.get(key), dict) and 'value' in cfg_dict[key]:
            return cfg_dict[key]['value']
        return cfg_dict.get(key, default)
    
    # Check if graph conditioning is used
    use_graph_conditioning = _get_cfg_value(config.get('model_config', {}), 'use_graph_conditioning', False)
    
    if verbose:
        model_type = "Graph-Conditioned" if use_graph_conditioning else "Standard"
        print(f"Model type: {model_type} InterventionalPFN")
    
    # Create and load the appropriate model
    if use_graph_conditioning:
        model = GraphConditionedInterventionalPFNSklearn(
            config_path=config_path,
            checkpoint_path=checkpoint_path
        )
    else:
        model = InterventionalPFNSklearn(
            config_path=config_path,
            checkpoint_path=checkpoint_path
        )
    
    model.load()
    
    if verbose:
        print("✓ Model loaded successfully")
    
    return model, config


def extract_attention_weights(model, X_obs, T_obs, Y_obs, X_intv, T_intv, adj_matrix=None):
    """
    Extract attention weights from the model's feature attention layers, properly applying
    soft attention bias logic for graph-conditioned models.
    
    Args:
        model: Loaded model (sklearn wrapper)
        X_obs, T_obs, Y_obs, X_intv, T_intv: Input data
        adj_matrix: Adjacency matrix (for graph-conditioned models)
        
    Returns:
        List of attention weight tensors, one per layer (properly bias-corrected)
    """
    # Get the underlying PyTorch model
    pytorch_model = model.model
    
    print(f"  Model has {len(pytorch_model.blocks)} blocks")
    
    # Detect if model uses soft attention bias
    use_soft_bias = any(hasattr(block, 'bias_edge') for block in pytorch_model.blocks)
    print(f"  Soft attention bias detected: {use_soft_bias}")
    
    attention_weights = []
    
    # Store original forward methods and patch them
    original_forwards = []
    
    for i, block in enumerate(pytorch_model.blocks):
        if hasattr(block, 'feat_attn'):
            # Store original
            original_forwards.append(block.feat_attn.forward)
            
            # Get bias parameters for this block if they exist
            bias_edge = None
            bias_no_edge = None
            if use_soft_bias and hasattr(block, 'bias_edge'):
                bias_edge = block.bias_edge
                bias_no_edge = block.bias_no_edge
                print(f"  Block {i} bias parameters: edge={bias_edge[:2]}, no_edge={bias_no_edge[:2]}")
            
            # Create a patched version that captures attention weights WITH bias application
            def make_patched_forward(original_forward, layer_idx, bias_edge_param, bias_no_edge_param):
                def patched_forward(query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None, average_attn_weights=True):
                    
                    # If we have soft bias and adjacency matrix, we need to apply bias to the attention mask
                    modified_attn_mask = attn_mask
                    
                    if use_soft_bias and bias_edge_param is not None and adj_matrix is not None:
                        # Apply the soft attention bias logic (following the model class implementation)
                        
                        # Get dimensions from the model architecture  
                        batch_size = query.shape[0]  # This should be B*S from the model
                        seq_len = query.shape[1]     # This should be F (features + T + Y)
                        num_heads = block.feat_attn.num_heads
                        
                        # Extract the adjacency matrix from the batch (remove batch dimension)
                        # adj_matrix has shape [batch, F, F] -> extract [F, F] 
                        if adj_matrix.dim() == 3 and adj_matrix.shape[0] == 1:
                            adj_for_bias = adj_matrix[0]  # Extract [F, F] from [1, F, F]
                        else:
                            adj_for_bias = adj_matrix
                            
                        # The adjacency matrix should match the feature dimensions
                        if adj_for_bias.shape[0] == seq_len and adj_for_bias.shape[1] == seq_len:
                            # CRITICAL: Follow the model's logic - transpose the adjacency matrix!
                            # The model transposes adjacency so that j can attend to i when i→j (causes)
                            # This is done in PartialGraphConditionedInterventionalPFN._prepare_attention_mask()
                            adj_for_bias = adj_for_bias.transpose(-2, -1)
                            print(f"  Layer {layer_idx}: Transposed adjacency matrix for attention bias")
                            
                            # Create the bias mask following the model class logic
                            device = query.device
                            
                            # adj_matrix values: -1 (no edge), 0 (unknown), 1 (edge)
                            edge_mask = (adj_for_bias == 1.0).float()      # Where edges exist  
                            no_edge_mask = (adj_for_bias == -1.0).float()  # Where no edges
                            
                            # Expand to batch dimension: (F, F) -> (B*S, F, F)  
                            edge_mask = edge_mask.unsqueeze(0).expand(batch_size, -1, -1)
                            no_edge_mask = no_edge_mask.unsqueeze(0).expand(batch_size, -1, -1)
                            
                            # Expand for num_heads: (B*S, F, F) -> (B*S*num_heads, F, F)
                            edge_mask = edge_mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
                            edge_mask = edge_mask.reshape(batch_size * num_heads, seq_len, seq_len)
                            
                            no_edge_mask = no_edge_mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
                            no_edge_mask = no_edge_mask.reshape(batch_size * num_heads, seq_len, seq_len)
                            
                            # Apply per-head biases (matching model class logic)
                            head_bias_edge = bias_edge_param.view(1, num_heads, 1, 1)
                            head_bias_edge = head_bias_edge.expand(batch_size, -1, seq_len, seq_len)
                            head_bias_edge = head_bias_edge.reshape(batch_size * num_heads, seq_len, seq_len)
                            
                            head_bias_no_edge = bias_no_edge_param.view(1, num_heads, 1, 1)
                            head_bias_no_edge = head_bias_no_edge.expand(batch_size, -1, seq_len, seq_len)
                            head_bias_no_edge = head_bias_no_edge.reshape(batch_size * num_heads, seq_len, seq_len)
                            
                            # Compute the bias mask: add for edges, subtract for no-edges
                            # This follows the exact logic from the model class:
                            # float_mask = edge_mask * head_bias_edge - no_edge_mask * head_bias_no_edge
                            bias_mask = edge_mask * head_bias_edge - no_edge_mask * head_bias_no_edge
                            
                            # The bias mask gets ADDED to attention scores (before softmax)
                            # PyTorch MultiheadAttention expects attn_mask to be additive
                            modified_attn_mask = bias_mask
                            
                            print(f"  Layer {layer_idx}: Applied soft bias mask, shape {bias_mask.shape}")
                            print(f"  Layer {layer_idx}: Adjacency [{adj_for_bias.shape}] -> bias mask [{bias_mask.shape}]")
                        
                        else:
                            print(f"  Layer {layer_idx}: Adjacency shape mismatch: {adj_for_bias.shape} vs seq_len {seq_len}")
                            print(f"  Layer {layer_idx}: Using fallback (no bias applied)")
                    
                    # Call the original forward with the (potentially modified) attention mask
                    result_with_weights = original_forward(
                        query, key, value,
                        key_padding_mask=key_padding_mask,
                        need_weights=True,  # Always request weights for our visualization
                        attn_mask=modified_attn_mask,  # Use our bias-corrected mask
                        average_attn_weights=False  # Don't average to see per-head
                    )
                    
                    # Extract and save the attention weights (these now include bias effects)
                    if isinstance(result_with_weights, tuple) and len(result_with_weights) >= 2:
                        output, attn_weights_tensor = result_with_weights[0], result_with_weights[1]
                        if attn_weights_tensor is not None:
                            print(f"  Layer {layer_idx}: Captured bias-corrected attention, shape {attn_weights_tensor.shape}")
                            attention_weights.append(attn_weights_tensor.detach().cpu())
                        else:
                            print(f"  Layer {layer_idx}: Attention weights are None")
                    else:
                        print(f"  Layer {layer_idx}: Unexpected return format: {type(result_with_weights)}")
                        output = result_with_weights if not isinstance(result_with_weights, tuple) else result_with_weights[0]
                    
                    # Return what the original caller expects (with the same parameters they requested)
                    return original_forward(
                        query, key, value,
                        key_padding_mask=key_padding_mask,
                        need_weights=need_weights,
                        attn_mask=modified_attn_mask,  # Use our bias-corrected mask consistently
                        average_attn_weights=average_attn_weights
                    )
                
                return patched_forward
            
            # Apply the patch
            block.feat_attn.forward = make_patched_forward(original_forwards[i], i, bias_edge, bias_no_edge)
            print(f"  Patched block {i} feat_attn")
    
    # Prepare inputs
    device = next(pytorch_model.parameters()).device
    print(f"  Model device: {device}")
    
    # Convert inputs to tensors and move to device
    if not torch.is_tensor(X_obs):
        X_obs = torch.tensor(X_obs, dtype=torch.float32)
    if not torch.is_tensor(T_obs):
        T_obs = torch.tensor(T_obs, dtype=torch.float32)
    if not torch.is_tensor(Y_obs):
        Y_obs = torch.tensor(Y_obs, dtype=torch.float32)
    if not torch.is_tensor(X_intv):
        X_intv = torch.tensor(X_intv, dtype=torch.float32)
    if not torch.is_tensor(T_intv):
        T_intv = torch.tensor(T_intv, dtype=torch.float32)
    
    X_obs = X_obs.to(device)
    T_obs = T_obs.to(device)
    Y_obs = Y_obs.to(device)
    X_intv = X_intv.to(device)
    T_intv = T_intv.to(device)
    
    if adj_matrix is not None:
        if not torch.is_tensor(adj_matrix):
            adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
        adj_matrix = adj_matrix.to(device)
        print(f"  Adjacency matrix shape: {adj_matrix.shape}")
    
    print(f"  Input shapes: X_obs={X_obs.shape}, T_obs={T_obs.shape}, Y_obs={Y_obs.shape}")
    print(f"  X_intv={X_intv.shape}, T_intv={T_intv.shape}")
    
    # Forward pass
    pytorch_model.eval()
    try:
        with torch.no_grad():
            print("  Starting forward pass...")
            if hasattr(pytorch_model, 'forward') and adj_matrix is not None:
                # Graph-conditioned model
                output = pytorch_model(X_obs, T_obs, Y_obs, X_intv, T_intv, adj_matrix)
                print(f"  Forward pass completed, output type: {type(output)}")
            else:
                # Standard model
                output = pytorch_model(X_obs, T_obs, Y_obs, X_intv, T_intv)
                print(f"  Forward pass completed, output type: {type(output)}")
    except Exception as e:
        print(f"  Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original forward methods
        for i, block in enumerate(pytorch_model.blocks):
            if hasattr(block, 'feat_attn') and i < len(original_forwards):
                block.feat_attn.forward = original_forwards[i]
        print(f"  Restored {len(original_forwards)} forward methods")
    
    print(f"  Total attention weights captured: {len(attention_weights)}")
    return attention_weights


def create_dataset_from_config(config: dict, num_samples: int = 20, seed: int = 42):
    """
    Create an InterventionalDataset using the model's training configuration.
    
    Args:
        config: Model configuration dictionary loaded from YAML
        num_samples: Number of samples to generate in the dataset
        seed: Random seed for reproducible data generation
        
    Returns:
        InterventionalDataset instance
    """
    # Extract the three main configs needed for the dataset
    scm_config = config.get('scm_config', {})
    dataset_config = config.get('dataset_config', {})
    preprocessing_config = config.get('preprocessing_config', {})
    
    # Override dataset size to generate fewer samples for visualization
    dataset_config_viz = dataset_config.copy()
    
    # Check if model uses ancestor matrices and configure appropriately
    model_config = config.get('model_config', {})
    use_ancestor_matrix = False
    if isinstance(model_config, dict):
        # Check if use_ancestor_matrix is set in model config
        use_ancestor_matrix = model_config.get('use_ancestor_matrix', {}).get('value', False)
    
    if use_ancestor_matrix:
        dataset_config_viz['return_ancestor_matrix'] = {'value': True}
        dataset_config_viz['return_adjacency_matrix'] = {'value': False}
        print("Using ancestor matrix format for dataset")
    else:
        dataset_config_viz['return_adjacency_matrix'] = {'value': True}
        dataset_config_viz['return_ancestor_matrix'] = {'value': False}
        print("Using adjacency matrix format for dataset")
    
    # Create the dataset with the same configuration used during training
    dataset = InterventionalDataset(
        scm_config=scm_config,
        preprocessing_config=preprocessing_config,
        dataset_config=dataset_config_viz,
        seed=seed,
        return_scm=True  # This returns (data, adjacency/ancestor, scm, processor, intervention_node)
    )
    
    print(f"Dataset created with {len(dataset)} samples using model's training config")
    print(f"SCM nodes: {scm_config.get('num_nodes', {}).get('value', 'unknown')}")
    print(f"Graph edge probability: {scm_config.get('graph_edge_prob', {}).get('value', 'unknown')}")
    
    return dataset


def sample_data_from_dataset(dataset, sample_idx: int = 0):
    """
    Sample data from the dataset for visualization.
    
    Args:
        dataset: InterventionalDataset instance
        sample_idx: Index of sample to use
        
    Returns:
        Tuple containing (X_obs, T_obs, Y_obs, X_intv, T_intv, adjacency_matrix, scm, processor)
    """
    # Get a sample from the dataset - following notebook pattern
    X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, adj, scm, processor, intervention_node = dataset[sample_idx]
    
    print(f"Sample {sample_idx} - Original data shapes:")
    print(f"  X_obs: {X_obs.shape}, T_obs: {T_obs.shape}, Y_obs: {Y_obs.shape}")
    print(f"  X_intv: {X_intv.shape}, T_intv: {T_intv.shape}, Y_intv: {Y_intv.shape}")
    print(f"  Adjacency matrix: {adj.shape}")
    
    # Subsample the data to make it more manageable for visualization
    max_obs_samples = 50
    max_intv_samples = 25
    
    # Subsample observational data if too large
    if X_obs.shape[0] > max_obs_samples:
        X_obs = X_obs[:max_obs_samples]
        T_obs = T_obs[:max_obs_samples] 
        Y_obs = Y_obs[:max_obs_samples]
        print(f"  Subsampled obs data to {max_obs_samples} samples")
    
    # Subsample interventional data if too large
    if X_intv.shape[0] > max_intv_samples:
        X_intv = X_intv[:max_intv_samples]
        T_intv = T_intv[:max_intv_samples]
        print(f"  Subsampled intv data to {max_intv_samples} samples")
    
    # The dataset already provides data in the correct format - just add batch dimension
    # for the PyTorch model (the sklearn wrapper expects different format)
    X_obs = X_obs.unsqueeze(0)  # Add batch dimension
    T_obs = T_obs.unsqueeze(0)  # Add batch dimension
    Y_obs = Y_obs.unsqueeze(0)  # Add batch dimension
    X_intv = X_intv.unsqueeze(0)  # Add batch dimension  
    T_intv = T_intv.unsqueeze(0)  # Add batch dimension
    adj = adj.unsqueeze(0)  # Add batch dimension
    
    print(f"Sample {sample_idx} - Final tensor shapes:")
    print(f"  Intervention node: {processor.intervened_feature}")
    print(f"  Target node: {processor.selected_target_feature}")
    print(f"  Kept features: {processor.kept_feature_indices}")
    print(f"  X_obs: {X_obs.shape}, T_obs: {T_obs.shape}, Y_obs: {Y_obs.shape}")
    print(f"  X_intv: {X_intv.shape}, T_intv: {T_intv.shape}")
    print(f"  Adjacency matrix: {adj.shape}")
    
    return X_obs, T_obs, Y_obs, X_intv, T_intv, adj, scm, processor


def create_sample_data(num_features: int, batch_size: int = 1, n_obs: int = 10, n_intv: int = 5):
    """
    Create sample data for visualization.
    
    Args:
        num_features: Number of features
        batch_size: Batch size
        n_obs: Number of observational samples
        n_intv: Number of interventional samples
        
    Returns:
        Tuple of (X_obs, T_obs, Y_obs, X_intv, T_intv)
    """
    torch.manual_seed(42)  # For reproducibility
    
    X_obs = torch.randn(batch_size, n_obs, num_features)
    T_obs = torch.randn(batch_size, n_obs, 1)
    Y_obs = torch.randn(batch_size, n_obs)
    X_intv = torch.randn(batch_size, n_intv, num_features)
    T_intv = torch.randn(batch_size, n_intv, 1)
    
    return X_obs, T_obs, Y_obs, X_intv, T_intv


def create_sample_graphs(num_features: int = 3):
    """
    Create sample adjacency matrices with different characteristics.
    
    Args:
        num_features: Number of features
        
    Returns:
        Dictionary of graph_name -> adjacency_matrix
    """
    num_nodes = num_features + 2  # +2 for T and Y
    
    # Fully connected graph (all edges present)
    adj_full = torch.ones(1, num_nodes, num_nodes, dtype=torch.float32)
    
    # Partially known graph (mix of all three states: -1, 0, 1)
    adj_partial = torch.zeros(1, num_nodes, num_nodes, dtype=torch.float32)
    
    # Example for 5 nodes (T, Y, X0, X1, X2)
    if num_nodes == 5:
        adj_partial[0] = torch.tensor([
            [ 1,  1,  0,  0, -1],  # T: edge to Y, unknown to X0/X1, no edge to X2
            [-1,  1,  0, -1,  0],  # Y: no edge to T, unknown to X0/X2
            [ 0,  1,  1, -1,  1],  # X0: edge to Y and X2, no edge to X1
            [ 0, -1, -1,  1,  0],  # X1: no edge to Y/X0, unknown to others
            [-1,  0,  0,  1,  1],  # X2: edge to X1, no edge to T
        ], dtype=torch.float32)
    else:
        # Generate a random partial graph for other sizes
        torch.manual_seed(42)
        adj_partial[0] = torch.randint(-1, 2, (num_nodes, num_nodes), dtype=torch.float32)
        # Ensure diagonal is 1 (self-loops)
        adj_partial[0].fill_diagonal_(1)
    
    # Sparse graph (mostly no edges)
    adj_sparse = -torch.ones(1, num_nodes, num_nodes, dtype=torch.float32)
    adj_sparse[0].fill_diagonal_(1)  # Self-loops
    # Add a few edges
    if num_nodes >= 3:
        adj_sparse[0, 0, 1] = 1  # T -> Y
        adj_sparse[0, 2, 1] = 1  # X0 -> Y
    
    return {
        'Fully Connected': adj_full,
        'Partially Known (IDK)': adj_partial,
        'Sparse Graph': adj_sparse,
    }


def visualize_attention_maps(model, config, save_dir: str = './attention_visualizations'):
    """
    Visualize attention maps using data sampled from the model's training configuration.
    
    Args:
        model: Loaded model
        config: Model configuration dictionary
        save_dir: Directory to save visualizations
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset from model config
    print("\nCreating dataset from model configuration...")
    try:
        dataset = create_dataset_from_config(config, seed=42)
    except Exception as e:
        print(f"Warning: Failed to create dataset from config: {e}")
        print("Falling back to synthetic data generation...")
        # Fall back to legacy method if dataset creation fails
        num_features = getattr(model.model, 'num_features', 5)
        X_obs, T_obs, Y_obs, X_intv, T_intv = create_sample_data(num_features)
        adjacency_matrix = torch.eye(num_features + 2).unsqueeze(0)  # Identity matrix as fallback
        graphs = {'Fallback Graph': adjacency_matrix}
        dataset = None
    
    # Check if model uses graph conditioning
    use_graph = hasattr(model, 'predict') and 'adjacency_matrix' in model.predict.__code__.co_varnames
    
    if dataset is not None:
        # Use multiple samples from the dataset to show different graph structures
        sample_indices = [0, 2, 5, 8]  # Sample a few different cases
        graphs = {}
        sample_X_obs = sample_T_obs = sample_Y_obs = sample_X_intv = sample_T_intv = None
        
        for i, sample_idx in enumerate(sample_indices):
            try:
                X_obs, T_obs, Y_obs, X_intv, T_intv, adj, scm, processor = sample_data_from_dataset(dataset, sample_idx)
                
                # Create descriptive name for this sample
                intervention_feature = processor.intervened_feature if hasattr(processor, 'intervened_feature') else 'Unknown'
                target_feature = processor.selected_target_feature if hasattr(processor, 'selected_target_feature') else 'Unknown'
                graph_name = f"Sample {sample_idx}: T{intervention_feature}→Y{target_feature}"
                
                graphs[graph_name] = adj
                
                # Use first successfully sampled data for all visualizations
                if sample_X_obs is None:
                    sample_X_obs, sample_T_obs, sample_Y_obs = X_obs, T_obs, Y_obs
                    sample_X_intv, sample_T_intv = X_intv, T_intv
                
            except Exception as e:
                print(f"Warning: Failed to sample data at index {sample_idx}: {e}")
                continue
        
        # If no samples were successfully loaded, fall back to synthetic data
        if not graphs:
            print("No samples loaded successfully, using synthetic data")
            num_features = 5
            sample_X_obs, sample_T_obs, sample_Y_obs, sample_X_intv, sample_T_intv = create_sample_data(num_features)
            graphs = {'Synthetic Graph': torch.eye(num_features + 2).unsqueeze(0)}
    else:
        # Use synthetic graphs as fallback
        sample_X_obs, sample_T_obs, sample_Y_obs, sample_X_intv, sample_T_intv = X_obs, T_obs, Y_obs, X_intv, T_intv
    
    # Determine focus nodes based on actual adjacency matrix size
    if dataset is not None and graphs:
        # Get the size from the first graph
        first_adj = next(iter(graphs.values()))
        adj_size = first_adj.shape[-1]  # Last dimension is the actual matrix size
        focus_nodes = adj_size
        # For 5-node graphs: T, Y, X0, X1, X2
        num_x_features = adj_size - 2  # Subtract T and Y
        feature_names = ['T', 'Y'] + [f'X{i}' for i in range(num_x_features)]
        print(f"Detected {adj_size}-node graphs, using {num_x_features} X features")
    else:
        # Fallback for synthetic data
        focus_nodes = 7
        feature_names = ['T', 'Y'] + [f'X{i}' for i in range(5)]
    
    for graph_name, adj_matrix in graphs.items():
        print(f"\nVisualizing attention for: {graph_name}")
        
        # Extract attention weights using the sampled data
        if use_graph:
            attention_weights = extract_attention_weights(
                model, sample_X_obs, sample_T_obs, sample_Y_obs, sample_X_intv, sample_T_intv, adj_matrix
            )
        else:
            print(f"  Skipping {graph_name} - model doesn't use graph conditioning")
            attention_weights = extract_attention_weights(
                model, sample_X_obs, sample_T_obs, sample_Y_obs, sample_X_intv, sample_T_intv
            )
        
        if not attention_weights:
            print(f"  Warning: No attention weights extracted for {graph_name}")
            continue
        
        n_layers = len(attention_weights)
        n_heads = attention_weights[0].shape[1] if len(attention_weights[0].shape) >= 2 else 1
        
        print(f"  Layers: {n_layers}, Heads per layer: {n_heads}")
        
        # Create visualization - focus on fewer layers and nodes
        fig_width = 4 * min(n_heads, 4) + 2  # Max 4 heads for readability
        fig_height = 4 * min(n_layers, 3) + 1  # Max 3 layers
        
        # Select layers to visualize (first, middle, last)
        if n_layers <= 3:
            layer_indices = list(range(n_layers))
        else:
            layer_indices = [0, n_layers // 2, n_layers - 1]
        
        # Limit heads to show
        heads_to_show = min(n_heads, 4)
        
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = fig.add_gridspec(len(layer_indices), heads_to_show + 1, 
                              width_ratios=[1.2] + [1] * heads_to_show,
                              hspace=0.4, wspace=0.4)
        
        # Plot adjacency matrix in the first column (if graph-conditioned) - focus on 7x7
        if use_graph:
            for row_idx in range(len(layer_indices)):
                ax = fig.add_subplot(gs[row_idx, 0])
                
                # Focus on first 7x7 submatrix
                adj_plot = adj_matrix[0, :focus_nodes, :focus_nodes].numpy()
                
                # Create custom colormap: red (-1), white (0), blue (1)
                cmap = plt.cm.RdBu_r
                im = ax.imshow(adj_plot, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
                
                ax.set_xticks(range(focus_nodes))
                ax.set_yticks(range(focus_nodes))
                ax.set_xticklabels(feature_names, fontsize=12)
                ax.set_yticklabels(feature_names, fontsize=12)
                ax.set_xlabel('To', fontsize=12, fontweight='bold')
                ax.set_ylabel('From', fontsize=12, fontweight='bold')
                
                if row_idx == 0:
                    ax.set_title(f'Input Graph\n({graph_name})', fontsize=13, fontweight='bold')
                
                # Add grid
                ax.set_xticks(np.arange(-0.5, focus_nodes, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, focus_nodes, 1), minor=True)
                ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
                
                # Add colorbar on first one only
                if row_idx == 0:
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label('Edge State', fontsize=10)
                    cbar.set_ticks([-1, 0, 1])
                    cbar.set_ticklabels(['No Edge', 'Unknown', 'Edge'], fontsize=9)
                
                # Add values to cells
                for i in range(focus_nodes):
                    for j in range(focus_nodes):
                        value = adj_plot[i, j]
                        text_color = 'white' if abs(value) > 0.5 else 'black'
                        ax.text(j, i, f'{value:.0f}', ha='center', va='center', 
                               color=text_color, fontsize=10, fontweight='bold')
        
        # Plot attention maps for each head - focus on 7x7
        for row_idx, layer_idx in enumerate(layer_indices):
            attn_weights = attention_weights[layer_idx]
            
            # Handle different attention weight shapes
            if len(attn_weights.shape) == 4:  # (batch*seq, heads, seq, seq)
                attn_per_head = attn_weights.mean(dim=0).numpy()  # Average over batch
            elif len(attn_weights.shape) == 3:  # (heads, seq, seq)
                attn_per_head = attn_weights.numpy()
            else:
                print(f"  Warning: Unexpected attention shape: {attn_weights.shape}")
                continue
            
            for head_idx in range(heads_to_show):
                col_idx = head_idx + 1 if use_graph else head_idx
                ax = fig.add_subplot(gs[row_idx, col_idx])
                
                # Get attention for this head - focus on 7x7 submatrix
                attn_feat = attn_per_head[head_idx, :focus_nodes, :focus_nodes]
                
                # Debug: print attention statistics
                if graph_name == 'Sparse Graph':
                    print(f"    Layer {layer_idx}, Head {head_idx}: min={attn_feat.min():.6f}, max={attn_feat.max():.6f}, mean={attn_feat.mean():.6f}")
                
                # Plot with better colorscale handling
                attn_max = attn_feat.max()
                if attn_max > 1e-6:  # Only use max if it's not essentially zero
                    im = ax.imshow(attn_feat, cmap='viridis', aspect='auto', 
                                  vmin=0, vmax=attn_max)
                else:
                    # Use a small fixed range if all values are near zero
                    im = ax.imshow(attn_feat, cmap='viridis', aspect='auto', 
                                  vmin=0, vmax=0.01)
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Attention', fontsize=9)
                cbar.ax.tick_params(labelsize=8)
                
                # Labels
                ax.set_xticks(range(focus_nodes))
                ax.set_yticks(range(focus_nodes))
                ax.set_xticklabels(feature_names, fontsize=11)
                ax.set_yticklabels(feature_names, fontsize=11)
                ax.set_xlabel('Key (To)', fontsize=10)
                ax.set_ylabel('Query (From)', fontsize=10)
                
                # Title
                if row_idx == 0:
                    ax.set_title(f'Head {head_idx}', fontsize=12, fontweight='bold')
                
                # Add text annotations with attention values
                for i in range(focus_nodes):
                    for j in range(focus_nodes):
                        if attn_max > 1e-6:
                            text_color = 'white' if attn_feat[i, j] > attn_max * 0.5 else 'black'
                            ax.text(j, i, f'{attn_feat[i, j]:.3f}',
                                   ha='center', va='center', color=text_color, fontsize=9)
                        else:
                            # For very small values, show more decimal places
                            ax.text(j, i, f'{attn_feat[i, j]:.4f}',
                                   ha='center', va='center', color='black', fontsize=8)
                
                # Add grid
                ax.set_xticks(np.arange(-0.5, focus_nodes, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, focus_nodes, 1), minor=True)
                ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
            
            # Add layer label on the left
            fig.text(0.02, 1 - (row_idx + 0.5) / len(layer_indices), 
                    f'Layer {layer_idx + 1}',
                    fontsize=14, fontweight='bold',
                    rotation=90, va='center', ha='center')
        
        # Title
        model_type = "Graph-Conditioned" if use_graph else "Standard"
        plt.suptitle(f'{model_type} Model - Attention Maps (First 5 Features)\nGraph: {graph_name}',
                     fontsize=15, fontweight='bold', y=0.98)
        
        # Save
        safe_name = graph_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
        save_path = save_dir / f'attention_map_{safe_name}_focused.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()


def visualize_model_architecture(model, config, save_dir: str = './attention_visualizations'):
    """
    Visualize model architecture and parameters.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the underlying PyTorch model
    pytorch_model = model.model
    
    # Extract architecture info
    n_layers = len(pytorch_model.blocks)
    d_model = getattr(pytorch_model, 'd_model', 'Unknown')
    
    # Get attention head info
    head_info = []
    for i, block in enumerate(pytorch_model.blocks):
        feat_heads = getattr(block.feat_attn, 'num_heads', 'Unknown')
        samp_heads = getattr(block.samp_attn_train, 'num_heads', 'Unknown')
        head_info.append((feat_heads, samp_heads))
    
    # Check for graph conditioning features
    use_graph = hasattr(pytorch_model, 'use_graph_conditioning') or 'Graph' in str(type(model))
    use_gcn = any(hasattr(block, 'gcn') for block in pytorch_model.blocks)
    use_adaln = any(hasattr(block, 'use_adaln') for block in pytorch_model.blocks)
    use_soft_bias = any(hasattr(block, 'bias_edge') for block in pytorch_model.blocks)
    
    # Create architecture summary plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Layer-wise head counts
    layers = list(range(1, n_layers + 1))
    feat_heads = [info[0] for info in head_info]
    samp_heads = [info[1] for info in head_info]
    
    ax1.bar([l - 0.2 for l in layers], feat_heads, 0.4, label='Feature Attention Heads', alpha=0.7)
    ax1.bar([l + 0.2 for l in layers], samp_heads, 0.4, label='Sample Attention Heads', alpha=0.7)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Number of Heads')
    ax1.set_title('Attention Heads per Layer')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Model features
    features = []
    feature_status = []
    
    if use_graph:
        features.append('Graph Conditioning')
        feature_status.append(1)
    if use_gcn:
        features.append('GCN Encoder')
        feature_status.append(1)
    if use_adaln:
        features.append('AdaLN')
        feature_status.append(1)
    if use_soft_bias:
        features.append('Soft Attention Bias')
        feature_status.append(1)
    
    # Add some standard features
    features.extend(['Feature Attention', 'Sample Attention', 'MLP Blocks'])
    feature_status.extend([1, 1, 1])
    
    colors = ['green' if status else 'red' for status in feature_status]
    ax2.barh(features, feature_status, color=colors, alpha=0.7)
    ax2.set_xlabel('Enabled')
    ax2.set_title('Model Features')
    ax2.set_xlim(0, 1.2)
    
    # 3. Parameter count by layer type
    param_counts = {'Total': 0}
    
    for name, param in pytorch_model.named_parameters():
        param_count = param.numel()
        param_counts['Total'] += param_count
        
        # Categorize parameters
        if 'feat_attn' in name:
            param_counts['Feature Attention'] = param_counts.get('Feature Attention', 0) + param_count
        elif 'samp_attn' in name:
            param_counts['Sample Attention'] = param_counts.get('Sample Attention', 0) + param_count
        elif 'mlp' in name:
            param_counts['MLP'] = param_counts.get('MLP', 0) + param_count
        elif 'gcn' in name:
            param_counts['GCN'] = param_counts.get('GCN', 0) + param_count
        elif 'bias_' in name:
            param_counts['Attention Bias'] = param_counts.get('Attention Bias', 0) + param_count
        else:
            param_counts['Other'] = param_counts.get('Other', 0) + param_count
    
    # Remove total and plot
    total_params = param_counts.pop('Total')
    if param_counts:
        ax3.pie(param_counts.values(), labels=param_counts.keys(), autopct='%1.1f%%')
        ax3.set_title(f'Parameter Distribution\n(Total: {total_params:,})')
    
    # 4. Architecture summary text
    ax4.axis('off')
    summary_text = f"""
Model Architecture Summary

Model Type: {'Graph-Conditioned' if use_graph else 'Standard'} InterventionalPFN
Layers: {n_layers}
Model Dimension: {d_model}
Total Parameters: {total_params:,}

Features:
• Graph Conditioning: {'✓' if use_graph else '✗'}
• GCN Encoder: {'✓' if use_gcn else '✗'}
• Adaptive LayerNorm: {'✓' if use_adaln else '✗'}
• Soft Attention Bias: {'✓' if use_soft_bias else '✗'}

Layer Configuration:
"""
    
    for i, (feat_h, samp_h) in enumerate(head_info[:3]):  # Show first 3 layers
        summary_text += f"• Layer {i+1}: {feat_h} feat heads, {samp_h} samp heads\n"
    
    if n_layers > 3:
        summary_text += f"... (and {n_layers-3} more layers)"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    save_path = save_dir / 'model_architecture.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved architecture visualization: {save_path}")
    plt.close()


def create_random_model(config_path: str, checkpoint_path: str):
    """
    Create a randomly initialized model with the same architecture as the trained model.
    
    Args:
        config_path: Path to the config file
        checkpoint_path: Path to the checkpoint (for getting architecture)
        
    Returns:
        Randomly initialized model with same architecture
    """
    print("Creating randomly initialized model...")
    
    # Load the trained model first to get the architecture
    trained_model, config = load_trained_model(config_path, checkpoint_path, verbose=False)
    
    # Reset all model parameters to random initialization
    def reset_weights(module):
        if hasattr(module, 'reset_parameters'):
            try:
                module.reset_parameters()
            except:
                pass  # Some modules might not support reset_parameters
        else:
            # Manual initialization for common layer types
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, (torch.nn.LayerNorm, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                if hasattr(module, 'weight') and module.weight is not None:
                    torch.nn.init.ones_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Embedding):
                torch.nn.init.normal_(module.weight, 0.0, 0.02)
    
    # Apply random initialization to all parameters
    trained_model.model.apply(reset_weights)
    
    print("✓ Random model created by resetting trained model weights")
    return trained_model, config


def main():
    """Main function with hardcoded paths for specific models."""
    
    print('='*70)
    print('Visualizing Attention Maps: Trained vs Random Models')
    print('='*70)
    print()
    
    # Hardcoded paths - modify these to visualize different models
    config_path = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/FirstTests/checkpoints/final_earlytest_16773250.0/step_200000_config.yaml"
    checkpoint_path = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/FirstTests/checkpoints/final_earlytest_16773250.0/step_200000.pt"
    trained_output_dir = "./ancestor_5node_attention_maps"
    random_output_dir = "./ancestor_5node_random_attention_maps"
    
    print(f"Config: {config_path}")
    print(f"Trained checkpoint: {checkpoint_path}")
    print(f"Trained output directory: {trained_output_dir}")
    print(f"Random output directory: {random_output_dir}")
    print()
    
    # Load trained model first to get the correct number of features
    print("Loading trained model...")
    try:
        trained_model, config = load_trained_model(config_path, checkpoint_path)
        
        # Get number of features from the model
        pytorch_model = trained_model.model
        num_features = getattr(pytorch_model, 'num_features', 3)
        print(f"Model expects {num_features} features")
        print()
    except Exception as e:
        print(f"Error loading trained model: {e}")
        return
    
    # Create random model
    try:
        random_model, _ = create_random_model(config_path, checkpoint_path)
    except Exception as e:
        print(f"Error creating random model: {e}")
        return
    
    # Visualize trained model
    print("=" * 50)
    print("TRAINED MODEL VISUALIZATIONS")
    print("=" * 50)
    
    # Visualize architecture
    print("Generating trained model architecture visualization...")
    visualize_model_architecture(trained_model, config, trained_output_dir)
    print()
    
    # Visualize attention maps using dataset from model config
    print("Generating trained model attention map visualizations...")
    visualize_attention_maps(trained_model, config, trained_output_dir)
    print()
    
    # Visualize random model
    print("=" * 50)
    print("RANDOM MODEL VISUALIZATIONS") 
    print("=" * 50)
    
    # Visualize architecture
    print("Generating random model architecture visualization...")
    visualize_model_architecture(random_model, config, random_output_dir)
    print()
    
    # Visualize attention maps using dataset from model config
    print("Generating random model attention map visualizations...")
    visualize_attention_maps(random_model, config, random_output_dir)
    print()
    
    print('='*70)
    print(f'✓ All visualizations complete!')
    print(f'Trained model output: {trained_output_dir}')
    print(f'Random model output: {random_output_dir}')
    print('='*70)


if __name__ == '__main__':
    main()