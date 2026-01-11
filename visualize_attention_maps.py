"""
Visualize attention maps with partial graph conditioning.

This script visualizes how the soft attention bias mechanism affects attention patterns
for different edge states in partial graphs:
- Edge exists (1): bias_edge is ADDED
- No edge (-1): bias_no_edge is SUBTRACTED  
- Unknown edge (0): NO bias applied

The visualization shows attention maps for different graph configurations and compares
how the biases modify the attention scores.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path

# Add the necessary directories to Python path
script_dir = Path(__file__).parent
causal_prior_dir = script_dir  # This is the CausalPriorFitting directory
src_dir = causal_prior_dir / 'src'
src_models_dir = causal_prior_dir / 'src' / 'models'

sys.path.insert(0, str(causal_prior_dir))
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(src_models_dir))

from PartialGraphConditionedInterventionalPFN import PartialGraphConditionedInterventionalPFN


def extract_attention_weights(model, X_obs, T_obs, Y_obs, X_intv, T_intv, adj_matrix):
    """
    Extract attention weights from the model's feature attention layers.
    
    Returns:
        List of attention weight tensors, one per layer
    """
    attention_weights = []
    
    # Hook to capture attention weights
    def attention_hook(module, args, kwargs, output):
        # Force need_weights=True in the forward_call
        pass
    
    def output_hook(module, input, output):
        # MultiheadAttention returns (attn_output, attn_weights) when need_weights=True
        if isinstance(output, tuple) and len(output) == 2 and output[1] is not None:
            attention_weights.append(output[1].detach().cpu())
    
    # Temporarily modify each block to return attention weights
    original_need_weights = []
    hooks = []
    
    for block in model.blocks:
        # Register hook on the feature attention module
        hook = block.feat_attn.register_forward_hook(output_hook)
        hooks.append(hook)
    
    # Monkey-patch the forward method to use need_weights=True
    original_forwards = []
    for block in model.blocks:
        original_forward = block.forward
        original_forwards.append(original_forward)
        
        def make_patched_forward(blk, orig_fwd):
            def patched_forward(x, N_train, N_test, attn_mask=None, graph_embeddings=None):
                # Call original forward but intercept feat_attn call
                B, S, F, D = x.shape
                
                # 1) Feature-wise attention with graph conditioning - pre-layer norm
                if blk.use_adaln and graph_embeddings is not None:
                    x_norm = blk.ln_feat(x, graph_embeddings)
                else:
                    x_norm = blk.ln_feat(x)
                
                # Reshape: (B, S, F, D) -> (B*S, F, D)
                x_flat = x_norm.reshape(B * S, F, D)
                
                # Process attention mask
                float_mask = None
                if attn_mask is not None:
                    if blk.use_soft_attention_bias and hasattr(blk, 'bias_edge') and blk.bias_edge is not None:
                        edge_mask = (attn_mask == 1.0).float()
                        no_edge_mask = (attn_mask == -1.0).float()
                        edge_mask_expanded = edge_mask.unsqueeze(1).expand(B, S, F, F).reshape(B * S, F, F)
                        no_edge_mask_expanded = no_edge_mask.unsqueeze(1).expand(B, S, F, F).reshape(B * S, F, F)
                        num_heads = blk.bias_edge.shape[0]
                        edge_mask_heads = edge_mask_expanded.unsqueeze(1).expand(B * S, num_heads, F, F).reshape(B * S * num_heads, F, F)
                        no_edge_mask_heads = no_edge_mask_expanded.unsqueeze(1).expand(B * S, num_heads, F, F).reshape(B * S * num_heads, F, F)
                        head_bias_edge = blk.bias_edge.view(1, num_heads, 1, 1).expand(B * S, num_heads, F, F).reshape(B * S * num_heads, F, F)
                        head_bias_no_edge = blk.bias_no_edge.view(1, num_heads, 1, 1).expand(B * S, num_heads, F, F).reshape(B * S * num_heads, F, F)
                        float_mask = edge_mask_heads * head_bias_edge - no_edge_mask_heads * head_bias_no_edge
                    else:
                        # Hard masking - expand for each sample in batch
                        mask_expanded = attn_mask.unsqueeze(1).expand(B, S, F, F).reshape(B * S, F, F)
                        float_mask = torch.where(mask_expanded == -1.0, torch.tensor(float('-inf'), device=x.device), torch.tensor(0.0, device=x.device))
                
                # Feature attention - USE need_weights=True
                x2, _ = blk.feat_attn(x_flat, x_flat, x_flat, attn_mask=float_mask, need_weights=True, average_attn_weights=False)
                x_flat = x_flat + blk.drop(x2)
                x = x_flat.reshape(B, S, F, D)
                
                # 2) Sample-wise attention - unchanged from original
                x_row = x.reshape(B, S, F, D)
                x_col = x.permute(0, 2, 1, 3).contiguous().reshape(B * F, S, D)
                x_train = x_col[:, :N_train, :]
                x_test = x_col[:, N_train:, :]
                
                x_train_norm = blk.ln_samp_train(x_train)
                x_train_attn, _ = blk.samp_attn_train(x_train_norm, x_train_norm, x_train_norm, need_weights=False)
                x_train = x_train + blk.drop(x_train_attn)
                
                if N_test > 0:
                    x_test_norm = blk.ln_samp_test(x_test)
                    x_train_norm_kv = blk.ln_samp_test(x_train)
                    x_test_attn, _ = blk.samp_attn_test(x_test_norm, x_train_norm_kv, x_train_norm_kv, need_weights=False)
                    x_test = x_test + blk.drop(x_test_attn)
                
                x_col = torch.cat([x_train, x_test], dim=1)
                x = x_col.reshape(B, F, S, D).permute(0, 2, 1, 3).contiguous()
                
                # 3) Position-wise MLP
                if blk.use_adaln and graph_embeddings is not None:
                    x_norm = blk.ln_mlp(x, graph_embeddings)
                else:
                    x_norm = blk.ln_mlp(x)
                x2 = blk.mlp(x_norm)
                x = x + blk.drop(x2)
                
                return x
            return patched_forward
        
        block.forward = make_patched_forward(block, original_forward)
    
    # Forward pass
    with torch.no_grad():
        _ = model(X_obs, T_obs, Y_obs, X_intv, T_intv, adj_matrix)
    
    # Restore original forward methods and remove hooks
    for i, block in enumerate(model.blocks):
        block.forward = original_forwards[i]
        hooks[i].remove()
    
    return attention_weights


def create_sample_graphs(num_features=3):
    """
    Create sample adjacency matrices with different characteristics.
    
    Args:
        num_features: Number of features (default 3 for 5x5 visualization)
    """
    # 5 features: T, Y, X0, X1, X2
    
    # Partially known graph (mix of all three states)
    adj_partial = torch.tensor([[
        [ 1,  1,  0,  0, -1],  # T: edge to Y, unknown to X0/X1, no edge to X2
        [-1,  1,  0, -1,  0],  # Y: no edge to T, unknown to X0/X2
        [ 0,  1,  1, -1,  1],  # X0: edge to Y and X2, no edge to X1
        [ 0, -1, -1,  1,  0],  # X1: no edge to Y/X0, unknown to others
        [-1,  0,  0,  1,  1],  # X2: edge to X1, no edge to T
    ]], dtype=torch.float32)
    
    # Fully known graph (no unknowns, only 1 and -1)
    adj_fully_known = torch.tensor([[
        [ 1,  1, -1, -1, -1],  # T: edge to Y, no edges to X0/X1/X2
        [-1,  1,  1, -1,  1],  # Y: edge to X0 and X2, no edge to T/X1
        [-1,  1,  1, -1,  1],  # X0: edge to Y and X2, no edge to T/X1
        [-1, -1, -1,  1, -1],  # X1: no edges except self-loop
        [-1,  1, -1,  1,  1],  # X2: edge to Y and X1, no edge to T/X0
    ]], dtype=torch.float32)
    
    return {
        'Partially Known': adj_partial,
        'Fully Known (No IDK)': adj_fully_known,
    }


def visualize_attention_maps(model, graphs, save_path='attention_maps_comparison.png'):
    """
    Visualize attention maps per head for first, middle, and last layers.
    """
    # Create sample data - use num_features from model
    num_features = model.num_features
    B, N, M = 1, 10, 5
    torch.manual_seed(42)
    X_obs = torch.randn(B, N, num_features)
    T_obs = torch.randn(B, N, 1)
    Y_obs = torch.randn(B, N)
    X_intv = torch.randn(B, M, num_features)
    T_intv = torch.randn(B, M, 1)
    
    # Show all nodes (T, Y, X0, X1, X2)
    total_nodes = 5
    # Use INTERNAL ordering to match attention maps: [X0, X1, X2, T, Y]
    feature_names = ['X₀', 'X₁', 'X₂', 'T', 'Y']
    n_layers = len(model.blocks)
    
    # Visualize all graphs
    for graph_idx, (graph_name, adj_matrix) in enumerate(graphs.items()):
        save_path_graph = save_path.replace('.png', f'_{graph_idx+1}_{graph_name.replace(" ", "_").replace("(", "").replace(")", "")}.png')
    # Visualize all graphs
    for graph_idx, (graph_name, adj_matrix) in enumerate(graphs.items()):
        save_path_graph = save_path.replace('.png', f'_{graph_idx+1}_{graph_name.replace(" ", "_").replace("(", "").replace(")", "")}.png')
    
        # Permute adjacency matrix from dataset order [T, Y, X0, X1, X2] to internal order [X0, X1, X2, T, Y]
        # Dataset indices: T=0, Y=1, X0=2, X1=3, X2=4
        # Internal indices: X0=0, X1=1, X2=2, T=3, Y=4
        perm = [2, 3, 4, 0, 1]  # [X0, X1, X2, T, Y]
        adj_matrix_internal = adj_matrix[:, perm, :][:, :, perm]
        
        # Extract attention for the graph
        attention_weights = extract_attention_weights(
            model, X_obs, T_obs, Y_obs, X_intv, T_intv, adj_matrix
        )
        
        # Use all layers
        layer_indices = list(range(n_layers))
        
        # Get number of heads from first layer
        n_heads = attention_weights[0].shape[1]  # (batch*seq, heads, seq, seq)
        
        # Create figure: rows for layers, columns for heads
        fig = plt.figure(figsize=(4 * n_heads + 2, 4 * len(layer_indices) + 1))
        
        # Create grid: one extra column for the adjacency matrix
        gs = fig.add_gridspec(len(layer_indices), n_heads + 1, 
                              width_ratios=[1.2] + [1] * n_heads,
                              hspace=0.3, wspace=0.3)
        
        # Plot adjacency matrix in the first column
        for row_idx in range(len(layer_indices)):
            ax = fig.add_subplot(gs[row_idx, 0])
            
            # Show full adjacency matrix (5x5) in INTERNAL order [X0, X1, X2, T, Y]
            adj_plot = adj_matrix_internal[0].numpy()
            
            # Create custom colormap: red (-1), white (0), blue (1)
            cmap = plt.cm.RdBu_r
            im = ax.imshow(adj_plot, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
            
            # Show all labels
            ax.set_xticks(range(total_nodes))
            ax.set_yticks(range(total_nodes))
            ax.set_xticklabels(feature_names, fontsize=10)
            ax.set_yticklabels(feature_names, fontsize=10)
            ax.set_xlabel('To', fontsize=11, fontweight='bold')
            ax.set_ylabel('From', fontsize=11, fontweight='bold')
            
            if row_idx == 0:
                ax.set_title(f'Input Graph\n({graph_name})', fontsize=12, fontweight='bold')
            
            # Add grid
            ax.set_xticks(np.arange(-0.5, total_nodes, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, total_nodes, 1), minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
            
            # Add colorbar on first one only
            if row_idx == 0:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Edge State', fontsize=9)
                cbar.set_ticks([-1, 0, 1])
                cbar.set_ticklabels(['No Edge', 'Unknown', 'Edge'], fontsize=8)
        
        # Plot attention maps for each head
        for row_idx, layer_idx in enumerate(layer_indices):
            attn_weights = attention_weights[layer_idx]
            
            # Average over batch dimension, get per-head attention: (heads, seq, seq)
            attn_per_head = attn_weights.mean(dim=0).numpy()  # (heads, seq, seq)
            
            for head_idx in range(n_heads):
                ax = fig.add_subplot(gs[row_idx, head_idx + 1])
                
                # Get attention for this head, all 5 features
                attn_feat = attn_per_head[head_idx, :total_nodes, :total_nodes]
                
                # Plot
                im = ax.imshow(attn_feat, cmap='viridis', aspect='auto', vmin=0, vmax=attn_feat.max())
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Attention', fontsize=8)
                cbar.ax.tick_params(labelsize=7)
                
                # Labels
                ax.set_xticks(range(total_nodes))
                ax.set_yticks(range(total_nodes))
                ax.set_xticklabels(feature_names, fontsize=9)
                ax.set_yticklabels(feature_names, fontsize=9)
                ax.set_xlabel('Key (To)', fontsize=9)
                ax.set_ylabel('Query (From)', fontsize=9)
                
                # Title
                if row_idx == 0:
                    ax.set_title(f'Head {head_idx}', fontsize=11, fontweight='bold')
                
                # Add text annotations with attention values
                for i in range(total_nodes):
                    for j in range(total_nodes):
                        text_color = 'white' if attn_feat[i, j] > attn_feat.max() * 0.5 else 'black'
                        ax.text(j, i, f'{attn_feat[i, j]:.2f}',
                               ha='center', va='center', color=text_color, fontsize=7)
                
                # Add grid
                ax.set_xticks(np.arange(-0.5, total_nodes, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, total_nodes, 1), minor=True)
                ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
            
            # Add layer label on the left
            fig.text(0.02, 1 - (row_idx + 0.5) / len(layer_indices), 
                    f'Layer {layer_idx + 1}',
                    fontsize=13, fontweight='bold',
                    rotation=90, va='center', ha='center')
        
        plt.suptitle('Per-Head Attention Maps with Partial Graph Conditioning\n' + 
                     f'Graph: {graph_name} | Soft Bias: +edge_bias for edges (1), -no_edge_bias for no-edges (-1), no bias for unknown (0)',
                     fontsize=13, fontweight='bold', y=0.995)
        
        plt.savefig(save_path_graph, dpi=300, bbox_inches='tight')
        print(f'Saved attention map visualization to: {save_path_graph}')
        plt.close()


def visualize_bias_effects(model, save_path='bias_effects.png'):
    """
    Visualize how the bias values affect attention scores.
    """
    # Extract bias values
    bias_edge_values = []
    bias_no_edge_values = []
    
    for name, param in model.named_parameters():
        if 'bias_edge' in name:
            bias_edge_values.append(param.data.cpu())
        if 'bias_no_edge' in name:
            bias_no_edge_values.append(param.data.cpu())
    
    n_layers = len(bias_edge_values)
    
    fig, axes = plt.subplots(2, n_layers, figsize=(5 * n_layers, 8))
    if n_layers == 1:
        axes = axes.reshape(-1, 1)
    
    for layer_idx in range(n_layers):
        bias_edge = bias_edge_values[layer_idx].numpy()
        bias_no_edge = bias_no_edge_values[layer_idx].numpy()
        
        n_heads = len(bias_edge)
        heads = np.arange(n_heads)
        
        # Plot bias values
        ax = axes[0, layer_idx]
        width = 0.35
        ax.bar(heads - width/2, bias_edge, width, label='bias_edge (added for edges)', 
               color='steelblue', alpha=0.8)
        ax.bar(heads + width/2, bias_no_edge, width, label='bias_no_edge (subtracted for no-edges)', 
               color='coral', alpha=0.8)
        
        ax.set_xlabel('Attention Head', fontsize=11)
        ax.set_ylabel('Bias Value', fontsize=11)
        ax.set_title(f'Layer {layer_idx + 1} Bias Values', fontsize=12, fontweight='bold')
        ax.set_xticks(heads)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Plot effective bias for different edge states
        ax = axes[1, layer_idx]
        edge_states = ['No Edge\n(-1)', 'Unknown\n(0)', 'Edge\n(1)']
        x_pos = np.arange(len(edge_states))
        
        for head_idx in range(n_heads):
            effective_biases = [
                -bias_no_edge[head_idx],  # No edge: subtract bias_no_edge
                0,                         # Unknown: no bias
                bias_edge[head_idx],       # Edge: add bias_edge
            ]
            ax.plot(x_pos, effective_biases, marker='o', label=f'Head {head_idx}', linewidth=2)
        
        ax.set_xlabel('Edge State', fontsize=11)
        ax.set_ylabel('Effective Bias (added to attention scores)', fontsize=11)
        ax.set_title(f'Layer {layer_idx + 1} Effective Bias by Edge State', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(edge_states, fontsize=10)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    plt.suptitle('Soft Attention Bias Values and Effects\n' +
                 'bias_edge ~ TruncNormal(0.5, 0.5), bias_no_edge ~ Normal(5, 1)',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Saved bias effects visualization to: {save_path}')
    plt.close()


def compare_attention_with_without_bias(save_path='attention_comparison.png'):
    """
    Compare attention maps with and without soft biases.
    """
    # Create two models: one with soft bias, one without
    model_with_bias = PartialGraphConditionedInterventionalPFN(
        num_features=3,
        d_model=64,
        depth=2,
        heads_feat=4,
        heads_samp=2,
        use_attention_masking=True,
        use_gcn=False,
        use_adaln=False,
        use_soft_attention_bias=True,
    )
    
    model_without_bias = PartialGraphConditionedInterventionalPFN(
        num_features=3,
        d_model=64,
        depth=2,
        heads_feat=4,
        heads_samp=2,
        use_attention_masking=True,
        use_gcn=False,
        use_adaln=False,
        use_soft_attention_bias=False,
    )
    
    # Use same random seed for both
    torch.manual_seed(42)
    
    # Sample data
    B, N, M, L = 1, 10, 5, 3
    X_obs = torch.randn(B, N, L)
    T_obs = torch.randn(B, N, 1)
    Y_obs = torch.randn(B, N)
    X_intv = torch.randn(B, M, L)
    T_intv = torch.randn(B, M, 1)
    
    # Partially known graph
    adj_partial = torch.tensor([[
        [ 1,  1,  0,  0, -1],  # T
        [-1,  1,  0, -1,  0],  # Y
        [ 0,  1,  1, -1,  1],  # X0
        [ 0, -1, -1,  1,  0],  # X1
        [-1,  0,  0,  1,  1],  # X2
    ]], dtype=torch.float32)
    
    feature_names = ['T', 'Y', 'X₀', 'X₁', 'X₂']
    
    # Extract attention
    attn_with_bias = extract_attention_weights(
        model_with_bias, X_obs, T_obs, Y_obs, X_intv, T_intv, adj_partial
    )
    attn_without_bias = extract_attention_weights(
        model_without_bias, X_obs, T_obs, Y_obs, X_intv, T_intv, adj_partial
    )
    
    n_layers = len(attn_with_bias)
    
    # Create figure
    fig, axes = plt.subplots(n_layers, 3, figsize=(15, 5 * n_layers))
    if n_layers == 1:
        axes = axes.reshape(1, -1)
    
    for layer_idx in range(n_layers):
        # With bias
        ax = axes[layer_idx, 0]
        attn_feat_with = attn_with_bias[layer_idx].mean(dim=0).mean(dim=0)[:5, :5].numpy()
        im = ax.imshow(attn_feat_with, cmap='viridis', aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(feature_names)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')
        ax.set_title(f'Layer {layer_idx + 1}: WITH Soft Bias', fontweight='bold')
        
        for i in range(5):
            for j in range(5):
                text_color = 'white' if attn_feat_with[i, j] > attn_feat_with.max() * 0.5 else 'black'
                ax.text(j, i, f'{attn_feat_with[i, j]:.3f}',
                       ha='center', va='center', color=text_color, fontsize=8)
        
        # Without bias
        ax = axes[layer_idx, 1]
        attn_feat_without = attn_without_bias[layer_idx].mean(dim=0).mean(dim=0)[:5, :5].numpy()
        im = ax.imshow(attn_feat_without, cmap='viridis', aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(feature_names)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')
        ax.set_title(f'Layer {layer_idx + 1}: WITHOUT Bias (Hard Masking)', fontweight='bold')
        
        for i in range(5):
            for j in range(5):
                text_color = 'white' if attn_feat_without[i, j] > attn_feat_without.max() * 0.5 else 'black'
                ax.text(j, i, f'{attn_feat_without[i, j]:.3f}',
                       ha='center', va='center', color=text_color, fontsize=8)
        
        # Difference
        ax = axes[layer_idx, 2]
        diff = attn_feat_with - attn_feat_without
        im = ax.imshow(diff, cmap='RdBu_r', aspect='auto', 
                      vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(feature_names)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')
        ax.set_title(f'Layer {layer_idx + 1}: Difference (With - Without)', fontweight='bold')
        
        for i in range(5):
            for j in range(5):
                text_color = 'white' if np.abs(diff[i, j]) > np.abs(diff).max() * 0.5 else 'black'
                ax.text(j, i, f'{diff[i, j]:+.3f}',
                       ha='center', va='center', color=text_color, fontsize=8)
    
    plt.suptitle('Attention Maps: Soft Bias vs Hard Masking\nPartial Graph with Edge States {-1, 0, 1}',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Saved comparison visualization to: {save_path}')
    plt.close()


def main():
    """Main visualization function."""
    print('='*70)
    print('Visualizing Attention Maps with Partial Graph Conditioning')
    print('='*70)
    print()
    
    # Create model with realistic architecture from training config
    # Based on experiments/FirstTests/configs/basic.yaml:
    # d_model=256, depth=6, heads_feat=4, heads_samp=4
    # Using 3 features for cleaner visualization (will show 5x5: T, Y, X0, X1, X2)
    print('Creating model with soft attention bias...')
    model = PartialGraphConditionedInterventionalPFN(
        num_features=3,
        d_model=256,
        depth=6,
        heads_feat=4,
        heads_samp=4,
        use_attention_masking=True,
        use_gcn=False,
        use_adaln=False,
        use_soft_attention_bias=True,
        dropout=0.0,
        hidden_mult=2,
    )
    model.eval()
    print('✓ Model created')
    print(f'  Architecture: {len(model.blocks)} layers, {model.blocks[0].feat_attn.num_heads} heads, d_model={256}')
    print()
    
    # Create sample graphs
    print('Creating sample graphs...')
    graphs = create_sample_graphs(num_features=3)
    for name, adj in graphs.items():
        print(f'  - {name}: shape={adj.shape}, unique values={torch.unique(adj).tolist()}')
    print()
    
    # Visualization 1: Attention maps for different graphs
    print('Generating attention map comparison...')
    visualize_attention_maps(model, graphs)
    print()
    
    # Visualization 2: Bias values and effects
    print('Generating bias effects visualization...')
    visualize_bias_effects(model)
    print()
    
    # Visualization 3: With vs without bias
    print('Generating soft bias vs hard masking comparison...')
    # Note: Commented out due to technical issues with attention weight extraction
    # when average_attn_weights=False with different mask formats
    # compare_attention_with_without_bias()
    print('  (Skipped - requires additional implementation for hard masking case)')
    print()
    
    print('='*70)
    print('✓ All visualizations complete!')
    print('='*70)
    print()
    print('Generated files:')
    print('  1. attention_maps_comparison.png - Attention patterns for different graphs')
    print('  2. bias_effects.png - Bias values and their effects by edge state')
    # print('  3. attention_comparison.png - Soft bias vs hard masking comparison')


if __name__ == '__main__':
    main()
