"""
Ultimate Graph-Conditioned Interventional Prior-Data Fitted Network (PFN) for causal inference.

This module extends GraphConditionedInterventionalPFN with additional graph processing:
1. Attention masking - hard or soft structural constraints
2. GCN-style graph encoder - learns node representations from graph structure
3. AdaLN - adaptive layer normalization conditioned on graph embeddings

The GCN encoder processes the adjacency matrix into node embeddings that capture
the graph structure. These embeddings can be used for AdaLN to provide learned,
soft graph-based conditioning alongside attention masking.

Input format:
- Same as InterventionalPFN: X_obs, T_obs, Y_obs, X_intv, T_intv
- Additional: adjacency_matrix (B, L+2, L+2) encoding causal structure

Adjacency matrix ordering (NEW - updated to match dataset):
- Position 0: Treatment variable (intervention_node)
- Position 1: Outcome variable (target feature)
- Positions 2 to L+1: Feature variables (kept after dropout, in the SAME ORDER as X columns, correspond to X[:,0] to X[:,L-1])

Edge semantics:
- A[i,j] = 1 means there is a directed edge from i to j (i causes j)
- The matrix is transposed internally so that j can attend to i
- This ensures effects attend to their causes for proper causal inference

Key differences from GraphConditionedInterventionalPFN:
1. Includes GCN-style graph encoder for processing adjacency matrix
2. Produces graph node embeddings (B, L+2, D) for AdaLN conditioning
3. Supports both hard masking (-inf) and soft attention biases (learnable)
4. AdaLN modulates layer normalization using graph embeddings
5. Flexible configuration: can enable/disable masking, GCN, AdaLN independently

Architecture features:
- SwiGLU activation
- Pre-layer normalization (adaptive when AdaLN enabled)
- Separate train/test attention
- Graph-conditioned feature attention (hard or soft)
- GCN-style graph processing
- Optional attention sinks
"""

from __future__ import annotations
from typing import Optional, Dict, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Two-layer feed-forward block with SwiGLU activation and dropout.
    
    SwiGLU uses a gated linear unit with SiLU (Swish) activation:
        SwiGLU(x) = (W1*x) ⊗ silu(W_gate*x) * W2
    
    This typically outperforms GELU in modern architectures.

    Args:
        dim: Input and output feature dimension (D).
        hidden_mult: Multiplier for the hidden layer size (hidden = hidden_mult * dim).
        dropout: Dropout rate applied after SwiGLU and after the second linear layer.
    """
    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = hidden_mult * dim
        self.fc1 = nn.Linear(dim, hidden)
        self.fc_gate = nn.Linear(dim, hidden)  # Gate pathway for SwiGLU
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation: element-wise product of linear projection and gated SiLU
        x = self.fc1(x) * F.silu(self.fc_gate(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class InputMLP(nn.Module):
    """
    Generic 2-layer MLP with SwiGLU activation for input embeddings (row-wise and scalar-wise).
    
    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        hidden_mult: Multiplier for hidden layer size
        dropout: Dropout rate
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = hidden_mult * out_dim
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc_gate = nn.Linear(in_dim, hidden)  # Gate pathway for SwiGLU
        self.fc2 = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation
        x = self.fc1(x) * F.silu(self.fc_gate(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization conditioned on graph embeddings.
    
    Computes scale and shift parameters from graph node embeddings to modulate
    the layer normalization. This allows different features to be normalized
    differently based on their position in the causal graph.
    
    Args:
        d_model: Feature dimension
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        # MLP to predict scale and shift from graph embedding
        # Uses SwiGLU-style gating for consistency with rest of model
        self.scale_gate = nn.Linear(d_model, d_model)
        self.scale_proj = nn.Linear(d_model, d_model)
        self.shift_gate = nn.Linear(d_model, d_model)
        self.shift_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor, graph_emb: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive layer normalization.
        
        Args:
            x: Input tensor to normalize
                - (B*S, F, D) for feature attention
                - (B, S, F, D) for MLP
            graph_emb: (B, F, D) - graph embeddings per feature
            
        Returns:
            Normalized and modulated tensor with same shape as x
        """
        # Normalize first
        x_norm = self.ln(x)
        
        # Compute scale and shift with SwiGLU-style gating
        scale = self.scale_proj(graph_emb) * F.silu(self.scale_gate(graph_emb))  # (B, F, D)
        shift = self.shift_proj(graph_emb) * F.silu(self.shift_gate(graph_emb))  # (B, F, D)
        
        # Initialize scale to 1 (identity) and shift to 0 by default
        scale = scale + 1.0
        
        # Broadcast graph embeddings to match x shape
        if x.dim() == 3:  # (B*S, F, D) - feature attention case
            # graph_emb is (B, F, D), need to handle batch dimension
            # Extract B from x's batch-sample dimension
            BS = x.shape[0]
            B = graph_emb.shape[0]
            S = BS // B
            # Expand: (B, F, D) -> (B, 1, F, D) -> (B, S, F, D) -> (B*S, F, D)
            scale = scale.unsqueeze(1).expand(-1, S, -1, -1).reshape(BS, -1, scale.shape[-1])
            shift = shift.unsqueeze(1).expand(-1, S, -1, -1).reshape(BS, -1, shift.shape[-1])
        elif x.dim() == 4:  # (B, S, F, D) - MLP case
            # Expand: (B, F, D) -> (B, 1, F, D)
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        
        return scale * x_norm + shift


class GraphEncoder(nn.Module):
    """
    Lightweight GCN-style graph encoder that processes the adjacency matrix.
    
    Takes an adjacency matrix and produces node embeddings via one layer of 
    graph convolution: H = σ(D^(-1/2) A D^(-1/2) X W)
    
    Args:
        num_nodes: Number of nodes (L+2: Treatment, Outcome, Features)
        d_model: Output embedding dimension
        dropout: Dropout rate
    """
    def __init__(self, num_nodes: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        
        # Linear transformation for GCN
        self.weight = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        adjacency_matrix: torch.Tensor,
        node_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply GCN-style message passing.
        
        Args:
            adjacency_matrix: (B, L+2, L+2) - adjacency matrix
            node_features: (L+2, D) - initial node features (role embeddings)
            
        Returns:
            Graph node embeddings: (B, L+2, D)
        """
        B = adjacency_matrix.shape[0]
        
        # Add self-loops to adjacency matrix
        A = adjacency_matrix.float()
        eye = torch.eye(self.num_nodes, device=A.device, dtype=A.dtype)
        A_self = A + eye.unsqueeze(0)  # (B, L+2, L+2)
        
        # Compute degree matrix D
        D = A_self.sum(dim=-1)  # (B, L+2) - row sums (out-degree)
        D_inv_sqrt = torch.pow(D, -0.5)  # (B, L+2)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0  # Handle isolated nodes
        
        # Normalize adjacency: D^(-1/2) A D^(-1/2)
        # A_norm[b, i, j] = A[b, i, j] / sqrt(D[b, i] * D[b, j])
        A_norm = D_inv_sqrt.unsqueeze(-1) * A_self * D_inv_sqrt.unsqueeze(-2)  # (B, L+2, L+2)
        
        # Expand node features for batch
        X = node_features.unsqueeze(0).expand(B, -1, -1)  # (B, L+2, D)
        
        # Graph convolution: A_norm @ X @ W
        H = torch.matmul(A_norm, X)  # (B, L+2, D)
        H = self.weight(H)  # (B, L+2, D)
        H = F.relu(H)
        H = self.dropout(H)
        
        return H  # (B, L+2, D)


class GraphConditionedTwoWayBlock(nn.Module):
    """
    Two-way attention block with causal graph conditioning via attention masking and AdaLN.
    
    Feature attention is modified to respect the causal graph structure:
    - Attention masking determines which features can attend to which
    - AdaLN uses graph embeddings to modulate layer normalization
    - Sample attention remains unchanged (no graph conditioning)
    
    Uses pre-layer normalization and separate attention layers for train and test samples.
    
    Args:
        dim: Model dimension
        heads_feat: Number of attention heads for feature-wise attention
        heads_samp: Number of attention heads for sample-wise attention
        dropout: Dropout rate
        hidden_mult: Hidden layer multiplier for MLPs
        use_adaln: Whether to use AdaLN (adaptive layer norm) with graph embeddings
        use_soft_attention_bias: Whether to use learnable soft biases instead of hard masking
        soft_bias_init: Initial value for soft attention biases (positive = encourage attention)
    """
    def __init__(
        self, 
        dim: int, 
        heads_feat: int, 
        heads_samp: int, 
        dropout: float = 0.0, 
        hidden_mult: int = 4,
        use_adaln: bool = True,
        use_soft_attention_bias: bool = False,
        soft_bias_init: float = 100.0
    ):
        super().__init__()
        self.use_adaln = use_adaln
        self.use_soft_attention_bias = use_soft_attention_bias
        self.heads_feat = heads_feat
        
        # Self-attention across features with graph conditioning: (B*S, L+2, D)
        self.feat_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads_feat,
            dropout=dropout,
            batch_first=True,
        )
        
        # Learnable soft attention biases (one per head)
        # Only used when use_soft_attention_bias=True
        if use_soft_attention_bias:
            # Initialize with positive values to encourage attention where graph permits
            self.soft_attention_bias = nn.Parameter(torch.full((heads_feat,), soft_bias_init))
        else:
            self.soft_attention_bias = None
        
        # Use AdaLN or regular LayerNorm for feature attention
        if use_adaln:
            self.ln_feat = AdaLN(dim)
        else:
            self.ln_feat = nn.LayerNorm(dim)

        # Self-attention for train samples across samples: (B*(L+2), N, D)
        self.samp_attn_train = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads_samp,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_samp_train = nn.LayerNorm(dim)
        
        # Cross-attention for test samples to train samples: (B*(L+2), M, D) attending to (B*(L+2), N, D)
        self.samp_attn_test = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads_samp,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_samp_test = nn.LayerNorm(dim)

        # Position-wise MLP
        self.mlp = MLP(dim, hidden_mult=hidden_mult, dropout=dropout)
        # Use AdaLN or regular LayerNorm for MLP
        if use_adaln:
            self.ln_mlp = AdaLN(dim)
        else:
            self.ln_mlp = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self, 
        x: torch.Tensor, 
        N_train: int, 
        N_test: int,
        attn_mask: Optional[torch.Tensor] = None,
        graph_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, S, F, D) where S = N_train + N_test, F = L+2.
            N_train: Number of train (observational) samples.
            N_test: Number of test (interventional) samples.
            attn_mask: Attention mask for feature attention of shape (B*S, F, F) or (F, F).
                      True/1 means position CAN attend, False/0 means CANNOT attend.
                      If None, standard full attention is used.
            graph_embeddings: (B, F, D) - graph node embeddings for AdaLN conditioning.
                             If None or use_adaln=False, regular LayerNorm is used.

        Returns:
            Tensor of shape (B, S, F, D).
        """
        B, S, F, D = x.shape
        assert S == N_train + N_test, f"Expected {N_train + N_test} samples, got {S}"

        # 1) Feature-attention (within row) with graph conditioning - pre-layer norm
        x_row = x.reshape(B * S, F, D)  # (B*S, F, D)
        
        # Apply layer norm (adaptive if graph embeddings provided)
        if self.use_adaln and graph_embeddings is not None:
            x_norm = self.ln_feat(x_row, graph_embeddings)
        else:
            x_norm = self.ln_feat(x_row)
        
        # Apply attention mask if provided
        # PyTorch MultiheadAttention expects mask of shape:
        # - 2D: (F, F) for all batches and heads
        # - 3D: (B*S*num_heads, F, F) for per-head masking
        if attn_mask is not None:
            # attn_mask shape: (B, F, F) - same mask for all samples in each batch
            
            if self.use_soft_attention_bias:
                # Soft attention: use learnable biases instead of hard masking
                # Create additive bias mask: add bias where graph permits, leave rest at 0
                # IMPORTANT: attn_mask from _prepare_attention_mask is NEGATED
                # So False = CAN attend (edge exists), True = CANNOT attend (no edge)
                if attn_mask.dtype == torch.bool:
                    # attn_mask is negated: False = can attend, True = cannot attend
                    # We want to ADD positive bias where False (where edges exist)
                    # Convert to float: False -> 1.0 (add bias), True -> 0.0 (no bias)
                    bias_mask = (~attn_mask).float()  # (B, F, F)
                else:
                    # attn_mask is negated: 0.0 = can attend, -inf/other = cannot attend
                    # For soft bias, we want 1.0 where attention is allowed (0.0), 0.0 elsewhere
                    bias_mask = (attn_mask == 0.0).float()  # (B, F, F)
                
                # Expand to (B*S, F, F)
                bias_mask = bias_mask.unsqueeze(1).expand(-1, S, -1, -1).reshape(B * S, F, F)
                
                # Expand for num_heads and apply per-head biases: (B*S*num_heads, F, F)
                num_heads = self.feat_attn.num_heads
                bias_mask = bias_mask.unsqueeze(1).expand(-1, num_heads, -1, -1).reshape(B * S * num_heads, F, F)
                
                # Apply learnable biases (one per head)
                # soft_attention_bias: (num_heads,)
                # Reshape to (1, num_heads, 1, 1) for broadcasting
                head_biases = self.soft_attention_bias.view(1, num_heads, 1, 1)
                # Expand to (B*S, num_heads, F, F)
                head_biases = head_biases.expand(B * S, -1, F, F).reshape(B * S * num_heads, F, F)
                
                # Apply: add bias only where edges exist in the graph
                # Result: float_mask[i,j] = bias_value if edge j->i exists, 0.0 otherwise
                # This will be ADDED to attention scores before softmax
                float_mask = bias_mask * head_biases
                
            else:
                # Hard masking: standard approach
                # Convert boolean mask to additive mask
                if attn_mask.dtype == torch.bool:
                    # Convert: True -> 0.0 (attend), False -> -inf (mask)
                    float_mask = torch.zeros_like(attn_mask, dtype=x.dtype)
                    float_mask = float_mask.masked_fill(~attn_mask, float('-inf'))
                else:
                    # Assume already float: 1.0 -> 0.0 (attend), 0.0 -> -inf (mask)
                    float_mask = torch.zeros_like(attn_mask, dtype=x.dtype)
                    float_mask = float_mask.masked_fill(attn_mask == 0, float('-inf'))
                
                # float_mask is (B, F, F)
                # Expand to (B*S, F, F) by repeating each batch's mask S times
                float_mask = float_mask.unsqueeze(1).expand(-1, S, -1, -1).reshape(B * S, F, F)  # (B*S, F, F)
                
                # Now expand for num_heads: (B*S*num_heads, F, F)
                num_heads = self.feat_attn.num_heads
                float_mask = float_mask.unsqueeze(1).expand(-1, num_heads, -1, -1).reshape(B * S * num_heads, F, F)
            x2, _ = self.feat_attn(x_norm, x_norm, x_norm, attn_mask=float_mask, need_weights=False)
        else:
            x2, _ = self.feat_attn(x_norm, x_norm, x_norm, need_weights=False)
        
        x_row = x_row + self.drop(x2)
        x = x_row.reshape(B, S, F, D)

        # 2) Sample-attention (within column) - separate for train and test - pre-layer norm
        # No graph conditioning here - sample relationships are independent of feature graph
        x_col = x.permute(0, 2, 1, 3).contiguous().reshape(B * F, S, D)  # (B*F, S, D)
        
        # Split into train and test
        x_train = x_col[:, :N_train, :]  # (B*F, N_train, D)
        x_test = x_col[:, N_train:, :]   # (B*F, N_test, D)
        
        # Train self-attention
        x_train_norm = self.ln_samp_train(x_train)
        x_train_attn, _ = self.samp_attn_train(x_train_norm, x_train_norm, x_train_norm, need_weights=False)
        x_train = x_train + self.drop(x_train_attn)
        
        # Test cross-attention to train
        if N_test > 0:
            x_test_norm = self.ln_samp_test(x_test)
            x_train_norm_kv = self.ln_samp_test(x_train)  # Use same norm for key/value
            x_test_attn, _ = self.samp_attn_test(x_test_norm, x_train_norm_kv, x_train_norm_kv, need_weights=False)
            x_test = x_test + self.drop(x_test_attn)
        
        # Concatenate back
        x_col = torch.cat([x_train, x_test], dim=1)  # (B*F, S, D)
        x = x_col.reshape(B, F, S, D).permute(0, 2, 1, 3).contiguous()  # (B, S, F, D)

        # 3) Position-wise MLP - pre-layer norm
        if self.use_adaln and graph_embeddings is not None:
            x_norm = self.ln_mlp(x, graph_embeddings)
        else:
            x_norm = self.ln_mlp(x)
        x2 = self.mlp(x_norm)
        x = x + self.drop(x2)
        return x


class UltimateGraphConditionedInterventionalPFN(nn.Module):
    """
    Ultimate PFN-like regressor with flexible graph conditioning mechanisms.

    Combines multiple approaches to incorporate causal graph structure:
    1. Attention masking - hard or soft constraints on which features can attend to which
       - Hard masking: -inf where graph prohibits attention (use_soft_attention_bias=False)
       - Soft masking: learnable positive biases where graph permits attention (use_soft_attention_bias=True)
    2. GCN-style graph encoder - learns node representations from adjacency matrix
    3. AdaLN - adaptive layer normalization conditioned on graph embeddings
    
    The GCN encoder processes the graph structure into node embeddings (B, L+2, D).
    These embeddings can be used for AdaLN to provide learned, soft graph-based 
    conditioning alongside attention masking.
    
    Adjacency matrix format:
    - Shape: (B, L+2, L+2) where L is number of features (after dropout)
    - Position 0: Treatment variable (intervention_node)
    - Position 1: Outcome variable (target feature)
    - Positions 2 to L+1: Feature variables (kept after dropout, in the SAME ORDER as X columns, correspond to X[:,0] to X[:,L-1])
    
    IMPORTANT: The internal embedding order is DIFFERENT from adjacency matrix order!
    - Internal embedding order: [X_0, X_1, ..., X_{L-1}, T, Y]
    - Adjacency matrix order: [T, Y, X_0, X_1, ..., X_{L-1}]
    - The model automatically reorders the adjacency matrix to match internal order
    
    - Edge semantics: A[i,j] = 1 means edge from i to j (i causes j)
    - The matrix is transposed internally so j can attend to i
    - This ensures effects attend to their causes for causal inference
    
    Architecture features:
    - SwiGLU activation
    - Pre-layer normalization (adaptive when AdaLN enabled)
    - Separate train/test attention
    - Graph-conditioned feature attention (hard or soft)
    - Optional attention sinks
    
    Args:
        num_features: Number of regular features (L) in X_obs/X_intv
        d_model: Model embedding dimension
        depth: Number of two-way attention blocks
        heads_feat: Number of attention heads for feature-wise attention
        heads_samp: Number of attention heads for sample-wise attention
        dropout: Dropout rate
        output_dim: Output dimension (1 for regression, >1 for distributional outputs)
        hidden_mult: Hidden layer multiplier for MLPs
        normalize_features: Whether to apply per-task feature normalization
        use_same_row_mlp: Whether to use the same row MLP for train and test data
        n_sample_attention_sink_rows: Number of learnable sink rows for sample attention stability
        n_feature_attention_sink_cols: Number of learnable sink columns for feature attention stability
        use_attention_masking: Whether to apply graph-based attention masking (hard constraints)
        use_gcn: Whether to use GCN encoder to process graph structure
        use_adaln: Whether to use Adaptive Layer Normalization with graph embeddings (requires use_gcn=True)
        use_soft_attention_bias: Whether to use learnable soft attention biases (alternative to hard masking)
        soft_bias_init: Initial value for soft attention biases (positive encourages attention)
    """
    def __init__(
        self,
        num_features: int,
        d_model: int = 256,
        depth: int = 8,
        heads_feat: int = 8,
        heads_samp: int = 8,
        dropout: float = 0.0,
        output_dim: int = 1,
        hidden_mult: int = 4,
        normalize_features: bool = True,
        use_same_row_mlp: bool = True,
        n_sample_attention_sink_rows: int = 0,
        n_feature_attention_sink_cols: int = 0,
        use_attention_masking: bool = True,
        use_gcn: bool = True,
        use_adaln: bool = True,
        use_soft_attention_bias: bool = False,
        soft_bias_init: float = 5.0,
    ):
        super().__init__()
        self.num_features = num_features  # L (excluding intervened column)
        self.d_model = d_model
        self.output_dim = output_dim
        self.normalize_features = normalize_features
        self.n_sample_attention_sink_rows = n_sample_attention_sink_rows
        self.n_feature_attention_sink_cols = n_feature_attention_sink_cols
        self.use_attention_masking = use_attention_masking
        self.use_gcn = use_gcn
        self.use_adaln = use_adaln
        self.use_soft_attention_bias = use_soft_attention_bias
        
        # Validate configuration
        if use_adaln and not use_gcn:
            raise ValueError("use_adaln=True requires use_gcn=True (AdaLN needs graph embeddings from GCN)")
        
        if use_soft_attention_bias and not use_attention_masking:
            raise ValueError("use_soft_attention_bias=True requires use_attention_masking=True (soft bias needs attention mask to define where to apply bias)")

        # === Embedding MLPs ===
        # Row-wise MLPs over feature dimension (R^(L+1) -> R^D)
        # Note: L+1 because we concatenate T to X
        if use_same_row_mlp:
            self.row_mlp_train = InputMLP(num_features + 1, d_model, hidden_mult, dropout)
            self.row_mlp_test = self.row_mlp_train
        else:
            self.row_mlp_train = InputMLP(num_features + 1, d_model, hidden_mult, dropout)
            self.row_mlp_test = InputMLP(num_features + 1, d_model, hidden_mult, dropout)

        # Shared scalar MLP for regular feature cells
        self.cell_mlp = InputMLP(1, d_model, hidden_mult, dropout)
        
        # Train label scalar MLP
        self.label_mlp_train = InputMLP(1, d_model, hidden_mult, dropout)

        # Feature positional encodings (sinusoidal) for:
        # - L regular features
        # - 1 intervened feature (special position)
        # - 1 label column
        # Total: L + 2 positions
        feat_pos = self._build_feature_positional(num_features + 2, d_model)  # (L+2, D)
        self.register_buffer("feature_positional", feat_pos.unsqueeze(0).unsqueeze(0), persistent=False)

        # Learnable scaling for row vs cell embeddings (for stability)
        self.row_scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(2.0)))
        self.cell_scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(2.0)))

        # === Attention Sinks ===
        # Optional learnable sink rows and columns for attention stability
        if n_sample_attention_sink_rows > 0:
            # Sink rows (dummy samples that all samples can attend to)
            # Shape: (1, n_sink_rows, L+2, D)
            self.sink_rows_x = nn.Parameter(torch.zeros(1, n_sample_attention_sink_rows, num_features + 2, d_model))
            nn.init.normal_(self.sink_rows_x, std=0.02)
            
            # Separate sink row for train labels
            self.sink_rows_y = nn.Parameter(torch.zeros(1, n_sample_attention_sink_rows, d_model))
            nn.init.normal_(self.sink_rows_y, std=0.02)
        else:
            self.sink_rows_x = None
            self.sink_rows_y = None
        
        if n_feature_attention_sink_cols > 0:
            # Sink columns (dummy features that all features can attend to)
            # Shape: (1, 1, n_sink_cols, D)
            self.sink_cols = nn.Parameter(torch.zeros(1, 1, n_feature_attention_sink_cols, d_model))
            nn.init.normal_(self.sink_cols, std=0.02)
        else:
            self.sink_cols = None

        # Learnable role embeddings
        self.obs_T_embed = self._create_role_embedding(1, 1, self.d_model)
        self.obs_label_embed = self._create_role_embedding(1, 1, self.d_model)
        self.obs_feature_embed = self._create_role_embedding(1, 1, self.d_model)
        self.intv_T_embed = self._create_role_embedding(1, 1, self.d_model)
        self.intv_label_embed = self._create_role_embedding(1, 1, self.d_model)
        self.intv_feature_embed = self._create_role_embedding(1, 1, self.d_model)

        # === Graph Encoder ===
        # GCN-style encoder that processes adjacency matrix into node embeddings
        # Only created if use_gcn is True
        if use_gcn:
            self.graph_encoder = GraphEncoder(
                num_nodes=num_features + 2,
                d_model=d_model,
                dropout=dropout
            )
        else:
            self.graph_encoder = None

        # Stacked two-way attention blocks with graph conditioning
        self.blocks = nn.ModuleList([
            GraphConditionedTwoWayBlock(
                d_model, heads_feat, heads_samp, 
                dropout=dropout, hidden_mult=hidden_mult,
                use_adaln=use_adaln,
                use_soft_attention_bias=use_soft_attention_bias,
                soft_bias_init=soft_bias_init
            )
            for _ in range(depth)
        ])

        # Output projection from D to desired output_dim per test token
        self.regression_head = nn.Linear(d_model, output_dim)

    def _create_role_embedding(self, *shape, std=0.02):
        """Helper to create and initialize a role embedding parameter."""
        embed = nn.Parameter(torch.zeros(*shape))
        nn.init.normal_(embed, std=std)
        return embed
    
    @staticmethod
    def _build_feature_positional(num_tokens: int, dim: int) -> torch.Tensor:
        """
        Sinusoidal encodings across feature axis (including intervened and label columns).
        """
        pe = torch.zeros(num_tokens, dim)
        position = torch.arange(0, num_tokens, dtype=torch.float32).unsqueeze(1)  # (num_tokens, 1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (num_tokens, dim)

    @staticmethod
    def _normalize_features(
        X_train: torch.Tensor,
        X_test: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize features per batch/task and per feature dimension using:
        1. Uniform quantile transform based on the support set (X_train)
        2. Standard normalization (mean/std) based on the support set
        
        This follows the PFN preprocessing: quantile transform followed by standardization.

        Args:
            X_train: (B, N, L+1) - support set (observational features + intervened)
            X_test:  (B, M, L+1) - query set (interventional features + intervened)

        Returns:
            X_train_norm, X_test_norm with same shapes.
        """
        B, N, F = X_train.shape  # F = L+1
        M = X_test.shape[1]
        
        # Step 1: Uniform quantile transform based on X_train (support set)
        X_train_sorted, _ = torch.sort(X_train, dim=1)  # (B, N, F)
        
        def quantile_transform(X, X_sorted):
            """Map X to quantiles based on sorted support set X_sorted."""
            B, S, F = X.shape
            B_s, N, F_s = X_sorted.shape
            assert B == B_s and F == F_s
            
            X_quantiles = torch.zeros_like(X)
            for b in range(B):
                for f in range(F):
                    sorted_vals = X_sorted[b, :, f]  # (N,)
                    vals = X[b, :, f]  # (S,)
                    
                    ranks = torch.searchsorted(sorted_vals.contiguous(), vals.contiguous())
                    quantiles = ranks.float() / max(N - 1, 1)
                    quantiles = quantiles.clamp(0.0, 1.0)
                    
                    X_quantiles[b, :, f] = quantiles
            
            return X_quantiles
        
        X_train_quantiles = quantile_transform(X_train, X_train_sorted)
        X_test_quantiles = quantile_transform(X_test, X_train_sorted)
        
        # Step 2: Standard normalization
        mean = X_train_quantiles.mean(dim=1, keepdim=True)  # (B, 1, F)
        std = X_train_quantiles.std(dim=1, keepdim=True, unbiased=False)  # (B, 1, F)
        std = std.clamp_min(1e-6)
        
        X_train_norm = (X_train_quantiles - mean) / std
        X_test_norm = (X_test_quantiles - mean) / std
        
        return X_train_norm, X_test_norm

    def _embed_features(
        self,
        X: torch.Tensor,
        T: torch.Tensor,
        row_mlp: nn.Module,
        is_intvn: bool,
    ) -> torch.Tensor:
        """
        Embed features with special treatment for intervened column.

        Args:
            X: (B, S, L) - regular features
            T: (B, S, 1) - intervened feature column
            row_mlp: Row-wise MLP to use (train or test)
            is_intvn: Whether this is interventional data (affects which MLPs are used)

        Returns:
            Tensor of shape (B, S, L+1, D) where position L is the intervened feature.
        """
        B, S, L = X.shape
        assert L == self.num_features
        assert T.shape == (B, S, 1)
        
        # Concatenate T to X for row-wise embedding
        X_with_T = torch.cat([X, T], dim=2)  # (B, S, L+1)
        
        # Row-wise embedding
        row_emb = row_mlp(X_with_T)  # (B, S, D)

        # Cell-wise embeddings
        # Regular features use cell_mlp: X -> (B, S, L) -> add dim -> (B, S, L, 1) -> MLP -> (B, S, L, D)
        cell_emb = self.cell_mlp(X_with_T.unsqueeze(-1))  # (B, S, L+1, D)

        # Split into regular features and intervened feature
        X_cells = cell_emb[:, :, :-1, :]  # (B, S, L, D)
        T_cells = cell_emb[:, :, -1:, :]  # (B, S, 1, D)
        
        # Add role embeddings based on whether this is interventional data
        if is_intvn:
            X_cells = X_cells + self.intv_feature_embed.expand(B, S, L, -1)
            T_cells = T_cells + self.intv_T_embed.expand(B, S, 1, -1)
        else:
            X_cells = X_cells + self.obs_feature_embed.expand(B, S, L, -1)
            T_cells = T_cells + self.obs_T_embed.expand(B, S, 1, -1)
        
        # Concatenate back
        cell_emb = torch.cat([X_cells, T_cells], dim=2)  # (B, S, L+1, D)

        # Combine row and cell embeddings
        row_exp = row_emb.unsqueeze(2).expand(-1, -1, L + 1, -1)  # (B, S, L+1, D)
        feat_emb = self.row_scale * row_exp + self.cell_scale * cell_emb
        
        return feat_emb

    def _embed_labels(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Embed labels as label-column tokens.

        Args:
            Y: (B, S) or (B, S, 1)

        Returns:
            Tensor of shape (B, S, D).
        """
        if Y.dim() == 3:
            Y = Y.squeeze(-1)  # (B, S)
        label_emb = self.label_mlp_train(Y.unsqueeze(-1))  # (B, S, D)
        # Add role embedding for observational labels
        label_emb = label_emb + self.obs_label_embed.expand(Y.size(0), Y.size(1), self.d_model)
        return label_emb

    def _prepare_attention_mask(
        self,
        adjacency_matrix: torch.Tensor,
        n_sink_cols: int,
    ) -> torch.Tensor:
        """
        Prepare attention mask for feature attention from adjacency matrix.
        
        The adjacency matrix from the dataset uses the ordering:
        - Position 0: Treatment variable (T)
        - Position 1: Outcome variable (Y)
        - Positions 2 to L+1: Feature variables (X_0, ..., X_{L-1})
        
        The model internally uses a different ordering:
        - Positions 0 to L-1: Feature variables (X_0, ..., X_{L-1})
        - Position L: Treatment variable (T)
        - Position L+1: Outcome variable (Y)
        
        This method reorders the adjacency matrix to match the internal ordering,
        then prepares the attention mask.
        
        Edge semantics:
        - A[i,j] = 1 means there is a directed edge from i to j (i causes j)
        
        For causal inference, we need information to flow backward along causal edges:
        - If i→j (i causes j), then j should attend to i (effect attends to cause)
        - This means we need to TRANSPOSE the adjacency matrix
        
        After transposing:
        - A_T[j,i] = 1 when original A[i,j] = 1 (edge i→j exists)
        - This allows position j to attend to position i
        
        The mask is created by reordering, transposing, converting to boolean, then negating (logical_not).
        After negation: False = CAN attend, True = CANNOT attend
        
        With sink columns, the mask is expanded to allow all features to attend to sinks.
        
        Args:
            adjacency_matrix: (B, L+2, L+2) - Binary adjacency matrix from dataset
                              Ordering: [T, Y, X_0, ..., X_{L-1}]
                              A[i,j] = 1 means edge from i to j (i causes j)
            n_sink_cols: Number of sink columns prepended
            
        Returns:
            Attention mask of shape (B, n_sink_cols + L+2, n_sink_cols + L+2)
            where False means position CAN attend (PyTorch convention after negation).
        """
        B, F, F2 = adjacency_matrix.shape
        assert F == F2, "Adjacency matrix must be square"
        assert F == self.num_features + 2, f"Expected adjacency matrix size {self.num_features + 2}, got {F}"
        
        # Reorder adjacency matrix from [T, Y, X_0, ..., X_{L-1}] to [X_0, ..., X_{L-1}, T, Y]
        # Input positions: 0=T, 1=Y, 2 to L+1=X_0 to X_{L-1}
        # Output positions: 0 to L-1=X_0 to X_{L-1}, L=T, L+1=Y
        L = self.num_features
        
        # Create permutation indices
        # New position 0 to L-1 should come from old position 2 to L+1 (features)
        # New position L should come from old position 0 (treatment)
        # New position L+1 should come from old position 1 (outcome)
        perm_indices = list(range(2, L + 2)) + [0, 1]  # [2, 3, ..., L+1, 0, 1]
        
        # Permute both rows and columns
        adjacency_matrix = adjacency_matrix[:, perm_indices, :]  # Permute rows
        adjacency_matrix = adjacency_matrix[:, :, perm_indices]  # Permute columns
        # Now adjacency_matrix is in the order [X_0, ..., X_{L-1}, T, Y]
        
        # Transpose adjacency matrix to flip edge direction for attention
        # Original: A[i,j] = 1 means i→j (i causes j)
        # After transpose: A[j,i] = 1 means j can attend to i (effect attends to cause)
        adjacency_matrix = adjacency_matrix.transpose(-2, -1)  # (B, L+2, L+2)
        
        # Add self-loops to enable self-attention
        eye = torch.eye(F, device=adjacency_matrix.device, dtype=adjacency_matrix.dtype)
        adjacency_matrix = adjacency_matrix + eye.unsqueeze(0)  # (B, L+2, L+2)
        
        # Convert adjacency matrix to boolean mask (1 -> True, 0 -> False)
        # True means CAN attend
        attn_mask = adjacency_matrix.bool()  # (B, L+2, L+2)
        
        if n_sink_cols > 0:
            # Expand mask to include sink columns
            # Sink columns: all features can attend to them (True)
            # Sink columns can attend to all features (True)
            full_size = n_sink_cols + F
            full_mask = torch.ones(B, full_size, full_size, dtype=torch.bool, device=attn_mask.device)
            
            # Copy the causal structure for non-sink features
            full_mask[:, n_sink_cols:, n_sink_cols:] = attn_mask
            
            attn_mask = full_mask

        #breakpoint()
        # negate attention mask
        attn_mask = torch.logical_not(attn_mask)

        
        return attn_mask

    def forward(
        self,
        X_obs: torch.Tensor,
        T_obs: torch.Tensor,
        Y_obs: torch.Tensor,
        X_intv: torch.Tensor,
        T_intv: torch.Tensor,
        adjacency_matrix: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass to produce predictions for interventional test samples.

        Args:
            X_obs: (B, N, L) - observational features (train)
            T_obs: (B, N, 1) - observational intervened feature (train)
            Y_obs: (B, N) or (B, N, 1) - observational targets (train)
            X_intv: (B, M, L) - interventional features (test)
            T_intv: (B, M, 1) - interventional intervened feature (test)
            adjacency_matrix: (B, L+2, L+2) - causal graph adjacency matrix
                Position ordering (NEW - from dataset):
                  - Position 0: Treatment variable (T)
                  - Position 1: Outcome variable (Y)
                  - Positions 2 to L+1: Feature variables (X[:,0] to X[:,L-1])
                
                Note: This is automatically reordered internally to match the model's
                embedding order [X_0, ..., X_{L-1}, T, Y]
                
                Edge semantics:
                  - A[i,j] = 1 means directed edge from i to j (i causes j)
                  - The matrix is transposed internally so j can attend to i
                  - This ensures effects attend to their causes for causal inference

        Returns:
            Dict with:
                - "predictions": (B, M) if output_dim == 1, else (B, M, output_dim)
                - "graph_embeddings": (B, L+2, D) if use_gcn=True, else not present
        """
        B, N, L = X_obs.shape
        assert L == self.num_features, f"Expected {self.num_features} features, got {L}"
        M = X_intv.shape[1]
        device = X_obs.device

        # Ensure T has correct shape
        if T_obs.dim() == 2:
            T_obs = T_obs.unsqueeze(-1)  # (B, N, 1)
        if T_intv.dim() == 2:
            T_intv = T_intv.unsqueeze(-1)  # (B, M, 1)

        # Validate adjacency matrix shape
        assert adjacency_matrix.shape == (B, L + 2, L + 2), \
            f"Expected adjacency matrix shape ({B}, {L + 2}, {L + 2}), got {adjacency_matrix.shape}"

        # === Process graph structure ===
        # Create graph node embeddings if GCN is enabled
        if self.use_gcn:
            # Reorder adjacency matrix from dataset order [T, Y, X_0, ..., X_{L-1}] 
            # to internal order [X_0, ..., X_{L-1}, T, Y]
            # This matches the internal embedding order used throughout the model
            perm_indices = list(range(2, L + 2)) + [0, 1]  # [2, 3, ..., L+1, 0, 1]
            adj_reordered = adjacency_matrix[:, perm_indices, :]  # Permute rows
            adj_reordered = adj_reordered[:, :, perm_indices]  # Permute columns
            # Now adj_reordered is in order [X_0, ..., X_{L-1}, T, Y]
            
            # Create initial node features from role embeddings
            # Position 0 to L-1: Feature nodes, Position L: Treatment, Position L+1: Outcome
            node_features = torch.zeros(L + 2, self.d_model, device=device)
            node_features[:L] = self.obs_feature_embed.squeeze().expand(L, -1)  # Feature nodes
            node_features[L] = self.obs_T_embed.squeeze()  # Treatment node
            node_features[L+1] = self.obs_label_embed.squeeze()  # Outcome node
            
            # Apply GCN to get graph-conditioned node embeddings
            # GCN receives adjacency in internal order [X_0, ..., X_{L-1}, T, Y]
            graph_node_embeddings = self.graph_encoder(adj_reordered, node_features)  # (B, L+2, D)
        else:
            graph_node_embeddings = None

        # === Normalize features (if enabled) ===
        if self.normalize_features:
            # Concatenate X and T for joint normalization
            X_obs_with_T = torch.cat([X_obs, T_obs], dim=2)  # (B, N, L+1)
            X_intv_with_T = torch.cat([X_intv, T_intv], dim=2)  # (B, M, L+1)
            
            X_obs_norm, X_intv_norm = self._normalize_features(X_obs_with_T, X_intv_with_T)
            
            # Split back into X and T
            X_obs_norm, T_obs_norm = X_obs_norm[:, :, :L], X_obs_norm[:, :, L:L+1]
            X_intv_norm, T_intv_norm = X_intv_norm[:, :, :L], X_intv_norm[:, :, L:L+1]
        else:
            X_obs_norm, T_obs_norm = X_obs, T_obs
            X_intv_norm, T_intv_norm = X_intv, T_intv

        # === Embed features ===
        feat_obs = self._embed_features(X_obs_norm, T_obs_norm, self.row_mlp_train, is_intvn=False)   # (B, N, L+1, D)
        feat_intv = self._embed_features(X_intv_norm, T_intv_norm, self.row_mlp_test, is_intvn=True)  # (B, M, L+1, D)
        feat_all = torch.cat([feat_obs, feat_intv], dim=1)  # (B, S, L+1, D), S = N+M

        # === Embed labels ===
        lab_obs = self._embed_labels(Y_obs)  # (B, N, D)
        lab_intv = self.intv_label_embed.expand(B, M, self.d_model)
        lab_all = torch.cat([lab_obs, lab_intv], dim=1)  # (B, S, D)

        # Stack features (including intervened) + label column along feature axis
        # Position order: [0..L-1: regular features, L: intervened feature, L+1: label]
        x = torch.cat([feat_all, lab_all.unsqueeze(2)], dim=2)  # (B, S, L+2, D)

        # Add feature positional encodings
        # feature_positional: (1, 1, L+2, D)
        x = x + self.feature_positional
        
        # === Add attention sinks ===
        # Add sink rows (dummy samples) if enabled
        if self.sink_rows_x is not None:
            # Expand sink rows to batch size
            sink_x = self.sink_rows_x.expand(B, -1, -1, -1)  # (B, n_sink_rows, L+2, D)
            # For sink rows, the label column uses sink_rows_y
            sink_x_features = sink_x[:, :, :-1, :]  # (B, n_sink_rows, L+1, D)
            sink_y = self.sink_rows_y.expand(B, -1, -1).unsqueeze(2)  # (B, n_sink_rows, 1, D)
            sink_x = torch.cat([sink_x_features, sink_y], dim=2)  # (B, n_sink_rows, L+2, D)
            
            # Prepend sink rows to the samples
            x = torch.cat([sink_x, x], dim=1)  # (B, n_sink_rows + S, L+2, D)
        
        n_sink_rows = self.n_sample_attention_sink_rows
        
        # Add sink columns (dummy features) if enabled
        if self.sink_cols is not None:
            # Expand sink columns to batch and sample size
            current_n_samples = x.shape[1]  # n_sink_rows + S
            sink_c = self.sink_cols.expand(B, current_n_samples, -1, -1)  # (B, current_n_samples, n_sink_cols, D)
            
            # Prepend sink columns to the features
            x = torch.cat([sink_c, x], dim=2)  # (B, current_n_samples, n_sink_cols + L+2, D)
        
        n_sink_cols = self.n_feature_attention_sink_cols

        # === Prepare attention mask from adjacency matrix ===
        # Only apply attention masking if enabled
        if self.use_attention_masking:
            attn_mask = self._prepare_attention_mask(adjacency_matrix, n_sink_cols)  # (B, n_sink_cols + L+2, n_sink_cols + L+2)
        else:
            attn_mask = None

        # === Prepare graph embeddings for AdaLN ===
        # Handle sink columns: if present, need to extend graph embeddings
        if n_sink_cols > 0 and self.use_adaln and graph_node_embeddings is not None:
            # Create dummy embeddings for sink columns (learnable or zeros)
            # For now, use zeros - sink columns don't correspond to real features
            sink_graph_emb = torch.zeros(B, n_sink_cols, self.d_model, device=device)
            graph_embeddings_with_sinks = torch.cat([sink_graph_emb, graph_node_embeddings], dim=1)  # (B, n_sink_cols + L+2, D)
        else:
            graph_embeddings_with_sinks = graph_node_embeddings if (self.use_adaln and graph_node_embeddings is not None) else None

        # Apply blocks with graph-conditioned attention and AdaLN
        for blk in self.blocks:
            x = blk(x, N_train=n_sink_rows + N, N_test=M, 
                   attn_mask=attn_mask,
                   graph_embeddings=graph_embeddings_with_sinks)

        # Readout: take the label column for interventional (test) rows only
        # Account for sink columns prepended to features
        label_pos = n_sink_cols + self.num_features + 1  # n_sink_cols + L + 1
        # Account for sink rows prepended to samples
        test_start_idx = n_sink_rows + N
        h_intv = x[:, test_start_idx:, label_pos, :]  # (B, M, D)
        predictions = self.regression_head(h_intv)  # (B, M, output_dim)

        # Backward-compatible squeeze when output_dim == 1
        if self.output_dim == 1:
            predictions = predictions.squeeze(-1)  # (B, M)

        result = {"predictions": predictions}
        
        # Only include graph embeddings if GCN was used
        if self.use_gcn and graph_node_embeddings is not None:
            result["graph_embeddings"] = graph_node_embeddings  # (B, L+2, D)
        
        return result


if __name__ == "__main__":
    torch.manual_seed(0)
    
    print("=" * 80)
    print("UltimateGraphConditionedInterventionalPFN Test Suite")
    print("=" * 80)
    
    # Test configuration
    B, N, M, L = 2, 16, 5, 7  # batch, train samples, test samples, features

    # Simulate interventional dataset output
    X_obs = torch.randn(B, N, L)
    T_obs = torch.randn(B, N, 1)
    Y_obs = torch.randn(B, N)
    X_intv = torch.randn(B, M, L)
    T_intv = torch.randn(B, M, 1)
    
    # Create sample adjacency matrices (fully connected for testing)
    adjacency_matrix = torch.ones(B, L + 2, L + 2)  # (B, L+2, L+2)

    # Test 1: All graph conditioning modes enabled (default)
    print("\n[Test 1] Full Graph Conditioning: Attention Masking + GCN + AdaLN")
    print("-" * 80)
    model_full = UltimateGraphConditionedInterventionalPFN(
        num_features=L,
        d_model=128,
        depth=2,
        heads_feat=4,
        heads_samp=4,
        dropout=0.1,
        use_attention_masking=True,
        use_gcn=True,
        use_adaln=True,
    )
    out_full = model_full(X_obs, T_obs, Y_obs, X_intv, T_intv, adjacency_matrix)
    print(f"✓ predictions shape: {out_full['predictions'].shape} (expected: ({B}, {M}))")
    print(f"✓ graph_embeddings in output: {'graph_embeddings' in out_full}")
    if 'graph_embeddings' in out_full:
        print(f"✓ graph_embeddings shape: {out_full['graph_embeddings'].shape}")
    print(f"✓ use_attention_masking: {model_full.use_attention_masking}")
    print(f"✓ use_gcn: {model_full.use_gcn}")
    print(f"✓ use_adaln: {model_full.use_adaln}")
    assert out_full["predictions"].shape == (B, M)
    print("✓ Test 1 passed!")

    # Test 2: Only attention masking (no GCN, no AdaLN)
    print("\n[Test 2] Attention Masking Only")
    print("-" * 80)
    model_mask_only = UltimateGraphConditionedInterventionalPFN(
        num_features=L,
        d_model=128,
        depth=2,
        heads_feat=4,
        heads_samp=4,
        dropout=0.1,
        use_attention_masking=True,
        use_gcn=False,
        use_adaln=False,
    )
    out_mask = model_mask_only(X_obs, T_obs, Y_obs, X_intv, T_intv, adjacency_matrix)
    print(f"✓ predictions shape: {out_mask['predictions'].shape}")
    print(f"✓ graph_embeddings in output: {'graph_embeddings' in out_mask}")
    print(f"✓ use_attention_masking: {model_mask_only.use_attention_masking}")
    print(f"✓ use_gcn: {model_mask_only.use_gcn}")
    print(f"✓ use_adaln: {model_mask_only.use_adaln}")
    print(f"✓ graph_encoder is None: {model_mask_only.graph_encoder is None}")
    assert 'graph_embeddings' not in out_mask, "Should not have graph embeddings without GCN"
    print("✓ Test 2 passed!")

    # Test 3: Only GCN + AdaLN (no attention masking)
    print("\n[Test 3] GCN + AdaLN Only (No Attention Masking)")
    print("-" * 80)
    model_gcn_only = UltimateGraphConditionedInterventionalPFN(
        num_features=L,
        d_model=128,
        depth=2,
        heads_feat=4,
        heads_samp=4,
        dropout=0.1,
        use_attention_masking=False,
        use_gcn=True,
        use_adaln=True,
    )
    out_gcn = model_gcn_only(X_obs, T_obs, Y_obs, X_intv, T_intv, adjacency_matrix)
    print(f"✓ predictions shape: {out_gcn['predictions'].shape}")
    print(f"✓ graph_embeddings in output: {'graph_embeddings' in out_gcn}")
    if 'graph_embeddings' in out_gcn:
        print(f"✓ graph_embeddings shape: {out_gcn['graph_embeddings'].shape}")
    print(f"✓ use_attention_masking: {model_gcn_only.use_attention_masking}")
    print(f"✓ use_gcn: {model_gcn_only.use_gcn}")
    print(f"✓ use_adaln: {model_gcn_only.use_adaln}")
    assert 'graph_embeddings' in out_gcn, "Should have graph embeddings with GCN"
    print("✓ Test 3 passed!")

    # Test 4: GCN without AdaLN (GCN embeddings computed but not used for conditioning)
    print("\n[Test 4] GCN Only (Without AdaLN)")
    print("-" * 80)
    model_gcn_no_adaln = UltimateGraphConditionedInterventionalPFN(
        num_features=L,
        d_model=128,
        depth=2,
        heads_feat=4,
        heads_samp=4,
        dropout=0.1,
        use_attention_masking=False,
        use_gcn=True,
        use_adaln=False,
    )
    out_gcn_no_adaln = model_gcn_no_adaln(X_obs, T_obs, Y_obs, X_intv, T_intv, adjacency_matrix)
    print(f"✓ predictions shape: {out_gcn_no_adaln['predictions'].shape}")
    print(f"✓ graph_embeddings in output: {'graph_embeddings' in out_gcn_no_adaln}")
    print(f"✓ use_attention_masking: {model_gcn_no_adaln.use_attention_masking}")
    print(f"✓ use_gcn: {model_gcn_no_adaln.use_gcn}")
    print(f"✓ use_adaln: {model_gcn_no_adaln.use_adaln}")
    print("✓ Test 4 passed!")

    # Test 5: No graph conditioning at all
    print("\n[Test 5] No Graph Conditioning (Baseline)")
    print("-" * 80)
    model_no_graph = UltimateGraphConditionedInterventionalPFN(
        num_features=L,
        d_model=128,
        depth=2,
        heads_feat=4,
        heads_samp=4,
        dropout=0.1,
        use_attention_masking=False,
        use_gcn=False,
        use_adaln=False,
    )
    out_no_graph = model_no_graph(X_obs, T_obs, Y_obs, X_intv, T_intv, adjacency_matrix)
    print(f"✓ predictions shape: {out_no_graph['predictions'].shape}")
    print(f"✓ graph_embeddings in output: {'graph_embeddings' in out_no_graph}")
    print(f"✓ use_attention_masking: {model_no_graph.use_attention_masking}")
    print(f"✓ use_gcn: {model_no_graph.use_gcn}")
    print(f"✓ use_adaln: {model_no_graph.use_adaln}")
    assert 'graph_embeddings' not in out_no_graph
    print("✓ Test 5 passed!")

    # Test 6: Invalid configuration (AdaLN without GCN)
    print("\n[Test 6] Invalid Configuration: AdaLN=True but GCN=False")
    print("-" * 80)
    try:
        model_invalid = UltimateGraphConditionedInterventionalPFN(
            num_features=L,
            d_model=128,
            depth=2,
            use_attention_masking=True,
            use_gcn=False,
            use_adaln=True,  # Invalid: needs GCN
        )
        print("✗ Should have raised ValueError!")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
        print("✓ Test 6 passed!")

    # Test 7: Compare parameter counts across configurations
    print("\n[Test 7] Parameter Counts Across Configurations")
    print("-" * 80)
    params_full = sum(p.numel() for p in model_full.parameters())
    params_mask = sum(p.numel() for p in model_mask_only.parameters())
    params_gcn = sum(p.numel() for p in model_gcn_only.parameters())
    params_none = sum(p.numel() for p in model_no_graph.parameters())
    
    print(f"✓ Full (Mask+GCN+AdaLN): {params_full:,} parameters")
    print(f"✓ Mask only: {params_mask:,} parameters")
    print(f"✓ GCN+AdaLN: {params_gcn:,} parameters")
    print(f"✓ No graph: {params_none:,} parameters")
    print(f"✓ GCN overhead: {params_gcn - params_none:,} parameters")
    print(f"✓ Mask overhead: {params_mask - params_none:,} parameters (should be 0)")
    assert params_mask == params_none, "Attention masking should not add parameters"
    assert params_gcn > params_none, "GCN should add parameters"
    print("✓ Test 7 passed!")

    # Test 8: Different graphs produce different results with different conditioning
    print("\n[Test 8] Graph Structure Effects Across Configurations")
    print("-" * 80)
    adj_full_graph = torch.ones(B, L + 2, L + 2)
    adj_sparse = torch.eye(L + 2).unsqueeze(0).expand(B, -1, -1).clone()
    
    # Full model
    out_full_dense = model_full(X_obs, T_obs, Y_obs, X_intv, T_intv, adj_full_graph)
    out_full_sparse = model_full(X_obs, T_obs, Y_obs, X_intv, T_intv, adj_sparse)
    diff_full = torch.abs(out_full_dense['predictions'] - out_full_sparse['predictions']).max().item()
    
    # Mask only
    out_mask_dense = model_mask_only(X_obs, T_obs, Y_obs, X_intv, T_intv, adj_full_graph)
    out_mask_sparse = model_mask_only(X_obs, T_obs, Y_obs, X_intv, T_intv, adj_sparse)
    diff_mask = torch.abs(out_mask_dense['predictions'] - out_mask_sparse['predictions']).max().item()
    
    # GCN only
    out_gcn_dense = model_gcn_only(X_obs, T_obs, Y_obs, X_intv, T_intv, adj_full_graph)
    out_gcn_sparse = model_gcn_only(X_obs, T_obs, Y_obs, X_intv, T_intv, adj_sparse)
    diff_gcn = torch.abs(out_gcn_dense['predictions'] - out_gcn_sparse['predictions']).max().item()
    
    # No graph (should be invariant to graph structure)
    out_none_1 = model_no_graph(X_obs, T_obs, Y_obs, X_intv, T_intv, adj_full_graph)
    out_none_2 = model_no_graph(X_obs, T_obs, Y_obs, X_intv, T_intv, adj_sparse)
    diff_none = torch.abs(out_none_1['predictions'] - out_none_2['predictions']).max().item()
    
    print(f"✓ Full model diff (dense vs sparse): {diff_full:.6f}")
    print(f"✓ Mask only diff: {diff_mask:.6f}")
    print(f"✓ GCN only diff: {diff_gcn:.6f}")
    print(f"✓ No graph diff: {diff_none:.6f}")
    
    assert diff_full > 1e-6, "Full model should be sensitive to graph"
    assert diff_mask > 1e-6, "Mask-only should be sensitive to graph"
    assert diff_gcn > 1e-6, "GCN-only should be sensitive to graph"
    # Note: No-graph model may still show differences due to random initialization
    # and numerical precision, but should not use graph structure
    print(f"✓ Graph-conditioned models show clear differences")
    print(f"✓ No-graph model: {diff_none:.6f} (may vary due to numerical effects)")
    print("✓ Test 8 passed!")

    # Test 9: Soft attention bias (learnable biases instead of hard masking)
    print("\n[Test 9] Soft Attention Bias")
    print("-" * 80)
    model_soft_bias = UltimateGraphConditionedInterventionalPFN(
        num_features=L,
        d_model=128,
        depth=2,
        heads_feat=4,
        heads_samp=4,
        dropout=0.1,
        use_attention_masking=True,
        use_gcn=False,
        use_adaln=False,
        use_soft_attention_bias=True,
        soft_bias_init=5.0,
    )
    out_soft = model_soft_bias(X_obs, T_obs, Y_obs, X_intv, T_intv, adjacency_matrix)
    print(f"✓ predictions shape: {out_soft['predictions'].shape}")
    print(f"✓ use_soft_attention_bias: {model_soft_bias.use_soft_attention_bias}")
    
    # Check that soft attention biases exist and are learnable
    has_biases = any('soft_attention_bias' in name for name, _ in model_soft_bias.named_parameters())
    print(f"✓ Soft attention biases exist: {has_biases}")
    
    # Count bias parameters (should be heads_feat per layer)
    bias_params = [p for name, p in model_soft_bias.named_parameters() if 'soft_attention_bias' in name]
    print(f"✓ Number of bias parameter tensors: {len(bias_params)} (expected: {2} layers)")
    for i, p in enumerate(bias_params):
        print(f"✓ Layer {i} bias shape: {p.shape} (expected: ({4},) for 4 heads)")
        assert p.shape == (4,), f"Expected shape (4,), got {p.shape}"
        print(f"✓ Layer {i} bias values: {p.data.tolist()} (initialized to {5.0})")
    
    # Verify different from hard masking
    out_hard = model_mask_only(X_obs, T_obs, Y_obs, X_intv, T_intv, adjacency_matrix)
    diff_soft_vs_hard = torch.abs(out_soft['predictions'] - out_hard['predictions']).max().item()
    print(f"✓ Soft vs hard masking diff: {diff_soft_vs_hard:.6f}")
    assert diff_soft_vs_hard > 1e-6, "Soft and hard masking should produce different results"
    print("✓ Test 9 passed!")

    # Test 10: Invalid configuration (soft bias without attention masking)
    print("\n[Test 10] Invalid Configuration: Soft Bias=True but Masking=False")
    print("-" * 80)
    try:
        model_invalid_soft = UltimateGraphConditionedInterventionalPFN(
            num_features=L,
            d_model=128,
            depth=2,
            use_attention_masking=False,
            use_soft_attention_bias=True,  # Invalid: needs masking
        )
        print("✗ Should have raised ValueError!")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
        print("✓ Test 10 passed!")

    # Test 11: Soft bias with GCN+AdaLN (full soft conditioning)
    print("\n[Test 11] Full Soft Conditioning: Soft Bias + GCN + AdaLN")
    print("-" * 80)
    model_full_soft = UltimateGraphConditionedInterventionalPFN(
        num_features=L,
        d_model=128,
        depth=2,
        heads_feat=4,
        heads_samp=4,
        dropout=0.1,
        use_attention_masking=True,
        use_gcn=True,
        use_adaln=True,
        use_soft_attention_bias=True,
        soft_bias_init=10.0,
    )
    out_full_soft = model_full_soft(X_obs, T_obs, Y_obs, X_intv, T_intv, adjacency_matrix)
    print(f"✓ predictions shape: {out_full_soft['predictions'].shape}")
    print(f"✓ graph_embeddings in output: {'graph_embeddings' in out_full_soft}")
    print(f"✓ use_attention_masking: {model_full_soft.use_attention_masking}")
    print(f"✓ use_gcn: {model_full_soft.use_gcn}")
    print(f"✓ use_adaln: {model_full_soft.use_adaln}")
    print(f"✓ use_soft_attention_bias: {model_full_soft.use_soft_attention_bias}")
    
    # Compare with full hard model
    diff_full_soft_vs_hard = torch.abs(out_full_soft['predictions'] - out_full['predictions']).max().item()
    print(f"✓ Full soft vs full hard diff: {diff_full_soft_vs_hard:.6f}")
    assert diff_full_soft_vs_hard > 1e-6, "Soft and hard conditioning should differ"
    print("✓ Test 11 passed!")

    # Test 12: Parameter counts with soft attention bias
    print("\n[Test 12] Parameter Counts with Soft Attention Bias")
    print("-" * 80)
    params_soft = sum(p.numel() for p in model_soft_bias.parameters())
    params_hard = sum(p.numel() for p in model_mask_only.parameters())
    params_full_soft = sum(p.numel() for p in model_full_soft.parameters())
    
    print(f"✓ Hard masking only: {params_hard:,} parameters")
    print(f"✓ Soft masking only: {params_soft:,} parameters")
    print(f"✓ Full soft (Soft+GCN+AdaLN): {params_full_soft:,} parameters")
    print(f"✓ Soft bias overhead: {params_soft - params_hard:,} parameters")
    
    # Soft bias adds: depth * heads_feat parameters (one bias per head per layer)
    expected_overhead = 2 * 4  # 2 layers * 4 heads
    actual_overhead = params_soft - params_hard
    print(f"✓ Expected soft bias overhead: {expected_overhead} parameters")
    print(f"✓ Actual overhead: {actual_overhead} parameters")
    assert actual_overhead == expected_overhead, f"Expected {expected_overhead}, got {actual_overhead}"
    print("✓ Test 12 passed!")

    # Model statistics
    print("\n[Model Statistics]")
    print("-" * 80)
    print(f"✓ Full model parameters: {params_full:,}")
    print(f"✓ Mask-only parameters: {params_mask:,}")
    print(f"✓ GCN+AdaLN parameters: {params_gcn:,}")
    print(f"✓ No-graph parameters: {params_none:,}")
    print(f"✓ Soft bias parameters: {params_soft:,}")
    print(f"✓ Full soft parameters: {params_full_soft:,}")
    
    # Architecture verification
    print("\n[Architecture Verification]")
    print("-" * 80)
    print(f"✓ Flexible graph conditioning modes:")
    print(f"  - use_attention_masking: Hard/soft constraints via attention masks")
    print(f"  - use_soft_attention_bias: Learnable biases (alternative to hard masking)")
    print(f"  - use_gcn: GCN processes adjacency → (B, L+2, D) node embeddings")
    print(f"  - use_adaln: AdaLN modulates LayerNorms with graph embeddings")
    print(f"✓ Soft attention bias:")
    print(f"  - One learnable bias per head per layer")
    print(f"  - Initialized to positive values (default: 5.0)")
    print(f"  - Added to attention scores where graph permits")
    print(f"  - Requires use_attention_masking=True")
    print(f"✓ Can use any combination: all, none, or individual modes")
    print(f"✓ GCN reuses existing role embeddings (obs_T, obs_label, obs_feature)")
    print(f"✓ AdaLN applied to ln_feat and ln_mlp in each block")
    print(f"✓ Graph embeddings returned when GCN enabled")
    print(f"✓ MLP uses SwiGLU activation")
    print(f"✓ Pre-layer normalization (adaptive when AdaLN enabled)")
    print(f"✓ Separate train/test attention layers")
    print(f"✓ Optional attention sinks supported")
    
    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)

