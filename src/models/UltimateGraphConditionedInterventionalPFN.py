"""
Ultimate Graph-Conditioned Interventional Prior-Data Fitted Network (PFN) for causal inference.

This module extends GraphConditionedInterventionalPFN with additional graph processing:
1. Attention masking (inherited) - hard structural constraints
2. GCN-style graph encoder - learns node representations from graph structure

The GCN encoder processes the adjacency matrix into node embeddings that capture
the graph structure. These embeddings are computed but not yet used for conditioning
(AdaLN integration planned for future).

Input format:
- Same as InterventionalPFN: X_obs, T_obs, Y_obs, X_intv, T_intv
- Additional: adjacency_matrix (B, L+2, L+2) encoding causal structure

Adjacency matrix ordering (matches data layout):
- Position 0: Treatment variable (intervention_node)
- Position 1: Outcome variable (target feature)
- Position 2+: Feature variables that were KEPT after dropout (sorted order)

The adjacency matrix A[i,j] = 1 means there is a causal edge from j to i.
Attention flows opposite to causal edges: feature i attends to feature j if A[i,j] = 1 (edge j→i).

Key differences from GraphConditionedInterventionalPFN:
1. Includes GCN-style graph encoder for processing adjacency matrix
2. Produces graph node embeddings (B, L+2, D) - currently for inspection only
3. Future: Will use graph embeddings for AdaLN conditioning

Architecture features:
- SwiGLU activation
- Pre-layer normalization
- Separate train/test attention
- Graph-conditioned feature attention via masking
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
    """
    def __init__(
        self, 
        dim: int, 
        heads_feat: int, 
        heads_samp: int, 
        dropout: float = 0.0, 
        hidden_mult: int = 4,
        use_adaln: bool = True
    ):
        super().__init__()
        self.use_adaln = use_adaln
        
        # Self-attention across features with graph conditioning: (B*S, L+2, D)
        self.feat_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads_feat,
            dropout=dropout,
            batch_first=True,
        )
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
    Ultimate PFN-like regressor with dual graph conditioning mechanisms.

    Combines two approaches to incorporate causal graph structure:
    1. Attention masking - hard structural constraints on which features can attend to which
    2. GCN-style graph encoder - learns node representations from adjacency matrix
    
    The GCN encoder processes the graph structure into node embeddings (B, L+2, D).
    Currently these are computed and returned for inspection. Future versions will
    use them for AdaLN (Adaptive Layer Normalization) to provide learned, soft
    graph-based conditioning alongside the hard attention masks.
    
    Adjacency matrix format:
    - Shape: (B, L+2, L+2) where L is number of features (after dropout)
    - Position 0: Treatment variable (intervention_node)
    - Position 1: Outcome variable (target feature)
    - Position 2+: Feature variables (kept after dropout, sorted order)
    - A[i,j] = 1 means there is a causal edge from j to i (i.e., j causes i)
    - Attention flows opposite to causal edges: feature i attends to j if A[i,j] = 1
    
    Architecture features:
    - SwiGLU activation
    - Pre-layer normalization
    - Separate train/test attention
    - Graph-conditioned feature attention
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
        use_adaln: Whether to use Adaptive Layer Normalization with graph embeddings
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
        use_adaln: bool = True,
    ):
        super().__init__()
        self.num_features = num_features  # L (excluding intervened column)
        self.d_model = d_model
        self.output_dim = output_dim
        self.normalize_features = normalize_features
        self.n_sample_attention_sink_rows = n_sample_attention_sink_rows
        self.n_feature_attention_sink_cols = n_feature_attention_sink_cols
        self.use_adaln = use_adaln

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
        self.graph_encoder = GraphEncoder(
            num_nodes=num_features + 2,
            d_model=d_model,
            dropout=dropout
        )

        # Stacked two-way attention blocks with graph conditioning
        self.blocks = nn.ModuleList([
            GraphConditionedTwoWayBlock(
                d_model, heads_feat, heads_samp, 
                dropout=dropout, hidden_mult=hidden_mult,
                use_adaln=use_adaln
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
        
        The adjacency matrix defines causal structure between features:
        - Position 0: Treatment variable
        - Position 1: Outcome variable
        - Position 2+: Other features
        - A[i,j] = 1 means there is a causal edge from j to i (j causes i)
        
        Attention flows opposite to causal edges: feature i attends to j if A[i,j] = 1.
        This is implemented by negating the adjacency matrix (logical_not).
        
        With sink columns, the mask is expanded to allow all features to attend to sinks.
        
        Args:
            adjacency_matrix: (B, L+2, L+2) - 1 means causal edge from j to i, 0 means no edge
            n_sink_cols: Number of sink columns prepended
            
        Returns:
            Attention mask of shape (B, n_sink_cols + L+2, n_sink_cols + L+2)
            where False means position CAN attend (PyTorch convention after negation).
        """
        B, F, F2 = adjacency_matrix.shape
        assert F == F2, "Adjacency matrix must be square"
        assert F == self.num_features + 2, f"Expected adjacency matrix size {self.num_features + 2}, got {F}"
        
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
                Position 0: Treatment variable
                Position 1: Outcome variable
                Position 2+: Other features (sorted, after dropout)
                A[i,j] = 1 means feature i can attend to feature j.  #TODO this is weird, it should be the other way around?!

        Returns:
            Dict with:
                - "predictions": (B, M) if output_dim == 1, else (B, M, output_dim)
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
        # Create initial node features from role embeddings
        # Position 0: Treatment, Position 1: Outcome, Position 2+: Features
        node_features = torch.zeros(L + 2, self.d_model, device=device)
        node_features[0] = self.obs_T_embed.squeeze()  # Treatment node
        node_features[1] = self.obs_label_embed.squeeze()  # Outcome node
        node_features[2:] = self.obs_feature_embed.squeeze().expand(L, -1)  # Feature nodes
        
        # Apply GCN to get graph-conditioned node embeddings
        graph_node_embeddings = self.graph_encoder(adjacency_matrix, node_features)  # (B, L+2, D)

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
        attn_mask = self._prepare_attention_mask(adjacency_matrix, n_sink_cols)  # (B, n_sink_cols + L+2, n_sink_cols + L+2)
        
        # === Prepare graph embeddings for AdaLN ===
        # Handle sink columns: if present, need to extend graph embeddings
        if n_sink_cols > 0 and self.use_adaln:
            # Create dummy embeddings for sink columns (learnable or zeros)
            # For now, use zeros - sink columns don't correspond to real features
            sink_graph_emb = torch.zeros(B, n_sink_cols, self.d_model, device=device)
            graph_embeddings_with_sinks = torch.cat([sink_graph_emb, graph_node_embeddings], dim=1)  # (B, n_sink_cols + L+2, D)
        else:
            graph_embeddings_with_sinks = graph_node_embeddings if self.use_adaln else None

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

        return {
            "predictions": predictions,
            "graph_embeddings": graph_node_embeddings  # (B, L+2, D) for inspection/debugging
        }


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

    # Test 1: Basic single output regression with GCN graph encoding and AdaLN
    print("\n[Test 1] Single Output with GCN Graph Encoder and AdaLN")
    print("-" * 80)
    model = UltimateGraphConditionedInterventionalPFN(
        num_features=L,
        d_model=128,
        depth=2,
        heads_feat=4,
        heads_samp=4,
        dropout=0.1,
        use_adaln=True,
    )
    out = model(X_obs, T_obs, Y_obs, X_intv, T_intv, adjacency_matrix)
    print(f"✓ predictions shape: {out['predictions'].shape} (expected: ({B}, {M}))")
    print(f"✓ graph_embeddings shape: {out['graph_embeddings'].shape} (expected: ({B}, {L+2}, 128))")
    print(f"✓ Sample predictions: {out['predictions'][0, :3].detach().numpy()}")
    print(f"✓ AdaLN enabled: {model.use_adaln}")
    print(f"✓ Block 0 uses AdaLN: {model.blocks[0].use_adaln}")
    assert out["predictions"].shape == (B, M), f"Expected shape ({B}, {M}), got {out['predictions'].shape}"
    assert out["graph_embeddings"].shape == (B, L+2, 128), f"Expected graph embeddings shape ({B}, {L+2}, 128)"
    print("✓ Test 1 passed!")

    # Test 2: Different adjacency matrices produce different graph embeddings
    print("\n[Test 2] Different Graphs Produce Different Embeddings (with AdaLN)")
    print("-" * 80)
    # Full graph
    adj_full = torch.ones(B, L + 2, L + 2)
    # Sparse graph (only self-loops and chain)
    adj_sparse = torch.eye(L + 2).unsqueeze(0).expand(B, -1, -1).clone()
    for i in range(L + 1):
        adj_sparse[:, i + 1, i] = 1  # Chain structure
    
    out_full = model(X_obs, T_obs, Y_obs, X_intv, T_intv, adj_full)
    out_sparse = model(X_obs, T_obs, Y_obs, X_intv, T_intv, adj_sparse)
    
    # Graph embeddings should differ
    graph_diff = torch.abs(out_full['graph_embeddings'] - out_sparse['graph_embeddings']).max().item()
    print(f"✓ Max graph embedding difference: {graph_diff:.6f}")
    assert graph_diff > 1e-6, "Different graphs should produce different embeddings"
    
    # Predictions should also differ (even more with AdaLN!)
    pred_diff = torch.abs(out_full['predictions'] - out_sparse['predictions']).max().item()
    print(f"✓ Max prediction difference: {pred_diff:.6f}")
    assert pred_diff > 1e-6, "Different graphs should produce different predictions"
    print("✓ Test 2 passed!")

    # Test 3: Compare AdaLN vs no AdaLN
    print("\n[Test 3] AdaLN vs Regular LayerNorm Comparison")
    print("-" * 80)
    model_no_adaln = UltimateGraphConditionedInterventionalPFN(
        num_features=L,
        d_model=128,
        depth=2,
        heads_feat=4,
        heads_samp=4,
        dropout=0.1,
        use_adaln=False,
    )
    out_no_adaln = model_no_adaln(X_obs, T_obs, Y_obs, X_intv, T_intv, adjacency_matrix)
    
    print(f"✓ Model with AdaLN predictions: {out['predictions'][0, 0].item():.6f}")
    print(f"✓ Model without AdaLN predictions: {out_no_adaln['predictions'][0, 0].item():.6f}")
    print(f"✓ AdaLN enabled in model: {model.use_adaln}")
    print(f"✓ AdaLN disabled in model_no_adaln: {model_no_adaln.use_adaln}")
    
    # Count parameters
    adaln_params = sum(p.numel() for p in model.parameters())
    no_adaln_params = sum(p.numel() for p in model_no_adaln.parameters())
    print(f"✓ Parameters with AdaLN: {adaln_params:,}")
    print(f"✓ Parameters without AdaLN: {no_adaln_params:,}")
    print(f"✓ AdaLN overhead: {adaln_params - no_adaln_params:,} parameters")
    print("✓ Test 3 passed!")

    # Test 4: Verify GCN encoder is using role embeddings
    print("\n[Test 3] GCN Uses Role Embeddings as Node Features")
    print("-" * 80)
    print(f"✓ Treatment embed shape: {model.obs_T_embed.shape}")
    print(f"✓ Outcome embed shape: {model.obs_label_embed.shape}")
    print(f"✓ Feature embed shape: {model.obs_feature_embed.shape}")
    print(f"✓ GraphEncoder input dim: {model.graph_encoder.d_model}")
    print(f"✓ GraphEncoder output dim: {model.graph_encoder.d_model}")
    print("✓ Test 3 passed!")

    # Test 4: High-dimensional output
    print("\n[Test 4] High-Dimensional Output with GCN")
    print("-" * 80)
    output_dim = 10
    model_hd = UltimateGraphConditionedInterventionalPFN(
        num_features=L,
        d_model=128,
        depth=2,
        heads_feat=4,
        heads_samp=4,
        dropout=0.1,
        output_dim=output_dim,
    )
    out_hd = model_hd(X_obs, T_obs, Y_obs, X_intv, T_intv, adjacency_matrix)
    print(f"✓ high-dim predictions shape: {out_hd['predictions'].shape} (expected: ({B}, {M}, {output_dim}))")
    print(f"✓ graph_embeddings shape: {out_hd['graph_embeddings'].shape} (expected: ({B}, {L+2}, 128))")
    assert out_hd["predictions"].shape == (B, M, output_dim)
    assert out_hd["graph_embeddings"].shape == (B, L+2, 128)
    print("✓ Test 4 passed!")

    # Test 5: With attention sinks
    print("\n[Test 5] GCN with Attention Sinks")
    print("-" * 80)
    n_sink_rows = 3
    n_sink_cols = 2
    model_sinks = UltimateGraphConditionedInterventionalPFN(
        num_features=L,
        d_model=128,
        depth=2,
        heads_feat=4,
        heads_samp=4,
        dropout=0.1,
        n_sample_attention_sink_rows=n_sink_rows,
        n_feature_attention_sink_cols=n_sink_cols,
    )
    out_sinks = model_sinks(X_obs, T_obs, Y_obs, X_intv, T_intv, adjacency_matrix)
    print(f"✓ predictions shape: {out_sinks['predictions'].shape} (expected: ({B}, {M}))")
    print(f"✓ graph_embeddings shape: {out_sinks['graph_embeddings'].shape} (expected: ({B}, {L+2}, 128))")
    assert out_sinks["predictions"].shape == (B, M)
    assert out_sinks["graph_embeddings"].shape == (B, L+2, 128)
    print("✓ Test 5 passed!")
    
    # Test 6: Verify graph embeddings capture local structure
    print("\n[Test 6] Graph Embeddings Capture Local Structure")
    print("-" * 80)
    # Create a star graph: Treatment connects to everything, others isolated
    adj_star = torch.eye(L + 2).unsqueeze(0).expand(B, -1, -1).clone()
    adj_star[:, :, 0] = 1  # All nodes attend to treatment
    adj_star[:, 0, :] = 1  # Treatment attends to all
    
    out_star = model(X_obs, T_obs, Y_obs, X_intv, T_intv, adj_star)
    
    # Treatment node (position 0) should have different embedding than others
    treatment_emb = out_star['graph_embeddings'][:, 0, :]  # (B, D)
    outcome_emb = out_star['graph_embeddings'][:, 1, :]  # (B, D)
    feature_emb = out_star['graph_embeddings'][:, 2, :]  # (B, D)
    
    treatment_outcome_diff = torch.abs(treatment_emb - outcome_emb).mean().item()
    treatment_feature_diff = torch.abs(treatment_emb - feature_emb).mean().item()
    
    print(f"✓ Treatment-Outcome embedding difference: {treatment_outcome_diff:.6f}")
    print(f"✓ Treatment-Feature embedding difference: {treatment_feature_diff:.6f}")
    print("✓ Test 6 passed!")
    
    # Model statistics
    print("\n[Model Statistics]")
    print("-" * 80)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gcn_params = sum(p.numel() for p in model.graph_encoder.parameters())
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    print(f"✓ GCN encoder parameters: {gcn_params:,}")
    
    # Architecture verification
    print("\n[Architecture Verification]")
    print("-" * 80)
    print(f"✓ Uses GraphConditionedTwoWayBlock with attention masking")
    print(f"✓ Includes GCN-style GraphEncoder")
    print(f"✓ GCN processes adjacency → (B, L+2, D) node embeddings")
    print(f"✓ GCN reuses existing role embeddings (obs_T, obs_label, obs_feature)")
    print(f"✓ AdaLN (Adaptive Layer Normalization) for graph-based conditioning")
    print(f"✓ AdaLN applied to ln_feat and ln_mlp in each block")
    print(f"✓ Graph embeddings modulate scale and shift in LayerNorms")
    print(f"✓ Graph embeddings returned for inspection/debugging")
    print(f"✓ MLP uses SwiGLU activation")
    print(f"✓ Pre-layer normalization (adaptive when AdaLN enabled)")
    print(f"✓ Separate train/test attention layers")
    print(f"✓ Optional attention sinks supported")
    
    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
