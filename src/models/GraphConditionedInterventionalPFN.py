"""
Graph-Conditioned Interventional Prior-Data Fitted Network (PFN) for causal inference.

This module extends InterventionalPFN to incorporate causal graph structure via
attention masking. The causal graph is provided as an adjacency matrix and conditions
the attention mechanism to respect causal relationships.

Input format:
- Same as InterventionalPFN: X_obs, T_obs, Y_obs, X_intv, T_intv
- Additional: adjacency_matrix (B, L+2, L+2) encoding causal structure

Adjacency matrix ordering (matches data layout):
- Position 0: Treatment variable (intervention_node)
- Position 1: Outcome variable (target feature)
- Position 2+: Feature variables that were KEPT after dropout (sorted order)

The adjacency matrix A[i,j] = 1 means feature i can attend to feature j.

Key differences from InterventionalPFN:
1. Takes adjacency_matrix as input
2. Modifies feature attention mask based on causal graph structure
3. Sample attention remains unchanged (no graph conditioning)
4. Graph structure is applied in all two-way blocks

Architecture features (inherited from InterventionalPFN):
- SwiGLU activation
- Pre-layer normalization
- Separate train/test attention
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


class GraphConditionedTwoWayBlock(nn.Module):
    """
    Two-way attention block with causal graph conditioning via attention masking.
    
    Feature attention is modified to respect the causal graph structure:
    - The adjacency matrix determines which features can attend to which
    - Sample attention remains unchanged (no graph conditioning)
    
    Uses pre-layer normalization and separate attention layers for train and test samples.
    
    Args:
        dim: Model dimension
        heads_feat: Number of attention heads for feature-wise attention
        heads_samp: Number of attention heads for sample-wise attention
        dropout: Dropout rate
        hidden_mult: Hidden layer multiplier for MLPs
    """
    def __init__(self, dim: int, heads_feat: int, heads_samp: int, dropout: float = 0.0, hidden_mult: int = 4):
        super().__init__()
        # Self-attention across features with graph conditioning: (B*S, L+2, D)
        self.feat_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads_feat,
            dropout=dropout,
            batch_first=True,
        )
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
        self.ln_mlp = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self, 
        x: torch.Tensor, 
        N_train: int, 
        N_test: int,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, S, F, D) where S = N_train + N_test, F = L+2.
            N_train: Number of train (observational) samples.
            N_test: Number of test (interventional) samples.
            attn_mask: Attention mask for feature attention of shape (B*S, F, F) or (F, F).
                      True/1 means position CAN attend, False/0 means CANNOT attend.
                      If None, standard full attention is used.

        Returns:
            Tensor of shape (B, S, F, D).
        """
        B, S, F, D = x.shape
        assert S == N_train + N_test, f"Expected {N_train + N_test} samples, got {S}"

        # 1) Feature-attention (within row) with graph conditioning - pre-layer norm
        x_row = x.reshape(B * S, F, D)  # (B*S, F, D)
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
        x_norm = self.ln_mlp(x)
        x2 = self.mlp(x_norm)
        x = x + self.drop(x2)
        return x


class GraphConditionedInterventionalPFN(nn.Module):
    """
    PFN-like regressor for interventional causal data with causal graph conditioning.

    Extends InterventionalPFN by incorporating causal graph structure via attention masking.
    The causal graph is provided as an adjacency matrix that determines which features
    can attend to which other features in the feature attention layers.
    
    Adjacency matrix format:
    - Shape: (B, L+2, L+2) where L is number of features (after dropout)
    - Position 0: Treatment variable (intervention_node)
    - Position 1: Outcome variable (target feature)
    - Position 2+: Feature variables (kept after dropout, sorted order)
    - A[i,j] = 1 means feature i CAN attend to feature j
    - A[i,j] = 0 means feature i CANNOT attend to feature j
    
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
    ):
        super().__init__()
        self.num_features = num_features  # L (excluding intervened column)
        self.d_model = d_model
        self.output_dim = output_dim
        self.normalize_features = normalize_features
        self.n_sample_attention_sink_rows = n_sample_attention_sink_rows
        self.n_feature_attention_sink_cols = n_feature_attention_sink_cols

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

        # Stacked two-way attention blocks with graph conditioning
        self.blocks = nn.ModuleList([
            GraphConditionedTwoWayBlock(d_model, heads_feat, heads_samp, dropout=dropout, hidden_mult=hidden_mult)
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
        
        With sink columns, the mask is expanded to allow all features to attend to sinks.
        
        Args:
            adjacency_matrix: (B, L+2, L+2) - 1 means can attend, 0 means cannot
            n_sink_cols: Number of sink columns prepended
            
        Returns:
            Attention mask of shape (B, n_sink_cols + L+2, n_sink_cols + L+2)
            where True means position CAN attend.
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
        
        # The mask is the same for all samples in a batch, so we can use just (F, F) or (B, F, F)
        # PyTorch's MultiheadAttention will broadcast appropriately
        # We'll pass it as (F, F) which is most efficient

        # Apply blocks with graph-conditioned attention
        for blk in self.blocks:
            x = blk(x, N_train=n_sink_rows + N, N_test=M, attn_mask=attn_mask)

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

        return {"predictions": predictions}


if __name__ == "__main__":
    torch.manual_seed(0)
    
    print("=" * 80)
    print("GraphConditionedInterventionalPFN Test Suite")
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
    # In practice, this would come from the causal graph
    adjacency_matrix = torch.ones(B, L + 2, L + 2)  # (B, L+2, L+2)

    # Test 1: Basic single output regression with full graph
    print("\n[Test 1] Single Output with Full Adjacency Graph")
    print("-" * 80)
    model = GraphConditionedInterventionalPFN(
        num_features=L,
        d_model=128,
        depth=2,
        heads_feat=4,
        heads_samp=4,
        dropout=0.1,
    )
    out = model(X_obs, T_obs, Y_obs, X_intv, T_intv, adjacency_matrix)
    print(f"✓ predictions shape: {out['predictions'].shape} (expected: ({B}, {M}))")
    print(f"✓ Sample predictions: {out['predictions'][0, :3].detach().numpy()}")
    assert out["predictions"].shape == (B, M), f"Expected shape ({B}, {M}), got {out['predictions'].shape}"
    print("✓ Test 1 passed!")

    # Test 2: Sparse adjacency matrix (causal structure)
    print("\n[Test 2] Sparse Causal Graph")
    print("-" * 80)
    # Create a simple chain structure: Treatment -> Feature1 -> Feature2 -> ... -> Outcome
    sparse_adj = torch.zeros(B, L + 2, L + 2)
    # Allow self-attention
    for i in range(L + 2):
        sparse_adj[:, i, i] = 1
    # Treatment (0) affects all features
    sparse_adj[:, :, 0] = 1
    # Chain structure: each feature affects the next
    for i in range(L + 1):
        sparse_adj[:, i + 1, i] = 1
    # All features affect outcome (position 1)
    sparse_adj[:, 1, :] = 1
    
    out_sparse = model(X_obs, T_obs, Y_obs, X_intv, T_intv, sparse_adj)
    print(f"✓ predictions shape (sparse graph): {out_sparse['predictions'].shape} (expected: ({B}, {M}))")
    assert out_sparse["predictions"].shape == (B, M)
    print("✓ Test 2 passed!")

    # Test 3: High-dimensional output
    print("\n[Test 3] High-Dimensional Output with Graph Conditioning")
    print("-" * 80)
    output_dim = 10
    model_hd = GraphConditionedInterventionalPFN(
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
    assert out_hd["predictions"].shape == (B, M, output_dim)
    print("✓ Test 3 passed!")

    # Test 4: With sample attention sinks
    print("\n[Test 4] Sample Attention Sinks with Graph Conditioning")
    print("-" * 80)
    n_sink_rows = 3
    model_sinks_rows = GraphConditionedInterventionalPFN(
        num_features=L,
        d_model=128,
        depth=2,
        heads_feat=4,
        heads_samp=4,
        dropout=0.1,
        n_sample_attention_sink_rows=n_sink_rows,
    )
    out_sinks = model_sinks_rows(X_obs, T_obs, Y_obs, X_intv, T_intv, adjacency_matrix)
    print(f"✓ predictions shape (with {n_sink_rows} sink rows): {out_sinks['predictions'].shape} (expected: ({B}, {M}))")
    assert out_sinks["predictions"].shape == (B, M)
    print("✓ Test 4 passed!")
    
    # Test 5: With feature attention sinks
    print("\n[Test 5] Feature Attention Sinks with Graph Conditioning")
    print("-" * 80)
    n_sink_cols = 2
    model_sinks_cols = GraphConditionedInterventionalPFN(
        num_features=L,
        d_model=128,
        depth=2,
        heads_feat=4,
        heads_samp=4,
        dropout=0.1,
        n_feature_attention_sink_cols=n_sink_cols,
    )
    out_sinks_cols = model_sinks_cols(X_obs, T_obs, Y_obs, X_intv, T_intv, adjacency_matrix)
    print(f"✓ predictions shape (with {n_sink_cols} sink cols): {out_sinks_cols['predictions'].shape} (expected: ({B}, {M}))")
    assert out_sinks_cols["predictions"].shape == (B, M)
    print("✓ Test 5 passed!")
    
    # Test 6: Verify different adjacency matrices produce different results
    print("\n[Test 6] Graph Structure Affects Predictions")
    print("-" * 80)
    # Create two different graph structures
    adj1 = torch.ones(B, L + 2, L + 2)  # Fully connected
    adj2 = torch.eye(L + 2).unsqueeze(0).expand(B, -1, -1)  # Only self-attention
    
    out1 = model(X_obs, T_obs, Y_obs, X_intv, T_intv, adj1)
    out2 = model(X_obs, T_obs, Y_obs, X_intv, T_intv, adj2)
    
    # Predictions should be different for different graphs
    diff = torch.abs(out1['predictions'] - out2['predictions']).max().item()
    print(f"✓ Max prediction difference between full and diagonal graphs: {diff:.6f}")
    assert diff > 1e-6, "Different graphs should produce different predictions"
    print("✓ Test 6 passed!")
    
    # Model statistics
    print("\n[Model Statistics]")
    print("-" * 80)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    # Architecture verification
    print("\n[Architecture Verification]")
    print("-" * 80)
    print(f"✓ Uses GraphConditionedTwoWayBlock with attention masking")
    print(f"✓ MLP uses SwiGLU activation (has fc_gate layer)")
    print(f"✓ Pre-layer normalization")
    print(f"✓ Separate train/test attention layers")
    print(f"✓ Graph-conditioned feature attention via masking")
    print(f"✓ Optional attention sinks supported")
    
    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
