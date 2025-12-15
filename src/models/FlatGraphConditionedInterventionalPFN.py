"""
Flat Graph-Conditioned Interventional Prior-Data Fitted Network (PFN).

This module implements a radically different approach to graph conditioning:
instead of using attention masking, the adjacency matrix is FLATTENED and
APPENDED to each cell input before the embedding layer.

Key differences from GraphConditionedInterventionalPFN:
1. No attention masking - uses standard full attention
2. Adjacency matrix is flattened: (B, L+2, L+2) -> (B, (L+2)^2)
3. Each cell gets the full flattened adjacency matrix appended to its input
4. Cell MLP input dimension: 1 + (L+2)^2 instead of just 1
5. Row MLP also gets the flattened adjacency matrix appended

This allows the model to learn how to use graph structure via the embedding layers
rather than hard-coding it via attention masks.

Input format (same as GraphConditionedInterventionalPFN):
- X_obs, T_obs, Y_obs: Observational data
- X_intv, T_intv: Interventional data
- adjacency_matrix: (B, L+2, L+2) causal graph

Adjacency matrix ordering:
- Position 0: Treatment variable
- Position 1: Outcome variable  
- Position 2+: Feature variables (after dropout, sorted)
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
    
    Args:
        dim: Input and output feature dimension (D).
        hidden_mult: Multiplier for the hidden layer size.
        dropout: Dropout rate.
    """
    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = hidden_mult * dim
        self.fc1 = nn.Linear(dim, hidden)
        self.fc_gate = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x) * F.silu(self.fc_gate(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class InputMLP(nn.Module):
    """
    Generic 2-layer MLP with SwiGLU activation for input embeddings.
    
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
        self.fc_gate = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x) * F.silu(self.fc_gate(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TwoWayBlock(nn.Module):
    """
    Standard two-way attention block (no graph conditioning via masking).
    
    Uses pre-layer normalization and separate attention layers for train and test samples.
    Graph conditioning is handled via the input embeddings, not attention masking.
    
    Args:
        dim: Model dimension
        heads_feat: Number of attention heads for feature-wise attention
        heads_samp: Number of attention heads for sample-wise attention
        dropout: Dropout rate
        hidden_mult: Hidden layer multiplier for MLPs
    """
    def __init__(self, dim: int, heads_feat: int, heads_samp: int, dropout: float = 0.0, hidden_mult: int = 4):
        super().__init__()
        # Self-attention across features: (B*S, L+2, D)
        self.feat_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads_feat,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_feat = nn.LayerNorm(dim)

        # Self-attention for train samples: (B*(L+2), N, D)
        self.samp_attn_train = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads_samp,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_samp_train = nn.LayerNorm(dim)
        
        # Cross-attention for test samples to train: (B*(L+2), M, D) -> (B*(L+2), N, D)
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
    ) -> torch.Tensor:
        """
        Args:
            x: (B, S, F, D) where S = N_train + N_test, F = L+2
            N_train: Number of train samples
            N_test: Number of test samples

        Returns:
            (B, S, F, D)
        """
        B, S, F, D = x.shape
        assert S == N_train + N_test

        # 1) Feature-attention (within row) - pre-layer norm
        x_row = x.reshape(B * S, F, D)
        x_norm = self.ln_feat(x_row)
        x2, _ = self.feat_attn(x_norm, x_norm, x_norm, need_weights=False)
        x_row = x_row + self.drop(x2)
        x = x_row.reshape(B, S, F, D)

        # 2) Sample-attention (within column) - separate for train and test
        x_col = x.permute(0, 2, 1, 3).contiguous().reshape(B * F, S, D)
        
        x_train = x_col[:, :N_train, :]
        x_test = x_col[:, N_train:, :]
        
        # Train self-attention
        x_train_norm = self.ln_samp_train(x_train)
        x_train_attn, _ = self.samp_attn_train(x_train_norm, x_train_norm, x_train_norm, need_weights=False)
        x_train = x_train + self.drop(x_train_attn)
        
        # Test cross-attention to train
        if N_test > 0:
            x_test_norm = self.ln_samp_test(x_test)
            x_train_norm_kv = self.ln_samp_test(x_train)
            x_test_attn, _ = self.samp_attn_test(x_test_norm, x_train_norm_kv, x_train_norm_kv, need_weights=False)
            x_test = x_test + self.drop(x_test_attn)
        
        x_col = torch.cat([x_train, x_test], dim=1)
        x = x_col.reshape(B, F, S, D).permute(0, 2, 1, 3).contiguous()

        # 3) Position-wise MLP - pre-layer norm
        x_norm = self.ln_mlp(x)
        x2 = self.mlp(x_norm)
        x = x + self.drop(x2)
        return x


class FlatGraphConditionedInterventionalPFN(nn.Module):
    """
    Interventional PFN with flat graph conditioning.
    
    Instead of using attention masking, the adjacency matrix is flattened and
    appended to each cell's input before embedding. This allows the model to
    learn how to use graph structure via the embedding layers.
    
    Key architecture:
    - Cell MLP input: [cell_value, flattened_adjacency_matrix]
    - Row MLP input: [feature_values, flattened_adjacency_matrix]
    - Standard full attention (no masking)
    
    Args:
        num_features: Number of regular features (L)
        d_model: Model embedding dimension
        depth: Number of two-way attention blocks
        heads_feat: Number of attention heads for feature attention
        heads_samp: Number of attention heads for sample attention
        dropout: Dropout rate
        output_dim: Output dimension (1 for regression, >1 for distributions)
        hidden_mult: Hidden layer multiplier for MLPs
        normalize_features: Whether to normalize features per task
        use_same_row_mlp: Whether to use same row MLP for train and test
        n_sample_attention_sink_rows: Number of learnable sink rows
        n_feature_attention_sink_cols: Number of learnable sink columns
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
        self.num_features = num_features  # L
        self.d_model = d_model
        self.output_dim = output_dim
        self.normalize_features = normalize_features
        self.n_sample_attention_sink_rows = n_sample_attention_sink_rows
        self.n_feature_attention_sink_cols = n_feature_attention_sink_cols

        # Adjacency matrix dimension when flattened
        # Shape: (L+2, L+2) -> (L+2)^2
        self.adj_dim = (num_features + 2) ** 2

        # === Embedding MLPs ===
        # Row-wise MLP: input = (L+1) features + flattened adjacency
        row_input_dim = num_features + 1 + self.adj_dim
        if use_same_row_mlp:
            self.row_mlp_train = InputMLP(row_input_dim, d_model, hidden_mult, dropout)
            self.row_mlp_test = self.row_mlp_train
        else:
            self.row_mlp_train = InputMLP(row_input_dim, d_model, hidden_mult, dropout)
            self.row_mlp_test = InputMLP(row_input_dim, d_model, hidden_mult, dropout)

        # Cell MLP: input = 1 scalar + flattened adjacency
        cell_input_dim = 1 + self.adj_dim
        self.cell_mlp = InputMLP(cell_input_dim, d_model, hidden_mult, dropout)
        
        # Label MLP: input = 1 scalar + flattened adjacency
        self.label_mlp_train = InputMLP(cell_input_dim, d_model, hidden_mult, dropout)

        # Feature positional encodings
        feat_pos = self._build_feature_positional(num_features + 2, d_model)
        self.register_buffer("feature_positional", feat_pos.unsqueeze(0).unsqueeze(0), persistent=False)

        # Learnable scaling for row vs cell embeddings
        self.row_scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(2.0)))
        self.cell_scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(2.0)))

        # === Attention Sinks ===
        if n_sample_attention_sink_rows > 0:
            self.sink_rows_x = nn.Parameter(torch.zeros(1, n_sample_attention_sink_rows, num_features + 2, d_model))
            nn.init.normal_(self.sink_rows_x, std=0.02)
            self.sink_rows_y = nn.Parameter(torch.zeros(1, n_sample_attention_sink_rows, d_model))
            nn.init.normal_(self.sink_rows_y, std=0.02)
        else:
            self.sink_rows_x = None
            self.sink_rows_y = None
        
        if n_feature_attention_sink_cols > 0:
            self.sink_cols = nn.Parameter(torch.zeros(1, 1, n_feature_attention_sink_cols, d_model))
            nn.init.normal_(self.sink_cols, std=0.02)
        else:
            self.sink_cols = None

        # Learnable role embeddings
        self.obs_T_embed = self._create_role_embedding(1, 1, d_model)
        self.obs_label_embed = self._create_role_embedding(1, 1, d_model)
        self.obs_feature_embed = self._create_role_embedding(1, 1, d_model)
        self.intv_T_embed = self._create_role_embedding(1, 1, d_model)
        self.intv_label_embed = self._create_role_embedding(1, 1, d_model)
        self.intv_feature_embed = self._create_role_embedding(1, 1, d_model)

        # Stacked two-way attention blocks (no masking)
        self.blocks = nn.ModuleList([
            TwoWayBlock(d_model, heads_feat, heads_samp, dropout=dropout, hidden_mult=hidden_mult)
            for _ in range(depth)
        ])

        # Output projection
        self.regression_head = nn.Linear(d_model, output_dim)

    def _create_role_embedding(self, *shape, std=0.02):
        """Helper to create role embedding parameter."""
        embed = nn.Parameter(torch.zeros(*shape))
        nn.init.normal_(embed, std=std)
        return embed
    
    @staticmethod
    def _build_feature_positional(num_tokens: int, dim: int) -> torch.Tensor:
        """Sinusoidal positional encodings."""
        pe = torch.zeros(num_tokens, dim)
        position = torch.arange(0, num_tokens, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    @staticmethod
    def _normalize_features(
        X_train: torch.Tensor,
        X_test: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize features using quantile transform + standardization.
        
        Args:
            X_train: (B, N, L+1)
            X_test: (B, M, L+1)
            
        Returns:
            Normalized X_train, X_test
        """
        B, N, F = X_train.shape
        M = X_test.shape[1]
        
        # Quantile transform
        X_train_sorted, _ = torch.sort(X_train, dim=1)
        
        def quantile_transform(X, X_sorted):
            B, S, F = X.shape
            X_quantiles = torch.zeros_like(X)
            for b in range(B):
                for f in range(F):
                    sorted_vals = X_sorted[b, :, f]
                    vals = X[b, :, f]
                    ranks = torch.searchsorted(sorted_vals.contiguous(), vals.contiguous())
                    quantiles = ranks.float() / max(X_sorted.shape[1] - 1, 1)
                    quantiles = quantiles.clamp(0.0, 1.0)
                    X_quantiles[b, :, f] = quantiles
            return X_quantiles
        
        X_train_q = quantile_transform(X_train, X_train_sorted)
        X_test_q = quantile_transform(X_test, X_train_sorted)
        
        # Standardization
        mean = X_train_q.mean(dim=1, keepdim=True)
        std = X_train_q.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        
        return (X_train_q - mean) / std, (X_test_q - mean) / std

    def _flatten_adjacency(self, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """
        Flatten adjacency matrix for concatenation with cell inputs.
        
        Args:
            adjacency_matrix: (B, L+2, L+2)
            
        Returns:
            Flattened adjacency: (B, (L+2)^2)
        """
        B = adjacency_matrix.shape[0]
        return adjacency_matrix.reshape(B, -1)

    def _embed_features(
        self,
        X: torch.Tensor,
        T: torch.Tensor,
        adj_flat: torch.Tensor,
        row_mlp: nn.Module,
        is_intvn: bool,
    ) -> torch.Tensor:
        """
        Embed features with flattened adjacency matrix appended to inputs.
        
        Args:
            X: (B, S, L) - regular features
            T: (B, S, 1) - intervention feature
            adj_flat: (B, (L+2)^2) - flattened adjacency matrix
            row_mlp: Row-wise MLP to use
            is_intvn: Whether this is interventional data
            
        Returns:
            (B, S, L+1, D)
        """
        B, S, L = X.shape
        
        # Concatenate features + intervention
        X_with_T = torch.cat([X, T], dim=2)  # (B, S, L+1)
        
        # Expand flattened adjacency for each sample
        adj_expanded = adj_flat.unsqueeze(1).expand(B, S, -1)  # (B, S, adj_dim)
        
        # Concatenate for row-wise embedding
        row_input = torch.cat([X_with_T, adj_expanded], dim=2)  # (B, S, L+1 + adj_dim)
        row_emb = row_mlp(row_input)  # (B, S, D)

        # Cell-wise embeddings: each cell gets [value, flattened_adjacency]
        # X_with_T: (B, S, L+1) -> (B, S, L+1, 1)
        X_with_T_expanded = X_with_T.unsqueeze(-1)  # (B, S, L+1, 1)
        # Expand adjacency for each cell
        adj_for_cells = adj_flat.unsqueeze(1).unsqueeze(2).expand(B, S, L + 1, -1)  # (B, S, L+1, adj_dim)
        # Concatenate value + adjacency
        cell_input = torch.cat([X_with_T_expanded, adj_for_cells], dim=-1)  # (B, S, L+1, 1 + adj_dim)
        cell_emb = self.cell_mlp(cell_input)  # (B, S, L+1, D)

        # Split and add role embeddings
        X_cells = cell_emb[:, :, :-1, :]  # (B, S, L, D)
        T_cells = cell_emb[:, :, -1:, :]  # (B, S, 1, D)
        
        if is_intvn:
            X_cells = X_cells + self.intv_feature_embed.expand(B, S, L, -1)
            T_cells = T_cells + self.intv_T_embed.expand(B, S, 1, -1)
        else:
            X_cells = X_cells + self.obs_feature_embed.expand(B, S, L, -1)
            T_cells = T_cells + self.obs_T_embed.expand(B, S, 1, -1)
        
        cell_emb = torch.cat([X_cells, T_cells], dim=2)  # (B, S, L+1, D)

        # Combine row and cell embeddings
        row_exp = row_emb.unsqueeze(2).expand(-1, -1, L + 1, -1)
        feat_emb = self.row_scale * row_exp + self.cell_scale * cell_emb
        
        return feat_emb

    def _embed_labels(self, Y: torch.Tensor, adj_flat: torch.Tensor) -> torch.Tensor:
        """
        Embed labels with flattened adjacency appended.
        
        Args:
            Y: (B, S) or (B, S, 1)
            adj_flat: (B, (L+2)^2)
            
        Returns:
            (B, S, D)
        """
        if Y.dim() == 3:
            Y = Y.squeeze(-1)
        B, S = Y.shape
        
        # Expand adjacency for each sample
        adj_expanded = adj_flat.unsqueeze(1).expand(B, S, -1)  # (B, S, adj_dim)
        
        # Concatenate label value + adjacency
        label_input = torch.cat([Y.unsqueeze(-1), adj_expanded], dim=-1)  # (B, S, 1 + adj_dim)
        label_emb = self.label_mlp_train(label_input)  # (B, S, D)
        
        # Add role embedding
        label_emb = label_emb + self.obs_label_embed.expand(B, S, -1)
        return label_emb

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
        Forward pass with flat graph conditioning.
        
        Args:
            X_obs: (B, N, L)
            T_obs: (B, N, 1)
            Y_obs: (B, N)
            X_intv: (B, M, L)
            T_intv: (B, M, 1)
            adjacency_matrix: (B, L+2, L+2)
            
        Returns:
            Dict with "predictions": (B, M) or (B, M, output_dim)
        """
        B, N, L = X_obs.shape
        assert L == self.num_features
        M = X_intv.shape[1]

        # Ensure correct shapes
        if T_obs.dim() == 2:
            T_obs = T_obs.unsqueeze(-1)
        if T_intv.dim() == 2:
            T_intv = T_intv.unsqueeze(-1)

        # Validate adjacency matrix
        assert adjacency_matrix.shape == (B, L + 2, L + 2), \
            f"Expected adjacency shape ({B}, {L+2}, {L+2}), got {adjacency_matrix.shape}"

        # Flatten adjacency matrix
        adj_flat = self._flatten_adjacency(adjacency_matrix)  # (B, (L+2)^2)

        # === Normalize features ===
        if self.normalize_features:
            X_obs_with_T = torch.cat([X_obs, T_obs], dim=2)
            X_intv_with_T = torch.cat([X_intv, T_intv], dim=2)
            X_obs_norm, X_intv_norm = self._normalize_features(X_obs_with_T, X_intv_with_T)
            X_obs_norm, T_obs_norm = X_obs_norm[:, :, :L], X_obs_norm[:, :, L:L+1]
            X_intv_norm, T_intv_norm = X_intv_norm[:, :, :L], X_intv_norm[:, :, L:L+1]
        else:
            X_obs_norm, T_obs_norm = X_obs, T_obs
            X_intv_norm, T_intv_norm = X_intv, T_intv

        # === Embed features with graph ===
        feat_obs = self._embed_features(X_obs_norm, T_obs_norm, adj_flat, self.row_mlp_train, is_intvn=False)
        feat_intv = self._embed_features(X_intv_norm, T_intv_norm, adj_flat, self.row_mlp_test, is_intvn=True)
        feat_all = torch.cat([feat_obs, feat_intv], dim=1)  # (B, S, L+1, D)

        # === Embed labels with graph ===
        lab_obs = self._embed_labels(Y_obs, adj_flat)  # (B, N, D)
        lab_intv = self.intv_label_embed.expand(B, M, self.d_model)
        lab_all = torch.cat([lab_obs, lab_intv], dim=1)  # (B, S, D)

        # Stack features + label column
        x = torch.cat([feat_all, lab_all.unsqueeze(2)], dim=2)  # (B, S, L+2, D)

        # Add positional encodings
        x = x + self.feature_positional

        # === Add attention sinks ===
        if self.sink_rows_x is not None:
            sink_x = self.sink_rows_x.expand(B, -1, -1, -1)
            sink_x_features = sink_x[:, :, :-1, :]
            sink_y = self.sink_rows_y.expand(B, -1, -1).unsqueeze(2)
            sink_x = torch.cat([sink_x_features, sink_y], dim=2)
            x = torch.cat([sink_x, x], dim=1)
        
        n_sink_rows = self.n_sample_attention_sink_rows
        
        if self.sink_cols is not None:
            current_n_samples = x.shape[1]
            sink_c = self.sink_cols.expand(B, current_n_samples, -1, -1)
            x = torch.cat([sink_c, x], dim=2)
        
        n_sink_cols = self.n_feature_attention_sink_cols

        # === Apply transformer blocks ===
        for blk in self.blocks:
            x = blk(x, N_train=n_sink_rows + N, N_test=M)

        # === Readout ===
        label_pos = n_sink_cols + self.num_features + 1
        test_start_idx = n_sink_rows + N
        h_intv = x[:, test_start_idx:, label_pos, :]  # (B, M, D)
        predictions = self.regression_head(h_intv)  # (B, M, output_dim)

        if self.output_dim == 1:
            predictions = predictions.squeeze(-1)

        return {"predictions": predictions}


if __name__ == "__main__":
    torch.manual_seed(42)
    
    print("=" * 80)
    print("FlatGraphConditionedInterventionalPFN Test Suite")
    print("=" * 80)
    
    # Test configuration
    B, N, M, L = 2, 16, 5, 3
    
    X_obs = torch.randn(B, N, L)
    T_obs = torch.randn(B, N, 1)
    Y_obs = torch.randn(B, N)
    X_intv = torch.randn(B, M, L)
    T_intv = torch.randn(B, M, 1)
    adjacency_matrix = torch.ones(B, L + 2, L + 2)

    # Test 1: Basic functionality
    print("\n[Test 1] Basic Single Output Regression with Flat Graph Conditioning")
    print("-" * 80)
    model = FlatGraphConditionedInterventionalPFN(
        num_features=L,
        d_model=128,
        depth=2,
        heads_feat=4,
        heads_samp=4,
        dropout=0.1,
    )
    out = model(X_obs, T_obs, Y_obs, X_intv, T_intv, adjacency_matrix)
    print(f"✓ predictions shape: {out['predictions'].shape} (expected: ({B}, {M}))")
    assert out["predictions"].shape == (B, M)
    print(f"✓ Sample predictions: {out['predictions'][0, :3].detach().numpy()}")
    print("✓ Test 1 passed!")

    # Test 2: Different adjacency matrices produce different outputs
    print("\n[Test 2] Different Adjacency Matrices Produce Different Predictions")
    print("-" * 80)
    adj1 = torch.ones(B, L + 2, L + 2)  # Fully connected
    adj2 = torch.eye(L + 2).unsqueeze(0).expand(B, -1, -1)  # Diagonal only
    
    out1 = model(X_obs, T_obs, Y_obs, X_intv, T_intv, adj1)
    out2 = model(X_obs, T_obs, Y_obs, X_intv, T_intv, adj2)
    
    diff = torch.abs(out1['predictions'] - out2['predictions']).max().item()
    print(f"✓ Max prediction difference: {diff:.6f}")
    assert diff > 1e-6, "Different graphs should produce different predictions"
    print("✓ Test 2 passed!")

    # Test 3: High-dimensional output
    print("\n[Test 3] High-Dimensional Output")
    print("-" * 80)
    output_dim = 10
    model_hd = FlatGraphConditionedInterventionalPFN(
        num_features=L,
        d_model=128,
        depth=2,
        heads_feat=4,
        heads_samp=4,
        output_dim=output_dim,
    )
    out_hd = model_hd(X_obs, T_obs, Y_obs, X_intv, T_intv, adjacency_matrix)
    print(f"✓ predictions shape: {out_hd['predictions'].shape} (expected: ({B}, {M}, {output_dim}))")
    assert out_hd["predictions"].shape == (B, M, output_dim)
    print("✓ Test 3 passed!")

    # Test 4: With attention sinks
    print("\n[Test 4] With Attention Sinks")
    print("-" * 80)
    model_sinks = FlatGraphConditionedInterventionalPFN(
        num_features=L,
        d_model=128,
        depth=2,
        heads_feat=4,
        heads_samp=4,
        n_sample_attention_sink_rows=3,
        n_feature_attention_sink_cols=2,
    )
    out_sinks = model_sinks(X_obs, T_obs, Y_obs, X_intv, T_intv, adjacency_matrix)
    print(f"✓ predictions shape with sinks: {out_sinks['predictions'].shape} (expected: ({B}, {M}))")
    assert out_sinks["predictions"].shape == (B, M)
    print("✓ Test 4 passed!")

    # Test 5: Verify flattened adjacency is used
    print("\n[Test 5] Verify Flattened Adjacency Dimension")
    print("-" * 80)
    adj_dim = (L + 2) ** 2
    print(f"✓ Adjacency matrix shape: ({B}, {L+2}, {L+2})")
    print(f"✓ Flattened adjacency dimension: {adj_dim}")
    print(f"✓ Cell MLP input dimension: 1 + {adj_dim} = {1 + adj_dim}")
    print(f"✓ Row MLP input dimension: {L+1} + {adj_dim} = {L+1+adj_dim}")
    assert model.adj_dim == adj_dim
    print("✓ Test 5 passed!")

    # Model statistics
    print("\n[Model Statistics]")
    print("-" * 80)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    print("\n[Architecture Verification]")
    print("-" * 80)
    print(f"✓ Uses FLAT graph conditioning (no attention masking)")
    print(f"✓ Adjacency matrix is flattened and appended to each cell")
    print(f"✓ Standard full attention (TwoWayBlock)")
    print(f"✓ Graph structure learned via embedding layers")
    print(f"✓ SwiGLU activation, pre-layer norm")
    
    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
