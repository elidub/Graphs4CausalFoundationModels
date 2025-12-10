"""
Simple Prior-Data Fitted Network (PFN)-style regressor with two-way attention.

This module provides:
- A lightweight MLP block.
- A TwoWayBlock that alternates attention across **features (columns)** and **samples (rows)**.
- A SimplePFNRegressor that encodes tabular (X_train, y_train) and (X_test) into a 4D tensor
  and stacks TwoWayBlocks to produce predictions for test rows.

Tensor conventions
------------------
- B: batch size (independent tasks / problems per batch)
- N: number of training samples per task
- M: number of test samples per task
- L: number of features (columns) in X (== num_features)
- D: model embedding dimension (== d_model)
- S: total sequence length along the "sample" axis, S = N + M

Core idea
---------
Each TwoWayBlock performs:
1) **Feature-attention (within row):** Self-attention over the L feature tokens of a
   fixed row (sample). This lets features talk to each other.
2) **Sample-attention (within column):** Self-attention over the S sample tokens of a
   fixed feature (column). A custom mask allows test samples to attend to train
   samples but prevents information leakage **between test samples** and from
   **test→train**.

Embedding design
----------------
For X_train (B, N, L) and X_test (B, M, L), after *per-task feature normalization*:

- Train features:
    r_n^tr = f_row_tr(X_train[n, :])       in R^D        (row-wise MLP over L dims)
    c_{n,l} = f_cell(X_train[n, l])        in R^D        (shared scalar MLP for all features)
    h_{n,l}^tr = α * r_n^tr + β * c_{n,l} + e_l

- Test features:
    r_m^te = f_row_te(X_test[m, :])        in R^D
    c_{m,l} = f_cell(X_test[m, l])         in R^D        (same scalar MLP as train)
    h_{m,l}^te = α * r_m^te + β * c_{m,l} + e_l

- Train labels:
    h_n^{y,tr} = f_y_tr(y_n)               in R^D
    treated as an extra "label feature" (index L) with e_L.

- Test labels:
    learned [MASK] vector.

Normalization (Optional)
-------------------------
When normalize_features=True (default), before embedding we apply a two-step normalization 
per batch/task and per feature dimension:

  1) Uniform quantile transform based on the support set (X_train):
     - Sort X_train along sample axis to get empirical quantiles
     - Map each value in X_train and X_test to its quantile rank in [0, 1]
     - This makes features more uniform and robust to outliers
  
  2) Standard normalization (mean/std) based on support set:
     - Compute mean and std from quantile-transformed X_train
     - Apply (X_quantile - mean) / std to both X_train and X_test
     - std is clamped to >= 1e-6 for numerical stability

This follows PFN preprocessing: quantile transform followed by standardization.

When normalize_features=False, features are passed through as-is (useful if preprocessing 
is done externally).

Labels are *not* normalized inside the model (you can handle target scaling externally).
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
    
    SwiGLU is a gated linear unit variant that uses Swish (SiLU) activation:
    SwiGLU(x) = (W_1 x) ⊗ SiLU(W_2 x)
    where ⊗ is element-wise multiplication.

    Args:
        dim: Input and output feature dimension (D).
        hidden_mult: Multiplier for the hidden layer size (hidden = hidden_mult * dim).
        dropout: Dropout rate applied after activation and after the second linear layer.
    """
    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = hidden_mult * dim
        # SwiGLU uses two parallel projections for gating
        self.fc1 = nn.Linear(dim, hidden)
        self.fc_gate = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: x_out = fc1(x) * silu(fc_gate(x))
        x_linear = self.fc1(x)
        x_gate = F.silu(self.fc_gate(x))
        x = x_linear * x_gate
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class InputMLP(nn.Module):
    """
    Generic 2-layer MLP for input embeddings (row-wise and scalar-wise) with SwiGLU activation.
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = hidden_mult * out_dim
        # SwiGLU uses two parallel projections for gating
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc_gate = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: x_out = fc1(x) * silu(fc_gate(x))
        x_linear = self.fc1(x)
        x_gate = F.silu(self.fc_gate(x))
        x = x_linear * x_gate
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TwoWayBlock(nn.Module):
    """
    Alternating attention across features (columns) and samples (rows), followed by an MLP.
    Uses pre-layer normalization for better training stability.
    """
    def __init__(self, dim: int, heads_feat: int, heads_samp: int, dropout: float = 0.0, hidden_mult: int = 4):
        super().__init__()
        # Self-attention across features of the same sample: (B*S, L, D)
        self.feat_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads_feat,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_feat = nn.LayerNorm(dim)

        # Self-attention across samples of the same feature: (B*L, S, D)
        self.samp_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads_samp,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_samp = nn.LayerNorm(dim)

        # Position-wise MLP
        self.mlp = MLP(dim, hidden_mult=hidden_mult, dropout=dropout)
        self.ln_mlp = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sample_attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, S, L, D).
            sample_attn_mask: Optional boolean mask of shape (S, S),
                where True = masked (no attention). Broadcast across batch & heads.

        Returns:
            Tensor of shape (B, S, L, D).
        """
        B, S, L, D = x.shape

        # 1) Feature-attention (within row) with pre-layer norm
        x_row = x.reshape(B * S, L, D)                       # (B*S, L, D)
        x_row_norm = self.ln_feat(x_row)
        x2, _ = self.feat_attn(x_row_norm, x_row_norm, x_row_norm, need_weights=False)
        x_row = x_row + self.drop(x2)
        x = x_row.reshape(B, S, L, D)

        # 2) Sample-attention (within column) with pre-layer norm
        x_col = x.permute(0, 2, 1, 3).contiguous().reshape(B * L, S, D)  # (B*L, S, D)
        x_col_norm = self.ln_samp(x_col)
        if sample_attn_mask is not None:
            x2, _ = self.samp_attn(
                x_col_norm, x_col_norm, x_col_norm,
                attn_mask=sample_attn_mask,
                need_weights=False,
            )
        else:
            x2, _ = self.samp_attn(x_col_norm, x_col_norm, x_col_norm, need_weights=False)

        x_col = x_col + self.drop(x2)
        x = x_col.reshape(B, L, S, D).permute(0, 2, 1, 3).contiguous()    # (B, S, L, D)

        # 3) Position-wise MLP with pre-layer norm
        x_norm = self.ln_mlp(x)
        x2 = self.mlp(x_norm)
        x = x + self.drop(x2)
        return x


class SimplePFNRegressor(nn.Module):
    """
    PFN-like regressor for tabular data with two-way attention and structured embeddings.

    Optionally performs per-task, per-feature normalization of X_train/X_test internally.
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
        n_sample_attention_sink_rows: int = 0,
        n_feature_attention_sink_cols: int = 0,
    ):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.output_dim = output_dim
        self.normalize_features = normalize_features
        self.n_sample_attention_sink_rows = n_sample_attention_sink_rows
        self.n_feature_attention_sink_cols = n_feature_attention_sink_cols

        # === Embedding MLPs ===
        # Row-wise MLPs over feature dimension (R^L -> R^D)
        self.row_mlp_train = InputMLP(num_features, d_model, hidden_mult, dropout)
        self.row_mlp_test  = InputMLP(num_features, d_model, hidden_mult, dropout)

        # Shared scalar MLP for feature cells (train + test)
        self.cell_mlp = InputMLP(1, d_model, hidden_mult, dropout)

        # Train label scalar MLP
        self.label_mlp_train = InputMLP(1, d_model, hidden_mult, dropout)

        # Learned [MASK] token for the label column on test rows
        self.label_mask_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.label_mask_embed, std=0.02)

        # Feature positional encodings (sinusoidal) for L_sink + L features + 1 label column
        total_feature_tokens = n_feature_attention_sink_cols + num_features + 1
        feat_pos = self._build_feature_positional(total_feature_tokens, d_model)  # (L_sink+L+1, D)
        self.register_buffer("feature_positional", feat_pos.unsqueeze(0).unsqueeze(0), persistent=False)

        # Learnable scaling for row vs cell embeddings (for stability)
        self.row_scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(2.0)))
        self.cell_scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(2.0)))

        # Stacked two-way attention blocks
        self.blocks = nn.ModuleList([
            TwoWayBlock(d_model, heads_feat, heads_samp, dropout=dropout, hidden_mult=hidden_mult)
            for _ in range(depth)
        ])

        # Output projection from D to desired output_dim per test token
        self.regression_head = nn.Linear(d_model, output_dim)

        # Cache for sample-attention masks keyed by (N, M, device)
        self._mask_cache: Dict[Tuple[int, int, torch.device], torch.Tensor] = {}
        
        # Attention sink rows: fixed random rows added to each batch
        # These are initialized once and reused across all batches
        if n_sample_attention_sink_rows > 0:
            # Initialize sink rows as learnable parameters
            self.sink_rows_x = nn.Parameter(torch.randn(1, n_sample_attention_sink_rows, num_features))
            self.sink_rows_y = nn.Parameter(torch.randn(1, n_sample_attention_sink_rows))
            nn.init.normal_(self.sink_rows_x, mean=0.0, std=0.02)
            nn.init.normal_(self.sink_rows_y, mean=0.0, std=0.02)
        else:
            self.sink_rows_x = None
            self.sink_rows_y = None
        
        # Attention sink columns: fixed random columns (features) added to each batch
        # These are initialized once and reused across all batches
        if n_feature_attention_sink_cols > 0:
            # Initialize sink columns as learnable parameters
            # Note: We don't need separate train/test versions since columns are shared
            self.sink_cols = nn.Parameter(torch.randn(1, 1, n_feature_attention_sink_cols))
            nn.init.normal_(self.sink_cols, mean=0.0, std=0.02)
        else:
            self.sink_cols = None

    @staticmethod
    def _build_feature_positional(num_tokens: int, dim: int) -> torch.Tensor:
        """
        Sinusoidal encodings across feature axis (including label column).
        """
        pe = torch.zeros(num_tokens, dim)
        position = torch.arange(0, num_tokens, dtype=torch.float32).unsqueeze(1)  # (num_tokens, 1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (num_tokens, dim)

    def _build_sample_attn_mask(self, N_sink: int, N: int, M: int, device: torch.device) -> torch.Tensor:
        """
        Build or fetch a Boolean attention mask over S = N_sink + N + M samples.

        Masking policy:
            - Sink rows can attend to all other sink rows (no masking between sinks)
            - Train samples can attend to sink rows and other train samples
            - Train samples cannot attend to test samples
            - Test samples can attend to sink rows and train samples
            - Test samples cannot attend to each other
            
        Sequence order: [sink_rows | train_rows | test_rows]
        """
        key = (N_sink, N, M, device)
        if key in self._mask_cache:
            return self._mask_cache[key]

        S = N_sink + N + M
        mask = torch.zeros((S, S), dtype=torch.bool, device=device)
        
        # Sink rows (indices 0:N_sink) - no masking, they can attend to all sinks
        # Train rows (indices N_sink:N_sink+N) - cannot attend to test rows
        if M > 0:
            mask[N_sink:N_sink+N, N_sink+N:S] = True   # train cannot look at test
            
            # Test rows (indices N_sink+N:S) - can attend to sinks and train, but not other test
            mask[N_sink+N:S, N_sink+N:S] = True  # test cannot look at test
        
        self._mask_cache[key] = mask
        return mask

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
            X_train: (B, N, L) - support set (training features)
            X_test:  (B, M, L) - query set (test features)

        Returns:
            X_train_norm, X_test_norm with same shapes.
        """
        B, N, L = X_train.shape
        M = X_test.shape[1]
        
        # Step 1: Uniform quantile transform based on X_train (support set)
        # For each batch and feature, sort X_train to get empirical quantiles
        X_train_sorted, _ = torch.sort(X_train, dim=1)  # (B, N, L)
        
        # Function to map values to quantiles [0, 1]
        def quantile_transform(X, X_sorted):
            """Map X to quantiles based on sorted support set X_sorted."""
            # For each value in X, find its rank in X_sorted
            # Use searchsorted to find insertion indices
            B, S, L = X.shape
            B_s, N, L_s = X_sorted.shape
            assert B == B_s and L == L_s
            
            X_quantiles = torch.zeros_like(X)
            for b in range(B):
                for l in range(L):
                    # Get sorted values for this batch and feature
                    sorted_vals = X_sorted[b, :, l]  # (N,)
                    vals = X[b, :, l]  # (S,)
                    
                    # Use searchsorted to find ranks (insertion points)
                    # searchsorted finds where each val would be inserted to maintain order
                    ranks = torch.searchsorted(sorted_vals.contiguous(), vals.contiguous())
                    
                    # Convert ranks to quantiles in [0, 1]
                    # Add small epsilon to avoid division by zero when N=1
                    quantiles = ranks.float() / max(N - 1, 1)
                    quantiles = quantiles.clamp(0.0, 1.0)  # Ensure in [0, 1]
                    
                    X_quantiles[b, :, l] = quantiles
            
            return X_quantiles
        
        # Apply quantile transform to both train and test using X_train as reference
        X_train_quantiles = quantile_transform(X_train, X_train_sorted)  # (B, N, L)
        X_test_quantiles = quantile_transform(X_test, X_train_sorted)    # (B, M, L)
        
        # Step 2: Standard normalization (mean/std) on quantile-transformed features
        # Compute mean/std from the support set (X_train_quantiles) only
        mean = X_train_quantiles.mean(dim=1, keepdim=True)  # (B, 1, L)
        std = X_train_quantiles.std(dim=1, keepdim=True, unbiased=False)  # (B, 1, L)
        std = std.clamp_min(1e-6)
        
        # Normalize both train and test using support set statistics
        X_train_norm = (X_train_quantiles - mean) / std  # (B, N, L)
        X_test_norm = (X_test_quantiles - mean) / std    # (B, M, L)
        
        return X_train_norm, X_test_norm

    def _embed_train_features(self, X_train: torch.Tensor) -> torch.Tensor:
        """
        Embed train features.

        Args:
            X_train: (B, N, L) – assumed already normalized.

        Returns:
            Tensor of shape (B, N, L, D).
        """
        B, N, L = X_train.shape
        assert L == self.num_features

        row_emb = self.row_mlp_train(X_train)                 # (B, N, D)
        cell_emb = self.cell_mlp(X_train.unsqueeze(-1))       # (B, N, L, D)

        row_exp = row_emb.unsqueeze(2).expand(-1, -1, L, -1)  # (B, N, L, D)
        feat_emb = self.row_scale * row_exp + self.cell_scale * cell_emb
        return feat_emb

    def _embed_test_features(self, X_test: torch.Tensor) -> torch.Tensor:
        """
        Embed test features.

        Args:
            X_test: (B, M, L) – assumed already normalized.

        Returns:
            Tensor of shape (B, M, L, D).
        """
        B, M, L = X_test.shape
        assert L == self.num_features

        row_emb = self.row_mlp_test(X_test)                   # (B, M, D)
        cell_emb = self.cell_mlp(X_test.unsqueeze(-1))        # (B, M, L, D)

        row_exp = row_emb.unsqueeze(2).expand(-1, -1, L, -1)  # (B, M, L, D)
        feat_emb = self.row_scale * row_exp + self.cell_scale * cell_emb
        return feat_emb

    def _embed_train_labels(self, y_train: torch.Tensor) -> torch.Tensor:
        """
        Embed train labels as a label-column token per row.

        Args:
            y_train: (B, N) or (B, N, 1)

        Returns:
            Tensor of shape (B, N, D).
        """
        if y_train.dim() == 3:
            y_train = y_train.squeeze(-1)  # (B, N)
        label_emb = self.label_mlp_train(y_train.unsqueeze(-1))  # (B, N, D)
        return label_emb

    def forward(self, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass to produce predictions for all test samples.

        Args:
            X_train: (B, N, L) training features.
            y_train: (B, N) or (B, N, 1) training targets.
            X_test:  (B, M, L) test features.

        Returns:
            Dict with:
                - "predictions": (B, M) if output_dim == 1, else (B, M, output_dim)
        """
        B, N, L = X_train.shape
        assert L == self.num_features
        M = X_test.shape[1]
        device = X_train.device
        
        # === Prepend attention sink rows if configured ===
        N_sink = self.n_sample_attention_sink_rows
        if N_sink > 0 and self.sink_rows_x is not None and self.sink_rows_y is not None:
            # Expand sink rows across batch dimension
            sink_x = self.sink_rows_x.expand(B, -1, -1).to(device)  # (B, N_sink, L)
            sink_y = self.sink_rows_y.expand(B, -1).to(device)      # (B, N_sink)
            
            # Match y_train dimensions (could be (B, N) or (B, N, 1))
            if y_train.dim() == 3:
                sink_y = sink_y.unsqueeze(-1)  # (B, N_sink, 1) to match y_train
            
            # Prepend sink rows to training data
            X_train_with_sink = torch.cat([sink_x, X_train], dim=1)  # (B, N_sink+N, L)
            y_train_with_sink = torch.cat([sink_y, y_train], dim=1)  # (B, N_sink+N) or (B, N_sink+N, 1)
        else:
            X_train_with_sink = X_train
            y_train_with_sink = y_train
            N_sink = 0

        # === Normalize features (if enabled) ===
        if self.normalize_features:
            X_train_norm, X_test_norm = self._normalize_features(X_train_with_sink, X_test)
        else:
            X_train_norm, X_test_norm = X_train_with_sink, X_test

        # === Embed features ===
        feat_train = self._embed_train_features(X_train_norm)       # (B, N_sink+N, L, D)
        feat_test  = self._embed_test_features(X_test_norm)         # (B, M, L, D)
        feat_all   = torch.cat([feat_train, feat_test], dim=1)      # (B, S, L, D), S = N_sink+N+M

        # === Prepend attention sink columns if configured ===
        L_sink = self.n_feature_attention_sink_cols
        if L_sink > 0 and self.sink_cols is not None:
            # Expand sink columns across batch and sample dimensions
            # sink_cols shape: (1, 1, L_sink) -> (B, S, L_sink, D)
            S = feat_all.shape[1]  # Total samples: N_sink + N + M
            
            # First embed the sink column values through the cell MLP
            sink_cols_expanded = self.sink_cols.expand(B, S, -1).to(device)  # (B, S, L_sink)
            sink_feat = self.cell_mlp(sink_cols_expanded.unsqueeze(-1))      # (B, S, L_sink, D)
            
            # Prepend sink features to the feature dimension
            feat_all = torch.cat([sink_feat, feat_all], dim=2)               # (B, S, L_sink+L, D)
        else:
            L_sink = 0

        # === Embed labels ===
        lab_train = self._embed_train_labels(y_train_with_sink)     # (B, N_sink+N, D)
        lab_test  = self.label_mask_embed.expand(B, M, self.d_model)
        lab_all   = torch.cat([lab_train, lab_test], dim=1)         # (B, S, D)

        # Stack features + label column along feature axis -> (B, S, L_sink+L+1, D)
        x = torch.cat([feat_all, lab_all.unsqueeze(2)], dim=2)      # (B, S, L_sink+L+1, D)

        # Add feature positional encodings (shared over samples)
        # feature_positional: (1, 1, L_sink+L+1, D)
        x = x + self.feature_positional

        # Build row-wise attention mask and apply blocks
        samp_mask = self._build_sample_attn_mask(N_sink, N, M, device)  # (S, S)

        for blk in self.blocks:
            x = blk(x, sample_attn_mask=samp_mask)

        # Readout: take the label column for test rows only
        # Label column is now at position L_sink + L (after sink columns and regular features)
        # Test rows now start at index N_sink+N instead of N
        label_pos = L_sink + self.num_features
        h_test = x[:, N_sink+N:, label_pos, :]                      # (B, M, D)
        predictions = self.regression_head(h_test)                  # (B, M, output_dim)

        # Backward-compatible squeeze when output_dim == 1
        if self.output_dim == 1:
            predictions = predictions.squeeze(-1)                   # (B, M)

        return {"predictions": predictions}


if __name__ == "__main__":
    torch.manual_seed(0)
    B, N, M, num_feat = 2, 16, 5, 7

    Xtr = torch.randn(B, N, num_feat)
    Xte = torch.randn(B, M, num_feat)
    ytr = torch.randn(B, N)

    # Test with default single output
    print("=== Single Output (Backward Compatible) ===")
    model = SimplePFNRegressor(
        num_features=num_feat,
        d_model=128,
        depth=2,
        heads_feat=4,
        heads_samp=4,
        dropout=0.1,
    )
    out = model(Xtr, ytr, Xte)
    print("predictions shape:", out["predictions"].shape)  # (B, M)

    # Test with high-dimensional output
    print("\n=== High-Dimensional Output ===")
    output_dim = 10
    model_hd = SimplePFNRegressor(
        num_features=num_feat,
        d_model=128,
        depth=2,
        heads_feat=4,
        heads_samp=4,
        dropout=0.1,
        output_dim=output_dim,
    )
    out_hd = model_hd(Xtr, ytr, Xte)
    print("high-dim predictions shape:", out_hd["predictions"].shape)  # (B, M, output_dim)

    # Test BarDistribution-style params
    print("\n=== BarDistribution Compatibility Example ===")
    bar_params = 9
    model_bar = SimplePFNRegressor(
        num_features=num_feat,
        d_model=128,
        depth=2,
        heads_feat=4,
        heads_samp=4,
        dropout=0.1,
        output_dim=bar_params,
    )
    out_bar = model_bar(Xtr, ytr, Xte)
    print("BarDistribution params shape:", out_bar["predictions"].shape)  # (B, M, 9)
    print("BarDistribution params sample:", out_bar["predictions"][0, 0, :])

    # Test without normalization
    print("\n=== Without Internal Normalization ===")
    model_no_norm = SimplePFNRegressor(
        num_features=num_feat,
        d_model=128,
        depth=2,
        heads_feat=4,
        heads_samp=4,
        dropout=0.1,
        normalize_features=False,
    )
    out_no_norm = model_no_norm(Xtr, ytr, Xte)
    print("predictions shape (no norm):", out_no_norm["predictions"].shape)  # (B, M)
    print("Note: Features passed through as-is (useful if externally preprocessed)")
