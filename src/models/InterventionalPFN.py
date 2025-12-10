"""
Interventional Prior-Data Fitted Network (PFN) for causal inference.

This module extends SimplePFN to handle interventional data with special treatment
for intervened features (T_obs, T_intv). The architecture is similar to SimplePFN
but with dedicated embeddings for the intervened feature column.

Input format from InterventionalDataset:
- X_obs: (B, N, L) - observational features (train)
- T_obs: (B, N, 1) - intervened feature values (train)
- Y_obs: (B, N) - observational targets (train)
- X_intv: (B, M, L) - interventional features (test)
- T_intv: (B, M, 1) - intervened feature values (test)
- Y_intv: (B, M) - interventional targets (test) [not used during forward]

Key differences from SimplePFN:
1. T_obs and T_intv are treated as special feature columns with their own embedding
2. These intervened columns are concatenated to X_obs/X_intv to form complete feature matrices
3. The intervened column gets a special positional encoding to distinguish it
4. Otherwise follows the same two-way attention architecture as SimplePFN
"""

from __future__ import annotations
from typing import Optional, Dict, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Two-layer feed-forward block with GELU activations and dropout.

    Args:
        dim: Input and output feature dimension (D).
        hidden_mult: Multiplier for the hidden layer size (hidden = hidden_mult * dim).
        dropout: Dropout rate applied after GELU and after the second linear layer.
    """
    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = hidden_mult * dim
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class InputMLP(nn.Module):
    """
    Generic 2-layer MLP for input embeddings (row-wise and scalar-wise).
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = hidden_mult * out_dim
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TwoWayBlock(nn.Module):
    """
    Alternating attention across features (columns) and samples (rows), followed by an MLP.
    """
    def __init__(self, dim: int, heads_feat: int, heads_samp: int, dropout: float = 0.0, hidden_mult: int = 4):
        super().__init__()
        # Self-attention across features of the same sample: (B*S, L+1, D)
        self.feat_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads_feat,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_feat = nn.LayerNorm(dim)

        # Self-attention across samples of the same feature: (B*(L+1), S, D)
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
            x: Input tensor of shape (B, S, F, D) where F = L+1 (features + intervened column).
            sample_attn_mask: Optional boolean mask of shape (S, S),
                where True = masked (no attention). Broadcast across batch & heads.

        Returns:
            Tensor of shape (B, S, F, D).
        """
        B, S, F, D = x.shape

        # 1) Feature-attention (within row)
        x_row = x.reshape(B * S, F, D)                       # (B*S, F, D)
        x2, _ = self.feat_attn(x_row, x_row, x_row, need_weights=False)
        x_row = x_row + self.drop(x2)
        x_row = self.ln_feat(x_row)
        x = x_row.reshape(B, S, F, D)

        # 2) Sample-attention (within column)
        x_col = x.permute(0, 2, 1, 3).contiguous().reshape(B * F, S, D)  # (B*F, S, D)
        if sample_attn_mask is not None:
            x2, _ = self.samp_attn(
                x_col, x_col, x_col,
                attn_mask=sample_attn_mask,
                need_weights=False,
            )
        else:
            x2, _ = self.samp_attn(x_col, x_col, x_col, need_weights=False)

        x_col = x_col + self.drop(x2)
        x_col = self.ln_samp(x_col)
        x = x_col.reshape(B, F, S, D).permute(0, 2, 1, 3).contiguous()    # (B, S, F, D)

        # 3) Position-wise MLP
        x2 = self.mlp(x)
        x = x + self.drop(x2)
        x = self.ln_mlp(x)
        return x


class InterventionalPFN(nn.Module):
    """
    PFN-like regressor for interventional causal data with two-way attention.

    Handles the output of InterventionalDataset:
    - Observational data (train): X_obs, T_obs, Y_obs
    - Interventional data (test): X_intv, T_intv
    
    The intervened feature columns (T_obs, T_intv) get special embeddings and positional encodings.
    
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
    ):
        super().__init__()
        self.num_features = num_features  # L (excluding intervened column)
        self.d_model = d_model
        self.output_dim = output_dim
        self.normalize_features = normalize_features

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

        # Learnable role embeddings
        self.obs_T_embed = self._create_role_embedding(1, 1, self.d_model)
        self.obs_label_embed = self._create_role_embedding(1, 1, self.d_model)
        self.obs_feature_embed = self._create_role_embedding(1, 1, self.d_model)
        self.intv_T_embed = self._create_role_embedding(1, 1, self.d_model)
        self.intv_label_embed = self._create_role_embedding(1, 1, self.d_model)
        self.intv_feature_embed = self._create_role_embedding(1, 1, self.d_model)

        # Stacked two-way attention blocks
        self.blocks = nn.ModuleList([
            TwoWayBlock(d_model, heads_feat, heads_samp, dropout=dropout, hidden_mult=hidden_mult)
            for _ in range(depth)
        ])

        # Output projection from D to desired output_dim per test token
        self.regression_head = nn.Linear(d_model, output_dim)

        # Cache for sample-attention masks keyed by (N, M, device)
        self._mask_cache: Dict[Tuple[int, int, torch.device], torch.Tensor] = {}

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

    def _build_sample_attn_mask(self, N: int, M: int, device: torch.device) -> torch.Tensor:
        """
        Build or fetch a Boolean attention mask over S = N + M samples.

        Masking policy:
            - Prevent observational samples from attending to interventional samples.
            - Prevent interventional samples from attending to each other.
            - Allow interventional samples to attend to observational samples.
        """
        key = (N, M, device)
        if key in self._mask_cache:
            return self._mask_cache[key]

        S = N + M
        mask = torch.zeros((S, S), dtype=torch.bool, device=device)
        if M > 0:
            mask[:N, N:S] = True   # obs cannot look at intv
            mask[N:S, N:S] = True  # intv cannot look at intv
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
        cell_emb = self.cell_mlp(X_with_T.unsqueeze(-1))  # (B, S, L, D)

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

    def forward(
        self,
        X_obs: torch.Tensor,
        T_obs: torch.Tensor,
        Y_obs: torch.Tensor,
        X_intv: torch.Tensor,
        T_intv: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass to produce predictions for interventional test samples.

        Args:
            X_obs: (B, N, L) - observational features (train)
            T_obs: (B, N, 1) - observational intervened feature (train)
            Y_obs: (B, N) or (B, N, 1) - observational targets (train)
            X_intv: (B, M, L) - interventional features (test)
            T_intv: (B, M, 1) - interventional intervened feature (test)

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

        # Build sample-wise attention mask and apply blocks
        samp_mask = self._build_sample_attn_mask(N, M, device)  # (S, S)

        for blk in self.blocks:
            x = blk(x, sample_attn_mask=samp_mask)

        # Readout: take the label column at position L+1 for interventional (test) rows only
        label_pos = self.num_features + 1  # L+1 (after L regular + 1 intervened)
        h_intv = x[:, N:, label_pos, :]  # (B, M, D)
        predictions = self.regression_head(h_intv)  # (B, M, output_dim)

        # Backward-compatible squeeze when output_dim == 1
        if self.output_dim == 1:
            predictions = predictions.squeeze(-1)  # (B, M)

        return {"predictions": predictions}


if __name__ == "__main__":
    torch.manual_seed(0)
    B, N, M, L = 2, 16, 5, 7  # batch, train samples, test samples, features

    # Simulate interventional dataset output
    X_obs = torch.randn(B, N, L)
    T_obs = torch.randn(B, N, 1)
    Y_obs = torch.randn(B, N)
    X_intv = torch.randn(B, M, L)
    T_intv = torch.randn(B, M, 1)

    # Test with default single output
    print("=== Single Output (Regression) ===")
    model = InterventionalPFN(
        num_features=L,
        d_model=128,
        depth=2,
        heads_feat=4,
        heads_samp=4,
        dropout=0.1,
    )
    out = model(X_obs, T_obs, Y_obs, X_intv, T_intv)
    print("predictions shape:", out["predictions"].shape)  # (B, M)
    print("Sample prediction:", out["predictions"][0, :3])

    # Test with high-dimensional output
    print("\n=== High-Dimensional Output ===")
    output_dim = 10
    model_hd = InterventionalPFN(
        num_features=L,
        d_model=128,
        depth=2,
        heads_feat=4,
        heads_samp=4,
        dropout=0.1,
        output_dim=output_dim,
    )
    out_hd = model_hd(X_obs, T_obs, Y_obs, X_intv, T_intv)
    print("high-dim predictions shape:", out_hd["predictions"].shape)  # (B, M, output_dim)

    # Test without normalization
    print("\n=== Without Internal Normalization ===")
    model_no_norm = InterventionalPFN(
        num_features=L,
        d_model=128,
        depth=2,
        heads_feat=4,
        heads_samp=4,
        dropout=0.1,
        normalize_features=False,
    )
    out_no_norm = model_no_norm(X_obs, T_obs, Y_obs, X_intv, T_intv)
    print("predictions shape (no norm):", out_no_norm["predictions"].shape)  # (B, M)
    
    # Test parameter count
    print("\n=== Model Statistics ===")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
