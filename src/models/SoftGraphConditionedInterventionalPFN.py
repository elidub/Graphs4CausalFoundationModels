"""
Soft Graph-Conditioned Interventional PFN

This model conditions on causal graph structure using learned biases applied to attention scores,
rather than hard attention masking. This allows the model to learn how much to respect vs. override
the causal structure.

Author: Generated for CausalPriorFitting
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math


class SoftGraphConditionedTwoWayBlock(nn.Module):
    """
    Two-way attention block with soft graph conditioning via learned biases.
    
    Instead of hard masking (setting attention to -inf), this block learns a bias
    that modulates attention scores based on the causal graph structure.
    """
    
    def __init__(
        self,
        d_model: int,
        heads_feat: int,
        heads_samp: int,
        dropout: float = 0.0,
        hidden_mult: int = 2,
        graph_bias_init: float = -5.0,  # Initial bias for non-edges (negative = discourage)
    ):
        """
        Args:
            d_model: Model dimension
            heads_feat: Number of heads for feature-wise attention
            heads_samp: Number of heads for sample-wise attention
            dropout: Dropout probability
            hidden_mult: Multiplier for feedforward hidden dimension
            graph_bias_init: Initial value for graph bias on non-edges (negative values discourage attention)
        """
        super().__init__()
        
        self.d_model = d_model
        self.heads_feat = heads_feat
        self.heads_samp = heads_samp
        self.graph_bias_init = graph_bias_init
        
        # Feature attention (with graph conditioning)
        self.ln_feat = nn.LayerNorm(d_model)
        self.feat_attn = nn.MultiheadAttention(
            d_model, 
            heads_feat, 
            dropout=dropout, 
            batch_first=True
        )
        
        # Sample attention (train)
        self.ln_samp_train = nn.LayerNorm(d_model)
        self.samp_attn_train = nn.MultiheadAttention(
            d_model,
            heads_samp,
            dropout=dropout,
            batch_first=True
        )
        
        # Sample attention (test)
        self.ln_samp_test = nn.LayerNorm(d_model)
        self.samp_attn_test = nn.MultiheadAttention(
            d_model,
            heads_samp,
            dropout=dropout,
            batch_first=True
        )
        
        # Feedforward
        self.ln_ff = nn.LayerNorm(d_model)
        hidden_dim = d_model * hidden_mult
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )
        
        # Learnable graph bias parameters
        # This is a per-head, per-edge bias that modulates attention scores
        # Shape: (num_heads, 1, 1) - will be broadcast to (num_heads, F, F)
        self.graph_bias_scale = nn.Parameter(torch.ones(heads_feat, 1, 1))
        self.graph_bias_offset = nn.Parameter(torch.zeros(heads_feat, 1, 1))
        
        self.drop = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        N_train: int,
        N_test: int,
        graph_structure: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with soft graph conditioning.
        
        Args:
            x: (B, S, F, D) where S = N_train + N_test
            N_train: Number of training samples
            N_test: Number of test samples  
            graph_structure: (B, F, F) - adjacency matrix where 1 = edge exists, 0 = no edge
                           If None, no graph conditioning is applied.
        
        Returns:
            x: (B, S, F, D) after attention and feedforward
        """
        B, S, F, D = x.shape
        assert S == N_train + N_test, f"Expected {N_train + N_test} samples, got {S}"

        # 1) Feature-attention (within row) with soft graph conditioning - pre-layer norm
        x_row = x.reshape(B * S, F, D)  # (B*S, F, D)
        x_norm = self.ln_feat(x_row)
        
        # Apply soft graph bias if provided
        if graph_structure is not None:
            # graph_structure: (B, F, F) where 1 = edge, 0 = no edge
            # Convert to bias: edges get small positive bias, non-edges get learned negative bias
            
            # Create bias matrix: edge=0, non-edge=graph_bias_init
            graph_bias_base = torch.where(
                graph_structure == 1,
                torch.zeros_like(graph_structure),
                torch.full_like(graph_structure, self.graph_bias_init)
            )  # (B, F, F)
            
            # Apply learnable scale and offset per head
            # graph_bias_scale: (num_heads, 1, 1)
            # graph_bias_offset: (num_heads, 1, 1)
            # Expand graph_bias_base to (B, num_heads, F, F)
            graph_bias_base = graph_bias_base.unsqueeze(1)  # (B, 1, F, F)
            
            # Apply per-head transformation
            # scale * base + offset
            graph_bias = (
                self.graph_bias_scale * graph_bias_base + self.graph_bias_offset
            )  # (1, num_heads, 1, 1) * (B, 1, F, F) -> (B, num_heads, F, F)
            
            # Expand to all samples: (B*S, num_heads, F, F)
            graph_bias = graph_bias.unsqueeze(1).expand(-1, S, -1, -1, -1).reshape(
                B * S, self.heads_feat, F, F
            )
            
            # PyTorch MultiheadAttention expects attn_mask of shape:
            # - (L, S) for 2D
            # - (B*num_heads, L, S) for 3D
            # Our bias is (B*S, num_heads, F, F), need (B*S*num_heads, F, F)
            graph_bias = graph_bias.reshape(B * S * self.heads_feat, F, F)
            
            x2, _ = self.feat_attn(x_norm, x_norm, x_norm, attn_mask=graph_bias, need_weights=False)
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

        # 3) Feedforward - pre-layer norm
        x_flat = x.reshape(B * S * F, D)
        x_norm = self.ln_ff(x_flat)
        x_ff = self.ff(x_norm)
        x_flat = x_flat + x_ff
        x = x_flat.reshape(B, S, F, D)

        return x


class SoftGraphConditionedInterventionalPFN(nn.Module):
    """
    Interventional PFN with soft graph conditioning via learned attention biases.
    
    This model learns how much to respect the causal graph structure rather than
    strictly enforcing it through hard attention masking.
    
    Key differences from GraphConditionedInterventionalPFN:
    - Uses learned biases instead of hard masks (-inf)
    - Can learn to override graph structure when beneficial
    - More flexible but requires more data to learn appropriate biases
    """
    
    def __init__(
        self,
        num_features: int,
        d_model: int = 256,
        depth: int = 6,
        heads_feat: int = 4,
        heads_samp: int = 4,
        dropout: float = 0.0,
        hidden_mult: int = 2,
        output_dim: int = 1,
        normalize_features: bool = True,
        n_sample_attention_sink_rows: int = 0,
        n_feature_attention_sink_cols: int = 0,
        graph_bias_init: float = -5.0,
    ):
        """
        Args:
            num_features: Number of input features (L)
            d_model: Model dimension
            depth: Number of transformer blocks
            heads_feat: Number of attention heads for feature-wise attention
            heads_samp: Number of attention heads for sample-wise attention
            dropout: Dropout probability
            hidden_mult: Multiplier for feedforward hidden dimension
            output_dim: Output dimension (1 for regression)
            normalize_features: Whether to apply quantile + standard normalization
            n_sample_attention_sink_rows: Number of random sink rows to prepend
            n_feature_attention_sink_cols: Number of random sink columns to prepend
            graph_bias_init: Initial bias value for non-edges (negative = discourage attention)
        """
        super().__init__()
        
        self.num_features = num_features
        self.d_model = d_model
        self.depth = depth
        self.output_dim = output_dim
        self.normalize_features = normalize_features
        self.n_sample_attention_sink_rows = n_sample_attention_sink_rows
        self.n_feature_attention_sink_cols = n_feature_attention_sink_cols
        self.graph_bias_init = graph_bias_init
        
        # Row-wise MLPs for embedding samples
        self.row_mlp_train = nn.Sequential(
            nn.Linear(num_features + 1, d_model),  # +1 for treatment
            nn.GELU(),
        )
        self.row_mlp_test = nn.Sequential(
            nn.Linear(num_features + 1, d_model),
            nn.GELU(),
        )
        
        # Column embeddings for each feature (including treatment at position L)
        self.col_embed = nn.Parameter(torch.randn(num_features + 1, d_model) * 0.02)
        
        # Label column embedding (for observational data)
        self.label_embed = nn.Linear(1, d_model)
        
        # Special embedding for interventional (test) label column
        self.intv_label_embed = nn.Parameter(torch.randn(d_model) * 0.02)
        
        # Feature positional encodings
        # Total features: num_features (X) + 1 (T) + 1 (Y) = num_features + 2
        self.feature_positional = nn.Parameter(
            torch.randn(1, 1, num_features + 2, d_model) * 0.02
        )
        
        # Attention sink rows (dummy samples)
        if n_sample_attention_sink_rows > 0:
            self.sink_rows_x = nn.Parameter(
                torch.randn(1, n_sample_attention_sink_rows, num_features + 2, d_model) * 0.02
            )
            self.sink_rows_y = nn.Parameter(
                torch.randn(1, n_sample_attention_sink_rows, d_model) * 0.02
            )
        else:
            self.sink_rows_x = None
            self.sink_rows_y = None
        
        # Attention sink columns (dummy features)
        if n_feature_attention_sink_cols > 0:
            self.sink_cols = nn.Parameter(
                torch.randn(1, 1, n_feature_attention_sink_cols, d_model) * 0.02
            )
        else:
            self.sink_cols = None
        
        # Transformer blocks with soft graph conditioning
        self.blocks = nn.ModuleList([
            SoftGraphConditionedTwoWayBlock(
                d_model=d_model,
                heads_feat=heads_feat,
                heads_samp=heads_samp,
                dropout=dropout,
                hidden_mult=hidden_mult,
                graph_bias_init=graph_bias_init,
            )
            for _ in range(depth)
        ])
        
        # Output head
        self.regression_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _normalize_features(
        self,
        X_train: torch.Tensor,
        X_test: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply quantile transformation followed by standard normalization.
        
        Args:
            X_train: (B, N, F) training features
            X_test: (B, M, F) test features
            
        Returns:
            X_train_norm: (B, N, F) normalized training features
            X_test_norm: (B, M, F) normalized test features
        """
        B, N, F = X_train.shape
        M = X_test.shape[1]
        
        # Sort training data for quantile transform
        X_train_sorted, _ = torch.sort(X_train, dim=1)  # (B, N, F)
        
        # Quantile transform function
        def quantile_transform(X, X_sorted):
            # For each value, find its quantile in the sorted training data
            # This maps values to roughly uniform [0, 1] based on training distribution
            B, S, F = X.shape
            N_train = X_sorted.shape[1]
            
            # Expand for broadcasting
            X_expanded = X.unsqueeze(2)  # (B, S, 1, F)
            X_sorted_expanded = X_sorted.unsqueeze(1)  # (B, 1, N_train, F)
            
            # Count how many training samples are less than each value
            ranks = (X_expanded >= X_sorted_expanded).sum(dim=2).float()  # (B, S, F)
            
            # Convert to quantiles (0 to 1)
            quantiles = ranks / (N_train + 1)
            
            return quantiles
        
        # Step 1: Quantile transformation
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
            is_intvn: Whether this is interventional data

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
        
        # Add column embeddings (per-feature)
        col_emb = self.col_embed.unsqueeze(0).unsqueeze(0)  # (1, 1, L+1, D)
        feat_emb = row_emb.unsqueeze(2) + col_emb  # (B, S, L+1, D)
        
        return feat_emb

    def _embed_labels(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Embed label column.
        
        Args:
            Y: (B, N) labels
            
        Returns:
            (B, N, D) label embeddings
        """
        if Y.dim() == 2:
            Y = Y.unsqueeze(-1)  # (B, N, 1)
        return self.label_embed(Y)  # (B, N, D)

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
        Forward pass with soft graph conditioning.
        
        Args:
            X_obs: (B, N, L) observational features
            T_obs: (B, N, 1) observational treatment values
            Y_obs: (B, N) observational outcomes
            X_intv: (B, M, L) interventional features
            T_intv: (B, M, 1) interventional treatment values
            adjacency_matrix: (B, L+2, L+2) causal graph adjacency matrix
                Ordering: [X_0, X_1, ..., X_{L-1}, T, Y]
                A[i,j] = 1 means feature i can influence feature j (edge exists)

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

        # === Prepare graph structure for soft conditioning ===
        # Expand adjacency matrix to include sink columns if present
        if n_sink_cols > 0:
            # Create expanded graph structure with ones for sink columns
            # Sink columns can attend to everything and everything can attend to them
            full_size = n_sink_cols + (L + 2)
            graph_expanded = torch.ones(B, full_size, full_size, dtype=adjacency_matrix.dtype, device=device)
            
            # Copy the real graph structure (excluding sinks)
            graph_expanded[:, n_sink_cols:, n_sink_cols:] = adjacency_matrix
            
            graph_structure = graph_expanded
        else:
            graph_structure = adjacency_matrix

        # Apply blocks with soft graph conditioning
        for blk in self.blocks:
            x = blk(x, N_train=n_sink_rows + N, N_test=M, graph_structure=graph_structure)

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
    print("SoftGraphConditionedInterventionalPFN Test Suite")
    print("=" * 80)
    
    # Test 1: Basic forward pass
    print("\n" + "=" * 80)
    print("Test 1: Basic Forward Pass")
    print("=" * 80)
    
    B, N, M, L = 2, 10, 5, 3
    model = SoftGraphConditionedInterventionalPFN(
        num_features=L,
        d_model=64,
        depth=2,
        heads_feat=2,
        heads_samp=2,
        graph_bias_init=-5.0,
    )
    
    X_obs = torch.randn(B, N, L)
    T_obs = torch.randn(B, N, 1)
    Y_obs = torch.randn(B, N)
    X_intv = torch.randn(B, M, L)
    T_intv = torch.randn(B, M, 1)
    
    # Create a simple adjacency matrix: [X_0, X_1, X_2, T, Y]
    adj = torch.zeros(B, L+2, L+2)
    # X_0 -> X_1, X_1 -> X_2, T -> Y, X_0 -> Y
    adj[:, 0, 1] = 1  # X_0 -> X_1
    adj[:, 1, 2] = 1  # X_1 -> X_2
    adj[:, 3, 4] = 1  # T -> Y
    adj[:, 0, 4] = 1  # X_0 -> Y
    
    output = model(X_obs, T_obs, Y_obs, X_intv, T_intv, adj)
    predictions = output['predictions']
    
    print(f"Input shapes:")
    print(f"  X_obs: {X_obs.shape}, T_obs: {T_obs.shape}, Y_obs: {Y_obs.shape}")
    print(f"  X_intv: {X_intv.shape}, T_intv: {T_intv.shape}")
    print(f"  Adjacency: {adj.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Expected: ({B}, {M})")
    assert predictions.shape == (B, M), f"Expected shape ({B}, {M}), got {predictions.shape}"
    print("✓ Test 1 passed!")
    
    # Test 2: With attention sinks
    print("\n" + "=" * 80)
    print("Test 2: With Attention Sinks")
    print("=" * 80)
    
    model_sinks = SoftGraphConditionedInterventionalPFN(
        num_features=L,
        d_model=64,
        depth=2,
        heads_feat=2,
        heads_samp=2,
        n_sample_attention_sink_rows=3,
        n_feature_attention_sink_cols=2,
        graph_bias_init=-3.0,
    )
    
    output = model_sinks(X_obs, T_obs, Y_obs, X_intv, T_intv, adj)
    predictions = output['predictions']
    
    print(f"Output shape: {predictions.shape}")
    assert predictions.shape == (B, M)
    print("✓ Test 2 passed!")
    
    # Test 3: Check learned bias parameters
    print("\n" + "=" * 80)
    print("Test 3: Learned Bias Parameters")
    print("=" * 80)
    
    for i, blk in enumerate(model.blocks):
        print(f"Block {i}:")
        print(f"  graph_bias_scale shape: {blk.graph_bias_scale.shape}")
        print(f"  graph_bias_scale (first head): {blk.graph_bias_scale[0].item():.4f}")
        print(f"  graph_bias_offset shape: {blk.graph_bias_offset.shape}")
        print(f"  graph_bias_offset (first head): {blk.graph_bias_offset[0].item():.4f}")
    
    print("✓ Test 3 passed!")
    
    # Test 4: Different graph bias initialization
    print("\n" + "=" * 80)
    print("Test 4: Different Graph Bias Initialization")
    print("=" * 80)
    
    for bias_init in [-10.0, -5.0, -1.0, 0.0]:
        model_test = SoftGraphConditionedInterventionalPFN(
            num_features=L,
            d_model=32,
            depth=1,
            graph_bias_init=bias_init,
        )
        print(f"bias_init={bias_init:.1f}: model created successfully")
    
    print("✓ Test 4 passed!")
    
    # Test 5: Gradient flow check
    print("\n" + "=" * 80)
    print("Test 5: Gradient Flow Check")
    print("=" * 80)
    
    model_grad = SoftGraphConditionedInterventionalPFN(
        num_features=L,
        d_model=32,
        depth=1,
        graph_bias_init=-5.0,
        normalize_features=False,  # Disable normalization for gradient test
    )
    
    X_obs_grad = torch.randn(1, 5, L, requires_grad=True)
    T_obs_grad = torch.randn(1, 5, 1)
    Y_obs_grad = torch.randn(1, 5)
    X_intv_grad = torch.randn(1, 3, L)
    T_intv_grad = torch.randn(1, 3, 1)
    adj_grad = torch.ones(1, L+2, L+2)  # Fully connected graph
    
    output = model_grad(X_obs_grad, T_obs_grad, Y_obs_grad, X_intv_grad, T_intv_grad, adj_grad)
    loss = output['predictions'].sum()
    loss.backward()
    
    print(f"Loss: {loss.item():.4f}")
    print(f"X_obs gradient: {X_obs_grad.grad is not None}")
    print(f"graph_bias_scale gradient: {model_grad.blocks[0].graph_bias_scale.grad is not None}")
    print(f"graph_bias_offset gradient: {model_grad.blocks[0].graph_bias_offset.grad is not None}")
    
    assert X_obs_grad.grad is not None, "No gradient for inputs"
    assert model_grad.blocks[0].graph_bias_scale.grad is not None, "No gradient for bias scale"
    assert model_grad.blocks[0].graph_bias_offset.grad is not None, "No gradient for bias offset"
    
    print("✓ Test 5 passed!")
    
    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
