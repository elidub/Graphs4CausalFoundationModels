"""
Hybrid Graph-Conditioned Interventional PFN

This model uses a hybrid approach where half of the attention heads use hard graph-based
attention masking while the other half use standard unconstrained attention. This allows
the model to learn both graph-constrained patterns and discover additional relationships
beyond the provided causal structure.

Key features:
- Split attention heads: first half uses graph masking, second half is unconstrained
- Causal graph provided as adjacency matrix
- Flexible balance between respecting and learning beyond causal structure

Author: Generated for CausalPriorFitting
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math


class MLP(nn.Module):
    """
    Two-layer feed-forward block with SwiGLU activation and dropout.
    
    SwiGLU uses a gated linear unit with SiLU (Swish) activation:
        SwiGLU(x) = (W1*x) ⊗ silu(W_gate*x) * W2
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
    """Generic 2-layer MLP with SwiGLU activation for input embeddings."""
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


class HybridGraphConditionedTwoWayBlock(nn.Module):
    """
    Two-way attention block with hybrid graph conditioning.
    
    Feature attention heads are split:
    - First half: Uses hard attention masking based on causal graph
    - Second half: Standard unconstrained attention
    
    This allows the model to both respect the causal structure and learn
    additional patterns beyond it.
    
    Args:
        dim: Model dimension
        heads_feat: Number of attention heads for feature-wise attention (must be even)
        heads_samp: Number of attention heads for sample-wise attention
        dropout: Dropout rate
        hidden_mult: Hidden layer multiplier for MLPs
    """
    def __init__(self, dim: int, heads_feat: int, heads_samp: int, dropout: float = 0.0, hidden_mult: int = 4):
        super().__init__()
        
        if heads_feat % 2 != 0:
            raise ValueError(f"heads_feat must be even for hybrid attention, got {heads_feat}")
        
        self.dim = dim
        self.heads_feat = heads_feat
        self.heads_constrained = heads_feat // 2  # First half: constrained by graph
        self.heads_unconstrained = heads_feat // 2  # Second half: unconstrained
        
        # Feature attention with graph conditioning
        self.feat_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads_feat,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_feat = nn.LayerNorm(dim)

        # Sample attention (train)
        self.samp_attn_train = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads_samp,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_samp_train = nn.LayerNorm(dim)
        
        # Sample attention (test)
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
        Forward pass with hybrid graph conditioning.
        
        Args:
            x: (B, S, F, D) where S = N_train + N_test
            N_train: Number of training samples
            N_test: Number of test samples
            attn_mask: (B*S, F, F) or (F, F) - applied only to first half of heads
        
        Returns:
            x: (B, S, F, D)
        """
        B, S, F, D = x.shape
        assert S == N_train + N_test

        # 1) Feature-attention with hybrid masking - pre-layer norm
        x_row = x.reshape(B * S, F, D)  # (B*S, F, D)
        x_norm = self.ln_feat(x_row)
        
        # Apply hybrid attention mask if provided
        if attn_mask is not None:
            # Prepare mask: boolean to additive conversion
            # True/1 -> 0.0 (allow attention), False/0 -> -inf (block attention)
            if attn_mask.dtype == torch.bool:
                additive_mask = torch.zeros_like(attn_mask, dtype=x.dtype)
                additive_mask.masked_fill_(~attn_mask, float('-inf'))
            else:
                additive_mask = torch.where(
                    attn_mask == 1,
                    torch.zeros_like(attn_mask, dtype=x.dtype),
                    torch.full_like(attn_mask, float('-inf'), dtype=x.dtype)
                )
            
            # Expand mask to match multi-head attention format
            # Shape: (B, F, F) or (B*S, F, F) -> (B*S*num_heads, F, F)
            if additive_mask.dim() == 2:
                # (F, F) -> (B*S*num_heads, F, F)
                additive_mask = additive_mask.unsqueeze(0).repeat(B * S * self.heads_feat, 1, 1)
            elif additive_mask.dim() == 3:
                # (B, F, F) -> (B*S*num_heads, F, F)
                # First repeat for S dimension, then for heads dimension
                additive_mask = additive_mask.unsqueeze(1).repeat(1, S, 1, 1)  # (B, S, F, F)
                additive_mask = additive_mask.reshape(B * S, F, F)  # (B*S, F, F)
                additive_mask = additive_mask.unsqueeze(1).repeat(1, self.heads_feat, 1, 1)  # (B*S, heads, F, F)
                additive_mask = additive_mask.reshape(B * S * self.heads_feat, F, F)  # (B*S*heads, F, F)
            
            # Create hybrid mask: first half constrained, second half free
            # Split into head groups and modify
            hybrid_mask = additive_mask.reshape(B * S, self.heads_feat, F, F)
            # Set second half of heads to zero (no masking)
            hybrid_mask[:, self.heads_constrained:, :, :] = 0.0
            # Reshape back for PyTorch MultiheadAttention: (B*S*num_heads, F, F)
            hybrid_mask = hybrid_mask.reshape(B * S * self.heads_feat, F, F)
            
            x2, _ = self.feat_attn(x_norm, x_norm, x_norm, attn_mask=hybrid_mask, need_weights=False)
        else:
            x2, _ = self.feat_attn(x_norm, x_norm, x_norm, need_weights=False)
        
        x_row = x_row + self.drop(x2)
        x = x_row.reshape(B, S, F, D)

        # 2) Sample-attention (within column) - separate for train and test - pre-layer norm
        x_col = x.permute(0, 2, 1, 3).contiguous().reshape(B * F, S, D)
        
        # Split into train and test
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
        
        # Concatenate back
        x_col = torch.cat([x_train, x_test], dim=1)
        x = x_col.reshape(B, F, S, D).permute(0, 2, 1, 3).contiguous()

        # 3) Position-wise MLP - pre-layer norm
        x_norm = self.ln_mlp(x)
        x2 = self.mlp(x_norm)
        x = x + self.drop(x2)
        return x


class HybridGraphConditionedInterventionalPFN(nn.Module):
    """
    Interventional PFN with hybrid graph conditioning.
    
    Uses split attention heads where half are constrained by the causal graph
    and half are free to learn any pattern. This provides a balance between
    respecting known causal structure and discovering additional relationships.
    
    Architecture identical to GraphConditionedInterventionalPFN except for
    the hybrid attention mechanism in the two-way blocks.
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
        use_same_row_mlp: bool = True,
        n_sample_attention_sink_rows: int = 0,
        n_feature_attention_sink_cols: int = 0,
    ):
        """
        Args:
            num_features: Number of input features (L)
            d_model: Model dimension
            depth: Number of transformer blocks
            heads_feat: Number of attention heads for feature-wise attention (must be even!)
            heads_samp: Number of attention heads for sample-wise attention
            dropout: Dropout probability
            hidden_mult: Multiplier for feedforward hidden dimension
            output_dim: Output dimension (1 for regression)
            normalize_features: Whether to apply quantile + standard normalization
            use_same_row_mlp: Whether to use the same row MLP for train and test
            n_sample_attention_sink_rows: Number of random sink rows to prepend
            n_feature_attention_sink_cols: Number of random sink columns to prepend
        """
        super().__init__()
        
        if heads_feat % 2 != 0:
            raise ValueError(f"heads_feat must be even for hybrid attention, got {heads_feat}")
        
        self.num_features = num_features
        self.d_model = d_model
        self.depth = depth
        self.output_dim = output_dim
        self.normalize_features = normalize_features
        self.n_sample_attention_sink_rows = n_sample_attention_sink_rows
        self.n_feature_attention_sink_cols = n_feature_attention_sink_cols
        
        # Row-wise MLPs
        if use_same_row_mlp:
            self.row_mlp_train = InputMLP(num_features + 1, d_model, hidden_mult, dropout)
            self.row_mlp_test = self.row_mlp_train
        else:
            self.row_mlp_train = InputMLP(num_features + 1, d_model, hidden_mult, dropout)
            self.row_mlp_test = InputMLP(num_features + 1, d_model, hidden_mult, dropout)
        
        # Scalar MLPs
        self.cell_mlp = InputMLP(1, d_model, hidden_mult, dropout)
        self.label_mlp_train = InputMLP(1, d_model, hidden_mult, dropout)
        
        # Feature positional encodings
        feat_pos = self._build_feature_positional(num_features + 2, d_model)
        self.register_buffer("feature_positional", feat_pos.unsqueeze(0).unsqueeze(0), persistent=False)
        
        # Learnable scaling
        self.row_scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(2.0)))
        self.cell_scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(2.0)))
        
        # Attention sinks
        if n_sample_attention_sink_rows > 0:
            self.sink_rows_x = nn.Parameter(torch.randn(1, n_sample_attention_sink_rows, num_features + 2, d_model) * 0.02)
            self.sink_rows_y = nn.Parameter(torch.randn(1, n_sample_attention_sink_rows, d_model) * 0.02)
        else:
            self.sink_rows_x = None
            self.sink_rows_y = None
        
        if n_feature_attention_sink_cols > 0:
            self.sink_cols = nn.Parameter(torch.randn(1, 1, n_feature_attention_sink_cols, d_model) * 0.02)
        else:
            self.sink_cols = None
        
        # Role embeddings
        self.obs_T_embed = self._create_role_embedding(1, 1, d_model)
        self.obs_label_embed = self._create_role_embedding(1, 1, d_model)
        self.obs_feature_embed = self._create_role_embedding(1, 1, d_model)
        self.intv_T_embed = self._create_role_embedding(1, 1, d_model)
        self.intv_label_embed = self._create_role_embedding(1, 1, d_model)
        self.intv_feature_embed = self._create_role_embedding(1, 1, d_model)
        
        # Hybrid two-way attention blocks
        self.blocks = nn.ModuleList([
            HybridGraphConditionedTwoWayBlock(d_model, heads_feat, heads_samp, dropout=dropout, hidden_mult=hidden_mult)
            for _ in range(depth)
        ])
        
        # Output head
        self.regression_head = nn.Linear(d_model, output_dim)
        
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
    
    def _create_role_embedding(self, *shape, std=0.02):
        """Helper to create and initialize a role embedding parameter."""
        embed = nn.Parameter(torch.zeros(*shape))
        nn.init.normal_(embed, std=std)
        return embed
    
    @staticmethod
    def _build_feature_positional(num_tokens: int, dim: int) -> torch.Tensor:
        """Build sinusoidal positional encodings."""
        position = torch.arange(num_tokens, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pos_enc = torch.zeros(num_tokens, dim)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc
    
    @staticmethod
    def _normalize_features(
        X_train: torch.Tensor,
        X_test: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply quantile transformation followed by standard normalization."""
        B, N, F = X_train.shape
        M = X_test.shape[1]
        
        # Sort training data for quantile transform
        X_train_sorted, _ = torch.sort(X_train, dim=1)
        
        def quantile_transform(X, X_sorted):
            B, S, F = X.shape
            N_train = X_sorted.shape[1]
            
            X_expanded = X.unsqueeze(2)
            X_sorted_expanded = X_sorted.unsqueeze(1)
            
            ranks = (X_expanded >= X_sorted_expanded).sum(dim=2).float()
            quantiles = ranks / (N_train + 1)
            
            return quantiles
        
        X_train_quantiles = quantile_transform(X_train, X_train_sorted)
        X_test_quantiles = quantile_transform(X_test, X_train_sorted)
        
        mean = X_train_quantiles.mean(dim=1, keepdim=True)
        std = X_train_quantiles.std(dim=1, keepdim=True, unbiased=False)
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
        """Embed features with special treatment for intervened column."""
        B, S, L = X.shape
        assert L == self.num_features
        assert T.shape == (B, S, 1)
        
        # Concatenate T to X for row-wise embedding
        X_with_T = torch.cat([X, T], dim=2)
        
        # Row-wise embedding
        row_emb = row_mlp(X_with_T)  # (B, S, D)
        
        # Cell-wise embeddings
        X_cells = self.cell_mlp(X.unsqueeze(-1))  # (B, S, L, D)
        T_cells = self.cell_mlp(T.unsqueeze(-1))  # (B, S, 1, D)
        
        # Combine row and cell embeddings
        row_emb_expanded = row_emb.unsqueeze(2).expand(-1, -1, L + 1, -1)
        cell_embs = torch.cat([X_cells, T_cells], dim=2)  # (B, S, L+1, D)
        
        feat_emb = self.row_scale * row_emb_expanded + self.cell_scale * cell_embs
        
        # Add role embeddings
        if is_intvn:
            feat_emb[:, :, :L, :] += self.intv_feature_embed
            feat_emb[:, :, L:, :] += self.intv_T_embed
        else:
            feat_emb[:, :, :L, :] += self.obs_feature_embed
            feat_emb[:, :, L:, :] += self.obs_T_embed
        
        return feat_emb
    
    def _embed_labels(self, Y: torch.Tensor) -> torch.Tensor:
        """Embed label column."""
        if Y.dim() == 2:
            Y = Y.unsqueeze(-1)
        lab_emb = self.label_mlp_train(Y)  # (B, N, D)
        lab_emb += self.obs_label_embed
        return lab_emb
    
    def _prepare_attention_mask(
        self,
        adjacency_matrix: torch.Tensor,
        n_sink_cols: int,
    ) -> torch.Tensor:
        """
        Prepare attention mask from adjacency matrix.
        
        Args:
            adjacency_matrix: (B, L+2, L+2) causal graph
            n_sink_cols: Number of sink columns to add
        
        Returns:
            Attention mask: (B, n_sink_cols + L+2, n_sink_cols + L+2)
        """
        B, F, _ = adjacency_matrix.shape
        device = adjacency_matrix.device
        
        if n_sink_cols > 0:
            # Expand to include sink columns (all ones for sinks)
            full_size = n_sink_cols + F
            mask = torch.ones(B, full_size, full_size, dtype=adjacency_matrix.dtype, device=device)
            mask[:, n_sink_cols:, n_sink_cols:] = adjacency_matrix
            return mask
        else:
            return adjacency_matrix

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
        Forward pass with hybrid graph conditioning.
        
        Args:
            X_obs: (B, N, L) observational features
            T_obs: (B, N, 1) observational treatment
            Y_obs: (B, N) observational outcomes
            X_intv: (B, M, L) interventional features
            T_intv: (B, M, 1) interventional treatment
            adjacency_matrix: (B, L+2, L+2) causal graph
                Ordering: [X_0, ..., X_{L-1}, T, Y]
                A[i,j] = 1 means i can attend to j
        
        Returns:
            Dict with "predictions": (B, M) or (B, M, output_dim)
        """
        B, N, L = X_obs.shape
        assert L == self.num_features
        M = X_intv.shape[1]
        device = X_obs.device
        
        # Ensure T has correct shape
        if T_obs.dim() == 2:
            T_obs = T_obs.unsqueeze(-1)
        if T_intv.dim() == 2:
            T_intv = T_intv.unsqueeze(-1)
        
        # Validate adjacency matrix
        assert adjacency_matrix.shape == (B, L + 2, L + 2), \
            f"Expected adjacency matrix shape ({B}, {L + 2}, {L + 2}), got {adjacency_matrix.shape}"
        
        # Normalize features if enabled
        if self.normalize_features:
            X_obs_with_T = torch.cat([X_obs, T_obs], dim=2)
            X_intv_with_T = torch.cat([X_intv, T_intv], dim=2)
            
            X_obs_norm, X_intv_norm = self._normalize_features(X_obs_with_T, X_intv_with_T)
            
            X_obs_norm, T_obs_norm = X_obs_norm[:, :, :L], X_obs_norm[:, :, L:L+1]
            X_intv_norm, T_intv_norm = X_intv_norm[:, :, :L], X_intv_norm[:, :, L:L+1]
        else:
            X_obs_norm, T_obs_norm = X_obs, T_obs
            X_intv_norm, T_intv_norm = X_intv, T_intv
        
        # Embed features
        feat_obs = self._embed_features(X_obs_norm, T_obs_norm, self.row_mlp_train, is_intvn=False)
        feat_intv = self._embed_features(X_intv_norm, T_intv_norm, self.row_mlp_test, is_intvn=True)
        feat_all = torch.cat([feat_obs, feat_intv], dim=1)
        
        # Embed labels
        lab_obs = self._embed_labels(Y_obs)
        lab_intv = self.intv_label_embed.expand(B, M, self.d_model)
        lab_all = torch.cat([lab_obs, lab_intv], dim=1)
        
        # Stack features + label
        x = torch.cat([feat_all, lab_all.unsqueeze(2)], dim=2)  # (B, S, L+2, D)
        
        # Add feature positional encodings
        x = x + self.feature_positional
        
        # Add attention sinks
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
        
        # Prepare attention mask
        attn_mask = self._prepare_attention_mask(adjacency_matrix, n_sink_cols)
        
        # Apply transformer blocks with hybrid masking
        for blk in self.blocks:
            x = blk(x, N_train=n_sink_rows + N, N_test=M, attn_mask=attn_mask)
        
        # Readout from label column for interventional samples
        label_pos = n_sink_cols + self.num_features + 1
        test_start_idx = n_sink_rows + N
        h_intv = x[:, test_start_idx:, label_pos, :]
        predictions = self.regression_head(h_intv)
        
        if self.output_dim == 1:
            predictions = predictions.squeeze(-1)
        
        return {"predictions": predictions}


if __name__ == "__main__":
    torch.manual_seed(0)
    
    print("=" * 80)
    print("HybridGraphConditionedInterventionalPFN Test Suite")
    print("=" * 80)
    
    # Test 1: Basic forward pass
    print("\n" + "=" * 80)
    print("Test 1: Basic Forward Pass with Hybrid Masking")
    print("=" * 80)
    
    B, N, M, L = 2, 10, 5, 3
    model = HybridGraphConditionedInterventionalPFN(
        num_features=L,
        d_model=64,
        depth=2,
        heads_feat=4,  # Must be even
        heads_samp=2,
    )
    
    X_obs = torch.randn(B, N, L)
    T_obs = torch.randn(B, N, 1)
    Y_obs = torch.randn(B, N)
    X_intv = torch.randn(B, M, L)
    T_intv = torch.randn(B, M, 1)
    
    # Create adjacency matrix: [X_0, X_1, X_2, T, Y]
    adj = torch.zeros(B, L+2, L+2)
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
    assert predictions.shape == (B, M)
    print(f"✓ Test 1 passed!")
    print(f"Note: First {model.blocks[0].heads_constrained} heads use graph masking, "
          f"last {model.blocks[0].heads_unconstrained} heads are unconstrained")
    
    # Test 2: Verify heads_feat must be even
    print("\n" + "=" * 80)
    print("Test 2: Verify Even Number of Heads Required")
    print("=" * 80)
    
    try:
        bad_model = HybridGraphConditionedInterventionalPFN(
            num_features=L,
            d_model=64,
            depth=1,
            heads_feat=3,  # Odd number - should fail
            heads_samp=2,
        )
        print("✗ Test 2 failed: Should have raised ValueError for odd heads_feat")
    except ValueError as e:
        print(f"✓ Test 2 passed: Correctly rejected odd heads_feat")
        print(f"  Error message: {e}")
    
    # Test 3: With attention sinks
    print("\n" + "=" * 80)
    print("Test 3: With Attention Sinks")
    print("=" * 80)
    
    model_sinks = HybridGraphConditionedInterventionalPFN(
        num_features=L,
        d_model=64,
        depth=2,
        heads_feat=4,
        heads_samp=2,
        n_sample_attention_sink_rows=3,
        n_feature_attention_sink_cols=2,
    )
    
    output = model_sinks(X_obs, T_obs, Y_obs, X_intv, T_intv, adj)
    predictions = output['predictions']
    
    print(f"Output shape: {predictions.shape}")
    assert predictions.shape == (B, M)
    print("✓ Test 3 passed!")
    
    # Test 4: Compare with fully connected vs sparse graph
    print("\n" + "=" * 80)
    print("Test 4: Graph Structure Affects Predictions")
    print("=" * 80)
    
    adj_full = torch.ones(B, L+2, L+2)  # Fully connected
    adj_sparse = torch.eye(L+2).unsqueeze(0).expand(B, -1, -1)  # Only self-attention
    
    out_full = model(X_obs, T_obs, Y_obs, X_intv, T_intv, adj_full)
    out_sparse = model(X_obs, T_obs, Y_obs, X_intv, T_intv, adj_sparse)
    
    diff = torch.abs(out_full['predictions'] - out_sparse['predictions']).max().item()
    print(f"Max prediction difference: {diff:.6f}")
    assert diff > 1e-6, "Different graphs should produce different predictions"
    print("✓ Test 4 passed!")
    
    # Test 5: High-dimensional output
    print("\n" + "=" * 80)
    print("Test 5: High-Dimensional Output")
    print("=" * 80)
    
    output_dim = 10
    model_hd = HybridGraphConditionedInterventionalPFN(
        num_features=L,
        d_model=64,
        depth=2,
        heads_feat=4,
        heads_samp=2,
        output_dim=output_dim,
    )
    
    out_hd = model_hd(X_obs, T_obs, Y_obs, X_intv, T_intv, adj)
    print(f"Output shape: {out_hd['predictions'].shape}")
    print(f"Expected: ({B}, {M}, {output_dim})")
    assert out_hd['predictions'].shape == (B, M, output_dim)
    print("✓ Test 5 passed!")
    
    # Model statistics
    print("\n" + "=" * 80)
    print("Model Statistics")
    print("=" * 80)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"\nHybrid Configuration:")
    print(f"  Total feature attention heads: {model.blocks[0].heads_feat}")
    print(f"  Constrained heads (first half): {model.blocks[0].heads_constrained}")
    print(f"  Unconstrained heads (second half): {model.blocks[0].heads_unconstrained}")
    
    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
