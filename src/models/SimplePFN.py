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

For the label column (a special feature added alongside all input features),
train labels are encoded directly, while test labels are replaced by a learned
[MASK] embedding. A train-only positional encoding is added to the label column
to let the model distinguish it from feature columns.
"""

from __future__ import annotations
from typing import Optional, Dict
from contextlib import nullcontext

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

    Shape:
        - Input:  (*, dim)
        - Output: (*, dim)  (same shape as input)

    Notes:
        This is used position-wise across tokens (no change in sequence axes).
    """
    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = hidden_mult * dim
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Arbitrary leading shape with final dimension == dim.

        Returns:
            Tensor with same shape as ``x``.
        """
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TwoWayBlock(nn.Module):
    """
    Alternating attention across features (columns) and samples (rows), followed by an MLP.

    Pipeline:
        1) Feature self-attention within each sample (across L features).
        2) Sample self-attention within each feature (across S samples) using an
           optional attention mask to control information flow.
        3) Position-wise MLP.

    Args:
        dim: Embedding dimension D.
        heads_feat: Number of attention heads for feature-attention.
        heads_samp: Number of attention heads for sample-attention.
        dropout: Dropout rate used in residual/dropout connections and MLP.
        hidden_mult: Hidden size multiplier for the MLP.

    Input/Output shape:
        x: (B, S, L, D) → (B, S, L, D)
           Where S = N + M.

    Attention mask semantics (``sample_attn_mask``):
        - Shape expected by PyTorch MHA: (num_heads * S_contexts, S, S) when passed as ``attn_mask``.
          Here it is expanded to (heads_samp * B * L, S, S) to apply the same mask
          per feature-column and head.
        - True (or nonzero) entries mean **masked** (not allowed to attend).
    """
    def __init__(self, dim: int, heads_feat: int, heads_samp: int, dropout: float = 0.0, hidden_mult: int = 4, use_flash_attention: bool = False):
        super().__init__()
        # Self-attention across features of the same sample: (B*S, L, D)
        self.feat_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads_feat, batch_first=True)
        self.ln_feat = nn.LayerNorm(dim)

        # Self-attention across samples of the same feature: (B*L, S, D)
        self.samp_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads_samp, batch_first=True)
        self.ln_samp = nn.LayerNorm(dim)
        self.heads_samp = heads_samp

        # Position-wise MLP
        self.mlp = MLP(dim, hidden_mult=hidden_mult, dropout=dropout)
        self.ln_mlp = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

        # Attention backend toggle
        self.use_flash_attention = use_flash_attention

    def _sdp_context(self, device: torch.device):
        """
        Prefer flash attention on CUDA when enabled; fall back to other SDPA kernels otherwise.
        We keep math and mem_efficient enabled as fallbacks to avoid runtime errors when flash
        is not applicable (e.g., due to mask shape or device).
        """
        if self.use_flash_attention and device.type == 'cuda':
            try:
                from torch.backends.cuda import sdp_kernel
                return sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True)
            except Exception:
                return nullcontext()
        return nullcontext()

    def forward(self, x: torch.Tensor, sample_attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, S, L, D).
            sample_attn_mask: Optional boolean mask of shape (S, S),
                where True = masked (no attention). This will be expanded
                across features and heads for sample-attention.

        Returns:
            Tensor of shape (B, S, L, D).
        """
        B, S, L, D = x.shape

        # 1) Feature-attention (within row): allow features to interact per sample
        x_row = x.reshape(B * S, L, D)                       # (B*S, L, D)
        with self._sdp_context(x.device):
            x2, _ = self.feat_attn(x_row, x_row, x_row, need_weights=False)
        x_row = x_row + self.drop(x2)
        x_row = self.ln_feat(x_row)
        x = x_row.reshape(B, S, L, D)

        # 2) Sample-attention (within column): allow samples to interact per feature
        x_col = x.permute(0, 2, 1, 3).contiguous().reshape(B * L, S, D)  # (B*L, S, D)

        # Expand (S, S) → (heads_samp * B * L, S, S) for MHA if provided
        with self._sdp_context(x.device):
            if sample_attn_mask is not None:
                # Same mask shared across features and heads
                expanded_mask = sample_attn_mask.unsqueeze(0).expand(B * L * self.heads_samp, -1, -1)
                x2, _ = self.samp_attn(x_col, x_col, x_col, attn_mask=expanded_mask, need_weights=False)
            else:
                x2, _ = self.samp_attn(x_col, x_col, x_col, need_weights=False)

        x_col = x_col + self.drop(x2)
        x_col = self.ln_samp(x_col)
        x = x_col.reshape(B, L, S, D).permute(0, 2, 1, 3).contiguous()    # back to (B, S, L, D)

        # 3) Position-wise MLP
        x2 = self.mlp(x)
        x = x + self.drop(x2)
        x = self.ln_mlp(x)
        return x


class SimplePFNRegressor(nn.Module):
    """
    Simple PFN-like regressor for tabular data with two-way attention blocks.

    Inputs:
        - X_train: (B, N, L) numeric features for training samples.
        - y_train: (B, N) or (B, N, 1) training targets.
        - X_test:  (B, M, L) numeric features for test samples.

    Architecture:
        - Encode each scalar feature value into D using a shared linear ``value_encoder``.
        - Create a special **label feature column**:
            * Train rows: encode ``y_train`` via ``label_value_encoder`` and add a learned
              positional embedding (``label_positional``) to mark this column.
            * Test rows: use a learned ``label_mask_embed`` token (no ground-truth available).
        - Concatenate encoded feature columns and the label column along L → (L + 1 total tokens per row).
        - Stack ``depth`` TwoWayBlocks for bidirectional mixing.
        - Read out test-row label column representation and pass through ``regression_head``.

    Args:
        num_features: Number of input features L.
        d_model: Embedding dimension D.
        depth: Number of stacked TwoWayBlocks.
        heads_feat: Heads for feature-attention.
        heads_samp: Heads for sample-attention.
        dropout: Dropout rate for blocks and MLPs.
        output_dim: Output dimensionality per test sample (e.g., 1 for regression,
                    or K+4 for a BarDistribution parameterization).
        hidden_mult: Hidden size multiplier for the MLP inside each TwoWayBlock.

    Outputs:
        dict with key:
            - "predictions": (B, M) if output_dim == 1; else (B, M, output_dim)

    Notes:
        - The attention mask allows information flow **from train → test** but not
          **test → train** and disallows **test ↔ test** interactions, preventing
          leakage among test samples.
        - If you set ``output_dim`` to match a downstream probabilistic head (e.g.,
          BarDistribution's K+4), the predictions can be interpreted as its parameters.
    """
    def __init__(
        self,
        num_features: int,
        d_model: int = 256,
        depth: int = 8,
        heads_feat: int = 8,
        heads_samp: int = 8,
        dropout: float = 0.0,
        output_dim: int = 1,  # New parameter for high-dimensional output
        hidden_mult: int = 4,  # MLP hidden layer multiplier
        use_flash_attention: bool = False,
    ):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.output_dim = output_dim
        self.use_flash_attention = use_flash_attention

        # Per-cell encoders (shared across all feature columns / label column)
        self.value_encoder = nn.Linear(1, d_model)        # for X values
        self.label_value_encoder = nn.Linear(1, d_model)  # for y values (train rows)

        # Learned [MASK] token for the label column on test rows
        self.label_mask_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.label_mask_embed, std=0.02)

        # Positional encoding to tag the (train) label column
        self.label_positional = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.label_positional, std=0.02)

        # Stacked two-way attention blocks
        self.blocks = nn.ModuleList([
            TwoWayBlock(d_model, heads_feat, heads_samp, dropout=dropout, hidden_mult=hidden_mult, use_flash_attention=use_flash_attention)
            for _ in range(depth)
        ])

        # Output projection from D to desired output_dim per test token
        self.regression_head = nn.Linear(d_model, output_dim)

    @staticmethod
    def _build_sample_attn_mask(N: int, M: int, device: torch.device) -> torch.Tensor:
        """
        Build a Boolean attention mask over S = N + M samples.

        Masking policy:
            - Prevent train samples from attending to test samples.
            - Prevent test samples from attending to each other.
            - Allow test samples to attend to train samples (so predictions can
              condition on training data).

        Args:
            N: Number of train samples.
            M: Number of test samples.
            device: Target device for the mask.

        Returns:
            Tensor of shape (S, S) with dtype=bool, where True = masked (no attention).
        """
        S = N + M
        mask = torch.zeros((S, S), dtype=torch.bool, device=device)
        if M > 0:
            # Train cannot look at test
            mask[:N, N:S] = True
            # Test cannot look at test
            mask[N:S, N:S] = True
            # Test can look at train (mask remains False there)
        return mask

    def _encode_features(self, X_train: torch.Tensor, X_test: torch.Tensor) -> torch.Tensor:
        """
        Encode scalar features into D-dim embeddings for both train and test rows.

        Args:
            X_train: (B, N, L)
            X_test:  (B, M, L)

        Returns:
            Tensor of shape (B, S, L, D), where S = N + M.
        """
        B, N, num_feat = X_train.shape
        Xt = torch.cat([X_train, X_test], dim=1)       # (B, S, L)
        feat_in = Xt.unsqueeze(-1)                      # (B, S, L, 1)
        feat_enc = self.value_encoder(feat_in)          # (B, S, L, D)
        return feat_enc

    def _encode_labels(self, y_train: torch.Tensor, M: int) -> torch.Tensor:
        """
        Build the label-column embeddings for both train and test rows.

        Train rows:
            - Encode y via label_value_encoder and add label_positional.

        Test rows:
            - Use the learned label_mask_embed to avoid leaking targets.

        Args:
            y_train: (B, N) or (B, N, 1)
            M: Number of test samples.

        Returns:
            Tensor of shape (B, S, D), the label-column embeddings.
        """
        # Handle both 2D (B, N) and 3D (B, N, 1) input shapes
        if y_train.dim() == 3:
            # If y_train is (B, N, 1), squeeze the last dimension
            y_train = y_train.squeeze(-1)

        try:
            B, N = y_train.shape
        except Exception:
            B = len(y_train)
        # Train encodings for label column
        lab_train = self.label_value_encoder(y_train.unsqueeze(-1))  # (B, N, D)
        lab_train = lab_train + self.label_positional                 # tag the special column

        # Test label embeddings: learned [MASK] token
        lab_test = self.label_mask_embed.expand(B, M, -1)            # (B, M, D)
        lab_all = torch.cat([lab_train, lab_test], dim=1)            # (B, S, D)
        return lab_all

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

        Notes:
            - The internal representation ``x`` has shape (B, S, L+1, D) where the
              extra feature corresponds to the label column.
            - The readout selects the label-column token at index ``label_pos = L``.
        """
        B, N, num_feat = X_train.shape
        assert num_feat == self.num_features
        M = X_test.shape[1]
        device = X_train.device

        # One-time report: whether flash attention is requested and likely available
        if not hasattr(self, "_flash_status_reported"):
            self._flash_status_reported = False
        if not self._flash_status_reported:
            if not self.use_flash_attention:
                print("[SimplePFN] Flash attention: disabled by config (use_flash_attention=False)")
            elif device.type != 'cuda':
                print(f"[SimplePFN] Flash attention: not available on device '{device.type}' (requires CUDA)")
            else:
                # CUDA device and enabled in config; attempt to detect SDPA kernel manager
                try:
                    from torch.backends.cuda import sdp_kernel  # noqa: F401
                    print("[SimplePFN] Flash attention: enabled (CUDA) — will attempt SDPA flash kernels when applicable")
                except Exception:
                    print("[SimplePFN] Flash attention: CUDA detected but SDPA flash backend not importable; falling back to standard attention")
            self._flash_status_reported = True

        # Encode features and label column, then fuse as feature dimension L+1
        feat_enc = self._encode_features(X_train, X_test)     # (B, S, L, D)
        lab_enc = self._encode_labels(y_train, M)             # (B, S, D)
        x = torch.cat([feat_enc, lab_enc.unsqueeze(2)], dim=2)  # (B, S, L+1, D)

        # Build row-wise attention mask and apply blocks
        samp_mask = self._build_sample_attn_mask(N, M, device)

        for blk in self.blocks:
            x = blk(x, sample_attn_mask=samp_mask)

        # Readout: take the label column at position L for test rows only
        label_pos = self.num_features
        h_test = x[:, N:, label_pos, :]                  # (B, M, D)
        predictions = self.regression_head(h_test)       # (B, M, output_dim)

        # Backward-compatible squeeze when output_dim == 1
        if self.output_dim == 1:
            predictions = predictions.squeeze(-1)        # (B, M)

        return {"predictions": predictions}


if __name__ == "__main__":
    torch.manual_seed(0)
    B, N, M, num_feat = 2, 16, 5, 7

    Xtr = torch.randn(B, N, num_feat)
    Xte = torch.randn(B, M, num_feat)
    ytr = torch.randn(B, N)

    # Test with default single output
    print("=== Single Output (Backward Compatible) ===")
    model = SimplePFNRegressor(num_features=num_feat, d_model=128, depth=4, heads_feat=4, heads_samp=4, dropout=0.1)
    out = model(Xtr, ytr, Xte)
    print('predictions shape:', out['predictions'].shape)  # Should be (B, M)
    print('predictions:', out['predictions'])
    
    # Test with high-dimensional output
    print("\n=== High-Dimensional Output ===")
    output_dim = 10  # Example: 10-dimensional output per test sample
    model_hd = SimplePFNRegressor(
        num_features=num_feat, 
        d_model=128, 
        depth=4, 
        heads_feat=4, 
        heads_samp=4, 
        dropout=0.1,
        output_dim=output_dim
    )
    out_hd = model_hd(Xtr, ytr, Xte)
    print('high-dim predictions shape:', out_hd['predictions'].shape)  # Should be (B, M, output_dim)
    print('high-dim predictions:', out_hd['predictions'])
    
    # Test compatibility with BarDistribution
    print("\n=== BarDistribution Compatibility Example ===")
    # For BarDistribution with 5 bars, we need 5 + 4 = 9 parameters
    bar_params = 9
    model_bar = SimplePFNRegressor(
        num_features=num_feat, 
        d_model=128, 
        depth=4, 
        heads_feat=4, 
        heads_samp=4, 
        dropout=0.1,
        output_dim=bar_params
    )
    out_bar = model_bar(Xtr, ytr, Xte)
    print('BarDistribution params shape:', out_bar['predictions'].shape)  # Should be (B, M, 9)
    print('BarDistribution params sample:', out_bar['predictions'][0, 0, :])  # First test sample parameters
