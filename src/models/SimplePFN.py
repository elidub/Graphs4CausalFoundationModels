from __future__ import annotations
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
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


class TwoWayBlock(nn.Module):
    def __init__(self, dim: int, heads_feat: int, heads_samp: int, dropout: float = 0.0):
        super().__init__()
        self.feat_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads_feat, batch_first=True)
        self.ln_feat = nn.LayerNorm(dim)

        self.samp_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads_samp, batch_first=True)
        self.ln_samp = nn.LayerNorm(dim)
        self.heads_samp = heads_samp

        self.mlp = MLP(dim, hidden_mult=4, dropout=dropout)
        self.ln_mlp = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sample_attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, L, D = x.shape

        # feature-attention (within row)
        x_row = x.reshape(B * S, L, D)
        x2, _ = self.feat_attn(x_row, x_row, x_row, need_weights=False)
        x_row = x_row + self.drop(x2)
        x_row = self.ln_feat(x_row)
        x = x_row.reshape(B, S, L, D)

        # sample-attention (within column)
        x_col = x.permute(0, 2, 1, 3).contiguous().reshape(B * L, S, D)
        
        # Expand the mask for all feature columns and attention heads if provided
        if sample_attn_mask is not None:
            # The mask needs to be expanded for each feature column (B * L copies)
            # and for each attention head (heads_samp copies)
            expanded_mask = sample_attn_mask.unsqueeze(0).expand(B * L * self.heads_samp, -1, -1)
            x2, _ = self.samp_attn(x_col, x_col, x_col, attn_mask=expanded_mask, need_weights=False)
        else:
            x2, _ = self.samp_attn(x_col, x_col, x_col, need_weights=False)
            
        x_col = x_col + self.drop(x2)
        x_col = self.ln_samp(x_col)
        x = x_col.reshape(B, L, S, D).permute(0, 2, 1, 3).contiguous()

        # MLP
        x2 = self.mlp(x)
        x = x + self.drop(x2)
        x = self.ln_mlp(x)
        return x


class SimplePFNRegressor(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_model: int = 256,
        depth: int = 8,
        heads_feat: int = 8,
        heads_samp: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model

        # Per-cell encoders
        self.value_encoder = nn.Linear(1, d_model)
        self.label_value_encoder = nn.Linear(1, d_model)

        # Learned [MASK] for label on test rows
        self.label_mask_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.label_mask_embed, std=0.02)

        # Positional encoding for the train-y column
        self.label_positional = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.label_positional, std=0.02)

        # Stacked two-way blocks
        self.blocks = nn.ModuleList([
            TwoWayBlock(d_model, heads_feat, heads_samp, dropout=dropout)
            for _ in range(depth)
        ])

        # Output head - single regression value
        self.regression_head = nn.Linear(d_model, 1)

    @staticmethod
    def _build_sample_attn_mask(N: int, M: int, device: torch.device) -> torch.Tensor:
        """
        Build attention mask to prevent test samples from attending to each other
        and train samples from attending to test samples.
        
        Returns mask where True = masked (not attended to)
        """
        S = N + M
        mask = torch.zeros((S, S), dtype=torch.bool, device=device)
        if M > 0:
            # Prevent train samples from attending to test samples
            mask[:N, N:S] = True
            # Prevent test samples from attending to each other  
            mask[N:S, N:S] = True
            # Allow test samples to attend to train samples (this is correct)
        return mask

    def _encode_features(self, X_train: torch.Tensor, X_test: torch.Tensor) -> torch.Tensor:
        B, N, num_feat = X_train.shape
        Xt = torch.cat([X_train, X_test], dim=1)
        feat_in = Xt.unsqueeze(-1)
        feat_enc = self.value_encoder(feat_in)
        return feat_enc

    def _encode_labels(self, y_train: torch.Tensor, M: int) -> torch.Tensor:
        B, N = y_train.shape
        # train encodings
        lab_train = self.label_value_encoder(y_train.unsqueeze(-1))
        # add positional encoding for train-y-column
        lab_train = lab_train + self.label_positional
        # test label enc
        lab_test = self.label_mask_embed.expand(B, M, -1)
        lab_all = torch.cat([lab_train, lab_test], dim=1)
        return lab_all

    def forward(self, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, N, num_feat = X_train.shape
        assert num_feat == self.num_features
        M = X_test.shape[1]
        device = X_train.device

        feat_enc = self._encode_features(X_train, X_test)
        lab_enc = self._encode_labels(y_train, M).unsqueeze(2)
        x = torch.cat([feat_enc, lab_enc], dim=2)

        samp_mask = self._build_sample_attn_mask(N, M, device)

        for blk in self.blocks:
            x = blk(x, sample_attn_mask=samp_mask)

        label_pos = self.num_features
        h_test = x[:, N:, label_pos, :]
        predictions = self.regression_head(h_test).squeeze(-1)  # (B, M)
        return {"predictions": predictions}


if __name__ == "__main__":
    torch.manual_seed(0)
    B, N, M, num_feat = 2, 16, 5, 7

    Xtr = torch.randn(B, N, num_feat)
    Xte = torch.randn(B, M, num_feat)
    ytr = torch.randn(B, N)

    model = SimplePFNRegressor(num_features=num_feat, d_model=128, depth=4, heads_feat=4, heads_samp=4, dropout=0.1)
    out = model(Xtr, ytr, Xte)
    print('predictions:', out['predictions'].shape)
