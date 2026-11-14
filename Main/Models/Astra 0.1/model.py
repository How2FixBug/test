#!/usr/bin/env python3
"""ASTRA 0.1 — compact decoder-only Transformer language model."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.heads, self.head_width)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        return self.out(attended)


class TransformerBlock(nn.Module):
    def __init__(self, width: int, heads: int, ff_width: int, dropout: float):
        super().__init__()
        self.attn_norm = nn.LayerNorm(width)
        self.attention = CausalSelfAttention(width, heads, dropout)
        self.mlp_norm = nn.LayerNorm(width)
        self.gate = nn.Linear(width, ff_width, bias=False)
        self.up = nn.Linear(width, ff_width, bias=False)
        self.down = nn.Linear(ff_width, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.attn_norm(x)))
        normalized = self.mlp_norm(x)
        hidden = F.silu(self.gate(normalized)) * self.up(normalized)
        return x + self.dropout(self.down(hidden))


class AstraGPT(nn.Module):
    """Small GPT-style decoder with distinct layers and tied token embeddings."""
    def __init__(
        self, vocab_size: int, width: int = 384, heads: int = 6,
        ff_width: int = 1024, max_seq_len: int = 512,
        layers: int = 4, dropout: float = 0.05,
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be at least 1")
        self.vocab_size = vocab_size
        self.width = width
        self.max_seq_len = max_seq_len
        self.num_layers = layers
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(max_seq_len, width)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(width, heads, ff_width, dropout)
            for _ in range(layers)
        ])
        self.final_norm = nn.LayerNorm(width)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, length = input_ids.shape
        if length > self.max_seq_len:
            raise ValueError(
                f"Sequence length {length} exceeds max_seq_len={self.max_seq_len}"
            )
        positions = torch.arange(length, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        x = self.embedding_dropout(x)
        for block in self.blocks:
            x = block(x)
        return F.linear(self.final_norm(x), self.token_embedding.weight)
