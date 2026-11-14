#!/usr/bin/env python3
"""ASTRA 0.1 — LoopCore language model."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SharedLoopBlock(nn.Module):
    """One transformer-like block reused at every LoopCore pass."""

    def __init__(self, width: int, heads: int, ff_width: int, local_window: int):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.local_window = local_window

        self.attn_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.attn_out = nn.Linear(width, width, bias=False)

        self.memory_candidate = nn.Linear(width, width)
        self.memory_gate = nn.Linear(width * 2, width)
        self.memory_out = nn.Linear(width, width, bias=False)

        self.mlp_norm = nn.LayerNorm(width)
        self.mlp_gate = nn.Linear(width, ff_width, bias=False)
        self.mlp_up = nn.Linear(width, ff_width, bias=False)
        self.mlp_down = nn.Linear(ff_width, width, bias=False)

    def forward(self, x: torch.Tensor, memory: torch.Tensor):
        batch, length, width = x.shape
        normalized = self.attn_norm(x)
        qkv = self.qkv(normalized).view(
            batch, length, 3, self.heads, self.head_width
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # True entries are visible to attention: causal and limited to a
        # sliding local window. The mask is shared across the batch/heads.
        positions = torch.arange(length, device=x.device)
        distance = positions[:, None] - positions[None, :]
        attention_mask = (distance >= 0) & (distance < self.local_window)
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        x = x + self.attn_out(attended)

        pooled = x.mean(dim=1, keepdim=True)
        candidate = torch.tanh(self.memory_candidate(pooled))
        gate = torch.sigmoid(self.memory_gate(torch.cat((pooled, memory), dim=-1)))
        memory = gate * memory + (1.0 - gate) * candidate
        x = x + self.memory_out(memory)

        normalized = self.mlp_norm(x)
        hidden = F.silu(self.mlp_gate(normalized)) * self.mlp_up(normalized)
        x = x + self.mlp_down(hidden)
        return x, memory


class LoopCore(nn.Module):
    """Autoregressive model with a single shared block applied recurrently."""

    def __init__(
        self,
        vocab_size: int,
        width: int = 512,
        heads: int = 8,
        ff_width: int = 1792,
        max_seq_len: int = 512,
        local_window: int = 128,
        loop_passes: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.width = width
        self.max_seq_len = max_seq_len
        self.loop_passes = loop_passes

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(max_seq_len, width)
        self.pass_embedding = nn.Embedding(loop_passes, width)
        self.loop_controller = nn.Linear(width * 2, 1)
        self.shared_block = SharedLoopBlock(width, heads, ff_width, local_window)
        self.final_norm = nn.LayerNorm(width)

        self.apply(self._init_weights)
        # Tied input/output embeddings: fewer parameters and one shared
        # representation space for reading and predicting tokens.

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor):
        batch, length = input_ids.shape
        if length > self.max_seq_len:
            raise ValueError(
                f"Sequence length {length} exceeds max_seq_len={self.max_seq_len}"
            )
        positions = torch.arange(length, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        memory = x.new_zeros(batch, 1, self.width)

        for pass_index in range(self.loop_passes):
            pass_vector = self.pass_embedding.weight[pass_index].view(1, 1, -1)
            pooled = x.mean(dim=1, keepdim=True)
            control_input = torch.cat((pooled, pass_vector.expand(batch, -1, -1)), dim=-1)
            strength = torch.sigmoid(self.loop_controller(control_input))
            updated, memory = self.shared_block(x + pass_vector, memory)
            x = x + strength * (updated - x)

        x = self.final_norm(x)
        return F.linear(x, self.token_embedding.weight)
